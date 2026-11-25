"""Macro forecast vintages for a realistic out-of-sample test.

The standard out-of-sample test forecasts with the macro values that actually happened (a conditional forecast):
it measures the model's error but not the error of the macro forecast it would have had to use. With vintages -
the macro forecasts that were available at past dates (for example the baseline scenario published each quarter) -
each origin also forecasts with the forecast of its time, so the test shows what the forecast would really have been.

    oos:
      macro_vintages:
        file: data/vintages/               # a folder of files, one per vintage, the quarter in the file name
                                           #   (baseline_2018Q1.csv, 2018Q1.xlsx, 2018-03.csv ...); or one csv/xlsx
                                           #   file with a vintage column; or one xlsx with a sheet per vintage
        vintage_column: vintage            # in a single long file: the quarter each forecast was made
        date_column: null                  # null = data.macro.date_column
        max_age: 2                         # use a vintage at most this many quarters older than the origin
        splice: true                       # apply the vintage's growth (levels) or changes (rates) from the actual
                                           #   value at the origin, so revisions to history do not create jumps
        missing: actual                    # series not in the vintages: actual (their realised path, reported) or
                                           #   hold (flat at the last actual value)
        rank_on: forecast                  # the recommended model is scored on these errors (forecast) or on the
                                           #   realised-macro ones (realised)

Each vintage table has the date column and the macro series as columns (the same layout as the scenario files);
quarters up to the vintage date are its history, later quarters its forecast.
"""
import os
import re

import numpy as np
import pandas as pd

from ..utils.io import read_table, to_quarters

DEFAULTS = {"file": None, "vintage_column": "vintage", "date_column": None, "max_age": 2, "splice": True,
            "missing": "actual", "rank_on": "forecast"}


def settings(cfg):
    v = (cfg.get("oos") or {}).get("macro_vintages") or {}
    if isinstance(v, str):
        v = {"file": v}
    return {**DEFAULTS, **{k: x for k, x in v.items() if x is not None}}


def _quarter(text):
    """'baseline_2018Q1.csv' -> Period 2018Q1; '2018-03' -> 2018Q1; None if no date."""
    s = str(text)
    m = re.search(r"(19|20)(\d{2})[\s_\-]?[Qq]([1-4])", s)
    if m:
        return pd.Period(f"{m.group(1)}{m.group(2)}Q{m.group(3)}", freq="Q")
    m = re.search(r"(19|20)(\d{2})[\s_\-./]?(0[1-9]|1[0-2])", s)
    if m:
        return pd.Period(f"{m.group(1)}{m.group(2)}-{m.group(3)}", freq="M").asfreq("Q")
    return None


def _table(df, date_col):
    """A vintage table indexed by quarter, numeric columns only."""
    if date_col not in df.columns:
        date_col = df.columns[0]
    q = to_quarters(df[date_col])
    out = df.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    out.index = q
    out = out.groupby(level=0).last().sort_index()
    return out.dropna(axis=1, how="all")


def load(cfg, log=print):
    """{vintage quarter: table} from oos.macro_vintages.file, or {} when not set."""
    st = settings(cfg)
    path = st["file"]
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"oos.macro_vintages.file not found: {path}")
    dcol = st["date_column"] or cfg["data"]["macro"]["date_column"]
    out = {}
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith((".csv", ".xlsx", ".xls")):
                continue
            v = _quarter(os.path.splitext(f)[0])
            if v is None:
                log(f"  vintages: {f} has no quarter in its name - skipped")
                continue
            out[v] = _table(read_table(os.path.join(path, f)), dcol)
    elif path.lower().endswith((".xlsx", ".xls")) and len(pd.ExcelFile(path).sheet_names) > 1:
        for sh in pd.ExcelFile(path).sheet_names:
            v = _quarter(sh)
            if v is not None:
                out[v] = _table(pd.read_excel(path, sheet_name=sh), dcol)
    else:
        df = read_table(path)
        vc = st["vintage_column"]
        if vc not in df.columns:
            raise ValueError(f"vintage file {path} has no '{vc}' column (oos.macro_vintages.vintage_column)")
        for v, g in df.groupby(vc):
            q = _quarter(v) or to_quarters(pd.Series([v])).iloc[0]
            out[q] = _table(g.drop(columns=[vc]), dcol)
    if out:
        ks = sorted(out)
        log(f"  macro forecast vintages: {len(out)} ({ks[0]}..{ks[-1]}), "
            f"{len(set().union(*[set(t.columns) for t in out.values()]))} series")
    return dict(sorted(out.items()))


def pick(vints, origin, max_age):
    """The latest vintage made at or before the origin and at most max_age quarters old, or None."""
    ok = [v for v in vints if v <= origin and (origin - v).n <= int(max_age)]
    return max(ok) if ok else None


def frame_at(ctx, vints, origin, cfg):
    """The macro frame for forecasting from `origin` with the vintage of the time.

    Quarters up to the origin keep the actual history (the models are estimated on it); later quarters take the
    vintage's forecast, spliced on the actual value at the origin (growth for levels, changes for rates).

    Returns:
        (frame or None, info dict: vintage, age, series covered, series not covered).
    """
    st = settings(cfg)
    v = pick(vints, origin, st["max_age"])
    if v is None:
        return None, {"vintage": None}
    V = vints[v]
    M = ctx.base.copy()                                 # keeps the theme-factor key (attrs)
    fut = M.index[M.index > origin]
    cols = [c for c in V.columns if c in M.columns]
    meta = ctx.meta
    for c in cols:
        vv = V[c].reindex(fut)
        if vv.isna().all():
            continue
        if st["splice"]:
            a0 = M.at[origin, c] if origin in M.index else np.nan
            anchor = V[c].loc[:origin].dropna()
            v0 = anchor.iloc[-1] if len(anchor) else np.nan
            lvl = c in meta.index and str(meta.loc[c, "transform"]) == "dlog" and (V[c].dropna() > 0).all() \
                and pd.notna(a0) and a0 > 0
            if pd.notna(a0) and pd.notna(v0) and (v0 > 0 or not lvl):
                vv = a0 * vv / v0 if lvl else a0 + (vv - v0)
        M.loc[fut, c] = vv.where(vv.notna(), M.loc[fut, c]).values
    # constructed series (data.constructed: real rates, spreads, ...) are rebuilt from the spliced inputs
    built = []
    for item in (cfg["data"].get("constructed") or []):
        nm = item.get("name")
        if nm not in M.columns or nm in cols:
            continue
        try:
            new = M.eval(item["expression"], local_dict={"log": np.log, "exp": np.exp}, engine="python")
            M.loc[fut, nm] = pd.Series(new, index=M.index).loc[fut].values
            built.append(nm)
        except Exception:
            pass
    cols = cols + built
    missing = [c for c in M.columns if c not in cols and not str(c).startswith("FAC.")]
    if st["missing"] == "hold" and origin in M.index:
        for c in missing:
            M.loc[fut, c] = M.at[origin, c]
    return M, {"vintage": str(v), "age_quarters": int((origin - v).n), "covered": cols, "not_covered": missing}


def coverage(model, info):
    """Share of a model's driver series that came from the vintage (1 = all)."""
    from ..impact.analysis import used_codes
    codes = [c for c in used_codes(model, info.get("covered", []) + info.get("not_covered", []))]
    if not codes:
        return np.nan, []
    out = [c for c in codes if c in set(info.get("not_covered", []))]
    return 1 - len(out) / len(codes), out
