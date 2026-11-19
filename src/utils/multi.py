"""Several targets in one data file: which columns can be forecast, and the config for each one.

A data file can hold many target series side by side (date, cre_loans, heloc_balances, stage2_share, ...). With a
literature master (literature.master), each column is matched to the master's rows for that target, so every target
gets its own literature models and its own expected signs. Used by run_targets.py and the app.

Matching a column to a master target: targets.map in the config first ({column: master target}), then the same
name (case, spaces, hyphens and underscores ignored). A column with no match can still run with a metadata file
given in the config, without literature models.

The target form (growth or change) is suggested per target: targets.forms in the config first, then 'diff' for
rates, ratios, shares, margins and anything that is negative or crosses zero, else 'dlog'.
"""
import copy
import os
import re

import pandas as pd

from . import master as ms
from .io import read_table, to_quarters

DIFF_WORDS = re.compile(r"rate|ratio|share|margin|spread|yield|delinquen|arrears|stage_?[23]|write_?off|provision|"
                        r"coverage|pct|percent|_r$|nim", re.I)


def _key(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def columns(cfg):
    """The numeric columns of the target file (the date and bank columns left out).

    Args:
        cfg: configuration (data.bank).

    Returns:
        DataFrame indexed by quarter, one column per series.
    """
    b = cfg["data"]["bank"]
    df = read_table(b["file"], b.get("sheet"))
    dc = b["date_column"]
    skip = {dc, b.get("panel_column")}
    num = [c for c in df.columns if c not in skip and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    out = df[num].apply(pd.to_numeric, errors="coerce")
    out.index = to_quarters(df[dc])
    return out


def suggest_form(name, series, forms=None):
    """'diff' or 'dlog' for a target.

    Args:
        name: column name. series: its values. forms: {name: form} from the config (wins).
    """
    forms = {_key(k): v for k, v in (forms or {}).items()}
    if _key(name) in forms:
        return forms[_key(name)]
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) and (s <= 0).any():
        return "diff"
    return "diff" if DIFF_WORDS.search(str(name)) else "dlog"


def catalog(cfg, master=None):
    """Every column of the target file with its match in the master, history and suggested form.

    Args:
        cfg: configuration. master: literature master file (default literature.master).

    Returns:
        DataFrame: column, master_target, literature (models in the master), metadata (rows), first, last,
        quarters, form, ready.
    """
    X = columns(cfg)
    master = master or (cfg.get("literature") or {}).get("master")
    T = ms.targets(master) if master and ms.is_master(master) else pd.DataFrame(columns=["target"])
    names = list(T["target"]) if len(T) else []
    tcfg = cfg.get("targets") or {}
    mp = {_key(k): v for k, v in (tcfg.get("map") or {}).items()}
    mdf = (cfg["data"].get("metadata") or {}).get("file")
    own_meta = bool(mdf) and os.sep + os.path.join("inputs", "master") + os.sep not in str(mdf) and not ms.is_master(mdf)
    rows = []
    for c in X.columns:
        s = X[c].dropna()
        m = mp.get(_key(c)) or ms._match(names, c)
        m = ms._match(names, m) if m and names else m
        info = T.set_index("target").loc[m] if m in names else None
        rows.append({"column": c, "master_target": m or "", "literature": int(info["models"]) if info is not None and "models" in info else 0,
                     "metadata": int(info["metadata"]) if info is not None and "metadata" in info else 0,
                     "first": str(s.index[0]) if len(s) else "", "last": str(s.index[-1]) if len(s) else "",
                     "quarters": len(s), "form": suggest_form(c, s, tcfg.get("forms")),
                     "ready": bool(len(s) >= 40 and (info is not None and info.get("metadata", 0) > 0
                                                     or own_meta))})
    return pd.DataFrame(rows)


def target_config(base, row, run_prefix=None):
    """The config for one target, as a dict (written as YAML by the caller).

    Args:
        base: the raw config dict (as in the YAML file). row: a catalog row. run_prefix: run name prefix.
    """
    c = copy.deepcopy(base)
    col = row["column"]
    c["run_name"] = f"{run_prefix or base.get('run_name', 'run')}_{_key(col)}"
    c.setdefault("data", {}).setdefault("bank", {})["target_column"] = col
    lit = c.setdefault("literature", {})
    if row.get("master_target") and lit.get("master"):
        lit["target"] = row["master_target"]
        lit.pop("workbook", None)                  # the master's rows, not the base config's workbook
        if row.get("metadata", 0) > 0:             # the master's metadata for this target
            c["data"].setdefault("metadata", {}).pop("file", None)
        if row.get("literature", 0) == 0:
            lit["workbook"] = None
    t = c.setdefault("target", {})
    t["transform"] = row["form"]
    t["label"] = (base.get("targets") or {}).get("labels", {}).get(col) or str(col).replace("_", " ")
    if row["form"] == "diff":
        t["display_scale"], t["display_units"] = None, None
        t["units"] = (base.get("targets") or {}).get("units", {}).get(col) or ""
    else:
        u = (base.get("targets") or {}).get("units", {}).get(col)
        if u:
            t["units"], t["display_scale"], t["display_units"] = u, None, None
    if not row.get("master_target") and lit.get("master"):
        lit.pop("master", None)                    # no literature for this column: metadata from the config
    return c


def outdir(base_cfg_loaded, run_name):
    return os.path.join(os.path.dirname(base_cfg_loaded["out_dir"]), run_name)
