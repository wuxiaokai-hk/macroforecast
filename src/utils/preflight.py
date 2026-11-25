"""Pre-flight check of a config and its data: finds name, file and date problems in seconds, before a long run.

Every check returns a finding with a level (ok, warn, error), the setting concerned and, for problems, how to fix
it. run_pipeline.py and run_oos.py run it first and stop on errors (--skip-check to override);
`python tools/check_config.py --config ...` runs it on its own.
"""
import os

import numpy as np
import pandas as pd

from .io import read_table, to_quarters

OK, WARN, ERROR = "ok", "warn", "error"
TRANSFORMS = {"dlog", "diff", "level", "lvl", "log"}


class Report:
    """Collects findings and prints them grouped by area."""

    def __init__(self):
        """Start an empty report."""
        self.items = []

    def add(self, level, area, msg, fix=""):
        """Record one finding.

        Args:
            level: ok, warn or error. area: config section or file. msg: what was found. fix: how to fix it.
        """
        self.items.append({"level": level, "area": area, "message": msg, "fix": fix})

    @property
    def errors(self):
        """Findings that stop a run."""
        return [i for i in self.items if i["level"] == ERROR]

    @property
    def warnings(self):
        """Findings worth a look that do not stop a run."""
        return [i for i in self.items if i["level"] == WARN]

    def text(self, show_ok=True):
        """The report as text, grouped by area.

        Args:
            show_ok: include the checks that passed.

        Returns:
            str.
        """
        mark = {OK: "  ok ", WARN: " WARN", ERROR: "ERROR"}
        out, area = [], None
        for i in self.items:
            if i["level"] == OK and not show_ok:
                continue
            if i["area"] != area:
                area = i["area"]
                out.append(f"\n[{area}]")
            out.append(f"  {mark[i['level']]}  {i['message']}")
            if i["fix"] and i["level"] != OK:
                out.append(f"         -> {i['fix']}")
        n_e, n_w = len(self.errors), len(self.warnings)
        out.append(f"\n{n_e} error(s), {n_w} warning(s)" + ("" if n_e else ": ready to run"))
        return "\n".join(out)


def _q(v):
    """A quarter from a config value (None stays None).

    Args:
        v: "2020Q1", a date, or None.

    Returns:
        Period or None.
    """
    if v in (None, "", "null"):
        return None
    return to_quarters([v])[0]


def _gap(a, b):
    """Quarters from b to a (a - b).

    Args:
        a, b: Periods.

    Returns:
        int.
    """
    return a.ordinal - b.ordinal


def _read(path, sheet, rep, area, what):
    """Read a CSV/Excel file, recording an error if it is missing or unreadable.

    Args:
        path: file. sheet: Excel sheet. rep: Report. area: section. what: description for messages.

    Returns:
        DataFrame or None.
    """
    if not path:
        rep.add(ERROR, area, f"no file given for {what}", "set the file name in the config")
        return None
    if not os.path.exists(path):
        rep.add(ERROR, area, f"{what}: file not found: {path}",
                "check the path; relative paths start at the project folder (where run_pipeline.py is)")
        return None
    try:
        return read_table(path, sheet)
    except Exception as e:
        rep.add(ERROR, area, f"{what}: cannot read {path}: {type(e).__name__}: {e}",
                "for Excel give `sheet:`; for CSV check the separator is a comma")
        return None


def _dates(df, col, rep, area, what):
    """Parse a date column into quarters, recording problems.

    Args:
        df: table. col: date column. rep: Report. area: section. what: description.

    Returns:
        PeriodIndex or None.
    """
    if col not in df.columns:
        rep.add(ERROR, area, f"{what}: date column '{col}' not found (columns: {list(df.columns)[:8]} ...)",
                f"set date_column to one of the column names above")
        return None
    try:
        return to_quarters(df[col])
    except Exception as e:
        rep.add(ERROR, area, f"{what}: cannot read dates in '{col}' ({e}); first values: {list(df[col].astype(str)[:3])}",
                "use quarter labels like 2020Q1 or dates like 2020-03-31")
        return None


def check(cfg):
    """Check a loaded config and its input files.

    Args:
        cfg: configuration from load_config.

    Returns:
        Report.
    """
    rep = Report()
    d, s = cfg["data"], cfg["sample"]

    # ---------------------------------------------------------------- bank
    b = d["bank"]
    bank = _read(b["file"], b.get("sheet"), rep, "data.bank", "target file")
    level = None
    if bank is not None:
        q = _dates(bank, b["date_column"], rep, "data.bank", "target file")
        if b["target_column"] not in bank.columns:
            rep.add(ERROR, "data.bank", f"target column '{b['target_column']}' not found; columns: {list(bank.columns)}",
                    "set target_column to the balance you want to forecast")
        elif q is not None:
            v = pd.to_numeric(bank[b["target_column"]], errors="coerce")
            v.index = q
            if b.get("frequency", "Q").upper() == "M":
                v = v.groupby(level=0).last()
            elif v.index.duplicated().any():
                rep.add(ERROR, "data.bank", "several rows per quarter", "set frequency: M for monthly data")
            level = v.dropna().sort_index()
            gaps = pd.period_range(level.index[0], level.index[-1], freq="Q").difference(level.index)
            if (level <= 0).any():
                rep.add(ERROR, "data.bank", "the balance has zero or negative values", "the model uses log growth: check the column")
            if len(gaps):
                rep.add(WARN, "data.bank", f"{len(gaps)} missing quarters, e.g. {[str(g) for g in gaps[:3]]}",
                        "fill them, or start the sample after the gap")
            rep.add(OK, "data.bank", f"'{b['target_column']}': {level.index[0]} to {level.index[-1]} ({len(level)} quarters)")

    # ---------------------------------------------------------------- events
    e = d.get("events") or {}
    if e.get("file"):
        ev = _read(e["file"], e.get("sheet"), rep, "data.events", "events file")
        if ev is not None:
            miss = [c for c in (e["date_column"], e["series_column"], e["amount_column"]) if c not in ev.columns]
            if miss:
                rep.add(ERROR, "data.events", f"columns {miss} not found; columns: {list(ev.columns)}",
                        "fix date_column / series_column / amount_column, or set events.file: null")
            else:
                n = int((ev[e["series_column"]].astype(str) == b["target_column"]).sum())
                rep.add(OK if n else WARN, "data.events", f"{n} event(s) for '{b['target_column']}'",
                        "" if n else f"the series column must contain '{b['target_column']}' for events of the target")
    else:
        rep.add(OK, "data.events", "no events file (no inorganic adjustments)")

    # ---------------------------------------------------------------- macro
    m = d["macro"]
    macro = {}
    for name, f in (m.get("scenarios") or {}).items():
        df = _read(f, m.get("sheet"), rep, "data.macro", f"scenario '{name}'")
        if df is None:
            continue
        q = _dates(df, m["date_column"], rep, "data.macro", f"scenario '{name}'")
        if q is None:
            continue
        df = df.drop(columns=m["date_column"]).apply(pd.to_numeric, errors="coerce")
        df.index = q
        macro[name] = df.sort_index()
    if m.get("base_scenario") not in (m.get("scenarios") or {}):
        rep.add(ERROR, "data.macro", f"base_scenario '{m.get('base_scenario')}' is not one of {list((m.get('scenarios') or {}))}",
                "set base_scenario to the name of your baseline scenario")
    base = macro.get(m.get("base_scenario"))
    if base is not None:
        cols = set(base.columns)
        for name, df in macro.items():
            diff = cols ^ set(df.columns)
            if diff:
                rep.add(WARN, "data.macro", f"scenario '{name}' has {len(diff)} column(s) not shared with the base, e.g. {sorted(diff)[:3]}",
                        "all scenario files should have the same variables")
        rep.add(OK, "data.macro", f"{len(macro)} scenario(s); base '{m['base_scenario']}': {base.shape[1]} variables, "
                                  f"{base.index[0]} to {base.index[-1]}")

    # ---------------------------------------------------------------- metadata
    md = d["metadata"]
    meta = None
    mdf = _read(md["file"], md.get("sheet"), rep, "data.metadata", "metadata file")
    if mdf is not None:
        cmap = md["columns"]
        miss = {k: v for k, v in cmap.items() if v not in mdf.columns}
        if miss:
            rep.add(ERROR, "data.metadata", f"columns not found: {miss}; the file has {list(mdf.columns)}",
                    "under metadata.columns, map each name on the left to YOUR column name")
        else:
            meta = mdf.rename(columns={v: k for k, v in cmap.items()})[list(cmap)]
            meta["mnemonic"] = meta["mnemonic"].astype(str).str.strip()
            dup = meta["mnemonic"][meta["mnemonic"].duplicated()].unique()
            if len(dup):
                rep.add(WARN, "data.metadata", f"{len(dup)} duplicated mnemonic(s), e.g. {list(dup[:3])}", "keep one row per variable")
            meta = meta.drop_duplicates("mnemonic").set_index("mnemonic")
            bad_t = sorted(set(meta["transform"].astype(str).str.strip()) - TRANSFORMS)
            if bad_t:
                rep.add(WARN, "data.metadata", f"transform values not recognised: {bad_t[:5]}",
                        "use dlog for levels and indexes, diff for rates and percentages")
            sg = pd.to_numeric(meta["expected_sign"], errors="coerce")
            if (~sg.isin([-1, 0, 1])).any():
                rep.add(WARN, "data.metadata", f"{int((~sg.isin([-1, 0, 1])).sum())} expected_sign value(s) not -1, 0 or 1",
                        "blank or other values count as 0 (no prior; not selectable)")
            n_sign = int((sg.fillna(0) != 0).sum())
            rep.add(OK if n_sign else ERROR, "data.metadata", f"{len(meta)} variables, {n_sign} with an expected sign",
                    "" if n_sign else "with features.require_sign_prior, variables need a +1/-1 sign to be selected")
            if base is not None:
                lvl_codes = [c for c in meta.index[meta["transform"].astype(str).str.strip() == "dlog"]
                             if c in base.columns]
                nonpos = [c for c in lvl_codes if (pd.to_numeric(base[c], errors="coerce") <= 0).any()]
                if nonpos:
                    rep.add(WARN, "data.metadata",
                            f"{len(nonpos)} series with transform dlog have zero or negative values, e.g. {nonpos[:4]}",
                            "their log growth does not exist in those quarters: such forms are not offered as candidates "
                            "and literature drivers using them are left out. Fix the values, or set the transform to diff")
                no_meta = [c for c in base.columns if c not in meta.index]
                no_data = [c for c in meta.index if c not in base.columns]
                lvl = OK if len(no_meta) < 0.2 * base.shape[1] else WARN
                rep.add(lvl, "data.metadata", f"{base.shape[1] - len(no_meta)} of {base.shape[1]} macro variables have a metadata row"
                        + (f"; missing e.g. {no_meta[:3]}" if no_meta else ""),
                        "variables without a row are ignored: add rows to the metadata file")
                if no_data:
                    rep.add(WARN, "data.metadata", f"{len(no_data)} metadata row(s) with no column in the macro file, e.g. {no_data[:3]}",
                            "names must match exactly (case and dots included)")

    # ---------------------------------------------------------------- constructed
    for c in d.get("constructed") or []:
        if base is None:
            break
        try:
            base.eval(c["expression"], local_dict={"log": np.log, "exp": np.exp}, engine="python")
            rep.add(OK, "data.constructed", f"{c['name']} = {c['expression']}")
        except Exception as ex:
            rep.add(WARN, "data.constructed", f"{c['name']} cannot be built ({type(ex).__name__}: {ex}); it will be skipped",
                    "rewrite the expression with your variable names (wrap names with dots in backticks), or delete it")

    # ---------------------------------------------------------------- features
    fe = cfg["features"]
    names = set(base.columns) if base is not None else set()
    names |= {c["name"] for c in d.get("constructed") or []}
    if fe.get("deflator"):
        rep.add(OK if fe["deflator"] in names else ERROR, "features", f"deflator '{fe['deflator']}'"
                + (" found" if fe["deflator"] in names else " not in the macro data"),
                "set features.deflator to your CPI series, or null")
    if meta is not None:
        geos = set(meta["geography"].astype(str))
        themes = set(meta["theme"].astype(str))
        for key, have, what in (("allowed_geographies", geos, "geography"), ("excluded_themes", themes, "theme"),
                                ("real_themes", themes, "theme")):
            vals = fe.get(key) or []
            missing = [v for v in vals if v not in have]
            if missing:
                rep.add(WARN if key != "allowed_geographies" or len(missing) < len(vals) else ERROR, "features",
                        f"{key}: {missing} not found in the metadata {what} column",
                        f"use values from: {sorted(have)[:12]}" + (" ..." if len(have) > 12 else ""))
        for sec in ("screen_first", "en_first"):
            pg = (cfg.get(sec) or {}).get("primary_geography")
            if pg and pg not in geos:
                rep.add(WARN, sec, f"primary_geography '{pg}' not in the metadata geographies",
                        f"use one of {sorted(geos)[:10]} (or null)")
        req = (cfg.get("final_model") or {}).get("required_themes") or []
        if [t for t in req if t not in themes]:
            rep.add(ERROR, "final_model", f"required_themes {[t for t in req if t not in themes]} not in the metadata themes",
                    "use theme names from the metadata, or []")
        keep = meta
        if fe.get("allowed_geographies"):
            keep = keep[keep["geography"].astype(str).isin(fe["allowed_geographies"])]
        keep = keep[~keep["theme"].astype(str).isin(fe.get("excluded_themes") or [])]
        if fe.get("require_sign_prior", True):
            keep = keep[pd.to_numeric(keep["expected_sign"], errors="coerce").fillna(0) != 0]
        rep.add(OK if len(keep) >= 5 else ERROR, "features", f"{len(keep)} variables pass the geography, theme and sign filters",
                "relax allowed_geographies / excluded_themes, or add expected signs" if len(keep) < 5 else "")

    # ---------------------------------------------------------------- sample
    if level is not None:
        first, last = level.index[0], level.index[-1]
        H = int(s["horizon"])
        he, es = _q(s["hist_end"]), _q(s["estimation_start"])
        if he is None or he not in level.index:
            rep.add(ERROR, "sample", f"hist_end {s['hist_end']} is not a quarter of the target data ({first} to {last})",
                    f"set hist_end: {last}")
            he = last
        if es is None or es not in level.index:
            rep.add(ERROR, "sample", f"estimation_start {s['estimation_start']} is not a quarter of the target data",
                    f"set estimation_start: {first + 9}")
        elif _gap(es, first) < 8:
            rep.add(WARN, "sample", f"estimation_start {es} leaves only {_gap(es, first)} quarters for lags and transforms",
                    f"use {first + 9} or later")
        else:
            rep.add(OK, "sample", f"estimation {es} to {he} ({_gap(he, es) + 1} quarters)")
        bs, be = _q(s["backtest_start"]), _q(s.get("backtest_end"))
        if bs is not None and es is not None:
            if bs <= es + 20:
                rep.add(WARN if bs > es + 8 else ERROR, "sample",
                        f"backtest_start {bs} leaves {_gap(bs, es)} quarters of estimation data before the first origin",
                        f"start the backtest around {es + 40} (10 years of data) if the sample allows")
            be_eff = be or (he - 1)
            if be is not None and be >= he:
                rep.add(ERROR, "sample", f"backtest_end {be} must be before hist_end {he}", f"set backtest_end: null or {he - 1}")
            n_full = max(0, _gap(min(be_eff, he - H), bs) + 1)
            n_all = max(0, _gap(be_eff, bs) + 1)
            if n_all <= 0:
                rep.add(ERROR, "sample", f"no backtest origins between {bs} and {be_eff}", "check backtest_start / backtest_end")
            else:
                rep.add(OK if n_full >= 6 else WARN, "sample",
                        f"{n_all} backtest origins; {n_full} have an actual {H} quarters ahead",
                        f"move backtest_start earlier for more {H}-quarter errors" if n_full < 6 else "")
        out = [q for q in s.get("dummy_quarters") or [] if not (first <= _q(q) <= he)]
        if out:
            rep.add(WARN, "sample", f"dummy_quarters outside the sample: {out}", "remove them (they would do nothing)")
        if base is not None:
            if base.index[0] > (es or first):
                rep.add(ERROR, "sample", f"macro history starts {base.index[0]}, after estimation_start {es}",
                        f"set estimation_start to {base.index[0] + 9} or later")
            need = he + H
            for name, df in macro.items():
                if df.index[-1] < need:
                    rep.add(WARN, "data.macro", f"scenario '{name}' ends {df.index[-1]}: forecasts stop before {need} (hist_end + horizon)",
                            "extend the scenario files, or shorten sample.horizon")
        o = cfg.get("oos") or {}
        fo = _q(o.get("first_origin"))
        if fo is not None and es is not None:
            if fo <= es + 20 or fo >= he:
                rep.add(ERROR if fo >= he else WARN, "oos", f"first_origin {fo}: the first re-selection has "
                        f"{_gap(fo, es)} quarters of data", f"set oos.first_origin around {es + 40}")
            else:
                rep.add(OK, "oos", f"out-of-sample origins from {fo} every {o.get('step', 4)} quarters")
        if int(o.get("holdout_quarters") or 0) >= _gap(he, es or first) // 2:
            rep.add(WARN, "oos", f"holdout_quarters {o['holdout_quarters']} is more than half the sample", "use 8-12")

    # ---------------------------------------------------------------- same-growth models and models.only
    mcfg = cfg.get("models") or {}
    for item in mcfg.get("same_growth") or []:
        var = str(item.get("variable", "") if isinstance(item, dict) else item).strip()
        if not var:
            continue
        code = var
        try:
            from ..models.literature import load as _load_lit
            tbl = _load_lit(cfg) if (cfg.get("literature") or {}).get("table") else None
            if tbl is not None and var in tbl.index:
                code = tbl.loc[var, "mnemonic"] if tbl.loc[var, "active"] else None
        except Exception:
            pass
        if not code or code not in names:
            rep.add(WARN, "models", f"same_growth: '{var}' is not a series in the data (or its concept is not mapped): "
                                    "its models are skipped", "use a literature concept that is mapped, or a series name")
    only = [str(x) for x in (mcfg.get("only") or [])]
    if only:
        rep.add(OK, "models", f"only these models run (plus the AR(1) benchmark): {', '.join(only)}")

    # ---------------------------------------------------------------- yes/no written where a list is expected
    from .config import SWITCH_NOTES
    for n in SWITCH_NOTES:
        rep.add(WARN, "config", n, "write a list (e.g. [arimax, theta]) or [] to be explicit")
    for sec, key in (("features", "allowed_geographies"), ("screen_first", "primary_geography"),
                     ("en_first", "primary_geography"), ("features", "excluded_themes")):
        v = (cfg.get(sec) or {}).get(key)
        if isinstance(v, bool) or (isinstance(v, list) and any(isinstance(x, bool) for x in v)):
            rep.add(ERROR, "config", f"{sec}.{key} contains {v}: YAML reads ON / YES / NO / OFF / TRUE as true or false",
                    "put the name in quotes, e.g. primary_geography: \"ON\" or allowed_geographies: [\"ON\", \"CA\"]")

    # ---------------------------------------------------------------- literature
    lit = cfg.get("literature") or {}
    if lit.get("workbook"):
        wb = lit["workbook"]
        if not os.path.exists(wb):
            rep.add(ERROR, "literature", f"workbook not found: {wb}", "point literature.workbook to the .xlsx, or set it to null")
        else:
            try:
                from ..models.literature import read_workbook
                models, conc = read_workbook(wb)
                cids = set(conc["concept_id"].astype(str)) if "concept_id" in conc else set()
                ok_tr = set(cfg["features"]["transforms_level"]) | set(cfg["features"]["transforms_rate"]) | {"rdlog", "ryoy"}
                rep.add(OK if models else ERROR, "literature", f"workbook: {len(models)} model(s), {len(conc)} concept(s)",
                        "" if models else "add at least one model to sheet 'models'")
                if models and not any(m_.get("core") for m_ in models):
                    rep.add(WARN, "literature", "no model is marked core = yes", "mark the rows of one model core = yes: it feeds the hybrids, regime and deep models")
                for m_ in models:
                    for f in m_["features"]:
                        v, tr, lg = f.split("|")
                        if v not in cids and v not in names:
                            rep.add(ERROR, "literature", f"model '{m_['name']}': '{v}' is neither a concept in sheet 'concepts' nor a series in the data",
                                    "use a concept_id from sheet 'concepts' or an exact series name")
                        from ..features.transforms import window as _win
                        if tr not in ok_tr and not _win(tr):
                            rep.add(WARN, "literature", f"model '{m_['name']}': transform '{tr}' of {v} is not among the transforms tried",
                                    f"use one of {sorted(ok_tr)}")
                        if m_.get("signs") and m_["signs"].get(f) not in (-1, 0, 1):
                            rep.add(WARN, "literature", f"model '{m_['name']}': expected_sign of {v} is {m_['signs'].get(f)}",
                                    "use 1, -1 or 0 (0 = no sign rule)")
                from ..models.literature import clean_settings
                raw_ = read_workbook(wb, full=True)["settings"]
                st_ = clean_settings(raw_)
                if not raw_:
                    rep.add(WARN, "literature", "the workbook has no 'settings' sheet: defaults are used",
                            "re-create it with python tools/literature_template.py (keeps your models and concepts)")
                defl = st_.get("ecm_deflator") or "inflation"
                if st_["ecm_enabled"] and defl not in cids and defl not in names:
                    rep.add(ERROR, "literature", f"settings: ecm_deflator '{defl}' is neither a concept nor a series",
                            "use a concept_id (e.g. inflation) or your CPI series name")
                thr_raw = str(raw_.get("regime_threshold", "") or "").strip()
                thr = st_["regime_threshold"]
                if thr_raw and thr_raw.lower() != "nan" and not thr:
                    rep.add(WARN, "literature", f"settings: regime_threshold '{thr_raw}' is not variable|transform|Llag: the default is used",
                            "write e.g. house_prices|dlog|L1, or leave it empty")
                elif thr and thr.split("|")[0] not in cids and thr.split("|")[0] not in names:
                    rep.add(ERROR, "literature", f"settings: regime_threshold '{thr}' cannot be resolved",
                            "use a concept_id or one of your series, e.g. house_prices|dlog|L1")
                for key in ("channels", "ecm_short_run"):
                    if isinstance(st_[key], list):
                        unk = [x for x in st_[key] if x.split("|")[0] not in cids | set(names)]
                        if unk:
                            rep.add(ERROR, "literature", f"settings: {key} has unknown variables {unk}",
                                    "use 'table' / 'core', or a comma list of concepts")
                lr = st_["ecm_long_run"]
                if lr != "table":
                    bad = [x for x in lr if x.split(":")[0] not in cids | set(names)]
                    if bad:
                        rep.add(ERROR, "literature", f"settings: ecm_long_run has unknown variables {bad}",
                                "use 'table' or a comma list of concepts / series")
            except Exception as ex:
                rep.add(ERROR, "literature", f"workbook cannot be read: {ex}", "use the template: python tools/literature_template.py")
    if lit.get("table"):
        if not os.path.exists(lit["table"]):
            rep.add(ERROR, "literature", f"table not found: {lit['table']}", "point literature.table to the CSV, or set it to null")
        else:
            if lit["table"].endswith(".csv"):
                t = pd.read_csv(lit["table"])
            else:
                sh = pd.read_excel(lit["table"], sheet_name=None)
                t = next((v for k, v in sh.items() if k.strip().lower() == "concepts"), next(iter(sh.values())))
            if "moodys_name" in t.columns and "series_name" not in t.columns:   # older workbooks
                t = t.rename(columns={"moodys_name": "series_name"})
            t["series_name"] = t["series_name"].fillna("").astype(str).str.strip()
            inc = t["include"].astype(str).str.lower().isin(["yes", "y", "true", "1"])
            core = t.get("core", pd.Series("no", index=t.index)).astype(str).str.lower().isin(["yes", "y", "true", "1"])
            for _, r in t[inc].iterrows():
                nm, cid = r["series_name"], r["concept_id"]
                is_core = bool(core.loc[_])
                if not nm and not is_core:
                    rep.add(OK, "literature", f"{cid}: not mapped (the models that use it go without it)")
                elif not nm:
                    rep.add(ERROR, "literature", f"concept '{cid}'{' (core)' if is_core else ''} has no series_name",
                            "put your series name in the series_name column, or include = no")
                elif nm not in names:
                    rep.add(ERROR if is_core else WARN, "literature", f"concept '{cid}' -> '{nm}' is not in the macro data",
                            "fix the name in series_name (it must match a macro column exactly)")
                else:
                    rep.add(OK, "literature", f"{cid} -> {nm}" + (" (core)" if is_core else ""))
    return rep


def preflight(cfg, log=print, strict=True):
    """Run the check at the start of a run: print problems and stop on errors.

    Args:
        cfg: configuration. log: message function. strict: raise SystemExit on errors.

    Returns:
        Report.
    """
    rep = check(cfg)
    if rep.errors or rep.warnings:
        log("Pre-flight check of the config and data:" + rep.text(show_ok=False))
    if rep.errors and strict:
        log("\nStopped before running: fix the errors above (or run with --skip-check).")
        raise SystemExit(2)
    return rep
