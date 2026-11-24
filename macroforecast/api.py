"""One call that runs the whole project: give the target series, the macro metadata and the literature workbook;
everything else has a default and can be changed in `options`. (macroforecast.Forecaster wraps this.)

    import macroforecast as mf

    res = mf.run(target="data/bank_data.csv",                 # date + balance columns (csv or xlsx)
                 metadata="data/macro_metadata.csv",
                 literature="config/literature_workbook.xlsx")
    res.comparison          # every model ranked by backtest error
    res.oos                 # out-of-sample test (re-selection at each origin)
    res.best                # recommended model, its error and the basis
    res.forecasts           # scenario forecasts of the target
    res.out_dir             # all tables, charts, formula and calculator

Anything else goes in `options`: nested ({"sample": {"horizon": 8}}), dotted ({"sample.horizon": 8}) or a list
of "key=value" strings (["sample.horizon=8", "en_first.target_n=8"]). Keys are the same as in the YAML config.
"""
import copy
import glob
import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = os.path.join(PROJECT_ROOT, "config", "example_config.yaml")
DATE_KEYS = ("hist_end", "estimation_start", "backtest_start")


# ----------------------------------------------------------------------------------------------- options
def _set(d, dotted, value):
    """Set d["a"]["b"]["c"] = value for dotted = "a.b.c" (creating the levels)."""
    keys = dotted.split(".")
    for k in keys[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _merge(base, over):
    """Recursive merge, `over` wins (lists and values are replaced, dicts merged)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def parse_options(options):
    """Options in any of the accepted forms -> one nested dict.

    Args:
        options: None, a dict (nested and/or dotted keys), a list of "key=value" strings, or a YAML file path.

    Returns:
        nested dict.
    """
    if options is None:
        return {}
    if isinstance(options, str) and os.path.exists(options):
        with open(options, encoding="utf-8") as f:
            options = yaml.safe_load(f) or {}
    if isinstance(options, (list, tuple)):
        pairs = {}
        for item in options:
            if "=" not in str(item):
                raise ValueError(f"option '{item}' should look like key=value, e.g. sample.horizon=8")
            k, v = str(item).split("=", 1)
            pairs[k.strip()] = yaml.safe_load(v.strip())       # 8 -> int, [a, b] -> list, true -> bool
        options = pairs
    if not isinstance(options, dict):
        raise TypeError("options must be a dict, a list of 'key=value' strings or a YAML file")
    out = {}
    for k, v in options.items():
        if "." in str(k):
            nested = {}
            _set(nested, str(k), v)
            out = _merge(out, nested)
        else:
            out = _merge(out, {k: v})
    return out


def _abs(p):
    """A path given by the user, relative to where they run Python (not the project folder)."""
    return None if p is None else os.path.abspath(os.path.expanduser(str(p)))


# ----------------------------------------------------------------------------------------------- target
def _target_file(target, target_column, date_column, inputs_dir):
    """Accept a file, a pandas Series or a DataFrame; return (file, target_column, date_column).

    A Series or DataFrame is written to <out>/inputs/target.csv so the run is reproducible.
    """
    if isinstance(target, (pd.Series, pd.DataFrame)):
        df = target.to_frame(target.name or target_column or "target") if isinstance(target, pd.Series) \
            else target.copy()
        if date_column is None and "date" in df.columns:
            date_column = "date"
        if date_column is None or date_column not in df.columns:
            df = df.reset_index()
            date_column = df.columns[0] if date_column is None or date_column not in df.columns else date_column
        df[date_column] = df[date_column].astype(str)
        os.makedirs(inputs_dir, exist_ok=True)
        path = os.path.join(inputs_dir, "target.csv")
        df.to_csv(path, index=False)
        target = path
    path = _abs(target)
    if not os.path.exists(path):
        raise FileNotFoundError(f"target file not found: {path}")
    from src.utils.io import read_table
    head = read_table(path)
    if date_column is None:
        date_column = "date" if "date" in head.columns else head.columns[0]
    if date_column not in head.columns:
        raise KeyError(f"date column '{date_column}' not in {path}: columns are {list(head.columns)}")
    if target_column is None:
        num = [c for c in head.columns if c != date_column and pd.api.types.is_numeric_dtype(head[c])]
        if len(num) != 1:
            raise ValueError(f"{path} has {len(num)} numeric columns {num}: say which one with target_column=")
        target_column = num[0]
    if target_column not in head.columns:
        raise KeyError(f"target column '{target_column}' not in {path}: columns are {list(head.columns)}")
    return path, target_column, date_column


def infer_dates(cfg):
    """Sample dates from the data, used for any of hist_end / estimation_start / backtest_start not given.

    hist_end           last quarter with a balance
    estimation_start   first quarter with both a balance and macro data, plus room for lags and 4-quarter changes
    backtest_start     8 years before hist_end, but at least 10 years after estimation_start

    Returns:
        dict of the inferred dates (strings like 2018Q1).
    """
    from src.utils.io import read_table, to_quarters
    b = cfg["data"]["bank"]
    df = read_table(b["file"], b.get("sheet"))
    s = pd.to_numeric(df[b["target_column"]], errors="coerce")
    s.index = to_quarters(df[b.get("date_column", "date")])
    s = s.dropna().sort_index()
    m = cfg["data"]["macro"]
    base = m["scenarios"][m["base_scenario"]]
    mq = to_quarters(read_table(base, m.get("sheet"))[m.get("date_column", "date")])
    lag = int((cfg.get("features") or {}).get("max_lag", 4))
    first = max(s.index[0], mq.min())
    end = s.index[-1]
    est = first + lag + 5                                   # 4 for year-on-year changes, 1 for the growth rate
    bt = max(end - 31, est + 40)
    if bt >= end - 4:
        raise ValueError(f"only {end - first} quarters of data ({first}-{end}): too short for a backtest; "
                         "set sample.estimation_start and sample.backtest_start in options")
    return {"hist_end": str(end), "estimation_start": str(est), "backtest_start": str(bt)}


# ----------------------------------------------------------------------------------------------- results
class Results:
    """What a run produced. Tables are read from the output folder when first asked for."""

    def __init__(self, out_dir, config_file, log_lines=None):
        self.out_dir = out_dir
        self.config_file = config_file
        self.log = log_lines or []

    def __repr__(self):
        b = self.best or {}
        return (f"Results({self.out_dir})\n  best: {b.get('best_model', '?')} "
                f"({b.get('best_error_pct', '?')}% {b.get('horizon', '')}, {b.get('basis', '')})\n"
                f"  tables: res.comparison, res.by_horizon, res.oos, res.forecasts, res.drivers, res.formula; "
                f"res.table('<name>'), res.files()")

    def _csv(self, sub, name):
        p = os.path.join(self.out_dir, sub, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    @property
    def config(self):
        with open(self.config_file, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def comparison(self):
        """Every model ranked by backtest balance error (comparison/1_model_comparison.csv)."""
        return self._csv("comparison", "1_model_comparison.csv")

    @property
    def by_horizon(self):
        return self._csv("comparison", "2_backtest_by_horizon.csv")

    @property
    def drivers(self):
        return self._csv("comparison", "3_drivers.csv")

    @property
    def forecasts(self):
        """Scenario forecasts of the balance for every model (comparison/6_scenario_forecasts.csv)."""
        return self._csv("comparison", "6_scenario_forecasts.csv")

    @property
    def oos(self):
        """Out-of-sample summary (oos/1_oos_summary.csv), or None if the test was not run."""
        return self._csv("oos", "1_oos_summary.csv")

    @property
    def formula(self):
        """The recommended model's terms (formula/1_terms.csv)."""
        return self._csv("formula", "1_terms.csv")

    @property
    def best(self):
        """Most accurate model: out of sample when available, otherwise backtest."""
        from src.utils.saved import _best
        return _best(self.out_dir)

    @property
    def summary(self):
        p = os.path.join(self.out_dir, "summary.md")
        return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

    def files(self, pattern="*"):
        """Output files (not saved runs, traces or the cache), relative to out_dir."""
        hits = glob.glob(os.path.join(self.out_dir, "**", pattern), recursive=True)
        skip = ("saved_runs", "trace")
        return sorted(os.path.relpath(h, self.out_dir) for h in hits
                      if os.path.isfile(h) and not os.path.relpath(h, self.out_dir).startswith(skip))

    def table(self, name):
        """Any CSV of the run by (part of) its name, e.g. res.table('B2_en_stability')."""
        hits = [f for f in self.files("*.csv") if name in os.path.basename(f)]
        if not hits:
            raise KeyError(f"no table matching '{name}'; see res.files('*.csv')")
        return pd.read_csv(os.path.join(self.out_dir, hits[0]))

    def models(self):
        """The fitted models, ready to forecast (see src/utils/saved.py)."""
        from src.utils.saved import load_models
        return load_models(self.out_dir)


# ----------------------------------------------------------------------------------------------- run
def build_config(target, metadata, literature=None, *, target_column=None, date_column=None, macro=None,
                 base_scenario=None, events=None, options=None, base_config=None, run_name=None,
                 output_dir=None, panel=None):
    """The configuration a run will use (without running anything).

    Returns:
        (config dict ready to write as YAML, out_dir).
    """
    with open(_abs(base_config) if base_config else DEFAULT_BASE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    opts = parse_options(options)
    cfg = _merge(cfg, opts)

    # the example config's same-growth models name mortgage concepts: keep those the workbook has
    if base_config is None and "same_growth" not in (opts.get("models") or {}):
        cfg.setdefault("models", {})
        keep = []
        if literature is not None:
            try:
                cs = pd.read_excel(_abs(literature), sheet_name="concepts")
                lt = (opts.get("literature") or {}).get("target") or run_name
                if "target" in cs and lt:                      # a master file: this target's concepts
                    k = lambda x: str(x).strip().lower().replace(" ", "_").replace("-", "_")
                    cs = cs[cs["target"].map(k) == k(lt)]
                ids = set(cs["concept_id"].astype(str))
            except Exception:
                ids = set()
            keep = [c for c in cfg["models"].get("same_growth", []) if c in ids]
        cfg["models"]["same_growth"] = keep

    out_root = _abs(output_dir) if output_dir else os.path.join(PROJECT_ROOT, cfg.get("output_dir", "outputs"))
    cfg["output_dir"] = out_root

    d = cfg.setdefault("data", {})
    bank = d.setdefault("bank", {})
    inputs = os.path.join(out_root, run_name or "macroforecast_inputs", "inputs")
    if target is not None:
        f, col, dcol = _target_file(target, target_column or (None if base_config is None else bank.get("target_column")),
                                    date_column, inputs)
        bank.update(file=f, target_column=col, date_column=dcol, sheet=None)
        if events is None and base_config is None:
            d["events"] = {"file": None}
    if panel:                                    # several banks in one file: the column naming the bank
        bank["panel_column"] = panel
        d.setdefault("events", {}).setdefault("bank_column", panel)
    cfg["run_name"] = run_name or (cfg.get("run_name") if base_config else f"{bank['target_column']}")
    if events is not None:
        d.setdefault("events", {})["file"] = _abs(events)

    if metadata is not None:
        d.setdefault("metadata", {})["file"] = _abs(metadata)
    if macro is not None:
        scen = {"Baseline": macro} if isinstance(macro, (str, os.PathLike)) else dict(macro)
        d.setdefault("macro", {})["scenarios"] = {k: _abs(v) for k, v in scen.items()}
        d["macro"]["base_scenario"] = base_scenario or next(iter(scen))
    elif base_scenario:
        d["macro"]["base_scenario"] = base_scenario
    else:                                                   # base-config paths are relative to the project
        d["macro"]["scenarios"] = {k: (v if os.path.isabs(v) else os.path.join(PROJECT_ROOT, v))
                                   for k, v in d["macro"]["scenarios"].items()}
    if macro is not None and base_config is None and d.get("constructed"):   # drop built series whose inputs are missing
        import re
        from src.utils.io import read_table
        cols = set(read_table(d["macro"]["scenarios"][d["macro"]["base_scenario"]]).columns)
        d["constructed"] = [c for c in d["constructed"]
                            if set(re.findall(r"`([^`]+)`", c.get("expression", ""))) <= cols]
    for key in ("bank", "events", "metadata"):
        p = (d.get(key) or {}).get("file")
        if p and not os.path.isabs(p):
            d[key]["file"] = os.path.join(PROJECT_ROOT, p)

    lit = cfg.setdefault("literature", {})
    if literature is not None:
        lit["workbook"] = _abs(literature)
    elif base_config is None:                                # no workbook: the theory-based models are off
        lit["workbook"] = None
        m = cfg.setdefault("models", {})
        m.setdefault("literature", False)
        m["literature"] = (opts.get("models") or {}).get("literature", False)
        m["hybrids"] = (opts.get("models") or {}).get("hybrids", False)
        aa = m.setdefault("auto_ardl", {})
        aa["on"] = [x for x in aa.get("on", ["core", "screen_first"]) if x != "core"] or ["screen_first"]
    elif lit.get("workbook") and not os.path.isabs(lit["workbook"]):
        lit["workbook"] = os.path.join(PROJECT_ROOT, lit["workbook"])

    given = (opts.get("sample") or {})
    if base_config is None or target is not None:
        cfg.setdefault("sample", {})
        for k in DATE_KEYS:
            if k not in given:
                cfg["sample"].pop(k, None)
        missing = [k for k in DATE_KEYS if k not in cfg["sample"]]
        if missing:
            guess = infer_dates(cfg)
            for k in missing:
                cfg["sample"][k] = guess[k]
            if "backtest_end" not in given:
                cfg["sample"]["backtest_end"] = None
            if "first_origin" not in (opts.get("oos") or {}):
                cfg.setdefault("oos", {})["first_origin"] = None
    return cfg, os.path.join(out_root, cfg["run_name"])


def _script(name):
    """run_pipeline / run_oos imported as modules from the project folder."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    return importlib.import_module(name)


def run(target, metadata, literature=None, *, target_column=None, date_column=None, macro=None,
        base_scenario=None, events=None, options=None, base_config=None, run_name=None, output_dir=None,
        oos=True, deck=False, approach=None, force=False, cache=True, trace=False, check=True, panel=None):
    """Run the whole project: selection, every model, backtest, scenarios, formula and (by default) the
    out-of-sample test. Returns a Results object.

    Args:
        target: the series to forecast: a csv/xlsx file with a date column and the balance, or a pandas Series /
            DataFrame indexed by quarter. Inorganic events are not removed unless `events` is given.
        metadata: macro metadata file (mnemonic, description, geography, theme, transform, expected_sign).
        literature: literature workbook (.xlsx). None switches the literature, hybrid and same-growth models off.
        target_column / date_column: needed only when the file has several numeric columns / no 'date' column.
        macro: scenario files: {"Baseline": path, "S4 Severe": path, ...} or one path. Default: the example scenario
            files of the base config (data/scenario_*.csv in the project).
        base_scenario: the scenario whose history is used for estimation (default: the first one).
        events: file of inorganic events (date, series, amount), or None.
        options: any other setting, in config keys: dict (nested or dotted), list of "key=value" strings, or a
            YAML file. Example: {"sample.horizon": 8, "en_first.target_n": 8, "models.machine_learning": []}.
            Sample dates not given are taken from the data (see infer_dates).
        base_config: a YAML config to start from instead of config/example_config.yaml.
        run_name: output folder name (default: the target column). output_dir: parent folder (default outputs/).
        oos: also run the out-of-sample test. deck: also build the PowerPoint deck (needs Quarto).
        approach: "screen_first", "en_first" or None (the config's list). force: recompute every step.
        cache: reuse unchanged steps. trace: write the detailed function trace. check: stop on pre-flight errors.

    Returns:
        Results.
    """
    cfg, out_dir = build_config(target, metadata, literature, target_column=target_column,
                                date_column=date_column, macro=macro, base_scenario=base_scenario, events=events,
                                options=options, base_config=base_config, run_name=run_name,
                                output_dir=output_dir, panel=panel)
    os.makedirs(out_dir, exist_ok=True)
    cfg_file = os.path.join(out_dir, "run_config.yaml")
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write("# Written by macroforecast.run(); rerun with: python run_pipeline.py --config <this file>\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    from src.utils.config import load_config
    from src.utils.io import log
    from src.utils.preflight import preflight
    from src.utils import docs, trace as tr
    s = cfg["sample"]
    log(f"macroforecast: target {cfg['data']['bank']['target_column']}, sample {s['estimation_start']}-"
        f"{s['hist_end']}, backtest from {s['backtest_start']} -> {out_dir}")
    args = SimpleNamespace(approach=approach, force=force, force_step=[], no_cache=not cache,
                           skip_check=not check, no_trace=not trace)
    if cfg["data"]["bank"].get("panel_column"):      # several banks: the panel run
        c = load_config(cfg_file)
        _script("run_panel").check(c, strict=check)
        from src.panel import run as panel_run
        panel_run.run(c, out_dir, oos=oos, log=log, force=force)
        if (c.get("panel") or {}).get("per_bank_full"):
            _per_bank_full(c, cfg, out_dir, oos, check, force=force)
        return Results(out_dir, cfg_file)
    steps = [("run_pipeline", "run_pipeline.py", "trace")] + ([("run_oos", "run_oos.py", "trace_oos")] if oos else [])
    for mod_name, script, folder in steps:
        mod = _script(mod_name)
        c = load_config(cfg_file)
        c["trace"]["folder"] = folder
        preflight(c, log, strict=check)
        tracer = tr.start(c, script, vars(mod), enabled=trace)
        try:
            mod.run(c, args)
        finally:
            tr.finish(tracer, {"config": cfg_file, "run_name": c["run_name"]}, log)
    docs.refresh(load_config(cfg_file), log)
    if deck:
        subprocess.call([sys.executable, os.path.join(PROJECT_ROOT, "tools", "make_deck.py"), "--config", cfg_file])
    res = Results(out_dir, cfg_file)
    with open(os.path.join(out_dir, "macroforecast_result.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg_file, "best": res.best, "oos": bool(oos)}, f, indent=1)
    return res


def _per_bank_full(c, cfg, out_dir, oos, check, force=False):
    """The complete single-series project for the chosen banks (every model family), in <out>/banks/<bank>/.

    panel.per_bank_full: true (every bank in the panel) or a list of bank names. A bank whose data, events,
    macro files, literature, settings and code are unchanged since its last run is not run again.
    """
    import copy as _copy
    from src.panel import reuse
    from src.panel.data import bank_events, load_long
    from src.utils.io import log
    levels = load_long(c)
    chosen = reuse.own_run_banks(c, levels)
    skipped = [b for b in levels if b not in chosen]
    if skipped:
        log(f"Own single-series runs: {', '.join(chosen) or 'none'} (not requested: {', '.join(skipped)})")
    for b in chosen:
        lvl = levels[b]
        ev = bank_events(c, b)
        comps = reuse.bank_components(c, b, lvl, ev)
        fp = reuse.bank_path(out_dir, b)
        ok, why = reuse.compare(comps, fp, need_oos=oos)
        done = os.path.exists(os.path.join(os.path.dirname(fp), "macroforecast_result.json"))
        if ok and done and not force:
            log(f"Full single-series run for {b}: nothing changed ({why}) - reusing it")
            continue
        log(f"Full single-series run for {b} ({'forced' if force else why}) ...")
        o = _copy.deepcopy(cfg)
        for k in ("out_dir", "config_file"):
            o.pop(k, None)
        o["data"]["bank"].pop("panel_column", None)
        o.pop("panel", None)
        inp = os.path.join(out_dir, "banks", "_inputs")
        os.makedirs(inp, exist_ok=True)
        ef = None
        if ev:
            ef = os.path.join(inp, f"{b}_events.csv".replace(" ", "_"))
            pd.DataFrame({"date": [str(q) for q in ev], "series": c["data"]["bank"]["target_column"],
                          "amount": list(ev.values())}).to_csv(ef, index=False)
        series = lvl.rename(c["data"]["bank"]["target_column"])
        series.index = series.index.astype(str)
        for k in ("hist_end", "estimation_start", "backtest_start", "backtest_end"):
            o["sample"].pop(k, None)
        o.setdefault("oos", {})["first_origin"] = None
        run(series.rename_axis("date"), o["data"]["metadata"]["file"], (o.get("literature") or {}).get("workbook"),
            target_column=c["data"]["bank"]["target_column"], events=ef, options=o, run_name=str(b).replace(" ", "_"),
            output_dir=os.path.join(out_dir, "banks"), oos=oos, check=check, force=force)
        reuse.save(fp, comps, oos, {"bank": b})


def load(out_dir):
    """Results of an earlier run, without running anything: mf.load('outputs/loan_balance')."""
    out_dir = _abs(out_dir)
    return Results(out_dir, os.path.join(out_dir, "run_config.yaml"))
