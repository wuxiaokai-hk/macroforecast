"""Load the YAML configuration, fill defaults, resolve paths and check that the required names are there."""
import copy
import os

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    "run_name": "run",
    # how the target is modelled: dlog (log growth: balances, deposits, investment) or diff (change: interest
    # rates, spreads); auto = dlog when every value is positive. label/units are used in titles and tables.
    "target": {"transform": "auto", "label": None, "units": None, "display_scale": None, "display_units": None},
    "output_dir": "outputs",
    "data": {
        "bank": {"sheet": None, "date_column": "date", "frequency": "Q", "panel_column": None},
        "events": {"file": None, "sheet": None, "date_column": "date", "series_column": "series",
                   "amount_column": "amount", "bank_column": None},
        "macro": {"date_column": "date", "sheet": None},
        "metadata": {"sheet": None, "columns": {"mnemonic": "mnemonic", "description": "description",
                                                "geography": "geography", "theme": "theme",
                                                "transform": "transform", "expected_sign": "expected_sign"}},
        "constructed": [],
    },
    "sample": {"horizon": 12, "report_horizons": [4, 8, 12], "dummy_quarters": [], "backtest_end": None,
               "backtest_step": 1, "reestimate_every": 4, "reestimate_quarter": 1},
    "features": {"max_lag": 4, "adf_p": 0.05, "transforms_level": ["dlog", "yoy", "dlog2", "acc", "gap"],
                 "transforms_rate": ["diff", "diff2", "diff4", "lvl", "rgap"], "real_themes": [],
                 "moving_average": [], "volatility": [], "volatility_sign": -1,
                 "deflator": None, "allowed_geographies": [], "excluded_themes": [], "require_sign_prior": True,
                 "form_rule": "all", "lags": None},
    "approaches": ["screen_first", "en_first", "bayes_first"],
    "screen_first": {"screen_p": 0.05, "cluster_corr": 0.75, "primary_geography": None, "rep_tolerance": 0.75,
                     "shortlist_max": 25},
    "en_first": {"l1_ratio": 0.5, "target_n": 10, "unpenalized_controls": True, "sign_constraints": True,
                 "stability": {"enabled": True, "n_subsamples": 100, "fraction": 0.75, "block_length": 8,
                               "threshold": 0.6, "seed": 42},
                 "group_corr": 0.75, "primary_geography": None, "rep_tolerance": 0.75},
    "bayes_first": {"g": "bric", "model_size": 4, "max_vars": 6, "max_candidates": 40, "candidate_corr": 0.75,
                    "iterations": 20000,
                    "burn_in": 2000, "seed": 42, "threshold": 0.5, "group_corr": 0.75, "primary_geography": None,
                    "rep_tolerance": 0.75, "average": {"enabled": True, "mass": 0.9, "max_models": 20}},
    "literature": {"master": None, "target": None, "names": None, "workbook": None, "table": None, "apply_signs_to_metadata": True, "models": [], "ecm": None,
                   "channels": "table"},
    "models": {"only": [], "tested": {"enabled": True},
               "same_growth": ["disposable_income", "house_prices"],
               "auto_ardl": {"enabled": True, "criterion": "bic", "max_p": 4, "max_q": 4, "levels_counterpart": True,
                             "on": ["core", "screen_first"], "levels": True},
               "horizon_combination": {"enabled": True, "buckets": [[1, 4], [5, 8], [9, 12]], "top_k": 3,
                                       "min_history": 3, "criterion": "growth"},
               "benchmarks": True, "literature": True, "hybrids": True, "hybrid_max_add": 2, "channel_prune_t": 1.0,
               "machine_learning": ["ridge", "random_forest", "gradient_boosting", "neural_net"],
               "machine_learning_settings": {},
               "foundation": {"enabled": "auto", "device": "cpu", "models": [
                   {"name": "Chronos-Bolt (own history)", "id": "amazon/chronos-bolt-small"},
                   {"name": "Chronos-2 (with drivers)", "id": "amazon/chronos-2", "covariates": "core"}]},
               "statistical": ["arimax", "theta"],
               "regime": {"models": ["markov", "threshold", "smooth_transition", "tvp", "break_adaptive"],
                          "threshold": "house_prices|dlog|L1", "settings": {}},
               "deep": {"enabled": "auto", "models": ["tide", "nhits", "tft"], "settings": {}}},
    "cache": {"enabled": True, "dir": "cache"},
    "save": {"fitted_models": True, "keep_runs": 20, "formula_quarters": 4, "origin_models": True,
             "fitted_values": True, "panels": True},
    "docs": {"enabled": True, "folder": "docs", "examples": True},
    "trace": {"enabled": True, "folder": "trace", "preview_rows": 20, "save_tables": True, "detail_calls": 1,
              "distinct_labels": 8, "keep_runs": 10, "max_table_cells": 200000,
              "local_variables": True, "local_preview_rows": 8, "library_functions": "default", "follow": "auto"},
    "factors": {"enabled": True, "targeted": True, "t_threshold": 1.65, "min_members": 3, "n_factors": 1,
                "exclude_outcomes": True, "max_vars": 4, "t_min": 1.0},
    "stability": {"enabled": True, "draws": 100, "q": 8, "threshold": 0.6, "block": 4, "cluster_corr": 0.75,
                  "seed": 7, "max_vars": 4, "t_min": 1.0},
    "impact": {"enabled": True, "focus": [], "max_controls": 4, "control_corr": 0.9, "controls_same_theme": False, "excluded_control_themes": "auto",
               "compare_horizon": 8, "tied_alpha": 0.10, "tied_max": 8, "tied_max_ratio": 2.0, "tied_min": 3,
               "control_t": 1.0, "horizons": [1, 4, 8, 12],
               "lookalike_corr": 0.75, "draws": 300, "seed": 42, "attribution": True, "attribution_models": "explanatory+best", "what_if": [], "episodes": []},
    "recommend": {"weights": {"accuracy": 0.5, "interpretation": 0.25, "sensitivity": 0.25}, "max_error_ratio": 1.5,
                  "min_coverage": 0.25, "max_drivers": 5, "transparency": {}, "require_stable": True},
    "oos": {"first_origin": None, "step": 4, "horizon": None, "window": "expanding", "window_length": 40,
            "holdout_quarters": 8, "macro_vintages": {"file": None},
            "window_check": {"enabled": True, "recent_quarters": [60]}},
    "final_model": {"min_vars": 1, "max_vars": 3, "t_min": 1.96, "vif_max": 5.0, "one_per_theme": True,
                    "required_themes": [], "n_oos_candidates": 300, "selection_tolerance": 0.05},
}

REQUIRED = [
    ("data", "bank", "file"), ("data", "bank", "target_column"),
    ("data", "macro", "scenarios"), ("data", "macro", "base_scenario"),
    ("data", "metadata", "file"),
    ("sample", "hist_end"), ("sample", "estimation_start"), ("sample", "backtest_start"),
]


SWITCH_NOTES = []


def _fix_switches(cfg, base=None, path=()):
    """Read true/false/yes/no written where a list or a block is expected the way it is meant.

    YAML turns the words true, yes, on (and ON, the Ontario code) into True, and false, no, off into False. A switch
    written in place of a list or block (e.g. statistical: yes, deep: false) would otherwise stop the run with
    "'bool' object is not iterable". True keeps the default list or block (switched on), False switches it off.

    Args:
        cfg: merged configuration (changed in place). base: the defaults at the same level. path: keys so far.

    Returns:
        list of "path: value -> meaning" notes, for the log.
    """
    base = DEFAULTS if base is None else base
    notes = []
    for k, d in base.items():
        if k not in cfg:
            continue
        v, where = cfg[k], ".".join(path + (k,))
        if isinstance(v, bool) and isinstance(d, list):
            cfg[k] = copy.deepcopy(d) if v else []
            notes.append(f"{where}: {v} read as {'the default list ' + str(d) if v else 'switched off ([])'}")
        elif isinstance(v, bool) and isinstance(d, dict):
            cfg[k] = copy.deepcopy(d)
            if "enabled" in d:
                cfg[k]["enabled"] = v if not v else d["enabled"]
            elif "models" in d and not v:
                cfg[k]["models"] = []
            notes.append(f"{where}: {v} read as {'the default settings' if v else 'switched off'}")
        elif isinstance(v, dict) and isinstance(d, dict):
            notes += _fix_switches(v, d, path + (k,))
    return notes


def _merge(base, over):
    """Recursively merge a user config over the defaults (user values win).

    Args:
        base: defaults. over: user values.

    Returns:
        new dict.
    """
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _path(p):
    """Resolve a path relative to the project folder.

    Args:
        p: path or None.

    Returns:
        absolute path or None.
    """
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def load_config(path):
    """Read a YAML config, fill defaults, check required keys and normalise values.

    Args:
        path: config file.

    Returns:
        dict (also holds out_dir and config_file).

    Raises:
        KeyError / ValueError on missing keys or invalid settings.
    """
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    d_ = user.get("data") or {}
    if isinstance(d_.get("target"), dict):               # data.target is another name for data.bank
        d_["bank"] = {**(d_.get("bank") or {}), **d_.pop("target")}
    cfg = _merge(DEFAULTS, user)
    SWITCH_NOTES[:] = _fix_switches(cfg)             # reported by the pre-flight check
    if cfg["literature"].get("master") and not cfg["data"]["metadata"].get("file"):
        cfg["data"]["metadata"]["file"] = cfg["literature"]["master"]   # the master holds the metadata too
    missing = []
    for keys in REQUIRED:
        node = cfg
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if node in (None, "", {}):
            missing.append(".".join(keys))
    if missing:
        raise ValueError(f"Config {path} is missing: {', '.join(missing)}")
    d = cfg["data"]
    d["bank"]["file"] = _path(d["bank"]["file"])
    d["events"]["file"] = _path(d["events"]["file"])
    d["metadata"]["file"] = _path(d["metadata"]["file"])
    lit = cfg["literature"]
    for k in ("master", "workbook", "names"):
        if lit.get(k):
            lit[k] = _path(lit[k])
    from .master import apply as apply_master     # one master file for many targets: take this target's rows
    used = apply_master(cfg, os.path.join(_path(cfg["output_dir"]), cfg["run_name"], "inputs", "master"))
    if used:
        lit["target"] = used
    d["macro"]["scenarios"] = {k: _path(v) for k, v in d["macro"]["scenarios"].items()}
    if d["macro"]["base_scenario"] not in d["macro"]["scenarios"]:
        raise ValueError(f"base_scenario '{d['macro']['base_scenario']}' is not one of the scenarios")
    for label, f in [("bank", d["bank"]["file"]), ("metadata", d["metadata"]["file"])] + \
            [(f"scenario {k}", v) for k, v in d["macro"]["scenarios"].items()]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{label} file not found: {f}")
    for k in ("hist_end", "estimation_start", "backtest_start"):
        cfg["sample"][k] = str(cfg["sample"][k])
    if cfg["sample"].get("backtest_end"):
        cfg["sample"]["backtest_end"] = str(cfg["sample"]["backtest_end"])
    for k in ("backtest_step", "reestimate_every", "reestimate_quarter"):
        cfg["sample"][k] = int(cfg["sample"][k])
    if not 1 <= cfg["sample"]["reestimate_quarter"] <= 4:
        raise ValueError("sample.reestimate_quarter must be 1, 2, 3 or 4")
    cfg["sample"]["dummy_quarters"] = [str(q) for q in cfg["sample"]["dummy_quarters"]]
    H = int(cfg["sample"]["horizon"])                      # the full horizon is always one of the comparison points
    cfg["sample"]["report_horizons"] = sorted({int(h) for h in cfg["sample"]["report_horizons"] if int(h) <= H} | {H})
    cfg["out_dir"] = os.path.join(_path(cfg["output_dir"]), cfg["run_name"])
    cfg["cache"]["dir"] = _path(cfg["cache"]["dir"])
    if cfg["literature"].get("workbook"):                  # the workbook replaces literature.models and .table
        cfg["literature"]["workbook"] = _path(cfg["literature"]["workbook"])
        if os.path.exists(cfg["literature"]["workbook"]):
            from ..models.literature import apply_settings, read_workbook
            wb = read_workbook(cfg["literature"]["workbook"], full=True)
            cfg["literature"]["models"] = wb["models"]
            cfg["literature"]["table"] = cfg["literature"]["workbook"]
            apply_settings(cfg, wb["settings"])
    if cfg["literature"].get("table"):
        cfg["literature"]["table"] = _path(cfg["literature"]["table"])
    if not cfg["oos"]["first_origin"]:                    # default: 3 years after the backtest start
        cfg["oos"]["first_origin"] = str(pd.Period(cfg["sample"]["backtest_start"], freq="Q") + 12)
    cfg["oos"]["first_origin"] = str(cfg["oos"]["first_origin"])
    if cfg["oos"]["window"] not in ("expanding", "rolling"):
        raise ValueError("oos.window must be 'expanding' or 'rolling'")
    for key, ap in (("factors", "factors_first"), ("stability", "stable_first")):   # new approaches, on by default
        if cfg[key].get("enabled", True) and ap not in cfg["approaches"]:
            cfg["approaches"] = list(cfg["approaches"]) + [ap]
        if not cfg[key].get("enabled", True):
            cfg["approaches"] = [a for a in cfg["approaches"] if a != ap]
    imp = cfg["impact"]                                    # the explanatory model is one more approach
    if imp.get("enabled", True) and "focus_first" not in cfg["approaches"]:
        cfg["approaches"] = list(cfg["approaches"]) + ["focus_first"]
    if not imp.get("enabled", True):
        cfg["approaches"] = [a for a in cfg["approaches"] if a != "focus_first"]
    cfg["config_file"] = os.path.abspath(path)
    return cfg
