"""
Macro-driver forecasting pipeline - entry point.

    python run_pipeline.py                                   # uses config/example_config.yaml
    python run_pipeline.py --config config/my_portfolio.yaml
    python run_pipeline.py --config config/my_portfolio.yaml --approach en_first

Everything specific to a data set (file names, sheet names, column names, dates, settings) is in the
config file, so the same code runs on any portfolio by pointing it at another config.

Folders
  config/            run configurations (one YAML per data set)
  data/              input files (simulated example data)
  src/utils/         config loading, package set-up, reading files
  src/features/      transformations, stationarity, candidate features
  src/selection/     univariate screen, clustering, Elastic Net (+ stability selection)
  src/modelling/     ARDL model, shared final search, backtest / diagnostics / scenarios
  src/pipelines/     screen_first.py (approach A), en_first.py (approach B)
  src/reporting/     tables, charts, summary
  tools/             simulate_data.py (makes the example data)
  outputs/<run_name>/screen_first, en_first, comparison
"""
import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.utils.env import ensure_packages  # noqa: E402

ensure_packages()

import warnings  # noqa: E402

import pandas as pd  # noqa: E402

from src.features.transforms import all_features, candidate_forms  # noqa: E402
from src.modelling.ardl import ARDL  # noqa: E402
from src.modelling.evaluate import evaluate, horizon_summary  # noqa: E402
from src.models import horizon, zoo  # noqa: E402
from src.pipelines import bayes_first, en_first, factors_first, focus_first, screen_first, stable_first  # noqa: E402
from src.reporting import formula, report, story  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.cache import StepCache, hash_code, hash_frame, hash_obj  # noqa: E402
from src.utils.io import Context, log  # noqa: E402
from src.utils import docs, saved, trace  # noqa: E402
from src.utils.preflight import preflight  # noqa: E402

warnings.filterwarnings("ignore")
APPROACHES = {"screen_first": screen_first, "en_first": en_first, "bayes_first": bayes_first,
              "focus_first": focus_first, "factors_first": factors_first, "stable_first": stable_first}


def main():
    """Command line entry: parse arguments, load the config, start the trace and run the pipeline."""
    ap = argparse.ArgumentParser(description="Macro-driver forecasting pipeline")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--approach", choices=["both", "all", "screen_first", "en_first", "bayes_first", "focus_first", "factors_first", "stable_first"], default=None,
                    help="overrides 'approaches' in the config")
    ap.add_argument("--force", action="store_true", help="recompute every step (ignore the cache)")
    ap.add_argument("--force-step", action="append", default=[],
                    help="recompute one step: features, screen_first, en_first or evaluate (repeatable)")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    ap.add_argument("--skip-check", action="store_true", help="do not stop on errors found by the pre-flight check")
    ap.add_argument("--no-trace", action="store_true", help="do not write the detailed trace (outputs/<run>/trace/)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preflight(cfg, log, strict=not args.skip_check)         # names, files and dates checked in seconds
    tracer = trace.start(cfg, "run_pipeline.py", globals(), enabled=not args.no_trace)
    try:
        run(cfg, args)
    finally:
        trace.finish(tracer, {"config": os.path.relpath(cfg["config_file"], HERE), "run_name": cfg["run_name"]}, log)
        docs.refresh(cfg, log)                       # function reference, if the code changed


def run(cfg, args):
    """The full pipeline: data -> features -> both selection approaches -> model zoo -> backtest,
    diagnostics and scenarios for every model -> tables, charts and summary.

    Args:
        cfg: configuration. args: parsed command-line arguments.
    """
    if args.approach is None:
        approaches = list(cfg["approaches"])
    else:
        approaches = ["screen_first", "en_first"] if args.approach == "both" else \
            list(APPROACHES) if args.approach == "all" else [args.approach]
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log(f"Run '{cfg['run_name']}' -> {out}")

    c = cfg["cache"]
    cache = StepCache(c["dir"], enabled=c["enabled"] and not args.no_cache, force=args.force,
                      force_steps=args.force_step)

    log("Loading data ...")
    ctx = Context(cfg)
    log(f"  target: {cfg['data']['bank']['target_column']}, {ctx.idx[0]}–{ctx.idx[-1]} ({len(ctx.idx)} quarters); "
        f"{ctx.base.shape[1]} macro columns; {len(ctx.macro)} scenarios")

    # ---- fingerprints of the inputs (content, not file names) ----
    h_y, h_level, h_meta = hash_frame(ctx.y), hash_frame(ctx.level), hash_frame(ctx.meta)
    h_hist = hash_frame(ctx.hist)
    h_scen = {k: hash_frame(v) for k, v in ctx.macro.items()}
    cs = {k: hash_obj(cfg[k]) for k in ("sample", "features", "screen_first", "en_first", "bayes_first", "final_model")}
    cs["factors_first"] = hash_obj(cfg["factors"])
    cs["stable_first"] = hash_obj(cfg["stability"])
    cs["focus_first"] = hash_obj({k: cfg["impact"].get(k) for k in ("focus", "max_controls", "control_corr", "controls_same_theme", "excluded_control_themes", "control_t")}
                                 | {"literature": cfg["literature"].get("models")})
    log("Steps (cached = loaded because nothing it depends on changed):")
    cache.run("data", {"target": h_y, "balance": h_level, "metadata": h_meta, "macro_history": h_hist,
                       "scenarios": hash_obj(h_scen)}, lambda: True, log)

    def build_features():
        forms_ = candidate_forms(cfg, ctx.hist, ctx.meta)
        if forms_.empty:
            raise RuntimeError("No candidate variables left: check allowed_geographies / excluded_themes / metadata")
        return forms_, all_features(cfg, ctx, forms_)
    feat_comp = {"macro_history": h_hist, "metadata": h_meta, "config.features": cs["features"],
                 "config.sample": cs["sample"], "code": hash_code("features/transforms.py", "utils/io.py")}
    forms, F = cache.run("features", feat_comp, build_features, log)
    h_feat = hash_obj(feat_comp)
    log(f"  {forms['mnemonic'].nunique()} variables, {int(forms['stationary'].sum())} stationary forms, "
        f"{F.shape[1]} features (with lags)")
    report.write_tables(os.path.join(out, "common"), {"stationarity_tests": forms})

    code_files = {
        "screen_first": ["selection/screening.py", "selection/clustering.py", "modelling/search.py",
                         "modelling/ardl.py", "pipelines/screen_first.py"],
        "en_first": ["selection/elastic_net.py", "selection/screening.py", "selection/clustering.py",
                     "modelling/search.py", "modelling/ardl.py", "pipelines/en_first.py"],
        "bayes_first": ["selection/bayes.py", "selection/screening.py", "selection/clustering.py",
                        "modelling/search.py", "modelling/ardl.py", "pipelines/bayes_first.py"],
        "focus_first": ["selection/double_selection.py", "pipelines/focus_first.py"],
        "factors_first": ["features/factors.py", "selection/small_search.py", "pipelines/factors_first.py"],
        "stable_first": ["selection/stability.py", "selection/small_search.py", "pipelines/stable_first.py"],
    }
    results = []
    for a in approaches:
        comp_a = {"features": h_feat, "target": h_y, "metadata": h_meta, f"config.{a}": cs[a],
                  "config.final_model": cs["final_model"], "config.sample": cs["sample"],
                  "code": hash_code(*code_files[a])}
        r = cache.run(a, comp_a, lambda a=a: APPROACHES[a].run(cfg, ctx, F, forms), log)
        log(f"  {a}: drivers {r['features']}")
        report.write_tables(os.path.join(out, a), r["tables"])
        if "B2_en_stability" in r["tables"]:
            report.chart_stability(os.path.join(out, a, "en_stability.png"), r["tables"]["B2_en_stability"],
                                   cfg["en_first"]["stability"]["threshold"])
        results.append(r)

    log("Building the competing models (benchmarks, data-driven, literature, hybrid, machine learning, AI) ...")
    sel = {r_a: zoo.selection_info(r_a, r) for r_a, r in zip(approaches, results)}
    models, notes, extra = zoo.build(cfg, ctx, sel, log)
    log(f"  {len(models)} models: " + ", ".join(f"{k} {v}" for k, v in notes["family"].value_counts(sort=False).items()))
    ev_code = hash_code("modelling/ardl.py", "modelling/evaluate.py", "modelling/search.py", "features/transforms.py",
                        "models/ecm.py", "models/ml.py", "models/combination.py", "models/foundation.py",
                        "models/deep.py", "models/regime.py", "models/simple.py", "models/autoardl.py")
    ev = {}
    for m in models:
        comp_m = {"model": hash_obj(m.spec()), "target": h_y, "balance": h_level, "metadata": h_meta,
                  "scenarios": hash_obj(h_scen), "config.sample": cs["sample"],
                  "config.deflator": hash_obj(cfg["features"]["deflator"]), "code": ev_code}
        ev[m.name] = cache.run(f"evaluate:{m.name}", comp_m, lambda m=m: evaluate(ctx, m), log)
    for k, v in ev.items():                     # lags chosen during estimation (auto-ARDL): update the model list
        if v.get("fitted") is not None and k in set(notes["model"]):
            try:
                m_ = pickle.loads(v["fitted"])
            except Exception:
                continue
            if hasattr(m_, "lag_summary"):
                i_ = notes.index[notes["model"] == k][0]
                notes.loc[i_, "drivers"] = ", ".join(m_.drivers())
                notes.loc[i_, "how it is specified"] = f"{notes.loc[i_, 'how it is specified']}. Chosen on all the " \
                                                       f"data: {m_.lag_summary()}"
    by_h = pd.concat([v["by_horizon"] for v in ev.values()])
    H = cfg["sample"]["horizon"]
    hz = cfg["sample"]["report_horizons"]
    tests = pd.concat([v["test_forecasts"] for v in ev.values() if isinstance(v.get("test_forecasts"), pd.DataFrame)],
                      ignore_index=True)
    # horizon combinations: the best model for each forecast horizon, chosen from errors already known at each origin
    hc_rows, hc_choices = horizon.combine(tests, cfg, H, lambda n: zoo.wanted(cfg, n, horizon.FAMILY))
    hc_extra = {}
    if hc_rows is not None:
        log(f"Horizon combinations ({', '.join(f'{a}-{z}' for a, z in horizon.buckets(cfg, H))} quarters ahead) ...")
        by_h = pd.concat([by_h, horizon.by_horizon(hc_rows)], ignore_index=True)
        tests = pd.concat([tests, hc_rows.drop(columns=["error", "chosen"])], ignore_index=True)
        B, fch = horizon.final_choice(tests, cfg, H)
        scen_all = pd.concat([v["scenarios"] for v in ev.values()])
        hc_extra["scen"] = horizon.scenarios(scen_all, B, fch, lambda n: zoo.wanted(cfg, n, horizon.FAMILY))
        for k, nm in horizon.NAMES.items():
            if nm in set(hc_rows["model"]):
                txt, drows = horizon.describe(B, fch, k)
                notes = pd.concat([notes, pd.DataFrame([{
                    "model": nm, "family": horizon.FAMILY, "drivers": txt,
                    "how it is specified": ("For each block of quarters ahead, the model with the smallest growth error "
                                            if k == "best" else "For each block of quarters ahead, the best "
                                            f"{cfg['models']['horizon_combination'].get('top_k', 3)} models weighted by "
                                            "1/error ") + "on forecasts already realised at each origin (no look-ahead)",
                    "source": ""}])], ignore_index=True)
                hc_extra.setdefault("drivers", []).append(pd.DataFrame(drows))
                hc_extra.setdefault("diag", []).append({"model": nm, "family": horizon.FAMILY, "note": "combination: "
                                                        "residual tests do not apply; see each member model"})
                log(f"  {nm}: {txt}")
    comp = horizon_summary(by_h, hz)
    comp = comp.merge(notes[["model", "drivers", "how it is specified", "source"]], on="model", how="left")
    comp["avg_bias_1_%d_pct" % H] = by_h.groupby("model", sort=False)["bias_exCOVID_pct"].mean().reindex(comp["model"]).values
    comp = comp.round(3)
    diag = pd.DataFrame([v["diagnostics"] for v in ev.values()] + hc_extra.get("diag", []))
    drivers = pd.concat([v["drivers"] for v in ev.values()] + hc_extra.get("drivers", []))
    scen = pd.concat([v["scenarios"] for v in ev.values()] + ([hc_extra["scen"]] if len(hc_extra.get("scen", [])) else []))
    yearend = scen[scen["date"].str.endswith("Q4")].pivot_table(index=["model", "date"], columns="scenario",
                                                                 values="balance_forecast", sort=False).round(1).reset_index()
    coef_paths = pd.concat([v["coef_path"].assign(model=k).reset_index().rename(columns={"index": "origin"})
                            for k, v in ev.items()])
    overlap = None
    if len(results) == 2:
        overlap = ", ".join(sorted(set(results[0]["features"]) & set(results[1]["features"])))

    cdir = os.path.join(out, "comparison")
    tables = {"1_model_comparison": comp, "2_backtest_by_horizon": by_h.round(4), "3_drivers": drivers,
              "4_diagnostics": diag, "5_scenario_yearend": yearend, "6_scenario_forecasts": scen,
              "7_coefficient_paths": coef_paths, "9_model_list": notes, "12_backtest_forecasts": tests.round(6)}
    if hc_choices is not None:
        tables["13_horizon_choices"] = hc_choices
    for k, v in extra.items():
        tables[f"10_{k}"] = v
    regimes = [v["regimes"] for v in ev.values() if isinstance(v.get("regimes"), pd.DataFrame)]
    if regimes:
        tables["11_regimes"] = pd.concat(regimes).round(4)
    report.write_tables(cdir, tables)
    if regimes:
        report.chart_regimes(os.path.join(cdir, "regimes.png"), tables["11_regimes"], ctx.level)
    report.chart_family_bars(os.path.join(cdir, "model_comparison.png"), comp, hz)
    report.chart_horizon(os.path.join(cdir, "backtest_by_horizon.png"), by_h, report.best_per_family(comp, H))
    report.chart_scenarios(os.path.join(cdir, "scenarios.png"), scen, ctx.level, report.scenario_models(comp, H))
    report.summary_md(os.path.join(out, "summary.md"), cfg, results, comp, diag, overlap)
    trace.add_story(lambda: story.build(cfg, ctx, forms, F, results, models, notes, ev, comp), log)

    log(f"\nResults (balance error, dummy quarters excluded; average over 1–h quarters):")
    log(f"  {'model':<44s} {'family':<20s} " + " ".join(f"{'1-' + str(h):>7s}" for h in hz))
    for _, r in comp.sort_values(f"avg_1_{H}_pct").iterrows():
        log(f"  {r['model']:<44s} {r['family']:<20s} " + " ".join(f"{r[f'avg_1_{h}_pct']:6.2f}%" for h in hz))
    if cfg["save"].get("fitted_models", True):
        saved.save_all(out, ev, ctx, log)
    all_fitted = {}
    for k, v in ev.items():
        if v.get("fitted") is not None:
            try:
                all_fitted[k] = pickle.loads(v["fitted"])
            except Exception:
                pass
    if (cfg.get("impact") or {}).get("enabled", True):
        log("\nHow the economy affects the target (impact/) ...")
        try:
            from src.impact import analysis as impact
            impact.run(cfg, ctx, all_fitted, out, log)
        except Exception as ex:
            log(f"  impact analysis not written ({type(ex).__name__}: {ex})")
    log("\nThe recommended model: every model scored on forecasting, interpretation and sensitivity ...")
    order, why_rec = None, None
    try:
        from src.selection import scorecard
        S = scorecard.score(cfg, all_fitted, comp, out, "backtest", log)
        if S is not None:
            order, why_rec = list(S["model"]), scorecard.why_text(S.iloc[0], S.attrs["weights"], S.attrs["col"])
    except Exception as ex:
        log(f"  scorecard not written ({type(ex).__name__}: {ex})")
    fitted = {k: m_ for k, m_ in all_fitted.items() if formula.has_formula(m_)}
    try:
        formula.write(cfg, ctx, fitted, comp, "backtest", log, order=order, why_rec=why_rec)
    except Exception as ex:
        log(f"  formula: not written ({type(ex).__name__}: {ex})")
    steps = cache.write_manifest(out, log)
    report.write_tables(os.path.join(out, "comparison"), {"8_run_steps": steps})
    saved.snapshot(cfg, "pipeline", log)
    log(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
