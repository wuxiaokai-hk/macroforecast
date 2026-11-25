"""
Out-of-sample test - entry point.

    python run_oos.py                                    # example config
    python run_oos.py --config config/my_portfolio.yaml
    python run_oos.py --config config/my_portfolio.yaml --approach en_first

At each forecast origin the full selection is re-run with data up to that origin only, then the chosen
model forecasts the next quarters. Compared with: the same approach using the drivers chosen on the full
sample (look-ahead), and the AR(1) benchmark. Settings are in the `oos:` section of the config.
Results go to outputs/<run_name>/oos/. Each origin's selection is cached, so re-runs are fast.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.utils.env import ensure_packages  # noqa: E402

ensure_packages()

import warnings  # noqa: E402

import pandas as pd  # noqa: E402

from src.features.transforms import all_features, candidate_forms  # noqa: E402
from src.pipelines import bayes_first, en_first, focus_first, screen_first  # noqa: E402
from src.reporting import formula, report, story  # noqa: E402
from src.utils.cache import StepCache, hash_code, hash_frame, hash_obj  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import Context, log  # noqa: E402
from src.utils import docs, saved, trace  # noqa: E402
from src.utils.preflight import preflight  # noqa: E402
from src.models import horizon, zoo  # noqa: E402
from src.validation import oos  # noqa: E402

warnings.filterwarnings("ignore")
APPROACHES = {"screen_first": screen_first, "en_first": en_first, "bayes_first": bayes_first,
              "focus_first": focus_first}


def main():
    """Command line entry for the out-of-sample test."""
    ap = argparse.ArgumentParser(description="Out-of-sample test with re-selection at each origin")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--approach", choices=["both", "all", "screen_first", "en_first", "bayes_first", "focus_first"], default=None)
    ap.add_argument("--force", action="store_true", help="recompute every origin")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip-check", action="store_true", help="do not stop on errors found by the pre-flight check")
    ap.add_argument("--no-trace", action="store_true", help="do not write the detailed trace")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["trace"]["folder"] = "trace_oos"
    preflight(cfg, log, strict=not args.skip_check)         # names, files and dates checked in seconds
    tracer = trace.start(cfg, "run_oos.py", globals(), enabled=not args.no_trace)
    try:
        run(cfg, args)
    finally:
        trace.finish(tracer, {"config": os.path.relpath(cfg["config_file"], HERE), "run_name": cfg["run_name"]}, log)
        docs.refresh(cfg, log)                       # function reference, if the code changed


def run(cfg, args):
    """Out-of-sample test with re-selection at every origin, Diebold-Mariano tests, holdout and charts.

    Args:
        cfg: configuration. args: parsed command-line arguments.
    """
    names = list(cfg["approaches"]) if args.approach is None else \
        (["screen_first", "en_first"] if args.approach == "both" else list(APPROACHES) if args.approach == "all"
         else [args.approach])
    approaches = {n: APPROACHES[n] for n in names}
    out = os.path.join(cfg["out_dir"], "oos")
    os.makedirs(out, exist_ok=True)
    cache = StepCache(cfg["cache"]["dir"], enabled=cfg["cache"]["enabled"] and not args.no_cache, force=args.force)

    log("Loading data ...")
    ctx = Context(cfg)

    # full-sample drivers (the ones the normal pipeline reports), for the look-ahead comparison
    log("Full-sample selection (for comparison) ...")
    full = {}
    for n, mod in approaches.items():
        full[n] = oos.select_at(cfg, ctx, mod, ctx.end, cache, n)["features"]
        log(f"  {n}: {full[n]}")

    store = {} if (cfg.get("save") or {}).get("origin_models", True) else None
    E, picks, H = oos.run_oos(cfg, ctx, approaches, full, cache, store)
    if store:                                    # every model as estimated at every origin
        coefs = store.pop("__coefficients__", [])
        saved.save_origin_models(os.path.join(out, "fitted_models"), store, log)
        if coefs:
            pd.DataFrame(coefs).to_csv(os.path.join(out, "fitted_coefficients_by_origin.csv"), index=False)
    hc_rows, hc_choices = horizon.combine(E, cfg, H, lambda n: zoo.wanted(cfg, n, horizon.FAMILY))
    if hc_rows is not None:                       # horizon combinations, chosen from errors realised at each origin
        E = pd.concat([E, hc_rows.drop(columns=["chosen"]).assign(drivers=hc_rows["chosen"])], ignore_index=True)
    hz = sorted({h for h in cfg["sample"]["report_horizons"] if h <= H} or {H})
    by_h, summ, dm, stab = oos.summarise(E, picks, H, hz)
    log("Holdout ...")
    hold = oos.holdout(cfg, ctx, approaches, cache)

    tables = {"1_oos_summary": summ.round(4), "2_oos_by_horizon": by_h.round(4), "3_dm_tests_vs_benchmark": dm.round(4),
              "4_drivers_by_origin": picks, "5_driver_stability": stab.round(3), "6_all_forecast_errors": E}
    if hold is not None:
        tables["7_holdout"] = hold.round(4)
    if hc_choices is not None:
        tables["8_horizon_choices"] = hc_choices
    report.write_tables(out, tables)
    report.chart_oos(os.path.join(out, "oos_mape_by_horizon.png"), by_h, summ, H)
    report.chart_family_bars(os.path.join(out, "oos_model_comparison.png"),
                             summ[summ["family"] != "Look-ahead check"], hz)
    report.chart_driver_timeline(os.path.join(out, "drivers_by_origin.png"),
                                 picks[picks["model"].isin([oos.PRETTY[a] for a in approaches])])
    if hold is not None:
        report.chart_holdout(os.path.join(out, "holdout.png"), hold, ctx.level, report.best_per_family(summ, max(hz)))

    trace.add_story(lambda: story.build_oos(cfg, picks, summ, dm, hold, H), log)
    log(f"\nOut-of-sample results (re-selected at every origin; dummy quarters excluded; "
        f"{len(picks['origin'].unique()) if len(picks) else 0} origins; average error over 1–h quarters):")
    log(f"  {'model':<46s} {'family':<20s} " + " ".join(f"{'1-' + str(h):>7s}" for h in hz) + "  Theil U")
    for _, r in summ.sort_values(f"avg_1_{max(hz)}_pct").iterrows():
        log(f"  {r['model']:<46s} {r['family']:<20s} " + " ".join(f"{r[f'avg_1_{h}_pct']:6.2f}%" for h in hz)
            + f"  {r[f'theil_U_1_{H}']:.2f}")
    if hold is not None:
        last = hold.groupby("model", sort=False).tail(1).sort_values("error_pct", key=abs)
        log(f"\nHoldout (selected and estimated to {hold['train_end'].iloc[0]}, error after {cfg['oos']['holdout_quarters']} quarters):")
        for _, r in last.iterrows():
            log(f"  {r['model']:<46s} {r['error_pct']:+.2f}%")
    fitted = {k: m for k, m in saved.load_models(cfg["out_dir"]).items() if formula.has_formula(m)}
    if fitted:
        try:
            formula.write(cfg, ctx, fitted, summ, "out of sample", log)
        except Exception as ex:
            log(f"  formula: not written ({type(ex).__name__}: {ex})")
    else:
        log("  formula: run run_pipeline.py first (it saves the fitted models the formula is taken from)")
    saved.snapshot(cfg, "oos", log)
    log(f"\nDone. Outputs in {out}")

if __name__ == "__main__":
    main()
