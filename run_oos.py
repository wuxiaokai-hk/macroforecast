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
from src.pipelines import bayes_first, en_first, factors_first, focus_first, screen_first, stable_first  # noqa: E402
from src.reporting import formula, report, story  # noqa: E402
from src.utils.cache import StepCache, hash_code, hash_frame, hash_obj  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import Context, log  # noqa: E402
from src.utils import docs, saved, trace  # noqa: E402
from src.utils.preflight import preflight  # noqa: E402
from src.models import horizon, zoo  # noqa: E402
from src.validation import oos, vintages  # noqa: E402

warnings.filterwarnings("ignore")
APPROACHES = {"screen_first": screen_first, "en_first": en_first, "bayes_first": bayes_first,
              "focus_first": focus_first, "factors_first": factors_first, "stable_first": stable_first}


def main():
    """Command line entry for the out-of-sample test."""
    ap = argparse.ArgumentParser(description="Out-of-sample test with re-selection at each origin")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--approach", choices=["both", "all", "screen_first", "en_first", "bayes_first", "focus_first", "factors_first", "stable_first"], default=None)
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
    vout, wout = {}, {}
    E, picks, H = oos.run_oos(cfg, ctx, approaches, full, cache, store, vintage_out=vout, window_out=wout)
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
    summ_fc = None
    EV = vout.get("errors")
    if EV is not None and len(EV):                # the same test with the macro forecast of each origin's time
        hv, _ = horizon.combine(EV, cfg, H,
                                lambda n: zoo.wanted(cfg, n, horizon.FAMILY))
        if hv is not None:
            EV = pd.concat([EV, hv.drop(columns=["chosen"]).assign(drivers=hv["chosen"])], ignore_index=True)
        by_h_fc, summ_fc, dm_fc, _ = oos.summarise(EV, picks, H, hz)
        col = f"avg_1_{max(hz)}_pct"
        cmp_ = summ[["model", "family", col]].merge(summ_fc[["model", col]], on="model",
                                                     suffixes=("_realised_macro", "_forecast_macro"))
        cmp_["macro_forecast_cost_pct"] = cmp_[f"{col}_forecast_macro"] - cmp_[f"{col}_realised_macro"]
        cmp_ = cmp_.sort_values(f"{col}_forecast_macro")
        cov = vout.get("coverage")
        tables.update({"9_oos_summary_forecast_macro": summ_fc.round(4),
                       "10_oos_by_horizon_forecast_macro": by_h_fc.round(4),
                       "11_dm_tests_forecast_macro": dm_fc.round(4),
                       "12_realised_vs_forecast_macro": cmp_.round(4),
                       "13_vintage_coverage": cov, "14_all_forecast_errors_forecast_macro": EV})
        log(f"\nWith the macro forecast available at each origin (vintages {', '.join(sorted(EV['vintage'].dropna().unique())[:3])}"
            f"...): average error over 1-{max(hz)} quarters, realised macro -> forecast macro")
        for _, r in cmp_.head(12).iterrows():
            log(f"  {r['model']:<46s} {r[f'{col}_realised_macro']:6.2f}% -> {r[f'{col}_forecast_macro']:6.2f}%")
        if cov is not None and len(cov) and (cov["drivers_from_vintage"] < 1).any():
            low = cov[cov["drivers_from_vintage"] < 1]
            log(f"  note: {low['model'].nunique()} model(s) use series missing from the vintages "
                f"({', '.join(sorted(set(', '.join(low['drivers_not_in_vintage']).split(', ')) - {''})[:6])}); "
                f"those take their {vintages.settings(cfg)['missing']} path (see 13_vintage_coverage)")
    WD, WS = oos.window_check(E, wout.get("errors"), cfg, hz)
    if WD is not None:                            # does the older history help or hurt?
        tables.update({"15_window_check": WD.round(4), "16_window_check_summary": WS.round(4),
                       "17_all_forecast_errors_other_windows": wout["errors"]})
        log(f"\nWindow check: {oos.base_window_label(cfg)} vs a shorter, more recent window "
            f"(same drivers, same origins; average error over 1-{max(hz)} quarters):")
        for _, r in WS.iterrows():
            log(f"  {r['other_window']} ({r['origins_compared']} origins with enough history): better for "
                f"{r[f'other_window_better_1_{max(hz)}']} of {r['models']} models, "
                f"median change {r['median_change_pct']:+.1f}%, significant (10%) for {r['significant_either_way']}; "
                f"best model {r['best_error_main_pct']:.2f}% ({r['main_window']}) vs {r['best_error_other_pct']:.2f}%")
        top = WD.loc[WD["model"].isin(summ.nsmallest(8, f"avg_1_{max(hz)}_pct")["model"])]
        for _, r in top.iterrows():
            log(f"    {r['model']:<44s} {r[f'main_avg_1_{max(hz)}_pct']:6.2f}% -> {r[f'other_avg_1_{max(hz)}_pct']:6.2f}%"
                f"  [{r['other_window']}] {r['verdict']}")
        report.chart_window_check(os.path.join(out, "window_check.png"), WD, max(hz))
    report.write_tables(out, tables)
    if summ_fc is not None:
        report.chart_family_bars(os.path.join(out, "oos_model_comparison_forecast_macro.png"),
                                 summ_fc[summ_fc["family"] != "Look-ahead check"], hz)
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
    all_fitted = saved.load_models(cfg["out_dir"])
    order, why_rec = None, None
    use_fc = summ_fc is not None and vintages.settings(cfg)["rank_on"] == "forecast"
    try:
        from src.selection import scorecard
        log("\nThe recommended model, scored out of sample (forecasting, interpretation, sensitivity):")
        S = scorecard.score(cfg, all_fitted, summ_fc if use_fc else summ, cfg["out_dir"],
                            "out of sample, with the macro forecasts of the time" if use_fc else "out of sample", log)
        if S is not None:
            order, why_rec = list(S["model"]), scorecard.why_text(S.iloc[0], S.attrs["weights"], S.attrs["col"])
    except Exception as ex:
        log(f"  scorecard not written ({type(ex).__name__}: {ex})")
    fitted = {k: m for k, m in all_fitted.items() if formula.has_formula(m)}
    if fitted:
        try:
            formula.write(cfg, ctx, fitted, summ_fc if use_fc else summ,
                          "out of sample, macro forecasts of the time" if use_fc else "out of sample", log,
                          order=order, why_rec=why_rec)
        except Exception as ex:
            log(f"  formula: not written ({type(ex).__name__}: {ex})")
    else:
        log("  formula: run run_pipeline.py first (it saves the fitted models the formula is taken from)")
    saved.snapshot(cfg, "oos", log)
    log(f"\nDone. Outputs in {out}")

if __name__ == "__main__":
    main()
