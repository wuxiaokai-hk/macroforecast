"""Compare the panel models on any metric, for all banks, some banks, or one bank, from a finished panel run.

No re-estimation: it reads the saved forecast errors (outputs/<run>/panel/4_backtest_forecasts.csv and
oos/4_forecasts.csv).

    python tools/panel_compare.py --run outputs/panel_example                       # MAPE, every bank
    python tools/panel_compare.py --run outputs/panel_example --metric rmse --oos   # out-of-sample test
    python tools/panel_compare.py --run outputs/panel_example --banks "Bank A" "Bank C"
    python tools/panel_compare.py --run outputs/panel_example --bank "Bank B"       # one bank's comparison
    python tools/panel_compare.py --run outputs/panel_example --metric combined --penalty 1 --horizon 8 --save

Metrics: mape, rmse, p90, cvar (average of the worst tail_share of errors), worst, combined
(mape + penalty x (cvar - mape)). --save writes the tables next to the run's leaderboards (10_..12_ files).
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.panel import compare  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="outputs/<run> (or its panel/ folder)")
    ap.add_argument("--metric", default="mape", choices=list(compare.METRICS))
    ap.add_argument("--banks", nargs="*", help="banks to include (default: the active ones, still reporting)")
    ap.add_argument("--all-banks", action="store_true", help="include the banks that stopped reporting too")
    ap.add_argument("--bank", help="also show this bank's comparison")
    ap.add_argument("--oos", action="store_true", help="the out-of-sample test instead of the backtest")
    ap.add_argument("--horizon", type=int, help="quarters 1..h (default: all)")
    ap.add_argument("--tail-share", type=float, default=0.10)
    ap.add_argument("--penalty", type=float, default=0.5)
    ap.add_argument("--weighting", choices=["equal", "size"], default="equal")
    ap.add_argument("--benchmark", help="model for the 'vs' columns (default: the AR(1) benchmark)")
    ap.add_argument("--include-covid", action="store_true", help="keep the dummy (COVID) windows")
    ap.add_argument("--save", action="store_true", help="write the tables into the run folder")
    a = ap.parse_args(argv)
    pdir = a.run if os.path.basename(os.path.normpath(a.run)) == "panel" else os.path.join(a.run, "panel")
    E = compare.load_errors(pdir, a.oos)
    import json
    try:
        act = json.load(open(os.path.join(pdir, "fitted", "manifest.json"), encoding="utf-8")).get("active_banks")
    except Exception:
        act = None
    banks = a.banks if a.banks else ("all" if a.all_banks else "active")
    res = compare.evaluate(E, a.metric, banks, a.horizon, a.tail_share, a.penalty, a.weighting, a.benchmark,
                           exclude_covid=not a.include_covid, active=act)
    st = res["settings"]
    pd.set_option("display.width", 200)
    print(f"{'Out-of-sample test' if a.oos else 'Backtest'}: {len(st['banks'])} bank(s), quarters 1-{st['horizon']}, "
          f"ranked on the {st['metric_long']}; benchmark: {st['benchmark']}")
    cols = ["rank", "model", "score", "mape", "rmse", "p90", "cvar", "worst", "bias", "vs_benchmark_pct", "banks_won",
            "banks_better_than_benchmark", "banks"]
    print(res["overall"][[c for c in cols if c in res["overall"]]].round(3).to_string(index=False))
    if a.bank:
        b = compare.one_bank(res, a.bank)
        if b.empty:
            raise SystemExit(f"no bank '{a.bank}'; banks are {st['banks']}")
        print(f"\n{a.bank}:")
        print(b[[c for c in ["rank_in_bank", "model", "score", "mape", "rmse", "p90", "cvar", "worst", "bias",
                             "vs_benchmark_pct", "forecasts"] if c in b]].round(3).to_string(index=False))
    if a.save:
        compare.write(res, os.path.join(pdir, "oos") if a.oos else pdir)
        print(f"\nSaved to {os.path.join(pdir, 'oos') if a.oos else pdir} (10_..12_ files)")


if __name__ == "__main__":
    main()
