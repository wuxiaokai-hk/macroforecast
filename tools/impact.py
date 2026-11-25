"""
How the economy affects the target, from a finished run: sensitivities, what-if shocks, historical episodes,
scenario attribution and how far each effect can be trusted. run_pipeline.py runs it at the end of every run;
use this to try other shocks without re-running anything.

    python tools/impact.py --config config/my.yaml
    python tools/impact.py --config config/my.yaml --shock unemployment:+2:4 --shock FRBOC.IC:+1:2
    python tools/impact.py --config config/my.yaml --shock "Recession=unemployment:+2:4,FRMORT5Y.IC:-1:2"
    python tools/impact.py --config config/my.yaml --episode 2008Q3:2009Q4 --episode 2022Q1:2023Q2

--shock VARIABLE:SIZE[:QUARTERS[:START]]   VARIABLE = literature concept, series name or feature; SIZE in points
    for rates (+1 = one percentage point) and in percent of the level for prices, income and other levels;
    QUARTERS = build-up (default 1); START = first forecast quarter hit (default 1). The shift is permanent.
    NAME=SHOCK,SHOCK,... combines several shocks into one what-if.
--episode START:END   replay the changes of a historical period on top of the base scenario.

Writes outputs/<run>/impact/ (see src/impact/analysis.py).
"""
import argparse
import os
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from src.impact import analysis  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import Context  # noqa: E402
from src.utils.saved import load_models  # noqa: E402


def parse_shock(text):
    """'unemployment:+2:4' -> {'variable': 'unemployment', 'size': 2.0, 'quarters': 4}."""
    p = [x.strip() for x in text.split(":")]
    if len(p) < 2:
        raise ValueError(f"shock '{text}': write VARIABLE:SIZE[:QUARTERS[:START]]")
    out = {"variable": p[0], "size": float(p[1])}
    if len(p) > 2 and p[2]:
        out["quarters"] = int(p[2])
    if len(p) > 3 and p[3]:
        out["start"] = int(p[3])
    return out


def parse_what_if(texts):
    """--shock arguments -> impact.what_if entries."""
    out = []
    for t in texts or []:
        name, _, body = t.partition("=") if "=" in t else ("", "", t)
        shocks = [parse_shock(x) for x in body.split(",") if x.strip()]
        out.append({"name": name or ", ".join(f"{s['variable']} {s['size']:+g}" for s in shocks), "shocks": shocks})
    return out


def make(config, run_dir=None, what_if=None, episodes=None, log=print):
    """Run the impact analysis on a finished run.

    Args:
        config: config file. run_dir: the run folder (default outputs/<run_name>).
        what_if: list of {name, shocks}; episodes: list of {name, start, end}; None = the config's.

    Returns:
        dict of tables.
    """
    cfg = load_config(config)
    run_dir = run_dir or cfg["out_dir"]
    models = load_models(run_dir)
    if not models:
        raise SystemExit(f"No fitted models in {run_dir}: run run_pipeline.py first (with save.fitted_models: true).")
    ctx = Context(cfg)
    return analysis.run(cfg, ctx, models, run_dir, log, what_if=what_if, episodes=episodes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--shock", action="append", default=None, help="VARIABLE:SIZE[:QUARTERS[:START]] (repeatable)")
    ap.add_argument("--episode", action="append", default=None, help="START:END, e.g. 2008Q3:2009Q4 (repeatable)")
    a = ap.parse_args()
    wi = parse_what_if(a.shock) if a.shock else None
    eps = [{"name": e, "start": e.split(":")[0], "end": e.split(":")[1]} for e in a.episode] if a.episode else None
    make(a.config, a.run_dir, wi, eps)


if __name__ == "__main__":
    main()
