"""
The process story of a run, for an audience: what happened at every step, with the run's own numbers and charts.

    python tools/process_story.py --config config/example_config.yaml
    python tools/process_story.py --config config/my_portfolio.yaml --format html      # or pptx, or both (default)
    python tools/process_story.py --config config/my.yaml --run-dir outputs/my_run/saved_runs/2026-11-01_1200

Run run_pipeline.py first (and run_oos.py for the out-of-sample sections; without it the story says so and
ranks on the backtest). Writes, in outputs/<run_name>/process_story/:

    process_story.html   one long page: every step with what was done, why, what happened, the chart and the table
    process_story.pptx   one slide per step (about 50 slides); the full text of each step is in the speaker notes
    charts/              the charts as PNG files

Needs python-pptx and matplotlib (pip install python-pptx matplotlib). No Quarto.
"""
import argparse
import os
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from src.reporting import process_story as PS, story_html, story_pptx  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def make(config, run_dir=None, fmt="both", out_dir=None, log=print):
    """Build the story of a finished run.

    Args:
        config: config file. run_dir: the run's output folder (default outputs/<run_name>).
        fmt: html, pptx or both. out_dir: where to write (default <run_dir>/process_story).

    Returns:
        dict with the paths written and any section that could not be built.
    """
    cfg = load_config(config)
    run_dir = run_dir or cfg["out_dir"]
    out = out_dir or os.path.join(run_dir, "process_story")
    os.makedirs(out, exist_ok=True)
    st = PS.Story(cfg, run_dir, os.path.join(out, "charts"))
    secs = st.build()
    title = f"How the {st.TARGET} forecast was built"
    res = {"sections": len(secs), "warnings": st.warnings}
    if fmt in ("html", "both"):
        res["html"] = story_html.build(secs, os.path.join(out, "process_story.html"), title)
    if fmt in ("pptx", "both"):
        res["pptx"] = story_pptx.build(secs, os.path.join(out, "process_story.pptx"),
                                       f"{st.TARGET} forecast · run {cfg['run_name']}")
    for w in st.warnings:
        log(f"  section skipped: {w}")
    log(f"Process story: {len(secs)} sections -> " + ", ".join(res[k] for k in ("html", "pptx") if k in res))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--run-dir", default=None, help="the run's output folder (default outputs/<run_name>)")
    ap.add_argument("--format", default="both", choices=["html", "pptx", "both"])
    ap.add_argument("--out", default=None, help="output folder (default <run-dir>/process_story)")
    a = ap.parse_args()
    make(a.config, a.run_dir, a.format, a.out)


if __name__ == "__main__":
    main()
