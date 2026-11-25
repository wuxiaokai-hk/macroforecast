"""
The process story of a run, for an audience: what happened at every step, with the run's own numbers and charts.

    python tools/process_story.py --config config/example_config.yaml
    python tools/process_story.py --config config/my_portfolio.yaml --format html      # or pptx, or both (default)
    python tools/process_story.py --config config/my.yaml --run-dir outputs/my_run/saved_runs/2026-11-01_1200
    python tools/process_story.py --config config/my.yaml --theme midnight     # another look (see --list-themes)
    python tools/process_story.py --config config/my.yaml --theme all          # one copy per theme, to compare

Themes: classic (default), corporate, midnight, minimal, forest, boardroom, ocean. Set a default for a project
with `process_story: {theme: corporate}` in the config; --theme overrides it.

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

from src.reporting import process_story as PS, story_html, story_pptx, themes as TH  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def make(config, run_dir=None, fmt="both", out_dir=None, log=print, theme=None):
    """Build the story of a finished run.

    Args:
        config: config file. run_dir: the run's output folder (default outputs/<run_name>).
        fmt: html, pptx or both. out_dir: where to write (default <run_dir>/process_story).
        theme: a theme name (src/reporting/themes.py), "all" for one copy per theme in <out>/themes/<name>/,
            or None for the config's process_story.theme (default classic).

    Returns:
        dict with the paths written and any section that could not be built ("all": a dict per theme).
    """
    cfg = load_config(config)
    run_dir = run_dir or cfg["out_dir"]
    out = out_dir or os.path.join(run_dir, "process_story")
    theme = theme or ((cfg.get("process_story") or {}).get("theme")) or TH.DEFAULT
    if str(theme).lower() == "all":
        res = {}
        for name in TH.THEMES:
            res[name] = make(config, run_dir, fmt, os.path.join(out, "themes", name), log, name)
        _gallery(res, os.path.join(out, "themes", "index.html"))
        log(f"Theme gallery: {os.path.join(out, 'themes', 'index.html')}")
        return res
    TH.get(theme)                                             # fail early on an unknown name
    os.makedirs(out, exist_ok=True)
    PS.apply_theme(theme)
    st = PS.Story(cfg, run_dir, os.path.join(out, "charts"))
    secs = st.build()
    title = f"How the {st.TARGET} forecast was built"
    res = {"sections": len(secs), "warnings": st.warnings}
    if fmt in ("html", "both"):
        res["html"] = story_html.build(secs, os.path.join(out, "process_story.html"), title, theme=theme)
    if fmt in ("pptx", "both"):
        res["pptx"] = story_pptx.build(secs, os.path.join(out, "process_story.pptx"),
                                       f"{st.TARGET} forecast · run {cfg['run_name']}", theme=theme)
    for w in st.warnings:
        log(f"  section skipped: {w}")
    PS.apply_theme(TH.DEFAULT)
    log(f"Process story ({theme}): {len(secs)} sections -> " + ", ".join(res[k] for k in ("html", "pptx") if k in res))
    return res


def _gallery(res, path):
    """A small page linking every theme's copy."""
    rows = "".join(
        f'<li><b>{TH.THEMES[n]["label"]}</b> <code>--theme {n}</code><br><span>{TH.THEMES[n]["note"]}</span><br>'
        + " · ".join(f'<a href="{n}/{os.path.basename(r[k])}">{k.upper()}</a>' for k in ("html", "pptx") if k in r)
        + "</li>" for n, r in res.items())
    with open(path, "w", encoding="utf-8") as f:
        f.write('<!DOCTYPE html><meta charset="utf-8"><title>Process story themes</title><style>body{font:16px/1.6 '
                'sans-serif;max-width:760px;margin:40px auto;padding:0 16px}li{margin:14px 0}span{color:#666}</style>'
                f'<h1>Process story themes</h1><ul>{rows}</ul>')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--run-dir", default=None, help="the run's output folder (default outputs/<run_name>)")
    ap.add_argument("--format", default="both", choices=["html", "pptx", "both"])
    ap.add_argument("--out", default=None, help="output folder (default <run-dir>/process_story)")
    ap.add_argument("--theme", default=None, help="look of the deck and page: " + ", ".join(TH.THEMES)
                    + ", or all (default: the config's process_story.theme, else classic)")
    ap.add_argument("--list-themes", action="store_true", help="list the themes and stop")
    a = ap.parse_args()
    if a.list_themes:
        print(TH.describe())
        return
    make(a.config, a.run_dir, a.format, a.out, theme=a.theme)


if __name__ == "__main__":
    main()
