"""
Build the styled PowerPoint deck for a run (the same design as report/forecast_deck.pptx).

    python tools/make_deck.py --config config/my_portfolio.yaml            -> outputs/<run>/<run>_deck.pptx
    python tools/make_deck.py --config config/my.yaml --title "Forecasting the Quebec Loan Balance" --simulated

Needs Quarto (https://quarto.org/docs/get-started/) and, in the Python that runs this tool: pip install python-pptx jupyter
Run the pipeline (run_pipeline.py, and run_oos.py for the out-of-sample slides) first: the deck reads its outputs.
Steps: report/make_reference.py builds the template if it is missing; quarto renders report/forecast_deck.qmd
with the run's config; report/style_deck.py (post-render) applies the design; the file is copied to the run folder.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(HERE, "report")
sys.path.insert(0, HERE)


def find_quarto():
    """Path of the quarto executable, or None (QUARTO environment variable, PATH, usual install folders)."""
    cands = [os.environ.get("QUARTO"), shutil.which("quarto"), "/Applications/quarto/bin/quarto",
             os.path.expanduser("~/Applications/quarto/bin/quarto"), r"C:\Program Files\Quarto\bin\quarto.exe",
             os.path.expanduser(r"~\AppData\Local\Programs\Quarto\bin\quarto.exe")]
    cands += sorted(glob.glob("/tmp/claude-0/quarto/*/bin/quarto"))
    return next((c for c in cands if c and os.path.exists(c)), None)


NEED = {"pptx": "python-pptx", "jupyter_core": "jupyter", "nbformat": "jupyter", "nbclient": "jupyter",
        "ipykernel": "jupyter", "yaml": "pyyaml", "openpyxl": "openpyxl", "tabulate": "tabulate"}


def missing_packages(py=None):
    """pip names of the packages the deck needs that are missing in a Python.

    Args:
        py: path of the Python executable (None = the one running this code).

    Returns:
        sorted list of pip package names (empty = ready).
    """
    py = py or sys.executable
    chk = subprocess.run([py, "-c", "import importlib.util as u; print(' '.join(m for m in %r if u.find_spec(m) is None))"
                          % list(NEED)], capture_output=True, text=True)
    return sorted({NEED[m] for m in chk.stdout.split()}) if chk.returncode == 0 else []


def build_deck(config, title=None, simulated=False, region="", out=None, log=print, run_dir=None):
    """Render the deck for a config and return the path of the .pptx.

    Args:
        config: config file. title: deck title (None = 'Forecasting the <region> Loan Balance').
        simulated: add the 'simulated data' notes. region: name used in the text ('' = from the config).
        out: target .pptx (None = outputs/<run>/<run>_deck.pptx). log: function for progress lines.
        run_dir: results folder to read (None = outputs/<run>; or a saved run, outputs/<run>/saved_runs/<...>).

    Returns:
        path of the deck.

    Raises:
        RuntimeError: quarto is missing or the render failed (the message has the log tail).
    """
    from src.utils.config import load_config
    cfg = load_config(config)
    q = find_quarto()
    if not q:
        raise RuntimeError("Quarto is not installed. Install it from https://quarto.org/docs/get-started/ "
                           "(or set the QUARTO environment variable to its path), then try again.")
    reg = region or (cfg.get("screen_first") or {}).get("primary_geography") or ""
    tt_ = (cfg.get("target") or {}).get("label") or str(cfg["data"]["bank"]["target_column"]).replace("_", " ").title()
    title = title or (f"Forecasting the {reg} {tt_}" if reg else f"Forecasting the {tt_}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")   # UTF-8 files and output on Windows too
    env.setdefault("QUARTO_PYTHON", sys.executable)          # Quarto uses this Python, the one running this tool
    py = env["QUARTO_PYTHON"]
    missing = missing_packages(py)
    if missing:
        raise RuntimeError("The deck needs these Python packages, which are not installed in " + py + ":\n    "
                           + ", ".join(missing) + "\nInstall them with:\n    \"" + py + "\" -m pip install "
                           + " ".join(missing) + "\nthen build the presentation again.")
    if not os.path.exists(os.path.join(REPORT, "reference.pptx")):
        log("building the template (report/make_reference.py)")
        subprocess.run([env.get("QUARTO_PYTHON", sys.executable), os.path.join(REPORT, "make_reference.py")],
                       cwd=REPORT, env=env, check=True)
    tmp = f"_deck_{cfg['run_name']}.pptx"
    env.update(DECK_CONFIG=os.path.abspath(config), DECK_OUTPUTS=os.path.dirname(cfg["out_dir"]),
               DECK_SIMULATED="True" if simulated else "False", DECK_REGION=reg,
               DECK_RUN_DIR=os.path.abspath(run_dir) if run_dir else "")
    cmd = [q, "render", "forecast_deck.qmd", "--to", "pptx", "--output", tmp, "-M", f"title:{title}"]
    log("rendering: " + " ".join(cmd[1:4]) + " …")
    p = subprocess.Popen(cmd, cwd=REPORT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    lines = []
    for line in p.stdout:
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)          # drop terminal colour codes
        lines.append(line.rstrip())
        if line.strip():
            log(line.rstrip())
    if p.wait() != 0 or not os.path.exists(os.path.join(REPORT, tmp)):
        raise RuntimeError("The deck could not be rendered:\n" + "\n".join(lines[-25:]))
    out = out or os.path.join(cfg["out_dir"], f"{cfg['run_name']}_deck.pptx")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    shutil.move(os.path.join(REPORT, tmp), out)
    log(f"deck: {out}")
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    a.add_argument("--title", default=None)
    a.add_argument("--region", default="")
    a.add_argument("--simulated", action="store_true", help="add the 'simulated data' notes")
    a.add_argument("--out", default=None)
    a.add_argument("--run-dir", default=None, help="a saved run folder (outputs/<run>/saved_runs/<...>) instead of the latest")
    a = a.parse_args()
    try:
        build_deck(a.config, a.title, a.simulated, a.region, a.out, run_dir=a.run_dir)
    except RuntimeError as ex:
        sys.exit(str(ex))
