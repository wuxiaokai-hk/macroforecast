"""Write the literature workbook template (sheets models, concepts, settings, references, how to fill) from the current config.

    python tools/literature_template.py                       -> config/literature_workbook.xlsx
    python tools/literature_template.py --out my_lit.xlsx --config config/my_portfolio.yaml

Fill sheet 'models' with the published models you want to compare (one row per driver) and sheet 'concepts'
with your series names, then set `literature.workbook` in the config (or upload it in the app).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.models.literature import load, read_workbook, write_workbook  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    """Command line entry."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "config", "literature_workbook.xlsx"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    wb = cfg["literature"].get("workbook")
    full = read_workbook(wb, full=True) if wb and os.path.exists(wb) else {}
    write_workbook(a.out, cfg["literature"]["models"], load(cfg), full.get("settings"), full.get("references"))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
