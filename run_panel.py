"""
Panel run - several banks (or regions, or portfolios) with the same target and common macro drivers.

    python run_panel.py --config config/panel_example.yaml
    python run_panel.py --config config/panel_example.yaml --no-oos

The target file has one row per bank and quarter: date, bank name, balance (data.bank.panel_column names the bank
column). Drivers are chosen on all banks together, then each bank is forecast by the panel models
(separate, pooled fixed effects, partial pooling, top-down share) and an AR(1) benchmark, with the same backtest,
out-of-sample test and scenarios as run_pipeline.py. Results: outputs/<run_name>/panel/.
"""
import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.utils.env import ensure_packages  # noqa: E402

ensure_packages()

import warnings  # noqa: E402

import pandas as pd  # noqa: E402

from src.panel import run as panel_run  # noqa: E402
from src.panel.data import load_long  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import log  # noqa: E402
from src.utils.preflight import preflight  # noqa: E402

warnings.filterwarnings("ignore")


def total_cfg(cfg):
    """A copy of a panel config that points at the total of the banks (for the pre-flight check, which checks
    one series; the panel file itself is checked by load_long)."""
    levels = load_long(cfg)
    d = os.path.join(cfg["out_dir"], "panel", "inputs")
    os.makedirs(d, exist_ok=True)
    tot = pd.concat(levels.values(), axis=1).sum(axis=1, min_count=1)
    f = os.path.join(d, "panel_total_check.csv")
    pd.DataFrame({"date": [str(q) for q in tot.index], "total": tot.values}).to_csv(f, index=False)
    c = copy.deepcopy(cfg)
    c["data"]["bank"].update(file=f, target_column="total", date_column="date", sheet=None, frequency="Q")
    c["data"]["events"]["file"] = None
    return c, levels


def check(cfg, strict=True):
    """Pre-flight check on the total of the banks."""
    c, levels = total_cfg(cfg)
    preflight(c, log, strict=strict)
    log(f"Panel file: {len(levels)} banks ({', '.join(list(levels)[:8])}{', ...' if len(levels) > 8 else ''})")


def main():
    ap = argparse.ArgumentParser(description="Panel run: several banks, common macro drivers")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "panel_example.yaml"))
    ap.add_argument("--no-oos", action="store_true", help="skip the out-of-sample test")
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not cfg["data"]["bank"].get("panel_column"):
        raise SystemExit("data.bank.panel_column is not set: name the column that holds the bank")
    check(cfg, strict=not a.skip_check)
    panel_run.run(cfg, cfg["out_dir"], oos=not a.no_oos, log=log)


if __name__ == "__main__":
    main()
