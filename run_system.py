"""
System run - several related targets (loans, deposits, the bank's own rates) analysed together.

    python run_system.py --config config/system_example.yaml

Every target runs through the complete single-target project, in an order where each target can use the
earlier ones as drivers (rates -> loans -> deposits). Then the whole chain is backtested jointly (earlier
targets' FORECASTS feed the later ones), tested out of sample with re-selection, challenged by a BVARX,
reconciled to identities (total loans = mortgage + commercial) and checked against ratios (loan-to-deposit).
Results: outputs/<run_name>/system/ and outputs/<run_name>/targets/<target>/.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.utils.env import ensure_packages  # noqa: E402

ensure_packages()

import warnings  # noqa: E402

from src.system import run as system  # noqa: E402
from src.utils.io import log  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="System run: several related targets")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "system_example.yaml"))
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    system.run(a.config, a.output_dir, log)


if __name__ == "__main__":
    main()
