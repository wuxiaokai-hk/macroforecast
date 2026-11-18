"""Check a config and its data files before running (a few seconds).

    python tools/check_config.py --config config/my_data.yaml          problems only
    python tools/check_config.py --config config/my_data.yaml --all    every check, passed ones too

Exit code 0 when there are no errors, 2 otherwise.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.utils.config import load_config  # noqa: E402
from src.utils.preflight import check  # noqa: E402


def main():
    """Command line entry."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--all", action="store_true", help="also list the checks that passed")
    a = ap.parse_args()
    try:
        cfg = load_config(a.config)
    except Exception as e:
        print(f"The config cannot be loaded: {type(e).__name__}: {e}")
        sys.exit(2)
    rep = check(cfg)
    print(f"Checking {a.config}" + rep.text(show_ok=a.all or not (rep.errors or rep.warnings)))
    sys.exit(2 if rep.errors else 0)


if __name__ == "__main__":
    main()
