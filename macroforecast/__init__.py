"""macroforecast: explainable forecasting of a quarterly series from macro drivers.

Balances, deposits, investment, prices (log growth) or interest rates and spreads (quarterly change): the same
selection, model zoo, backtest, out-of-sample test, scenario forecasts and saved fitted models.

    from macroforecast import Forecaster
    fc = Forecaster(target="loan_balance").fit(data="data/bank_data.csv", metadata="data/macro_metadata.csv",
                                               literature="config/literature_workbook.xlsx",
                                               hyperparameters={"sample.horizon": 8})
    fc.leaderboard(); fc.predict(); fc.coefficients(); fc.fitted_values()

    import macroforecast as mf
    res = mf.run(...)                    # the same as one function call
    mf.build_config(...)                 # the config a run would use, without running
    mf.Forecaster.load("outputs/<run>")  # results of an earlier run
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:                     # works without installing: the project folder holds src/
    _sys.path.insert(0, _ROOT)

from src.utils.env import ensure_packages as _ensure  # noqa: E402

_ensure()

from .api import DEFAULT_BASE, PROJECT_ROOT, Results, build_config, infer_dates, load, parse_options, run  # noqa: E402,F401
from .forecaster import PRESETS, Forecaster  # noqa: E402,F401
from .system import System  # noqa: E402,F401

__version__ = "2.0.0"
__all__ = ["Forecaster", "System", "PRESETS", "run", "load", "build_config", "infer_dates", "parse_options", "Results",
           "DEFAULT_BASE", "PROJECT_ROOT"]
