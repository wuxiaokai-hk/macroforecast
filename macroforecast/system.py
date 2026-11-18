"""Several related targets as one system (loans, deposits, own rates).

    from macroforecast import System
    sy = System("config/system_example.yaml").fit()      # or System({...the same as a dict...})
    sy.chain()                  # order, form, what each target uses, best model, drivers
    sy.leaderboard()            # per target: chain vs chain with actual inputs vs BVARX vs reconciled
    sy.links()                  # how each target responds to the earlier ones (its chosen model)
    sy.scenarios("deposits")    # joint scenario paths
    sy.ratios()                 # e.g. loan-to-deposit under every scenario vs its historical range
    sy.sur_coefficients(); sy.sur_correlation(); sy.fan("deposits", "S4 Severe")
    sy.vecm_coefficients(); sy.cointegration()
    sy.target("mortgage")       # the target's complete single-target results (a Forecaster)
    sy = System.load("outputs/system_example")
"""
import os

import pandas as pd
import yaml

from .api import PROJECT_ROOT, _abs
from .forecaster import Forecaster


class System:
    def __init__(self, config=None, path=None):
        self.config = config
        self.path = _abs(path) if path else None

    def fit(self, output_dir=None):
        """Run the whole system (see src/system/run.py). Returns self."""
        from src.system import run as srun
        from src.utils.io import log
        cfg = self.config
        if isinstance(cfg, dict):                      # a dict: written next to the outputs for the record
            root = _abs(output_dir) if output_dir else os.path.join(PROJECT_ROOT, "outputs")
            d = os.path.join(root, cfg.get("run_name", "system"))
            os.makedirs(d, exist_ok=True)
            f = os.path.join(d, "system_config.yaml")
            yaml.safe_dump(cfg, open(f, "w", encoding="utf-8"), sort_keys=False)
            cfg = f
        cfg = str(cfg)
        if not os.path.exists(cfg) and os.path.exists(os.path.join(PROJECT_ROOT, cfg)):
            cfg = os.path.join(PROJECT_ROOT, cfg)            # a path relative to the project folder
        res = srun.run(_abs(cfg), output_dir, log)
        self.path = os.path.dirname(res["folder"])
        return self

    @classmethod
    def load(cls, path):
        return cls(path=path)

    @property
    def is_panel(self):
        """True for a panel system (several banks, system.panel_column)."""
        return bool(self.path) and os.path.isdir(os.path.join(self.path, "system_panel"))

    def _t(self, name):
        if not self.path:
            raise RuntimeError("call fit() first, or System.load(path)")
        return pd.read_csv(os.path.join(self.path, "system_panel" if self.is_panel else "system", f"{name}.csv"))

    def by_bank(self, target=None, bank=None):
        """Panel system: accuracy of every target for every bank (chain and chain with actual inputs)."""
        T = self._t("2_by_bank")
        for k, v in (("target", target), ("bank", bank)):
            if v is not None:
                T = T[T[k] == v]
        return T.reset_index(drop=True)

    def target_leaderboard(self, name, bank=None):
        """Panel system: one target's panel models ranked across banks (or for one bank)."""
        d = os.path.join(self.path, "targets", name, "panel")
        T = pd.read_csv(os.path.join(d, "2_leaderboard_by_bank.csv" if bank else "1_leaderboard_overall.csv"))
        return T[T["bank"] == bank].reset_index(drop=True) if bank else T

    def chain(self):
        return self._t("0_chain")

    def leaderboard(self, test="backtest", target=None):
        """test: backtest | oos. Methods: Chain (earlier targets' forecasts feed the later ones), Chain, actual
        inputs (what each target's own backtest measures), BVARX, Chain, reconciled (identities).
        Panel system: mean across banks of Panel chain and Panel chain, actual inputs (sy.by_bank() per bank)."""
        T = self._t("1_system_leaderboard")
        if "test" in T:
            T = T[T["test"] == test]
        if target:
            T = T[T["target"] == target]
        return T.reset_index(drop=True)

    def links(self):
        return self._t("6_links")

    def scenarios(self, target=None, method=None, scenario=None):
        T = self._t("4_scenario_paths")
        for k, v in (("target", target), ("method", method), ("scenario", scenario)):
            if v is not None:
                T = T[T[k] == v]
        return T.reset_index(drop=True)

    def ratios(self, outside_only=False):
        T = self._t("5_ratios")
        return T[T["outside_history"]].reset_index(drop=True) if outside_only else T

    def bvarx_coefficients(self):
        return self._t("7_bvarx_coefficients")

    def sur_coefficients(self, target=None):
        """The chain's equations estimated jointly (SUR, feasible GLS): coefficient, standard error, t."""
        T = self._t("7b_sur_coefficients")
        return T[T["equation"] == target].reset_index(drop=True) if target else T

    def sur_correlation(self):
        """Correlation of the equations' surprises: what SUR uses, and what makes the fan charts joint."""
        return self._t("7c_sur_residual_correlation").set_index("target")

    def fan(self, target=None, scenario=None):
        """SUR fan charts: 5/25/50/75/95th percentiles of every target's level (joint draws)."""
        T = self._t("4b_sur_fan")
        for k, v in (("target", target), ("scenario", scenario)):
            if v is not None:
                T = T[T[k] == v]
        return T.reset_index(drop=True)

    def vecm_coefficients(self):
        """VECM: error-correction (adjustment) coefficients, lags, macro terms, and the estimated long runs."""
        return self._t("7d_vecm_coefficients")

    def cointegration(self):
        """ADF test of each long-run relation and the Johansen trace test on the targets' levels."""
        return self._t("7e_cointegration")

    def target(self, name):
        """One target's complete single-target results (a panel system: use target_leaderboard)."""
        if self.is_panel:
            return self.target_leaderboard(name)
        return Forecaster.load(os.path.join(self.path, "targets", name))

    def __repr__(self):
        if not self.path:
            return f"System({self.config!r}, not fitted)"
        c = self.chain()
        kind = "panel system" if self.is_panel else "system"
        return f"System({self.path}, {kind})\n" + "\n".join(f"  {r.order}. {r.target}: {r.best_model}" for r in c.itertuples())
