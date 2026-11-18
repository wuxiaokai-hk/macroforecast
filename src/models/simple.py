"""Same-growth models: the target grows at the same rate as another variable.

    exact:              g_t = x_t                       (constant ratio of the target to the variable)
    with average gap:   g_t = c + x_t                   (c = average of g - x over the estimation window)

g_t is quarterly log growth of the target and x_t the quarterly log growth of the chosen variable
(e.g. household disposable income: a stable debt-to-income ratio). Nothing else is estimated: the slope is
fixed at 1, so the model is a transparent rule of thumb and a benchmark for the estimated models.
"""
import numpy as np
import pandas as pd

from ..modelling.ardl import make_panel

FAMILY = "Same growth as a driver"


class SameGrowth:
    """Target growth equals the growth of one variable (optionally plus a constant gap). Same interface as ARDL."""
    kind = "same_growth"

    def __init__(self, name, feature, gap=False, family=FAMILY, note=""):
        """Create the rule.

        Args:
            name: display name. feature: the variable's growth as a feature name, e.g. "FYPD.ION|dlog|L0".
            gap: add the average historical gap (target growth minus the variable's growth) as a constant.
            family: model family. note: free text for the model list.
        """
        self.name, self.features, self.gap = name, [feature], bool(gap)
        self.family, self.note = family, note
        self.params, self.res, self.ar = None, None, False

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): target, dummies and the variable's growth.

        Args:
            ctx: Context. mdf: macro DataFrame (history or a scenario).

        Returns:
            DataFrame with y, the feature and the D_ dummies.
        """
        return make_panel(ctx, mdf, self.features)

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): settings that define the model (for the cache).

        Returns:
            dict.
        """
        return {"class": "SameGrowth", "features": self.features, "gap": self.gap}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): the one variable.

        Returns:
            list with one feature name.
        """
        return list(self.features)

    def fit(self, P, start, end, fast=False):
        """Part of the model interface (see src/models/zoo.py): the constant gap (if any) over start..end.

        Quarters with a one-off dummy (e.g. COVID) are left out of the average.

        Args:
            P: panel. start, end: estimation window. fast: unused (nothing heavy to skip).

        Returns:
            self, with params = {const, feature: 1.0}.
        """
        f = self.features[0]
        W = P.loc[start:end]
        d = W["y"] - W[f]
        dummies = [c for c in W.columns if c.startswith("D_")]
        if dummies:
            d = d[~(W[dummies] != 0).any(axis=1)]
        c = float(d.dropna().mean()) if self.gap and d.notna().any() else 0.0
        self.params = pd.Series({"const": c, f: 1.0})
        self.resid = d.dropna() - c
        return self

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): g = const + x for each quarter ahead.

        Args:
            P: panel. origin: last known quarter. H: quarters ahead.

        Returns:
            array of up to H forecasts of quarterly log growth.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        x = P[self.features[0]].iloc[pos + 1:pos + 1 + H].values
        return self.params["const"] + np.asarray(x, dtype=float)

    def detail_rows(self, ctx, P):
        """Rows for the model detail page: the fixed slope, the gap and how well the rule fitted history.

        Args:
            ctx: Context. P: panel.

        Returns:
            list of dict rows (model, term, coef, reading).
        """
        f = self.features[0]
        c = float(self.params["const"]) if self.params is not None else 0.0
        rows = [{"model": self.name, "term": f, "coef": 1.0,
                 "reading": "fixed at 1: the target grows at the same rate as this variable"},
                {"model": self.name, "term": "const", "coef": c,
                 "reading": (f"average gap: the target grew {c * 400:+.2f} pp a year faster than the variable"
                             if self.gap else "no gap (exact same growth)")}]
        r = getattr(self, "resid", None)
        if r is not None and len(r):
            rows.append({"model": self.name, "term": "fit over history", "coef": float(np.sqrt((r ** 2).mean())),
                         "reading": "root mean squared gap between target growth and the rule, per quarter (log points)"})
        return rows
