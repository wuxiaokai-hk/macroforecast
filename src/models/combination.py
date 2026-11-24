"""Forecast combinations: an average of member models' forecasts.

weighting = "equal"    : simple average (robust; Stock & Watson 2004, Timmermann 2006).
weighting = "inv_mse"  : Bates-Granger weights proportional to 1 / MSE of each member's recent forecasts
                         (made only with data up to the estimation end), shrunk half-way towards equal weights,
                         which guards against chasing noise with ~100 observations.
"""
import numpy as np


class Combination:
    """Forecast combination: a weighted average of member models' forecasts.

    weighting "equal" gives equal weights; "inv_mse" gives Bates-Granger weights from each member's recent
    out-of-sample errors (last `window` origins, horizons 1..hmax), shrunk by `shrink` towards equal weights.
    """
    family = "Hybrid"

    def __init__(self, name, members, weighting="equal", family="Hybrid", note="", window=16, hmax=4, shrink=0.5,
                 weights=None):
        """Create a combination.

        Args:
            name: display name. members: list of models. weighting: "equal" or "inv_mse".
            family, note: as for ARDL. window: origins used for the weights. hmax: horizons used.
            shrink: share of the way back to equal weights (0 = pure inverse-MSE).
        """
        self.name, self.members, self.weighting = name, list(members), weighting
        self.family, self.note = family, note
        self.window, self.hmax, self.shrink = window, hmax, shrink
        self.weights = np.full(len(self.members), 1 / len(self.members))
        if weighting == "fixed" and weights is not None:     # given weights (e.g. posterior model probabilities)
            w = np.asarray(weights, float)
            self.weights = w / w.sum()
        self.params, self.res = None, None

    @property
    def features(self):
        """All drivers of the members, without duplicates.

        Returns:
            list of feature names.
        """
        return list(dict.fromkeys(f for m in self.members for f in m.features))

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): settings of the combination and of every member.

        Returns:
            dict.
        """
        return {"class": "Combination", "weighting": self.weighting, "members": [m.spec() for m in self.members],
                "window": self.window, "hmax": self.hmax, "shrink": self.shrink,
                "weights": [round(float(w), 6) for w in self.weights] if self.weighting == "fixed" else None}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): members with their current weights.

        Returns:
            list of str like "Screen first (40%)".
        """
        if self.weighting == "fixed":               # averaged models: show each one's drivers
            return [f"{' + '.join(m.features) or 'no macro driver'} ({w:.0%})" for m, w in zip(self.members, self.weights)]
        return [f"{m.name} ({w:.0%})" for m, w in zip(self.members, self.weights)]

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): one panel holding the columns every member needs.

        Args:
            ctx: Context. mdf: macro DataFrame.

        Returns:
            DataFrame.
        """
        P = None
        for m in self.members:
            Pm = m.panel(ctx, mdf)
            P = Pm if P is None else P.join(Pm[[c for c in Pm.columns if c not in P.columns]])
        return P

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): estimate the weights (inv_mse only) and every member.

        Args:
            P: panel. start, end: estimation window (Periods).

        Returns:
            self.
        """
        if self.weighting == "inv_mse":
            self.weights = self._weights(P, start, end)
        for m in self.members:
            m.fit(P, start, end)
        return self

    def _weights(self, P, start, end):
        """Members' squared errors of cumulative growth, 1..hmax quarters ahead, over the last `window` origins."""
        e_pos = P.index.get_loc(end)
        s_pos = P.index.get_loc(start)
        origins = [o for o in range(e_pos - self.window - self.hmax + 1, e_pos - self.hmax + 1) if o - s_pos >= 40]
        if len(origins) < 4:
            return np.full(len(self.members), 1 / len(self.members))
        y = P["y"].values
        dummy_cols = [c for c in P.columns if c.startswith("D_")]
        busy = P[dummy_cols].sum(axis=1).values > 0 if dummy_cols else np.zeros(len(P), bool)
        mse = []
        for m in self.members:
            err = []
            for o in origins:
                if busy[o + 1:o + 1 + self.hmax].any():
                    continue
                m.fit(P, start, P.index[o], fast=True)
                fc = m.forecast(P, P.index[o], self.hmax)
                err.append(np.cumsum(fc) - np.cumsum(y[o + 1:o + 1 + len(fc)]))
            mse.append(np.mean(np.square(err)) if err else np.nan)
        mse = np.array(mse)
        if np.isnan(mse).any() or (mse <= 0).any():
            return np.full(len(self.members), 1 / len(self.members))
        w = (1 / mse) / (1 / mse).sum()
        eq = np.full(len(w), 1 / len(w))
        return self.shrink * eq + (1 - self.shrink) * w

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): weighted average of the members' forecasts.

        Args:
            P: panel. origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        fcs = [m.forecast(P, origin, H) for m in self.members]
        n = min(len(f) for f in fcs)
        return np.sum([w * f[:n] for w, f in zip(self.weights, fcs)], axis=0)
