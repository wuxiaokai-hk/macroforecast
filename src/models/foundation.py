"""Pretrained time-series foundation models (zero-shot: no training on our data).

  Chronos-Bolt (Amazon, 2024)  forecasts target growth from its own history only. It cannot use scenarios,
                               so every scenario gets the same forecast; it is a strong "AI benchmark".
  Chronos-2   (Amazon, 2025)   also takes covariates, with their future values: the literature-core drivers and
                               their scenario paths, so it does respond to the scenarios.

Requires `pip install chronos-forecasting` (installs PyTorch). The weights are downloaded from Hugging Face
on first use (chronos-bolt-small about 190 MB, chronos-2 about 480 MB) and cached. Without the package or
without internet access the models are skipped with a message; the rest of the pipeline is unaffected. To
run offline, download the model folders once and put their local paths in models.foundation.models[].id.

Inference takes well under a second per forecast on a normal CPU.
Context: the organic quarterly log change of the balance, with the dummy (COVID) quarters set to missing,
which these models handle natively.
"""
import numpy as np

from ..modelling.ardl import make_panel

_PIPES = {}
_ERRORS = {}


def available():
    """Check whether chronos-forecasting and PyTorch are installed.

    Returns:
        (ok, reason) where reason explains how to install when ok is False.
    """
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}. Install with: pip install chronos-forecasting"


def load(model_id, device="cpu"):
    """Load (once) a Chronos pipeline. Returns None and remembers the error if it cannot be loaded.

    Args:
        model_id: Hugging Face id or local folder. device: cpu or cuda.
    """
    if model_id in _PIPES:
        return _PIPES[model_id]
    if model_id in _ERRORS:
        return None
    try:
        import torch
        from chronos import BaseChronosPipeline
        _PIPES[model_id] = BaseChronosPipeline.from_pretrained(model_id, device_map=device, dtype=torch.float32)
        return _PIPES[model_id]
    except Exception as e:
        _ERRORS[model_id] = f"{type(e).__name__}: {str(e)[:300]}"
        return None


def load_error(model_id):
    """Why a foundation model could not be loaded (empty if it loaded).

    Args:
        model_id: Hugging Face id or local folder.

    Returns:
        str.
    """
    return _ERRORS.get(model_id, "")


class FoundationModel:
    """Pretrained zero-shot forecaster (Chronos-Bolt or Chronos-2); nothing is estimated on our data."""
    family = "AI foundation model"

    def __init__(self, name, model_id, covariates=None, device="cpu", note=""):
        """Create a foundation model.

        Args:
            name: display name. model_id: Hugging Face id or local path. covariates: driver features
            (Chronos-2 only; empty = own history only). device: "cpu" or "cuda". note: free text.
        """
        self.name, self.model_id, self.device = name, model_id, device
        self.features = list(covariates or [])
        self.family, self.note = "AI foundation model", note
        self.params, self.res = None, None

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): model id and covariates.

        Returns:
            dict.
        """
        return {"class": "Foundation", "model_id": self.model_id, "covariates": self.features}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): covariates, or "(own history only)".

        Returns:
            list of str.
        """
        return self.features or ["(own history only)"]

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): panel with the target, covariates and dummies.

        Args:
            ctx: Context. mdf: macro DataFrame.

        Returns:
            DataFrame.
        """
        return make_panel(ctx, mdf, self.features)

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): zero-shot: only remembers where the context window starts.

        Args:
            P: panel. start, end: window (end is not used).

        Returns:
            self.
        """
        self.start = start                     # zero-shot: nothing is estimated
        return self

    def _context(self, P, origin):
        """History passed to the model: from start (or the first complete covariate row) to the origin;
        COVID quarters set to missing.

        Args:
            P: panel. origin: last known quarter.

        Returns:
            (target series, covariate frame) for the context window.
        """
        W = P.loc[self.start:origin]
        if self.features:                      # covariates must be complete in the context window
            first = W[self.features].notna().all(axis=1).idxmax()
            W = W.loc[first:]
        y = W["y"].astype(float).copy()
        dummy_cols = [c for c in P.columns if c.startswith("D_")]
        if dummy_cols:
            y[W[dummy_cols].sum(axis=1) > 0] = np.nan
        return W, y.values

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): median forecast from the pretrained model.

        Args:
            P: panel (future covariate values come from it). origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        if H <= 0:
            return np.zeros(0)
        pipe = load(self.model_id, self.device)
        if pipe is None:
            raise RuntimeError(f"{self.name}: model could not be loaded ({load_error(self.model_id)})")
        import torch
        W, y = self._context(P, origin)
        if not self.features:
            q, _ = pipe.predict_quantiles(torch.tensor(y, dtype=torch.float32), prediction_length=H, quantile_levels=[0.5])
            q = q[0] if isinstance(q, list) else q
            return np.asarray(q).reshape(-1, H, 1)[0, :, 0].astype(float)
        fut = P.iloc[pos + 1:pos + 1 + H]
        inp = [{"target": y,
                "past_covariates": {f: W[f].values.astype(float) for f in self.features},
                "future_covariates": {f: fut[f].values.astype(float) for f in self.features}}]
        q, _ = pipe.predict_quantiles(inp, prediction_length=H, quantile_levels=[0.5])
        return np.asarray(q[0]).reshape(-1, H, 1)[0, :, 0].astype(float)
