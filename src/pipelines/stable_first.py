"""Approach F - 'Stable first': stability selection (src/selection/stability.py) keeps the variables a lasso picks in
most sub-samples of the history; each gets its best form and lag; the lowest-BIC subset with the expected signs is
the model. The selection probabilities say how sure the data are about every driver."""
import pandas as pd

from ..selection import stability
from ..selection.small_search import best_forms, subset_search
from ..utils.io import log


def run(cfg, ctx, F, forms):
    """Stability selection -> best form and lag -> lowest-BIC subset with the expected signs.

    Returns:
        dict with the features and the tables written to stable_first/.
    """
    st = stability.settings(cfg)
    T, stable, info = stability.select(cfg, ctx, F, log)
    log(f"[stable_first] {info['variables']} variables in {info['groups']} groups; {len(stable)} picked in at least "
        f"{st['threshold']:.0%} of {info['draws']} sub-samples (expected false ones at most "
        f"{info['expected_false_kept_at_most']:.2f}): {stable}")
    if not stable and len(T):                         # nothing stable: the most stable few, flagged
        stable = list(T["representative"].head(2))
        log(f"[stable_first] nothing reached the threshold; using the two most stable: {stable}")
    bf = best_forms(ctx, F, stable, float(st["t_min"]))
    best, S = subset_search(ctx, F, [f for f, _ in bf.values()], int(st["max_vars"]))
    log(f"[stable_first] chosen: {best}")
    tables = {"F1_selection_probabilities": T, "F2_settings": pd.DataFrame([info]),
              "F3_best_form": pd.DataFrame([{"variable": k, "feature": f, "t": t} for k, (f, t) in bf.items()]),
              "F4_search": S}
    funnel = [("Candidate variables", info["variables"]), ("Groups of near-identical variables", info["groups"]),
              (f"Picked in >= {st['threshold']:.0%} of sub-samples", len(stable)), ("Chosen", len(best))]
    return {"name": "Stable first", "features": best, "tables": tables, "funnel": funnel}
