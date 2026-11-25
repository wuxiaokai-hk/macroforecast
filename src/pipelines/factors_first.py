"""Approach E - 'Factors first': the model is built from the theme factors (src/features/factors.py) only.

Every series takes part through its theme's index, so the drivers are stable concepts ("the housing index", "the
labour-market index") rather than one of many near-identical series. Each factor gets its best lag; the subset of
up to factors.max_vars factors with the lowest BIC and the expected signs is the model."""
import pandas as pd

from ..selection.small_search import best_forms, subset_search
from ..utils.io import log


def run(cfg, ctx, F, forms):
    """Theme factors -> best lag each -> lowest-BIC subset with the expected signs.

    Returns:
        dict with the features and the tables written to factors_first/.
    """
    st = cfg.get("factors") or {}
    facs = sorted({c.split("|")[0] for c in F.columns if c.startswith("FAC.")})
    if not facs:
        log("[factors_first] no theme factors in this run (factors.enabled: false, or too few series per theme)")
        return {"name": "Factors first", "features": [], "tables": {}, "funnel": []}
    bf = best_forms(ctx, F, facs, float(st.get("t_min", 1.0)))
    best, T = subset_search(ctx, F, [f for f, _ in bf.values()], int(st.get("max_vars", 4)))
    log(f"[factors_first] {len(facs)} factors, {len(bf)} with a usable lag; chosen: {best}")
    ft = getattr(ctx, "factor_table", pd.DataFrame())
    tables = {"E1_factors": ft, "E2_loadings": getattr(ctx, "factor_loadings", pd.DataFrame()),
              "E3_best_lag": pd.DataFrame([{"factor": k, "feature": f, "t": t} for k, (f, t) in bf.items()]),
              "E4_search": T}
    funnel = [("Series in the factors", int(ft["members"].sum()) if len(ft) else 0), ("Theme factors", len(facs)),
              ("Factors with a usable lag", len(bf)), ("Chosen", len(best))]
    return {"name": "Factors first", "features": best, "tables": tables, "funnel": funnel}
