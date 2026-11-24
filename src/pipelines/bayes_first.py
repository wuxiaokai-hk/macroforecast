"""Approach C - 'Bayesian first':
one feature per variable -> Bayesian model averaging with sign restrictions (MC3) -> inclusion probabilities ->
drivers with probability >= threshold (median probability model) -> group correlated picks -> shared final search.
The averaged forecast of the top models is also built as its own model ('Bayesian model average')."""
import pandas as pd

from ..modelling.search import exhaustive_search
from ..selection.bayes import bma_select
from ..selection.clustering import cluster, representatives
from ..utils.io import log


def run(cfg, ctx, F, forms):
    """Bayesian first approach: BMA inclusion probabilities -> group correlated picks -> exhaustive search with the
    same rules as the other approaches.

    Args:
        cfg: configuration. ctx: Context. F: all candidate features. forms: transforms per series.

    Returns:
        dict with the final features, the search table, the BMA tables (written to bayes_first/) and the
        averaged models.
    """
    b = cfg["bayes_first"]
    log(f"[bayes_first] Bayesian model averaging (MC3, {b['iterations']} draws) ...")
    r = bma_select(cfg, ctx, F)
    inc = r["inclusion"]
    m = r["meta"].iloc[0]
    log(f"[bayes_first] {int(m['candidates'])} candidates, {int(m['models_visited'])} models visited, "
        f"posterior mean size {m['posterior_mean_model_size']:.1f}")
    sel = inc[inc["inclusion_probability"] >= b["threshold"]].copy()
    if len(sel) < 2:
        log(f"[bayes_first] fewer than 2 drivers with probability >= {b['threshold']} - using the 3 most probable")
        sel = inc.head(3).copy()
    sel["score"] = sel["inclusion_probability"]
    Fs = F[sel["feature"]].iloc[ctx.est0:ctx.end + 1]
    sel["cluster"] = cluster(Fs, b["group_corr"]).reindex(sel["feature"]).values
    grp = representatives(sel.set_index("feature", drop=False), "score", b["primary_geography"], b["rep_tolerance"])
    cands = grp[grp["is_representative"]].sort_values("score", ascending=False)
    log(f"[bayes_first] {len(sel)} probable drivers -> {len(cands)} groups; searching ...")
    R, chosen = exhaustive_search(cfg, ctx, F[cands["feature"]], cands.reset_index(drop=True).set_index("feature"),
                                  "bayes_first")
    log(f"[bayes_first] chosen: {chosen}")
    tables = {"C1_bma_settings": r["meta"], "C2_bma_inclusion": inc, "C3_bma_top_models": r["top_models"],
              "C4_groups": grp.reset_index(drop=True), "C5_search": R,
              "C6_search_rules": pd.DataFrame(list(R.attrs.get("rules", {}).items()), columns=["step", "combinations"])}
    funnel = [("Candidate variables", forms["mnemonic"].nunique()),
              ("Features (forms x lags)", F.shape[1]),
              ("Candidates in the BMA (one per variable)", int(m["candidates"])),
              (f"Inclusion probability >= {b['threshold']}", int((inc["inclusion_probability"] >= b["threshold"]).sum())),
              ("Groups after correlation check", len(cands)), ("Final drivers", len(chosen))]
    return {"name": "Bayesian first", "features": chosen, "tables": tables, "funnel": funnel,
            "bma_models": r["models"]}
