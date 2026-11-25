"""Approach D - 'Focus first' (the explanatory model):
the drivers whose effect the user wants to know (impact.focus) are always in; the controls around them are chosen
by double selection (src/selection/double_selection.py). No search may drop a focus driver, so the model answers
"how does this variable affect the target", while the other approaches answer "what forecasts best".
It is backtested and ranked like every other model, so its accuracy is known; src/impact uses it for the effects."""
import pandas as pd

from ..selection.double_selection import double_selection, resolve_focus
from ..utils.io import log


def run(cfg, ctx, F, forms):
    """Focus drivers + double-selection controls.

    Args:
        cfg: configuration. ctx: Context. F: all candidate features. forms: transforms per series.

    Returns:
        dict with the features (focus first, then controls) and the tables written to focus_first/.
    """
    focus = resolve_focus(cfg, ctx, F, forms, log)
    log(f"[focus_first] focus drivers: {list(focus['feature'])}")
    controls, tab, st = double_selection(cfg, ctx, F, focus, log)
    log(f"[focus_first] double selection: {st['selected_any']} candidates picked by at least one equation "
        f"({st['excluded_near_duplicates']} near-duplicates of a focus driver excluded); controls: {controls}")
    tables = {"D1_focus_drivers": focus, "D2_double_selection": tab,
              "D3_settings": pd.DataFrame([st])}
    funnel = [("Candidate variables", forms["mnemonic"].nunique()), ("Features (forms x lags)", F.shape[1]),
              ("Focus drivers", len(focus)), ("Candidates picked by double selection", st["selected_any"]),
              ("Controls used", len(controls))]
    return {"name": "Focus first", "features": list(focus["feature"]) + controls, "tables": tables,
            "funnel": funnel, "focus": list(focus["feature"])}
