"""Charts of the impact analysis: sensitivity, response paths, scenario attribution, what-if."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils import target as tg  # noqa: E402

INK, MAIN, BAND, GREY, POS, NEG = "#1B2A41", "#1F6F8B", "#9CC3D3", "#B8C2CC", "#2E9E83", "#C2553F"
PALETTE = ["#1F6F8B", "#E3A72F", "#2E9E83", "#C2553F", "#5B4B8A", "#C77B30", "#8796A8", "#0F8A8A"]


def _unit():
    return "%" if tg.is_log() else "target units"


def sensitivity(path, sens, h):
    """Effect of a typical rise in each focus driver after h quarters: explanatory model with its 90% band, and
    the range of the tied models."""
    d = sens[sens["horizon"] == h].reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(d) + 1.6))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        if np.isfinite(r.get("typical_tied_min", np.nan)):
            ax.plot([r["typical_tied_min"], r["typical_tied_max"]], [yi - 0.18] * 2, color=GREY, lw=6,
                    solid_capstyle="butt", label="tied models (range)" if yi == y[0] else None)
        if np.isfinite(r.get("typical_band_lo", np.nan)):
            ax.plot([r["typical_band_lo"], r["typical_band_hi"]], [yi + 0.12] * 2, color=BAND, lw=3,
                    label="90% band" if yi == y[0] else None)
        v = r["typical_explanatory"]
        ax.plot(v, yi + 0.12, "o", color=POS if v >= 0 else NEG, ms=9,
                label="explanatory model" if yi == y[0] else None)
        ax.annotate(f"{v:+.2f}", (v, yi + 0.12), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['description'][:48]}\n{r['typical_shock']}" for _, r in d.iterrows()], fontsize=9)
    ax.set_xlabel(f"effect on the balance after {h} quarters ({_unit()})")
    ax.set_title("What a typical rise in each driver does to the balance", loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def response_paths(path, resp, ex_name, title, kind="typical"):
    """Quarter-by-quarter response to each shock: tied models in grey, the explanatory model with its band."""
    R = resp[resp["shock_type"] == kind] if "shock_type" in resp else resp
    shocks = list(dict.fromkeys(R["shock"]))
    if not shocks:
        return
    n = len(shocks)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows), squeeze=False)
    for ax, s in zip(axes.flat, shocks):
        d = R[R["shock"] == s]
        for m, g in d[(d["kind"] == "point") & (d["model"] != ex_name)].groupby("model"):
            ax.plot(g["quarter"], g["balance_effect"], color=GREY, lw=1)
        lo = d[(d["model"] == ex_name) & (d["kind"] == "band_lo")]
        hi = d[(d["model"] == ex_name) & (d["kind"] == "band_hi")]
        if len(lo) and len(hi):
            ax.fill_between(lo["quarter"], lo["balance_effect"], hi["balance_effect"], color=BAND, alpha=0.6, lw=0)
        e = d[(d["model"] == ex_name) & (d["kind"] == "point")]
        ax.plot(e["quarter"], e["balance_effect"], color=MAIN, lw=2.2)
        ax.axhline(0, color=INK, lw=0.7)
        lab = d["shock_label"].iloc[0] if "shock_label" in d and d["shock_label"].notna().any() else ""
        ax.set_title(f"{s} {lab}".strip(), fontsize=10, loc="left")
        ax.set_xlabel("quarters after the shock", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel(f"balance effect ({_unit()})", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def attribution(path, att, ex_name, h):
    """Each scenario against the base, split by driver group (explanatory model)."""
    a = att[(att["model"] == ex_name) & (att["horizon"] == h)]
    if a.empty:
        return
    scen = list(dict.fromkeys(a["scenario"]))
    groups = list(a.groupby("group")["contribution"].apply(lambda s: s.abs().max()).sort_values(ascending=False).index)
    col = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(scen) + 1.8))
    for i, s in enumerate(scen[::-1]):
        g = a[a["scenario"] == s].set_index("group")["contribution"]
        pos, neg = 0.0, 0.0
        for gr in groups:
            v = float(g.get(gr, 0.0))
            if v >= 0:
                ax.barh(i, v, left=pos, color=col[gr], label=gr if i == 0 else None)
                pos += v
            else:
                ax.barh(i, v, left=neg, color=col[gr], label=gr if i == 0 else None)
                neg += v
        tot = float(a.loc[a["scenario"] == s, "total"].iloc[0])
        ax.plot(tot, i, "D", color=INK, ms=7, label="total" if i == 0 else None)
        ax.annotate(f"{tot:+.2f}", (tot, i), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    ax.set_yticks(range(len(scen)))
    ax.set_yticklabels(scen[::-1])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel(f"difference from the base scenario after {h} quarters ({_unit()})")
    ax.set_title("Why the scenarios differ: contribution of each driver", loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def all(out, cfg, ctx, T, ex_name, h):   # noqa: A001
    """Draw every impact chart that has data."""
    if len(T["3_sensitivity"]):
        sensitivity(os.path.join(out, "sensitivity.png"), T["3_sensitivity"], h)
    if len(T["2_responses"]):
        response_paths(os.path.join(out, "responses.png"), T["2_responses"], ex_name,
                       "Response of the balance to a typical rise in each driver")
    if len(T["5_attribution"]):
        attribution(os.path.join(out, "attribution.png"), T["5_attribution"], ex_name, h)
    if len(T["6_what_if"]):
        response_paths(os.path.join(out, "what_if.png"), T["6_what_if"].assign(shock_type="typical"), ex_name,
                       "What-if: response of the balance")
