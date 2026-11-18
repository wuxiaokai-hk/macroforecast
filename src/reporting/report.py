"""Write tables (CSV + one Excel workbook per folder), charts and a markdown summary."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "lines.linewidth": 2, "legend.frameon": False,
    "legend.labelcolor": INK2, "figure.dpi": 110,
})


def write_tables(folder, tables):
    """Write each table as CSV and all of them as sheets of tables.xlsx (with sized columns).

    Args:
        folder: output folder. tables: dict name -> DataFrame.
    """
    os.makedirs(folder, exist_ok=True)
    with pd.ExcelWriter(os.path.join(folder, "tables.xlsx"), engine="openpyxl") as w:
        for name, df in tables.items():
            df.to_csv(os.path.join(folder, f"{name}.csv"), index=False)
            sheet = re.sub(r"[\[\]\*\?/\\:]", "_", name)[:31]
            df.to_excel(w, sheet_name=sheet, index=False)
            ws = w.sheets[sheet]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200]) + 2
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width))


FAMILY_ORDER = ["Benchmark", "Data-driven", "Literature", "Hybrid", "Auto ARDL", "Same growth as a driver",
                "Horizon combination", "Regime switching", "Statistical time series", "Machine learning",
                "Deep learning", "AI foundation model"]
FAMILY_COLOR = {"Benchmark": "#8a8984", "Data-driven": PAL[0], "Literature": PAL[3], "Hybrid": PAL[2],
                "Machine learning": PAL[6], "AI foundation model": PAL[4],
                "Statistical time series": PAL[1], "Deep learning": PAL[7], "Regime switching": "#0f8a8a",
                "Auto ARDL": "#3b5ba5", "Same growth as a driver": "#8a6d3b", "Horizon combination": "#b5487a"}


def rank_column(comp, H):
    """The error column to rank by: avg_1_<H>_pct, or the longest available horizon up to H (then any).

    With few forecast origins the longest horizon that has errors can fall between the report horizons,
    so the exact column may not exist.

    Args:
        comp: table with avg_1_<h>_pct columns. H: preferred horizon.
    """
    want = f"avg_1_{int(H)}_pct"
    if want in comp.columns:
        return want
    hs = sorted(int(c.split("_")[2]) for c in comp.columns if c.startswith("avg_1_") and c.endswith("_pct"))
    if not hs:
        raise KeyError(f"no avg_1_<h>_pct column to rank by (wanted {want})")
    below = [h for h in hs if h <= int(H)]
    return f"avg_1_{(below or hs)[-1]}_pct"


def best_per_family(comp, H):
    """AR(1) plus the most accurate model of every other family (for charts that cannot show 20 lines).

    Args:
        comp: model comparison table.
    """
    c = comp.sort_values(rank_column(comp, H))
    keep = ["AR(1) benchmark"] + [c[c["family"] == f]["model"].iloc[0] for f in FAMILY_ORDER[1:] if (c["family"] == f).any()]
    return list(dict.fromkeys(m for m in keep if m in set(comp["model"])))


def scenario_models(comp, H):
    """Models shown in the scenario chart: the data-driven ones and the best literature and hybrid models.

    Args:
        comp: model comparison table.
    """
    c = comp.sort_values(rank_column(comp, H))
    out = list(c[c["family"] == "Data-driven"]["model"])
    for f in ("Literature", "Hybrid"):
        if (c["family"] == f).any():
            out.append(c[c["family"] == f]["model"].iloc[0])
    return out[:4]


def chart_horizon(path, by_h, models=None):
    """Line chart of balance error by quarters ahead, one line per model, coloured by family.

    Args:
        path: PNG to write. by_h: backtest errors by horizon. models: subset to plot (None = all).
    """
    fig, ax = plt.subplots(figsize=(9.5, 5))
    d0 = by_h if models is None else by_h[by_h["model"].isin(models)]
    for name, d in d0.groupby("model", sort=False):
        fam = d["family"].iloc[0] if "family" in d else ""
        ax.plot(d["horizon_q"], d["MAPE_exCOVID_pct"], color=FAMILY_COLOR.get(fam, PAL[0]), marker="o", ms=4,
                ls="--" if "benchmark" in name.lower() else "-", label=name)
    ax.set_xlabel("Quarters ahead")
    ax.set_ylabel("MAPE of the balance (%)")
    ax.set_ylim(0, None)
    ax.set_title("Rolling backtest: balance error by horizon (best model of each family)")
    ax.legend(fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_family_bars(path, comp, horizons):
    """Average balance error over 1..h quarters for every model, grouped by family; one panel per horizon.

    Args:
        comp: model comparison table.
    """
    c = comp.copy()
    c["fo"] = c["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}).fillna(99)
    H = max(horizons)
    c = c.sort_values(["fo", f"avg_1_{H}_pct"], ascending=[True, True]).iloc[::-1]
    fig, axs = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons) + 3, 0.34 * len(c) + 1.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, h in zip(axs, horizons):
        v = c[f"avg_1_{h}_pct"].values
        ax.barh(c["model"], v, color=[FAMILY_COLOR.get(f, PAL[0]) for f in c["family"]], height=0.7)
        for y_, x_ in enumerate(v):
            ax.text(x_ + 0.01, y_, f"{x_:.2f}", va="center", fontsize=8, color=INK2)
        ax.set_title(f"1–{h} quarters ahead")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, np.nanmax(c[[f"avg_1_{x}_pct" for x in horizons]].values) * 1.15)
    axs[0].tick_params(axis="y", labelsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAMILY_ORDER if f in set(c["family"])]
    fig.legend(handles, [f for f in FAMILY_ORDER if f in set(c["family"])], loc="lower center",
               ncol=6, fontsize=9, bbox_to_anchor=(0.55, -0.02))
    fig.suptitle("Average balance error (MAPE %, dummy quarters excluded), by model family", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_scenarios(path, scen, level, models):
    """Balance paths under each Moody's scenario, one panel per model.

    Args:
        path: PNG. scen: scenario forecasts. level: actual balance. models: models to show.
    """
    fig, axs = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.4), sharey=True, squeeze=False)
    hx = level.index[-16:].to_timestamp()
    cols = [PAL[0], PAL[2], PAL[3], PAL[7], PAL[6], PAL[4]]
    for ax, m in zip(axs[0], models):
        ax.plot(hx, level.iloc[-16:] / 1000, color=INK, label="Actual")
        for i, (sname, d) in enumerate(scen[scen["model"] == m].groupby("scenario", sort=False)):
            xs = pd.PeriodIndex(d["date"], freq="Q").to_timestamp()
            ax.plot(np.r_[hx[-1:], xs], np.r_[level.iloc[-1], d["balance_forecast"].values] / 1000,
                    color=cols[i % len(cols)], label=sname)
        ax.set_title(m)
        ax.legend(fontsize=9)
    axs[0][0].set_ylabel("Balance (thousands of input units)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_stability(path, freq, threshold, top=25):
    """Bar chart of Elastic Net stability-selection frequencies.

    Args:
        path: PNG. freq: selection frequency table. threshold: frequency needed to keep a feature. top: bars shown.
    """
    d = freq.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(d) + 1.2))
    ax.barh(d["feature"], d["selection_frequency"],
            color=[PAL[0] if v >= threshold else "#b9b8b2" for v in d["selection_frequency"]], height=0.65)
    ax.axvline(threshold, color=INK2, ls=":", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of subsamples in which the feature was selected")
    ax.set_title("Elastic Net stability selection (blue = kept)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summary_md(path, cfg, results, comparison, diag, overlap):
    """Write the run summary (summary.md): settings, chosen drivers, comparison and diagnostics.

    Args:
        path: file to write. cfg: configuration. results: per-approach results. comparison: model comparison.
        diag: diagnostics table. overlap: drivers shared by the approaches.
    """
    H = cfg["sample"]["horizon"]
    hz = cfg["sample"]["report_horizons"]
    L = [f"# Forecasting run: {cfg['run_name']}", "",
         f"Config: `{cfg['config_file']}`  ",
         f"Target: `{cfg['data']['bank']['target_column']}` (organic quarterly log change). "
         f"Estimation {cfg['sample']['estimation_start']}–{cfg['sample']['hist_end']}; backtest origins from "
         f"{cfg['sample']['backtest_start']}, 1–{H} quarters ahead; models compared at {', '.join(map(str, hz))} quarters.", ""]
    for r in results:
        L += [f"## {r['name']}", "", "| Step | Count |", "|---|---|"] + [f"| {a} | {b} |" for a, b in r["funnel"]]
        L += ["", "Final drivers: " + ", ".join(f"`{f}`" for f in r["features"]), ""]
    cols = ["model", "family"] + [f"avg_1_{h}_pct" for h in hz] + [f"at_{h}q_pct" for h in hz] + [f"rank_1_{H}"]
    c = comparison.sort_values(f"avg_1_{H}_pct")[cols]
    L += [f"## All models (balance MAPE %, dummy quarters excluded; avg_1_h = average over 1..h quarters, at_hq = at h)", "",
          c.to_markdown(index=False) if hasattr(c, "to_markdown") else c.to_string(index=False), ""]
    if overlap is not None:
        L += [f"Drivers chosen by both data-driven approaches: {overlap or 'none'}", ""]
    d = diag[diag["n_obs"].notna()] if "n_obs" in diag else diag
    L += ["## Diagnostics (regression models)", "", d.to_markdown(index=False) if hasattr(d, "to_markdown") else d.to_string(index=False)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def chart_oos(path, by_h, summ, H):
    """Out-of-sample error by horizon: AR(1), the best model of each family, and the look-ahead variants.

    Args:
        summ: out-of-sample summary per model.
    """
    keep = best_per_family(summ[summ["family"] != "Look-ahead check"], max(by_h["horizon_q"]))
    keep += [m for m in summ["model"] if "full-sample" in m]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for m in keep:
        d = by_h[by_h["model"] == m]
        if d.empty:
            continue
        fam = d["family"].iloc[0]
        ls = "--" if "benchmark" in m.lower() else (":" if fam == "Look-ahead check" else "-")
        ax.plot(d["horizon_q"], d["MAPE_pct"], color=FAMILY_COLOR.get(fam, "#b9b8b2"), ls=ls, marker="o", ms=3.5, label=m)
    ax.set_xlabel("Quarters ahead")
    ax.set_ylabel("MAPE of the balance (%)")
    ax.set_ylim(0, None)
    ax.set_title("Out-of-sample error, everything re-selected at each origin (dotted: drivers fixed from the full sample)")
    ax.legend(fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_driver_timeline(path, picks):
    """Which drivers each approach picks at each out-of-sample origin (dot chart).

    Args:
        path: PNG. picks: drivers by origin (model, origin, drivers).
    """
    aps = list(dict.fromkeys(picks["model"]))
    drivers = sorted({d for x in picks["drivers"] for d in x.split(", ") if d})
    origins = list(dict.fromkeys(picks["origin"]))
    fig, axs = plt.subplots(1, len(aps), figsize=(5.5 * len(aps) + 2, 0.35 * len(drivers) + 1.8), sharey=True,
                            squeeze=False)
    for ax, apn in zip(axs[0], aps):
        g = picks[picks["model"] == apn].set_index("origin")
        for j, o in enumerate(origins):
            used = set(g.loc[o, "drivers"].split(", ")) if o in g.index else set()
            ys = [i for i, d in enumerate(drivers) if d in used]
            ax.scatter([j] * len(ys), ys, s=60, color=PAL[0])
        ax.set_xticks(range(len(origins)))
        ax.set_xticklabels(origins, rotation=60, fontsize=8)
        ax.set_yticks(range(len(drivers)))
        ax.set_yticklabels(drivers, fontsize=8)
        ax.set_title(apn)
        ax.set_xlabel("Forecast origin")
    fig.suptitle("Drivers chosen at each origin (data up to that origin only)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_holdout(path, hold, level, models=None):
    """Holdout chart: actual balance and each model's forecast of the last 12 quarters.

    Args:
        path: PNG. hold: holdout forecasts. level: actual balance. models: subset to show.
    """
    if models:
        hold = hold[hold["model"].isin(models)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    first = hold.iloc[0]
    act = hold.drop_duplicates("date")
    start = pd.Period(first["train_end"], freq="Q")
    hist = level.loc[:start].iloc[-12:]
    ax.plot(hist.index.to_timestamp(), hist / 1000, color=INK, label="Actual (to training end)")
    x0, y0 = start.to_timestamp(), level.loc[start]
    xs = pd.PeriodIndex(act["date"], freq="Q").to_timestamp()
    ax.plot([x0] + list(xs), np.r_[y0, act["actual_organic"].values] / 1000, color=INK, ls="--",
            label="Actual (held out, organic)")
    for m, d in hold.groupby("model", sort=False):
        fam = d["family"].iloc[0] if "family" in d else ""
        ax.plot([x0] + list(pd.PeriodIndex(d["date"], freq="Q").to_timestamp()), np.r_[y0, d["forecast"].values] / 1000,
                color=FAMILY_COLOR.get(fam, PAL[0]), marker="o", ms=3, ls="--" if "benchmark" in m.lower() else "-",
                label=f"{m}")
    ax.axvline(start.to_timestamp(), color=INK2, ls=":", lw=1)
    ax.set_title(f"Holdout: models chosen and estimated on data to {first['train_end']}")
    ax.set_ylabel("Balance (thousands of input units)")
    ax.legend(fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_regimes(path, reg, level):
    """Regimes over history: target growth on top, then each regime-switching model's regime indicator.

    Args:
        path: PNG. reg: regime table (model, date, value, meaning). level: actual balance.
    """
    models = list(dict.fromkeys(reg["model"]))
    fig, axs = plt.subplots(len(models) + 1, 1, figsize=(10, 1.55 * (len(models) + 1) + 0.6), sharex=True)
    from ..utils import target as tg
    g = tg.per_year(tg.growth(level))
    axs[0].plot(g.index.to_timestamp(), g.values, color=INK, lw=1.2)
    axs[0].set_title(("Growth of the target (annualised %" if tg.is_log() else "Change of the target (annualised, units")
                     + ", includes inorganic events)", fontsize=9, loc="left")
    for ax, m in zip(axs[1:], models):
        d = reg[reg["model"] == m]
        x = pd.PeriodIndex(d["date"], freq="Q").to_timestamp()
        ax.fill_between(x, 0, d["value"].values, step="mid", color="#0f8a8a", alpha=0.35, lw=0)
        ax.plot(x, d["value"].values, color="#0f8a8a", lw=1.1, drawstyle="steps-mid")
        ax.set_title(f"{m}: {d['meaning'].iloc[0]}", fontsize=9, loc="left")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
