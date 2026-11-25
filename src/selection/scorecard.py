"""The recommended model: every model competes on equal terms, scored on forecasting, interpretation and sensitivity.

    score = w_acc * accuracy + w_int * interpretation + w_sens * sensitivity        (weights: recommend.weights)

Accuracy (0-1)       best error / the model's error: average balance error over 1..H quarters ahead, out of sample
                     when run_oos.py has run, else the rolling backtest. Gate: a model whose error is more than
                     recommend.max_error_ratio x the best is not eligible, however readable it is.
Interpretation (0-1) transparency of the model type x the average of
                       - signs: share of driver coefficients with the expected sign (models with coefficients)
                       - parsimony: 1 with at most recommend.max_drivers drivers, then max_drivers / drivers
                       - economic drivers: share of its drivers that are not other outcomes of the economy (credit
                         aggregates, deposits: impact.excluded_control_themes), which explain nothing about it
                     Transparency: a written equation (regression, error correction, auto-ARDL) 1.0; same growth as a
                     driver 0.9; ARIMAX / combinations of equations 0.7-0.8; regime switching 0.6; horizon
                     combinations 0.5; machine learning 0.2; deep learning and foundation models 0.1
                     (recommend.transparency overrides any family).
Sensitivity (0-1)    does the model respond to the economy the way it should? The average of
                       - coverage: share of the focus drivers it responds to (a typical rise, src/impact)
                       - direction: share of those responses with the expected sign
                       - scenarios: share of the scenarios whose gap from the base scenario has the same sign as the
                         explanatory model's (the model built to measure effects)
                     A model without macro drivers (AR(1), mean growth) scores 0: it cannot answer "what if".
                     Gate: a model must respond to at least recommend.min_coverage of the focus drivers (default a
                     quarter, i.e. one of four) to be recommended; 0 switches the gate off.

Models without a fitted object (horizon combinations) are scored on what is available. Look-ahead checks are never
eligible. Writes outputs/<run>/recommendation/: 1_scorecard.csv, recommended.json, summary.md, scorecard.png.
"""
import json
import os

import numpy as np
import pandas as pd

TRANSPARENCY = {"Benchmark": 1.0, "Literature": 1.0, "Data-driven": 1.0, "Explanatory": 1.0, "Auto ARDL": 1.0,
                "Hybrid": 0.9, "Same growth as a driver": 0.9, "Statistical time series": 0.7,
                "Regime switching": 0.6, "Horizon combination": 0.5, "Machine learning": 0.2, "Deep learning": 0.1,
                "AI foundation model": 0.1}
DEFAULTS = {"weights": {"accuracy": 0.5, "interpretation": 0.25, "sensitivity": 0.25}, "max_error_ratio": 1.5,
            "min_coverage": 0.25, "max_drivers": 5, "transparency": {}}


def settings(cfg):
    r = dict(DEFAULTS)
    r.update({k: v for k, v in (cfg.get("recommend") or {}).items() if v is not None})
    w = {**DEFAULTS["weights"], **(r.get("weights") or {})}
    tot = sum(float(v) for v in w.values()) or 1.0
    r["weights"] = {k: float(v) / tot for k, v in w.items()}
    return r


def _transparency(name, family, model, table):
    from ..reporting.formula import has_formula
    if model is not None and has_formula(model):
        return 1.0, "written equation"
    cls = type(model).__name__ if model is not None else ""
    if cls == "Combination":
        return 0.7, "average of models"
    v = table.get(family, 0.5)
    return float(v), family.lower() if family else "unknown type"


def _n_drivers(model, drivers_text):
    feats = getattr(model, "features", None) if model is not None else None
    if feats:
        return len({str(f).split("|")[0] for f in feats})
    s = str(drivers_text or "")
    if not s or s == "(none)" or s == "nan":
        return 0
    return len({d.replace("long run: ", "").split("|")[0].split(":")[0].strip() for d in s.split(", ") if d.strip()})


def _outcome_codes(cfg):
    """Series whose theme is another outcome of the economy (same rule as the explanatory model's controls)."""
    try:
        from ..selection.double_selection import outcome_themes
        from ..utils.io import load_metadata
        meta = load_metadata(cfg)

        class _C:
            pass
        c = _C()
        c.meta = meta
        th = outcome_themes(cfg, c)
        return set(meta.index[meta["theme"].isin(th)])
    except Exception:
        return set()


def _codes(model, drivers_text):
    feats = getattr(model, "features", None) if model is not None else None
    if feats:
        return {str(f).split("|")[0] for f in feats}
    s = str(drivers_text or "")
    if not s or s in ("(none)", "nan"):
        return set()
    return {d.replace("long run: ", "").split("|")[0].split(":")[0].strip() for d in s.split(", ") if "|" in d}


def score(cfg, models, ranking, run_dir, basis, log=print):
    """Score every model and pick the recommended one.

    Args:
        cfg: configuration. models: {name: fitted model} (may miss some, e.g. horizon combinations).
        ranking: table with model, family, drivers and avg_1_<h>_pct columns (backtest comparison or OOS summary).
        run_dir: the run folder (reads comparison/ and impact/). basis: "backtest" or "out of sample".

    Returns:
        DataFrame scorecard sorted best first (eligible first), or None.
    """
    st = settings(cfg)
    W = st["weights"]
    cols = sorted([c for c in ranking.columns if c.startswith("avg_1_") and c.endswith("_pct")],
                  key=lambda c: int(c.split("_")[2]))
    if not cols:
        return None
    col = cols[-1]
    R = ranking[(ranking["family"] != "Look-ahead check") & ranking[col].notna()].copy()
    if R.empty:
        return None
    best_err = float(R[col].min())
    rd = lambda *p: pd.read_csv(os.path.join(run_dir, *p)) if os.path.exists(os.path.join(run_dir, *p)) \
        else pd.DataFrame()  # noqa: E731
    drv = rd("comparison", "3_drivers.csv")
    allr = rd("impact", "7_all_model_responses.csv")
    scen = rd("comparison", "6_scenario_forecasts.csv")
    h_imp = int((cfg.get("impact") or {}).get("compare_horizon", 8))
    if len(allr) and h_imp not in set(allr["horizon"]):
        h_imp = int(allr["horizon"].max())
    focus_n = allr["shock"].nunique() if len(allr) else 0
    ex_name = next((n for n in R["model"] if str(n).startswith("Explanatory")), None)
    base = cfg["data"]["macro"]["base_scenario"]
    gaps = {}
    if len(scen):
        hq = scen.groupby(["model", "scenario"], sort=False).nth(min(h_imp, int(cfg["sample"]["horizon"])) - 1)
        hq = hq.set_index(["model", "scenario"])["balance_forecast"]
        for m in hq.index.get_level_values(0).unique():
            v = hq.loc[m]
            if base in v.index:
                gaps[m] = (v.drop(base) / v[base] - 1)
    ref = gaps.get(ex_name)
    if ref is None and gaps:                                    # no explanatory model: the median model
        ref = pd.concat(gaps.values(), axis=1).median(axis=1)
    outcome = _outcome_codes(cfg)
    rows = []
    for _, r in R.iterrows():
        nm, fam = r["model"], r["family"]
        m = models.get(nm)
        acc = best_err / float(r[col]) if float(r[col]) > 0 else 1.0
        ratio = float(r[col]) / best_err if best_err > 0 else 1.0
        tr, tr_why = _transparency(nm, fam, m, {**TRANSPARENCY, **(st.get("transparency") or {})})
        d = drv[drv["model"] == nm] if len(drv) else drv
        judged = d[d["sign_ok"].isin(["yes", "no"])] if len(d) else d
        signs = float((judged["sign_ok"] == "yes").mean()) if len(judged) else np.nan
        nd = _n_drivers(m, r.get("drivers"))
        pars = 1.0 if nd <= st["max_drivers"] else st["max_drivers"] / nd
        cs = _codes(m, r.get("drivers"))
        econ = (1 - len(cs & outcome) / len(cs)) if cs else np.nan
        interp = tr * float(np.nanmean([signs, pars, econ]))
        a = allr[(allr["model"] == nm) & (allr["horizon"] == h_imp)] if len(allr) else allr
        if nd == 0 and (m is None or not getattr(m, "features", None)) and not len(a):
            cov = dirn = scen_ok = 0.0
        else:
            resp = a[a["balance_effect"].abs() > 1e-9] if len(a) else a
            cov = len(resp) / focus_n if focus_n else np.nan
            dirn = float((np.sign(resp["balance_effect"]) == np.sign(resp["expected_sign"])).mean()) \
                if len(resp) else (0.0 if focus_n else np.nan)
            g = gaps.get(nm)
            if g is not None and ref is not None:
                common = g.index.intersection(ref.index)
                scen_ok = float((np.sign(g[common]) == np.sign(ref[common])).mean()) if len(common) else np.nan
            else:
                scen_ok = np.nan
        sens = float(np.nanmean([cov, dirn, scen_ok])) if not all(np.isnan([cov, dirn, scen_ok])) else 0.0
        total = W["accuracy"] * acc + W["interpretation"] * interp + W["sensitivity"] * sens
        why_not = []
        if ratio > st["max_error_ratio"]:
            why_not.append(f"error {ratio:.2f}x the best (limit {st['max_error_ratio']:g}x)")
        mc = float(st.get("min_coverage", 0) or 0)
        if mc > 0 and focus_n and not (np.isfinite(cov) and cov >= mc - 1e-9):
            why_not.append(f"responds to {0 if not np.isfinite(cov) else cov:.0%} of the focus drivers "
                           f"(needs {mc:.0%}): cannot answer what-if questions about them")
        eligible = not why_not
        responds = not (mc > 0 and focus_n and not (np.isfinite(cov) and cov >= mc - 1e-9))
        rows.append({"model": nm, "family": fam, "error_pct": float(r[col]), "error_vs_best": ratio,
                     "accuracy": acc, "transparency": tr, "model_type": tr_why, "sign_share": signs, "drivers": nd,
                     "parsimony": pars, "economic_drivers": econ, "interpretation": interp, "coverage": cov, "direction": dirn,
                     "scenario_agreement": scen_ok, "sensitivity": sens, "score": total, "eligible": eligible,
                     "responds_to_economy": responds,
                     "not_eligible_because": "; ".join(why_not)})
    # order: eligible models by score; then those that at least respond to the economy (fallbacks, e.g. for the
    # formula when no eligible model has one); then the rest
    S = pd.DataFrame(rows).sort_values(["eligible", "responds_to_economy", "score"],
                                       ascending=[False, False, False]).reset_index(drop=True)
    S.insert(0, "rank", range(1, len(S) + 1))
    S.attrs.update(col=col, basis=basis, weights=W, horizon=h_imp)
    write(S, cfg, run_dir, basis, col, W, st, log)
    return S


def why_text(row, W, col):
    h = col.split("_")[2]
    return (f"best balance of forecasting, interpretation and sensitivity (score {row['score']:.2f}: accuracy "
            f"{row['accuracy']:.2f}, interpretation {row['interpretation']:.2f}, sensitivity {row['sensitivity']:.2f}; "
            f"weights {W['accuracy']:.0%}/{W['interpretation']:.0%}/{W['sensitivity']:.0%}); average error over 1-{h} "
            f"quarters {row['error_pct']:.2f}%, {row['error_vs_best']:.2f}x the most accurate model")


def write(S, cfg, run_dir, basis, col, W, st, log):
    out = os.path.join(run_dir, "recommendation")
    os.makedirs(out, exist_ok=True)
    S.round(4).to_csv(os.path.join(out, "1_scorecard.csv"), index=False)
    rec = S.iloc[0]
    most_acc = S.sort_values("error_pct").iloc[0]
    if not bool(rec["eligible"]):
        log("  scorecard: no model passed the gates (recommend.max_error_ratio, recommend.min_coverage); "
            "the highest score is used")
    j = {"model": rec["model"], "family": rec["family"], "score": float(rec["score"]),
         "accuracy": float(rec["accuracy"]), "interpretation": float(rec["interpretation"]),
         "sensitivity": float(rec["sensitivity"]), "error_pct": float(rec["error_pct"]),
         "most_accurate": most_acc["model"], "most_accurate_error_pct": float(most_acc["error_pct"]),
         "runner_up": S.iloc[1]["model"] if len(S) > 1 else None, "basis": basis, "error_column": col,
         "weights": W, "max_error_ratio": st["max_error_ratio"], "min_coverage": st.get("min_coverage"),
         "passed_gates": bool(rec["eligible"]), "why": why_text(rec, W, col)}
    json.dump(j, open(os.path.join(out, "recommended.json"), "w", encoding="utf-8"), indent=1)
    L = [f"# Recommended model: {rec['model']}", "", f"Chosen from all {len(S)} models ({basis}): {j['why']}.", "",
         f"Most accurate model: {most_acc['model']} ({most_acc['error_pct']:.2f}%).", "",
         "| rank | model | family | error % | accuracy | interpretation | sensitivity | score | eligible |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in S.head(15).iterrows():
        L.append(f"| {r['rank']} | {r['model']} | {r['family']} | {r['error_pct']:.2f} | {r['accuracy']:.2f} | "
                 f"{r['interpretation']:.2f} | {r['sensitivity']:.2f} | {r['score']:.2f} | "
                 f"{'yes' if r['eligible'] else 'no'} |")
    L += ["", "Accuracy = best error / model error. Interpretation = transparency of the model type x average of "
          "(share of coefficients with the expected sign, parsimony, share of drivers that are not other outcomes). "
          "Sensitivity = average of (share of focus drivers "
          "it responds to, share of those in the expected direction, share of scenarios moving the same way as the "
          "explanatory model). Weights and the eligibility limit: config `recommend`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    try:
        chart(S, os.path.join(out, "scorecard.png"), W)
    except Exception:
        pass
    log(f"  recommended ({basis}): {rec['model']} - score {rec['score']:.2f} (accuracy {rec['accuracy']:.2f}, "
        f"interpretation {rec['interpretation']:.2f}, sensitivity {rec['sensitivity']:.2f}); most accurate: "
        f"{most_acc['model']}")


def chart(S, path, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = S.head(12).iloc[::-1]
    f, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.4))
    left = np.zeros(len(d))
    for k, c in (("accuracy", "#1F6F8B"), ("interpretation", "#E3A72F"), ("sensitivity", "#2E9E83")):
        v = d[k].values * W[k]
        ax.barh(range(len(d)), v, left=left, color=c, label=f"{k} ({W[k]:.0%})")
        left += v
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([("✓ " if e else "✗ ") + m[:46] for m, e in zip(d["model"], d["eligible"])], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("weighted score (✗ = not eligible: too far behind the most accurate model, or no response "
                  "to the focus drivers)")
    ax.set_title("Model scorecard: forecasting, interpretation and sensitivity", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)
