"""The data story: how the data moves through the model, one step at a time, with this run's own numbers.

Written for a reader who knows econometrics but not this model. Each step says what goes in, the operation (as
an equation where there is one), the rule applied, what comes out (with a table of this run's data) and where
in the call tree it happens. One variable is followed through every step, and one forecast is taken apart
into its pieces.

The story is built from each stage's outputs (not from inside the functions), so it is complete even when a step
was loaded from the cache. It is added to the run trace (trace.html, "Data story" view) by
`trace.add_story(...)`; with the trace switched off nothing is built.
"""
import numpy as np
import pandas as pd

from ..features.transforms import TR_DESC, feature_matrix, parse_feature, transform
from ..utils import target as tg


def _rank(t, H):
    from .report import rank_column
    return rank_column(t, H)

FORMULA = {"dlog": "Δln x_t = ln x_t − ln x_{t−1}", "yoy": "ln x_t − ln x_{t−4}", "dlog2": "ln x_t − ln x_{t−2}",
           "acc": "(ln x_t − ln x_{t−4}) − (ln x_{t−4} − ln x_{t−8})", "gap": "ln x_t − mean(ln x_{t−11..t})",
           "rdlog": "Δln x_t − Δln P_t   (P = deflator)", "ryoy": "(ln x_t − ln x_{t−4}) − (ln P_t − ln P_{t−4})",
           "diff": "x_t − x_{t−1}", "diff2": "x_t − x_{t−2}", "diff4": "x_t − x_{t−4}", "lvl": "x_t",
           "rgap": "x_t − mean(x_{t−11..t})"}


# ------------------------------------------------------------------------------------------------- helpers
def fmt(v, d=4):
    """Short display of a value for a story table.

    Args:
        v: value. d: significant digits for floats.

    Returns:
        str, number or None.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return float(f"{v:.{d}g}")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Period):
        return str(v)
    return v if isinstance(v, (int, str)) else str(v)


def tbl(df, label, rows=10, note="", highlight=None):
    """A table for the story: first rows, values shortened.

    Args:
        df: DataFrame. label: caption. rows: rows kept. note: one line under it.
        highlight: column and value whose rows are marked (the followed variable), e.g. ("feature", "X|dlog|L1").

    Returns:
        dict (label, columns, rows, n, note, marked).
    """
    df = df.reset_index(drop=True)
    marked = []
    if highlight and highlight[0] in df.columns:
        marked = [i for i, v in enumerate(df[highlight[0]].head(rows)) if str(v).startswith(str(highlight[1]))]
    return {"label": label, "note": note, "columns": [str(c) for c in df.columns], "n": len(df), "marked": marked,
            "rows": [[fmt(v) for v in r] for r in df.head(rows).itertuples(index=False)]}


def step(stage, title, text, equation=None, rules=None, numbers=None, tables=None, follow=None, functions=None,
         files=None):
    """One step of the story.

    Args:
        stage: short stage label. title: the step's finding. text: list of short paragraphs.
        equation: lines of the equation(s). rules: the decision rules. numbers: list of (label, value).
        tables: list from tbl(). follow: what happens to the followed variable here.
        functions: functions where it happens (keys of the call tree). files: output files with the full tables.

    Returns:
        dict.
    """
    return {"stage": stage, "title": title, "text": list(text), "equation": equation or [], "rules": rules or [],
            "numbers": [[k, fmt(v)] for k, v in (numbers or [])], "tables": tables or [], "follow": follow,
            "functions": functions or [], "files": files or []}


def _pick_follow(cfg, results):
    """The feature followed through the story: config trace.follow, else the first driver chosen by the first approach.

    Args:
        cfg: configuration. results: approach results.

    Returns:
        feature name.
    """
    f = (cfg.get("trace") or {}).get("follow")
    if f and f != "auto":
        return f
    for r in results:
        if r["features"]:
            return r["features"][0]
    return None


# ------------------------------------------------------------------------------------------------- the run
def build(cfg, ctx, forms, F, results, models, notes, ev, comp):
    """The story of one pipeline run.

    Args:
        cfg: configuration. ctx: Context. forms: transforms and ADF results. F: candidate features.
        results: approach results. models: the model zoo. notes: model list. ev: evaluation per model.
        comp: model comparison table.

    Returns:
        dict with the followed variable and the list of steps.
    """
    fol = _pick_follow(cfg, results)
    fcode, ftr, flag = parse_feature(fol) if fol and fol.count("|") == 2 else (fol, None, None)
    fdesc = ctx.meta.loc[fcode, "description"] if fcode in ctx.meta.index else fcode
    S, smp, fe = [], cfg["sample"], cfg["features"]
    bank = cfg["data"]["bank"]
    hist_end = ctx.idx[ctx.end]
    est = ctx.idx[ctx.est0]

    # 1 ---------------------------------------------------------------- data in
    show = pd.DataFrame({"quarter": ctx.idx.astype(str), "target series": ctx.level.reindex(ctx.idx).values})
    if fcode in ctx.hist:
        show[f"{fcode} (raw)"] = ctx.hist[fcode].values
    S.append(step("1 · Data", f"{len(ctx.idx)} quarters of the target and {ctx.base.shape[1]} macro series go in",
                  [f"The target is `{bank['target_column']}`, {ctx.idx[0]} to {ctx.idx[-1]}. Each of the "
                   f"{len(ctx.macro)} scenario files holds the same {ctx.base.shape[1]} series: history, then "
                   f"the scenario path. The base scenario's history is used for estimation.",
                   f"The variable followed through this story is **{fol}**: {fdesc}."],
                  numbers=[("quarters of target data", len(ctx.idx)), ("macro series", ctx.base.shape[1]),
                           ("scenarios", len(ctx.macro)), ("metadata rows", len(ctx.meta)), ("last actual quarter", str(hist_end))],
                  tables=[tbl(show.iloc[-8:], "The last 8 quarters of history (balance in the file's units)", 8)],
                  follow=f"Raw values of {fcode} are read from the base scenario file.",
                  functions=["src.utils.io.Context.__init__", "src.utils.io.load_bank", "src.utils.io.load_macro",
                             "src.utils.io.load_metadata"]))

    # 2 ---------------------------------------------------------------- organic growth
    raw = tg.growth(ctx.level)
    events = (raw - ctx.y).abs() > 1e-12
    ev_q = list(ctx.idx[events.reindex(ctx.idx).fillna(False).values])
    rows = []
    for q in ev_q[:3] or [ctx.idx[1]]:
        p = ctx.idx.get_loc(q)
        for j in range(max(1, p - 1), min(len(ctx.idx), p + 2)):
            L, Lp = ctx.level.iloc[j], ctx.level.iloc[j - 1]
            A = L - tg.level_after(Lp, ctx.y.iloc[j])
            rows.append({"quarter": str(ctx.idx[j]), "L_t": L, "L_t−1": Lp, "A_t (event)": round(A, 1) if abs(A) > 1e-6 else 0.0,
                         "raw growth ln(L_t/L_t−1) (%)": raw.iloc[j] * 100, "organic y_t (%)": ctx.y.iloc[j] * 100})
    yv = ctx.y.iloc[ctx.est0:ctx.end + 1]
    S.append(step("2 · Target", "The target is organic quarterly log growth, with one-off events taken out",
                  ["Acquisitions and portfolio sales move the balance without any macro cause. They are listed in the "
                   "events file and removed, so the model explains only organic growth.",
                   f"One dummy per quarter in `dummy_quarters` ({', '.join(smp['dummy_quarters']) or 'none'}) absorbs "
                   "quarters no macro variable can explain; the same quarters are left out of every error measure."],
                  equation=["y_t = ln[(L_t − A_t) / L_{t−1}]", "A_t = size of the one-off event in quarter t (0 if none)",
                            "D_q = 1 in quarter q, 0 otherwise (one per dummy quarter)"],
                  numbers=[("events removed", len(ev_q)), ("mean organic growth, annualised", f"{tg.per_year(yv.mean()):.2f}" + ("%" if tg.is_log() else " units")),
                           ("s.d. of quarterly growth", f"{yv.std() * 100:.2f} pp"), ("estimation sample", f"{est} to {hist_end}")],
                  tables=[tbl(pd.DataFrame(rows), "Around each event: raw versus organic growth", 9,
                              "Organic growth removes the jump A_t, so the event quarter looks like a normal quarter.")],
                  functions=["src.utils.io.organic_growth"]))

    # 3 ---------------------------------------------------------------- relevance filter
    meta = ctx.meta
    have = [c for c in ctx.hist.columns if c in meta.index]
    geo_ok = [c for c in have if not fe["allowed_geographies"] or any(str(meta.loc[c, "geography"]).startswith(g) for g in fe["allowed_geographies"])]
    th_ok = [c for c in geo_ok if meta.loc[c, "theme"] not in (fe["excluded_themes"] or [])]
    sg_ok = [c for c in th_ok if not fe["require_sign_prior"] or meta.loc[c, "expected_sign"] != 0]
    funnel = pd.DataFrame([("macro series in the files", ctx.base.shape[1]), ("with a metadata row", len(have)),
                           (f"geography in {fe['allowed_geographies'] or 'any'}", len(geo_ok)),
                           (f"theme not in {fe['excluded_themes'] or '[]'}", len(th_ok)),
                           ("with an expected sign (+1 or −1)", len(sg_ok))], columns=["filter", "variables left"])
    by_theme = meta.loc[sg_ok].groupby("theme").size().sort_values(ascending=False).rename("variables").reset_index()
    S.append(step("3 · Relevance", f"{len(sg_ok)} of {ctx.base.shape[1]} series are economically relevant candidates",
                  ["Before any statistics, the metadata removes series that cannot be a sensible driver: other regions, "
                   "industry detail, and variables without an expected sign. The expected sign is the prior the "
                   "literature table and the metadata give (house prices +, unemployment −)."],
                  rules=[f"geography starts with one of {fe['allowed_geographies']}" if fe["allowed_geographies"] else "any geography",
                         f"theme not in {fe['excluded_themes']}", "expected_sign ≠ 0" if fe["require_sign_prior"] else "sign prior not required"],
                  tables=[tbl(funnel, "How many variables each filter leaves", 10), tbl(by_theme, "Candidates by theme", 12)],
                  follow=(f"{fcode} ({meta.loc[fcode, 'geography']}, theme {meta.loc[fcode, 'theme']}, expected sign "
                          f"{int(meta.loc[fcode, 'expected_sign']):+d}) passes." if fcode in sg_ok else None),
                  functions=["src.features.transforms.candidate_forms"]))

    # 4 ---------------------------------------------------------------- transforms
    order = list(FORMULA)
    trs = sorted(set(forms["transform"]), key=lambda t: (order.index(t) if t in order else len(order), t))
    from ..features.transforms import tr_desc
    tr_tab = pd.DataFrame([(t, FORMULA.get(t, tr_desc(t)), tr_desc(t), int((forms["transform"] == t).sum())) for t in trs],
                          columns=["transform", "formula", "meaning", "series tried"])
    tabs = [tbl(tr_tab, "The transforms: levels and indexes get the log forms, rates the difference forms", 14)]
    if fcode in ctx.hist:
        fr = forms[forms["mnemonic"] == fcode]
        W = pd.DataFrame({"quarter": ctx.idx.astype(str), "raw": ctx.hist[fcode].values})
        for t in fr["transform"]:
            W[t] = transform(ctx.hist, fcode, t, fe["deflator"]).values
        tabs.append(tbl(W.iloc[ctx.end - 7:ctx.end + 1], f"{fcode} in each of its forms, last 8 quarters", 8))
    S.append(step("4 · Transforms", "Each candidate is tried in several economically meaningful forms",
                  ["A level (an index, a balance) enters as growth: quarterly, yearly, two-quarter, its acceleration, or its "
                   "gap from a 3-year average. A rate enters as a change or a level. Series in the income, consumption and "
                   "credit themes are also tried in real terms (deflated by the CPI)."],
                  equation=[f"{r.transform}: {r.formula}" for r in tr_tab.itertuples()],
                  tables=tabs, follow=f"{fcode} is tried as: {', '.join(forms[forms['mnemonic'] == fcode]['transform'])}." if fcode in ctx.hist else None,
                  functions=["src.features.transforms.transform", "src.features.transforms.candidate_forms"]))

    # 5 ---------------------------------------------------------------- ADF
    st = forms.groupby("transform").agg(tested=("stationary", "size"), stationary=("stationary", "sum")).reset_index()
    fr = forms[forms["mnemonic"] == fcode][["transform", "adf_p", "stationary"]].rename(columns={"adf_p": "ADF p-value"})
    S.append(step("5 · Stationarity", f"{int(forms['stationary'].sum())} of {len(forms)} forms are stationary (ADF p < {fe['adf_p']})",
                  ["A regression of a stationary target on a non-stationary regressor can be spurious, so every form "
                   "must pass an augmented Dickey-Fuller test on the full history. Lag length is chosen by AIC."],
                  equation=["Δz_t = α + γ z_{t−1} + Σ_{j=1..p} φ_j Δz_{t−j} + ε_t", "H0: γ = 0 (unit root); keep the form if p < "
                            f"{fe['adf_p']}"],
                  rules=[f"ADF p-value < {fe['adf_p']}", "at least 30 observations and non-zero variance"],
                  tables=[tbl(st, "Forms tested and passing, by transform", 14), tbl(fr, f"{fcode}: test result of each form", 10)],
                  follow=f"Stationary forms of {fcode}: {', '.join(fr[fr['stationary']]['transform']) or 'none'}.",
                  functions=["src.features.transforms.candidate_forms"], files=["common/stationarity_tests.csv"]))

    # 6 ---------------------------------------------------------------- lags
    L = int(fe["max_lag"])
    tabs = []
    if fol in F.columns:
        base_f = f"{fcode}|{ftr}|L0"
        X = feature_matrix(ctx.hist, [base_f, fol], fe["deflator"]).reindex(ctx.idx)
        X.insert(0, "quarter", ctx.idx.astype(str))
        tabs.append(tbl(X.iloc[ctx.end - 5:ctx.end + 1], f"Lagging shifts the series down by {flag} quarter(s): {fol}", 6,
                        "Each row is the value the regression uses in that quarter."))
    S.append(step("6 · Lags", f"{F.shape[1]} candidate regressors: each stationary form at lags 0 to {L}",
                  ["Credit responds to the economy with a delay, so each stationary form enters at lags 0 to "
                   f"{L} quarters. A candidate's name is `mnemonic|transform|Llag`. Candidates with gaps in the "
                   f"estimation window ({est} to {hist_end}) or no variance are dropped."],
                  equation=["x^{(k)}_t = z_{t−k},  k = 0 … " + str(L)],
                  numbers=[("stationary forms", int(forms["stationary"].sum())), ("× lags", L + 1), ("candidates kept", F.shape[1])],
                  tables=tabs, follow=f"{fcode} as {ftr} at lag {flag} is the column {fol}." if fol in F.columns else None,
                  functions=["src.features.transforms.all_features"]))

    # 7-10 ------------------------------------------------------------ screen first
    r_sf = next((r for r in results if r["name"] == "Screen first"), None)
    if r_sf:
        sc, best = r_sf["tables"]["A1_screen_all"], r_sf["tables"]["A2_screen_best_per_variable"]
        s = cfg["screen_first"]
        top = best.sort_values("abs_t", ascending=False)
        fs = sc[sc["mnemonic"] == fcode].sort_values("abs_t", ascending=False)
        cols = ["feature", "description", "coef", "t_hac", "p_hac", "expected_sign", "sign_ok", "passed"]
        dm = None
        if fol in F.columns:
            dm = pd.DataFrame({"quarter": ctx.idx.astype(str), "y_t (%)": ctx.y.values * 100, "const": 1.0,
                               "y_t−1 (%)": ctx.y.shift(1).values * 100, f"x_t = {fol}": F[fol].values})
            for c in ctx.dummies.columns[:2]:
                dm[c] = ctx.dummies[c].values
            dm = dm.iloc[ctx.end - 5:ctx.end + 1]
        S.append(step("7 · Screen", f"{int(best['passed'].sum())} variables have a significant effect with the expected sign, one at a time",
                      ["Approach A, *screen first*, asks of every candidate separately whether it explains target growth "
                       "beyond momentum and the dummies. Standard errors are Newey-West (HAC, 4 lags) because quarterly "
                       "residuals are autocorrelated. Of all forms and lags of a variable, the one with the largest |t| "
                       "represents it."],
                      equation=["y_t = a + ρ y_{t−1} + Σ_q δ_q D_{q,t} + β x_t + e_t   (OLS, HAC s.e.)"],
                      rules=[f"p-value of β < {s['screen_p']}", "sign(β) = expected sign", "best form/lag per variable = largest |t|"],
                      numbers=[("regressions run", len(sc)), ("variables screened", best["mnemonic"].nunique()),
                               ("variables passing", int(best["passed"].sum()))],
                      tables=([tbl(dm, f"What one of these regressions sees: the last 6 rows of its data, for {fol}", 6,
                                   "One row per quarter of the estimation sample; growth in %. Only 2 of the dummy columns are "
                                   "shown: each is 1 in its own quarter and 0 elsewhere.")] if dm is not None else []) +
                             [tbl(top[cols], "Strongest variables (best form of each)", 10, highlight=("feature", fcode)),
                              tbl(fs[cols], f"Every form and lag of {fcode}, strongest first", 8)],
                      follow=(f"{fcode}'s best form is {fs.iloc[0]['feature']} with t = {fs.iloc[0]['t_hac']:.2f}: "
                              + ("passes." if fs.iloc[0]["passed"] else "fails.")) if len(fs) else None,
                      functions=["src.selection.screening.screen"],
                      files=["screen_first/A1_screen_all.csv", "screen_first/A2_screen_best_per_variable.csv"]))

        cl, sh = r_sf["tables"]["A3_clusters"], r_sf["tables"]["A4_shortlist"]
        tabs = []
        frow = cl[cl["mnemonic"] == fcode]
        if len(frow):
            k = frow.iloc[0]["cluster"]
            mem = cl[cl["cluster"] == k].copy()
            Fm = F[[c for c in mem["feature"] if c in F.columns]].iloc[ctx.est0:ctx.end + 1]
            if frow.iloc[0]["feature"] in Fm:
                mem["corr with followed"] = mem["feature"].map(Fm.corr()[frow.iloc[0]["feature"]])
            tabs.append(tbl(mem[["feature", "description", "geography", "abs_t", "corr with followed", "is_representative"]
                                if "corr with followed" in mem else ["feature", "description", "abs_t", "is_representative"]],
                            f"The cluster of {fcode}: near-duplicates, one representative", 12, highlight=("feature", fcode)))
        big = cl["cluster"].value_counts().index[0]
        mem = cl[cl["cluster"] == big]
        tabs.append(tbl(mem[["feature", "description", "geography", "abs_t", "is_representative"]].sort_values("abs_t", ascending=False),
                        f"The largest cluster ({len(mem)} near-duplicates): one representative goes on", 14))
        S.append(step("8 · Clusters", f"{len(cl)} passing variables collapse into {cl['cluster'].nunique()} distinct signals",
                      ["Many passing variables measure the same thing (house prices in 15 cities). Highly correlated "
                       "candidates are grouped, and one representative is kept per group, so the search compares "
                       "different economic channels rather than near-copies."],
                      equation=["distance d_ij = 1 − |corr(x_i, x_j)|  (estimation window)",
                                f"average-linkage clustering, cut at d = {1 - s['cluster_corr']:.2f}  (|corr| ≥ {s['cluster_corr']})"],
                      rules=["representative = largest |t| in the cluster",
                             f"unless a {s['primary_geography']} member reaches {s['rep_tolerance']:.0%} of that |t|: then the best "
                             f"{s['primary_geography']} member"],
                      tables=tabs + [tbl(sh[["feature", "description", "theme", "abs_t", "cluster_size"]],
                                         f"Shortlist: the {len(sh)} strongest representatives go to the search", 12,
                                         highlight=("feature", fcode))],
                      follow=((f"{fcode} has no close substitute: it is a cluster of its own, so it goes on." if int(frow.iloc[0]["cluster_size"]) == 1
                               else f"{fcode} is in a cluster of {int(frow.iloc[0]['cluster_size'])}; "
                               + ("it is the representative." if frow.iloc[0]["is_representative"] else "another member represents it."))
                              if len(frow) else None),
                      functions=["src.selection.clustering.cluster", "src.selection.clustering.representatives"],
                      files=["screen_first/A3_clusters.csv", "screen_first/A4_shortlist.csv"]))

        R, rules = r_sf["tables"]["A5_search"], r_sf["tables"].get("A6_search_rules")
        fm = cfg["final_model"]
        S.append(step("9 · Search", f"Every combination of {fm['min_vars']}–{fm['max_vars']} drivers is estimated; the chosen one is {', '.join(r_sf['features'])}",
                      ["All combinations of shortlisted drivers are estimated as ARDL(1) models. Combinations that break an "
                       "economic or statistical rule are dropped; the best by BIC are then ranked by how well they would "
                       "have forecast the balance in a rolling backtest. Among those within "
                       f"{fm['selection_tolerance']:.0%} of the best error, the one with the fewest drivers (then the "
                       "largest minimum |t|) wins."],
                      equation=["y_t = a + ρ y_{t−1} + Σ_q δ_q D_{q,t} + Σ_j β_j x_{j,t} + e_t",
                                "BIC = n ln(SSR/n) + k ln n",
                                "backtest error at origin o, horizon h:  e_{o,h} = exp(Σ_{s=1..h} ŷ_{o+s} − Σ_{s=1..h} y_{o+s}) − 1",
                                "rank by mean |e_{o,h}| over origins and h = 1 … " + str(smp["horizon"])],
                      rules=[f"every |t| ≥ {fm['t_min']}", "every sign as expected", f"VIF ≤ {fm['vif_max']}",
                             "at most one driver per theme" if fm["one_per_theme"] else "themes may repeat"],
                      tables=([tbl(rules, "What each rule removed", 12)] if rules is not None else []) +
                             [tbl(R[["features", "k", "min_abs_t", "max_vif", "bic", "oos_mape_pct", "selected"]],
                                  "Best combinations by backtest error (chosen one first)", 8, highlight=("features", fol))],
                      follow=(f"{fol} is in the chosen model." if fol in r_sf["features"] else None),
                      functions=["src.modelling.search.exhaustive_search", "src.modelling.search.fast_backtest"],
                      files=["screen_first/A5_search.csv"]))

    # 10 ---------------------------------------------------------------- EN first
    r_en = next((r for r in results if r["name"] == "Elastic Net first"), None)
    if r_en:
        e = cfg["en_first"]
        t = r_en["tables"]
        tabs = [tbl(t["B1_en_full_sample"], f"Full-sample Elastic Net: λ set so {e['target_n']} drivers are non-zero", 12)]
        if "B2_en_stability" in t:
            tabs.append(tbl(t["B2_en_stability"], f"Stability selection: share of {e['stability']['n_subsamples']} block subsamples "
                            "in which each feature was picked", 12, highlight=("feature", fcode)))
        if "B5_search_rules" in t:
            tabs.append(tbl(t["B5_search_rules"], "Final search on the stable picks: what each rule removed", 12))
        S.append(step("10 · Elastic Net", f"Approach B picks {', '.join(r_en['features'])} by penalised regression on all {F.shape[1]} candidates",
                      ["Approach B, *Elastic Net first*, lets a penalised regression choose among all candidates at once. "
                       "Momentum and dummies are partialled out first (Frisch-Waugh), so only macro drivers are shrunk. "
                       "Everything is standardised as glmnet does. With sign constraints, a driver can only enter with its "
                       "expected sign. Features chosen in at least "
                       f"{e['stability']['threshold']:.0%} of subsamples of consecutive-quarter blocks are kept, grouped by "
                       "correlation, and passed to the same final search as approach A."],
                      equation=["ỹ, X̃ = residuals of y and each x on (1, y_{t−1}, D)  (Frisch-Waugh)",
                                f"min_b (1/2n)‖ỹ − X̃b‖² + λ[α‖b‖₁ + (1−α)/2 ‖b‖²],  α = {e['l1_ratio']}",
                                "λ = the largest value at which target_n drivers are non-zero"],
                      rules=[f"target_n = {e['target_n']}", f"stability threshold {e['stability']['threshold']}",
                             f"groups at |corr| ≥ {e['group_corr']}", "then the same rules and backtest as step 9"],
                      tables=tabs, functions=["src.selection.elastic_net.elastic_net_select",
                                              "src.selection.elastic_net.stability_selection", "src.selection.elastic_net._prepare"],
                      files=["en_first/B1_en_full_sample.csv", "en_first/B2_en_stability.csv", "en_first/B4_search.csv"]))

    r_b = next((r for r in results if r["name"] == "Bayesian first"), None)
    if r_b:
        b = cfg["bayes_first"]
        t = r_b["tables"]
        tabs = [tbl(t["C2_bma_inclusion"][["feature", "inclusion_probability", "posterior_mean_std_coef", "description"]],
                    "Inclusion probability of each candidate: the posterior probability of the models that contain it",
                    12, highlight=("feature", fcode)),
                tbl(t["C3_bma_top_models"][["drivers", "prob", "cumulative_prob"]], "The most probable models", 8),
                tbl(t["C1_bma_settings"], "Prior and search", 3)]
        S.append(step("10b · Bayesian", f"Approach C picks {', '.join(r_b['features'])} by Bayesian model averaging",
                      ["Approach C, *Bayesian first*, weighs every combination of drivers by how well it explains the "
                       "target, against a prior that favours small models. One form and lag per variable is kept, "
                       "near-duplicates are grouped (so copies of a series do not split the probability), and models "
                       "with a wrong-signed driver get no weight. A driver's inclusion probability is the total "
                       f"probability of the models that contain it; those with at least {b['threshold']} form the median "
                       "probability model, grouped by correlation and passed to the same final search. The averaged "
                       "forecast of the most probable models is also a model: 'Bayesian model average'."],
                      equation=["p(M | y) ∝ p(y | M) p(M)",
                                "log p(y | M) = (n−1−k)/2 log(1+g) − (n−1)/2 log(1 + g(1 − R²_M)),  g = max(n, K²)",
                                f"p(M) beta-binomial on the number of drivers, prior mean {b['model_size']}",
                                "PIP_j = Σ_{M ∋ j} p(M | y)"],
                      rules=[f"at most {b['max_vars']} drivers per model", f"{b['max_candidates']} candidates",
                             f"near-duplicates grouped at |corr| ≥ {b.get('candidate_corr', 0.75)}",
                             f"MC3: {b['iterations']} draws after {b['burn_in']}", f"threshold {b['threshold']}"],
                      tables=tabs, functions=["src.selection.bayes.bma_select", "src.selection.bayes.mc3",
                                              "src.selection.bayes.prepare"],
                      files=["bayes_first/C2_bma_inclusion.csv", "bayes_first/C3_bma_top_models.csv",
                             "bayes_first/C5_search.csv"]))

    # 11-14 ------------------------------------------------------------ the chosen model
    m = next((x for x in models if x.name == "Screen first"), None) or next((x for x in models if x.name == "Elastic Net first"), None)
    if m is not None:
        S += _model_steps(cfg, ctx, m, ev, fol)

    # 15 ---------------------------------------------------------------- the zoo
    fam = notes.groupby("family", sort=False)["model"].apply(lambda s: "; ".join(s)).reset_index()
    fam.insert(1, "models", notes.groupby("family", sort=False).size().values)
    H = smp["horizon"]
    cc = comp.sort_values(f"avg_1_{H}_pct")[["model", "family"] + [c for c in comp.columns if c.startswith("avg_1_")]]
    S.append(step("14 · Competitors", f"{len(notes)} models in {notes['family'].nunique()} families face the same backtest",
                  ["The chosen drivers are one answer. Benchmarks, published specifications, hybrids of theory and data, "
                   "regime-switching, machine-learning, statistical and deep-learning models get the same data, the same "
                   "origins, the same error measure and the same scenarios. The out-of-sample test (run_oos.py) then "
                   "repeats the whole selection at each origin."],
                  tables=[tbl(fam, "Families and their models", 12), tbl(cc, "Backtest error, average over 1 … h quarters (%)", 12)],
                  functions=["src.models.zoo.build", "src.modelling.evaluate.evaluate"],
                  files=["comparison/9_model_list.csv", "comparison/1_model_comparison.csv"]))
    return {"follow": fol, "follow_description": fdesc, "steps": S}


def _model_steps(cfg, ctx, m, ev, fol):
    """Steps 11-14 for the chosen model: estimation, one backtest origin worked through, the forecast taken apart.

    Args:
        cfg: configuration. ctx: Context. m: the model. ev: evaluation per model. fol: followed feature.

    Returns:
        list of steps.
    """
    from ..modelling.search import backtest_origins, covid_mask, refit_flags
    out = []
    start = pd.Period(cfg["sample"]["estimation_start"], freq="Q")
    end = ctx.idx[ctx.end]
    P = m.panel(ctx, ctx.base)
    m.fit(P, start, end)
    r = m.res
    coef = pd.DataFrame({"term": r.params.index, "coef": r.params.values, "s.e. (HAC)": r.bse.values,
                         "t": r.tvalues.values, "p": r.pvalues.values})
    coef = coef[~coef["term"].str.startswith("D_")]
    rho = r.params.get("y_lag1", 0.0)
    lr = [(f, r.params[f] / (1 - rho)) for f in m.features]
    d = ev.get(m.name, {}).get("diagnostics", {}) or {}
    diag = pd.DataFrame([(k, d.get(k)) for k in ("adj_R2", "Ljung-Box p (4)", "Breusch-Pagan p", "Jarque-Bera p", "CUSUM p",
                                                 "max VIF", "wrong-sign drivers") if k in d], columns=["check", "value"])
    X = m._X(P).loc[:end]
    X = X[[c for c in X.columns if not c.startswith("D_")]].copy()
    X.insert(0, "y_t", P["y"].loc[:end])
    X = X.iloc[-6:]
    X.insert(0, "quarter", X.index.astype(str))
    out.append(step("11 · Estimate", f"{m.name}: {len(m.features)} drivers, adjusted R² {r.rsquared_adj:.2f}",
                    [f"The chosen drivers are re-estimated on the full sample, {start} to {end}, with Newey-West standard "
                     "errors. The long-run effect of a driver is its coefficient divided by (1 − ρ): a permanent one-unit "
                     "change in the driver changes quarterly growth by that much once momentum has played out."],
                    equation=["y_t = a + ρ y_{t−1} + Σ_q δ_q D_{q,t} + " + " + ".join(f"β{i + 1}·x{i + 1}_t" for i in range(len(m.features))) + " + e_t",
                              "long-run effect of x_j = β_j / (1 − ρ)"],
                    numbers=[("observations", int(r.nobs)), ("ρ (momentum)", rho)] + [(f"long-run effect of {f}", v) for f, v in lr],
                    tables=[tbl(X, "The regression's data: last 6 rows of y and X (dummy columns not shown)", 6),
                            tbl(coef, "Coefficients (COVID dummies not shown)", 10, highlight=("term", fol)),
                            tbl(diag, "Residual checks", 10)],
                    follow=f"{fol}: β = {r.params[fol]:.4f}, t = {r.tvalues[fol]:.2f}." if fol in r.params else None,
                    functions=["src.modelling.ardl.ARDL.fit", "src.modelling.evaluate.diagnostics"],
                    files=["comparison/3_drivers.csv", "comparison/4_diagnostics.csv"]))

    # one backtest origin, worked through
    H = cfg["sample"]["horizon"]
    origins = backtest_origins(ctx)
    flags = refit_flags(ctx, origins)
    M = covid_mask(ctx, origins, H)
    pick = next((i for i in range(len(origins) - 1, -1, -1) if origins[i] + H <= ctx.end and not M[i].any()), None)
    if pick is not None:
        o = origins[pick]
        j = max(k for k in range(pick + 1) if flags[k] or k == 0)
        fit_end = ctx.idx[origins[j]]
        m.fit(P, start, fit_end)
        Ph = P.loc[:end]
        fc = m.forecast(Ph, ctx.idx[o], H)
        act = ctx.y.values[o + 1:o + 1 + len(fc)]
        L0 = ctx.level.iloc[o]
        rows = [{"h": h + 1, "quarter": str(ctx.idx[o + h + 1]), "forecast ŷ (%)": fc[h] * 100, "actual y (%)": act[h] * 100,
                 "forecast balance": tg.level_after(L0, fc[:h + 1].sum()),
                 "actual organic balance": tg.level_after(L0, act[:h + 1].sum()),
                 "error %": float(tg.error(fc[:h + 1].sum(), act[:h + 1].sum())) * 100} for h in range(len(fc))]
        bh = ev.get(m.name, {}).get("by_horizon")
        out.append(step("12 · Backtest", f"At origin {ctx.idx[o]}, {m.name} is {abs(rows[-1]['error %']):.2f}% "
                        f"{'above' if rows[-1]['error %'] > 0 else 'below'} the actual balance {H} quarters later",
                        [f"The backtest pretends to stand at each origin from {cfg['sample']['backtest_start']}"
                         f"{' to ' + str(cfg['sample']['backtest_end']) if cfg['sample'].get('backtest_end') else ''}, "
                         "estimate with data up to then, and forecast 1 … "
                         f"{H} quarters ahead with the realised macro values. Coefficients are re-estimated every "
                         f"{cfg['sample']['reestimate_every']} quarter(s): this origin uses the coefficients estimated at "
                         f"{fit_end}. Lagged growth is replaced by the model's own forecast after the first quarter.",
                         "The error is measured on the balance, not on growth, because the balance is what is reported. "
                         "Origins whose window touches the dummy quarters are left out."],
                        equation=["ŷ_{o+h} = â + ρ̂ ŷ_{o+h−1} + Σ_j β̂_j x_{j,o+h}   (ŷ_o = y_o)",
                                  "balance_{o+h} = L_o · exp(Σ_{s≤h} ŷ_{o+s});   error_h = exp(Σ ŷ − Σ y) − 1"],
                        numbers=[("origins", len(origins)), ("re-estimations", int(flags.sum())),
                                 ("origins with a full window (no dummy quarter)", int((~M.any(axis=1)).sum()))],
                        tables=[tbl(pd.DataFrame(rows), f"Origin {ctx.idx[o]}: forecast against actual, quarter by quarter", H)] +
                               ([tbl(bh[["horizon_q", "MAPE_exCOVID_pct", "bias_exCOVID_pct", "n_origins"]],
                                     "Over all origins: mean absolute balance error by horizon (%)", H)] if bh is not None else []),
                        functions=["src.modelling.evaluate.backtest", "src.modelling.search.backtest_origins",
                                   "src.modelling.search.refit_flags", "src.modelling.ardl.ARDL.forecast"],
                        files=["comparison/2_backtest_by_horizon.csv"]))

    # the forecast taken apart
    m.fit(P, start, end)
    p = m.params
    pos = P.index.get_loc(end)
    yprev, L = P["y"].iloc[pos], ctx.level.iloc[ctx.end]
    rows = []
    for h in range(1, min(H, len(P) - pos - 1) + 1):
        x = P.iloc[pos + h]
        parts = {"constant a": p.get("const", 0.0), "ρ·ŷ_{t−1}": p.get("y_lag1", 0.0) * yprev}
        for f in m.features:
            parts[f"β·{f}"] = p[f] * x[f]
        yh = sum(parts.values())
        L = tg.level_after(L, yh)
        row = {"quarter": str(P.index[pos + h])}
        row.update({k: v * 100 for k, v in parts.items()})
        row.update({"ŷ = sum (%)": yh * 100, "annualised %": tg.per_year(yh), "balance": L})
        rows.append(row)
        yprev = yh
    sc = ev.get(m.name, {}).get("scenarios")
    tabs = [tbl(pd.DataFrame(rows), "Baseline scenario: each quarter's growth forecast (%) = the sum of its parts (percentage points)", H)]
    if sc is not None and len(sc):
        ye = sc[sc["date"].str.endswith("Q4")].pivot_table(index="date", columns="scenario", values="balance_forecast", sort=False)
        tabs.append(tbl(ye.round(1).reset_index().rename(columns={"date": "year end"}), "Year-end balance under each scenario", 8))
    out.append(step("13 · Forecast", f"The baseline forecast taken apart: what each driver contributes, quarter by quarter",
                    [f"From the last actual quarter, {end}, the model is iterated forward with the scenario's path of each "
                     "driver. Each column below is one term of the equation; their sum is the quarter's growth, and the "
                     "balance compounds it. The other scenarios differ only through the drivers' paths."],
                    equation=["ŷ_{T+h} = a + ρ ŷ_{T+h−1} + Σ_j β_j x^{scenario}_{j,T+h}", "L_{T+h} = L_{T+h−1} · exp(ŷ_{T+h})"],
                    tables=tabs, follow=f"The column β·{fol} is the followed variable's contribution." if fol in m.features else None,
                    functions=["src.modelling.evaluate.scenario_forecasts", "src.modelling.ardl.ARDL.forecast"],
                    files=["comparison/6_scenario_forecasts.csv", "comparison/5_scenario_yearend.csv"]))
    return out


# ------------------------------------------------------------------------------------------------- run_oos
def build_oos(cfg, picks, summ, dm, hold, H):
    """The story of the out-of-sample test (run_oos.py).

    Args:
        cfg: configuration. picks: drivers chosen at each origin. summ: summary per model. dm: Diebold-Mariano
        tests. hold: holdout forecasts. H: horizon.

    Returns:
        dict with the steps.
    """
    o = cfg["oos"]
    S = []
    S.append(step("1 · Re-selection", f"The whole selection is re-run at {picks['origin'].nunique()} origins with data up to each one only",
                  ["The rolling backtest in run_pipeline.py chose the drivers once, with all the data, so it flatters the "
                   "data-driven models. Here, at each origin, the stationarity tests, the screen, the clusters, the "
                   "Elastic Net and the search are all re-run on data that ends at the origin; every model is then "
                   f"re-estimated and forecasts 1 … {H} quarters ahead."],
                  rules=[f"origins from {o['first_origin']} every {o['step']} quarters", f"{o['window']} window"],
                  tables=[tbl(picks, "Drivers chosen at each origin", 30)],
                  functions=["src.validation.oos.run_oos", "src.validation.oos.select_at", "src.validation.oos.models_at"],
                  files=["oos/4_drivers_by_origin.csv", "oos/5_driver_stability.csv"]))
    cols = ["model", "family"] + [c for c in summ.columns if c.startswith("avg_1_")]
    S.append(step("2 · Accuracy", f"Out of sample, {summ.sort_values(_rank(summ, H)).iloc[0]['model']} has the smallest error",
                  ["Errors are measured on the balance as in the backtest. The *look-ahead check* rows use the drivers "
                   "chosen with all the data: the gap between them and the re-selected rows is the optimism of choosing "
                   "drivers with hindsight."],
                  equation=["error_{o,h} = exp(Σ_{s≤h} ŷ_{o+s} − Σ_{s≤h} y_{o+s}) − 1", "avg_1_h = mean of |error| over origins and horizons 1 … h"],
                  tables=[tbl(summ.sort_values(_rank(summ, H))[cols], "Average balance error over 1 … h quarters (%)", 30)],
                  functions=["src.validation.oos.summarise"], files=["oos/1_oos_summary.csv"]))
    S.append(step("3 · Significance", "Diebold-Mariano tests: is a model's advantage over AR(1) larger than chance?",
                  ["The loss differential between a model and the AR(1) benchmark is tested for a zero mean, with a HAC "
                   "variance (h − 1 lags) and the Harvey-Leybourne-Newbold small-sample correction. A negative statistic "
                   "means the model is better. With few origins, the test has little power."],
                  equation=["d_t = e²_{model,t} − e²_{AR(1),t}", "DM = d̄ / √(V̂(d̄)),  HLN: DM · √[(n + 1 − 2h + h(h−1)/n) / n] ~ t(n−1)"],
                  tables=[tbl(dm, "Tests at the comparison horizons", 30)], functions=["src.validation.oos.dm_test"],
                  files=["oos/3_dm_tests_vs_benchmark.csv"]))
    if hold is not None and len(hold):
        last = hold.groupby("model", sort=False).tail(1)[["model", "family", "drivers", "error_pct"]]
        S.append(step("4 · Holdout", f"Chosen and estimated without the last {o['holdout_quarters']} quarters, then compared with them",
                      ["The final check repeats everything with data that stops "
                       f"{o['holdout_quarters']} quarters before the end and forecasts those quarters."],
                      tables=[tbl(last.sort_values("error_pct", key=abs), "Balance error at the end of the holdout (%)", 30)],
                      functions=["src.validation.oos.holdout"], files=["oos/7_holdout.csv"]))
    return {"steps": S}
