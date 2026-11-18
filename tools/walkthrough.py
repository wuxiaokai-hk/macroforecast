"""
Step-by-step walkthrough of the whole pipeline, like stepping through it in a debugger.

    python tools/walkthrough.py                                  # small example (config/mini_config.yaml)
    python tools/walkthrough.py --config config/my_portfolio.yaml

It calls the same functions as run_pipeline.py, one at a time, in the same order. After each step it
  - saves that step's result to outputs/<run_name>/walkthrough/<NN>_<name>.csv (or .txt),
  - records which file and function did the work, the code, and what is now "in memory".
At the end it writes outputs/<run_name>/walkthrough/walkthrough.html: open it in a browser and click
through the steps. It also checks that the step-by-step results equal what the pipelines return.

Use a small config: every intermediate table goes into the page.
"""
import argparse
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.utils.env import ensure_packages  # noqa: E402

ensure_packages()

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.linear_model import ElasticNet  # noqa: E402

from src.features import transforms as T  # noqa: E402
from src.modelling import ardl as A, evaluate as EV, search as S  # noqa: E402
from src.pipelines import en_first, screen_first  # noqa: E402
from src.selection import clustering as CL, elastic_net as EN, screening as SC  # noqa: E402
from src.utils import config as CF, io as IO  # noqa: E402
from src.utils.cache import StepCache  # noqa: E402
from src.validation import oos as OOS  # noqa: E402

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- recording helpers
def _cell(v):
    """Make a value JSON-safe for the walkthrough page."""
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        return float(f"{v:.5g}")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    if isinstance(v, pd.Period):
        return str(v)
    return v if isinstance(v, (int, str)) or v is None else str(v)


def table(df, head=40, tail=0, index=False):
    """A DataFrame as JSON for the page: the first `head` rows (and last `tail`), with the total count."""
    d = df.reset_index() if index else df
    n = len(d)
    part = d if n <= head + tail else pd.concat([d.head(head), d.tail(tail)]) if tail else d.head(head)
    rows = [[_cell(v) for v in r] for r in part.itertuples(index=False)]
    gap = head if (tail and n > head + tail) else None
    return {"kind": "table", "columns": [str(c) for c in d.columns], "rows": rows, "n": n, "gap": gap}


def describe_obj(v):
    """Type and size of an object for the walkthrough tables."""
    if isinstance(v, pd.DataFrame):
        return "DataFrame", f"{v.shape[0]} × {v.shape[1]}"
    if isinstance(v, pd.Series):
        return "Series", f"{len(v)}"
    if isinstance(v, np.ndarray):
        return "ndarray", " × ".join(map(str, v.shape))
    if isinstance(v, dict):
        return "dict", f"{len(v)} keys"
    if isinstance(v, list):
        return "list", f"{len(v)} items"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return type(v).__name__, f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)
    if hasattr(v, "__dict__"):
        return type(v).__name__, ""
    return type(v).__name__, str(v)[:60]


class Recorder:
    """Collects the walkthrough steps (code, explanation, outputs) and writes the HTML page."""
    def __init__(self, folder):
        """Create a recorder writing into folder."""
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.steps, self.env, self.n = [], {}, 0

    def save(self, name, obj):
        """Save an intermediate object (CSV for tables, JSON otherwise)."""
        path = os.path.join(self.folder, name)
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            obj.to_csv(path, index=isinstance(obj, pd.Series) or not isinstance(obj.index, pd.RangeIndex))
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(obj))
        return os.path.relpath(path, HERE)

    def step(self, group, title, call, funcs, explain, outputs, new_vars, saved=(), pipeline_file=None,
             checks=None):
        """Record one walkthrough step with its code, explanation, outputs and checks."""
        self.n += 1
        code = []
        for fn in funcs:
            src = inspect.getsource(fn)
            file = os.path.relpath(inspect.getsourcefile(fn), HERE)
            line = inspect.getsourcelines(fn)[1]
            code.append({"name": fn.__qualname__, "file": file, "line": line, "source": src})
        changed = set(new_vars)
        self.env.update(new_vars)
        env = [{"name": k, "type": describe_obj(v)[0], "value": describe_obj(v)[1], "new": k in changed}
               for k, v in self.env.items()]
        self.steps.append({"n": self.n, "group": group, "title": title, "call": call, "code": code,
                           "explain": explain, "outputs": outputs, "env": env,
                           "saved": [self.save(f"{self.n:02d}_{nm}", ob) for nm, ob in saved],
                           "pipeline_file": pipeline_file, "checks": checks or []})
        print(f"  step {self.n:2d}  {title}")


def chart(title, series, x_label="", y_label="", kind="line", x_type="cat"):
    """A chart block for the walkthrough page."""
    return {"kind": "chart", "title": title, "series": series, "x_label": x_label, "y_label": y_label,
            "type": kind, "x_type": x_type}


def text(title, body):
    """A text block for the walkthrough page."""
    return {"kind": "text", "title": title, "body": body}


def note(body):
    """A note block for the walkthrough page."""
    return {"kind": "note", "body": body}


def kv(title, pairs):
    """A key-value table block for the walkthrough page."""
    return {"kind": "kv", "title": title, "pairs": [[k, _cell(v)] for k, v in pairs]}


# --------------------------------------------------------------------------- the walkthrough
def main():
    """Run the step-by-step walkthrough on the mini example data and write walkthrough.html."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config", "mini_config.yaml"))
    ap.add_argument("--skip-oos", action="store_true", help="leave out the out-of-sample step")
    args = ap.parse_args()

    cfg = CF.load_config(args.config)
    W = Recorder(os.path.join(cfg["out_dir"], "walkthrough"))
    run = cfg["run_name"]
    print(f"Walkthrough of '{run}' -> {W.folder}")

    # 1 ----------------------------------------------------------------- config
    flat = []
    for sec in ("sample", "features", "screen_first", "en_first", "final_model", "oos"):
        for k, v in cfg[sec].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat.append({"section": sec, "setting": f"{k}.{k2}", "value": _cell(v2)})
            else:
                flat.append({"section": sec, "setting": k, "value": _cell(v)})
    flat = pd.DataFrame(flat)
    W.step("Shared start", "Read the config", "cfg = load_config('" + os.path.relpath(args.config, HERE) + "')",
           [CF.load_config],
           "The YAML file is merged with the defaults in src/utils/config.py, file paths are made absolute and "
           "the required names are checked. Every later step reads its settings from <code>cfg</code>; "
           "nothing about file or column names is written in the code.",
           [table(flat, head=80)], {"cfg": cfg}, saved=[("config_settings.csv", flat)])

    # 2 ----------------------------------------------------------------- bank data
    level = IO.load_bank(cfg)
    lv = level.rename("loan_balance").to_frame()
    lv.index = lv.index.astype(str)
    W.step("Shared start", "Load the target", "level = load_bank(cfg)", [IO.load_bank, IO.to_quarters],
           f"Reads <code>{os.path.relpath(cfg['data']['bank']['file'], HERE)}</code>, takes the column named in "
           f"<code>data.bank.target_column</code> ({cfg['data']['bank']['target_column']}), turns the dates into "
           f"quarters and cuts at <code>sample.hist_end</code>. Result: {len(level)} quarters, "
           f"{level.index[0]}–{level.index[-1]}.",
           [table(lv, head=8, tail=6, index=True),
            chart("Loan balance", [{"name": "loan_balance", "x": list(lv.index), "y": [float(v) for v in level]}])],
           {"level": level}, saved=[("loan_balance.csv", lv)])

    # 3 ----------------------------------------------------------------- organic growth
    y = IO.organic_growth(cfg, level)
    raw = np.log(level).diff()
    ev = IO.read_table(cfg["data"]["events"]["file"]) if cfg["data"]["events"]["file"] else pd.DataFrame()
    ev = ev[ev[cfg["data"]["events"]["series_column"]] == cfg["data"]["bank"]["target_column"]] if len(ev) else ev
    g = pd.DataFrame({"balance": level, "raw_dlog": raw, "organic_dlog": y})
    g["removed_event"] = 0.0
    for q, a in zip(IO.to_quarters(ev[cfg["data"]["events"]["date_column"]]), ev[cfg["data"]["events"]["amount_column"]]) if len(ev) else []:
        if q in g.index:
            g.loc[q, "removed_event"] = a
    g.index = g.index.astype(str)
    ev_q = [str(q) for q in IO.to_quarters(ev[cfg["data"]["events"]["date_column"]])] if len(ev) else []
    around = g.loc[[q for q in g.index if any(abs(pd.Period(q, "Q").ordinal - pd.Period(e, "Q").ordinal) <= 3
                                               for e in ev_q)]] if ev_q else g.head(8)
    W.step("Shared start", "Remove inorganic events → target y", "y = organic_growth(cfg, level)",
           [IO.organic_growth],
           "The target is the quarterly log change of the balance with acquisitions and other one-off jumps "
           "taken out: y<sub>t</sub> = ln((L<sub>t</sub> − A<sub>t</sub>) / L<sub>t−1</sub>), where A<sub>t</sub> "
           "is the event amount from the events file. Only the quarter of the event changes. "
           + (f"Here the event in {', '.join(ev_q)} is removed." if ev_q else "No events file was given."),
           [table(ev.reset_index(drop=True)) if len(ev) else note("No events."),
            table(around.reset_index().rename(columns={"index": "date"}), head=20),
            chart("Quarterly growth, raw vs organic (%)",
                  [{"name": "raw", "x": list(g.index[1:]), "y": [float(v) * 100 for v in g["raw_dlog"].iloc[1:]]},
                   {"name": "organic (target y)", "x": list(g.index[1:]),
                    "y": [float(v) * 100 for v in g["organic_dlog"].iloc[1:]]}])],
           {"y": y}, saved=[("target_growth.csv", g)])

    # 4 ----------------------------------------------------------------- macro + metadata
    macro = IO.load_macro(cfg)
    meta = IO.load_metadata(cfg)
    base = macro[cfg["data"]["macro"]["base_scenario"]]
    W.step("Shared start", "Load the macro scenarios and the metadata",
           "macro = load_macro(cfg); meta = load_metadata(cfg)", [IO.load_macro, IO.load_metadata],
           f"One file per scenario ({', '.join(macro)}); each holds history plus that scenario's path to "
           f"{base.index[-1]}. The metadata says, for each mnemonic, its geography, theme, whether it is a level "
           "(<code>dlog</code>) or a rate (<code>diff</code>), and the expected sign of its effect on loans. "
           "Your own column names are mapped through <code>data.metadata.columns</code>.",
           [table(meta, index=True),
            table(base.round(3).assign(date=base.index.astype(str)).set_index("date"), head=6, tail=4, index=True)],
           {"macro": macro, "meta": meta},
           saved=[("metadata.csv", meta), ("macro_baseline.csv", base)])

    # 5 ----------------------------------------------------------------- constructed
    macro, meta = IO.add_constructed(cfg, macro, meta)
    base = macro[cfg["data"]["macro"]["base_scenario"]]
    names = [c["name"] for c in cfg["data"]["constructed"]]
    if names:
        cols = sorted({c for c in base.columns if any(f"`{c}`" in k["expression"] for k in cfg["data"]["constructed"])}) + names
        hx = pd.Period(cfg["sample"]["hist_end"], freq="Q")
        cv = base.loc[hx - 7:hx, cols].round(3)
        cv.index = cv.index.astype(str)
        outs = [table(pd.DataFrame(cfg["data"]["constructed"])[["name", "expression", "theme", "expected_sign"]]),
                table(cv, index=True)]
    else:
        cv, outs = pd.DataFrame(), [note("No constructed variables in this config.")]
    W.step("Shared start", "Build constructed variables", "macro, meta = add_constructed(cfg, macro, meta)",
           [IO.add_constructed],
           "Spreads or real rates listed under <code>data.constructed</code> are computed in every scenario file "
           "with a pandas expression and added to the metadata, so from here on they are ordinary candidates.",
           outs, {"macro": macro, "meta": meta}, saved=[("constructed.csv", cv)])

    # 6 ----------------------------------------------------------------- context
    ctx = IO.Context(cfg)
    dm = ctx.dummies.copy()
    dm.index = dm.index.astype(str)
    dshow = dm.loc[str(pd.Period(cfg["sample"]["dummy_quarters"][0], "Q") - 2):
                   str(pd.Period(cfg["sample"]["dummy_quarters"][-1], "Q") + 2)] if len(dm.columns) else dm.head()
    W.step("Shared start", "Put it together: the Context object", "ctx = Context(cfg)", [IO.Context.__init__],
           "<code>Context</code> runs steps 2–5 in one go and adds the sample positions and the dummies. "
           "<code>ctx.hist</code> is the base-scenario macro data on the history quarters only, so nothing "
           "from a scenario path leaks into estimation. One dummy per quarter in "
           "<code>sample.dummy_quarters</code>.",
           [kv("Sample", [("first quarter of y", ctx.idx[0]), ("estimation start (ctx.est0)", f"{ctx.idx[ctx.est0]}  (position {ctx.est0})"),
                          ("estimation end (ctx.end)", f"{ctx.idx[ctx.end]}  (position {ctx.end})"),
                          ("observations used", ctx.end - ctx.est0 + 1),
                          ("macro columns", ctx.base.shape[1]), ("scenarios", ", ".join(ctx.macro))]),
            table(dshow, index=True, head=20)],
           {"ctx": ctx, "ctx.hist": ctx.hist, "ctx.dummies": ctx.dummies}, saved=[("dummies.csv", dm)])

    # 7 ----------------------------------------------------------------- filter + transforms + ADF
    f = cfg["features"]
    why = []
    for code in ctx.hist.columns:
        if code not in ctx.meta.index:
            why.append((code, "no metadata row"))
            continue
        geo, th = ctx.meta.loc[code, "geography"], ctx.meta.loc[code, "theme"]
        if f["allowed_geographies"] and not any(geo.startswith(g_) for g_ in f["allowed_geographies"]):
            why.append((code, f"geography '{geo}' not in allowed_geographies"))
        elif th in (f["excluded_themes"] or []):
            why.append((code, f"theme '{th}' is excluded"))
        elif f["require_sign_prior"] and ctx.meta.loc[code, "expected_sign"] == 0:
            why.append((code, "expected_sign is 0 (no prior) and require_sign_prior is on"))
    dropped = pd.DataFrame(why, columns=["mnemonic", "dropped because"])
    forms = T.candidate_forms(cfg, ctx.hist, ctx.meta)
    if f["deflator"] in dropped["mnemonic"].values:
        dropped.loc[dropped["mnemonic"] == f["deflator"], "dropped because"] += " (still used as the deflator for real transforms)"
    W.step("Shared start", "Relevance filter, transformations, ADF test", "forms = candidate_forms(cfg, ctx.hist, ctx.meta)",
           [T.candidate_forms, T.transform],
           f"Variables outside <code>allowed_geographies</code>, in an excluded theme, or with no sign prior are dropped. "
           f"Every remaining variable is transformed in every allowed way (levels: {', '.join(f['transforms_level'])}"
           f"{', plus real growth for ' + ', '.join(f['real_themes']) if f['real_themes'] else ''}; rates: "
           f"{', '.join(f['transforms_rate'])}) and each transformed series gets an ADF unit-root test on history. "
           f"Only forms with p &lt; {f['adf_p']} (stationary) go on. Result: {forms['mnemonic'].nunique()} variables, "
           f"{len(forms)} forms, {int(forms['stationary'].sum())} stationary.",
           [table(dropped) if len(dropped) else note("Nothing dropped by the filter."), table(forms, head=80)],
           {"forms": forms}, saved=[("stationarity_tests.csv", forms), ("filter_dropped.csv", dropped)],
           pipeline_file=f"outputs/{run}/common/stationarity_tests.csv")

    # 8 ----------------------------------------------------------------- example transform
    ex = next((c for c in forms["mnemonic"] if forms[forms["mnemonic"] == c]["transform"].isin(["dlog"]).any()),
              forms["mnemonic"].iloc[0])
    trs = list(forms[forms["mnemonic"] == ex]["transform"])
    tx = pd.DataFrame({"level": ctx.hist[ex]})
    for tr in trs:
        tx[tr] = T.transform(ctx.hist, ex, tr, f["deflator"])
    tx.index = tx.index.astype(str)
    adf = forms[forms["mnemonic"] == ex][["transform", "adf_p", "stationary"]]
    W.step("Shared start", f"Close-up: the transformations of {ex}",
           f"T.transform(ctx.hist, '{ex}', tr) for tr in {trs}", [T.transform],
           f"What step 7 did for one variable, {ctx.meta.loc[ex, 'description']}. "
           "dlog = ln(x<sub>t</sub>) − ln(x<sub>t−1</sub>); yoy = ln(x<sub>t</sub>) − ln(x<sub>t−4</sub>); "
           "gap = ln(x<sub>t</sub>) minus its trailing 12-quarter mean; rdlog / ryoy subtract the same change in "
           "the deflator. All windows look backwards only.",
           [table(adf), table(tx.round(4), head=6, tail=6, index=True),
            chart(f"{ex}: transformed series", [{"name": tr, "x": list(tx.index), "y": [None if pd.isna(v) else float(v) for v in tx[tr]]}
                                               for tr in trs if tr != "level"])],
           {f"transforms of {ex}": tx}, saved=[(f"transforms_{ex}.csv", tx)])

    # 9 ----------------------------------------------------------------- lags -> features
    F = T.all_features(cfg, ctx, forms)
    fl = pd.DataFrame({"feature": F.columns})
    fl[["mnemonic", "transform", "lag"]] = [T.parse_feature(c) for c in F.columns]
    fl["description"] = [T.describe(c, ctx.meta) for c in F.columns]
    Fw = F.iloc[ctx.est0:ctx.end + 1]
    Fshow = Fw.iloc[:, :6].round(4)
    Fshow.index = Fshow.index.astype(str)
    W.step("Shared start", "Add lags: the candidate feature matrix F", "F = all_features(cfg, ctx, forms)",
           [T.all_features, T.feature_series],
           f"Each stationary form is lagged 0–{f['max_lag']} quarters. A feature is named "
           "<code>MNEMONIC|transform|L&lt;lag&gt;</code>. Features with a gap or no variation inside the estimation "
           f"window are dropped. Result: F has {F.shape[1]} columns ({len(forms[forms['stationary']])} stationary forms × "
           f"{f['max_lag'] + 1} lags, minus any dropped).",
           [table(fl, head=60), table(Fshow, head=5, tail=3, index=True)],
           {"F": F}, saved=[("feature_list.csv", fl), ("feature_matrix.csv", F)])

    # ================================================================== Approach A
    s = cfg["screen_first"]
    sc, best = SC.screen(cfg, ctx, F)
    top = sc.iloc[0]["feature"]
    rows_ = slice(ctx.est0, ctx.end + 1)
    C = SC.controls(ctx).iloc[rows_]
    C = C.loc[:, (C.std() > 0) | (C.columns == "const")]
    res = sm.OLS(ctx.y.iloc[rows_], C.assign(**{top: F[top].iloc[rows_]})).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    W.step("A · Screen first", "Univariate screen", "sc, best = screen(cfg, ctx, F)",
           [SC.screen, SC.controls],
           "Each feature is added on its own to the base regression "
           "y<sub>t</sub> = a + ρ·y<sub>t−1</sub> + dummies + β·z<sub>t</sub>, with Newey-West (HAC, 4 lags) "
           f"standard errors. It passes if p &lt; {s['screen_p']} and the sign of β matches the expected sign. "
           f"{len(sc)} regressions; {int(sc['passed'].sum())} features pass. The text box shows the full "
           f"regression for the strongest feature, <code>{top}</code>.",
           [table(sc[["feature", "description", "coef", "t_hac", "p_hac", "expected_sign", "sign_ok", "passed"]], head=40),
            text(f"Regression for {top} (statsmodels summary)", res.summary().as_text())],
           {"sc": sc}, saved=[("A1_screen_all.csv", sc), ("A1_top_regression.txt", res.summary().as_text())],
           pipeline_file=f"outputs/{run}/screen_first/A1_screen_all.csv")

    passed = best[best["passed"]].copy()
    W.step("A · Screen first", "Keep the best form and lag per variable", "best = (inside screen) … groupby('mnemonic').head(1)",
           [SC.screen],
           "Several forms and lags of the same variable usually pass. Only the one with the largest |t| is kept per "
           f"variable, so one variable cannot fill the shortlist. {len(passed)} of {len(best)} variables have a passing form.",
           [table(best[["mnemonic", "feature", "description", "t_hac", "p_hac", "passed"]])],
           {"best": best, "passed": passed}, saved=[("A2_screen_best_per_variable.csv", best)],
           pipeline_file=f"outputs/{run}/screen_first/A2_screen_best_per_variable.csv")

    Fp = F[passed["feature"]].iloc[ctx.est0:ctx.end + 1]
    passed["cluster"] = CL.cluster(Fp, s["cluster_corr"]).reindex(passed["feature"]).values
    clus = CL.representatives(passed.set_index("feature", drop=False), "abs_t", s["primary_geography"], s["rep_tolerance"])
    corr = Fp.corr().round(2)
    corr.index = [c.split("|")[0] for c in corr.index]
    corr.columns = [c.split("|")[0] for c in corr.columns]
    W.step("A · Screen first", "Cluster near-duplicates, keep one per cluster",
           "passed['cluster'] = cluster(Fp, 0.75); clus = representatives(passed, 'abs_t', 'Ontario', 0.75)",
           [CL.cluster, CL.representatives],
           f"Distance between two features = 1 − |correlation|; average-linkage clustering cut at |ρ| = {s['cluster_corr']}. "
           f"In each cluster the feature with the highest |t| is kept, unless an {s['primary_geography']} series has at least "
           f"{s['rep_tolerance']:.0%} of that |t|, in which case the {s['primary_geography']} one is kept. "
           f"{len(passed)} variables → {clus['cluster'].nunique()} clusters.",
           [table(corr, index=True), table(clus[["feature", "description", "geography", "abs_t", "cluster", "cluster_size", "is_representative"]]
                                           .sort_values(["cluster", "abs_t"], ascending=[True, False]))],
           {"clus": clus}, saved=[("A3_clusters.csv", clus.reset_index(drop=True)), ("A3_correlations.csv", corr)],
           pipeline_file=f"outputs/{run}/screen_first/A3_clusters.csv")

    shortlist = clus[clus["is_representative"]].sort_values("abs_t", ascending=False).head(s["shortlist_max"])
    W.step("A · Screen first", "Shortlist", "shortlist = clus[clus.is_representative].head(shortlist_max)",
           [screen_first.run],
           f"The cluster representatives, strongest first, up to <code>shortlist_max</code> = {s['shortlist_max']}. "
           f"{len(shortlist)} candidates go into the final search.",
           [table(shortlist[["feature", "description", "theme", "abs_t", "expected_sign"]].reset_index(drop=True))],
           {"shortlist": shortlist}, saved=[("A4_shortlist.csv", shortlist.reset_index(drop=True))],
           pipeline_file=f"outputs/{run}/screen_first/A4_shortlist.csv")

    def rule_funnel(Fc, info):
        fm = cfg["final_model"]
        feats = list(Fc.columns)
        themes, signs = info.loc[feats, "theme"].values, info.loc[feats, "expected_sign"].values
        rows = np.arange(ctx.est0, ctx.end + 1)
        yv, D = ctx.y.values, ctx.dummies.values
        Dr = D[rows][:, D[rows].std(axis=0) > 0]
        base_ = np.column_stack([np.ones(len(rows)), yv[rows - 1], Dr])
        cnt = {"combinations": 0, "rejected: two drivers from one theme": 0, "rejected: a driver with |t| < t_min": 0,
               "rejected: a wrong sign": 0, "rejected: VIF above vif_max": 0, "passed all rules": 0}
        import itertools
        for k in range(fm["min_vars"], fm["max_vars"] + 1):
            for combo in itertools.combinations(range(len(feats)), k):
                cnt["combinations"] += 1
                if fm["one_per_theme"] and len(set(themes[list(combo)])) < k:
                    cnt["rejected: two drivers from one theme"] += 1
                    continue
                Fk = Fc.values[rows][:, combo]
                b, se, _ = A.ols_np(np.column_stack([base_, Fk]), yv[rows])
                if np.any(np.abs(b[-k:] / se[-k:]) < fm["t_min"]):
                    cnt["rejected: a driver with |t| < t_min"] += 1
                    continue
                sg = signs[list(combo)]
                if np.any((sg != 0) & (np.sign(b[-k:]) != sg)):
                    cnt["rejected: a wrong sign"] += 1
                    continue
                if np.max(A.vif(Fk)) > fm["vif_max"]:
                    cnt["rejected: VIF above vif_max"] += 1
                    continue
                cnt["passed all rules"] += 1
        return list(cnt.items())

    fm = cfg["final_model"]
    search_explain = (f"Every combination of {fm['min_vars']}–{fm['max_vars']} candidates is fitted as an ARDL "
                      "(lagged y, dummies, the drivers). A combination is kept only if every driver has "
                      f"|t| ≥ {fm['t_min']}, the expected sign, VIF ≤ {fm['vif_max']}"
                      f"{', and no two drivers share a theme' if fm['one_per_theme'] else ''}. The survivors are "
                      f"backtested from {cfg['sample']['backtest_start']} (1–{cfg['sample']['horizon']} quarters ahead, "
                      "dummy quarters left out of the error) and ranked by balance MAPE. The simplest model within "
                      f"{fm['selection_tolerance']:.0%} of the best MAPE is chosen (<code>selected = True</code>, first row).")
    info_a = shortlist.reset_index(drop=True).set_index("feature")
    RA, chosen_a = S.exhaustive_search(cfg, ctx, F[shortlist["feature"]], info_a, "screen_first")
    W.step("A · Screen first", "Final search → Screen-first drivers",
           "RA, chosen_a = exhaustive_search(cfg, ctx, F[shortlist], info, 'screen_first')",
           [S.exhaustive_search, S.fast_backtest],
           search_explain + f" <b>Chosen: {', '.join(chosen_a)}</b>.",
           [kv("How many combinations each rule removed", rule_funnel(F[shortlist["feature"]], info_a)),
            table(RA, head=25)],
           {"RA": RA, "chosen_a": chosen_a}, saved=[("A5_search.csv", RA)],
           pipeline_file=f"outputs/{run}/screen_first/A5_search.csv")

    # ================================================================== Approach B
    e = cfg["en_first"]
    rows = np.arange(ctx.est0, ctx.end + 1)
    X, yy, nm, sg = EN._prepare(cfg, ctx, F, rows)
    y_raw = ctx.y.values[rows]
    prep = pd.DataFrame({"column": nm, "expected_sign": [int(v) for v in sg], "mean": X.mean(0), "sd": X.std(0)})
    prep["flipped (× −1)"] = prep["expected_sign"] < 0
    W.step("B · Elastic Net first", "Prepare X and y the glmnet way", "X, y, names, signs = _prepare(cfg, ctx, F, rows)",
           [EN._prepare, SC.controls],
           "Three things happen. (1) Lagged y and the dummies are regressed out of y and of every feature, which is "
           "the same as glmnet's <code>penalty.factor = 0</code> for those columns: they are never shrunk. "
           "(2) X and y are standardised with the population sd, as glmnet does. (3) Features expected to lower loans "
           "are multiplied by −1 so a single <code>positive=True</code> constraint enforces every expected sign "
           "(glmnet <code>lower.limits</code>). Check: after partialling out, y is uncorrelated with lagged y.",
           [kv("Shapes and checks", [("rows (quarters)", X.shape[0]), ("columns", X.shape[1]),
                                    ("mean of y after", round(float(yy.mean()), 6)), ("sd of y after", round(float(yy.std()), 6)),
                                    ("corr(y before, lagged y)", round(float(np.corrcoef(y_raw, ctx.y.values[rows - 1])[0, 1]), 3)),
                                    ("corr(y after, lagged y)", round(float(np.corrcoef(yy, ctx.y.values[rows - 1])[0, 1]), 6))]),
            table(prep, head=60)],
           {"X": X, "y_std": yy}, saved=[("B0_prepared_columns.csv", prep)])

    n = len(yy)
    lam_max = np.max(np.abs(X.T @ yy)) / (n * max(e["l1_ratio"], 1e-3))
    grid = lam_max * np.logspace(0, -3, 150)
    m = ElasticNet(alpha=grid[0], l1_ratio=e["l1_ratio"], positive=e["sign_constraints"], fit_intercept=False,
                   warm_start=True, max_iter=20000, tol=1e-7)
    path, entered = [], {}
    for lam in grid:
        m.set_params(alpha=lam)
        m.fit(X, yy)
        nzn = sorted(set(np.array(nm)[m.coef_ != 0]))
        for c in nzn:
            entered.setdefault(c, lam)
        path.append((lam, len(nzn)))
        if len(nzn) >= e["target_n"]:
            break
    full = EN.elastic_net_select(cfg, ctx, F)
    full["entered_at_lambda"] = full["feature"].map(entered)
    W.step("B · Elastic Net first", "Elastic Net: lower λ until target_n features are in",
           "full = elastic_net_select(cfg, ctx, F)", [EN.elastic_net_select, EN._select_at_target],
           f"λ starts at λ<sub>max</sub> = {lam_max:.4f}, where nothing is selected, and is lowered on a log grid "
           f"(warm starts). It stops at the first λ where {e['target_n']} features are non-zero "
           f"(<code>target_n</code>); here λ = {full['lambda'].iloc[0]:.4f} after {len(path)} grid points. "
           "No cross-validation, so for the same data the answer is always the same. "
           f"α (glmnet) = l1_ratio (sklearn) = {e['l1_ratio']}.",
           [chart("Features selected as λ falls", [{"name": "selected", "x": [float(np.log10(a)) for a, _ in path],
                                                   "y": [b for _, b in path]}], x_label="log10 λ (falls to the right)",
                  y_label="features", x_type="num", kind="step"),
            table(full)],
           {"full": full}, saved=[("B1_en_full_sample.csv", full),
                                  ("B1_lambda_path.csv", pd.DataFrame(path, columns=["lambda", "n_selected"]))],
           pipeline_file=f"outputs/{run}/en_first/B1_en_full_sample.csv")

    st = e["stability"]
    freq = EN.stability_selection(cfg, ctx, F)
    sel = freq[freq["selection_frequency"] >= st["threshold"]].copy()
    if len(sel) < 2:
        sel = freq.head(max(3, e["target_n"] // 2)).copy()
    sel["score"] = sel["selection_frequency"]
    nb = int(np.ceil(len(rows) / st["block_length"]))
    W.step("B · Elastic Net first", "Stability selection", "freq = stability_selection(cfg, ctx, F)",
           [EN.stability_selection],
           f"The estimation window is cut into {nb} blocks of {st['block_length']} consecutive quarters. "
           f"{st['n_subsamples']} times, {st['fraction']:.0%} of the blocks are drawn at random (seed {st['seed']}) and "
           "step 15–16 is repeated on those quarters only. A feature is kept if it is selected in at least "
           f"{st['threshold']:.0%} of the subsamples. {len(sel)} features are kept.",
           [table(freq.assign(kept=freq["selection_frequency"] >= st["threshold"]), head=40)],
           {"freq": freq, "sel": sel}, saved=[("B2_en_stability.csv", freq)],
           pipeline_file=f"outputs/{run}/en_first/B2_en_stability.csv")

    sel["mnemonic"] = sel["feature"].map(lambda c: T.parse_feature(c)[0])
    for col in ("theme", "geography", "expected_sign", "description"):
        sel[col] = sel["mnemonic"].map(ctx.meta[col])
    sel["expected_sign"] = sel["expected_sign"].astype(int)
    Fs = F[sel["feature"]].iloc[ctx.est0:ctx.end + 1]
    sel["cluster"] = CL.cluster(Fs, e["group_corr"]).reindex(sel["feature"]).values
    grp = CL.representatives(sel.set_index("feature", drop=False), "score", e["primary_geography"], e["rep_tolerance"])
    cands = grp[grp["is_representative"]].sort_values("score", ascending=False)
    W.step("B · Elastic Net first", "Group correlated picks, keep one per group",
           "grp = representatives(sel, 'score', 'Ontario', 0.75)", [CL.cluster, CL.representatives],
           f"Same clustering as step 12, at |ρ| = {e['group_corr']}, but the score is the stability frequency "
           f"instead of |t|. {len(sel)} features → {len(cands)} groups.",
           [table(grp[["feature", "description", "geography", "score", "cluster", "cluster_size", "is_representative"]]
                  .sort_values(["cluster", "score"], ascending=[True, False]))],
           {"grp": grp, "cands": cands}, saved=[("B3_groups.csv", grp.reset_index(drop=True))],
           pipeline_file=f"outputs/{run}/en_first/B3_groups.csv")

    info_b = cands.reset_index(drop=True).set_index("feature")
    RB, chosen_b = S.exhaustive_search(cfg, ctx, F[cands["feature"]], info_b, "en_first")
    W.step("B · Elastic Net first", "Final search → Elastic-Net-first drivers",
           "RB, chosen_b = exhaustive_search(cfg, ctx, F[cands], info, 'en_first')",
           [S.exhaustive_search],
           "The same search as step 14, on the group representatives. " + f"<b>Chosen: {', '.join(chosen_b)}</b>.",
           [kv("How many combinations each rule removed", rule_funnel(F[cands["feature"]], info_b)), table(RB, head=25)],
           {"RB": RB, "chosen_b": chosen_b}, saved=[("B4_search.csv", RB)],
           pipeline_file=f"outputs/{run}/en_first/B4_search.csv")

    # ================================================================== Evaluate
    models = [A.ARDL("AR(1) benchmark", []), A.ARDL("Screen first", chosen_a), A.ARDL("Elastic Net first", chosen_b)]
    ev_ = {}
    summaries = []
    for mdl in models:
        ev_[mdl.name] = EV.evaluate(ctx, mdl)
        P = A.make_panel(ctx, ctx.base, mdl.features)
        mdl.fit(P, pd.Period(cfg["sample"]["estimation_start"], freq="Q"), ctx.idx[ctx.end])
        summaries.append(f"===== {mdl.name} =====\n" + mdl.res.summary().as_text())
    drivers = pd.concat([v["drivers"] for v in ev_.values()])
    W.step("Evaluate", "Fit the final ARDL models on the full sample",
           "m = ARDL(name, drivers).fit(P, estimation_start, hist_end); driver_table(ctx, m)",
           [A.make_panel, A.ARDL.fit, EV.driver_table],
           "Each set of drivers (and the AR(1) benchmark with none) is estimated by OLS with Newey-West standard errors: "
           "y<sub>t</sub> = a + ρ·y<sub>t−1</sub> + Σ b<sub>j</sub>·x<sub>j</sub> + dummies. The driver table turns each "
           "coefficient into a sentence, including the long-run effect b / (1 − ρ).",
           [table(drivers[["model", "term", "coef", "t", "p", "reading"]]),
            text("Full regression output", "\n\n".join(summaries))],
           {"models": [m_.name for m_ in models]},
           saved=[("drivers.csv", drivers), ("final_regressions.txt", "\n\n".join(summaries))],
           pipeline_file=f"outputs/{run}/comparison/3_drivers.csv")

    diag = pd.DataFrame([v["diagnostics"] for v in ev_.values()])
    W.step("Evaluate", "Residual diagnostics", "diagnostics(ctx, model, P)", [EV.diagnostics],
           "Ljung-Box (autocorrelation left in the residuals; want p &gt; 0.05), Breusch-Pagan (changing variance), "
           "Jarque-Bera (normality), Durbin-Watson (near 2 is good), CUSUM (parameter stability), the largest driver "
           "p-value and VIF, and any driver with the wrong sign.",
           [table(diag.T.reset_index().rename(columns={"index": "test", **{i: n_ for i, n_ in enumerate(diag["model"])}}).iloc[1:])],
           {"diag": diag}, saved=[("diagnostics.csv", diag)], pipeline_file=f"outputs/{run}/comparison/4_diagnostics.csv")

    by_h = pd.concat([v["by_horizon"] for v in ev_.values()])
    comp = by_h.groupby("model", sort=False)["MAPE_exCOVID_pct"].mean().round(3).rename("avg_MAPE_1_to_H_pct").reset_index()
    W.step("Evaluate", "Rolling backtest", "backtest(ctx, model, P)", [EV.backtest, A.ARDL.forecast],
           f"At every quarter from {cfg['sample']['backtest_start']} the model is re-estimated on data up to that "
           f"quarter and forecasts 1–{cfg['sample']['horizon']} quarters ahead with the realised macro values. "
           "Growth forecasts are cumulated into the balance and compared with the organic balance. Windows that "
           "touch a dummy quarter are left out of the MAPE. The drivers were chosen with all the data, so this is "
           "optimistic; step 24 removes that.",
           [table(comp), chart("Balance MAPE by horizon (%)", [{"name": nm_, "x": list(d["horizon_q"].astype(int)),
                                                                 "y": [float(v) for v in d["MAPE_exCOVID_pct"]]}
                                                                for nm_, d in by_h.groupby("model", sort=False)],
                                x_label="quarters ahead", y_label="MAPE %", x_type="num"),
            table(by_h.round(4), head=30)],
           {"by_h": by_h}, saved=[("backtest_by_horizon.csv", by_h)],
           pipeline_file=f"outputs/{run}/comparison/2_backtest_by_horizon.csv")

    scen = pd.concat([v["scenarios"] for v in ev_.values()])
    hist16 = ctx.level.iloc[-12:]
    series = [{"name": "Actual", "x": list(hist16.index.astype(str)), "y": [float(v) for v in hist16]}]
    for (mn, sn), d in scen.groupby(["model", "scenario"], sort=False):
        if mn == "AR(1) benchmark":
            continue
        series.append({"name": f"{mn} · {sn}", "x": [str(hist16.index[-1])] + list(d["date"]),
                       "y": [float(hist16.iloc[-1])] + [float(v) for v in d["balance_forecast"]]})
    W.step("Evaluate", "Scenario forecasts", "scenario_forecasts(ctx, model)", [EV.scenario_forecasts],
           "Each model is estimated once on history, then fed each scenario's macro path. Forecast growth is "
           "cumulated from the last actual balance.",
           [chart("Balance under each scenario", series),
            table(scen.round(4), head=12, tail=6)],
           {"scen": scen}, saved=[("scenario_forecasts.csv", scen)],
           pipeline_file=f"outputs/{run}/comparison/6_scenario_forecasts.csv")

    # check against the pipelines
    ra = screen_first.run(cfg, ctx, F, forms)
    rb = en_first.run(cfg, ctx, F, forms)
    checks = [{"check": "Screen first drivers equal screen_first.run()", "ok": ra["features"] == chosen_a,
               "detail": ", ".join(ra["features"])},
              {"check": "Elastic Net first drivers equal en_first.run()", "ok": rb["features"] == chosen_b,
               "detail": ", ".join(rb["features"])},
              {"check": "A5 search table identical", "ok": ra["tables"]["A5_search"].equals(RA), "detail": f"{len(RA)} rows"},
              {"check": "B4 search table identical", "ok": rb["tables"]["B4_search"].equals(RB), "detail": f"{len(RB)} rows"}]

    # ================================================================== OOS
    if not args.skip_oos:
        cache = StepCache(cfg["cache"]["dir"], enabled=False)
        E, picks, H = OOS.run_oos(cfg, ctx, {"screen_first": screen_first, "en_first": en_first},
                                  {"screen_first": chosen_a, "en_first": chosen_b}, cache)
        bh, summ, dmt, stab = OOS.summarise(E, picks, H)
        W.step("Evaluate", "Out-of-sample test with re-selection",
               "E, picks, H = run_oos(cfg, ctx, approaches, full_sample_drivers, cache)",
               [OOS.run_oos, OOS.select_at, OOS.truncated, OOS.dm_test],
               f"At each origin from {cfg['oos']['first_origin']} (every {cfg['oos']['step']} quarters) steps 7–19 are "
               "re-run using data up to that origin only, then the chosen model forecasts ahead. 're-selected' is the "
               "honest error; 'full-sample drivers' uses the drivers from steps 14 and 19 and shows how much the "
               "normal backtest flatters the models. Diebold-Mariano tests compare each model with AR(1).",
               [table(summ.round(3)), table(picks), table(dmt.round(3), head=30), table(stab.round(3))],
               {"E": E, "picks": picks}, saved=[("oos_summary.csv", summ), ("oos_drivers_by_origin.csv", picks),
                                                ("oos_dm_tests.csv", dmt)],
               pipeline_file=f"outputs/{run}/oos/1_oos_summary.csv")

    W.step("Check", "Check: step-by-step = pipeline", "screen_first.run(...); en_first.run(...)",
           [screen_first.run, en_first.run],
           "The pipelines in <code>src/pipelines/</code> run steps 10–14 and 15–19 in one call each. "
           "Running them here and comparing confirms this walkthrough did exactly what <code>run_pipeline.py</code> does.",
           [table(pd.DataFrame(checks))], {}, checks=checks)

    # ------------------------------------------------------------------ write page
    meta_page = {"run": run, "config": os.path.relpath(args.config, HERE), "folder": os.path.relpath(W.folder, HERE),
                 "n_vars": int(ctx.base.shape[1]), "n_features": int(F.shape[1]),
                 "chosen": {"Screen first": chosen_a, "Elastic Net first": chosen_b},
                 "mape": dict(zip(comp["model"], comp["avg_MAPE_1_to_H_pct"]))}
    data = json.dumps({"meta": meta_page, "steps": W.steps}, default=_cell)
    with open(os.path.join(W.folder, "walkthrough.json"), "w", encoding="utf-8") as fh:
        fh.write(data)
    with open(os.path.join(HERE, "tools", "walkthrough_template.html"), encoding="utf-8") as fh:
        tpl = fh.read()
    page = tpl.replace("/*__DATA__*/null", data.replace("</", "<\\/"))
    with open(os.path.join(W.folder, "walkthrough_body.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(os.path.join(W.folder, "walkthrough.html"), "w", encoding="utf-8") as fh:
        fh.write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                 '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>\n' + page +
                 "\n</body></html>\n")
    print(f"\nChecks: {'all passed' if all(c['ok'] for c in checks) else 'FAILED: ' + str(checks)}")
    print(f"Open {os.path.join(W.folder, 'walkthrough.html')}")


if __name__ == "__main__":
    main()
