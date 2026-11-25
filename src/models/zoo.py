"""All competing models, built from the selection results at a given estimation end.

Families
  Benchmark            AR(1) (past target growth only); mean growth (random walk with drift in the balance)
  Data-driven          the two selection approaches (screen first, Elastic Net first)
  Literature           specifications taken from published studies (config: literature.models), plus an
                       error-correction model with a theory long-run relation (literature.ecm)
  Hybrid               theory + data: core + data add-ons; theory long run + data short run; theory channels with
                       data-chosen measures; 50/50 combination; performance-weighted combination
  Machine learning     ridge, random forest, gradient boosting, neural network on the pooled drivers
  Regime switching     Markov switching, threshold and smooth-transition ARDLs, time-varying parameters,
                       break-adaptive ARDL, all on the literature core drivers
  Statistical / Deep   ARIMAX, Theta; TiDE, N-HiTS, TFT (small networks trained on this data)
  AI foundation model  Chronos-Bolt (own history) and Chronos-2 (with the literature drivers and their scenarios)
  Same growth          the target grows at the same rate as a chosen variable (exact, and with the average gap)
                       (config: models.same_growth)
  Auto ARDL            EViews-style ARDL with lags chosen by AIC/BIC (literature core, data-chosen drivers) and an
                       ARDL in levels with the Pesaran-Shin-Smith bounds test (config: models.auto_ardl)

models.only (config) limits the run to the listed models or families; the AR(1) benchmark is always kept.

The same function is used for the full-sample evaluation and, with data truncated at each origin, for the
out-of-sample test, so every model is re-selected with the information available at the time.
"""
import pandas as pd

from ..features.transforms import all_features, candidate_forms, parse_feature
from ..modelling.ardl import ARDL
from ..selection.screening import screen
from . import foundation, hybrid, literature as litmod
from .combination import Combination
from .ecm import ECM
from . import deep, regime
from .ml import LABELS, MLModel
from .simple import SameGrowth
from . import autoardl

APPROACH_NAMES = {"screen_first": "Screen first", "en_first": "Elastic Net first", "bayes_first": "Bayesian first",
                  "focus_first": "Explanatory: focus drivers", "factors_first": "Factors first",
                  "stable_first": "Stable first"}
APPROACH_FAMILY = {"focus_first": "Explanatory"}


def selection_info(approach, result):
    """What the zoo needs from one approach's result (kept small so it can be cached at every OOS origin).

    Args:
        result: the dict returned by the approach's run().
    """
    out = {"features": list(result["features"])}
    if result.get("focus"):
        out["focus"] = list(result["focus"])
    if result.get("bma_models"):
        out["bma_models"] = [(list(f), float(w)) for f, w in result["bma_models"]]
    t = result.get("tables", {})
    if "A4_shortlist" in t:
        out["shortlist"] = t["A4_shortlist"][["feature", "theme", "expected_sign"]].reset_index(drop=True)
    if "A2_screen_best_per_variable" in t:
        out["best"] = t["A2_screen_best_per_variable"][["feature", "theme", "coef", "abs_t", "expected_sign"]].reset_index(drop=True)
    return out


def _valid(ctx, features):
    """Split features into those whose series exist in the macro data and those that do not.

    Args:
        ctx: Context. features: feature names.

    Returns:
        (available, missing) lists.
    """
    ok = [f for f in features if parse_feature(f)[0] in ctx.hist.columns]
    return ok, [f for f in features if f not in ok]


def literature_models(cfg, ctx, log, used=None):
    """Literature specifications with concepts resolved to series names through the literature table.

    Args:
        used: dict filled with concept_id -> models that use it (for the table report).
    """
    table = getattr(ctx, "lit_table", None)
    out = []
    for spec in (cfg.get("literature") or {}).get("models", []):
        feats, dropped, signs = [], [], {}
        for f in spec["features"]:
            r = litmod.resolve_feature(f, table)
            (feats if r else dropped).append(r or f)
            if r and f in (spec.get("signs") or {}):
                signs[r] = spec["signs"][f]
        feats, missing = _valid(ctx, feats)
        if dropped:
            log(f"  literature model '{spec['name']}': {dropped} switched off or not mapped in the literature table")
        if missing:
            log(f"  literature model '{spec['name']}': {missing} not in the macro data - left out")
        if used is not None:
            for f in spec["features"]:
                used.setdefault(f.split("|")[0], []).append(spec["name"])
        if feats:
            out.append({**spec, "features": feats, "signs": {f: v for f, v in signs.items() if f in feats}})
    return out


def _screen_tables(cfg, ctx, sel):
    """Shortlist and per-variable screen results, from screen first if it ran, otherwise computed here."""
    s = sel.get("screen_first", {})
    if "shortlist" in s and "best" in s:
        return s["shortlist"], s["best"]
    forms = candidate_forms(cfg, ctx.hist.iloc[:ctx.end + 1], ctx.meta)
    F = all_features(cfg, ctx, forms)
    _, best = screen(cfg, ctx, F)
    passed = best[best["passed"]]
    return passed[["feature", "theme", "expected_sign"]].head(cfg["screen_first"]["shortlist_max"]), best


def wanted(cfg, name=None, family=None, names=()):
    """Is a model in models.only? Empty or missing models.only = every model.

    Entries match a family name, a model name, or the start of a model name (case does not matter).
    The AR(1) benchmark is always kept: every comparison and test is made against it.

    Args:
        cfg: configuration. name: model name. family: model family.
        names: for a whole family before its models exist: the names its models would get.

    Returns:
        bool.
    """
    only = [str(x).strip().lower() for x in ((cfg.get("models") or {}).get("only") or []) if str(x).strip()]
    if not only or name == "AR(1) benchmark":
        return True
    f = str(family or "").lower()
    cands = [str(n).lower() for n in ([name] if name is not None else list(names))]
    return any(o == f for o in only) or any(o == n or n.startswith(o) for o in only for n in cands)


FAMILIES = ["Benchmark", "Data-driven", "Literature", "Hybrid", "Same growth as a driver", "Auto ARDL",
            "Horizon combination", "Regime switching",
            "Statistical time series", "Machine learning", "Deep learning", "AI foundation model"]
FAMILIES_LOWER = {f.lower() for f in FAMILIES}


def same_growth_models(cfg, ctx, log=print):
    """The same-growth rules listed in models.same_growth, two per variable (exact and with the average gap).

    Each entry is a literature concept (e.g. disposable_income), a series name, or a dict
    {variable, lag (quarters, default 0), gap: both | yes | no}.

    Args:
        cfg: configuration. ctx: Context (literature table, metadata, macro history). log: message function.

    Returns:
        list of (model, how-it-is-specified) pairs.
    """
    out = []
    table = getattr(ctx, "lit_table", None)
    from ..features.transforms import RATE_TRS
    for item in (cfg.get("models") or {}).get("same_growth") or []:
        spec = item if isinstance(item, dict) else {"variable": item}
        var = str(spec.get("variable", "")).strip()
        if not var:
            continue
        code = litmod.resolve(var, table) if table is not None else var
        code = code or var
        if code not in ctx.hist.columns:
            log(f"  same-growth model for '{var}' skipped: no such series (map the concept in the literature workbook "
                "or use a series name)")
            continue
        if code in ctx.meta.index and str(ctx.meta.loc[code, "transform"]) in RATE_TRS:
            log(f"  same-growth model for '{var}' skipped: {code} is a rate, and the growth of a rate is not meaningful")
            continue
        lag = int(spec.get("lag", 0) or 0)
        f = f"{code}|dlog|L{lag}"
        label = (ctx.meta.loc[code, "description"] if code in ctx.meta.index else code)
        label = str(label).split(",")[0][:45]
        when = "same quarter" if lag == 0 else f"{lag} quarter{'s' if lag > 1 else ''} earlier"
        gaps = {"both": [False, True], "yes": [True], "true": [True], "no": [False], "false": [False]}
        for g in gaps.get(str(spec.get("gap", "both")).lower(), [False, True]):
            nm = f"Same growth as {label}" + (" + average gap" if g else "")
            how = (f"Target growth = growth of {code} ({when})" + ("; plus the average historical gap" if g else
                                                                  "; nothing estimated (constant target-to-variable ratio)"))
            out.append((SameGrowth(nm, f, gap=g), how))
    return out


def build(cfg, ctx, sel, log=print):
    """sel: {approach: selection_info(...)}. Returns (models, notes table, extra tables)."""
    mc = cfg["models"]
    models, notes, extra = [], [], {}

    def add(m, how, source=""):
        if not wanted(cfg, m.name, m.family):
            return
        models.append(m)
        notes.append({"model": m.name, "family": m.family, "drivers": ", ".join(m.drivers()) or "(none)",
                      "how it is specified": how, "source": source})

    if mc.get("benchmarks", True):
        add(ARDL("AR(1) benchmark", [], family="Benchmark"), "Last quarter's target growth only")
        add(ARDL("Mean growth benchmark", [], ar=False, family="Benchmark"),
            "Average historical growth: a random walk with drift in the balance")
    for ap, info in sel.items():
        if not info.get("features"):
            continue
        if ap == "focus_first":
            m_ = ARDL(APPROACH_NAMES[ap], info["features"], family=APPROACH_FAMILY[ap])
            m_.focus = list(info.get("focus") or [])
            add(m_, f"Focus drivers {', '.join(m_.focus)} always in; controls by double selection (predict the target "
                    "or a focus driver). Built to measure effects, not to win the forecast race (see focus_first/)")
            continue
        add(ARDL(APPROACH_NAMES[ap], info["features"], family="Data-driven"),
            "Selected from all candidates by the data (see the approach's folder)")
    bm = (sel.get("bayes_first") or {}).get("bma_models")
    if bm and (cfg.get("bayes_first") or {}).get("average", {}).get("enabled", True):
        valid = [(f, w) for f, w in bm if not _valid(ctx, f)[1]]
        if valid:
            add(Combination("Bayesian model average", [ARDL(f"model {i + 1}", f) for i, (f, _) in enumerate(valid)],
                            "fixed", family="Data-driven", weights=[w for _, w in valid]),
                f"Average of the {len(valid)} most probable models of the Bayesian first approach, weighted by their "
                "posterior probabilities (Fernandez-Ley-Steel g-prior, beta-binomial model prior, sign restrictions)")
    if (cfg["models"].get("same_growth") or []) and (not cfg["models"].get("only") or any(
            wanted(cfg, m_.name, m_.family) for m_, _ in same_growth_models(cfg, ctx, lambda *_: None))):
        for m_, how_ in same_growth_models(cfg, ctx, log):
            add(m_, how_)

    used = {}
    lit = literature_models(cfg, ctx, log, used)
    table = getattr(ctx, "lit_table", None)
    core_spec = next((m for m in lit if m.get("core")), lit[-1] if lit else None)
    core = core_spec["features"] if core_spec else []
    ecm_cfg = (cfg.get("literature") or {}).get("ecm")
    champion = sel.get("screen_first", sel.get("en_first", {})).get("features", [])

    def long_run_items():
        spec = ecm_cfg.get("long_run", "table")
        items = litmod.long_run(table) if spec in (None, "table") else \
            [{**i, "mnemonic": litmod.resolve(i["mnemonic"], table)} for i in spec]
        return [i for i in items if i["mnemonic"] and i["mnemonic"] in ctx.base.columns]

    def ecm(name, short_run, family, how):
        defl = litmod.resolve(ecm_cfg.get("deflator", cfg["features"]["deflator"]), table) or cfg["features"]["deflator"]
        return ECM(name, short_run, long_run_items(), defl, family=family, note=how)

    if mc.get("literature", True):
        for spec in lit:
            m_ = ARDL(spec["name"], spec["features"], family="Literature")
            m_.signs = dict(spec.get("signs") or {})             # expected signs from the workbook's models sheet
            add(m_, "Fixed specification from the literature", spec.get("source", ""))
        if ecm_cfg and core and long_run_items():
            sr = core if ecm_cfg.get("short_run", "core") == "core" else \
                _valid(ctx, [x for x in (litmod.resolve_feature(f, table) for f in ecm_cfg["short_run"]) if x])[0]
            for i in long_run_items():
                cid = next((c for c in (table.index if table is not None else []) if table.loc[c, "mnemonic"] == i["mnemonic"]), i["mnemonic"])
                used.setdefault(cid, []).append(ecm_cfg.get("name", "Literature ECM") + " (long run)")
            add(ecm(ecm_cfg.get("name", "Literature ECM"), sr, "Literature", ""),
                "Long-run relation of the real target to " + ", ".join(f"{i['mnemonic']} ({i.get('form', 'log')})"
                                                                  for i in long_run_items()) + "; short run: literature core",
                ecm_cfg.get("source", ""))

    acfg = mc.get("auto_ardl") or {}
    if acfg.get("enabled", True):
        mp, mq, ic = int(acfg.get("max_p", 4)), int(acfg.get("max_q", 4)), str(acfg.get("criterion", "bic")).lower()
        how = (f"Lags of target growth (1-{mp}) and of each driver (0-{mq}) chosen together by {ic.upper()} on a common "
               "sample, as EViews' automatic ARDL; chosen again at every re-estimation")
        sets = {"core": ("Auto-ARDL: literature core", core, (core_spec or {}).get("signs") or {}),
                "screen_first": ("Auto-ARDL: screen first drivers", sel.get("screen_first", {}).get("features", []), {}),
                "en_first": ("Auto-ARDL: Elastic Net drivers", sel.get("en_first", {}).get("features", []), {}),
                "bayes_first": ("Auto-ARDL: Bayesian drivers", sel.get("bayes_first", {}).get("features", []), {})}
        for key in acfg.get("on", ["core", "screen_first"]):
            if key not in sets or not sets[key][1]:
                continue
            nm, feats, sg = sets[key]
            bsig = {autoardl.base_of(f): v for f, v in sg.items()}
            add(autoardl.AutoARDL(nm, feats, mp, mq, ic, signs=bsig), how)
        if acfg.get("levels", True) and table is not None:
            items = [i for i in litmod.long_run(table) if i["mnemonic"] in ctx.hist.columns]
            defl = litmod.resolve((ecm_cfg or {}).get("deflator", cfg["features"]["deflator"]), table) \
                or cfg["features"]["deflator"]
            if items and defl in ctx.hist.columns:
                add(autoardl.LevelsARDL("Auto-ARDL in levels (bounds test)", items, defl, mp, mq, ic),
                    "ln(real organic target) on the long-run levels of the literature error-correction model, lags by "
                    f"{ic.upper()}; Pesaran-Shin-Smith bounds test for a long-run relation",
                    "Pesaran, Shin & Smith (2001)")

    if mc.get("hybrids", True) and core and champion:
        shortlist, best = _screen_tables(cfg, ctx, sel)
        h1, extra["hybrid_core_addons_search"] = hybrid.core_plus_addons(cfg, ctx, core, shortlist, mc.get("hybrid_max_add", 2))
        add(ARDL("Hybrid: theory core + data add-ons", h1, family="Hybrid"),
            "Literature core always in; data may add up to "
            f"{mc.get('hybrid_max_add', 2)} drivers from other themes under the usual rules")
        if ecm_cfg and long_run_items():
            add(ecm("Hybrid: theory long run + data short run", champion, "Hybrid", ""),
                "Theory long-run relation (as in the literature ECM) with the data-chosen drivers in the short run")
        chs = (cfg.get("literature") or {}).get("channels", "table")
        if isinstance(chs, str) and chs.lower() == "off":
            channels = []
        elif isinstance(chs, (list, tuple)) and chs and all(isinstance(c, str) for c in chs):
            channels = litmod.channels(table, ctx.meta, only=chs)       # a list of concept ids from the workbook
        elif isinstance(chs, (list, tuple)):
            channels = list(chs)                                         # already {concept, theme, sign} items
        else:                                                            # "table", empty, TRUE/yes
            channels = litmod.channels(table, ctx.meta)
        if channels:
            h3, extra["hybrid_channels"] = hybrid.channels_data_measures(cfg, ctx, best, channels, mc.get("channel_prune_t", 1.0))
            if h3:
                add(ARDL("Hybrid: theory channels, data measures", h3, family="Hybrid"),
                    "Theory fixes the channels and signs; data picks the best series, form and lag in each channel")
        add(Combination("Hybrid: 50/50 data + theory", [ARDL("data", champion), ARDL("theory", core)], "equal"),
            "Equal-weight average of the data-driven champion and the literature core")
        members = [ARDL("data", champion), ARDL("theory", core)]
        if "en_first" in sel and sel["en_first"]["features"] != champion:
            members.insert(1, ARDL("elastic net", sel["en_first"]["features"]))
        if ecm_cfg and long_run_items():
            members.append(ecm("ecm", core, "Hybrid", ""))
        add(Combination("Hybrid: performance-weighted", members, "inv_mse"),
            "Weights from each member's recent out-of-sample errors (Bates-Granger), shrunk half-way to equal: "
            + ", ".join(m.name for m in members))

    pool = list(dict.fromkeys(core + [f for info in sel.values() for f in info["features"]]))
    settings = mc.get("machine_learning_settings") or {}
    for kind in mc.get("machine_learning") or []:
        add(MLModel(LABELS[kind], pool, kind, settings.get(kind)),
            "Lagged target growth, dummies and the pooled drivers (literature core + data-chosen)")

    fcfg = mc.get("foundation") or {}
    en = fcfg.get("enabled", "auto")
    if en not in (False, "false", "off") and wanted(cfg, family="AI foundation model",
                                                     names=[m.get("name", "") for m in fcfg.get("models", [])]):
        ok, why = foundation.available()
        if not ok:
            msg = f"  foundation models skipped: {why}"
            if en is True:
                raise RuntimeError(msg)
            log(msg)
        else:
            for m in fcfg.get("models", []):
                cov = m.get("covariates")
                cov = core if cov == "core" else (_valid(ctx, cov)[0] if cov else [])
                if foundation.load(m["id"], fcfg.get("device", "cpu")) is None:
                    log(f"  {m['name']} skipped: {foundation.load_error(m['id'])}")
                    continue
                add(foundation.FoundationModel(m["name"], m["id"], cov, fcfg.get("device", "cpu")),
                    "Pretrained, zero-shot (not trained on this data)" + ("; uses the literature drivers and their scenario paths"
                                                                         if cov else "; own history only, same forecast in every scenario"),
                    m.get("source", ""))
    for kind in mc.get("statistical") or []:
        if kind == "arimax":
            add(deep.ARIMAX("ARIMAX: literature drivers", core),
                "Regression on the literature core drivers with ARMA(1,1) errors (statsmodels SARIMAX)")
        elif kind == "theta":
            add(deep.Theta("Theta (own history)"),
                "Theta method (M3 competition winner); target growth alone, same forecast in every scenario")

    rcfg = mc.get("regime") or {}
    if rcfg.get("models") and core:
        thr = rcfg.get("threshold") or core[0]
        thr = litmod.resolve_feature(thr, table) if table is not None else thr
        if not thr or not _valid(ctx, [thr])[0]:
            log(f"  regime models: threshold variable {rcfg.get('threshold')} not available, using {core[0]}")
            thr = core[0]
        how = {"markov": "Two unobserved regimes with their own mean growth and volatility; regime probability carried "
                         "forward with the estimated transition matrix",
               "threshold": f"All coefficients differ below / above an estimated threshold of {thr}; the scenario path "
                            "decides the regime in each future quarter",
               "smooth_transition": f"Coefficients move gradually between two regimes as {thr} rises (logistic weight)",
               "tvp": "Coefficients drift as random walks (Kalman filter); forecasts use the latest coefficients",
               "break_adaptive": "Tests for a structural break at each re-estimation; estimates on the data after "
                                 "the latest significant break"}
        for kind in rcfg["models"]:
            add(regime.build(kind, core, thr, (rcfg.get("settings") or {}).get(kind)),
                how[kind] + "; literature core drivers")

    dcfg = mc.get("deep") or {}
    en = dcfg.get("enabled", "auto")
    if en not in (False, "false", "off") and dcfg.get("models") and wanted(
            cfg, family="Deep learning", names=[deep.DEEP_LABELS.get(k, k) for k in dcfg["models"]]):
        ok, why = deep.deep_available()
        if not ok:
            if en is True:
                raise RuntimeError(f"deep models: {why}")
            log(f"  deep-learning models skipped: {why}")
        else:
            for kind in dcfg["models"]:
                feats = [] if kind in deep.NO_COV else core
                how = {"tide": "drivers' scenario paths as known future inputs",
                       "tft": "drivers' scenario paths as known future inputs (attention + LSTM)",
                       "tsmixer": "drivers' scenario paths as known future inputs",
                       "nhits": "drivers as past inputs (scenario path used beyond the first 4 quarters)",
                       "nbeats": "target growth alone; same forecast in every scenario"}[kind]
                add(deep.DeepModel(deep.DEEP_LABELS[kind], kind, feats, dcfg.get("settings")),
                    f"Small network trained on this data, average of several seeds; {how}")
    if table is not None:
        extra["literature_variables"] = litmod.status(table, set(ctx.hist.columns), used)
    return models, pd.DataFrame(notes), extra
