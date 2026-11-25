"""The process story: what happened in a run, step by step, for an audience (not at the code level).

Built from a finished run's output tables (and the config's own data files), so it describes exactly what the run
did and found. Each section has:

    part      the chapter it belongs to
    title     the finding, as a sentence
    takeaway  one line: what the audience should remember
    what      what was done, in plain words
    why       why it is done that way
    result    what happened in this run (with its numbers)
    read      how to read the chart or table
    numbers   a few key figures shown as tiles
    chart     a PNG made here       table  a small DataFrame

The same sections are rendered as a long HTML page (src/reporting/story_html.py) and a PowerPoint deck with the
full text in the speaker notes (src/reporting/story_pptx.py). Entry point: tools/process_story.py.
"""
import json
import os
import re

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ------------------------------------------------------------------------------------------------- palette
C = dict(ink="#1B2A41", navy="#0E3B4F", teal="#1F6F8B", gold="#E3A72F", green="#2E9E83", red="#C2553F",
         grey="#8796A8", purple="#5B4B8A", orange="#C77B30", plum="#8E4A6E", sea="#0F8A8A", mid="#4A5866",
         light="#EEF3F6", line="#D9E1E8", sky="#C9DCE6")
FAM_COL = {"Benchmark": C["grey"], "Data-driven": C["teal"], "Literature": C["gold"], "Hybrid": C["green"],
           "Machine learning": C["purple"], "AI foundation model": C["red"], "Statistical time series": C["orange"],
           "Deep learning": C["plum"], "Regime switching": C["sea"], "Auto ARDL": "#3B5BA5",
           "Same growth as a driver": "#8A6D3B", "Horizon combination": "#B5487A", "Look-ahead check": "#C9D3DC",
           "Explanatory": "#2F6B4F"}
FAM_ORDER = list(FAM_COL)
SCEN_COL = [C["navy"], C["green"], C["gold"], C["red"], C["purple"], C["orange"], C["sea"]]
TR_WORDS = {
    "dlog": ("quarterly growth", "the % change from the previous quarter"),
    "yoy": ("annual growth", "the % change from the same quarter a year earlier"),
    "dlog2": ("2-quarter growth", "the % change over the last two quarters"),
    "acc": ("acceleration", "how much faster or slower annual growth is than a year ago"),
    "gap": ("gap from trend", "how far the level sits above or below its average of the last 3 years"),
    "rdlog": ("real quarterly growth", "quarterly growth after removing inflation"),
    "ryoy": ("real annual growth", "annual growth after removing inflation"),
    "diff": ("quarterly change", "the change in the rate from the previous quarter, in percentage points"),
    "diff2": ("2-quarter change", "the change over the last two quarters, in percentage points"),
    "diff4": ("annual change", "the change from the same quarter a year earlier, in percentage points"),
    "lvl": ("level", "the rate itself"),
    "rgap": ("gap from trend", "how far the rate sits above or below its average of the last 3 years"),
}


def tr_words(tr):
    """Plain name and meaning of a transform code (moving windows included)."""
    if tr in TR_WORDS:
        return TR_WORDS[tr]
    m = re.match(r"(d?)(ma|vol)(\d+)$", str(tr))
    if m:
        k = m.group(3)
        if m.group(2) == "ma":
            return (f"{k}-quarter average {'change' if m.group(1) else 'growth'}",
                    f"the average quarterly {'change' if m.group(1) else 'growth'} over the last {k} quarters")
        return (f"{k}-quarter volatility", f"how much quarterly {'changes' if m.group(1) else 'growth'} varied over the last {k} quarters")
    return (str(tr), str(tr))


def apply_theme(name=None):
    """Chart colours from a theme (src/reporting/themes.py); call before Story(...).build()."""
    from . import themes as TH
    t = TH.get(name)
    h = lambda k: "#" + k                                    # noqa: E731
    ch = [h(x) for x in t["chart"]]
    C.update(ink=h(t["ink"]), navy=h(t["title"]), teal=ch[0], gold=ch[1], green=ch[2], red=ch[3], purple=ch[4],
             orange=ch[5], grey=ch[6], sea=ch[7], mid=h(t["mid"]), light=h(t["tile"]), line=h(t["line"]),
             sky=h(t["cover_sub"]), bg=h(t["card"]), box=h(t["head_bg"]), box2=h(t.get("box2", t["kicker"])),
             boxfg=h(t["head_fg"]), boxsub=h(t["cover_sub"]) if TH.is_dark(dict(bg=t["cover_bg"])) else "#DDDDDD",
             hl=h(t["highlight"]))
    FAM_COL.update({"Benchmark": C["grey"], "Data-driven": C["teal"], "Literature": C["gold"], "Hybrid": C["green"],
                    "Machine learning": C["purple"], "AI foundation model": C["red"],
                    "Statistical time series": C["orange"], "Regime switching": C["sea"]})
    _style()
    return t


def _style():
    plt.rcParams.update({"figure.facecolor": C.get("bg", "#FFFFFF"), "axes.facecolor": C.get("bg", "#FFFFFF"),
                         "savefig.facecolor": C.get("bg", "#FFFFFF"), "text.color": C["ink"],
                         "legend.labelcolor": C["ink"]})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": C["line"],
                         "axes.labelcolor": C["mid"], "xtick.color": C["mid"], "ytick.color": C["mid"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": C["line"], "grid.linewidth": 0.6, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "axes.titlecolor": C["ink"], "legend.frameon": False,
                         "legend.fontsize": 9, "figure.dpi": 100})


# ------------------------------------------------------------------------------------------------- helpers
def pct(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}%"


def num(v, d=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:,.{d}f}"


def plural(n, word, many=None):
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def parse(f):
    """(code, transform, lag) of a feature name 'CODE|tr|L1'; (f, None, None) otherwise."""
    if isinstance(f, str) and f.count("|") == 2:
        c, t, l = f.split("|")
        try:
            return c, t, int(l.lstrip("L"))
        except ValueError:
            return c, t, None
    return f, None, None


class Story:
    """Reads a run and builds its sections.

    Args:
        cfg: configuration (load_config). out: the run's output folder. charts: folder for the PNGs.
    """

    def __init__(self, cfg, out, charts):
        self.cfg, self.out, self.charts = cfg, out, charts
        os.makedirs(charts, exist_ok=True)
        _style()
        self.sections = []
        self.warnings = []

    # ------------------------------------------------------------------------------------------------- reading
    def read(self, *p):
        f = os.path.join(self.out, *p)
        return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()

    def add(self, part, title, takeaway="", what=(), why=(), result=(), read="", numbers=(), chart=None,
            table=None, table_note="", kind="content"):
        self.sections.append({"part": part, "title": title, "takeaway": takeaway, "what": list(what),
                              "why": list(why), "result": list(result), "read": read,
                              "numbers": [(str(a), str(b)) for a, b in numbers], "chart": chart, "table": table,
                              "table_note": table_note, "kind": kind})

    def fig(self, name, w=9, h=4.4):
        f, ax = plt.subplots(figsize=(w, h))
        return f, ax, os.path.join(self.charts, f"{name}.png")

    def save(self, f, path):
        f.tight_layout()
        f.savefig(path, dpi=200, facecolor=C.get("bg", "#FFFFFF"))
        plt.close(f)
        return path

    def label(self, f):
        """Plain label of a feature: 'House price index, Ontario: quarterly growth, 1 quarter earlier'."""
        code, tr, lag = parse(f)
        if tr is None:
            return str(f)
        d = self.meta.loc[code, "description"] if code in self.meta.index else code
        when = "same quarter" if lag == 0 else (f"{lag} quarter earlier" if lag == 1 else f"{lag} quarters earlier")
        if str(code).startswith("FAC.") and tr == "lvl":
            return f"{d}, {when}"
        return f"{d}: {tr_words(tr)[0]}, {when}"

    def short(self, f, n=46):
        code, tr, lag = parse(f)
        if tr is None:
            return str(f)
        d = self.meta.loc[code, "description"] if code in self.meta.index else code
        d = re.sub(r"\s*\(.*?\)", "", d)
        s = f"{d} ({tr_words(tr)[0]}, L{lag})"
        return s if len(s) <= n else s[:n - 1] + "…"

    # ------------------------------------------------------------------------------------------------- setup
    def load(self):
        from ..utils.io import Context
        cfg = self.cfg
        self.ctx = ctx = Context(cfg)
        self.meta = ctx.meta
        self.level = ctx.level
        self.form = ctx.target_form
        self.growth_word = "growth" if self.form == "dlog" else "change"
        t = cfg.get("target") or {}
        self.TARGET = str(t.get("label") or str(cfg["data"]["bank"]["target_column"]).replace("_", " "))
        if t.get("display_scale"):
            self.SCALE, self.UNITS = float(t["display_scale"]), str(t.get("display_units") or t.get("units") or "")
        elif t.get("units"):
            self.SCALE, self.UNITS = 1.0, str(t["units"])
        else:
            self.SCALE, self.UNITS = (1000.0, "$ billions") if self.form == "dlog" else (1.0, "units")
        if self.form != "dlog" and self.UNITS == "$ billions" and not t.get("units") is None and t.get("units") != "$ billions":
            self.SCALE, self.UNITS = 1.0, str(t["units"])          # a rate run carrying the example's display units
        s = cfg["sample"]
        self.H = int(s.get("horizon", 12))
        self.HZ = sorted({int(h) for h in s.get("report_horizons", [4, 8, 12]) if int(h) <= self.H} | {self.H})
        self.HM = max(self.HZ)
        self.EU = "%" if self.form == "dlog" else ""       # error unit
        self.MAXV = int((cfg.get("final_model") or {}).get("max_vars", 3))
        R = self.read
        self.comp, self.by_h, self.drv, self.diag = (R("comparison", f) for f in (
            "1_model_comparison.csv", "2_backtest_by_horizon.csv", "3_drivers.csv", "4_diagnostics.csv"))
        self.yearend, self.scen, self.paths, self.mlist = (R("comparison", f) for f in (
            "5_scenario_yearend.csv", "6_scenario_forecasts.csv", "7_coefficient_paths.csv", "9_model_list.csv"))
        self.litstat, self.btf = R("comparison", "10_literature_variables.csv"), R("comparison", "12_backtest_forecasts.csv")
        self.forms = R("common", "stationarity_tests.csv")
        self.a1, self.a2, self.a3, self.a4, self.a5, self.a6 = (R("screen_first", f) for f in (
            "A1_screen_all.csv", "A2_screen_best_per_variable.csv", "A3_clusters.csv", "A4_shortlist.csv",
            "A5_search.csv", "A6_search_rules.csv"))
        self.b1, self.b2, self.b3, self.b4, self.b5 = (R("en_first", f) for f in (
            "B1_en_full_sample.csv", "B2_en_stability.csv", "B3_groups.csv", "B4_search.csv", "B5_search_rules.csv"))
        self.c1, self.c2, self.c3, self.c5, self.c6 = (R("bayes_first", f) for f in (
            "C1_bma_settings.csv", "C2_bma_inclusion.csv", "C3_bma_top_models.csv", "C5_search.csv",
            "C6_search_rules.csv"))
        self.oos, self.oos_h, self.dm, self.picks, self.stab, self.hold = (R("oos", f) for f in (
            "1_oos_summary.csv", "2_oos_by_horizon.csv", "3_dm_tests_vs_benchmark.csv", "4_drivers_by_origin.csv",
            "5_driver_stability.csv", "7_holdout.csv"))
        self.terms, self.nextq = R("formula", "1_terms.csv"), R("formula", "2_next_quarters.csv")
        self.e1, self.e2, self.e3 = (R("factors_first", f) for f in ("E1_factors.csv", "E2_loadings.csv",
                                                                     "E3_best_lag.csv"))
        self.f1, self.f2 = R("stable_first", "F1_selection_probabilities.csv"), R("stable_first", "F2_settings.csv")
        self.d1, self.d2, self.d3 = (R("focus_first", f) for f in (
            "D1_focus_drivers.csv", "D2_double_selection.csv", "D3_settings.csv"))
        self.i_tied, self.i_resp, self.i_sens, self.i_rel, self.i_att, self.i_wi = (R("impact", f) for f in (
            "1_tied_models.csv", "2_responses.csv", "3_sensitivity.csv", "4_reliability.csv", "5_attribution.csv",
            "6_what_if.csv"))
        rc = f"avg_1_{self.HM}_pct"
        if self.comp.empty:
            raise SystemExit(f"No results in {self.out}: run run_pipeline.py with this config first")
        if rc not in self.comp:                       # longest horizon available
            hs = sorted(int(c.split("_")[2]) for c in self.comp.columns if re.match(r"avg_1_\d+_pct", c))
            self.HM = hs[-1]
            self.HZ = [h for h in self.HZ if h <= self.HM] or [self.HM]
        self.HAVE_OOS = not self.oos.empty and f"avg_1_{self.HM}_pct" in self.oos
        rank = (self.oos[self.oos["family"] != "Look-ahead check"] if self.HAVE_OOS else self.comp).copy()
        rank = rank.drop(columns=["drivers"], errors="ignore").merge(self.mlist[["model", "drivers"]], on="model",
                                                                      how="left")
        rank["n_vars"] = rank["drivers"].map(self._n_vars)
        rank["single_equation"] = rank["family"].isin(["Data-driven", "Literature", "Hybrid"]) & rank["n_vars"].notna()
        self.rank = rank.sort_values(f"avg_1_{self.HM}_pct").reset_index(drop=True)
        within = self.rank[self.rank["single_equation"] & self.rank["n_vars"].between(1, self.MAXV)]
        self.REC = (within.iloc[0] if len(within) else self.rank.iloc[0])
        self.SC = R("recommendation", "1_scorecard.csv")
        self.SCJ = {}
        rj = os.path.join(self.out, "recommendation", "recommended.json")
        if os.path.exists(rj):                     # the model scorecard decides (src/selection/scorecard.py)
            self.SCJ = json.load(open(rj, encoding="utf-8"))
            hit = self.rank[self.rank["model"] == self.SCJ.get("model")]
            if len(hit):
                self.REC = hit.iloc[0]
        self.AR = self.rank.set_index("model").loc["AR(1) benchmark"] if "AR(1) benchmark" in set(self.rank["model"]) else None
        self.basis = ("out of sample (drivers re-chosen at every origin)" if self.HAVE_OOS
                      else "rolling backtest (drivers chosen on all the data)")
        self.rec_drivers = [d for d in str(self.REC.get("drivers", "")).split(", ") if "|" in d]
        self.u = (lambda v: f"{v / self.SCALE:,.1f}") if self.form == "dlog" else (lambda v: f"{v / self.SCALE:,.2f}")
        self.ulab = self.UNITS

    @staticmethod
    def _n_vars(s):
        if not isinstance(s, str) or s == "(none)":
            return 0
        if "%" in s:
            return np.nan
        return len({d.replace("long run: ", "").split("|")[0].split(":")[0] for d in s.split(", ")})

    def err(self, v):
        return pct(v) if self.EU == "%" else num(v, 3)

    # ================================================================================================ sections
    def build(self):
        self.load()
        for f in (self.s_title, self.s_question, self.s_impact_headline, self.s_overview, self.s_target, self.s_events, self.s_macro,
                  self.s_signs, self.s_literature, self.s_transforms, self.s_stationarity, self.s_lags,
                  self.s_funnel, self.s_screen, self.s_bestform, self.s_clusters, self.s_shortlist, self.s_en,
                  self.s_stability, self.s_bayes, self.s_factors, self.s_stable, self.s_focus, self.s_search_rules, self.s_search_rank, self.s_search_pick, self.s_families, self.s_twins,
                  self.s_backtest_design, self.s_one_origin, self.s_all_origins, self.s_leaderboard,
                  self.s_by_horizon, self.s_oos_design, self.s_oos_gap, self.s_driver_stability, self.s_holdout,
                  self.s_scorecard, self.s_rec_model, self.s_rec_effects, self.s_rec_fit, self.s_coef_paths, self.s_diagnostics,
                  self.s_scen_inputs, self.s_scen_forecast, self.s_contrib, self.s_scen_check,
                  self.s_impact_responses, self.s_impact_reliability, self.s_impact_attribution, self.s_impact_whatif,
                  self.s_limits,
                  self.s_recommend, self.s_glossary, self.s_settings):
            try:
                f()
            except Exception as e:                    # one missing table never stops the story
                self.warnings.append(f"{f.__name__[2:]}: {type(e).__name__}: {e}")
        return self.sections

    # ---------------------------------------------------------------------------------------------- part 0
    def s_title(self):
        c = self.cfg
        self.add("", f"How the {self.TARGET} forecast was built",
                 f"A step-by-step account of run '{c['run_name']}': the data, every choice the process made, how "
                 f"the models were tested and what they forecast.",
                 kind="title",
                 numbers=[("Run", c["run_name"]), ("History", f"{self.ctx.idx[self.ctx.est0]} to {self.ctx.idx[self.ctx.end]}"),
                          ("Horizon", f"{self.H} quarters"), ("Models compared", str(len(self.comp)))])

    def s_question(self):
        L = self.level
        last, q = L.iloc[-1], L.index[-1]
        g1 = (L.iloc[-1] / L.iloc[-5] - 1) * 100 if self.form == "dlog" and len(L) > 4 else None
        scen = list(self.cfg["data"]["macro"]["scenarios"])
        self.add("1. The question", f"We forecast the {self.TARGET} for the next {self.H} quarters under "
                 f"{plural(len(scen), 'scenario')}",
                 f"The model must forecast well, be explainable driver by driver, and respond sensibly to "
                 f"economic scenarios.",
                 what=[f"The target is the {self.TARGET}, observed quarterly. The latest value ({q}) is "
                       f"{self.u(last)} {self.ulab}" + (f", {g1:+.1f}% over the last year." if g1 is not None else "."),
                       f"The forecast runs {self.H} quarters ahead of the last observed quarter, under each of the "
                       f"scenarios provided: {', '.join(scen)}.",
                       "Rather than forecasting the level directly, the model forecasts the quarterly "
                       f"{self.growth_word} and adds it up, quarter after quarter, to get the level."],
                 why=["Three requirements shape every choice in the process:",
                      "Accuracy: the forecast is judged on how close the forecast level comes to the actual level, "
                      "1 to 12 quarters ahead, in tests that only use information available at the time.",
                      "Explainability: the recommended model is a short equation with a few economic drivers, each "
                      "with the sign economic theory expects, so every movement in the forecast can be traced to a driver.",
                      "Scenario readiness: the drivers must be variables that the scenario providers forecast, so the "
                      "model can be run under each scenario."],
                 result=[f"The process compared {len(self.comp)} models from {self.comp['family'].nunique()} "
                         f"families before recommending one."],
                 numbers=[("Latest value", f"{self.u(last)} {self.ulab}"), ("As of", str(q)),
                          ("Forecast", f"{self.H} quarters"), ("Scenarios", str(len(scen)))])

    def s_overview(self):
        n_var = self.forms["mnemonic"].nunique() if not self.forms.empty else len(self.meta)
        n_feat = len(self.a1) if not self.a1.empty else None
        steps = [("Prepare the target", "Organic growth: one-off events removed"),
                 ("Collect macro data", f"{len(self.meta):,} series, expected signs"),
                 ("Build candidates", f"{n_var} variables x forms x lags = {num(n_feat) if n_feat else 'many'} candidates"),
                 ("Choose drivers", "Two independent approaches, same final search"),
                 ("Compete", f"{len(self.comp)} models in {self.comp['family'].nunique()} families"),
                 ("Test", "Backtest and out-of-sample test, 1-12 quarters"),
                 ("Recommend", f"{self.REC['model']}"),
                 ("Forecast", "Every scenario, explained driver by driver")]
        f, ax, p = self.fig("overview", 11, 4.6)
        ax.axis("off")
        ax.grid(False)
        for i, (h, s) in enumerate(steps):
            col_, row = i % 4, i // 4
            x, yb = col_ * 0.25, 0.53 - row * 0.5
            ax.add_patch(plt.Rectangle((x + 0.008, yb), 0.234, 0.42,
                                       color=C.get("box", C["navy"]) if i not in (3, 5) else C.get("box2", C["teal"]),
                                       transform=ax.transAxes))
            ax.text(x + 0.03, yb + 0.33, f"{i + 1}", ha="left", va="center", color=C.get("hl", C["gold"]), fontsize=20,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(x + 0.03, yb + 0.21, h, ha="left", va="center", color=C.get("boxfg", "white"), fontsize=12.5,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(x + 0.03, yb + 0.12, "\n".join(_wrap(s, 30)), ha="left", va="top", color=C.get("boxsub", "#C9DCE6"),
                    fontsize=9.5, transform=ax.transAxes)
        self.add("1. The question", "Eight steps take raw data to a tested, explained forecast",
                 "Each step narrows the possibilities or tests what is left; the rest of this document follows them in order.",
                 what=[f"{i + 1}. {h}: {s}." for i, (h, s) in enumerate(steps)],
                 why=["The process is fully automatic and repeatable: the same data and settings always give the "
                      "same result, and every intermediate table is saved, so any number in this document can be "
                      "checked in the run's output folder."],
                 chart=self.save(f, p))

    # ---------------------------------------------------------------------------------------------- part 2: data
    def s_target(self):
        L, y = self.level, self.ctx.y
        dq = [str(q) for q in self.cfg["sample"].get("dummy_quarters", [])]
        f, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        p = os.path.join(self.charts, "target.png")
        x = L.index.to_timestamp()
        axs[0].plot(x, L.values / self.SCALE, color=C["navy"], lw=2)
        axs[0].set_title(f"{self.TARGET} ({self.ulab})", loc="left")
        yy = (y * 100 if self.form == "dlog" else y).reindex(L.index)
        cols = [C["red"] if str(q) in dq else C["teal"] for q in L.index]
        axs[1].bar(x, yy.values, width=70, color=cols)
        axs[1].set_title(f"Quarterly organic {self.growth_word}" + (" (%)" if self.form == "dlog" else "")
                         + "; red: quarters given their own dummy", loc="left")
        axs[1].axhline(0, color=C["mid"], lw=0.8)
        yv = yy.dropna()
        st = yv.loc[self.ctx.idx[self.ctx.est0]:]
        avg, sd = st.mean(), st.std()
        hi, lo = st.idxmax(), st.idxmin()
        dd = 2 if self.form == "dlog" else 3
        self.add("2. The data", (f"The {self.TARGET} grew on average {avg:.2f}% a quarter" if self.form == "dlog" else
                                 f"The {self.TARGET} changed by {avg:+.3f} {self.UNITS} a quarter on average")
                 + f" since {self.ctx.idx[self.ctx.est0]}",
                 "This is the history the models learn from; its swings are what the drivers must explain.",
                 what=[f"The history runs from {L.index[0]} to {L.index[-1]} ({len(L)} quarters). Models are "
                       f"estimated from {self.ctx.idx[self.ctx.est0]}.",
                       f"The model works with the quarterly {self.growth_word} of the {self.TARGET}, because changes are "
                       "stable over time while the level keeps rising; a level-on-level regression would find "
                       "relations between any two rising series (spurious regression)."],
                 result=[f"Average quarterly {self.growth_word}: {avg:.{dd}f}{self.EU}; standard deviation {sd:.{dd}f}{self.EU}.",
                         f"Largest rise: {hi} ({yv[hi]:+.{dd}f}{self.EU}); largest fall or weakest quarter: {lo} ({yv[lo]:+.{dd}f}{self.EU}).",
                         (f"{plural(len(dq), 'quarter')} ({', '.join(dq)}) get their own dummy variable: their growth "
                          "was driven by a one-time shock (the pandemic), so they are not allowed to distort the "
                          "estimated effects of the drivers, and they are left out when forecast errors are scored."
                          if dq else "No quarters get their own dummy variable.")],
                 read=f"Top: the {self.TARGET}. Bottom: its quarterly {self.growth_word}, the series every model actually explains.",
                 numbers=[("Quarters", str(len(L))), (f"Avg {self.growth_word}", f"{avg:.{dd}f}{self.EU}"),
                          ("Volatility", f"{sd:.{dd}f}{self.EU}"), ("Dummy quarters", str(len(dq)))],
                 chart=self.save(f, p))

    def s_events(self):
        e = self.cfg["data"]["events"]
        from ..utils.io import read_table, to_quarters
        if not (e.get("file") and os.path.exists(e["file"])):
            self.add("2. The data", "No one-off events were removed",
                     "Reported growth is used as is.",
                     what=["One-off events (acquisitions, portfolio sales, write-offs, reclassifications) change the "
                           "balance without any economic cause. When an events file is given, their amounts are "
                           "removed from the quarter they happened in, giving 'organic' growth."],
                     result=["This run has no events file, so reported and organic growth are the same."])
            return
        ev = read_table(e["file"], e.get("sheet"))
        ev = ev[ev[e["series_column"]].astype(str) == self.cfg["data"]["bank"]["target_column"]].copy()
        ev["quarter"] = [str(q) for q in to_quarters(ev[e["date_column"]])]
        L = self.level
        rows = []
        for _, r in ev.iterrows():
            q = pd.Period(r["quarter"], freq="Q")
            if q not in L.index or q == L.index[0]:
                continue
            prev, cur, a = L.shift(1).loc[q], L.loc[q], float(r[e["amount_column"]])
            rep = (np.log(cur / prev) * 100) if self.form == "dlog" else cur - prev
            org = (np.log((cur - a) / prev) * 100) if self.form == "dlog" else cur - a - prev
            desc = next((str(r[c]) for c in ev.columns if c.lower() in ("description", "event", "note", "comment")), "")
            rows.append({"Quarter": str(q), "Event": desc, "Amount": f"{a / self.SCALE:,.2f}",
                         "Reported growth": f"{rep:+.2f}{self.EU}", "Organic growth": f"{org:+.2f}{self.EU}"})
        T = pd.DataFrame(rows)
        self.add("2. The data", f"{plural(len(T), 'one-off event')} removed to get organic growth",
                 "The model learns from growth with an economic cause, not from acquisitions or portfolio moves.",
                 what=["Each event's amount is taken out of the balance in the quarter it happened: organic growth = "
                       "ln((balance - event amount) / previous balance).",
                       "The balance itself is not changed; only the growth the models learn from."],
                 why=["An acquisition can add several percent to the balance in one quarter. Left in, it would look "
                      "like a sudden burst of demand that no macro driver can explain, distorting every estimate."],
                 result=[f"Removed: {', '.join(T['Quarter'])}." if len(T) else "No event matched the target."],
                 table=T, table_note=f"Amounts in {self.ulab}.")

    def s_macro(self):
        m = self.meta
        th = m["theme"].replace("", "(none)").value_counts()
        geo = m["geography"].value_counts()
        scen = self.cfg["data"]["macro"]["scenarios"]
        f, ax, p = self.fig("macro_themes", 9, 4.2)
        th.sort_values().plot.barh(ax=ax, color=C["teal"])
        ax.set_ylabel("")
        ax.set_title("Macro series by theme", loc="left")
        for i, v in enumerate(th.sort_values().values):
            ax.text(v, i, f" {v}", va="center", fontsize=8, color=C["mid"])
        fe = self.cfg.get("features") or {}
        self.add("2. The data", f"{len(m):,} macro series in {len(th)} themes are the candidate drivers",
                 "Any of these could, in principle, drive the target; the process has to find the few that do.",
                 what=[f"Each scenario file ({', '.join(scen)}) holds the same {len(m):,} series: history up to the "
                       "last quarter, then the scenario's forecast.",
                       "A metadata file describes each series: its description, theme (output, housing, rates, "
                       "labour...), geography, natural transformation and expected effect on the target."],
                 why=["Only variables that the scenarios forecast can be drivers: otherwise the model could not be "
                      "run under a scenario."],
                 result=[f"Largest themes: " + ", ".join(f"{k} ({v})" for k, v in th.head(5).items()) + ".",
                         f"Geographies: " + ", ".join(f"{k} ({v})" for k, v in geo.head(5).items())
                         + (", ..." if len(geo) > 5 else "") + ".",
                         (f"Only {', '.join(fe['allowed_geographies'])} series are considered as drivers"
                          if fe.get("allowed_geographies") else "All geographies are considered")
                         + (f"; themes left out: {', '.join(fe['excluded_themes'])}." if fe.get("excluded_themes") else ".")],
                 numbers=[("Series", f"{len(m):,}"), ("Themes", str(len(th))), ("Geographies", str(len(geo))),
                          ("Scenarios", str(len(scen)))],
                 chart=self.save(f, p))

    def s_signs(self):
        m = self.meta.copy()
        m["sign"] = m["expected_sign"].map({1: "raises the target", -1: "lowers the target", 0: "no expectation"})
        tab = pd.crosstab(m["theme"].replace("", "(none)"), m["sign"])
        tab = tab.loc[tab.sum(axis=1).sort_values().index]
        f, ax, p = self.fig("signs", 9, 4.4)
        left = np.zeros(len(tab))
        for k, col in (("raises the target", C["green"]), ("lowers the target", C["red"]), ("no expectation", C["line"])):
            if k in tab:
                ax.barh(tab.index, tab[k], left=left, color=col, label=k)
                left += tab[k].values
        ax.legend(loc="lower right")
        ax.set_title("Expected effect on the target, by theme", loc="left")
        npos, nneg, nz = (int((m["expected_sign"] == s).sum()) for s in (1, -1, 0))
        self.add("2. The data", f"Every series carries an expected sign: {npos} raise the target, {nneg} lower it",
                 "The expected sign is the economic theory built into the process: a driver with the wrong sign is never chosen.",
                 what=["Before any estimation, each series is given the direction in which it should move the target: "
                       "+1 (e.g. higher house prices, more income), -1 (e.g. higher interest rates, more "
                       "unemployment) or 0 (no clear expectation).",
                       "The signs come from the metadata file and from the literature review: when the literature "
                       "workbook names a series for a concept, the concept's sign is applied to it."],
                 why=["With hundreds of candidates, some will fit history by chance with an economically absurd sign "
                      "(say, unemployment raising loan growth). Such a model would forecast nonsense under a recession "
                      "scenario. Requiring the expected sign removes them.",
                      "Series with sign 0 are kept out of the data-driven selection entirely, so the user decides "
                      "which variables are allowed to compete."],
                 result=[f"{npos} series expected to raise the target, {nneg} to lower it, {nz} without an expectation "
                         "(not eligible for selection)."],
                 numbers=[("Raise", str(npos)), ("Lower", str(nneg)), ("No expectation", str(nz))],
                 chart=self.save(f, p))

    def s_literature(self):
        L = self.litstat
        if L.empty:
            return
        T = L[["concept", "effect_on_loan_growth", "series_name", "status"]].copy() if "effect_on_loan_growth" in L else \
            L[[c for c in L.columns if c in ("concept", "series_name", "status")]].copy()
        T.columns = ["Concept", "Expected effect", "Our series", "Status"][:T.shape[1]]
        n_use = int((L["status"] == "in use").sum())
        lm = self.mlist[self.mlist["family"] == "Literature"]
        self.add("2. The data", f"The literature review names {len(L)} economic concepts; {n_use} are used by models",
                 "Published research tells us which channels drive the target and in which direction.",
                 what=["A literature review collected the published models of the target and the economic concepts "
                       "they rely on (for loans: house prices, income, interest rates, unemployment, ...).",
                       "Each concept is mapped to one of our macro series. The published models are then rebuilt with "
                       "our series and compete with everything else."],
                 why=["Theory-based models are the benchmark for any data-driven choice: if a model chosen from the "
                      "data cannot beat the published specifications, the data-driven choice is not worth it."],
                 result=[f"{plural(len(lm), 'published model')} rebuilt: {', '.join(m.replace('Literature: ', '') for m in lm['model'])}."
                         if len(lm) else "No literature models in this run.",
                         f"Concepts in use: {n_use} of {len(L)}; the others are mapped but unused, unmapped or "
                         "switched off."],
                 table=T, table_note="Status: in use = used by at least one literature or hybrid model.")

    # ---------------------------------------------------------------------------------------------- part 3: candidates
    def s_transforms(self):
        F = self.forms
        if F.empty:
            return
        cnt = F["transform"].value_counts()
        T = pd.DataFrame([{"Form": tr_words(t)[0], "What it measures": tr_words(t)[1], "Variables": int(n)}
                          for t, n in cnt.items()])
        self.add("3. Building candidates", f"Each variable is tried in up to {cnt.size} forms",
                 "The same variable can matter as its growth, its annual change, its acceleration or its gap from trend.",
                 what=["Each series is transformed into the forms that make economic sense for it: growth rates for "
                       "series in levels (prices, income, GDP), changes for rates (interest rates, unemployment).",
                       "Moving averages or volatility can be added as further forms."],
                 why=["Economic theory says that house prices matter, but not whether it is last quarter's growth, the "
                      "annual growth or the gap from trend that drives borrowing. Trying the reasonable forms lets "
                      "the data answer that, while each form keeps a clear economic meaning."],
                 result=[f"{F['mnemonic'].nunique()} variables give {len(F)} variable-form pairs."],
                 table=T)

    def s_stationarity(self):
        F = self.forms
        if F.empty:
            return
        st = F.groupby("transform")["stationary"].agg(["sum", "count"]).reset_index()
        st["share"] = st["sum"] / st["count"] * 100
        f, ax, p = self.fig("stationarity", 9, 3.8)
        st = st.sort_values("share")
        ax.barh([tr_words(t)[0] for t in st["transform"]], st["share"], color=C["teal"])
        ax.set_xlim(0, 118)
        ax.set_title("Share of variables whose form passes the ADF test (%)", loc="left")
        for i, (v, n, c) in enumerate(zip(st["share"], st["sum"], st["count"])):
            ax.text(v, i, f" {int(n)}/{int(c)}", va="center", fontsize=8, color=C["mid"])
        npass, ntot = int(F["stationary"].sum()), len(F)
        self.add("3. Building candidates", f"{npass} of {ntot} variable forms are stationary and can be used",
                 "Only forms that move around a stable average are allowed, so relations are not spurious.",
                 what=["Each form is tested with the Augmented Dickey-Fuller (ADF) test. A form passes when the test "
                       "rejects a unit root at the 5% level: it fluctuates around a stable mean instead of wandering."],
                 why=["Two trending series always look related, even when they have nothing to do with each other. "
                      "Using only stationary forms on both sides of the equation avoids such spurious relations."],
                 result=[f"{F[F['stationary']]['mnemonic'].nunique()} of {F['mnemonic'].nunique()} variables have at "
                         "least one stationary form. Levels of trending series are the forms that fail, as expected."],
                 read="Each bar: the share of variables for which that form passes; the label gives the count.",
                 numbers=[("Forms tested", str(ntot)), ("Stationary", str(npass))],
                 chart=self.save(f, p))

    def s_lags(self):
        if self.a1.empty:
            return
        lags = sorted(self.a1["lag"].unique())
        self.add("3. Building candidates", f"Lags of {lags[0]} to {lags[-1]} quarters give {len(self.a1):,} candidate drivers",
                 "Each stationary form enters at every lag, because effects take time to reach the target.",
                 what=[f"Each stationary form is tried at lags {', '.join(map(str, lags))}: the same quarter and "
                       "up to four quarters earlier.",
                       "A driver at lag 1 means: this quarter's target growth responds to the driver's value one "
                       "quarter earlier. That also makes forecasting easier, because the first forecast quarter uses "
                       "an observed value."],
                 why=["Rate changes take several quarters to reach new lending; house prices act faster. Fixing the lag "
                      "in advance would impose one timing on every channel."],
                 result=[f"{len(self.a1):,} candidate drivers (variable x form x lag) go into the selection."],
                 numbers=[("Candidates", f"{len(self.a1):,}"), ("Lags", f"{lags[0]}-{lags[-1]}")])

    def s_funnel(self):
        steps = []
        if not self.a1.empty:
            steps = [("Macro series", len(self.meta)),
                     ("With a stationary form", self.forms[self.forms["stationary"]]["mnemonic"].nunique() if not self.forms.empty else None),
                     ("Candidate drivers (form x lag)", len(self.a1)),
                     ("Significant, right sign", int(self.a1["passed"].sum())),
                     ("Variables (best form each)", int(self.a2["passed"].sum()) if "passed" in self.a2 else len(self.a2)),
                     ("Clusters of look-alikes", int(self.a3["cluster"].nunique()) if not self.a3.empty else None),
                     ("Shortlist", len(self.a4)),
                     ("Combinations tried", self._rule(self.a6, "combinations")),
                     ("Passed every rule", self._rule(self.a6, "passed every rule")),
                     ("Chosen drivers", len(str(self.mlist.set_index("model").loc["Screen first", "drivers"]).split(", "))
                      if "Screen first" in set(self.mlist["model"]) else None)]
        steps = [(a, b) for a, b in steps if b is not None]
        if not steps:
            return
        f, ax, p = self.fig("funnel", 9, 4.6)
        ax.grid(False)
        ax.axis("off")
        mx = np.log10(max(b for _, b in steps) + 1)
        for i, (a, b) in enumerate(steps):
            w = 0.15 + 0.85 * np.log10(b + 1) / mx
            yv = len(steps) - i
            ax.barh(yv, w, left=(1 - w) / 2, color=C["navy"] if i < len(steps) - 1 else C["gold"], height=0.8)
            ax.text(0.5, yv, f"{b:,}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            ax.text(1.02, yv, a, ha="left", va="center", color=C["ink"], fontsize=9)
        ax.set_xlim(0, 1.6)
        ax.set_title("Screen-first approach: from all series to the chosen drivers (width on a log scale)", loc="left")
        self.add("3. Building candidates", f"The funnel: {steps[0][1]:,} series narrowed to {steps[-1][1]} drivers",
                 "Each stage removes candidates for a stated reason; nothing is chosen by hand.",
                 what=[f"{a}: {b:,}." for a, b in steps],
                 why=["The funnel matters for credibility: the chosen drivers survived every test, out of thousands of "
                      "possibilities, and the rules that removed the others are economic (sign), statistical "
                      "(significance, collinearity) and practical (forecast accuracy)."],
                 read="Each bar is one stage; widths are on a log scale so the late stages stay visible.",
                 chart=self.save(f, p))

    @staticmethod
    def _rule(T, key):
        if T.empty:
            return None
        r = T[T["step"] == key]
        return int(r["combinations"].iloc[0]) if len(r) else None

    # ---------------------------------------------------------------------------------------------- part 4: approach A
    def s_screen(self):
        A = self.a1
        if A.empty:
            return
        f, ax, p = self.fig("screen", 9, 4.2)
        t = A["t_hac"].clip(-15, 15)
        ok = A["passed"]
        ax.hist([t[ok], t[~ok]], bins=60, stacked=True, color=[C["teal"], C["line"]],
                label=["significant with the expected sign", "rejected"])
        for v in (-1.96, 1.96):
            ax.axvline(v, color=C["red"], lw=1, ls="--")
        ax.legend(loc="upper left")
        ax.set_title("t-statistic of each candidate, tested one at a time", loc="left")
        ax.set_xlabel("t-statistic (Newey-West); dashed: ±1.96")
        n_sig = int((A["p_hac"] < 0.05).sum())
        n_wrong = int(((A["p_hac"] < 0.05) & ~A["sign_ok"]).sum())
        top = A[A["passed"]].nlargest(8, "abs_t")
        T = pd.DataFrame({"Candidate": [self.short(x, 60) for x in top["feature"]], "Theme": top["theme"],
                          "Effect per unit": [f"{v:+.3f}" for v in top["coef"]], "t": [f"{v:.1f}" for v in top["t_hac"]]})
        self.add("4. Choosing drivers: screen first", f"{int(A['passed'].sum()):,} of {len(A):,} candidates pass the "
                 "one-at-a-time screen", "The first filter: does the candidate help explain growth on its own, with the right sign?",
                 what=["Each candidate is added, alone, to a base equation: target growth = constant + last quarter's "
                       "growth + dummy quarters + candidate.",
                       "It passes when its coefficient is significant (p < 0.05, using Newey-West standard errors "
                       "that allow for autocorrelation) and has the expected sign."],
                 why=["Testing candidates on top of last quarter's growth asks the right question: does this variable "
                      "add information beyond the target's own momentum?"],
                 result=[f"{n_sig:,} candidates are significant; {n_wrong:,} of them have the wrong sign and are "
                         f"rejected; {int(A['passed'].sum()):,} pass.",
                         f"The strongest single candidate: {self.label(top['feature'].iloc[0])} (t = "
                         f"{top['t_hac'].iloc[0]:.1f})." if len(top) else ""],
                 read="Blue: candidates that pass; grey: rejected. Candidates beyond the dashed lines are significant.",
                 chart=self.save(f, p), table=T, table_note="The 8 strongest candidates that pass.")

    def s_bestform(self):
        A2 = self.a2
        if A2.empty:
            return
        A = self.a1
        ex = A2[A2["passed"]].iloc[0]["mnemonic"] if "passed" in A2 else A2.iloc[0]["mnemonic"]
        E = A[(A["mnemonic"] == ex) & A["passed"]].nlargest(6, "abs_t")
        T = pd.DataFrame({"Form and lag": [f"{tr_words(t)[0]}, lag {l}" for t, l in zip(E["transform"], E["lag"])],
                          "t": [f"{v:.1f}" for v in E["t_hac"]], "Kept": ["yes"] + ["no"] * (len(E) - 1)})
        A2 = A2[A2["passed"]] if "passed" in A2 else A2
        self.add("4. Choosing drivers: screen first", f"Only the best form and lag of each variable is kept: {len(A2)} variables",
                 "One variable, one representative: the selection compares variables, not versions of the same one.",
                 what=["For each variable, the form and lag with the highest |t| among those that pass is kept; the "
                       "others are dropped."],
                 why=["Two versions of the same variable (say, annual and quarterly growth of house prices) say almost "
                      "the same thing. Keeping both would crowd the shortlist with duplicates."],
                 result=[f"{len(A2)} variables remain. Example: for {self.meta.loc[ex, 'description'] if ex in self.meta.index else ex}, "
                         f"{len(E)} versions pass; the strongest is kept."],
                 table=T, table_note="Example: the passing versions of the strongest variable.")

    def s_clusters(self):
        A3 = self.a3
        if A3.empty:
            return
        ncl = A3["cluster"].nunique()
        big = A3.groupby("cluster").size().sort_values(ascending=False)
        c0 = big.index[0]
        E = A3[A3["cluster"] == c0].sort_values("abs_t", ascending=False).head(8)
        T = pd.DataFrame({"Variable": [self.short(x, 58) for x in E["feature"]], "Geography": E["geography"],
                          "t": [f"{v:.1f}" for v in E["t_hac"]],
                          "Representative": ["yes" if v else "" for v in E["is_representative"]]})
        thr = (self.cfg.get("screen_first") or {}).get("cluster_threshold", 0.75)
        self.add("4. Choosing drivers: screen first", f"Look-alike variables are grouped: {len(A3)} variables form {ncl} clusters",
                 "One representative per cluster, so the final search compares different economic forces.",
                 what=[f"Variables whose histories are correlated above {thr} (in absolute value) are put in the same "
                       "cluster, using hierarchical clustering.",
                       "Each cluster keeps one representative: the strongest, except that a series for the primary "
                       "geography is preferred when it is nearly as strong."],
                 why=["House prices in ten Ontario cities move together. They are not ten separate forces, and putting "
                      "two of them in one equation would make both coefficients unreliable (collinearity)."],
                 result=[f"The largest cluster has {big.iloc[0]} members; {int((big == 1).sum())} variables are alone in their cluster."],
                 table=T, table_note=f"The largest cluster (first {len(E)} members).")

    def s_shortlist(self):
        A4 = self.a4
        if A4.empty:
            return
        T = pd.DataFrame({"Variable": [self.short(x, 60) for x in A4["feature"]], "Theme": A4["theme"],
                          "t": [f"{v:.1f}" for v in A4["t_hac"]]}).head(25)
        self.add("4. Choosing drivers: screen first", f"The {len(A4)} strongest representatives form the shortlist",
                 "The shortlist is what the final search combines.",
                 what=["Cluster representatives are ranked by |t| and the strongest are kept (the shortlist size is a setting)."],
                 why=["The final search tries every combination, so the number of combinations grows fast with the "
                      "shortlist; 25 candidates already give thousands of combinations of up to 3 drivers."],
                 result=[f"Themes on the shortlist: " + ", ".join(f"{k} ({v})" for k, v in A4["theme"].value_counts().items()) + "."],
                 table=T)

    # ---------------------------------------------------------------------------------------------- part 5: approach B
    def s_en(self):
        B1 = self.b1
        if B1.empty:
            return
        lam = B1["lambda"].iloc[0] if "lambda" in B1 else None
        enc = self.cfg.get("en_first") or {}
        T = pd.DataFrame({"Selected": [self.short(x, 62) for x in B1["feature"]],
                          "Standardised effect": [f"{v:+.3f}" for v in B1["std_coef"]]})
        self.add("5. Choosing drivers: Elastic Net first", f"Elastic Net considers all {len(self.a1):,} candidates at once "
                 f"and keeps {len(B1)}", "The second approach looks at all candidates together instead of one at a time.",
                 what=["An Elastic Net regression puts every candidate in one equation and shrinks their coefficients "
                       "toward zero; only candidates that really add information keep a non-zero coefficient.",
                       "Last quarter's growth and the dummies are not penalised: their effect is removed first, so the "
                       "candidates compete only on what they add.",
                       "Coefficients are constrained to the expected sign, and the penalty is set so that about "
                       f"{enc.get('target_n', 10)} candidates remain."],
                 why=["Screening one at a time can keep a variable that only looks strong because it stands in for "
                      "another. Looking at all candidates jointly is the classic answer; having two independent "
                      "approaches also shows how robust the choice of drivers is."],
                 result=[f"On the full sample, {len(B1)} candidates keep a non-zero coefficient"
                         + (f" (penalty λ = {lam:.3g})." if lam is not None else ".")],
                 table=T)

    def s_stability(self):
        B2 = self.b2
        if B2.empty:
            return
        enc = self.cfg.get("en_first") or {}
        thr = float((enc.get("stability") or {}).get("threshold", 0.6)) if isinstance(enc.get("stability"), dict) else 0.6
        top = B2.nlargest(15, "selection_frequency").iloc[::-1]
        f, ax, p = self.fig("stability", 9, 5)
        ax.barh([self.short(x, 50) for x in top["feature"]], top["selection_frequency"] * 100,
                color=[C["teal"] if v >= thr else C["line"] for v in top["selection_frequency"]])
        ax.axvline(thr * 100, color=C["red"], ls="--", lw=1)
        ax.set_xlim(0, 100)
        ax.set_title("How often each candidate is selected across subsamples (%)", loc="left")
        ax.tick_params(axis="y", labelsize=8.5)
        n_st = int((B2["selection_frequency"] >= thr).sum())
        G = self.b3
        self.add("5. Choosing drivers: Elastic Net first", f"{n_st} candidates are selected in at least {thr:.0%} of "
                 "100 subsamples", "Stability selection keeps only the picks that do not depend on a few particular quarters.",
                 what=["The Elastic Net is re-run on 100 subsamples, each using 75% of the quarters drawn in blocks of "
                       "consecutive quarters (so the time pattern is kept).",
                       f"A candidate is kept when it is selected in at least {thr:.0%} of the subsamples. The survivors "
                       "are grouped by correlation, one per group, as in the first approach."],
                 why=["A single Elastic Net run can pick a variable because of one unusual episode. A variable that is "
                      "picked again and again, whatever quarters are left out, reflects a durable relation."],
                 result=[f"{n_st} stable candidates, which form {G['cluster'].nunique() if not G.empty and 'cluster' in G else 'n/a'} "
                         "groups after the correlation check; their representatives go to the final search."],
                 read="Blue bars cross the dashed threshold and survive.",
                 chart=self.save(f, p))

    def s_bayes(self):
        C2 = self.c2
        if C2.empty:
            return
        bc = self.cfg.get("bayes_first") or {}
        thr = float(bc.get("threshold", 0.5))
        top = C2.nlargest(15, "inclusion_probability").iloc[::-1]
        f, ax, p = self.fig("bma_inclusion", 9, 5)
        ax.barh([self.short(x, 50) for x in top["feature"]], top["inclusion_probability"] * 100,
                color=[C["teal"] if v >= thr else C["line"] for v in top["inclusion_probability"]])
        ax.axvline(thr * 100, color=C["red"], ls="--", lw=1)
        ax.set_xlim(0, 100)
        ax.set_title("Inclusion probability of each candidate (%)", loc="left")
        ax.tick_params(axis="y", labelsize=8.5)
        m = self.c1.iloc[0] if not self.c1.empty else {}
        n_in = int((C2["inclusion_probability"] >= thr).sum())
        T = None
        if not self.c3.empty:
            t3 = self.c3.head(6)
            T = pd.DataFrame({"Model (drivers)": [self.short(x, 90) for x in t3["drivers"]],
                              "Probability": [f"{v:.0%}" for v in t3["prob"]]})
        self.add("5. Choosing drivers: Bayesian first", f"{n_in} candidates appear in at least {thr:.0%} of the "
                 "probability-weighted models", "The third approach weighs every combination of drivers instead of "
                 "choosing one.",
                 what=["Every combination of candidates (up to a maximum size) is a possible model. Each gets a "
                       "probability: how well it explains growth, against a prior that prefers small models.",
                       "Models in which a driver has the wrong sign get no probability. Near-identical candidates (the "
                       "same series for several regions) are grouped first, so they do not split the probability.",
                       "A candidate's inclusion probability is the total probability of the models that contain it. "
                       f"Those with at least {thr:.0%} go to the final search; the most probable models are also "
                       "averaged into a forecast of their own ('Bayesian model average')."],
                 why=["Choosing one model ignores that several are almost as good. Averaging over them, weighted by "
                      "how probable each is, is the textbook answer to model uncertainty (Fernandez, Ley & Steel 2001; "
                      "the IMF uses a sign-constrained version in its stress tests)."],
                 result=[f"{int(m.get('candidates', 0))} candidates; {int(m.get('models_visited', 0)):,} models "
                         f"visited; on average {float(m.get('posterior_mean_model_size', 0)):.1f} drivers per model.",
                         f"{n_in} candidates reach {thr:.0%}; the top model alone has "
                         + (f"{self.c3['prob'].iloc[0]:.0%}" if not self.c3.empty else "n/a") + " of the probability."],
                 read="Blue bars cross the dashed threshold and go to the final search.",
                 chart=self.save(f, p), table=T, table_note="The most probable models.")

    # ---------------------------------------------------------------------------------------------- part 6: search
    def s_search_rules(self):
        T = self.a6
        if T.empty:
            return
        rows = T.set_index("step")["combinations"]
        tot = int(rows.get("combinations", 0))
        rem = [(k, int(v)) for k, v in rows.items() if k not in ("combinations", "passed every rule",
                                                                  "backtested (best by BIC)",
                                                                  "within the tolerance of the best error")]
        f, ax, p = self.fig("rules", 9, 4.2)
        names = ["All combinations"] + [f"− {k}" for k, _ in rem] + ["Pass every rule"]
        vals, bottoms, cur = [tot], [0], tot
        for _, v in rem:
            cur -= v
            vals.append(v)
            bottoms.append(cur)
        vals.append(cur)
        bottoms.append(0)
        cols = [C["navy"]] + [C["red"]] * len(rem) + [C["green"]]
        ax.bar(range(len(vals)), vals, bottom=bottoms, color=cols)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            ax.text(i, b + v, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([_wrapl(n, 14) for n in names], fontsize=8)
        ax.set_title("Screen-first final search: combinations removed by each rule", loc="left")
        fm = self.cfg.get("final_model") or {}
        self.add("6. The final search", f"{tot:,} combinations tried; {int(rows.get('passed every rule', 0)):,} pass every rule",
                 "Each rule removes models that would be hard to defend, before accuracy is even looked at.",
                 what=[f"Every combination of {fm.get('min_vars', 1)} to {fm.get('max_vars', 3)} shortlisted drivers is "
                       "estimated in the full equation (constant, last quarter's growth, dummies, drivers).",
                       "A combination is kept only if: no two drivers come from the same theme; every driver is "
                       f"significant (|t| ≥ {fm.get('t_min', 1.96)}); every sign is as expected; and the drivers are "
                       f"not near-copies of each other (VIF ≤ {fm.get('vif_max', 5)})."],
                 why=["One driver per theme keeps the model readable (one housing driver, one labour driver...). The "
                      "significance and sign rules make each driver defensible on its own. The VIF rule avoids "
                      "unstable coefficients."],
                 result=[f"{k}: {v:,} combinations removed." for k, v in rem if v] +
                        [f"{int(rows.get('passed every rule', 0)):,} combinations pass every rule."],
                 read="Red bars: combinations removed by each rule, in the order the rules are applied.",
                 chart=self.save(f, p))

    def s_search_rank(self):
        S = self.a5
        if S.empty or "oos_mape_pct" not in S:
            return
        f, ax, p = self.fig("search_scatter", 9, 4.4)
        ks = sorted(S["k"].unique())
        for k, col in zip(ks, (C["sky"], C["teal"], C["navy"], C["purple"])):
            s = S[S["k"] == k]
            ax.scatter(s["bic"], s["oos_mape_pct"], s=14, color=col, label=f"{k} driver{'s' if k > 1 else ''}", alpha=0.8)
        sel = S[S["selected"]]
        ax.scatter(sel["bic"], sel["oos_mape_pct"], s=120, facecolor="none", edgecolor=C["gold"], lw=2.2, label="chosen")
        ax.set_xlabel("BIC (in-sample fit; lower is better)")
        ax.set_ylabel(f"Backtest error 1-{self.H}Q ({self.EU or 'units'})")
        ax.legend()
        ax.set_title("The best 300 by BIC, then ranked by backtest error", loc="left")
        best = S.nsmallest(1, "oos_mape_pct").iloc[0]
        tol = (self.cfg.get("final_model") or {}).get("selection_tolerance", 0.05)
        top = S.nsmallest(8, "oos_mape_pct")
        T = pd.DataFrame({"Drivers": ["; ".join(self.short(d, 40) for d in str(x).split(", ")) for x in top["features"]],
                          "Error": [self.err(v) for v in top["oos_mape_pct"]], "Weakest |t|": [f"{v:.1f}" for v in top["min_abs_t"]],
                          "Chosen": ["yes" if v else "" for v in top["selected"]]})
        corr = S[["bic", "oos_mape_pct"]].corr().iloc[0, 1]
        self.add("6. The final search", "Surviving models are ranked by forecast accuracy, not by fit",
                 "A model that fits the past well is not necessarily the one that forecasts best.",
                 what=["The survivors are sorted by BIC (fit penalised for the number of drivers) and the best 300 go on.",
                       "Each of these is then forecast from every backtest origin, 1 to "
                       f"{self.H} quarters ahead, re-estimated each time on the data available then, and compared "
                       "with the actual balance. The score is the average absolute error of the forecast level."],
                 why=["Fit and forecast accuracy disagree more often than one might think: the correlation between BIC "
                      f"and backtest error across these 300 models is only {corr:.2f}."],
                 result=[f"The most accurate combination has an average error of {self.err(best['oos_mape_pct'])}."],
                 read="Each dot is a model. Further left: better fit; lower: more accurate forecasts. Gold ring: the chosen model.",
                 chart=self.save(f, p), table=T, table_note="The 8 most accurate combinations.")
        self._tol = tol

    def s_search_pick(self):
        rows = []
        for name, S, R in (("Screen first", self.a5, self.a6), ("Elastic Net first", self.b4, self.b5),
                           ("Bayesian first", self.c5, self.c6)):
            if S.empty:
                continue
            sel = S[S["selected"]]
            if not len(sel):
                continue
            r = sel.iloc[0]
            best = S["oos_mape_pct"].min()
            rows.append({"Approach": name, "Chosen drivers": "; ".join(self.short(d, 70) for d in str(r["features"]).split(", ")),
                         "Error": self.err(r["oos_mape_pct"]), "Best error": self.err(best),
                         "Within tolerance": str(self._rule(R, "within the tolerance of the best error") or "")})
        if not rows:
            return
        tol = (self.cfg.get("final_model") or {}).get("selection_tolerance", 0.05)
        T = pd.DataFrame(rows)
        same = len(rows) == 2 and rows[0]["Chosen drivers"] == rows[1]["Chosen drivers"]
        self.add("6. The final search", "The simplest model close to the best is chosen",
                 "Among near-equals, fewer drivers win: they are easier to explain and less likely to be overfitted.",
                 what=[f"All models within {tol:.0%} of the best backtest error are considered equivalent. Among them, "
                       "the one with the fewest drivers wins; on a tie, the one whose weakest driver is most significant."],
                 why=["A difference of a few percent in backtest error is within noise; an extra driver is a real cost "
                      "(one more thing to explain, one more input that can go wrong in a scenario)."],
                 result=[f"{r['Approach']}: {r['Chosen drivers']} (error {r['Error']}; "
                         f"{plural(int(r['Within tolerance'] or 0), 'model')} within the tolerance)." for r in rows] +
                        (["Both approaches arrive at the same drivers, which is strong evidence for them."] if same else
                         ["The two approaches choose different drivers: both models go on to the competition."]),
                 table=T)

    # ---------------------------------------------------------------------------------------------- part 7: competition
    def s_families(self):
        M = self.mlist.copy()
        fams = [f for f in FAM_ORDER if f in set(M["family"])] + sorted(set(M["family"]) - set(FAM_ORDER))
        DESC = {"Benchmark": "Naive rules any model must beat: last quarter's growth carried forward (AR(1)), or average growth.",
                "Data-driven": "The two models chosen in the previous steps.",
                "Literature": "Published specifications rebuilt with our series; drivers and lags fixed by the source.",
                "Hybrid": "Theory and data combined: the literature core plus data-chosen drivers, or averages of the two.",
                "Auto ARDL": "Lag lengths chosen automatically by information criterion (as in EViews' automatic ARDL).",
                "Same growth as a driver": "The target simply grows like one driver (e.g. like income), a common rule of thumb.",
                "Horizon combination": "A different model for each block of quarters ahead, chosen on past accuracy.",
                "Regime switching": "Relations that change with the cycle: Markov switching, thresholds, time-varying coefficients.",
                "Statistical time series": "ARIMAX and Theta: time-series models with or without drivers.",
                "Machine learning": "Ridge, random forest, gradient boosting, neural network on the same drivers.",
                "Deep learning": "Neural forecasting models (TiDE, N-HiTS, TFT).",
                "AI foundation model": "Pre-trained time-series models (e.g. Chronos) used without training on our data."}
        T = pd.DataFrame([{"Family": f, "Models": int((M["family"] == f).sum()), "What they are": DESC.get(f, "")} for f in fams])
        self.add("7. The competition", f"{len(M)} models from {len(fams)} families compete on equal terms",
                 "The chosen equation is only recommended if it holds its own against every other approach.",
                 what=["Every family is forecast in exactly the same way: same origins, same horizons, same data available "
                       "at each origin, same error measure."],
                 why=["A model can only be called good relative to the alternatives. Benchmarks show whether the drivers "
                      "add anything at all; machine learning and regime models show whether a more flexible form would "
                      "be worth its loss of transparency."],
                 table=T)

    def s_backtest_design(self):
        s = self.cfg["sample"]
        o = sorted(self.btf["origin"].unique()) if not self.btf.empty else []
        step, every = int(s.get("backtest_step", 1)), int(s.get("reestimate_every", 1))
        f, ax, p = self.fig("backtest_design", 9, 3.6)
        ax.grid(False)
        idx = self.ctx.idx
        if o:
            show = o[:: max(1, len(o) // 8)][:8]
            for i, org in enumerate(show):
                po = pd.Period(org, freq="Q")
                ax.barh(i, (po - idx[0]).n, left=0, color=C["sky"], height=0.55)
                ax.barh(i, self.H, left=(po - idx[0]).n, color=C["teal"], height=0.55)
                ax.text((po - idx[0]).n + self.H + 0.5, i, f"origin {org}", va="center", fontsize=8, color=C["mid"])
            ax.set_yticks([])
            ticks = list(range(0, len(idx), 8))
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(idx[t]) for t in ticks], fontsize=8)
            ax.set_title("Each origin: estimate on the light part, forecast the dark part, compare with actuals", loc="left")
        dq = s.get("dummy_quarters", [])
        self.add("8. Testing", f"Every model is tested from {len(o)} forecast origins, 1 to {self.H} quarters ahead",
                 "The backtest replays history: at each origin, models only see the data available at that time.",
                 what=[f"Starting at {s['backtest_start']}, the process pretends to stand at each origin "
                       + ("quarter" if step == 1 else f"(every {step} quarters)") + ", estimates every model on the data up to "
                       f"that quarter, forecasts the next {self.H} quarters, and compares the forecast level with the actual one.",
                       "Re-estimation: " + ("at every origin." if every == 1 else f"every {every} quarters; in between, "
                                            "coefficients are kept, as in production."),
                       "Scenario drivers are set to their actual values over the forecast period, so the test measures "
                       "the model, not the scenario."],
                 why=["This is the only honest test of a forecasting model: accuracy on data it has not seen. The error "
                      "is measured on the level (what users care about), so mistakes in growth add up over the quarters "
                      "exactly as they would in a real forecast."],
                 result=[f"Origins: {o[0]} to {o[-1]} ({len(o)})." if o else "",
                         f"Forecast windows that include a dummy quarter ({', '.join(dq)}) are left out of the error, "
                         "because no model could have foreseen the pandemic." if dq else ""],
                 numbers=[("Origins", str(len(o))), ("Horizon", f"1-{self.H}Q")],
                 chart=self.save(f, p) if o else None)

    def s_one_origin(self):
        B = self.btf
        if B.empty:
            return
        rec = self.REC["model"]
        b = B[B["model"].isin([rec, "AR(1) benchmark"]) & ~B["covid_window"]]
        if b.empty:
            return
        org = sorted(b[b["model"] == rec]["origin"].unique())
        full = [o for o in org if b[(b["model"] == rec) & (b["origin"] == o)]["horizon_q"].max() >= self.H]
        org = (full or org)[len(full or org) // 2]
        f, ax, p = self.fig("one_origin", 9, 4.2)
        L = self.level
        po = pd.Period(org, freq="Q")
        h0 = L.loc[po - 8:po + self.H]
        ax.plot(h0.index.to_timestamp(), h0.values / self.SCALE, color=C["ink"], lw=2, label="actual")
        for m, col in ((rec, C["gold"]), ("AR(1) benchmark", C["grey"])):
            s = b[(b["model"] == m) & (b["origin"] == org)].sort_values("horizon_q")
            if len(s):
                xs = [po.to_timestamp()] + [pd.Period(d, freq="Q").to_timestamp() for d in s["date"]]
                ys = [s["balance_at_origin"].iloc[0] / self.SCALE] + list(s["forecast_balance"] / self.SCALE)
                ax.plot(xs, ys, color=col, lw=2, ls="--", marker="o", ms=3, label=m)
        ax.axvline(po.to_timestamp(), color=C["mid"], lw=0.8)
        ax.legend()
        ax.set_title(f"Forecast made at {org} ({self.ulab})", loc="left")
        s = b[(b["model"] == rec) & (b["origin"] == org)].sort_values("horizon_q")
        a = b[(b["model"] == "AR(1) benchmark") & (b["origin"] == org)].sort_values("horizon_q")
        T = s[s["horizon_q"].isin(self.HZ)][["horizon_q", "date", "forecast_balance", "actual_balance_organic", "error_pct"]].copy()
        T.columns = ["Quarters ahead", "Quarter", "Forecast", "Actual", "Error " + (self.EU or self.UNITS)]
        T["Forecast"] = T["Forecast"].map(self.u)
        T["Actual"] = T["Actual"].map(self.u)
        T[T.columns[-1]] = T[T.columns[-1]].map(lambda v: f"{v:+.2f}" if self.EU else f"{v:+.3f}")
        self.add("8. Testing", f"One origin up close: the forecast made at {org}",
                 "Every number in the accuracy tables is built from forecasts like this one.",
                 what=[f"At {org}, {rec} is estimated on data up to {org}, then forecasts growth quarter by quarter; "
                       "each quarter's growth is added to the previous level to give the forecast level.",
                       ("The error at each horizon is the percentage gap between forecast and actual level." if self.EU
                        else f"The error at each horizon is the gap between forecast and actual level, in {self.UNITS}.")],
                 result=[f"{rec}: error {s[s['horizon_q'] == s['horizon_q'].max()]['error_pct'].iloc[0]:+.3f}{self.EU} after "
                         f"{int(s['horizon_q'].max())} quarters" + (f"; AR(1) benchmark: {a[a['horizon_q'] == a['horizon_q'].max()]['error_pct'].iloc[0]:+.3f}{self.EU}." if len(a) else ".")],
                 read=f"Black: actual {self.TARGET}. Dashed: forecasts from the origin (vertical line).",
                 chart=self.save(f, p), table=T)

    def s_all_origins(self):
        B = self.btf
        if B.empty:
            return
        rec = self.REC["model"]
        b = B[B["model"] == rec]
        if b.empty:
            return
        f, ax, p = self.fig("all_origins", 9, 4.2)
        L = self.level.loc[pd.Period(b["origin"].min(), freq="Q") - 6:]
        ax.plot(L.index.to_timestamp(), L.values / self.SCALE, color=C["ink"], lw=2.2, label="actual", zorder=5)
        for org, s in b.groupby("origin"):
            s = s.sort_values("horizon_q")
            po = pd.Period(org, freq="Q")
            xs = [po.to_timestamp()] + [pd.Period(d, freq="Q").to_timestamp() for d in s["date"]]
            ys = [s["balance_at_origin"].iloc[0] / self.SCALE] + list(s["forecast_balance"] / self.SCALE)
            ax.plot(xs, ys, color=C["red"] if s["covid_window"].any() else C["teal"], lw=1, alpha=0.6)
        ax.set_title(f"{rec}: every backtest forecast ({self.ulab}); red: windows with a dummy quarter (not scored)", loc="left")
        e = b[~b["covid_window"]]
        bias = e.groupby("horizon_q")["error_pct"].mean()
        self.add("8. Testing", "All the backtest forecasts of the recommended model, at once",
                 "The spread of the lines around the actual series shows how much to trust a forecast at each horizon.",
                 what=[f"Each thin line is one forecast of {rec}, made at one origin and running up to {self.H} quarters ahead."],
                 result=[f"Average error by horizon (scored windows): " +
                         ", ".join(f"{h}Q {self.err(e[e['horizon_q'] == h]['error_pct'].abs().mean())}" for h in self.HZ
                                   if h in set(e["horizon_q"])) + ".",
                         f"Average bias at {self.HM} quarters: {bias.get(self.HM, np.nan):+.3f}{self.EU} "
                         f"({'forecasts tend to be too high' if bias.get(self.HM, 0) > 0 else 'forecasts tend to be too low'})."],
                 read="A good model's lines stay close to the black line and do not all err in the same direction.",
                 chart=self.save(f, p))

    def s_twins(self):
        T = self.read("comparison", "10_tested_twins.csv")
        if T.empty:
            return
        added = T[T["outcome"] == "twin added"]
        if added.empty:
            return
        src = self.rank.set_index("model")
        col = f"avg_1_{self.HM}_pct"
        rows = []
        for _, r in added.iterrows():
            if r["model"] in src.index and r["twin"] in src.index:
                rows.append({"model": r["model"], "original": src.loc[r["model"], col], "tested": src.loc[r["twin"], col]})
        R = pd.DataFrame(rows)
        res = []
        if len(R):
            R["better"] = R["tested"] < R["original"]
            res = [f"The tested twin is more accurate for {int(R['better'].sum())} of {len(R)} models ({self.basis}).",
                   "Largest gains: " + ", ".join(f"{r['model']} ({self.err(r['original'])} -> {self.err(r['tested'])})"
                                                 for _, r in R.assign(g=R['tested'] / R['original']).sort_values('g')
                                                 .head(3).iterrows()) + "."]
        Tt = pd.DataFrame({"Model": added["model"], "Drivers kept after the tests": [
            ", ".join(self.short(d, 34) for d in str(x).split(", ") if "|" in d) for x in added["tested_drivers"]]})
        if len(R):
            Tt["Original error"] = added["model"].map(dict(zip(R["model"], R["original"].map(self.err)))).fillna("")
            Tt["Tested error"] = added["model"].map(dict(zip(R["model"], R["tested"].map(self.err)))).fillna("")
        self.add("7. The competition", f"{len(added)} models also run as a tested twin",
                 "Every model with drivers competes twice: as specified, and with only drivers that pass the tests.",
                 what=["The data-driven approaches keep only drivers that pass the tests (|t| at least 1.96, the "
                       "expected sign, VIF at most 5, one per theme). Literature, Auto-ARDL, regime, ARIMAX and "
                       "machine-learning models keep their drivers whatever the tests say.",
                       "Each of them also runs as a '(tested)' twin: starting from its own drivers, the weakest failing "
                       "driver is tried at its other lags and dropped if no lag passes, until every driver passes. The "
                       "twin keeps the model's form."],
                 why=["A driver that the data cannot pin down adds noise to the forecast and an effect nobody can "
                      "defend; but theory or a flexible model may still use it well. Running both lets the tests "
                      "decide which matters more for each model."],
                 result=res, table=Tt)

    def s_leaderboard(self):
        base = self.comp.copy()
        col = f"avg_1_{self.HM}_pct"
        t = base.sort_values(col).copy()
        if len(t) > 20:                                # the best 19 and the benchmark
            t = pd.concat([t.head(19), t[(t["model"] == "AR(1) benchmark") & ~t["model"].isin(t.head(19)["model"])]])
        f, ax, p = self.fig("leaderboard", 9, 6)
        t = t.iloc[::-1]
        ax.barh(t["model"], t[col], color=[FAM_COL.get(x, C["teal"]) for x in t["family"]])
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_title(f"Backtest: average error over 1-{self.HM} quarters ({self.EU or 'units'})", loc="left")
        rec = self.REC["model"]
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == rec:
                lbl.set_fontweight("bold")
        hand = [plt.Rectangle((0, 0), 1, 1, color=FAM_COL.get(fm, C["teal"])) for fm in FAM_ORDER if fm in set(t["family"])]
        ax.legend(hand, [fm for fm in FAM_ORDER if fm in set(t["family"])], fontsize=7.5, loc="lower right")
        top = base.sort_values(col).head(10)
        T = pd.DataFrame({"Model": top["model"], "Family": top["family"],
                          **{f"1-{h}Q": [self.err(v) for v in top[f"avg_1_{h}_pct"]] for h in self.HZ if f"avg_1_{h}_pct" in top}})
        ar = base.set_index("model")[col].get("AR(1) benchmark", np.nan)
        nb = int((base[col] < ar).sum()) if ar == ar else None
        self.add("8. Testing", f"Backtest leaderboard: {base.sort_values(col).iloc[0]['model']} is the most accurate",
                 "With drivers chosen on all the data; the out-of-sample test that follows is the stricter one.",
                 what=["Each model's score is its average absolute error of the level over 1 to "
                       f"{self.HM} quarters ahead, across all scored origins."],
                 result=[f"{nb} of {len(base)} models beat the AR(1) benchmark ({self.err(ar)})." if nb is not None else "",
                         f"Best of each family: " + "; ".join(f"{r['model']} ({self.err(r[col])})"
                                                              for _, r in base.sort_values(col).groupby("family").head(1).iterrows()) + "."],
                 read="Bars coloured by family; shorter is better. Bold: the recommended model.",
                 chart=self.save(f, p), table=T)

    def s_by_horizon(self):
        H = self.oos_h if self.HAVE_OOS and not self.oos_h.empty else self.by_h
        if H.empty:
            return
        col = "MAPE_exCOVID_pct" if "MAPE_exCOVID_pct" in H else [c for c in H.columns if "mape" in c.lower()][0]
        rec = self.REC["model"]
        pick = [rec, "AR(1) benchmark"]
        rk = self.rank
        for fam in ("Literature", "Machine learning", "Hybrid"):
            r = rk[(rk["family"] == fam) & ~rk["model"].isin(pick)]
            if len(r):
                pick.append(r.iloc[0]["model"])
        f, ax, p = self.fig("by_horizon", 9, 4.2)
        cols = [C["gold"], C["grey"], C["teal"], C["purple"], C["green"]]
        for m, c in zip(pick, cols):
            s = H[H["model"] == m].sort_values("horizon_q")
            if len(s):
                ax.plot(s["horizon_q"], s[col], marker="o", ms=4, lw=2.4 if m == rec else 1.6, color=c, label=m)
        ax.set_xlabel("quarters ahead")
        ax.set_ylabel(f"average error ({self.EU or 'units'})")
        ax.legend(fontsize=8)
        ax.set_title(("Out of sample" if H is self.oos_h else "Backtest") + ": error at each horizon", loc="left")
        r = H[H["model"] == rec].set_index("horizon_q")[col]
        a = H[H["model"] == "AR(1) benchmark"].set_index("horizon_q")[col]
        gain = {h: (1 - r.get(h) / a.get(h)) * 100 for h in self.HZ if h in r.index and h in a.index}
        g = list(gain.values())
        widens = len(g) > 1 and g[-1] > g[0]
        self.add("8. Testing", "Errors grow with the horizon" + ("; the gap over the benchmark widens" if widens else
                                                                 " for every model"),
                 "Drivers matter most at longer horizons, where momentum alone fades." if widens else
                 "How the advantage over the benchmark changes with the horizon.",
                 what=["The same errors, now shown horizon by horizon rather than averaged."],
                 why=["One quarter ahead, last quarter's growth is a good guide and every model does well. Further out, "
                      "a model needs to know where the economy is heading: that is what the drivers bring."],
                 result=[f"{rec} vs the AR(1) benchmark: " + ", ".join(f"{g:.0f}% less error at {h}Q" for h, g in gain.items()) + "."
                         if gain else ""],
                 chart=self.save(f, p))

    def s_oos_design(self):
        if not self.HAVE_OOS:
            self.add("8. Testing", "The out-of-sample test was not run",
                     "Run run_oos.py to re-choose the drivers at every origin and get the stricter ranking.",
                     what=["In the backtest above, the drivers were chosen once using all the data, including the "
                           "quarters later used for testing. The out-of-sample test repeats the whole selection at every "
                           "origin with only the data available then."])
            return
        o = self.cfg["oos"]
        n = self.picks["origin"].nunique() if not self.picks.empty else 0
        self.add("8. Testing", f"Out of sample: the whole process is repeated at {n} origins",
                 "The strictest test: the drivers themselves are chosen without seeing the future.",
                 what=["The backtest above has one weakness: the drivers were chosen once, on all the data, so the "
                       "choice already knew which variables worked in the test period (look-ahead).",
                       f"The out-of-sample test removes it: at each of {n} origins"
                       + (f" (every {o.get('step', 4)} quarters)" if o.get("step") else "") + ", everything is redone "
                       "with data up to that origin only: transformations, screening, Elastic Net, the final search, "
                       "estimation and the forecast. Different origins can choose different drivers."],
                 why=["This is exactly what would have happened if the process had been used in real time. Its ranking "
                      "is the basis for the recommendation."],
                 numbers=[("Origins", str(n)), ("Basis", "re-selection")])

    def s_oos_gap(self):
        if not self.HAVE_OOS:
            return
        col = f"avg_1_{self.HM}_pct"
        m = self.comp[["model", "family", col]].merge(self.oos[["model", col]], on="model", suffixes=("_bt", "_oos"))
        m = m.sort_values(f"{col}_oos").head(16).iloc[::-1]
        f, ax, p = self.fig("oos_gap", 9, 5.8)
        for i, r in enumerate(m.itertuples()):
            ax.plot([getattr(r, f"{col}_bt"), getattr(r, f"{col}_oos")], [i, i], color=C["line"], lw=2.5, zorder=1)
        ax.scatter(m[f"{col}_bt"], range(len(m)), color=C["sky"], s=40, label="backtest (drivers chosen on all data)", zorder=2)
        ax.scatter(m[f"{col}_oos"], range(len(m)), color=[FAM_COL.get(x, C["teal"]) for x in m["family"]], s=50,
                   label="out of sample (re-chosen)", zorder=3)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m["model"], fontsize=8.5)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"Average error 1-{self.HM}Q: backtest vs out of sample", loc="left")
        top = self.rank.head(10)
        T = pd.DataFrame({"Model": top["model"], "Family": top["family"],
                          **{f"1-{h}Q": [self.err(v) for v in top[f"avg_1_{h}_pct"]] for h in self.HZ if f"avg_1_{h}_pct" in top}})
        dd = self.comp.set_index("model")[col]
        gap = {r["model"]: r[col] - dd.get(r["model"], np.nan) for _, r in self.rank.iterrows()}
        dd_models = [x for x in ("Screen first", "Elastic Net first", "Bayesian first", "Bayesian model average")
                     if x in gap]
        self.add("8. Testing", f"Out of sample, {self.rank.iloc[0]['model']} is the most accurate",
                 "Models whose drivers were chosen from the data lose the most once the choice can no longer see the future.",
                 what=["The two scores of each model side by side: with drivers chosen on all the data (light dot) and "
                       "re-chosen at every origin (coloured dot)."],
                 why=["The gap measures how much a model benefited from hindsight in its choice of drivers. Models with "
                      "fixed drivers (literature, benchmarks) barely move; data-chosen models move most."],
                 result=[f"{x}: {self.err(dd.get(x))} in the backtest, {self.err(dd.get(x) + gap[x])} out of sample." for x in dd_models]
                        + [f"Recommended model ({self.REC['model']}): {self.err(self.REC[col])} out of sample."],
                 read="Each line joins a model's two scores; the longer the line, the more the backtest flattered it.",
                 chart=self.save(f, p), table=T, table_note="Out-of-sample ranking (top 10).")

    def s_driver_stability(self):
        P = self.picks
        if P.empty:
            return
        P = P[P["model"].isin(["Screen first", "Elastic Net first", "Bayesian first"])] if "model" in P else P
        fcol = "features" if "features" in P else [c for c in P.columns if "driver" in c or "feature" in c][0]
        rows = []
        for _, r in P.iterrows():
            for d in str(r[fcol]).split(", "):
                if "|" in d:
                    rows.append({"model": r.get("model", ""), "origin": r["origin"], "var": d.split("|")[0]})
        D = pd.DataFrame(rows)
        if D.empty:
            return
        n_or = P["origin"].nunique()
        cnt = D.groupby("var")["origin"].nunique().sort_values(ascending=False).head(14)
        f, ax, p = self.fig("driver_stability", 9, 4.6)
        c2 = cnt.iloc[::-1]
        lbl = [re.sub(r"\s*\(.*?\)", "", self.meta.loc[v, "description"])[:48] if v in self.meta.index else v for v in c2.index]
        ax.barh(lbl, c2.values / n_or * 100, color=C["teal"])
        ax.set_xlim(0, 100)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_title(f"Share of the {n_or} origins at which each variable is chosen (either approach, %)", loc="left")
        self.add("8. Testing", "Which variables keep being chosen, origin after origin",
                 "Drivers that come back at most origins reflect durable relations; one-off picks are noise.",
                 what=["At each out-of-sample origin, both approaches choose their drivers afresh. Counting how often "
                       "each variable is chosen shows how stable the choice is."],
                 result=[f"Most frequent: " + ", ".join(f"{(self.meta.loc[v, 'description'] if v in self.meta.index else v)} "
                                                       f"({c}/{n_or})" for v, c in cnt.head(3).items()) + "."],
                 read="Longer bars: chosen at more origins. The form and lag may differ from origin to origin.",
                 chart=self.save(f, p))

    def s_holdout(self):
        Hd = self.hold
        if Hd.empty:
            return
        n = self.cfg["oos"].get("holdout_quarters", 8)
        last = Hd.groupby("model", sort=False).tail(1).copy()
        last["abs"] = last["error_pct"].abs()
        last = last.sort_values("abs").head(10)
        T = pd.DataFrame({"Model": last["model"], "Error after the holdout": [f"{v:+.2f}%" for v in last["error_pct"]]})
        self.add("8. Testing", f"Holdout: the last {n} quarters were hidden from the whole process",
                 "A final check on the most recent period, which no choice and no estimate has seen.",
                 what=[f"Everything is redone with the last {n} quarters removed, then the forecast for those quarters is "
                       "compared with what happened."],
                 why=["The most recent period is the most relevant for the next forecast; a model that does well over "
                      "history but misses the latest quarters deserves caution."],
                 result=[f"Training ends at {Hd['train_end'].iloc[0]}." if "train_end" in Hd else ""],
                 table=T, table_note=f"Error of the level after {n} quarters (10 closest).")

    # ---------------------------------------------------------------------------------------------- part 9: the model
    def s_rec_model(self):
        rec = self.REC["model"]
        D = self.drv[self.drv["model"] == rec]
        col = f"avg_1_{self.HM}_pct"
        T = pd.DataFrame({"Term": [self._term(t) for t in D["term"]], "Coefficient": [f"{v:+.4f}" for v in D["coef"]],
                          "t": [f"{v:.1f}" if v == v else "" for v in D["t"]],
                          "Expected sign": [("+" if s > 0 else "-") if s == s and s else "" for s in D["expected_sign"]],
                          "Sign OK": [str(s) if isinstance(s, str) else "" for s in D["sign_ok"]]})
        eq = None
        rj = os.path.join(self.out, "formula", "recommended.json")
        if os.path.exists(rj):
            j = json.load(open(rj, encoding="utf-8"))
            if j.get("formula_model") == rec:
                eq = j.get("equation")
        ar = self.AR[col] if self.AR is not None else np.nan
        sc = self.SCJ
        rule = (f"Choice rule: every model competes; the one with the best balance of forecasting accuracy, "
                f"interpretation and sensitivity to the economy (model scorecard, previous page), {self.basis}."
                if sc else
                f"Choice rule: among single-equation models (data-driven, literature, hybrid) with 1 to {self.MAXV} "
                f"drivers, the one with the lowest average error over 1-{self.HM} quarters, {self.basis}.")
        self.add("9. The recommended model", f"Recommended: {rec}",
                 ("The best balance of accuracy, interpretation and sensitivity among all models, judged "
                  f"{self.basis}." if sc else
                  f"The most accurate model with at most {self.MAXV} drivers and a written equation, judged {self.basis}."),
                 what=[rule,
                       "Drivers: " + "; ".join(self.label(d) for d in self.rec_drivers) + "." if self.rec_drivers else ""],
                 why=["A single equation can be written down, checked by hand, put in a spreadsheet and explained to a "
                      "committee; a forest of trees cannot. When the most accurate model overall has no equation, the "
                      "gap between the two is reported so the cost of transparency is known."],
                 result=[f"Average error 1-{self.HM}Q: {self.err(self.REC[col])}; AR(1) benchmark: {self.err(ar)} "
                         f"({(1 - self.REC[col] / ar) * 100:.0f}% less error)." if ar == ar else "",
                         f"Most accurate model overall: {self.rank.iloc[0]['model']} ({self.err(self.rank.iloc[0][col])})."],
                 numbers=[("Error 1-%dQ" % self.HM, self.err(self.REC[col])),
                          ("vs AR(1)", f"{(1 - self.REC[col] / ar) * 100:.0f}% less" if ar == ar else "n/a"),
                          ("Drivers", str(len(self.rec_drivers)))],
                 table=T, table_note=("Equation: " + eq.replace("\n", ";  ")) if eq else "")

    def _term(self, t):
        if t == "const":
            return "Constant"
        if t == "y_lag1":
            return "Target growth last quarter"
        if t.startswith("D_"):
            return f"Dummy {t[2:]}"
        return self.short(t, 60)

    def s_rec_effects(self):
        rec = self.REC["model"]
        D = self.drv[(self.drv["model"] == rec) & self.drv["term"].isin(self.rec_drivers)]
        if D.empty:
            return
        f, ax, p = self.fig("effects", 9, 3.4)
        v = D["balance_effect_pct_8q"].fillna(0) if "balance_effect_pct_8q" in D else D["coef"]
        lbl = [_wrapl(self.short(t, 70), 30) for t in D["term"]]
        ax.barh(lbl, v, color=[C["green"] if x > 0 else C["red"] for x in v])
        ax.axvline(0, color=C["mid"], lw=0.8)
        ax.set_title(f"Effect on the {self.TARGET} after 8 quarters, per unit of each driver ({self.EU or self.UNITS})", loc="left")
        self.add("9. The recommended model", "What each driver does, in plain words",
                 "Each driver's effect has the sign theory expects and a size that can be checked.",
                 what=[str(r) for r in D["reading"]] if "reading" in D else [],
                 why=["These sentences are the explanation a user can give for any change in the forecast: when a "
                      "driver's scenario path changes, the forecast moves by these amounts."],
                 result=[f"{self.short(t, 60)}: {e:+.2f}{self.EU or ' ' + self.UNITS} on the {self.TARGET} after 8 quarters per unit."
                         for t, e in zip(D["term"], v)],
                 read=f"Green: raises the {self.TARGET}; red: lowers it. 'Unit' = 1 percentage point of the driver's form.",
                 chart=self.save(f, p))

    def s_rec_fit(self):
        if not self.rec_drivers:
            return
        from ..features.transforms import feature_matrix
        X = feature_matrix(self.ctx.base, self.rec_drivers).reindex(self.ctx.idx)
        y = self.ctx.y
        f, axs = plt.subplots(len(self.rec_drivers) + 1, 1, figsize=(9, 1.6 * (len(self.rec_drivers) + 1) + 0.6), sharex=True)
        p = os.path.join(self.charts, "drivers_history.png")
        x = self.ctx.idx.to_timestamp()
        axs[0].plot(x, (y * 100 if self.form == "dlog" else y).values, color=C["navy"], lw=1.6)
        axs[0].set_title(f"Target: quarterly {self.growth_word}", loc="left", fontsize=9.5)
        cor = []
        for ax, d, c in zip(axs[1:], self.rec_drivers, (C["teal"], C["gold"], C["green"], C["purple"])):
            ax.plot(x, X[d].values * (100 if parse(d)[1] not in ("diff", "diff2", "diff4", "lvl", "rgap") else 1), color=c, lw=1.6)
            ax.set_title(self.short(d, 80), loc="left", fontsize=9.5)
            cor.append(pd.concat([y, X[d]], axis=1).dropna().corr().iloc[0, 1])
        self.add("9. The recommended model", "The drivers next to the target",
                 "The drivers' swings line up with the target's: that is the relation the model exploits.",
                 what=["Top: the target's quarterly growth. Below: each driver, in the form and with the lag the model uses."],
                 result=[f"Correlation with the target: " + ", ".join(f"{self.short(d, 80)} {c:+.2f}"
                                                                     for d, c in zip(self.rec_drivers, cor)) + "."],
                 read="Lags are already applied: a peak in a driver lines up with the target peak it helps explain.",
                 chart=self.save(f, p))

    def s_coef_paths(self):
        P = self.paths
        rec = self.REC["model"]
        if P.empty or "model" not in P:
            return
        s = P[P["model"] == rec]
        cols = [d for d in self.rec_drivers if d in s]
        if s.empty or not cols:
            return
        f, ax, p = self.fig("coef_paths", 9, 3.8)
        x = [pd.Period(o, freq="Q").to_timestamp() for o in s["origin"]]
        for d, c in zip(cols, (C["teal"], C["gold"], C["green"], C["purple"])):
            v = s[d].values
            ax.plot(x, v / np.nanmean(np.abs(v)), color=c, lw=1.8, label=self.short(d, 50))
        ax.axhline(0, color=C["mid"], lw=0.8)
        ax.legend(fontsize=8)
        ax.set_title("Each coefficient re-estimated at every origin (relative to its average size)", loc="left")
        rng = {d: (s[d].min(), s[d].max()) for d in cols}
        flips = [d for d, (a, b) in rng.items() if a * b < 0]
        self.add("9. The recommended model", "The coefficients are stable as the sample grows" if not flips else
                 "Some coefficients move a lot as the sample grows",
                 "A relation that holds whatever the estimation window is one we can rely on in a forecast.",
                 what=["At every backtest origin the model is re-estimated; this chart follows each coefficient over "
                       "those re-estimations."],
                 result=[f"{self.short(d, 60)}: from {a:+.4f} to {b:+.4f}." for d, (a, b) in rng.items()] +
                        ([f"Sign changes over the sample: {', '.join(self.short(d, 40) for d in flips)}."] if flips else
                         ["No coefficient changes sign."]),
                 read="Lines near 1 throughout: the coefficient keeps its size. Crossing 0: it changes sign.",
                 chart=self.save(f, p))

    def s_diagnostics(self):
        rec = self.REC["model"]
        D = self.diag[self.diag["model"] == rec]
        if D.empty:
            return
        r = D.iloc[0]
        checks = [("Autocorrelation (Ljung-Box, 4 lags)", "Ljung-Box p (4)", "errors are not predictable from past errors"),
                  ("Autocorrelation (Ljung-Box, 8 lags)", "Ljung-Box p (8)", "same, over two years"),
                  ("Constant variance (Breusch-Pagan)", "Breusch-Pagan p", "errors are as large in calm and turbulent times"),
                  ("Normal errors (Jarque-Bera)", "Jarque-Bera p", "no extreme surprises beyond the normal"),
                  ("Stable relation (CUSUM)", "CUSUM p", "no break in the relation over the sample")]
        rows = []
        for name, c, meaning in checks:
            v = r.get(c)
            if v == v and v is not None:
                rows.append({"Check": name, "p-value": f"{v:.3f}", "Result": "pass" if v >= 0.05 else "flag",
                             "Meaning when it passes": meaning})
        T = pd.DataFrame(rows)
        npass = int((T["Result"] == "pass").sum()) if len(T) else 0
        flag = list(T.loc[T["Result"] == "flag", "Check"]) if len(T) else []
        self.add("9. The recommended model", f"The model passes {npass} of {len(T)} statistical checks",
                 "The errors behave like random noise: the model has not left a predictable pattern unexplained."
                 if not flag else f"Flagged: {', '.join(flag)}. The forecasts stay unbiased, but the standard errors "
                 "and any forecast intervals should be read with care.",
                 what=["Standard residual tests on the estimated equation. A p-value above 0.05 means the test finds no problem."],
                 why=["Predictable errors would mean the model is missing something (a lag, a driver, a break), and its "
                      "forecast intervals would be too narrow."],
                 result=[f"Fit: adjusted R² {r.get('adj_R2', np.nan):.2f} on {int(r.get('n_obs', 0))} quarters; largest "
                         f"VIF {r.get('max VIF', np.nan):.2f}; Durbin-Watson {r.get('Durbin-Watson', np.nan):.2f}."],
                 table=T)

    # ---------------------------------------------------------------------------------------------- part 10: forecasts
    def s_scen_inputs(self):
        if not self.rec_drivers:
            return
        from ..features.transforms import feature_matrix
        sc = self.ctx.macro
        hist_end = self.ctx.idx[self.ctx.end]
        f, axs = plt.subplots(1, len(self.rec_drivers), figsize=(9, 3.6))
        axs = np.atleast_1d(axs)
        p = os.path.join(self.charts, "scenario_inputs.png")
        for ax, d in zip(axs, self.rec_drivers):
            k = 100 if parse(d)[1] not in ("diff", "diff2", "diff4", "lvl", "rgap") else 1
            for (nm, df), c in zip(sc.items(), SCEN_COL):
                X = feature_matrix(df, [d])[d].loc[hist_end - 12:hist_end + self.H]
                ax.plot(X.index.to_timestamp(), X.values * k, color=c, lw=1.8, label=nm)
            ax.axvline(hist_end.to_timestamp(), color=C["mid"], lw=0.8)
            ax.set_title(_wrapl(self.short(d, 60), 34), loc="left", fontsize=9)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
        axs[0].legend(fontsize=7.5)
        self.add("10. The forecast", "The scenarios, seen through the model's drivers",
                 "The forecast differs across scenarios only through these paths.",
                 what=["Each scenario provides a future path for every macro series; the model only uses its drivers, "
                       "transformed and lagged exactly as in the estimation."],
                 result=[f"Scenarios: {', '.join(sc)}. The vertical line marks the last observed quarter ({hist_end})."],
                 chart=self.save(f, p))

    def s_scen_forecast(self):
        rec = self.REC["model"]
        S = self.scen[self.scen["model"] == rec]
        if S.empty:
            return
        f, ax, p = self.fig("scenario_forecast", 9, 4.4)
        L = self.level.loc[self.level.index[-16]:]
        ax.plot(L.index.to_timestamp(), L.values / self.SCALE, color=C["ink"], lw=2.2, label="history")
        last = (L.index[-1].to_timestamp(), L.iloc[-1] / self.SCALE)
        rows = []
        for (nm, s), c in zip(S.groupby("scenario", sort=False), SCEN_COL):
            s = s.sort_values("date")
            xs = [last[0]] + [pd.Period(d, freq="Q").to_timestamp() for d in s["date"]]
            ys = [last[1]] + list(s["balance_forecast"] / self.SCALE)
            ax.plot(xs, ys, color=c, lw=2, label=nm)
            r = {"Scenario": nm}
            for h in self.HZ:
                if len(s) >= h:
                    v = s["balance_forecast"].iloc[h - 1]
                    r[f"After {h}Q"] = (f"{self.u(v)} ({(v / self.level.iloc[-1] - 1) * 100:+.1f}%)" if self.form == "dlog"
                                        else f"{self.u(v)} ({v - self.level.iloc[-1]:+.2f})")
            rows.append(r)
        ax.legend()
        ax.set_title(f"{rec}: the {self.TARGET} under each scenario ({self.ulab})", loc="left")
        T = pd.DataFrame(rows)
        self.add("10. The forecast", f"The forecast under each scenario, {self.H} quarters ahead",
                 "The spread between the scenarios shows how much the forecast depends on the economy.",
                 what=["Starting from the last observed level, the model forecasts growth quarter by quarter under each "
                       "scenario: last quarter's growth (the model's own forecast after the first quarter) plus each "
                       "driver's scenario value times its coefficient."],
                 result=[f"After {self.HM} quarters: " + "; ".join(f"{r['Scenario']} {r.get(f'After {self.HM}Q', 'n/a')}"
                                                                   for r in rows) + "."],
                 read="In brackets: change from the last observed level" + (" (%)." if self.form == "dlog" else f" ({self.UNITS})."),
                 chart=self.save(f, p), table=T, table_note=f"Level in {self.ulab}; change from {self.level.index[-1]} in brackets.")

    def s_contrib(self):
        """Forecast growth taken apart: constant + persistence x last growth + coefficient x each driver
        (single equations whose terms are the constant, last quarter's growth, dummies and drivers)."""
        from ..features.transforms import feature_matrix
        rec = self.REC["model"]
        D = self.drv[self.drv["model"] == rec].set_index("term")["coef"]
        other = [t for t in D.index if t not in ("const", "y_lag1") and not t.startswith("D_") and "|" not in t]
        S = self.scen[self.scen["model"] == rec]
        if D.empty or other or S.empty:
            return
        base = self.cfg["data"]["macro"]["base_scenario"]
        s = S[S["scenario"] == base].sort_values("date").head(min(8, self.H))
        if s.empty:
            return
        drivers = [t for t in D.index if "|" in t]
        X = feature_matrix(self.ctx.macro[base], drivers)
        g = list(s["dlog_forecast"])
        prev = [float(self.ctx.y.iloc[self.ctx.end])] + g[:-1]
        q = [pd.Period(d, freq="Q") for d in s["date"]]
        parts = {"last quarter's growth": [D.get("y_lag1", 0.0) * v for v in prev]}
        for d in drivers:
            parts[self.short(d, 70)] = [D[d] * X[d].get(t, np.nan) for t in q]
        P = pd.DataFrame(parts, index=[str(t) for t in q])
        if "const" in D.index:
            c0 = float(D["const"])
        else:                                          # not in the table: the constant is what the pieces leave
            c0 = float(np.median(np.array(g) - P.sum(axis=1).values))
        P.insert(0, "constant", c0)
        k = 100 if self.form == "dlog" else 1
        check = (P.sum(axis=1) - np.array(g)).abs().max()
        f, ax, p = self.fig("contributions", 9, 4.8)
        x = np.arange(len(P))
        pos, neg = np.zeros(len(P)), np.zeros(len(P))
        for c, col in zip(P.columns, (C["grey"], C["sky"], C["teal"], C["gold"], C["green"], C["purple"])):
            v = P[c].values * k
            ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=col, label=c)
            ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=col)
            pos += np.where(v > 0, v, 0)
            neg += np.where(v < 0, v, 0)
        ax.plot(x, np.array(g) * k, color=C["ink"], marker="o", lw=1.5, label="forecast growth")
        ax.set_xticks(x)
        ax.set_xticklabels(P.index, fontsize=8)
        ax.axhline(0, color=C["mid"], lw=0.8)
        ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        ax.set_title(f"{base}: forecast quarterly {self.growth_word} taken apart" + (" (% points)" if k == 100 else ""), loc="left")
        r0 = P.iloc[0]
        T = (P * k).round(3)
        T.insert(0, "Quarter", P.index)
        T["Total"] = [f"{v * k:.3f}" for v in g]
        unit = "%" if k == 100 else ""
        self.add("10. The forecast", "Every forecast quarter, taken apart driver by driver",
                 "Nothing in the forecast is a black box: each quarter's growth is a sum of named pieces.",
                 what=["Forecast growth = constant + persistence x last quarter's growth + coefficient x each driver's "
                       "scenario value. Each bar segment is one of these products; after the first quarter, 'last "
                       "quarter's growth' is the model's own forecast."],
                 why=["This is how a change in the forecast is explained: when a scenario path changes, the driver's "
                      "segment changes by its coefficient times the change."],
                 result=[f"{P.index[0]}: {self.growth_word} {g[0] * k:+.2f}{unit} = " + " + ".join(f"{c} {v * k:+.2f}" for c, v in r0.items()) + "."]
                        + ([] if check < 1e-6 else [f"(Pieces differ from the model's forecast by up to {check * k:.3f}: rounding of inputs.)"]),
                 read="Bars above zero push growth up, below zero pull it down; the line is the total.",
                 chart=self.save(f, p), table=T, table_note=f"{base} scenario; values in {'% points' if k == 100 else 'units'}.")

    def s_scen_check(self):
        rec = self.REC["model"]
        Y = self.yearend[self.yearend["model"] == rec] if not self.yearend.empty else pd.DataFrame()
        if Y.empty:
            return
        sc = [c for c in Y.columns if c not in ("model", "date")]
        def rank(n):
            n = n.lower()
            return 0 if "severe" in n or "stress" in n else 1 if "down" in n or "adverse" in n else 2 if "base" in n else 3 if "up" in n else None
        order = [c for c in sorted(sc, key=lambda c: (rank(c) is None, rank(c))) if rank(c) is not None]
        last = Y.iloc[-1]
        ok = all(last[a] <= last[b] for a, b in zip(order, order[1:])) if len(order) > 1 else None
        T = Y.copy()
        for c in sc:
            T[c] = T[c].map(self.u)
        T = T.drop(columns="model").rename(columns={"date": "Year end"})
        self.add("10. The forecast", "Scenario ordering check: " + ("passed" if ok else "look closer" if ok is False else "not applicable"),
                 "Severe below downside below baseline below upside, as the scenario names promise.",
                 what=["The year-end levels of each scenario are compared. A sensible model ranks them in the order of "
                       "the scenario names (severe < downside < baseline < upside)."],
                 result=[("The order holds at the last year end." if ok else
                          "The order does not hold at the last year end: check the drivers' scenario paths.")
                         if ok is not None else "Scenario names without a severity keyword are not checked."],
                 table=T, table_note=f"Year-end level, {self.ulab}.")

    # ---------------------------------------------------------------------------------------------- part 11: closing
    def s_limits(self):
        rec = self.REC["model"]
        col = f"avg_1_{self.HM}_pct"
        best = self.rank.iloc[0]
        items = [("History is short", f"{len(self.ctx.idx[self.ctx.est0:self.ctx.end + 1])} quarters of estimation data. "
                  "Few recessions are in the sample, so effects in a severe downturn are extrapolated."),
                 ("Pandemic quarters", "Handled with dummies and excluded from scoring; a similar shock would not be "
                  "captured by the drivers."),
                 ("Scenario dependence", "The forecast is only as good as the scenario paths of the drivers."),
                 ("Driver choice can shift", "Out of sample, the chosen drivers change between origins; the "
                  "recommendation should be re-run when new data arrive.")]
        if best["model"] != rec:
            items.append(("Accuracy vs transparency", f"The most accurate model overall ({best['model']}, "
                          f"{self.err(best[col])}) is not the recommended one ({self.err(self.REC[col])}); the "
                          "difference is the price of an explainable equation."))
        T = pd.DataFrame(items, columns=["Limitation", "What it means and how it is managed"])
        self.add("12. Conclusion", "Known limitations and how they are managed",
                 "No model is perfect; these are the points to keep in mind when using the forecast.", table=T)

    def s_scorecard(self):
        S = self.SC
        if S.empty:
            return
        W = self.SCJ.get("weights") or {"accuracy": 0.5, "interpretation": 0.25, "sensitivity": 0.25}
        d = S.head(12).iloc[::-1]
        f, ax, p = self.fig("scorecard", 9, 0.42 * len(d) + 1.5)
        left = np.zeros(len(d))
        for k, c in (("accuracy", C["teal"]), ("interpretation", C["gold"]), ("sensitivity", C["green"])):
            v = d[k].values * W[k]
            ax.barh(range(len(d)), v, left=left, color=c, label=f"{k} ({W[k]:.0%})")
            left += v
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([("" if e else "(not eligible) ") + self.short(m, 44) for m, e in zip(d["model"], d["eligible"])],
                           fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Weighted score of each model", loc="left")
        T = pd.DataFrame({"Model": S["model"].head(8), "Error": [self.err(v) for v in S["error_pct"].head(8)],
                          "Accuracy": S["accuracy"].head(8).map("{:.2f}".format),
                          "Interpretation": S["interpretation"].head(8).map("{:.2f}".format),
                          "Sensitivity": S["sensitivity"].head(8).map("{:.2f}".format),
                          "Score": S["score"].head(8).map("{:.2f}".format),
                          "Eligible": ["yes" if e else "no" for e in S["eligible"].head(8)]})
        out_ = S[~S["eligible"]]
        top_out = out_[out_["score"] > S.iloc[0]["score"]]
        self.add("9. The recommended model", f"All {len(S)} models scored: {S.iloc[0]['model']} balances best",
                 "No family is preferred: every model competes on forecasting, interpretation and sensitivity.",
                 what=["Accuracy: the best average error divided by the model's (1 = the most accurate).",
                       "Interpretation: how transparent the model type is (a written equation scores 1, machine "
                       "learning 0.2), times the average of three checks: coefficients with the expected sign, at most "
                       f"{int((self.cfg.get('recommend') or {}).get('max_drivers', 5))} drivers, and drivers that are "
                       "not other outcomes of the economy (credit, deposits).",
                       "Sensitivity: does the model respond to the focus drivers, in the expected direction, and do its "
                       "scenarios move the same way as the explanatory model's?",
                       f"To be recommended, a model must be within {self.SCJ.get('max_error_ratio', 1.5):g}x the best "
                       "error and respond to at least one focus driver."],
                 why=["The most accurate model is not always the most useful: a model that cannot say how the economy "
                      "moves the target, or whose drivers cannot be explained, is hard to defend and cannot answer "
                      "what-if questions."],
                 result=[f"Recommended: {S.iloc[0]['model']} (score {S.iloc[0]['score']:.2f}); most accurate: "
                         f"{self.SCJ.get('most_accurate', '')}."] +
                        [f"{r['model']} scores higher but is not eligible: {r['not_eligible_because']}."
                         for _, r in top_out.head(2).iterrows()],
                 read=f"Weights: accuracy {W['accuracy']:.0%}, interpretation {W['interpretation']:.0%}, sensitivity "
                      f"{W['sensitivity']:.0%} (config recommend.weights).",
                 chart=self.save(f, p), table=T)

    def s_recommend(self):
        rec = self.REC["model"]
        col = f"avg_1_{self.HM}_pct"
        ar = self.AR[col] if self.AR is not None else np.nan
        self.add("12. Conclusion", f"Recommendation: use {rec}",
                 f"{self.err(self.REC[col])} average error over 1-{self.HM} quarters, "
                 + (f"{(1 - self.REC[col] / ar) * 100:.0f}% less than the benchmark, " if ar == ar else "")
                 + "with drivers that make economic sense.",
                 what=[f"Use {rec} for the {self.TARGET} forecast under each scenario.",
                       "Drivers: " + "; ".join(self.label(d) for d in self.rec_drivers) + "." if self.rec_drivers else "",
                       "Re-run the process each quarter when new data arrive; check whether the drivers and the ranking change.",
                       "Keep the runner-up models as challengers and report their forecasts alongside."],
                 kind="closing")

    # ---------------------------------------------------------------------------------------------- impact
    IMP = "11. How the economy affects the target"

    def _ex_name(self):
        r = self.i_resp
        names = [m for m in r["model"].unique() if str(m).startswith("Explanatory")
                 and not str(m).endswith("(tested)")] if not r.empty else []
        return names[0] if names else "Explanatory: focus drivers"

    def _hm(self):
        hs = sorted(self.i_sens["horizon"].unique()) if not self.i_sens.empty else []
        h = int((self.cfg.get("impact") or {}).get("compare_horizon", 8))
        return h if h in hs else (hs[-1] if hs else h)

    def _label(self, concept):
        r = self.i_rel[self.i_rel["concept"] == concept] if not self.i_rel.empty else self.i_rel
        return (r["label"].iloc[0], r["reasons"].iloc[0]) if len(r) else ("", "")

    def s_impact_headline(self):
        S = self.i_sens
        if S.empty:
            return
        h = self._hm()
        d = S[S["horizon"] == h].copy()
        d = d.reindex(d["typical_explanatory"].abs().sort_values().index)
        f, ax, p = self.fig("impact_sensitivity", 9, 0.7 * len(d) + 1.6)
        y = np.arange(len(d))
        ax.barh(y, d["typical_explanatory"], color=[C["green"] if v >= 0 else C["red"] for v in d["typical_explanatory"]],
                height=0.5)
        if "typical_band_lo" in d:
            ax.errorbar(d["typical_explanatory"], y, xerr=[d["typical_explanatory"] - d["typical_band_lo"],
                                                           d["typical_band_hi"] - d["typical_explanatory"]],
                        fmt="none", ecolor=C["ink"], capsize=3, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrapl(f"{r['description']} ({r['typical_shock']})", 38) for _, r in d.iterrows()],
                           fontsize=8.5)
        ax.axvline(0, color=C["mid"], lw=0.8)
        ax.set_title(f"Effect of a typical rise in each driver on the {self.TARGET} after {h} quarters "
                     f"({self.EU or self.UNITS}); bars = 90% band", loc="left")
        top = d.iloc[-1]
        T = pd.DataFrame({"Driver": d["description"].iloc[::-1].values,
                          "Typical rise": d["typical_shock"].iloc[::-1].values,
                          f"Effect after {h}q": [f"{v:+.2f}{self.EU}" for v in d["typical_explanatory"].iloc[::-1]],
                          "Can we trust it?": [self._label(c)[0] for c in d["concept"].iloc[::-1]]})
        from ..impact.analysis import sentence
        self.add("1. The question", f"How the economy affects the {self.TARGET}: {top['description']} matters most",
                 "The first question a user asks: if the economy moves, what happens to us, and when?",
                 what=[sentence(None, r, h, f"the {self.TARGET}") for _, r in d.iloc[::-1].iterrows()],
                 why=["These effects come from an explanatory model built for the purpose: the drivers of interest "
                      "are always in, and the variables held fixed around them are chosen so their effects are not "
                      "biased (double selection, chapter 5). The forecast competition (chapters 7-8) answers a "
                      "different question: what forecasts best.",
                      f"A 'typical rise' is one standard deviation of the driver's change over four quarters, so the "
                      "drivers can be compared with each other."],
                 result=[f"{r['description']}: {self._label(r['concept'])[0]}" + (
                     f" ({self._label(r['concept'])[1]})" if self._label(r['concept'])[0] != "Reliable" else "")
                     for _, r in d.iloc[::-1].iterrows()],
                 read="Bars: the explanatory model's estimate; whiskers: its 90% band. Chapter 11 has the details.",
                 chart=self.save(f, p), table=T)

    def _chosen(self, model):
        r = self.mlist[self.mlist["model"] == model] if not self.mlist.empty else self.mlist
        return [d for d in str(r["drivers"].iloc[0]).split(", ") if "|" in d] if len(r) else []

    def s_factors(self):
        E = self.e1
        if E.empty:
            return
        f, ax, p = self.fig("factors", 9, 0.38 * len(E) + 1.4)
        d = E.sort_values("members")
        ax.barh(d["mnemonic"].str.replace("FAC.", "", regex=False), d["members"], color=C["teal"], label="in the index")
        ax.barh(d["mnemonic"].str.replace("FAC.", "", regex=False), d["of"] - d["members"], left=d["members"],
                color=C["line"], label="not related to the target")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Series summarised by each theme index", loc="left")
        ch = self._chosen("Factors first")
        T = pd.DataFrame({"Index": E["mnemonic"], "Series": E["members"].astype(int).astype(str) + " of " +
                          E["of"].astype(int).astype(str), "Variance explained": E["variance_share"].map("{:.0%}".format),
                          "Expected sign": E["expected_sign"].map({1: "+", -1: "-", 0: "?"})})
        self.add("5. Choosing drivers: factors first (theme indexes)",
                 f"{len(E)} theme indexes summarise {int(E['members'].sum())} series",
                 "Instead of choosing one of many near-identical series, use all of them through one index per theme.",
                 what=["For each theme, the series related to the target (|t| at least 1.65 at their best lag) are "
                       "standardised and summarised by their first principal component (targeted predictors, Bai & "
                       "Ng 2008; diffusion indexes, Stock & Watson 2002).",
                       "The indexes are candidates for every approach, and 'Factors first' builds a model from them "
                       "only: each index at its best lag, then the lowest-BIC subset with the expected signs."],
                 why=["Macro data are dense: many series carry a little of the same information. An index uses all of "
                      "it, does not depend on which city happened to fit best, and stays easy to explain."],
                 result=[f"Factors first chose: {', '.join(self.label(c) for c in ch)}." if ch else ""],
                 chart=self.save(f, p), table=T)

    def s_stable(self):
        T0 = self.f1
        if T0.empty:
            return
        info = self.f2.iloc[0] if not self.f2.empty else {}
        d = T0.head(12).iloc[::-1]
        thr = float(info.get("threshold", 0.6))
        f, ax, p = self.fig("stability_probabilities", 9, 0.36 * len(d) + 1.4)
        ax.barh([self.short(str(x), 40) for x in d["description"]], d["probability"],
                color=[C["green"] if v >= thr else C["line"] for v in d["probability"]])
        ax.axvline(thr, color=C["red"], lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_title(f"Share of {int(info.get('draws', 100))} half-samples in which each group was picked", loc="left")
        ch = self._chosen("Stable first")
        st_ = T0[T0["probability"] >= thr]
        self.add("5. Choosing drivers: stable first (stability selection)",
                 f"{len(st_)} driver group(s) picked in at least {thr:.0%} of the half-samples",
                 "A driver worth keeping is one the data pick again and again, not once.",
                 what=[f"A lasso is run on {int(info.get('draws', 100))} random halves of the history (in 4-quarter "
                       f"blocks); each time the first {int(info.get('q', 8))} variables to enter are recorded "
                       "(Meinshausen & Bühlmann 2010).",
                       "Variables count, not forms or lags; closely related variables (correlation above 0.75) vote "
                       f"together: {int(info.get('variables', 0))} variables in {int(info.get('groups', 0))} groups.",
                       "The kept groups get their best form and lag; the lowest-BIC subset with the expected signs is "
                       "the model."],
                 why=["One selection on one sample is fragile with 80 quarters and thousands of candidates. The share "
                      "of half-samples that pick a driver says how sure the data are about it."],
                 result=[f"Expected number of falsely kept groups at most {float(info.get('expected_false_kept_at_most', np.nan)):.1f} "
                         "(Meinshausen-Bühlmann bound).",
                         f"Stable first chose: {', '.join(self.label(c) for c in ch)}." if ch else ""],
                 chart=self.save(f, p))

    def s_focus(self):
        D1, D2 = self.d1, self.d2
        if D1.empty:
            return
        st_ = self.d3.iloc[0] if not self.d3.empty else {}
        used = D2[D2["used"]] if not D2.empty and "used" in D2 else D2
        T = pd.DataFrame({"Role": ["focus"] * len(D1) + ["control"] * len(used),
                          "Driver": [self.short(x, 70) for x in list(D1["feature"]) + list(used.get("feature", []))],
                          "Why": list(D1["how"].str.split(";").str[0]) + [f"predicts {x}" for x in used.get("selected_by", [])]})
        self.add("5. Choosing drivers: focus first (explanatory model)",
                 f"{len(D1)} focus drivers always in, {len(used)} control(s) chosen around them",
                 "A fourth model is built to measure effects, not to win the forecast race.",
                 what=["The focus drivers are the variables whose effect we want to know (impact.focus; by default "
                       "the literature core). Each gets the form and lag with the strongest effect of the expected sign.",
                       "Controls are chosen by double selection: a candidate is kept if it predicts the target or any "
                       "focus driver (rigorous lasso, Belloni, Chernozhukov & Hansen 2014). Alternative measures of "
                       "the same concept and other outcomes of the economy (credit, deposits) are not allowed as "
                       "controls; controls with |t| below 1 are dropped."],
                 why=["A forecasting search keeps whatever forecasts best, which may be a variable that moves with "
                      "the target for reasons unrelated to the economy we want to describe. Holding fixed the "
                      "variables that predict both the target and a driver is what keeps the driver's effect unbiased."],
                 result=[f"{int(st_.get('candidates', 0)):,} candidates considered as controls; "
                         f"{int(st_.get('selected_any', 0))} picked by at least one equation; "
                         f"{int(st_.get('controls_used', 0))} used."],
                 table=T)

    def s_impact_responses(self):
        R = self.i_resp
        if R.empty:
            return
        ex = self._ex_name()
        R = R[R["shock_type"] == "typical"]
        shocks = list(dict.fromkeys(R["shock"]))
        n = len(shocks)
        cols = min(n, 2)
        rows = int(np.ceil(n / cols))
        f, axes = plt.subplots(rows, cols, figsize=(9, 2.6 * rows), squeeze=False)
        for ax, sh in zip(axes.flat, shocks):
            d = R[R["shock"] == sh]
            for m, g in d[(d["kind"] == "point") & (d["model"] != ex)].groupby("model"):
                ax.plot(g["quarter"], g["balance_effect"], color=C["line"], lw=1)
            lo, hi = (d[(d["model"] == ex) & (d["kind"] == k)] for k in ("band_lo", "band_hi"))
            if len(lo) and len(hi):
                ax.fill_between(lo["quarter"], lo["balance_effect"], hi["balance_effect"], color=C["teal"],
                                alpha=0.18, lw=0)
            e = d[(d["model"] == ex) & (d["kind"] == "point")]
            ax.plot(e["quarter"], e["balance_effect"], color=C["teal"], lw=2.2)
            ax.axhline(0, color=C["mid"], lw=0.7)
            ax.set_title(f"{sh} {d['shock_label'].iloc[0]}", loc="left", fontsize=10)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        p = os.path.join(self.charts, "impact_responses.png")
        self.add(self.IMP, "How the effect builds up, quarter by quarter",
                 "Effects arrive with the drivers' lags and then settle as the target's own persistence fades.",
                 what=["Each driver's whole future path is shifted up by a typical amount from the first forecast "
                       "quarter, and stays there; the other drivers keep their baseline paths. The response is the "
                       "shocked forecast minus the baseline forecast.",
                       "A shock to one series also moves its look-alikes (same theme, correlated history), so a model "
                       "that measures the concept with another series responds too."],
                 why=["The shape answers 'when': how many quarters before the effect shows, and whether it keeps "
                      "growing or levels off."],
                 read=f"Thick line: the explanatory model, with its 90% band; thin grey lines: the most accurate "
                      f"models ({len(self.i_tied[self.i_tied['in_set']]) if not self.i_tied.empty else 0}).",
                 chart=self.save(f, p))

    def s_impact_reliability(self):
        Rl = self.i_rel
        if Rl.empty:
            return
        T = pd.DataFrame({"Driver": Rl["description"], "t": [f"{v:.1f}" for v in Rl["t"]],
                          "Sign stable": [f"{v:.0%}" if pd.notna(v) else "" for v in Rl["sign_stability"]],
                          "VIF": [f"{v:.1f}" if pd.notna(v) else "" for v in Rl["vif"]],
                          "Models agree": [f"{v:.0%}" if pd.notna(v) else "" for v in Rl["tied_models_agree"]],
                          "Verdict": Rl["label"], "Why": Rl["reasons"]})
        n_ok = int((Rl["label"] == "Reliable").sum())
        tied = self.i_tied[self.i_tied["in_set"]] if not self.i_tied.empty else self.i_tied
        self.add(self.IMP, f"{n_ok} of {len(Rl)} effects pass every reliability check",
                 "An effect is only worth acting on if the data pin it down.",
                 what=["Six checks per driver: significance (|t| ≥ 1.96); the expected sign; the sign held when the "
                       "model was re-estimated through the backtest (≥ 80%); not entangled with other drivers "
                       "(VIF ≤ 5); the effect does not depend on the controls (it moves by less than half without "
                       "them); and the most accurate models agree on its direction (≥ 75%).",
                       "Reliable = all checks passed; use with care = one failed; weak = two or more."],
                 why=["With a few decades of quarterly data, some effects are estimated precisely and others are "
                      "not. Saying which is which keeps a weak effect from being presented as a fact."],
                 result=[f"Models used for the range: {', '.join(tied['model'])}." if len(tied) else ""],
                 table=T)

    def s_impact_attribution(self):
        A = self.i_att
        if A.empty:
            return
        ex, h = self._ex_name(), self._hm()
        a = A[(A["model"] == ex) & (A["horizon"] == h)]
        if a.empty:
            return
        scen = list(dict.fromkeys(a["scenario"]))
        groups = list(a.groupby("group")["contribution"].apply(lambda v: v.abs().max()).sort_values(ascending=False).index)
        pal = [C["teal"], C["gold"], C["green"], C["red"], C["purple"], C["orange"], C["grey"], C["sea"]]
        f, ax, p = self.fig("impact_attribution", 9, 0.8 * len(scen) + 1.8)
        for i, sc in enumerate(scen[::-1]):
            g = a[a["scenario"] == sc].set_index("group")["contribution"]
            pos = neg = 0.0
            for j, gr in enumerate(groups):
                v = float(g.get(gr, 0.0))
                ax.barh(i, v, left=pos if v >= 0 else neg, color=pal[j % len(pal)], label=gr if i == 0 else None)
                pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
            ax.plot(float(a.loc[a["scenario"] == sc, "total"].iloc[0]), i, "D", color=C["ink"],
                    label="total" if i == 0 else None)
        ax.set_yticks(range(len(scen)))
        ax.set_yticklabels(scen[::-1])
        ax.axvline(0, color=C["mid"], lw=0.8)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title(f"Difference from the base scenario after {h} quarters ({self.EU or self.UNITS}), by driver",
                     loc="left")
        res = []
        for sc in scen:
            g = a[a["scenario"] == sc].sort_values("contribution", key=abs, ascending=False)
            res.append(f"{sc}: {g['total'].iloc[0]:+.2f}{self.EU} in total; "
                       + ", ".join(f"{r['group']} {r['contribution']:+.2f}" for _, r in g.head(3).iterrows()))
        self.add(self.IMP, "Why the scenarios differ, driver by driver",
                 "Each scenario's gap from the base scenario, split into the part each driver explains.",
                 what=["For each scenario, the drivers are switched from their base path to the scenario path one "
                       "group at a time, in every possible order; each group's contribution is its average effect "
                       "(Shapley value), so the parts add up exactly to the total."],
                 why=["In a real scenario the drivers move together. This split is more realistic than single-driver "
                      "shocks, and it shows which assumption of the scenario does the work."],
                 result=res, chart=self.save(f, p))

    def s_impact_whatif(self):
        W = self.i_wi
        if W.empty:
            return
        ex = self._ex_name()
        hs = [h for h in self.HZ if h in set(W["quarter"])]
        w = W[(W["kind"] == "point") & (W["quarter"].isin(hs))]
        rows = []
        for sh, g in w.groupby("shock", sort=False):
            e = g[g["model"] == ex].set_index("quarter")["balance_effect"]
            rows.append([sh] + [f"{e.get(h, np.nan):+.2f}{self.EU}" for h in hs]
                        + [(lambda o: f"{o.min():+.2f} to {o.max():+.2f}" if len(o) else "no other model responds")(
                            g.loc[(g["model"] != ex) & (g["balance_effect"].abs() > 1e-9), "balance_effect"])])
        T = pd.DataFrame(rows, columns=["What-if"] + [f"after {h}q" for h in hs] + ["range across models"])
        self.add(self.IMP, "What-if: the shocks you asked about",
                 "Your own combinations of shocks, and historical episodes replayed on top of the base scenario.",
                 what=["Set in the config (impact.what_if, impact.episodes), in the app, or with "
                       "python tools/impact.py --shock VARIABLE:SIZE:QUARTERS --episode START:END.",
                       "An episode replays each macro series' quarter-by-quarter changes from that period on top of "
                       "the last actual values, then the base scenario resumes."],
                 table=T)

    def s_glossary(self):
        G = [("Organic growth", "Growth of the balance after removing one-off events such as acquisitions."),
             ("Driver", "A macro variable, in a given form and lag, that enters the forecast equation."),
             ("Form", "How a variable is measured: quarterly growth, annual growth, change, gap from trend..."),
             ("Lag", "How many quarters earlier the driver's value is taken."),
             ("Expected sign", "The direction in which theory says a driver should move the target."),
             ("Stationary", "Fluctuates around a stable average; required to avoid spurious relations."),
             ("t-statistic", "How many standard errors a coefficient is away from zero; |t| ≥ 1.96 is significant at 5%."),
             ("Newey-West", "Standard errors that remain valid when errors are correlated over time."),
             ("Elastic Net", "A regression that shrinks coefficients toward zero and keeps only the useful candidates."),
             ("Stability selection", "Repeating a selection on many subsamples and keeping what is picked consistently."),
             ("VIF", "Variance inflation factor: how much a driver is explained by the other drivers; above 5 is a warning."),
             ("BIC", "Bayesian information criterion: fit penalised for the number of parameters."),
             ("Backtest", "Forecasting history from past origins with the data available at each one."),
             ("Out of sample", "A backtest in which the drivers are also re-chosen at each origin."),
             ("Average error (MAPE)", "Average absolute percentage gap between forecast and actual level."),
             ("AR(1) benchmark", "Forecast that carries last quarter's growth forward with its estimated persistence."),
             ("Dummy quarter", "A quarter given its own indicator so a one-time shock does not distort the estimates."),
             ("Explanatory model", "The focus drivers always in, controls chosen by double selection: built to measure "
                                   "effects, not to forecast best."),
             ("Double selection", "Keeping every control that predicts the target or a focus driver, so the driver's "
                                  "effect is not biased by what was left out (Belloni, Chernozhukov & Hansen 2014)."),
             ("Typical rise", "One standard deviation of a driver's change over four quarters."),
             ("Shapley value", "A driver's average contribution over every order in which the drivers can be switched; "
                               "the contributions add up to the total.")]
        self.add("Appendix", "Glossary", "", table=pd.DataFrame(G, columns=["Term", "Meaning"]))

    def s_settings(self):
        c = self.cfg
        s, fm = c["sample"], c.get("final_model") or {}
        rows = [("Run name", c["run_name"]), ("Target", self.TARGET), ("Target form", self.form),
                ("Estimation start", s.get("estimation_start")), ("Last observed quarter", s.get("hist_end")),
                ("Backtest start", s.get("backtest_start")), ("Horizon", f"{self.H} quarters"),
                ("Dummy quarters", ", ".join(map(str, s.get("dummy_quarters", []))) or "none"),
                ("Drivers per model", f"{fm.get('min_vars', 1)} to {fm.get('max_vars', 3)}"),
                ("Significance rule", f"|t| ≥ {fm.get('t_min', 1.96)}"), ("VIF limit", fm.get("vif_max", 5)),
                ("Selection tolerance", f"{fm.get('selection_tolerance', 0.05):.0%}"),
                ("Approaches", ", ".join(c.get("approaches", []))),
                ("Ranking basis", self.basis)]
        self.add("Appendix", "Settings used in this run", "Every number in this document comes from these settings and "
                 "the run's output tables.", table=pd.DataFrame(rows, columns=["Setting", "Value"]).astype(str),
                 table_note=f"Output folder: {self.out}")


def _wrap(s, n):
    words, lines, cur = str(s).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def _wrapl(s, n):
    return "\n".join(_wrap(s, n))
