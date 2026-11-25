"""macroforecast app: upload your files, review the settings it proposes, check, run and see the results.

    pip install streamlit
    streamlit run app.py

Runs on your computer and opens in your browser. Nothing leaves your machine. Every setting the app writes is
an ordinary config file (config/<name>.yaml), so a run started here can be repeated from the command line.
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import new_data_setup as setup  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.preflight import check  # noqa: E402

st.set_page_config(page_title="macroforecast", page_icon="📈", layout="wide")
_v = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
if _v < (1, 23):
    st.error(f"This app needs Streamlit 1.23 or later (you have {st.__version__}): pip install --upgrade streamlit")
    st.stop()
S = st.session_state

ML = ["ridge", "random_forest", "gradient_boosting", "neural_net"]
REGIME = ["markov", "threshold", "smooth_transition", "tvp", "break_adaptive"]
STAT = ["arimax", "theta"]
DEEP = ["tide", "nhits", "tft", "nbeats", "tsmixer"]


# --------------------------------------------------------------------------------------------- helpers
def wide_image(path):
    """Show an image at the page width, whatever the Streamlit version (the setting was renamed in 1.40).

    Args:
        path: image file.
    """
    for kw in ({"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            return st.image(path, **kw)
        except TypeError:
            continue


class _Box:
    """Stand-in for st.status on Streamlit versions before 1.26: an expander whose label cannot change."""

    def __init__(self, label):
        """Open an expander.

        Args:
            label: its title.
        """
        self.exp = st.expander(label, expanded=True)

    def __enter__(self):
        """Enter the expander."""
        self.exp.__enter__()
        return self

    def __exit__(self, *a):
        """Leave the expander."""
        return self.exp.__exit__(*a)

    def update(self, label="", state="", expanded=True):
        """Show the final status as a message instead of relabelling.

        Args:
            label: message. state: complete or error. expanded: unused.
        """
        (st.error if state == "error" else st.success)(label)


def status_box(label):
    """st.status where available, an expander otherwise.

    Args:
        label: title.

    Returns:
        context manager with update().
    """
    return st.status(label, expanded=True) if hasattr(st, "status") else _Box(label)


def save_upload(f, folder):
    """Save an uploaded file to the project's data folder and return its path.

    Args:
        f: Streamlit UploadedFile. folder: target folder.

    Returns:
        path.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f.name)
    with open(path, "wb") as out:
        out.write(f.getbuffer())
    return path


def quarters(a, b):
    """Quarter labels from a to b.

    Args:
        a, b: quarter labels.

    Returns:
        list of str.
    """
    return [str(q) for q in pd.period_range(pd.Period(a, "Q"), pd.Period(b, "Q"), freq="Q")]


def q_select(label, options, value, key, help=None):
    """A quarter picker that keeps a value missing from the options.

    Args:
        label: label. options: quarter labels. value: current value. key: widget key. help: tooltip.

    Returns:
        the chosen quarter.
    """
    opts = list(options)
    if value not in opts:
        opts = [value] + opts
    return st.selectbox(label, opts, index=opts.index(value), key=key, help=help)


def show_report(rep):
    """Pre-flight findings: errors and warnings with their fixes, the rest folded.

    Args:
        rep: Report from src.utils.preflight.check.
    """
    if rep.errors:
        st.error(f"{len(rep.errors)} problem(s) to fix before running")
    elif rep.warnings:
        st.warning(f"Ready to run, with {len(rep.warnings)} warning(s)")
    else:
        st.success("Ready to run: every check passed")
    for i in rep.errors + rep.warnings:
        box = st.error if i["level"] == "error" else st.warning
        box(f"**{i['area']}** · {i['message']}" + (f"\n\n→ {i['fix']}" if i["fix"] else ""))
    with st.expander(f"All {len(rep.items)} checks"):
        st.dataframe(pd.DataFrame(rep.items), hide_index=True, use_container_width=True)


def multi_key(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def recommendation_section(out):
    """The recommended model: every model scored on forecasting, interpretation and sensitivity."""
    import json
    jf = os.path.join(out, "recommendation", "recommended.json")
    if not os.path.exists(jf):
        return
    j = json.load(open(jf, encoding="utf-8"))
    st.subheader(f"Recommended model: {j['model']}")
    st.markdown(f"Chosen from all models ({j['basis']}): {j['why']}.")
    if j.get("most_accurate") and j["most_accurate"] != j["model"]:
        st.caption(f"Most accurate model: {j['most_accurate']} ({j['most_accurate_error_pct']:.2f}%).")
    cols = ["rank", "model", "family", "error_pct", "accuracy", "interpretation", "sensitivity", "score", "eligible"]
    S = read_result(os.path.join(out, "recommendation", "1_scorecard.csv"), need=cols)
    if os.path.exists(os.path.join(out, "recommendation", "scorecard.png")):
        wide_image(os.path.join(out, "recommendation", "scorecard.png"))
    with st.expander("The scorecard: every model"):
        extra = ["not_eligible_because", "model_type", "sign_share", "drivers", "economic_drivers", "coverage",
                 "direction", "scenario_agreement"]
        if len(S):
            st.dataframe(S[cols + [c for c in extra if c in S.columns]].round(3), hide_index=True,
                         use_container_width=True)
        w = j.get("weights", {})
        st.caption(f"Score = {w.get('accuracy', 0):.0%} accuracy + {w.get('interpretation', 0):.0%} interpretation + "
                   f"{w.get('sensitivity', 0):.0%} sensitivity. Eligible: error within {j.get('max_error_ratio')}x the "
                   f"best and a response to at least {float(j.get('min_coverage') or 0):.0%} of the focus drivers. "
                   "Change them in the config (recommend) or in ③ Settings.")


def impact_section(out, cfg_path):
    """How the economy affects the target: sensitivities, reliability, scenario attribution and a what-if form."""
    d = os.path.join(out, "impact")
    sens = read_result(os.path.join(d, "3_sensitivity.csv"), need=("horizon", "concept", "series"))
    if not len(sens):
        return
    st.subheader("How the economy affects the target")
    rel = read_result(os.path.join(d, "4_reliability.csv"), need=("concept", "label", "reasons"))
    hs = sorted(sens["horizon"].unique())
    h = st.select_slider("Quarters after the shock", hs, value=8 if 8 in hs else hs[-1], key="imp_h")
    from src.impact.analysis import sentence
    for _, r in sens[sens["horizon"] == h].iterrows():
        lab = rel.loc[rel["concept"] == r["concept"]] if len(rel) else rel
        label = lab["label"].iloc[0] if len(lab) else ""
        icon = {"Reliable": "🟢", "Use with care": "🟡", "Weak": "🔴"}.get(label, "")
        why = f" — {lab['reasons'].iloc[0]}" if len(lab) and label != "Reliable" else ""
        st.markdown(f"{icon} {sentence(None, r, h)} **{label}**{why}")
    st.caption("Main estimate: the explanatory model (focus drivers + double-selection controls), with a 90% band from "
               "its coefficient uncertainty. Range: the most accurate models. A typical rise = one standard deviation "
               "of the series' 4-quarter change; the shift is permanent and also moves the series' look-alikes. "
               "Associations estimated from history, holding the other drivers at their baseline.")
    for p in ("sensitivity.png", "attribution.png", "responses.png"):
        if os.path.exists(os.path.join(d, p)):
            wide_image(os.path.join(d, p))
    with st.expander("Reliability checks and the models compared"):
        if len(rel):
            st.dataframe(rel, hide_index=True, use_container_width=True)
        t = read_result(os.path.join(d, "1_tied_models.csv"))
        if len(t):
            st.dataframe(t, hide_index=True, use_container_width=True)
    with st.form("what_if_form"):
        st.markdown("**What-if**: shift one or more variables and see the response (about 15 seconds).")
        opts = list(dict.fromkeys(list(sens["concept"]) + list(sens["series"])))
        rows = []
        for i in range(3):
            c1, c2, c3 = st.columns([2, 1, 1])
            v = c1.selectbox(f"Variable {i + 1}", ["(none)"] + opts, key=f"wi_v{i}",
                             help="A focus concept, or type any series name in the box below")
            size = c2.number_input("Size", value=0.0, step=0.25, key=f"wi_s{i}",
                                   help="Points for rates (+1 = one percentage point), percent of the level otherwise")
            q = c3.number_input("Build-up (quarters)", 1, 12, 1, key=f"wi_q{i}")
            if v != "(none)" and size != 0:
                rows.append(f"{v}:{size:+g}:{int(q)}")
        other = st.text_input("Other shocks (VARIABLE:SIZE:QUARTERS, comma-separated)", "", key="wi_other")
        rows += [x.strip() for x in other.split(",") if x.strip()]
        ep = st.text_input("Replay a historical episode (START:END, e.g. 2008Q3:2009Q4)", "", key="wi_ep")
        name = st.text_input("Name", "My what-if", key="wi_name")
        go = st.form_submit_button("Run the what-if")
    if go:
        args = ["--run-dir", out]
        if rows:
            args += ["--shock", f"{name}=" + ",".join(rows)]
        if ep.strip():
            args += ["--episode", ep.strip()]
        if len(args) == 2:
            st.warning("Set at least one shock or an episode.")
        elif run_script(os.path.join("tools", "impact.py"), cfg_path, args) != 0:
            st.error("The what-if could not be run: see the log above.")
    w = read_result(os.path.join(d, "6_what_if.csv"), need=("kind", "quarter", "model", "shock", "balance_effect"))
    if len(w):
        w = w[(w["kind"] == "point") & (w["quarter"].isin(hs))]
        if len(w):
            st.markdown("**What-if results** (balance effect; explanatory model and the range of the most accurate models)")
            ex = w[w["model"].astype(str).str.startswith("Explanatory")].pivot_table(index="shock", columns="quarter",
                                                                       values="balance_effect", sort=False)
            rng = w[w["balance_effect"].abs() > 1e-9].groupby(["shock", "quarter"])["balance_effect"].agg(
                ["min", "max", "count"]).round(2)
            st.dataframe(ex.round(2).add_prefix("after q"), use_container_width=True)
            with st.expander("Range across models"):
                st.dataframe(rng, use_container_width=True)
            if os.path.exists(os.path.join(d, "what_if.png")):
                wide_image(os.path.join(d, "what_if.png"))


def story_theme_picker():
    """The look of the process story: a theme name, or "all" for one copy of each to compare."""
    from src.reporting import themes as TH
    names = list(TH.THEMES) + ["all"]
    lab = {**{k: f"{v['label']}: {v['note']}" for k, v in TH.THEMES.items()},
           "all": "All themes: one copy of each, to compare (takes longer)"}
    return st.selectbox("Style of the deck and page", names, format_func=lambda k: lab[k], key="story_theme",
                        help="The same content in another look. Charts follow the theme. Set a project default "
                             "with `process_story: {theme: corporate}` in the config.")


def story_gallery(story_dir):
    """Download buttons for every theme's copy."""
    from src.reporting import themes as TH
    base = os.path.join(story_dir, "themes")
    if not os.path.isdir(base):
        return
    st.markdown("**Every theme** (open the HTML or the deck of the one you like, then pick it above and rebuild)")
    for name, t in TH.THEMES.items():
        files = [os.path.join(base, name, f) for f in ("process_story.html", "process_story.pptx")]
        if not any(os.path.exists(f) for f in files):
            continue
        c0, c1, c2 = st.columns([3, 1, 1])
        c0.markdown(f"**{t['label']}** · {t['note']}")
        for col_, f_, mime_ in zip((c1, c2), files, ("text/html",
                                   "application/vnd.openxmlformats-officedocument.presentationml.presentation")):
            if os.path.exists(f_):
                col_.download_button(os.path.splitext(f_)[1][1:].upper(), open(f_, "rb").read(),
                                     file_name=f"process_story_{name}{os.path.splitext(f_)[1]}", mime=mime_,
                                     key=f"dl_theme_{name}_{os.path.splitext(f_)[1]}")


def run_script(script, cfg_path, extra=()):
    """Run run_pipeline.py or run_oos.py and stream its log into the page.

    Args:
        script: script file name. cfg_path: config. extra: more command-line arguments.

    Returns:
        exit code.
    """
    cmd = [sys.executable, "-u", os.path.join(HERE, script), "--config", cfg_path, *extra]
    lines, steps = [], 0
    with status_box(f"Running {script} …") as box:
        bar = st.progress(0.0, text="starting")
        area = st.empty()
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")   # UTF-8 on Windows too
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding="utf-8", errors="replace", env=env)
        for ln in p.stdout:
            if "Warning" in ln or "warnings.warn" in ln or "litlogger" in ln:
                continue
            lines.append(ln.rstrip())
            if "[computed]" in ln or "[  cached]" in ln or ln.startswith("origin "):
                steps += 1
                bar.progress(min(0.98, steps / 40), text=ln.strip()[:90])
            area.code("\n".join(lines[-25:]), language=None)
        code = p.wait()
        bar.progress(1.0, text=f"finished in {time.time() - t0:.0f} s")
        box.update(label=f"{script} {'finished' if code == 0 else 'stopped with an error'} ({time.time() - t0:.0f} s)",
                   state="complete" if code == 0 else "error", expanded=code != 0)
    S["last_log"] = "\n".join(lines)
    return code


def is_panel(cfg):
    """True when the config's target file holds several banks (data.bank.panel_column)."""
    return bool(((cfg.get("data") or {}).get("bank") or {}).get("panel_column"))


def run_check(cfg_path):
    """Pre-flight check of a config; a panel config is checked on the total of its banks."""
    cfg = load_config(cfg_path)
    if is_panel(cfg):
        from run_panel import total_cfg
        return check(total_cfg(cfg)[0])
    return check(cfg)


def run_buttons(cfg_path, key=""):
    """The run buttons: panel analysis for a panel config, otherwise the pipeline and the out-of-sample test."""
    cfg = load_config(cfg_path)
    if is_panel(cfg):
        try:
            from src.panel.reuse import status
            stt = status(cfg)
            n_re = int((stt["panel"] != "reuse").iloc[0]) if len(stt) else 1
            st.markdown("**What a run would do** (results are reused when nothing they depend on changed)")
            st.dataframe(stt, hide_index=True, use_container_width=True)
            if not n_re:
                st.caption("The panel results are up to date: running again only reloads them, unless you force it. "
                           "With the out-of-sample test, they are reused only if the last run had it too.")
        except Exception as e:
            st.caption(f"Cannot compare with the last run: {e}")
        force = st.checkbox("Re-estimate everything, even what is unchanged", False, key=f"pf{key}")
        extra = ["--force"] if force else []
        c1, c2 = st.columns(2)
        if c1.button("Run the panel analysis (selection on all banks, backtest, scenarios)", type="primary", key=f"rp{key}"):
            run_script("run_panel.py", cfg_path, ["--no-oos", *extra])
        if c2.button("Run it with the out-of-sample test (slower: re-selects at every origin)", key=f"rpo{key}"):
            run_script("run_panel.py", cfg_path, extra)
    else:
        c1, c2 = st.columns(2)
        if c1.button("Run the pipeline (selection, backtest, scenarios)", type="primary", key=f"rr{key}"):
            run_script("run_pipeline.py", cfg_path)
        if c2.button("Run the out-of-sample test (slower: re-selects at every origin)", key=f"ro{key}"):
            run_script("run_oos.py", cfg_path)


PANEL_KINDS = {"separate": "Separate (each bank alone)", "pooled": "Pooled fixed effects (one effect for all banks)",
               "partial": "Partial pooling (each bank pulled toward the pooled effect)",
               "topdown": "Top-down (total of all banks x each bank's share; balances only)"}


def panel_results(out, cfg):
    """Results of a panel run (outputs/<run>/panel/)."""
    pdir = os.path.join(out, "panel")
    f_all = os.path.join(pdir, "1_leaderboard_overall.csv")
    if not os.path.exists(f_all):
        st.info("No panel results yet: run the panel analysis in ④.")
        return
    from src.panel import compare
    use_oos_avail = os.path.exists(os.path.join(pdir, "oos", "4_forecasts.csv"))
    man = {}
    try:
        import json
        man = json.load(open(os.path.join(pdir, "fitted", "manifest.json"), encoding="utf-8"))
    except Exception:
        pass
    unit = "%" if man.get("target_form", "dlog") == "dlog" else " (target units)"
    st.caption(f"Panel drivers (chosen on the estimation sample): {', '.join(man.get('panel_drivers') or []) or 'none'}. "
               f"Literature core: {', '.join(man.get('core') or []) or 'none'}.")
    sc_ = read_result(os.path.join(pdir, "0_sample_choice.csv"), need=("chosen", "label", "banks", "reason"))
    if len(sc_) and sc_["chosen"].astype(bool).any():
        ch = sc_[sc_["chosen"].astype(bool)].iloc[0]
        st.info(f"**Estimation sample: {ch['label']}** ({int(ch['banks'])} credit unions). {ch['reason']}.")
        if len(sc_) > 1:
            with st.expander("How the sample was chosen"):
                st.caption("Each sample chooses its own drivers; the pooled model is then backtested and only the "
                           "credit unions still reporting are scored (they are the ones to forecast).")
                st.dataframe(sc_.drop(columns=[c for c in ("reason",) if c in sc_]).rename(columns={
                    "score_active": "Error for the active credit unions", "exited_included": "Exited included"}),
                    hide_index=True, use_container_width=True)

    # ---- how to compare: metric, test, horizon, which credit unions
    ev0 = compare.settings(cfg)
    st.subheader("Comparing the models")
    k1, k2, k3 = st.columns([1.2, 1.4, 1])
    mets = list(compare.METRICS)
    metric = k1.selectbox("Rank on", mets, index=mets.index(ev0["metric"]), format_func=compare.METRICS.get,
                          key="pe_metric",
                          help="MAPE: average absolute error. RMSE and the tail measures punish a model that is right "
                               "on average but sometimes far off: 90th percentile, average of the worst 10%, largest "
                               "error, or MAPE plus a penalty on the worst errors.")
    tests = (["Out of sample (drivers re-selected at every origin)"] if use_oos_avail else []) + ["Backtest"]
    test = k2.radio("Test", tests, key="pe_test", horizontal=True)
    use_oos = test.startswith("Out")
    E = compare.load_errors(pdir, use_oos)
    Hmax = int(E["horizon_q"].max())
    H = k3.slider("Quarters ahead (1..h)", 1, Hmax, min(int(ev0.get("horizon") or Hmax), Hmax), key="pe_h")
    all_banks = sorted(E["bank"].astype(str).unique())
    act = [b for b in (man.get("active_banks") or []) if b in all_banks]
    inc_exit = False
    if act and len(act) < len(all_banks):
        inc_exit = st.checkbox(f"Include the {len(all_banks) - len(act)} credit unions that stopped reporting",
                               False, key="pe_exit",
                               help="By default only the credit unions still reporting are compared: they are the "
                                    "ones you forecast. Their earlier forecasts are in the backtest either way.")
    pool = all_banks if (inc_exit or not act) else act
    pick = st.multiselect(f"Credit unions in the comparison (empty = all {len(pool)})", pool, default=[], key="pe_banks")
    with st.expander("More settings"):
        q1, q2, q3, q4 = st.columns(4)
        tail = q1.number_input("Worst share (tail)", 0.02, 0.5, float(ev0["tail_share"]), 0.01, key="pe_tail")
        pen = q2.number_input("Penalty (MAPE + tail)", 0.0, 3.0, float(ev0["penalty"]), 0.1, key="pe_pen")
        wgt = q3.radio("Average across credit unions", ["equal", "size"], key="pe_w",
                       index=["equal", "size"].index(ev0["weighting"]),
                       help="equal: every credit union counts the same; size: weighted by its latest balance")
        mods = sorted(E["model"].unique())
        bm0 = next((m for m in mods if m.startswith("AR(1)")), mods[0])
        bench = q4.selectbox("Benchmark", mods, index=mods.index(bm0), key="pe_bench")
        cov = st.checkbox("Leave out the dummy (COVID) windows", bool(ev0["exclude_covid"]), key="pe_cov")
    res = compare.evaluate(E, metric, pick or pool, H, tail, pen, wgt, bench, exclude_covid=cov)
    O, used = res["overall"], res["settings"]
    b0 = O.iloc[0]
    st.success(f"**Most accurate {'out of sample' if use_oos else 'in the backtest'} on the {used['metric_long']}: "
               f"{b0['model']}** ({b0['score']:.2f}{unit} over 1–{H} quarters, average across {int(b0['banks'])} "
               f"credit union(s)); best for {int(b0['banks_won'])} of them"
               + (f", better than {bench} for {int(b0['banks_better_than_benchmark'])}."
                  if "banks_better_than_benchmark" in O else "."))
    st.markdown(f"**All {'selected ' if pick else ''}credit unions: average comparison**")
    show = ["rank", "model", "score", "mape", "rmse", "p90", "cvar", "worst", "bias", "vs_benchmark_pct",
            "banks_won", "banks_better_than_benchmark", "median_bank_score", "pooled_score"]
    names = {"score": f"Score ({compare.METRICS[metric]})", "mape": "MAPE", "rmse": "RMSE", "p90": "90th pct",
             "cvar": f"Worst {tail:.0%} avg", "worst": "Largest", "bias": "Bias (+ = over)",
             "vs_benchmark_pct": f"vs {bench} (%)", "banks_won": "Best for", "banks_better_than_benchmark":
             "Beat benchmark", "median_bank_score": "Median CU score", "pooled_score": "Pooled score"}
    st.dataframe(O[[c for c in show if c in O]].round(3).rename(columns=names), hide_index=True,
                 use_container_width=True)
    g1, g2 = st.columns(2)
    g1.caption(f"Score by model ({compare.METRICS[metric]}, lower is better)")
    g1.bar_chart(O.set_index("model")["score"].sort_values())
    g2.caption("Score by quarter ahead, average across the credit unions (top 6 models and the benchmark)")
    top = list(O["model"].head(6)) + ([bench] if bench not in list(O["model"].head(6)) else [])
    hz = res["by_horizon"]
    g2.line_chart(hz[hz["model"].isin(top)].pivot_table(index="horizon_q", columns="model", values="score"))

    st.subheader("One credit union")
    banks = pick or pool
    b = st.selectbox("Credit union", banks, key="panel_bank")
    lb = compare.one_bank(res, b)
    best = lb["model"].iloc[0] if len(lb) else None
    if best:
        st.markdown(f"Most accurate model for **{b}** on the {used['metric_long']}: **{best}**")
    shw = ["rank_in_bank", "model", "score", "mape", "rmse", "p90", "cvar", "worst", "bias", "vs_benchmark_pct", "forecasts"]
    st.dataframe(lb[[c for c in shw if c in lb]].round(3).rename(columns={**names, "rank_in_bank": "Rank",
                                                                          "forecasts": "Forecasts"}),
                 hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("This comparison (all credit unions), CSV", O.to_csv(index=False).encode(),
                       file_name=f"panel_comparison_{metric}.csv", key="pe_dl1")
    d2.download_button("Every credit union x model, CSV", res["by_bank"].to_csv(index=False).encode(),
                       file_name=f"panel_comparison_by_cu_{metric}.csv", key="pe_dl2")
    for p_ in (os.path.join(pdir, "panel_forecasts.png"),):
        if os.path.exists(p_):
            wide_image(p_)
    S_ = read_result(os.path.join(pdir, "7_scenario_forecasts.csv"),
                     need=("bank", "model", "date", "scenario", "level_forecast"))
    models_b = sorted(S_[S_["bank"] == b]["model"].unique()) if len(S_) else []
    if models_b:
        mdl = st.selectbox("Model for the scenario chart", models_b,
                           index=models_b.index(best) if best in models_b else 0, key="panel_bank_model")
        d = S_[(S_["bank"] == b) & (S_["model"] == mdl)].pivot_table(index="date", columns="scenario",
                                                                     values="level_forecast")
        try:
            from src.panel.data import load_long
            hist = load_long(cfg).get(b)
            if hist is not None:
                hh = hist.iloc[-16:]
                hh.index = hh.index.astype(str)
                d = pd.concat([hh.rename("history").to_frame(), d], axis=0)
        except Exception:
            pass
        st.line_chart(d)
    T = read_result(os.path.join(pdir, "6_shrinkage.csv"),
                    need=("bank", "model", "term", "own", "own_se", "pooled", "weight_on_own", "shrunk"))
    if len(T):
        T = T[T["bank"] == b]
        if len(T):
            st.markdown("**Partial pooling for this bank**: its own estimate, the pooled one, and the weight on its own "
                        "(w = τ² / (τ² + se²); a short or noisy history gets a small weight).")
            st.dataframe(T[["model", "term", "own", "own_se", "pooled", "weight_on_own", "shrunk"]].round(4),
                         hide_index=True, use_container_width=True)
    C = read_result(os.path.join(pdir, "5_coefficients.csv"), need=("bank",))
    if len(C):
        with st.expander("Coefficients"):
            st.markdown("**Shared coefficients** (pooled fixed effects, and the total in top-down), with Driscoll–Kraay "
                        "standard errors (robust to the same shock hitting every bank in a quarter)")
            st.dataframe(C[C["bank"].isin(["(common)", "(total)"])].round(5), hide_index=True, use_container_width=True)
            st.markdown(f"**{b}'s equation in every model**")
            st.dataframe(C[C["bank"] == b].round(5), hide_index=True, use_container_width=True)
    with st.expander("Driver selection on all banks"):
        for f_, lab in (("P3_search.csv", "Combinations that pass the rules (lowest pooled BIC first; the chosen one marked)"),
                        ("P2_shortlist.csv", "Shortlist"), ("P1_panel_screen.csv", "Screen (top 50)")):
            t_ = read_result(os.path.join(pdir, f_))
            if len(t_):
                st.markdown(f"**{lab}**")
                st.dataframe(t_.head(50), hide_index=True, use_container_width=True)
        t_ = read_result(os.path.join(pdir, "oos", "5_drivers_by_origin.csv"))
        if len(t_):
            st.markdown("**Drivers chosen at each out-of-sample origin**")
            st.dataframe(t_, hide_index=True, use_container_width=True)
    st.subheader("Downloads")
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f_ in glob.glob(os.path.join(pdir, "**", "*"), recursive=True):
            if os.path.isfile(f_) and "inputs" not in f_:
                z.write(f_, os.path.relpath(f_, pdir))
    st.download_button("All panel results (tables, charts, fitted models) as a zip", buf.getvalue(),
                       file_name=f"{cfg['run_name']}_panel_results.zip", key="dl_panel_zip")
    st.caption("The PowerPoint deck covers single-series runs; for a bank, set `panel.per_bank_full: true` to get its "
               "complete single-series results in outputs/<run>/banks/<bank>/.")


def friendly(df):
    """Error columns renamed for reading: avg_1_4_pct -> "Error, 1–4 quarters (%)".

    Args:
        df: table with avg_1_h_pct columns.

    Returns:
        DataFrame.
    """
    ren = {c: f"Error, 1–{c.split('_')[2]} quarters (%)" for c in df.columns if c.startswith("avg_1_") and c.endswith("_pct")}
    return df.rename(columns={**ren, "model": "Model", "family": "Family"}).round(3)


@st.cache_data
def lit_template():
    """The literature workbook template (the example's models, concepts, settings and references) as .xlsx bytes."""
    return open(os.path.join(HERE, "config", "literature_workbook.xlsx"), "rb").read()


def _read(out, *p):
    """A result table, or an empty one if the file is not there.

    Args:
        out: run output folder. p: path parts.

    Returns:
        DataFrame.
    """
    return read_result(os.path.join(out, *p))


def read_result(f, need=()):
    """A result CSV, or an empty table if it is missing, empty (0 bytes / no columns) or lacks a needed column.

    A run that stopped half-way, or an older run made before a column existed, then shows nothing for that table
    instead of crashing the app with EmptyDataError or KeyError.

    Args:
        f: path of the CSV. need: column names the caller will index; if any is absent, an empty table is returned.

    Returns:
        DataFrame.
    """
    if not f or not os.path.exists(f) or os.path.getsize(f) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if any(c not in df.columns for c in need):
        return pd.DataFrame()
    return df


def formula_section(out):
    """The recommended model as a formula: equation, terms, the next quarters, the Excel calculator; and the most
    accurate model overall when it has no formula.

    Args:
        out: run output folder (or a saved run).
    """
    import json
    fdir = os.path.join(out, "formula")
    jf = os.path.join(fdir, "recommended.json")
    if not os.path.exists(jf):
        return
    info = json.load(open(jf, encoding="utf-8"))
    st.subheader("The model to use by hand: formula and the next quarters")
    st.markdown(f"**{info['formula_model']}** ({info.get('formula_family', '')}): {info['why']}.")
    if not info.get("best_has_formula", True):
        who = ("The recommended model" if os.path.exists(os.path.join(out, "recommendation", "recommended.json"))
               else "The most accurate model overall")
        st.info(f"{who}, **{info['best_model']}** ({info['best_family']}, "
                f"{info['best_error_pct']:.2f}%), has no formula you can write down. Both are shown: the formula model "
                "here, its forecast below.")
    st.code(info["equation"], language=None)
    T = _read(out, "formula", "1_terms.csv")
    if len(T):
        st.dataframe(T[["symbol", "coefficient", "definition", "meaning"]], hide_index=True, use_container_width=True)
    st.caption("g(t) = ln(L(t)/L(t-1)), quarterly growth of the target; L = target level; X(t-k) = series X, k quarters back. "
               f"Estimated on {info.get('window', '')}.")
    W = _read(out, "formula", "2_next_quarters.csv")
    if len(W):
        st.markdown(f"**Next {info.get('quarters', 4)} quarters, term by term** (identical to the model's own forecast)")
        tabs_ = st.tabs(list(dict.fromkeys(W["scenario"])))
        for t_, (sc_, d_) in zip(tabs_, W.groupby("scenario", sort=False)):
            with t_:
                st.dataframe(d_.drop(columns=["scenario"]).set_index("date").T.round(6), use_container_width=True)
    xl = os.path.join(fdir, "recommended_model_calculator.xlsx")
    if os.path.exists(xl):
        st.download_button("⬇ Excel calculator (live formulas: change the drivers, the forecast updates)",
                           open(xl, "rb").read(), file_name=f"{info['formula_model'][:40]}_calculator.xlsx".replace(":", ""),
                           key=f"dl_calc_{out}")
    B = _read(out, "formula", "3_best_overall_next_quarters.csv")
    if len(B):
        st.markdown(f"**{info['best_model']}** (most accurate overall, no formula): forecast for the same quarters")
        st.dataframe(B.pivot_table(index="date", columns="scenario", values="balance", sort=False).round(1),
                     use_container_width=True)


def test_forecasts(out, m):
    """Forecast against actual for one model at every test origin (backtest and out of sample).

    Args:
        out: run folder. m: model name.
    """
    bt = _read(out, "comparison", "12_backtest_forecasts.csv")
    oo = _read(out, "oos", "6_all_forecast_errors.csv")
    srcs = {}
    if len(bt):
        srcs["Rolling backtest"] = bt[bt["model"] == m]
    if len(oo) and "forecast_balance" in oo:
        srcs["Out of sample (re-selected at each origin)"] = oo[oo["model"] == m]
    srcs = {k: v for k, v in srcs.items() if len(v)}
    if not srcs:
        return
    st.markdown("**Forecasts in the test samples**: every forecast the model made, next to what happened")
    c1, c2 = st.columns([2, 1])
    which = c1.radio("Test", list(srcs), horizontal=True, key=f"tf_src_{m}")
    D = srcs[which]
    origins = list(dict.fromkeys(D["origin"]))
    o = c2.selectbox("Forecast made at", origins, index=len(origins) - 1, key=f"tf_o_{which}_{m}")
    d = D[D["origin"] == o]
    show = d[["date", "horizon_q", "forecast_growth_pct", "actual_growth_pct", "forecast_balance",
              "actual_balance_organic", "error_pct"]].rename(columns={
        "date": "Quarter", "horizon_q": "Quarters ahead", "forecast_growth_pct": "Forecast growth (% q/q, log)",
        "actual_growth_pct": "Actual growth (% q/q, log)", "forecast_balance": "Forecast balance",
        "actual_balance_organic": "Actual balance (organic)", "error_pct": "Error (%)"})
    try:
        import altair as alt
        long = pd.concat([d.assign(series="forecast", value=d["forecast_balance"]),
                          d.assign(series="actual (organic)", value=d["actual_balance_organic"])])
        ch = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("date:O", title=None), y=alt.Y("value:Q", title="balance", scale=alt.Scale(zero=False)),
            color="series:N", tooltip=["date", "series", alt.Tooltip("value:Q", format=",.1f")]).properties(height=240)
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.line_chart(d.set_index("date")[["forecast_balance", "actual_balance_organic"]])
    st.dataframe(show.round(3), hide_index=True, use_container_width=True)
    st.caption("Organic = without one-off acquisitions and sales, the basis of every error. The full table for every "
               "model and origin is in comparison/12_backtest_forecasts.csv and oos/6_all_forecast_errors.csv.")


def model_details(out, comp, H):
    """The detail page of one model: what it is, its drivers (coefficients, importance, scenario effect), checks,
    error by horizon, coefficients over time, scenario paths and regimes.

    Args:
        out: run output folder. comp: model comparison. H: horizon.
    """
    st.subheader("Model details")
    names = list(comp.sort_values(f"avg_1_{H}_pct")["model"])
    m = st.selectbox("Model", names, key="detail_model")
    ml = _read(out, "comparison", "9_model_list.csv")
    row = ml[ml["model"] == m]
    if len(row):
        r = row.iloc[0]
        st.markdown(f"**{m}** · {r['family']}  \n{r.get('how it is specified', '') or ''}"
                    + (f"  \nSource: {r['source']}" if isinstance(r.get("source"), str) and r["source"] else "")
                    + f"  \nDrivers: `{r['drivers']}`")
    drv = _read(out, "comparison", "3_drivers.csv")
    d = drv[drv["model"] == m].drop(columns=["model"]).dropna(axis=1, how="all")
    if len(d):
        st.markdown("**Drivers**: coefficients with HAC standard errors, t and p for regressions; importance (%) for trees and "
                    "TFT; weights for combinations. The two *scenario effect* columns work for every model type, including "
                    "the networks: how much the driver's baseline path moves the forecast, compared with holding the driver "
                    "at its last actual value.")
        names = {"term": "Driver", "coef": "Coefficient / importance / weight",
                 "se_hac": "s.e. (HAC)", "t": "t", "p": "p", "expected_sign": "Expected sign", "sign_ok": "Sign as expected",
                 "importance_decoder_pct": "TFT importance looking ahead (%)",
                 "growth_effect_pp_per_year": "Scenario effect on growth (pp a year)",
                 "balance_effect_pct_8q": "Scenario effect on the balance after 8 quarters (%)",
                 "description": "What it is", "reading": "Reading"}
        order = [c for c in names if c in d.columns] + [c for c in d.columns if c not in names]
        d = d[order].rename(columns=names)
        d["What it is"] = d["What it is"].fillna("") if "What it is" in d else ""
        st.dataframe(d.round(4), hide_index=True, use_container_width=True)
    dg = _read(out, "comparison", "4_diagnostics.csv")
    dg = dg[dg["model"] == m]
    if len(dg) and dg.iloc[0].get("note") != "no regression residual tests for this model type":
        st.markdown("**Residual checks** (p-values: above 0.05 means no evidence of a problem)")
        st.dataframe(dg.drop(columns=["model", "family"]).dropna(axis=1, how="all").T.rename(columns={dg.index[0]: "value"}),
                     use_container_width=True)
    c1, c2 = st.columns(2)
    bh = _read(out, "comparison", "2_backtest_by_horizon.csv")
    oh = _read(out, "oos", "2_oos_by_horizon.csv")
    err = pd.DataFrame(index=range(1, H + 1))
    if len(bh):
        err["backtest"] = bh[bh["model"] == m].set_index("horizon_q")["MAPE_exCOVID_pct"]
    if len(oh) and "MAPE_pct" in oh:
        err["out of sample"] = oh[oh["model"] == m].set_index("horizon_q")["MAPE_pct"]
    with c1:
        st.markdown("**Balance error by quarters ahead (%)**")
        st.line_chart(err.dropna(how="all"))
    sc = _read(out, "comparison", "6_scenario_forecasts.csv")
    sc = sc[sc["model"] == m]
    with c2:
        if len(sc):
            st.markdown("**Balance forecast under each scenario**")
            try:
                import altair as alt
                ch = alt.Chart(sc).mark_line().encode(
                    x=alt.X("date:O", title=None, axis=alt.Axis(labelAngle=-90, labelOverlap=True)),
                    y=alt.Y("balance_forecast:Q", title="balance", scale=alt.Scale(zero=False)),
                    color=alt.Color("scenario:N", sort=list(dict.fromkeys(sc["scenario"]))),
                    tooltip=["scenario", "date", alt.Tooltip("balance_forecast:Q", format=",.0f")]).properties(height=300)
                st.altair_chart(ch, use_container_width=True)
            except Exception:
                st.line_chart(sc.pivot_table(index="date", columns="scenario", values="balance_forecast", sort=False))
    if m.startswith("Horizon combination"):
        ch = _read(out, "oos", "8_horizon_choices.csv")
        src_ = "out-of-sample test"
        if not len(ch):
            ch, src_ = _read(out, "comparison", "13_horizon_choices.csv"), "rolling backtest"
        if len(ch):
            col = "best" if "best per horizon" in m else "weighted"
            st.markdown(f"**Which model was used for each block of quarters, at each origin** ({src_}; 'fallback' = "
                        "too little history yet, so the model best over all horizons so far)")
            st.dataframe(ch.pivot_table(index="origin", columns="quarters_ahead", values=col, aggfunc="first"),
                         use_container_width=True)
    test_forecasts(out, m)
    cp = _read(out, "comparison", "7_coefficient_paths.csv")
    cp = cp[cp["model"] == m].dropna(axis=1, how="all") if len(cp) else cp
    if len(cp):
        keep = [c for c in cp.columns if c not in ("model", "origin", "const") and not c.startswith("D_")]
        if keep:
            st.markdown("**Coefficients at each backtest re-estimation** (stable lines = a stable model)")
            st.line_chart(cp.set_index("origin")[keep])
    rg = _read(out, "comparison", "11_regimes.csv")
    rg = rg[rg["model"] == m] if len(rg) else rg
    if len(rg):
        st.markdown(f"**Regime over history**: {rg['meaning'].iloc[0]}")
        st.line_chart(rg.set_index("date")["value"])
    pk = _read(out, "oos", "4_drivers_by_origin.csv")
    pk = pk[pk["model"] == m] if len(pk) else pk
    if len(pk):
        st.markdown("**Drivers chosen at each out-of-sample origin**")
        st.dataframe(pk.drop(columns=["model"]), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------------------------- sidebar
st.sidebar.title("macroforecast")
mode = st.sidebar.radio("Start from", ["New data set", "Existing configuration"], key="mode")
if mode == "New data set":
    name = st.sidebar.text_input("Name of this set-up", value=S.get("name", "my_portfolio"),
                                 help="Used for config/<name>.yaml and outputs/<name>/").strip().replace(" ", "_")
    S["name"] = name
    cfg_path = os.path.join(HERE, "config", f"{name}.yaml")
else:
    cfgs = sorted(glob.glob(os.path.join(HERE, "config", "*.yaml")))
    pick = st.sidebar.selectbox("Configuration", [os.path.basename(c) for c in cfgs])
    cfg_path = os.path.join(HERE, "config", pick)
    name = os.path.splitext(pick)[0]
st.sidebar.caption(f"Config file: `config/{os.path.basename(cfg_path)}`")
st.sidebar.divider()
if mode == "New data set":
    st.sidebar.markdown("**Steps**\n1. Files\n2. Data settings\n3. Sample and models\n4. Check and run\n5. Results")
else:
    st.sidebar.markdown("**Steps**\n\n- ④ Check and run\n- ⑤ Results\n- Edit any setting in the config file")

# --------------------------------------------------------------------------------------------- pages
if mode == "New data set":
    tabs = st.tabs(["① Files", "② Data settings", "③ Sample and models", "④ Check and run", "⑤ Results"])
else:
    tabs = [None, None, None] + list(st.tabs(["④ Check and run", "⑤ Results", "Edit the config file"]))

# ① files
if mode == "New data set":
    with tabs[0]:
        st.header("Upload your files")
        st.caption("CSV or Excel. Column and variable names can be anything: the next step maps them. "
                   "Files are copied to data/<name>/ in the project folder.")
        c1, c2 = st.columns(2)
        with c1:
            up_bank = st.file_uploader("Target series (required)", type=["csv", "xlsx", "xls"], key="up_bank",
                                       help="A date column and the balance; quarterly or monthly")
            up_meta = st.file_uploader("Variable metadata (recommended)", type=["csv", "xlsx", "xls"], key="up_meta",
                                       help="One row per macro variable: code, description, geography, theme, transform, sign. "
                                            "Missing columns are suggested.")
            up_events = st.file_uploader("One-off events (optional)", type=["csv", "xlsx", "xls"], key="up_events",
                                         help="Acquisitions and portfolio sales to take out of the growth")
        with c2:
            up_macro = st.file_uploader("Scenario files (required, one per scenario)", type=["csv", "xlsx", "xls"],
                                        accept_multiple_files=True, key="up_macro",
                                        help="History followed by the scenario path; same variables in every file")
            up_lit = st.file_uploader("Literature models workbook (required)", type=["xlsx"], key="up_lit",
                                      help="Sheet 'models': the published models to compare, one row per driver. "
                                           "Sheet 'concepts': each economic concept and your series for it. "
                                           "Or a master file (literature_master.xlsx) holding every target's "
                                           "workbook and metadata, with a 'target' column.")
            target = None
            if up_bank is not None:                       # a file with several targets: choose one
                try:
                    from src.utils.io import read_table as _rt
                    _bp = save_upload(up_bank, os.path.join(HERE, "data", name))
                    _bd = _rt(_bp, None)
                    _num = [c for c in _bd.columns if pd.to_numeric(_bd[c], errors="coerce").notna().mean() > 0.5
                            and not str(c).lower().startswith(("date", "quarter", "period"))]
                    if len(_num) > 1:
                        _pick = st.selectbox(f"Target to forecast ({len(_num)} series in the file)", ["(detect)"] + _num,
                                             key="target_pick",
                                             help="The file holds several series side by side. To run many of them "
                                                  "one after another, use run_targets.py (see the README).")
                        target = None if _pick == "(detect)" else _pick
                except Exception:
                    pass
            master_t = None
            if up_lit is not None:
                from src.utils import master as ms
                _lp = save_upload(up_lit, os.path.join(HERE, "data", name))
                if ms.is_master(_lp):
                    _T = ms.targets(_lp)
                    _names = list(_T["target"])
                    _tr = ms.target_rows(_lp)
                    _yours = {str(y).strip().lower(): t for t, y in zip(_tr["target"], _tr["your_name"]) if y}
                    _m = ms._match(_names, _yours.get(str(target).strip().lower()) or target) if target else None
                    master_t = st.selectbox("Target in the master file", _names, key="master_t",
                                            index=_names.index(_m) if _m in _names else 0,
                                            help="Its literature rows and its metadata are used. Filled from the "
                                                 "master's target_names sheet (your_name = your column) when you "
                                                 "have filled it. A metadata file uploaded on the left takes "
                                                 "precedence over the master's.")
                    _have = _tr[_tr["your_name"] != ""]
                    if len(_have):
                        st.caption(f"target_names: you have {len(_have)} of the master's series "
                                   f"({', '.join(f'{y} = {t}' for t, y in zip(_have['target'], _have['your_name']))}).")
                    st.caption(f"Master file: {len(_T)} targets. {master_t}: "
                               + ", ".join(f"{int(_T.set_index('target').loc[master_t].get(k, 0))} {k} rows"
                                           for k in ("models", "concepts", "metadata")))
            st.download_button("Download a template workbook (filled with the example's models)", lit_template(),
                               file_name="literature_workbook.xlsx", key="dl_tpl")
            sheet = st.text_input("Excel sheet name (if not the first sheet)", value="", key="sheet") or None
            if target is None and not st.session_state.get("target_pick"):
                target = st.text_input("Balance column (leave empty to detect)", value="", key="target") or None
            panel_in = st.text_input("Bank column, if the file holds several banks (leave empty to detect)", value="",
                                     key="panel_col", help="One row per bank and quarter: date, bank name, balance. "
                                                           "Detected when every date repeats once per value of a text column.") or None
        ready = up_bank is not None and up_macro and up_lit is not None
        if not ready:
            st.caption("Needed: the target series, at least one scenario file and the literature workbook.")
        if st.button("Read the files and propose settings", type="primary", disabled=not ready):
            folder = os.path.join(HERE, "data", name)
            bank = save_upload(up_bank, folder)
            macro = {setup.scenario_name(f.name): save_upload(f, os.path.join(folder, "scenarios")) for f in up_macro}
            meta = save_upload(up_meta, folder) if up_meta else None
            ev = save_upload(up_events, folder) if up_events else None
            litwb = save_upload(up_lit, folder)
            log = []
            if master_t:                                   # one target's workbook and metadata from the master
                from src.utils import master as ms
                x = ms.extract(litwb, master_t, os.path.join(folder, "master"))
                litwb = x["workbook"]
                meta = meta or x["metadata"]
                log.append(f"Master file: target {x['target']} -> {x['workbook'] or 'no workbook'}, {x['metadata'] or 'no metadata'}")
                S["master_target_used"] = x["target"]
                if not litwb:
                    log.append("This target has no literature rows in the master: the example's literature models "
                               "are proposed; switch the Literature and Hybrid families off in ③ if they do not apply.")
            try:
                S["P"] = setup.propose(bank, macro, meta, ev, sheet, target, None, name, log=log.append, literature=litwb,
                                       panel=panel_in)
                S["setup_log"] = "\n".join(log)
                st.success("Done: review the proposed settings in ② and ③.")
            except SystemExit as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        if "setup_log" in S:
            st.subheader("What was found")
            st.code(S["setup_log"], language=None)

    P = S.get("P")

    # ② data settings
    with tabs[1]:
        st.header("Data settings")
        if not P:
            st.info("Upload the files in ① first.")
        else:
            st.subheader("Dependent variable")
            st.caption("Choose the balance to forecast. The model explains its organic quarterly log growth "
                       "(one-off events removed) and turns the growth forecasts back into the balance.")
            c1, c2, c3 = st.columns(3)
            cols = P["bank"]["columns"]
            P["bank"]["date_column"] = c1.selectbox("Date column", cols, index=cols.index(P["bank"]["date_column"]))
            num = [c for c in cols if c != P["bank"]["date_column"]]
            P["bank"]["target_column"] = c2.selectbox("Balance to forecast (dependent variable)", num,
                                                      index=num.index(P["bank"]["target_column"]) if P["bank"]["target_column"] in num else 0)
            if S.get("master_target_used") and multi_key(P["bank"]["target_column"]) != multi_key(S["master_target_used"]):
                st.warning(f"The literature and metadata are those of '{S['master_target_used']}' from the master file. "
                           "To forecast another column with its own literature, choose it in ① and read the files again.")
            P["bank"]["frequency"] = c3.selectbox("Frequency", ["Q", "M"], index=["Q", "M"].index(P["bank"]["frequency"]),
                                                  format_func=lambda x: {"Q": "Quarterly", "M": "Monthly (quarter-end used)"}[x])
            popts = ["(one series)"] + [c for c in cols if c not in (P["bank"]["date_column"], P["bank"]["target_column"])]
            pcur = P["bank"].get("panel_column")
            psel = st.selectbox("Bank column (panel data: several banks in one file)", popts,
                                index=popts.index(pcur) if pcur in popts else 0, key="panel_sel",
                                help="Set when the file has one row per bank and quarter. Drivers are then chosen on all "
                                     "banks together and each bank is forecast by panel models.")
            P["bank"]["panel_column"] = None if psel == "(one series)" else psel
            if P["bank"]["panel_column"]:
                try:
                    from src.utils.io import read_table, to_quarters
                    bk = read_table(P["bank"]["file"], P["bank"]["sheet"])
                    dc_, tc_, pc_ = P["bank"]["date_column"], P["bank"]["target_column"], P["bank"]["panel_column"]
                    bq = pd.DataFrame({"bank": bk[pc_].astype(str).str.strip(), "q": to_quarters(bk[dc_]),
                                       "v": pd.to_numeric(bk[tc_], errors="coerce")})
                    dup = bq.duplicated(["bank", "q"]).any()
                    if dup and P["bank"]["frequency"] == "Q":
                        st.error("Some bank has several rows in the same quarter: choose Monthly, or check the bank column.")
                    bq = bq.dropna().sort_values("q").groupby(["bank", "q"], as_index=False).last()
                    summ = bq.groupby("bank").agg(first=("q", "first"), last=("q", "last"), quarters=("q", "size"),
                                                  last_value=("v", "last")).reset_index()
                    P["banks"] = summ
                    st.markdown(f"**Panel: {len(summ)} banks** by `{pc_}`")
                    short = summ[summ["quarters"] < 20]
                    summ = summ.assign(first=summ["first"].astype(str), last=summ["last"].astype(str))
                    # choose the banks: in the panel (pooled together) and, optionally, their own full run
                    names = list(summ["bank"])
                    ch = P.get("bank_choice") or {}
                    if ch.get("banks") != names:          # a new file or bank column: every bank in, no own runs
                        ch = {"banks": names, "in_panel": {b: True for b in names},
                              "own_run": {b: False for b in names}, "ver": ch.get("ver", 0) + 1}
                    a1, a2, a3, _ = st.columns([1, 1, 1.4, 2])
                    bset = None
                    if a1.button("Select all", key="bk_all"):
                        bset = ("in_panel", True)
                    if a2.button("Select none", key="bk_none"):
                        bset = ("in_panel", False)
                    if a3.button("Own run for selected", key="bk_own"):
                        ch["own_run"] = {b: bool(ch["in_panel"].get(b)) for b in names}
                        ch["ver"] += 1
                    if bset:
                        ch[bset[0]] = {b: bset[1] for b in names}
                        ch["ver"] += 1                     # a new editor, so the buttons override earlier ticks
                    tab = summ.assign(in_panel=[ch["in_panel"].get(b, True) for b in names],
                                      own_run=[ch["own_run"].get(b, False) for b in names])
                    ed = st.data_editor(
                        tab[["in_panel", "own_run", "bank", "first", "last", "quarters", "last_value"]],
                        hide_index=True, use_container_width=True, key=f"bk_ed_{ch['ver']}",
                        disabled=["bank", "first", "last", "quarters", "last_value"],
                        column_config={
                            "in_panel": st.column_config.CheckboxColumn(
                                "Estimate", help="Banks pooled together in the panel models. Leaving a bank out "
                                                 "changes the panel estimates of the others."),
                            "own_run": st.column_config.CheckboxColumn(
                                "Own full run", help="Also the complete single-series project for this bank on its "
                                                     "own (every model family); slower. Reused when nothing about "
                                                     "this bank changed.")})
                    chosen = [b for b, v in zip(ed["bank"], ed["in_panel"]) if v]
                    own = [b for b, v, o in zip(ed["bank"], ed["in_panel"], ed["own_run"]) if v and o]
                    ch.update(in_panel=dict(zip(ed["bank"], map(bool, ed["in_panel"]))),
                              own_run=dict(zip(ed["bank"], map(bool, ed["own_run"]))))
                    P["bank_choice"] = ch
                    P["panel_banks"], P["own_banks"] = chosen, own
                    if not chosen:
                        st.error("Tick at least one bank to estimate.")
                    elif len(chosen) == 1:
                        st.info(f"One credit union selected (**{chosen[0]}**): it runs as a single series - the "
                                "full project with every model family - not as a panel. Tick two or more to run "
                                "the panel.")
                    else:
                        st.caption(f"Estimating {len(chosen)} of {len(summ)} banks"
                                   + (f"; own full run for {', '.join(own)}" if own else "")
                                   + ". Results are reused when the chosen banks' data, the macro files and the "
                                     "settings are unchanged since the last run (see ④).")
                    if len(short):
                        st.warning(f"{', '.join(short['bank'])}: fewer than 20 quarters; they borrow from the other banks "
                                   "(fully pooled) instead of getting their own equation.")
                    if (bq["v"] <= 0).any():
                        st.error("Some balances are zero or negative: use a rate-type target (quarterly change) in ③.")
                    wide = bq[bq["bank"].isin(P.get("panel_banks") or names)].assign(q=bq["q"].astype(str)).pivot_table(
                        index="q", columns="bank", values="v")
                    st.line_chart(wide)
                except Exception as e:
                    st.warning(f"Cannot preview the banks: {e}")
            try:                                                  # preview of the chosen series
                if P["bank"]["panel_column"]:
                    raise StopIteration                           # the panel preview above replaces it
                from src.utils.io import read_table, to_quarters
                bk = read_table(P["bank"]["file"], P["bank"]["sheet"])
                y = pd.to_numeric(bk[P["bank"]["target_column"]], errors="coerce")
                y.index = to_quarters(bk[P["bank"]["date_column"]])
                y = (y.groupby(level=0).last() if P["bank"]["frequency"] == "M" else y).dropna().sort_index()
                g = (np.log(y).diff() * 100).rename("q/q growth (%)")
                y.index, g.index = y.index.astype(str), g.index.astype(str)
                p1, p2 = st.columns(2)
                p1.markdown(f"**{P['bank']['target_column']}**, {y.index[0]} to {y.index[-1]} ({len(y)} quarters)")
                p1.line_chart(y.rename("balance"))
                p2.markdown(f"**Quarterly log growth (%)**: mean {g.mean():.2f}, s.d. {g.std():.2f} (before removing one-off events)")
                p2.bar_chart(g.dropna())
                if (y <= 0).any():
                    st.error("This column has zero or negative values: the model needs a positive balance (it uses log growth).")
            except StopIteration:
                pass
            except Exception as e:
                st.warning(f"Cannot preview this column: {e}")

            st.subheader("Scenarios")
            sc = pd.DataFrame({"scenario": list(P["macro"]["files"]), "file": [os.path.basename(f) for f in P["macro"]["files"].values()]})
            sc = st.data_editor(sc, hide_index=True, disabled=["file"], key="sc_edit", use_container_width=True)
            paths = list(P["macro"]["files"].values())
            P["macro"]["files"] = {n: p for n, p in zip(sc["scenario"], paths)}
            names_ = list(P["macro"]["files"])
            base = P["macro"]["base"] if P["macro"]["base"] in names_ else names_[0]
            P["macro"]["base"] = st.selectbox("Base scenario (its history is used for estimation)", names_, index=names_.index(base))

            st.subheader("Metadata columns")
            if P["metadata"]["file"] and os.path.basename(P["metadata"]["file"]).endswith("_metadata_draft.csv"):
                st.warning(f"Some metadata was missing, so the app drafted it: `{setup.rel(P['metadata']['file'])}`. "
                           "Review the expected signs: they decide what can be selected.")
            try:
                mcols = list(pd.read_csv(P["metadata"]["file"], nrows=0).columns) if P["metadata"]["file"].endswith(".csv") \
                    else list(pd.read_excel(P["metadata"]["file"], nrows=0).columns)
            except pd.errors.EmptyDataError:
                mcols = []
            if not mcols:
                st.error(f"The metadata file `{setup.rel(P['metadata']['file'])}` is empty: upload it again in ①.")
                st.stop()
            cc = st.columns(6)
            for i, k in enumerate(["mnemonic", "description", "geography", "theme", "transform", "expected_sign"]):
                cur = P["metadata"]["columns"].get(k, mcols[0])
                P["metadata"]["columns"][k] = cc[i].selectbox(k, mcols, index=mcols.index(cur) if cur in mcols else 0, key=f"mc_{k}")

            st.subheader("Which variables are candidates")
            c1, c2 = st.columns(2)
            P["allowed"] = c1.multiselect("Geographies considered (empty = all)", P["geographies"], default=[g for g in P["allowed"] if g in P["geographies"]])
            P["primary"] = c2.selectbox("Main region (preferred when two series are near-duplicates)", P["geographies"],
                                        index=P["geographies"].index(P["primary"]) if P["primary"] in P["geographies"] else 0)
            c1, c2, c3 = st.columns(3)
            P["excluded_themes"] = c1.multiselect("Themes left out", P["themes"], default=P["excluded_themes"])
            P["real_themes"] = c2.multiselect("Themes also tried in real terms", P["themes"], default=P["real_themes"])
            series = sorted(P["base"].columns)
            dfl = P["deflator"] if P["deflator"] in series else None
            P["deflator"] = c3.selectbox("Price index (deflator)", [None] + series, index=([None] + series).index(dfl),
                                         format_func=lambda x: "none (no real transforms)" if x is None else x)

            st.subheader("Literature concepts → your series")
            st.caption("The published models refer to concepts. Pick your series for each; leave empty or untick to leave a concept out. "
                       "The four core concepts feed most theory-based models.")
            lit = P["lit"].copy()
            lit["include"] = lit["include"].astype(str).str.lower().isin(["yes", "y", "true", "1"])
            opts = series + [c[0] for c in P["constructed"]]
            show = lit[["concept", "include", "series_name", "core", "expected_sign", "setup_note"]].rename(
                columns={"series_name": "your series", "setup_note": "how it was matched"})
            ed = st.data_editor(show, hide_index=True, use_container_width=True, key="lit_edit",
                                disabled=["concept", "core", "expected_sign", "how it was matched"],
                                column_config={"your series": st.column_config.SelectboxColumn(options=opts),
                                               "include": st.column_config.CheckboxColumn()})
            P["lit"]["series_name"] = ed["your series"].values
            P["lit"]["include"] = ["yes" if v else "no" for v in ed["include"]]
            st.subheader("Published models to compare (from your workbook)")
            st.caption("One row per driver; rows with the same model name form one model. The variable is a concept above "
                       "or one of your series. Mark the rows of one model as core: the theory core used by the hybrids, "
                       "regime-switching, ARIMAX and the networks. Add or delete rows here; the workbook is saved with the config.")
            from src.features.transforms import parse_feature
            csg = dict(zip(P["lit"]["concept_id"].astype(str), pd.to_numeric(P["lit"]["expected_sign"], errors="coerce")))
            mrows = [{"model": m["name"], "source": m.get("source", ""), "variable": parse_feature(f)[0],
                      "transform": parse_feature(f)[1], "lag": parse_feature(f)[2],
                      "expected_sign": (m.get("signs") or {}).get(f, csg.get(parse_feature(f)[0])), "core": bool(m.get("core"))}
                     for m in P["lit_models"] for f in m["features"]]
            TRS = ["dlog", "yoy", "dlog2", "acc", "gap", "rdlog", "ryoy", "diff", "diff2", "diff4", "lvl", "rgap"]
            med = st.data_editor(pd.DataFrame(mrows, columns=["model", "variable", "transform", "lag", "expected_sign", "core", "source"]),
                                 hide_index=True, use_container_width=True, num_rows="dynamic", key="lit_models_edit",
                                 column_config={"variable": st.column_config.SelectboxColumn(options=list(P["lit"]["concept_id"]) + opts),
                                                "transform": st.column_config.SelectboxColumn(options=TRS),
                                                "lag": st.column_config.NumberColumn(min_value=0, max_value=8, step=1),
                                                "expected_sign": st.column_config.SelectboxColumn(options=[1, -1, 0], help="+1 or -1: the sign the literature expects in this model; flagged in the diagnostics if the estimate disagrees"),
                                                "core": st.column_config.CheckboxColumn(),
                                                "source": st.column_config.TextColumn(width="medium")})
            new_models = []
            for nm_, g in med.dropna(subset=["model", "variable", "transform"]).groupby("model", sort=False):
                feats_ = [f"{r.variable}|{r.transform}|L{int(r.lag if pd.notna(r.lag) else 0)}" for r in g.itertuples()]
                spec = {"name": nm_, "source": str(g["source"].dropna().iloc[0]) if g["source"].notna().any() else "",
                        "features": feats_,
                        "signs": {f: int(v) for f, v in zip(feats_, pd.to_numeric(g["expected_sign"], errors="coerce")) if pd.notna(v)}}
                if g["core"].fillna(False).astype(bool).any():
                    spec["core"] = True
                new_models.append(spec)
            P["lit_models"] = new_models
            if sum(1 for m in new_models if m.get("core")) != 1:
                st.warning("Mark exactly one model as core.")
            st.subheader("Literature settings (from your workbook)")
            st.caption("The error-correction model, the channels hybrid, the regime threshold and the sign rule. "
                       "Saved in the workbook's settings sheet.")
            from src.models.literature import SETTINGS
            cur = {k: d for k, d, _ in SETTINGS}
            cur.update(P.get("lit_settings") or {})
            def _cell(v):                                   # Excel TRUE/FALSE and empty cells as plain text
                if isinstance(v, bool):
                    return "yes" if v else "no"
                return "" if v is None or (isinstance(v, float) and v != v) else str(v)
            sdf = pd.DataFrame([(k, _cell(cur.get(k, d)), mng) for k, d, mng in SETTINGS], columns=["setting", "value", "meaning"])
            sed = st.data_editor(sdf, hide_index=True, use_container_width=True, disabled=["setting", "meaning"], key="lit_settings_edit")
            P["lit_settings"] = {k: ("" if pd.isna(v) else str(v)) for k, v in zip(sed["setting"], sed["value"])}
            with st.expander(f"Constructed variables ({len(P['constructed'])})"):
                for c in P["constructed"]:
                    st.markdown(f"**{c[0]}** = `{c[1]}` — {c[2]}")

    # ③ sample and models
    with tabs[2]:
        st.header("Sample and models")
        if not P:
            st.info("Upload the files in ① first.")
        else:
            d = P["dates"]
            allq = quarters(d["first"], d["hist_end"])
            st.subheader("Dates")
            st.caption(f"Target data from {P['series'].index[0]} to {P['series'].index[-1]}; the macro history allows estimation from {d['first']}.")
            c1, c2, c3 = st.columns(3)
            d["hist_end"] = q_select("Last actual quarter (hist_end)", allq, d["hist_end"], "q_he")
            d["estimation_start"] = q_select("Estimation starts", allq, d["estimation_start"], "q_es",
                                             help="Leave about 9 quarters after the data start for lags and yearly changes")
            d["oos_first"] = q_select("Out-of-sample test: first origin", allq, d["oos_first"], "q_fo")
            c1, c2, c3 = st.columns(3)
            d["backtest_start"] = c1.selectbox("Backtest: first origin", allq, index=allq.index(d["backtest_start"]) if d["backtest_start"] in allq else 0)
            d["backtest_end"] = c2.selectbox("Backtest: last origin", allq, index=allq.index(d["backtest_end"]) if d["backtest_end"] in allq else len(allq) - 1)
            S.setdefault("ov", {})
            ov = S["ov"]
            ov["sample.reestimate_every"] = str(c3.selectbox("Re-estimate every", [1, 2, 4], index=2, format_func=lambda x: f"{x} quarter(s)"))
            t1, t2, t3 = st.columns(3)
            from src.utils.multi import suggest_form
            try:
                _sug = suggest_form(P["bank"]["target_column"], P["series"])
            except Exception:
                _sug = "auto"
            ov["target.transform"] = t1.selectbox(
                "How the target is modelled", ["auto", "dlog", "diff"], index=2 if _sug == "diff" else 0,
                format_func=lambda k: {"auto": "auto (log growth if always positive)", "dlog": "log growth (balances, volumes)",
                                       "diff": "quarterly change (interest rates, spreads)"}[k],
                help="diff: errors are in the target's units (percentage points for a rate) instead of percent")
            lab_ = t2.text_input("Target name (titles, deck)", value="")
            if lab_:
                ov["target.label"] = setup.yq(lab_)
            un_ = t3.text_input("Units", value="")
            if un_:
                ov["target.units"] = setup.yq(un_)
            c1, c2, c3 = st.columns(3)
            ov["sample.horizon"] = str(c1.slider("Forecast horizon (quarters)", 4, 16, 12))
            ov["sample.backtest_step"] = str(c2.selectbox("Backtest origins every", [1, 2, 4], index=0, format_func=lambda x: f"{x} quarter(s)"))
            ov["oos.step"] = str(c3.selectbox("Out-of-sample origins every", [1, 2, 4], index=2, format_func=lambda x: f"{x} quarter(s)"))
            d["dummies"] = st.multiselect("One-off quarters (each gets a dummy and is left out of error measures)", allq, default=d["dummies"])

            if P["bank"].get("panel_column") and len(P.get("panel_banks") or []) == 1:
                st.caption(f"One credit union selected ({P['panel_banks'][0]}): the settings below are for its "
                           "single-series run. Tick two or more in ② for the panel.")
            if P["bank"].get("panel_column") and len(P.get("panel_banks") or []) > 1:
                st.subheader("Panel: several banks")
                st.caption("Drivers are chosen on all banks together (a pooled screen and search with Driscoll–Kraay "
                           "standard errors); then each bank is forecast by the panel models below and an AR(1) "
                           "benchmark, scored bank by bank. The competing single-series models below are used only "
                           "if you also ask for the full single-series run per bank.")
                p1, p2, p3 = st.columns([2, 1.2, 1])
                ov["panel.models"] = setup.yq(p1.multiselect("Panel models", list(PANEL_KINDS), default=list(PANEL_KINDS),
                                                             format_func=lambda k: PANEL_KINDS[k], key="pn_models"))
                ov["panel.driver_sets"] = setup.yq(p2.multiselect(
                    "Driver sets", ["panel", "core"], default=["panel", "core"], key="pn_sets",
                    format_func=lambda k: {"panel": "chosen on all banks", "core": "literature core"}[k]))
                ov["panel.min_obs"] = str(p3.number_input("Quarters for its own equation", 8, 80, 20, key="pn_min",
                                                          help="A bank with fewer is fully pooled"))
                bnames = list(P["banks"]["bank"]) if isinstance(P.get("banks"), pd.DataFrame) else []
                chosen = P.get("panel_banks") or bnames
                own = P.get("own_banks") or []
                # all banks ticked -> leave the list empty, so banks added to the file later are included
                ov["panel.banks"] = setup.yq([] if set(chosen) == set(bnames) else list(chosen))
                ov["panel.per_bank_full"] = setup.yq(list(own)) if own else "false"
                st.caption(f"Banks: {', '.join(chosen) if chosen else 'none'} (choose them in ②, next to the bank "
                           f"column); own full run: {', '.join(own) or 'none'}.")
                from src.panel.compare import METRICS as _PM
                ov["panel.evaluation.metric"] = st.selectbox(
                    "Rank the models on", list(_PM), format_func=_PM.get, key="pn_metric",
                    help="Used for the leaderboards and each credit union's best model. MAPE is the average absolute "
                         "error; RMSE and the tail measures punish occasional large misses. You can change it later "
                         "in ⑤ without re-estimating.")
                gone = []
                if isinstance(P.get("banks"), pd.DataFrame) and {"bank", "last"} <= set(P["banks"].columns):
                    _e = pd.PeriodIndex(P["banks"]["last"].astype(str), freq="Q")
                    gone = [f"{b} (last {q})" for b, q in zip(P["banks"]["bank"], _e) if (_e.max() - q).n >= 2]
                drop = st.checkbox("Leave out banks that stopped reporting (exited, failed, acquired)", False, key="pn_drop",
                                   help="Banks whose data end at least 2 quarters before the latest quarter. Keeping "
                                        "them uses their history (often their downturns) in the estimation; leaving "
                                        "them out estimates on surviving banks only.")
                ov["panel.drop_exited"] = "true" if drop else "false"
                if gone:
                    st.caption(("Left out: " if drop else "Would be left out: ") + ", ".join(gone))
                elif drop:
                    st.caption("No bank stopped reporting: nothing is left out.")
            st.subheader("Transforms tried for each candidate")
            st.caption("Levels and indexes (metadata transform = dlog) are tried in the level forms; rates (diff) in the rate forms. "
                       "Every form must pass the ADF test, and each is tried at lags 0 to the maximum.")
            LV = {"dlog": "q/q growth", "yoy": "y/y growth", "dlog2": "2-quarter growth", "acc": "acceleration", "gap": "gap from 3-year average"}
            RT = {"diff": "q/q change", "diff2": "2-quarter change", "diff4": "y/y change", "lvl": "level", "rgap": "gap from 3-year average"}
            c1, c2 = st.columns(2)
            ov["features.transforms_level"] = setup.yq(c1.multiselect("Level forms", list(LV), default=list(LV), format_func=lambda k: f"{k}: {LV[k]}"))
            ov["features.transforms_rate"] = setup.yq(c2.multiselect("Rate forms", list(RT), default=list(RT), format_func=lambda k: f"{k}: {RT[k]}"))
            c1, c2, c3 = st.columns(3)
            ov["features.moving_average"] = setup.yq(c1.multiselect(
                "Moving averages (quarters)", [6, 8, 12, 16], default=[],
                help="Average q/q growth (change for rates) over the last k quarters. 4 and 2 are left out: they equal "
                     "yoy/4 and dlog2/2."))
            ov["features.volatility"] = setup.yq(c2.multiselect(
                "Volatility windows (quarters)", [4, 6, 8, 12], default=[],
                help="Rolling standard deviation of q/q growth (change for rates) over the last k quarters"))
            ov["features.volatility_sign"] = str(c3.selectbox("Expected sign of volatility", [-1, 0, 1], index=0,
                                                              help="-1: more uncertainty, lower growth"))
            c1, c2 = st.columns(2)
            ov["features.max_lag"] = str(c1.slider("Longest lag (quarters)", 0, 8, 4))
            ov["features.adf_p"] = str(c2.selectbox("ADF test: keep a form if p <", [0.01, 0.05, 0.10], index=1))

            st.subheader("Driver selection")
            c1, c2, c3, c4 = st.columns(4)
            ov["final_model.max_vars"] = str(c1.slider("Most drivers in a model", 1, 6, 3))
            ov["final_model.t_min"] = str(c2.number_input("Smallest |t| allowed", 1.0, 4.0, 1.96, 0.01))
            ov["final_model.vif_max"] = str(c3.number_input("Largest VIF allowed", 2.0, 20.0, 5.0, 0.5))
            ov["approaches"] = setup.yq(c4.multiselect(
                "Selection approaches", ["screen_first", "en_first", "bayes_first", "factors_first", "stable_first"],
                default=["screen_first", "en_first", "bayes_first", "factors_first", "stable_first"],
                help="screen_first: one variable at a time; en_first: Elastic Net on all at once; bayes_first: Bayesian "
                     "model averaging (inclusion probabilities), which also adds the 'Bayesian model average' model; "
                     "factors_first: one index per theme from all its series; stable_first: variables a lasso picks "
                     "in most sub-samples (stability selection)"))
            ov["factors.enabled"] = "true" if "factors_first" in S["ov"].get("approaches", "factors_first") else "false"
            ov["stability.enabled"] = "true" if "stable_first" in S["ov"].get("approaches", "stable_first") else "false"
            ov["features.form_rule"] = c4.selectbox(
                "Forms per variable", ["all", "by_type"], index=0, key="form_rule",
                help="all: every transformation (growth, annual growth, acceleration, gaps ...); by_type: one form per "
                     "variable (growth for levels, change for rates), about 4x fewer candidates and fewer chance picks")

            st.subheader("How the economy affects the target")
            st.caption("An explanatory model built to measure effects rather than to win the forecast race: the focus "
                       "drivers are always in, and controls are chosen by double selection (they predict the target or a "
                       "focus driver). After the run: sensitivities, what-if shocks, why the scenarios differ, and how "
                       "far each effect can be trusted.")
            e1, e2, e3 = st.columns([1.2, 3, 1])
            imp_on = e1.checkbox("Explanatory model and impact analysis", True, key="imp_on")
            ov["impact.enabled"] = "true" if imp_on else "false"
            ftxt = e2.text_input("Focus drivers (comma-separated)", "", key="imp_focus",
                                 placeholder="e.g. unemployment, house_prices, FRMORT5Y.IC   (empty = the literature core)",
                                 help="Literature concept ids, series names, or exact features such as FLBR.ION|diff|L1. "
                                      "A concept or series without a form and lag gets the form and lag with the "
                                      "strongest effect of the expected sign.")
            ov["impact.focus"] = setup.yq([x.strip() for x in ftxt.split(",") if x.strip()])
            ov["impact.max_controls"] = str(e3.number_input("Most controls", 0, 8, 4, key="imp_mc"))

            st.subheader("Choosing the recommended model")
            st.caption("Every model competes. Score = accuracy + interpretation + sensitivity to the economy, with "
                       "these weights; a model must be within the error limit and respond to the focus drivers.")
            r1, r2, r3, r4, r5 = st.columns(5)
            wa = r1.slider("Accuracy", 0.0, 1.0, 0.5, 0.05, key="rec_wa")
            wi = r2.slider("Interpretation", 0.0, 1.0, 0.25, 0.05, key="rec_wi")
            ws = r3.slider("Sensitivity", 0.0, 1.0, 0.25, 0.05, key="rec_ws")
            ov["recommend.weights"] = f"{{accuracy: {wa:g}, interpretation: {wi:g}, sensitivity: {ws:g}}}"
            ov["recommend.max_error_ratio"] = str(r4.number_input("Error limit (x best)", 1.0, 10.0, 1.5, 0.1,
                                                                  key="rec_mer"))
            ov["recommend.min_coverage"] = str(r5.number_input("Must respond to (share of focus drivers)", 0.0, 1.0,
                                                               0.25, 0.05, key="rec_mc"))

            st.subheader("Competing models")
            c1, c2, c3 = st.columns(3)
            for k, lab in [("benchmarks", "Benchmarks (AR(1), mean growth)"), ("literature", "Literature models"), ("hybrids", "Hybrids")]:
                ov[f"models.{k}"] = "true" if c1.checkbox(lab, True, key=f"m_{k}") else "false"
            ov["models.machine_learning"] = setup.yq(c2.multiselect("Machine learning", ML, default=ML))
            ov["models.statistical"] = setup.yq(c2.multiselect("Statistical time series", STAT, default=STAT))
            ov["models.regime.models"] = setup.yq(c3.multiselect("Regime switching", REGIME, default=REGIME))
            deep_on = c3.checkbox("Deep learning (TiDE, N-HiTS, TFT; needs darts, adds about 6 minutes)", False, key="m_deep")
            ov["models.deep.enabled"] = "auto" if deep_on else "false"
            ov["models.foundation.enabled"] = "auto" if c3.checkbox("AI foundation models (Chronos; needs internet once)", False, key="m_fm") else "false"
            ov["models.tested.enabled"] = "true" if c1.checkbox(
                "Tested twins", True, key="m_tested",
                help="Every model with drivers also runs as '(tested)': its drivers re-checked until each passes the "
                     "same tests as the data-driven approaches (|t| ≥ 1.96, expected sign, VIF ≤ 5, one per theme); "
                     "failing drivers are re-lagged or dropped. The original stays.") else "false"

            st.markdown("**Auto-ARDL (as in EViews)**: the lags of target growth and of each driver are chosen by an "
                        "information criterion, again at every re-estimation; plus an ARDL in levels with the "
                        "Pesaran-Shin-Smith bounds test for a long-run relation.")
            a1, a2, a3, a4, a5 = st.columns([1.1, 1, 1, 1.6, 1.3])
            aa_on = a1.checkbox("Auto-ARDL models", True, key="aa_on")
            ov["models.auto_ardl.enabled"] = "true" if aa_on else "false"
            ov["models.auto_ardl.criterion"] = a2.selectbox("Criterion", ["bic", "aic", "hq"], key="aa_ic",
                                                            help="BIC (Schwarz) picks shorter lags; AIC longer ones")
            ov["models.auto_ardl.max_p"] = str(a3.number_input("Most lags of growth", 1, 8, 4, key="aa_p"))
            ov["models.auto_ardl.max_q"] = str(a3.number_input("Most lags of each driver", 0, 8, 4, key="aa_q"))
            ov["models.auto_ardl.on"] = setup.yq(a4.multiselect("Driver sets", ["core", "screen_first", "en_first", "bayes_first"],
                                                                default=["core", "screen_first"], key="aa_on_sets",
                                                                help="core = the literature core model's drivers"))
            ov["models.auto_ardl.levels"] = "true" if a5.checkbox("Levels ARDL + bounds test", True, key="aa_lv") else "false"

            st.markdown("**Horizon combinations**: the model best at 1-4 quarters ahead for those quarters, the best at 5-8 "
                        "for the next, and so on; chosen at each forecast origin from errors already known then.")
            h1, h2, h3, h4 = st.columns([1.1, 2, 1, 1.2])
            ov["models.horizon_combination.enabled"] = "true" if h1.checkbox("Horizon combinations", True, key="hc_on") else "false"
            btxt = h2.text_input("Blocks of quarters ahead", "1-4, 5-8, 9-12", key="hc_b",
                                 help="e.g. 1-2, 3-6, 7-12. Quarters not covered form one more block.")
            try:
                bl = [[int(x.split("-")[0]), int(x.split("-")[-1])] for x in btxt.replace(" ", "").split(",") if x]
                ov["models.horizon_combination.buckets"] = "[" + ", ".join(f"[{a}, {b}]" for a, b in bl) + "]"
            except ValueError:
                st.warning("Write the blocks as 1-4, 5-8, 9-12")
            ov["models.horizon_combination.top_k"] = str(h3.number_input("Models in the weighted one", 1, 6, 3, key="hc_k"))
            ov["models.horizon_combination.criterion"] = h4.selectbox("Judged by", ["growth", "balance"], key="hc_c",
                                                                      help="growth: error of the growth in those quarters "
                                                                           "(what is spliced); balance: balance error at "
                                                                           "those horizons")

            st.markdown("**Same growth as another variable**: the target grows at the same rate as the variable (two models "
                        "each: exactly, and plus the average historical gap). Pick level variables such as income, GDP "
                        "or house prices, not rates.")
            from src.features.transforms import RATE_TRS
            lit_ids = [c for c in P["lit"]["concept_id"].astype(str)
                       if str(P["lit"].set_index("concept_id").loc[c, "transform"] if "transform" in P["lit"] else "dlog")
                       not in RATE_TRS]
            sg_opts = lit_ids + [c for c in sorted(P["base"].columns) if c not in lit_ids]
            sg_def = [c for c in ["disposable_income", "house_prices"] if c in sg_opts]
            ov["models.same_growth"] = setup.yq(st.multiselect("Variables (literature concepts first, then your series)",
                                                               sg_opts, default=sg_def, key="m_same"))

            st.markdown("**Run only some models** (leave empty to run every model ticked above). Pick families or "
                        "single models; the AR(1) benchmark is always kept for the comparison.")
            from src.models.zoo import FAMILIES
            from src.models.ml import LABELS as ML_LABELS
            only_opts = FAMILIES + ["Screen first", "Elastic Net first", "Bayesian first", "Bayesian model average"] + [m["name"] for m in P["lit_models"]] \
                + [ML_LABELS[k] for k in ML if k in ML_LABELS] + ["Same growth as"]
            ov["models.only"] = setup.yq(st.multiselect("Models or families to run", list(dict.fromkeys(only_opts)),
                                                        default=[], key="m_only",
                                                        help="'Same growth as' keeps every same-growth model; a family "
                                                             "keeps all its models"))

    # ④ check and run
    with tabs[3]:
        st.header("Check and run")
        if not P:
            st.info("Upload the files in ① first.")
        else:
            if st.button("Save the configuration and check it", type="primary"):
                path, lit_path = setup.write_config(P, name, overrides=S.get("ov", {}), log=lambda *_: None)
                S["saved"] = path
                st.success(f"Saved `{setup.rel(path)}` and `{setup.rel(lit_path)}`")
            if S.get("saved") == cfg_path and os.path.exists(cfg_path):
                show_report(run_check(cfg_path))
                run_buttons(cfg_path, "new")
else:
    with tabs[3]:
        st.header("Check and run")
        if is_panel(load_config(cfg_path)):
            st.info("Panel configuration: several banks in one file, analysed together (run_panel.py).")
        show_report(run_check(cfg_path))
        run_buttons(cfg_path, "old")
    with tabs[5]:
        st.header("Edit the config file")
        st.caption("Every setting, with its explanation. Saved as it is; the check in ④ runs on the saved file.")
        txt = open(cfg_path, encoding="utf-8").read()
        new = st.text_area("config", txt, height=600, label_visibility="collapsed")
        if st.button("Save"):
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(new)
            st.success("Saved")

# ⑤ results
with tabs[4]:
    st.header("Results")
    try:
        out = load_config(cfg_path)["out_dir"]
    except Exception:
        out = None
    latest_out = out
    try:
        _cfg5 = load_config(cfg_path)
    except Exception:
        _cfg5 = None
    if out and _cfg5 is not None and is_panel(_cfg5):
        panel_results(out, _cfg5)
        st.subheader("Process story")
        st.caption("What happened at every step of this run, for an audience: the data, each selection stage with its "
                   "numbers, how the models were tested, the recommended model and the scenarios. A long HTML page and "
                   "a PowerPoint deck (about 50 slides, full text in the speaker notes). No Quarto needed; a few seconds.")
        story_dir = os.path.join(out, "process_story")
        story_files = [os.path.join(story_dir, f) for f in ("process_story.html", "process_story.pptx")]
        story_theme = story_theme_picker()
        if st.button("Rebuild the process story" if os.path.exists(story_files[0]) else "Build the process story",
                     key="build_story"):
            if run_script(os.path.join("tools", "process_story.py"), cfg_path,
                          ["--run-dir", out, "--theme", story_theme]) != 0:
                st.error("The process story could not be built: see the log above (python-pptx and matplotlib are needed).")
        if story_theme == "all":
            story_gallery(story_dir)
        s1, s2 = st.columns(2)
        for col_, f_, mime_ in zip((s1, s2), story_files, ("text/html",
                                   "application/vnd.openxmlformats-officedocument.presentationml.presentation")):
            if os.path.exists(f_):
                col_.download_button(f"⬇ {os.path.basename(f_)}", open(f_, "rb").read(), file_name=os.path.basename(f_),
                                     mime=mime_, key=f"dl_{os.path.basename(f_)}")
        if S.get("last_log"):
            with st.expander("Log of the last run"):
                st.code(S["last_log"][-20000:], language=None)
        st.stop()
    if out:
        from src.utils.saved import list_snapshots
        snaps = list_snapshots(out)
        if snaps:
            def _lab(x):
                t = x.get("saved", x["name"])
                t = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[9:11]}:{t[11:13]}" if len(t) >= 13 else t
                b = (f" · best {x['best_model']} {x['best_error_pct']:.2f}% ({x.get('basis', '')})"
                     if x.get("best_model") else "")
                return f"{t} · {x.get('script', '')}{b}"
            opts = ["Latest results"] + [_lab(x) for x in snaps]
            pick_run = st.selectbox("Results to show", opts, key="run_pick",
                                    help="Every run keeps a complete copy of its results (outputs/<run>/saved_runs/), "
                                         "so you can look at an earlier run without running anything again.")
            if pick_run != "Latest results":
                out = snaps[opts.index(pick_run) - 1]["path"]
                st.caption(f"Showing the saved run in `{os.path.relpath(out, HERE)}` (inputs used: its `inputs/` folder).")
    comp = read_result(os.path.join(out, "comparison", "1_model_comparison.csv"), need=("model", "family")) \
        if out else pd.DataFrame()
    hz = [int(c.split("_")[2]) for c in comp.columns if c.startswith("avg_1_") and c.endswith("_pct")]
    if not len(comp) or not hz:
        st.info("No results yet (or the last run stopped before writing the model comparison): run the pipeline in ④.")
    else:
        H = max(hz)
        cols = ["model", "family"] + [c for c in comp.columns if c.startswith("avg_1_")]
        oo = read_result(os.path.join(out, "oos", "1_oos_summary.csv"), need=("model", "family", f"avg_1_{H}_pct"))
        oo = oo[oo["family"] != "Look-ahead check"] if len(oo) else oo
        if len(oo):
            from src.reporting.report import rank_column
            oo = oo.sort_values(rank_column(oo, H))
            best = oo.iloc[0]
            ar = oo[oo["model"] == "AR(1) benchmark"]
            st.success(f"**Most accurate out of sample: {best['model']}** ({best['family']}): average balance error "
                       f"{best[f'avg_1_{H}_pct']:.2f}% over 1–{H} quarters"
                       + (f", against {ar.iloc[0][f'avg_1_{H}_pct']:.2f}% for the AR(1) benchmark" if len(ar) else "")
                       + f". {len(oo)} models compared.")
            st.subheader("Ranking out of sample (drivers re-selected at every origin)")
            st.caption("Average absolute error of the balance, in %, over 1 … h quarters ahead. Lower is better. Use this table to rank models.")
            st.dataframe(friendly(oo[[c for c in cols if c in oo.columns]]), hide_index=True, use_container_width=True)
        else:
            st.info("Run the out-of-sample test in ④ for the fair ranking; below is the backtest only.")
        recommendation_section(out)
        impact_section(out, cfg_path)
        formula_section(out)
        st.subheader("Rolling backtest' (drivers chosen with all the data: optimistic for data-driven models)")
        st.dataframe(friendly(comp.sort_values(f"avg_1_{H}_pct")[cols]), hide_index=True, use_container_width=True)
        imgs = [p for p in [os.path.join(out, "comparison", "model_comparison.png"), os.path.join(out, "comparison", "scenarios.png"),
                            os.path.join(out, "oos", "oos_mape_by_horizon.png"), os.path.join(out, "oos", "holdout.png"),
                            os.path.join(out, "comparison", "regimes.png")] if os.path.exists(p)]
        for p in imgs:
            wide_image(p)
        model_details(out, comp, H)
        st.subheader("Downloads")
        c = st.columns(4)
        files = [("Tables (Excel)", os.path.join(out, "comparison", "tables.xlsx")), ("Summary", os.path.join(out, "summary.md"))]
        tr = sorted(glob.glob(os.path.join(out, "trace", "*", "trace.html")))
        if tr:
            files.append(("Run trace and data story", tr[-1]))
        oo_x = os.path.join(out, "oos", "tables.xlsx")
        if os.path.exists(oo_x):
            files.append(("Out-of-sample tables", oo_x))
        for lab_, p_ in (("Test-sample forecasts (backtest, CSV)", os.path.join(out, "comparison", "12_backtest_forecasts.csv")),
                         ("Test-sample forecasts (out of sample, CSV)", os.path.join(out, "oos", "6_all_forecast_errors.csv")),
                         ("Fitted models (pickle)", os.path.join(out, "fitted", "models.pkl")),
                         ("Fitted models (plain JSON)", os.path.join(out, "fitted", "formula_models.json"))):
            if os.path.exists(p_):
                files.append((lab_, p_))
        for i, (lab, p) in enumerate(files):
            if os.path.exists(p):
                c[i % 4].download_button(lab, open(p, "rb").read(), file_name=os.path.basename(p), key=f"dl{i}")

        st.subheader("Presentation")
        st.caption("The styled PowerPoint deck (methodology, literature, model comparison, recommended model, scenarios), "
                   "built from this run's results; run the out-of-sample test in ④ first for the full deck. Needs Quarto (quarto.org) on this computer; takes about half a minute.")
        sys.path.insert(0, os.path.join(HERE, "tools"))
        from make_deck import find_quarto
        cfg_now = load_config(cfg_path)
        deck = os.path.join(out, f"{cfg_now['run_name']}_deck.pptx")
        run_dir_args = [] if out == latest_out else ["--run-dir", out]
        reg0 = (cfg_now.get("screen_first") or {}).get("primary_geography") or ""
        d1, d2, d3 = st.columns([3, 1, 1])
        tt_ = (cfg_now.get("target") or {}).get("label") or str(cfg_now["data"]["bank"]["target_column"]).replace("_", " ").title()
        title = d1.text_input("Title", value=f"Forecasting the {reg0} {tt_}" if reg0 else f"Forecasting the {tt_}",
                              key="deck_title")
        region = d2.text_input("Region in the text", value=reg0, key="deck_region")
        sim = d3.checkbox("Simulated-data notes", value=False, key="deck_sim",
                          help="Adds 'results on simulated data' to every slide: for demonstrations only")
        from make_deck import missing_packages
        miss = missing_packages(sys.executable)
        if miss:
            st.warning(f"The deck needs **{', '.join(miss)}** in the Python that runs this app, which is not where "
                       f"your `pip install` went:\n\n`{sys.executable}`")
            st.caption("Several Pythons on one computer are common (Anaconda, virtual environments, the system Python). "
                       "The button installs into the right one; or run the command below in a terminal.")
            st.code(f'"{sys.executable}" -m pip install ' + " ".join(miss), language=None)
            if st.button("Install them now", key="install_deck_pkgs"):
                with st.spinner("Installing …"):
                    r = subprocess.run([sys.executable, "-m", "pip", "install", *miss], capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
                if r.returncode == 0:
                    st.success("Installed. Click Build the presentation.")
                    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                else:
                    st.error("pip could not install them; the message is below. Run the command above in a terminal instead.")
                    st.code((r.stdout + r.stderr)[-3000:], language=None)
        elif not find_quarto():
            st.warning("Quarto is not installed, so the deck cannot be built here. Install it from "
                       "https://quarto.org/docs/get-started/ and restart the app (or set the QUARTO environment variable to its path).")
        elif st.button("Rebuild the presentation" if os.path.exists(deck) else "Build the presentation", type="primary"):
            extra = ["--title", title, "--region", region, "--out", deck] + run_dir_args + (["--simulated"] if sim else [])
            if run_script(os.path.join("tools", "make_deck.py"), cfg_path, extra) != 0:
                st.error("The deck could not be built: see the log above. Most often a missing package "
                         "(pip install jupyter python-pptx) or a pipeline run that has not finished.")
        if os.path.exists(deck):
            st.download_button("⬇ Download the presentation (.pptx)", open(deck, "rb").read(), file_name=os.path.basename(deck),
                               mime="application/vnd.openxmlformats-officedocument.presentationml.presentation", key="dl_deck")
        st.subheader("Process story")
        st.caption("What happened at every step of this run, for an audience: the data, each selection stage with its "
                   "numbers, how the models were tested, the recommended model and the scenarios. A long HTML page and "
                   "a PowerPoint deck (about 50 slides, full text in the speaker notes). No Quarto needed; a few seconds.")
        story_dir = os.path.join(out, "process_story")
        story_files = [os.path.join(story_dir, f) for f in ("process_story.html", "process_story.pptx")]
        story_theme = story_theme_picker()
        if st.button("Rebuild the process story" if os.path.exists(story_files[0]) else "Build the process story",
                     key="build_story"):
            if run_script(os.path.join("tools", "process_story.py"), cfg_path,
                          ["--run-dir", out, "--theme", story_theme]) != 0:
                st.error("The process story could not be built: see the log above (python-pptx and matplotlib are needed).")
        if story_theme == "all":
            story_gallery(story_dir)
        s1, s2 = st.columns(2)
        for col_, f_, mime_ in zip((s1, s2), story_files, ("text/html",
                                   "application/vnd.openxmlformats-officedocument.presentationml.presentation")):
            if os.path.exists(f_):
                col_.download_button(f"⬇ {os.path.basename(f_)}", open(f_, "rb").read(), file_name=os.path.basename(f_),
                                     mime=mime_, key=f"dl_{os.path.basename(f_)}")
        if S.get("last_log"):
            with st.expander("Log of the last run"):
                st.code(S["last_log"][-20000:], language=None)
