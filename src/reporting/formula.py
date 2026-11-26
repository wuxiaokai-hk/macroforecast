"""The recommended model as a formula you can use by hand, and its forecast for the next quarters.

Only models that are an explicit equation qualify: ARDL regressions (data-driven, literature, hybrid, auto-ARDL),
the error-correction model, the ARDL in levels and the same-growth rules. Machine learning, networks, regime-switching, combinations and
foundation models have no closed formula; when one of them is the most accurate overall it is reported next to
the best formula model, with its forecast but without an equation.

Outputs (outputs/<run>/formula/):
  formula.md                         the equation with its coefficients and what every symbol means
  1_terms.csv                        coefficient of every term
  2_next_quarters.csv                the next quarters under every scenario, term by term
  3_best_overall_next_quarters.csv   the most accurate model overall (when it is not the formula model)
  recommended_model_calculator.xlsx  the same calculation with live Excel formulas: change a driver's future
                                     value (yellow cells) and the growth and balance update
"""
import os
import re

import numpy as np
import pandas as pd

from ..features.transforms import TR_DESC, lookback, parse_feature, tr_desc, window
from ..utils import target as tg

FORMULA_TYPES = ("ARDL", "ECM", "SameGrowth", "AutoARDL", "LevelsARDL")
LOOKBACK = {"dlog": 1, "rdlog": 1, "dlog2": 2, "yoy": 4, "ryoy": 4, "acc": 8, "gap": 11, "rgap": 11,
            "diff": 1, "diff2": 2, "diff4": 4, "lvl": 0}


def has_formula(model):
    """True when the model is an explicit equation (ARDL, error-correction model, same-growth rule).

    Args:
        model: any model.

    Returns:
        bool.
    """
    return type(model).__name__ in FORMULA_TYPES


def choose(ranking, models, col, order=None):
    """The recommended model and the best model with a formula.

    With `order` (model names, best first: the scorecard, src/selection/scorecard.py) the first is the recommended
    model and the first with a formula is the formula model; without it, the most accurate ones.

    Args:
        ranking: table with model, family and the error column. models: dict name -> fitted model.
        col: error column (lower is better), e.g. avg_1_12_pct.

    Returns:
        (best overall row, best formula row or None) as Series.
    """
    r = ranking[(ranking["family"] != "Look-ahead check") & ranking[col].notna()].sort_values(col)
    if r.empty:
        return None, None
    if order:
        pos = {m: i for i, m in enumerate(order)}
        r = r.assign(_o=r["model"].map(pos).fillna(len(pos))).sort_values(["_o", col]).drop(columns="_o")
    best = r.iloc[0]
    f = r[r["model"].map(lambda m: m in models and has_formula(models[m]))]
    return best, (f.iloc[0] if len(f) else None)


# ------------------------------------------------------------------------------------------- the equation
def _x(code, k):
    """Symbol of a series k quarters back, e.g. FHPI.ION(t-2)."""
    return f"{code}(t)" if k == 0 else f"{code}(t-{k})"


def definition(feature, deflator=None):
    """What a feature is, as a formula in the raw series.

    Args:
        feature: "code|transform|Llag". deflator: price index used by real transforms.

    Returns:
        str, e.g. "ln(FHPI.ION(t-1) / FHPI.ION(t-2))".
    """
    code, tr, l = parse_feature(feature)
    X = lambda k: _x(code, l + k)                                   # noqa: E731
    P = lambda k: _x(deflator or "CPI", l + k)                       # noqa: E731
    return {"dlog": f"ln({X(0)} / {X(1)})", "dlog2": f"ln({X(0)} / {X(2)})", "yoy": f"ln({X(0)} / {X(4)})",
            "acc": f"ln({X(0)} / {X(4)}) - ln({X(4)} / {X(8)})",
            "gap": f"ln {X(0)} - average of ln {code} over {_x(code, l)}..{_x(code, l + 11)}",
            "rdlog": f"ln({X(0)} / {X(1)}) - ln({P(0)} / {P(1)})", "ryoy": f"ln({X(0)} / {X(4)}) - ln({P(0)} / {P(4)})",
            "diff": f"{X(0)} - {X(1)}", "diff2": f"{X(0)} - {X(2)}", "diff4": f"{X(0)} - {X(4)}", "lvl": X(0),
            "rgap": f"{X(0)} - average of {code} over {_x(code, l)}..{_x(code, l + 11)}"}.get(tr, _window_def(code, tr, l, feature))


def _window_def(code, tr, l, feature):
    """Definition of a moving-window transform (ma, vol, dma, dvol)."""
    w = window(tr)
    if not w:
        return feature
    rate, kind, k = w
    step = f"{code}(s) - {code}(s-1)" if rate else f"ln({code}(s) / {code}(s-1))"
    rng = f"over the {k} quarters ending {_x(code, l).replace(code + '(', '(')}"
    return f"{'average' if kind == 'ma' else 'standard deviation'} of {step} {rng}"


def _lr_def(item, deflator):
    """Definition of an error-correction long-run regressor."""
    c, form = item["mnemonic"], item.get("form", "log")
    return {"log": f"ln {c}(t)", "log_real": f"ln({c}(t) / {deflator}(t))", "level": f"{c}(t)",
            "real_rate": f"{c}(t) - 100 * ln({deflator}(t) / {deflator}(t-4))"}.get(form, c)


def terms(model, meta):
    """The equation term by term.

    Args:
        model: fitted formula model. meta: metadata (descriptions).

    Returns:
        DataFrame: symbol, term, coefficient, definition, meaning.
    """
    p = model.params
    if getattr(model, "kind", "") == "LevelsARDL":
        return terms_levels(model)
    rows = [{"symbol": "c", "term": "const", "coefficient": float(p.get("const", 0.0)), "definition": "1",
             "meaning": "constant (average quarterly growth not explained by the drivers)"}]
    for i in range(1, n_ylags(p) + 1):
        if f"y_lag{i}" in p:
            rows.append({"symbol": f"g(t-{i})", "term": f"y_lag{i}", "coefficient": float(p[f"y_lag{i}"]),
                         "definition": f"ln(L(t-{i}) / L(t-{i + 1}))",
                         "meaning": "last quarter's target growth" if i == 1 else f"target growth {i} quarters earlier"})
    defl = getattr(model, "deflator", None)
    for i, f in enumerate(model.features, 1):
        if f not in p:
            continue
        code, tr, lag = parse_feature(f)
        desc = meta.loc[code, "description"] if code in meta.index else code
        when = "same quarter" if lag == 0 else f"{lag} quarter{'s' if lag > 1 else ''} earlier"
        rows.append({"symbol": f"x{i}(t)", "term": f, "coefficient": float(p[f]), "definition": definition(f, defl),
                     "meaning": f"{desc}: {tr_desc(tr)}, {when}"})
    if "ect_lag1" in p and getattr(model, "beta", None) is not None:
        rows.append({"symbol": "ect(t-1)", "term": "ect_lag1", "coefficient": float(p["ect_lag1"]),
                     "definition": "ln(L*(t-1) / " + f"{defl}(t-1)) - [b0 + sum of b_k * z_k(t-1)], L* = organic balance "
                                   "(one-off acquisitions and sales removed)",
                     "meaning": "last quarter's gap of the real target from their long-run level (error correction)"})
        rows.append({"symbol": "b0", "term": "long run: constant", "coefficient": float(model.beta[0]),
                     "definition": "1", "meaning": "long-run relation: constant"})
        for k, (item, b) in enumerate(zip(model.long_run, model.beta[1:]), 1):
            rows.append({"symbol": f"z{k}(t)", "term": f"long run: {item['mnemonic']}", "coefficient": float(b),
                         "definition": _lr_def(item, defl), "meaning": f"long-run relation: {item['mnemonic']} "
                                                                       f"({item.get('form', 'log')})"})
    return pd.DataFrame(rows)


def n_ylags(p):
    """Number of lagged-growth terms in a parameter vector (y_lag1, y_lag2, ...)."""
    ks = [int(k[5:]) for k in p.index if str(k).startswith("y_lag") and str(k)[5:].isdigit()]
    return max(ks) if ks else 0


def terms_levels(model):
    """The terms of an ARDL in levels: y*(t) = c + sum a_i y*(t-i) + sum b_jk z_j(t-k).

    Args:
        model: fitted LevelsARDL.

    Returns:
        DataFrame: symbol, term, coefficient, definition, meaning.
    """
    p = model.params
    rows = [{"symbol": "c", "term": "const", "coefficient": float(p.get("const", 0.0)), "definition": "1",
             "meaning": "constant"}]
    for i in range(1, model.p + 1):
        rows.append({"symbol": f"y*(t-{i})", "term": f"ys_lag{i}", "coefficient": float(p.get(f"ys_lag{i}", 0.0)),
                     "definition": (f"ln(L*(t-{i}) / {model.deflator}(t-{i}))" if getattr(model, "real", True)
                                    and tg.is_log() else f"ln L*(t-{i})" if tg.is_log() else f"L*(t-{i})"),
                     "meaning": f"{model.target_label()}, {i} quarter{'s' if i > 1 else ''} earlier"})
    for j, (item, n) in enumerate(zip(model.long_run, model.lr_names), 1):
        for k in range(0, model.q[n] + 1):
            d = _lr_def(item, model.deflator).replace("(t)", f"(t-{k})" if k else "(t)").replace("(t-4)", f"(t-{k + 4})")
            rows.append({"symbol": f"z{j}(t-{k})" if k else f"z{j}(t)", "term": f"{n}|L{k}",
                         "coefficient": float(p.get(f"{n}|L{k}", 0.0)), "definition": d,
                         "meaning": f"{item['mnemonic']} ({item.get('form', 'log')})" + (f", {k} quarter(s) earlier" if k else "")})
    return pd.DataFrame(rows)


def equation_text(model, T):
    """The equation in one line plus the balance identity.

    Args:
        model: fitted formula model. T: terms() table.

    Returns:
        str (several lines).
    """
    def num(v):
        return f"{v:.4g}"
    def join(rows):
        out = ""
        rows = [(sym, v) for sym, v in rows if not (sym == "c" and v == 0 and len(rows) > 1)]
        for k, (sym, v) in enumerate(rows):
            piece = num(abs(v)) if sym in ("c", "b0") else (sym if abs(v) == 1 else f"{num(abs(v))}·{sym}")
            out += (("-" if v < 0 else "") + piece) if k == 0 else ((" - " if v < 0 else " + ") + piece)
        return out
    if getattr(model, "kind", "") == "LevelsARDL":
        if not tg.is_log():
            return "\n".join(["y*(t) = " + join(list(zip(T["symbol"], T["coefficient"]))),
                              "g(t) = y*(t) - y*(t-1)", "L(t) = L(t-1) + g(t)",
                              "y*(t) = L*(t), L* = organic level of the target"])
        return "\n".join(["y*(t) = " + join(list(zip(T["symbol"], T["coefficient"]))),
                          f"g(t) = y*(t) - y*(t-1) + ln({model.deflator}(t) / {model.deflator}(t-1))",
                          "L(t) = L(t-1) × exp(g(t))",
                          f"y*(t) = ln(L*(t) / {model.deflator}(t)), L* = organic level (real terms)"])
    is_lr = T["symbol"].str.match(r"^(b0|z\d)")
    s = "g(t) = " + join(list(zip(T.loc[~is_lr, "symbol"], T.loc[~is_lr, "coefficient"])))
    lines = [s, "L(t) = L(t-1) × exp(g(t))" if tg.is_log() else "L(t) = L(t-1) + g(t)"]
    if is_lr.any():
        z = join(list(zip(T.loc[is_lr, "symbol"], T.loc[is_lr, "coefficient"])))
        lines.append(f"ect(t) = ln(L*(t) / {model.deflator}(t)) - [{z}]   (L* = organic level)" if tg.is_log()
                     else f"ect(t) = L*(t) - [{z}]   (L* = organic level)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- worked forecast in Python
def worked(ctx, model, Q):
    """The next Q quarters under every scenario, term by term (identical to model.forecast).

    Args:
        ctx: Context. model: fitted formula model. Q: quarters.

    Returns:
        DataFrame: scenario, date, each driver's value and contribution, growth (log, % q/q, % annualised), balance.
    """
    p = model.params
    if getattr(model, "kind", "") == "LevelsARDL":
        return worked_levels(ctx, model, Q)
    origin = ctx.idx[ctx.end]
    L0 = float(ctx.level.iloc[-1])
    ny = n_ylags(p)
    rows = []
    for name, mdf in ctx.macro.items():
        P = model.panel(ctx, mdf)
        pos = P.index.get_loc(origin)
        q = min(Q, len(P) - pos - 1)
        past = list(P["y"].iloc[max(0, pos - max(ny, 1) + 1):pos + 1].values)
        L = L0
        ect, logL = None, None
        if "ect_lag1" in p and getattr(model, "beta", None) is not None:
            logL = float(P["__logL"].iloc[pos])
            z = np.r_[1.0, P[model.lr_names].iloc[pos].values]
            ect = (logL - P["__lcpi"].iloc[pos]) - z @ model.beta
        for h in range(1, q + 1):
            row = P.iloc[pos + h]
            r = {"scenario": name, "date": str(P.index[pos + h]), "constant": float(p.get("const", 0.0))}
            g = r["constant"]
            for i in range(1, ny + 1):
                if f"y_lag{i}" in p:
                    r[f"g(t-{i})"] = past[-i]
                    r[f"contribution g(t-{i})"] = p[f"y_lag{i}"] * past[-i]
                    g += r[f"contribution g(t-{i})"]
            for i, f in enumerate(model.features, 1):
                if f in p:
                    r[f"x{i} = {f}"] = float(row[f])
                    r[f"contribution x{i}"] = float(p[f] * row[f])
                    g += r[f"contribution x{i}"]
            if ect is not None:
                r["ect(t-1)"] = float(ect)
                r["contribution ect(t-1)"] = float(p["ect_lag1"] * ect)
                g += r["contribution ect(t-1)"]
            L = tg.level_after(L, g)
            r.update({"growth_log": g, "growth_pct_qq": tg.pct(g),
                      "growth_pct_annualised": tg.annualised(g), "balance": L})
            rows.append(r)
            past.append(g)
            if ect is not None:
                logL += g
                z = np.r_[1.0, row[model.lr_names].values]
                ect = (logL - row["__lcpi"]) - z @ model.beta
        check = np.asarray(model.forecast(P, origin, q))
        got = np.array([x["growth_log"] for x in rows if x["scenario"] == name])
        if len(check) and np.max(np.abs(check - got)) > 1e-9:
            raise RuntimeError(f"formula table differs from the model's forecast for {model.name} / {name}")
    return pd.DataFrame(rows)


def worked_levels(ctx, model, Q):
    """The next Q quarters of an ARDL in levels, term by term (identical to model.forecast).

    Args:
        ctx: Context. model: fitted LevelsARDL. Q: quarters.

    Returns:
        DataFrame: scenario, date, contributions, y*, inflation, growth, balance.
    """
    p = model.params
    origin = ctx.idx[ctx.end]
    L0 = float(ctx.level.iloc[-1])
    rows = []
    for name, mdf in ctx.macro.items():
        P = model.panel(ctx, mdf)
        pos = P.index.get_loc(origin)
        q = min(Q, len(P) - pos - 1)
        ys = list(P["__ys"].iloc[max(0, pos - model.p + 1):pos + 1].values)
        L = L0
        for h in range(1, q + 1):
            t = pos + h
            r = {"scenario": name, "date": str(P.index[t]), "constant": float(p.get("const", 0.0))}
            v = r["constant"]
            for i in range(1, model.p + 1):
                r[f"contribution y*(t-{i})"] = float(p.get(f"ys_lag{i}", 0.0) * ys[-i])
                v += r[f"contribution y*(t-{i})"]
            for j, n in enumerate(model.lr_names, 1):
                for k in range(0, model.q[n] + 1):
                    r[f"z{j} = {n[3:]} (t-{k})"] = float(P[n].iloc[t - k])
                    r[f"contribution z{j}(t-{k})"] = float(p.get(f"{n}|L{k}", 0.0) * P[n].iloc[t - k])
                    v += r[f"contribution z{j}(t-{k})"]
            infl = float(P["__lcpi"].iloc[t] - P["__lcpi"].iloc[t - 1])
            g = v - ys[-1] + infl
            L = tg.level_after(L, g)
            r.update({"y*": v, "inflation (log)": infl, "growth_log": g, "growth_pct_qq": tg.pct(g),
                      "growth_pct_annualised": tg.annualised(g), "balance": L})
            rows.append(r)
            ys.append(v)
        check = np.asarray(model.forecast(P, origin, q))
        got = np.array([x["growth_log"] for x in rows if x["scenario"] == name])
        if len(check) and np.max(np.abs(check - got)) > 1e-9:
            raise RuntimeError(f"formula table differs from the model's forecast for {model.name} / {name}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- Excel calculator
def _col(i):
    """Excel column letters for a 1-based index."""
    s = ""
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _excel_feature(tr, X, P, r, l):
    """Excel formula for a transform at row r with lag l. X(row) / P(row): cell of the series / deflator.

    Returns:
        formula text without the leading '='.
    """
    a = r - l
    w = window(tr)
    if w:                                    # moving windows over the last k quarters
        rate, kind, k = w
        if kind == "ma":
            return f"({X(a)}-{X(a - k)})/{k}" if rate else f"LN({X(a)}/{X(a - k)})/{k}"
        cur, prev = f"{X(a - k + 1)}:{X(a)}", f"{X(a - k)}:{X(a - 1)}"
        g = f"({cur}-{prev})" if rate else f"LN({cur}/{prev})"
        # SUMPRODUCT works on ranges in every Excel version (no array formula needed); same as pandas std (ddof 1)
        return f"SQRT(SUMPRODUCT(({g}-SUMPRODUCT({g})/{k})^2)/{k - 1})"
    if tr == "dlog":
        return f"LN({X(a)}/{X(a - 1)})"
    if tr == "dlog2":
        return f"LN({X(a)}/{X(a - 2)})"
    if tr == "yoy":
        return f"LN({X(a)}/{X(a - 4)})"
    if tr == "acc":
        return f"LN({X(a)}/{X(a - 4)})-LN({X(a - 4)}/{X(a - 8)})"
    if tr == "rdlog":
        return f"LN({X(a)}/{X(a - 1)})-LN({P(a)}/{P(a - 1)})"
    if tr == "ryoy":
        return f"LN({X(a)}/{X(a - 4)})-LN({P(a)}/{P(a - 4)})"
    if tr == "diff":
        return f"{X(a)}-{X(a - 1)}"
    if tr == "diff2":
        return f"{X(a)}-{X(a - 2)}"
    if tr == "diff4":
        return f"{X(a)}-{X(a - 4)}"
    if tr == "lvl":
        return f"{X(a)}"
    if tr == "gap":
        return f"LN({X(a)})-SUMPRODUCT(LN({X(a - 11)}:{X(a)}))/12"
    if tr == "rgap":
        return f"{X(a)}-AVERAGE({X(a - 11)}:{X(a)})"
    raise ValueError(tr)


def write_calculator(path, ctx, model, T, Q, info, best_other=None):
    """The recommended model as a live Excel calculator: one sheet per scenario.

    Blue cells are data (history and the scenario's future values); yellow cells are the future values to change;
    black cells are formulas. Each sheet rebuilds every driver from the raw series, then growth and the balance.

    Args:
        path: .xlsx to write. ctx: Context. model: fitted formula model. T: terms() table. Q: quarters ahead.
        info: dict of facts for the cover sheet (model, basis, error, window). best_other: optional DataFrame,
        the most accurate model overall when it has no formula (written as values).
    """
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    F = "Arial"
    BLUE = Font(name=F, color="0000FF")
    BOLD, TITLE = Font(name=F, bold=True), Font(name=F, bold=True, size=14)
    HEAD = Font(name=F, bold=True, color="FFFFFF")
    HFILL, YELLOW = PatternFill("solid", fgColor="1F6F8B"), PatternFill("solid", fgColor="FFFF00")
    GREY = PatternFill("solid", fgColor="EEF3F6")
    thin = Border(bottom=Side(style="thin", color="C9D3DC"))
    wb = Workbook()
    wb.calculation.fullCalcOnLoad = True                     # Excel computes every formula when the file opens

    # ---- cover / equation
    ws = wb.active
    ws.title = "Equation"
    ws["A1"], ws["A1"].font = f"Recommended model: {model.name}", TITLE
    lines = [f"Why this model: {info['why']}",
             f"Estimated on {info['window']} (quarterly). Balance at {info['origin']}: {info['L0']:,.1f}.",
             "How to use: open a scenario sheet, change the yellow cells (future values of the drivers); growth and "
             "the balance update. Blue = data and estimated coefficients, black = formulas, yellow = values to change."]
    for i, t in enumerate(lines, 2):
        ws.cell(i, 1, t).font = Font(name=F)
    r0 = len(lines) + 3
    ws.cell(r0, 1, "Equation").font = BOLD
    for k, line in enumerate(equation_text(model, T).split("\n"), 1):
        ws.cell(r0 + k, 1, line).font = Font(name="Consolas")
    r1 = r0 + 5
    hdr = ["Symbol", "Coefficient", "Definition", "Meaning", "Term in the code"]
    for j, h in enumerate(hdr, 1):
        c = ws.cell(r1, j, h)
        c.font, c.fill = HEAD, HFILL
    coef_ref = {}
    for i, (_, r) in enumerate(T.iterrows(), r1 + 1):
        ws.cell(i, 1, r["symbol"]).font = Font(name=F)
        c = ws.cell(i, 2, float(r["coefficient"]))
        c.font, c.number_format = BLUE, "0.000000"
        c.comment = Comment(f"Estimated by the pipeline on {info['window']}; change it only to test the formula.", "model")
        ws.cell(i, 3, r["definition"]).font = Font(name=F)
        ws.cell(i, 4, r["meaning"]).font = Font(name=F)
        ws.cell(i, 5, r["term"]).font = Font(name=F, color="808080")
        coef_ref[r["term"]] = f"Equation!$B${i}"
    rn = r1 + len(T) + 2
    notes = (["Symbols: g(t) = ln(L(t) / L(t-1)), quarterly growth of the target (log); L = target level; "
              "X(t-k) = series X, k quarters back; ln = natural log.",
              "Growth in % q/q = exp(g) - 1; annualised = exp(4g) - 1."] if tg.is_log() else
             ["Symbols: g(t) = L(t) - L(t-1), quarterly change of the target, in its own units; L = target level; "
              "X(t-k) = series X, k quarters back; ln = natural log.",
              "Change annualised = 4g."]) + [
             "One-off quarters (dummies) are zero in forecasts. The first lagged growth is the last actual organic growth "
             "(one-off events removed)."]
    for k, t in enumerate(notes):
        ws.cell(rn + k, 1, t).font = Font(name=F, italic=True, color="555555")
    for col, w in zip("ABCDE", (14, 14, 52, 70, 26)):
        ws.column_dimensions[col].width = w

    # ---- one sheet per scenario
    p = model.params
    defl = getattr(model, "deflator", None) or ctx.cfg["features"].get("deflator")
    feats = [f for f in model.features if f in p]
    series = []
    for f in feats:
        code, tr, lag = parse_feature(f)
        series.append(code)
        if tr in ("rdlog", "ryoy"):
            series.append(defl)
    ecm = "ect_lag1" in p and getattr(model, "beta", None) is not None
    if ecm:
        series += [i["mnemonic"] for i in model.long_run] + [model.deflator]
    series = list(dict.fromkeys(s_ for s_ in series if s_))
    ny = n_ylags(p)
    look = max([lookback(parse_feature(f)[1]) + parse_feature(f)[2] for f in feats] + [4 if ecm else 1, 1, ny])
    origin = ctx.idx[ctx.end]
    L0 = float(ctx.level.iloc[-1])
    yv = np.asarray(ctx.y.values if hasattr(ctx.y, "values") else ctx.y, dtype=float)
    LOG = tg.is_log()
    G_COL = "g: growth (log)" if LOG else "g: change"
    B_COL = "Level"
    YS_COL = "y* = ln(real organic level)" if LOG else "y* = organic level"
    LN_COL = "ln L" if LOG else "L (organic)"
    QQ_COL = "Growth % q/q" if LOG else "Change q/q"
    ANN_COL = "Growth % annualised" if LOG else "Change annualised (x4)"

    def f_level(b, g, r):                      # level from the previous row and this row's change
        return f"={b}{r - 1}*EXP({g}{r})" if LOG else f"={b}{r - 1}+{g}{r}"

    def f_qq(g, r):
        return f"=EXP({g}{r})-1" if LOG else f"={g}{r}"

    def f_ann(g, r):
        return f"=EXP(4*{g}{r})-1" if LOG else f"=4*{g}{r}"
    PCT = "0.00%" if LOG else "0.0000"
    levels = getattr(model, "kind", "") == "LevelsARDL"
    if levels:
        series = [i["mnemonic"] for i in model.long_run] + [model.deflator]
        series = list(dict.fromkeys(series))
        rr_ = any(i.get("form") == "real_rate" for i in model.long_run)
        look = max(max(model.q.values()) + (4 if rr_ else 0), model.p, 1)

    for sname, mdf in ctx.macro.items():
        sh = wb.create_sheet(re.sub(r"[\[\]\*\?/\\:]", "_", str(sname))[:31])
        sh["A1"], sh["A1"].font = f"{model.name}: {sname}", TITLE
        sh["A2"] = "Change the yellow cells (future values of the drivers). Blue = data, black = formulas."
        sh["A2"].font = Font(name=F, italic=True)
        hr = 4
        cols = ["Quarter", "Actual / forecast"] + [f"{c_} (raw)" for c_ in series]
        xcols = [] if levels else [f"x{model.features.index(f) + 1}: {f}" for f in feats]
        cols += xcols
        if levels:
            cols += [f"z{k}: {i['mnemonic']} ({i.get('form', 'log')})" for k, i in enumerate(model.long_run, 1)]
            cols += [YS_COL]
        if ecm:
            cols += [f"z{k}: {i['mnemonic']} ({i.get('form', 'log')})" for k, i in enumerate(model.long_run, 1)]
            cols += [LN_COL, "ect"]                      # ln L = organic log level of the target
        cols += [G_COL, QQ_COL, ANN_COL, B_COL]
        for j, h in enumerate(cols, 1):
            c = sh.cell(hr, j, h)
            c.font, c.fill = HEAD, HFILL
            c.alignment = Alignment(wrap_text=True, vertical="center")
        cidx = {h: j for j, h in enumerate(cols, 1)}
        P = model.panel(ctx, mdf)
        pos = P.index.get_loc(origin)
        q = min(Q, len(P) - pos - 1)
        quarters = list(mdf.index[mdf.index.get_loc(origin) - look:mdf.index.get_loc(origin) + q + 1])
        first = hr + 1
        orow = first + look                                   # the origin row (last actual quarter)
        for k, qt in enumerate(quarters):
            rr = first + k
            fut = rr > orow
            sh.cell(rr, 1, str(qt)).font = BOLD if rr == orow else Font(name=F)
            sh.cell(rr, 2, "forecast" if fut else "actual").font = Font(name=F, color="808080")
            for s_ in series:
                v = float(mdf.loc[qt, s_]) if s_ in mdf.columns and pd.notna(mdf.loc[qt, s_]) else None
                c = sh.cell(rr, cidx[f"{s_} (raw)"], v)
                c.font, c.number_format = BLUE, "#,##0.0000"
                if fut:
                    c.fill = YELLOW
            if rr % 2 == 0:
                for j in range(1, len(cols) + 1):
                    if sh.cell(rr, j).fill != YELLOW:
                        sh.cell(rr, j).fill = GREY

        def X(code):
            col_ = get_column_letter(cidx[f"{code} (raw)"])
            return lambda row: f"{col_}{row}"
        gcol, bcol = get_column_letter(cidx[G_COL]), get_column_letter(cidx[B_COL])
        # actual growth for the lags the equation needs (origin row and, with several lags, the rows above)
        for i in range(0, max(ny, 1)):
            c = sh.cell(orow - i, cidx[G_COL], float(yv[ctx.end - i]))
            c.font, c.number_format = BLUE, "0.000000"
        sh.cell(orow, cidx[G_COL]).comment = Comment(
            "Actual organic target growth (one-off events removed).", "model")
        c = sh.cell(orow, cidx[B_COL], L0)
        c.font, c.number_format = BLUE, "#,##0.0"
        rows_feat = range(orow + 1, orow + q + 1)
        if levels:
            zc = {}
            Pd = X(model.deflator)
            for k, item in enumerate(model.long_run, 1):
                name_ = f"z{k}: {item['mnemonic']} ({item.get('form', 'log')})"
                zc[model.lr_names[k - 1]] = get_column_letter(cidx[name_])
                Xi = X(item["mnemonic"])
                for rr in range(orow - model.q[model.lr_names[k - 1]], orow + q + 1):
                    form = item.get("form", "log")
                    fml = {"log": f"LN({Xi(rr)})", "log_real": f"LN({Xi(rr)}/{Pd(rr)})", "level": f"{Xi(rr)}",
                           "real_rate": f"{Xi(rr)}-100*LN({Pd(rr)}/{Pd(rr - 4)})"}[form]
                    sh.cell(rr, cidx[name_], "=" + fml).number_format = "0.000000"
            ysc = get_column_letter(cidx[YS_COL])
            for i in range(0, model.p):
                c = sh.cell(orow - i, cidx[YS_COL], float(P["__ys"].iloc[pos - i]))
                c.font, c.number_format = BLUE, "0.000000"
            sh.cell(orow, cidx[YS_COL]).comment = Comment(
                (("ln of the organic balance divided by the price index" if getattr(model, "real", True) else
                  "ln of the organic balance") if LOG else "the organic target level") +
                " (one-off acquisitions and sales removed).", "model")
            for rr in rows_feat:
                parts = [coef_ref["const"]] + [f"{coef_ref[f'ys_lag{i}']}*{ysc}{rr - i}" for i in range(1, model.p + 1)]
                parts += [f"{coef_ref[f'{n}|L{k}']}*{zc[n]}{rr - k}" for n in model.lr_names
                          for k in range(0, model.q[n] + 1)]
                sh.cell(rr, cidx[YS_COL], "=" + "+".join(parts)).number_format = "0.000000"
                sh.cell(rr, cidx[G_COL],
                        f"={ysc}{rr}-{ysc}{rr - 1}" + (f"+LN({Pd(rr)}/{Pd(rr - 1)})"
                                                       if LOG and getattr(model, "real", True) else "")
                        ).number_format = "0.000000"
                sh.cell(rr, cidx[QQ_COL], f_qq(gcol, rr)).number_format = PCT
                sh.cell(rr, cidx[ANN_COL], f_ann(gcol, rr)).number_format = PCT
                sh.cell(rr, cidx[B_COL], f_level(bcol, gcol, rr)).number_format = "#,##0.0000"
        if ecm:
            zc = {}
            for k, item in enumerate(model.long_run, 1):
                name_ = f"z{k}: {item['mnemonic']} ({item.get('form', 'log')})"
                zc[k] = get_column_letter(cidx[name_])
                Xi, Pd = X(item["mnemonic"]), X(model.deflator)
                for rr in range(orow, orow + q + 1):
                    form = item.get("form", "log")
                    fml = {"log": f"LN({Xi(rr)})", "log_real": f"LN({Xi(rr)}/{Pd(rr)})", "level": f"{Xi(rr)}",
                           "real_rate": f"{Xi(rr)}-100*LN({Pd(rr)}/{Pd(rr - 4)})"}[form]
                    sh.cell(rr, cidx[name_], "=" + fml).number_format = "0.000000"
            lnc = get_column_letter(cidx[LN_COL])
            Pd = X(model.deflator)
            lnL0 = float(P["__logL"].iloc[pos])
            c = sh.cell(orow, cidx[LN_COL], lnL0)
            c.font, c.number_format = BLUE, "0.000000"
            c.comment = Comment("Organic log level of the target at the last actual quarter: the log of the first balance plus "
                                "all organic growth since (one-off acquisitions and sales removed), as the model uses it.",
                                "model")
            for rr in range(orow + 1, orow + q + 1):
                sh.cell(rr, cidx[LN_COL], f"={lnc}{rr - 1}+{gcol}{rr}").number_format = "0.000000"
            for rr in range(orow, orow + q + 1):
                lr = "+".join([coef_ref["long run: constant"]] +
                              [f"{coef_ref['long run: ' + it['mnemonic']]}*{zc[k]}{rr}"
                               for k, it in enumerate(model.long_run, 1)])
                sh.cell(rr, cidx["ect"], f"={lnc}{rr}" + (f"-LN({Pd(rr)})" if LOG else "") + f"-({lr})"
                        ).number_format = "0.000000"
        for f, xc in zip(feats, xcols):
            code, tr, lag = parse_feature(f)
            for rr in rows_feat:
                fml = _excel_feature(tr, X(code), X(defl) if defl in series else X(code), rr, lag)
                sh.cell(rr, cidx[xc], "=" + fml).number_format = "0.000000"
        for rr in ([] if levels else rows_feat):
            parts = [coef_ref["const"]]
            for i in range(1, ny + 1):
                if f"y_lag{i}" in p:
                    parts.append(f"{coef_ref[f'y_lag{i}']}*{gcol}{rr - i}")
            for f, xc in zip(feats, xcols):
                parts.append(f"{coef_ref[f]}*{get_column_letter(cidx[xc])}{rr}")
            if ecm:
                parts.append(f"{coef_ref['ect_lag1']}*{get_column_letter(cidx['ect'])}{rr - 1}")
            sh.cell(rr, cidx[G_COL], "=" + "+".join(parts)).number_format = "0.000000"
            sh.cell(rr, cidx[QQ_COL], f_qq(gcol, rr)).number_format = PCT
            sh.cell(rr, cidx[ANN_COL], f_ann(gcol, rr)).number_format = PCT
            sh.cell(rr, cidx[B_COL], f_level(bcol, gcol, rr)).number_format = "#,##0.0000"
        for rr in range(orow, orow + q + 1):
            for j in range(1, len(cols) + 1):
                sh.cell(rr, j).border = thin
        for j in range(1, len(cols) + 1):
            sh.column_dimensions[get_column_letter(j)].width = 12 if j > 2 else (11 if j == 1 else 15)
        sh.row_dimensions[hr].height = 45
        sh.freeze_panes = sh.cell(hr + 1, 3)
        sh.cell(orow + q + 2, 1, "Coefficients come from sheet Equation; the rows above the bold quarter are the "
                                 "history the drivers need.").font = Font(name=F, italic=True, color="555555")
    if best_other is not None and len(best_other):
        sh = wb.create_sheet("Best overall (no formula)")
        sh["A1"], sh["A1"].font = f"Most accurate overall: {best_other['model'].iloc[0]}", TITLE
        sh["A2"] = ("This model has no closed formula (machine learning, network, regime-switching or combination): its "
                    "forecast is shown as values. To update it, re-run the pipeline or use the saved model (fitted/).")
        sh["A2"].font = Font(name=F, italic=True)
        cols = list(best_other.columns)
        for j, h in enumerate(cols, 1):
            c = sh.cell(4, j, h)
            c.font, c.fill = HEAD, HFILL
        for i, (_, r) in enumerate(best_other.iterrows(), 5):
            for j, h in enumerate(cols, 1):
                v = r[h]
                c = sh.cell(i, j, float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                c.font = BLUE if isinstance(v, (float, np.floating)) else Font(name=F)
                if isinstance(v, (float, np.floating)):
                    c.number_format = "#,##0.00"
        for j in range(1, len(cols) + 1):
            sh.column_dimensions[get_column_letter(j)].width = 22
    for s_ in wb.worksheets:
        for row in s_.iter_rows():
            for c in row:
                if c.font and c.font.name != F and c.font.name != "Consolas":
                    c.font = Font(name=F, bold=c.font.bold, italic=c.font.italic, color=c.font.color, size=c.font.size)
    wb.save(path)


# ------------------------------------------------------------------------------------------- main entry
def write(cfg, ctx, models, ranking, basis, log=print, order=None, why_rec=None):
    """Pick the recommended formula model, write the equation, the next quarters and the Excel calculator.

    Args:
        cfg: configuration. ctx: Context. models: dict name -> fitted model. ranking: model table with the
        error columns (out-of-sample summary or backtest comparison). basis: "out of sample" or "backtest".
        log: message function.

    Returns:
        dict with the chosen models and file paths, or None when no model qualifies.
    """
    cols = sorted([c for c in ranking.columns if c.startswith("avg_1_") and c.endswith("_pct")],
                  key=lambda c: int(c.split("_")[2]))
    if not cols or not models:
        return None
    col = cols[-1]
    hq = int(col.split("_")[2])
    best, fbest = choose(ranking, models, col, order)
    if fbest is None:
        log("  formula: no model with an explicit formula in this run")
        return None
    Q = int((cfg.get("save") or {}).get("formula_quarters", 4) or 4)
    m = models[fbest["model"]]
    out = os.path.join(cfg["out_dir"], "formula")
    os.makedirs(out, exist_ok=True)
    T = terms(m, ctx.meta)
    W = worked(ctx, m, Q)
    start = cfg["sample"]["estimation_start"]
    if order and best is not None and best["model"] == fbest["model"]:
        why = f"the recommended model: {why_rec}" if why_rec else "the recommended model (model scorecard)"
    elif order:
        why = (f"the highest-scoring model with a formula on the model scorecard ({basis}, average balance error over "
               f"1–{hq} quarters {fbest[col]:.2f}%)")
    else:
        why = (f"most accurate model with a formula ({basis}, average balance error over 1–{hq} quarters "
               f"{fbest[col]:.2f}%)")
    other = None
    if best is not None and best["model"] != fbest["model"]:
        why += (f"; the recommended model is {best['model']} ({best[col]:.2f}%), which has no formula" if order else
                f"; the most accurate overall is {best['model']} ({best[col]:.2f}%), which has no formula")
        scen_f = os.path.join(cfg["out_dir"], "comparison", "6_scenario_forecasts.csv")
        if os.path.exists(scen_f):
            S = pd.read_csv(scen_f)
            S = S[S["model"] == best["model"]]
            other = S.groupby("scenario", sort=False).head(Q)[["model", "scenario", "date", "dlog_forecast",
                                                                "balance_forecast"]].copy()
            other["growth_pct_qq"] = tg.pct(other["dlog_forecast"])
            other["growth_pct_annualised"] = tg.annualised(other["dlog_forecast"])
            other = other.rename(columns={"dlog_forecast": "growth_log", "balance_forecast": "balance"})
    info = {"why": why, "window": f"{start} to {ctx.idx[ctx.end]}", "origin": str(ctx.idx[ctx.end]),
            "L0": float(ctx.level.iloc[-1])}
    eq = equation_text(m, T)
    md = [f"# Recommended model with a formula: {m.name}", "", why + ".", "", "```", eq, "```", "",
          "| Symbol | Coefficient | Definition | Meaning |", "|---|---|---|---|"]
    md += [f"| {r.symbol} | {r.coefficient:.6g} | {r.definition} | {r.meaning} |" for r in T.itertuples()]
    md += ["", ("g(t) = ln(L(t)/L(t-1)), quarterly growth (log); L = target level; " if tg.is_log() else
                "g(t) = L(t) - L(t-1), quarterly change; L = target level; ") + "X(t-k) = series X k quarters back.",
           f"Estimated on {info['window']}. Next {Q} quarters: 2_next_quarters.csv; live Excel: "
           "recommended_model_calculator.xlsx."]
    if other is not None:
        md += ["", f"## {'Recommended' if order else 'Most accurate overall'}: {best['model']} (no formula)", "",
               "Its forecast for the same quarters is in 3_best_overall_next_quarters.csv."]
    open(os.path.join(out, "formula.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    T.to_csv(os.path.join(out, "1_terms.csv"), index=False)
    W.to_csv(os.path.join(out, "2_next_quarters.csv"), index=False)
    if other is not None:
        other.to_csv(os.path.join(out, "3_best_overall_next_quarters.csv"), index=False)
    elif os.path.exists(os.path.join(out, "3_best_overall_next_quarters.csv")):
        os.remove(os.path.join(out, "3_best_overall_next_quarters.csv"))
    import json
    json.dump({"formula_model": m.name, "formula_family": m.family, "formula_error_pct": float(fbest[col]),
               "best_model": best["model"], "best_family": best["family"], "best_error_pct": float(best[col]),
               "best_has_formula": best["model"] == fbest["model"], "basis": basis, "horizon_q": hq, "quarters": Q,
               "equation": eq, "why": why, "origin": info["origin"], "window": info["window"]},
              open(os.path.join(out, "recommended.json"), "w", encoding="utf-8"), indent=1)
    xl = os.path.join(out, "recommended_model_calculator.xlsx")
    try:
        write_calculator(xl, ctx, m, T, Q, info, other)
    except Exception as ex:                                   # the CSV and text outputs are still there
        log(f"  formula: Excel calculator not written ({type(ex).__name__}: {ex})")
        xl = None
    log(f"  formula: {m.name} ({why}) -> {os.path.relpath(out)}")
    return {"formula_model": m.name, "best_model": best["model"], "folder": out, "excel": xl}
