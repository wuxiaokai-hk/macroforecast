"""Set up a new data set in one command: writes a ready-to-check config from your files.

    python tools/new_data_setup.py --name my_portfolio \\
        --bank data/my_bank.xlsx --macro-dir data/my_scenarios/ --metadata data/my_metadata.csv

What it does (and prints, so you can review it):
  1. reads the loan file: finds the date column, the balance column (or use --target) and monthly vs quarterly;
  2. reads every scenario file in --macro-dir (or --macro Name=path ...): scenario names from the file names,
     the one with "base" in its name is the baseline;
  3. reads the metadata and matches its columns to the six the model needs (mnemonic, description,
     geography, theme, transform, expected_sign). Missing columns are filled with suggestions in a draft copy
     (data/<name>_metadata_draft.csv): transform from the data, theme and sign from keywords in the description;
     without --metadata the draft is built from the macro column names;
  4. maps the literature concepts (house prices, income, mortgage rate, ...) to your series by keywords in the
     descriptions (kept where your workbook already names a series), into config/<name>_literature.xlsx;
  5. sets the dates from the data: hist_end, estimation_start, backtest window, out-of-sample origins, COVID
     dummies if the sample covers 2020-21;
  6. writes config/<name>.yaml (a copy of the example config, comments kept) and runs the pre-flight check.

Review the printed summary, the metadata draft and the workbook, then run:  python run_pipeline.py --config config/<name>.yaml
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.utils.io import read_table, to_quarters  # noqa: E402

FIELDS = {"mnemonic": ["mnemonic", "code", "seriescode", "series", "variable", "ticker", "id", "name"],
          "description": ["description", "desc", "label", "title", "longname", "seriesname", "name"],
          "geography": ["geography", "geo", "region", "area", "country", "province", "location"],
          "theme": ["theme", "category", "group", "concept", "topic", "sector"],
          "transform": ["transform", "transformation", "tr"],
          "expected_sign": ["expectedsign", "sign", "prior", "direction"]}

# literature concepts: keywords searched in description + mnemonic (all lower case); 'avoid' lowers the score
CONCEPTS = {
    "house_prices": {"kw": ["house price", "home price", "housing price", "hpi", "residential property price"],
                     "avoid": ["rent", "construction"]},
    "disposable_income": {"kw": ["disposable income", "personal disposable"], "avoid": ["per capita"]},
    "real_gdp": {"kw": ["real gdp", "gdp, real", "real gross domestic", "gdp (real", "gdpr"], "avoid": ["nominal", "per capita"]},
    "mortgage_rate": {"kw": ["mortgage rate", "5-year mortgage", "5 year mortgage", "conventional mortgage", "rmort"],
                      "avoid": ["spread", "real", "arrears"], "prefer": ["5-year", "5 year", "5y", "five-year"]},
    "policy_rate": {"kw": ["policy rate", "overnight", "bank rate", "target rate", "fed funds", "rboc"], "avoid": ["spread"]},
    "unemployment": {"kw": ["unemployment rate", "jobless rate", "lbr."], "avoid": ["youth", "duration"]},
    "inflation": {"kw": ["consumer price index", "consumer price", "cpi"], "avoid": ["core", "excluding", "ex-"]},
}
YIELD5 = ["5-year government", "5 year government", "5-year bond", "5-year goc", "gov5y", "rgov5y"]
YIELD10 = ["10-year government", "10 year government", "10-year bond", "10-year goc", "gov10y", "rgov10y"]

THEME_KW = [("Interest rates", ["rate", "yield", "bond", "mortgage rate"]), ("House prices", ["house price", "home price", "hpi"]),
            ("Housing activity", ["housing start", "home sales", "resale", "permit", "construction"]),
            ("Labour", ["employment", "unemployment", "labour", "labor", "jobs", "participation"]),
            ("Income", ["income", "wage", "earnings", "compensation"]), ("Prices", ["price index", "cpi", "inflation", "deflator"]),
            ("Output", ["gdp", "output", "production"]), ("Consumption", ["retail", "consumption", "spending", "sales"]),
            ("Population", ["population", "migration", "household formation"]),
            ("Money & credit", ["credit", "loan", "money", "m2", "deposit", "mortgage"]),
            ("Financial markets", ["stock", "equity", "tsx", "s&p", "exchange rate", "dollar"])]
SIGN_KW = [(+1, ["house price", "home price", "hpi", "disposable income", "gdp", "employment", "income", "wage", "housing start",
                 "home sales", "resale", "retail", "population", "permit", "consumer confidence", "stock", "tsx"]),
           (-1, ["unemployment", "mortgage rate", "interest rate", "bond yield", "prime rate", "policy rate", "overnight",
                 "bankrupt", "insolvenc", "arrears", "spread"])]
COVID = ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2"]


def norm(s):
    """Lower-case a header and drop everything but letters and digits.

    Args:
        s: header.

    Returns:
        str.
    """
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_date_column(df):
    """The column that reads as quarters (columns named like date/period/quarter are tried first).

    Args:
        df: table.

    Returns:
        column name or None.
    """
    cols = sorted(df.columns, key=lambda c: 0 if re.search(r"date|period|quarter|time|month", str(c), re.I) else 1)
    for c in cols:
        try:
            q = to_quarters(df[c].dropna())
            if len(q) > 4 and q.min().year > 1900:
                return c
        except Exception:
            continue
    return None


def find_panel_column(df, dc, num):
    """The column naming the bank (or region, portfolio) when one file holds several series: a text column such
    that every date appears once per value of it. None when the file is a single series."""
    q = to_quarters(df[dc])
    if not pd.Series(q).duplicated().any():
        return None
    for c in df.columns:
        if c == dc or c in num:
            continue
        v = df[c].astype(str).str.strip()
        if 2 <= v.nunique() <= max(2, len(df) // 4) and not df.assign(_v=v).duplicated([dc, "_v"]).any():
            return c
    return None


def bank_summary(df, dc, target, panel):
    """One row per bank: first and last quarter, number of quarters, last value."""
    q = to_quarters(df[dc])
    d = pd.DataFrame({"bank": df[panel].astype(str).str.strip(), "q": q, "v": pd.to_numeric(df[target], errors="coerce")})
    d = d.dropna().sort_values("q").groupby(["bank", "q"], as_index=False).last()
    return d.groupby("bank").agg(first=("q", "first"), last=("q", "last"), quarters=("q", "size"),
                                 last_value=("v", "last")).reset_index()


def read_bank(path, sheet, target, log, panel=None):
    """Read the target file and detect its date column, balance column, frequency and (for several banks in one
    file) the bank column.

    Args:
        path: file. sheet: Excel sheet. target: balance column (None = detect). log: message function.
        panel: the bank column; None = detect; "" or False = the file is one series.

    Returns:
        dict with date_column, target_column, frequency, panel_column, banks (summary or None) and the quarterly
        series (the total of all banks for a panel).
    """
    df = read_table(path, sheet)
    dc = find_date_column(df)
    if dc is None:
        raise SystemExit(f"{path}: no column reads as dates or quarters (columns: {list(df.columns)})")
    num = [c for c in df.columns if c != dc and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    if target is None:
        loan = [c for c in num if re.search(r"loan|lend|balance|outstanding", str(c), re.I)]
        target = (loan or num or [None])[0]
        if len(num) > 1:
            log(f"  several numeric columns {num}: using '{target}' (choose another with --target)")
    if target not in df.columns:
        raise SystemExit(f"{path}: column '{target}' not found (columns: {list(df.columns)})")
    q = to_quarters(df[dc])
    pc = (panel if panel in df.columns else None) if panel else (None if panel in ("", False) else
                                                                 find_panel_column(df, dc, num))
    if pc:
        bk = df[pc].astype(str).str.strip()
        freq = "M" if pd.DataFrame({"b": bk, "q": q}).duplicated().any() else "Q"
        d = pd.DataFrame({"b": bk, "q": q, "v": pd.to_numeric(df[target], errors="coerce")}).dropna()
        d = d.groupby(["b", "q"], as_index=False).last()
        s = d.groupby("q")["v"].sum().sort_index()
        banks = bank_summary(df, dc, target, pc)
        log(f"  panel: '{pc}' names {len(banks)} banks ({', '.join(banks['bank'][:6])}"
            f"{', ...' if len(banks) > 6 else ''}); the total is used for dates and previews")
        return {"date_column": dc, "target_column": target, "frequency": freq, "series": s, "panel_column": pc,
                "banks": banks}
    freq = "M" if pd.Series(q).duplicated().any() else "Q"
    s = pd.to_numeric(df[target], errors="coerce")
    s.index = q
    s = (s.groupby(level=0).last() if freq == "M" else s).dropna().sort_index()
    return {"date_column": dc, "target_column": target, "frequency": freq, "series": s, "panel_column": None,
            "banks": None}


def scenario_name(path):
    """Scenario label from a file name (baseline, upside, downside, severe ... recognised).

    Args:
        path: file.

    Returns:
        str.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    low = stem.lower()
    for key, label in (("base", "Baseline"), ("upside", "Upside"), ("downside", "Downside"), ("severe", "Severe"),
                       ("adverse", "Adverse"), ("stress", "Stress")):
        if key in low:
            m = re.search(r"s(\d)\b|s(\d)_|_s(\d)", low)
            return (f"S{next(g for g in m.groups() if g)} " if m and label != "Baseline" else "") + label
    return stem


def read_macro(files, sheet, log):
    """Read the scenario files.

    Args:
        files: dict name -> path. sheet: Excel sheet. log: message function.

    Returns:
        (dict name -> DataFrame, date column, base scenario name).
    """
    out, dc = {}, None
    for name, f in files.items():
        df = read_table(f, sheet)
        c = find_date_column(df)
        if c is None:
            raise SystemExit(f"{f}: no date column found")
        if dc and c != dc:
            log(f"  warning: {f} uses date column '{c}', the first file uses '{dc}'")
        dc = dc or c
        x = df.drop(columns=c).apply(pd.to_numeric, errors="coerce")
        x.index = to_quarters(df[c])
        out[name] = x.sort_index()
    base = next((n for n in files if "base" in n.lower()), next(iter(files)))
    return out, dc, base


def map_columns(df):
    """Match metadata headers to the six fields the model needs.

    Args:
        df: metadata table.

    Returns:
        dict field -> your column (missing fields absent).
    """
    used, out = set(), {}
    heads = {norm(c): c for c in df.columns}
    for field, cands in FIELDS.items():
        for cnd in cands:
            hit = next((orig for n, orig in heads.items() if (n == cnd or n.startswith(cnd)) and orig not in used), None)
            if hit is not None:
                out[field] = hit
                used.add(hit)
                break
    return out


def guess_transform(x, text):
    """dlog for positive levels and indexes, diff for rates, percentages and series that go negative.

    Args:
        x: the series. text: description and mnemonic.

    Returns:
        "dlog" or "diff".
    """
    t = text.lower()
    if re.search(r"rate|yield|spread|percent|%|ratio|share|balance of|index of change|change", t) and "price" not in t:
        return "diff"
    v = pd.to_numeric(x, errors="coerce").dropna()
    return "dlog" if len(v) and (v > 0).all() else "diff"


def guess_theme(text):
    """Theme from keywords in the description.

    Args:
        text: description and mnemonic.

    Returns:
        str.
    """
    t = text.lower()
    for theme, kws in THEME_KW:
        if any(k in t for k in kws):
            return theme
    return "Other"


def guess_sign(text):
    """Expected effect on target growth from keywords (0 = no prior); a suggestion to review.

    Args:
        text: description and mnemonic.

    Returns:
        int.
    """
    t = text.lower()
    for sign, kws in SIGN_KW[::-1]:                   # negative keywords first ("unemployment" before "employment")
        if any(k in t for k in kws):
            return sign
    return 0


def build_metadata(meta_path, sheet, base, name, log):
    """Read the metadata (or start from the macro columns) and fill missing fields with suggestions.

    Args:
        meta_path: metadata file or None. sheet: Excel sheet. base: base-scenario DataFrame. name: set-up name.
        log: message function.

    Returns:
        (path of the metadata file to use, column mapping, metadata in model names).
    """
    if meta_path:
        md = read_table(meta_path, sheet)
        cmap = map_columns(md)
        if "mnemonic" not in cmap:
            raise SystemExit(f"{meta_path}: no column looks like the variable code (columns: {list(md.columns)})")
        log(f"  metadata columns: " + ", ".join(f"{k} <- '{v}'" for k, v in cmap.items()))
    else:
        md = pd.DataFrame({"mnemonic": list(base.columns)})
        cmap = {"mnemonic": "mnemonic"}
        log("  no metadata file: drafting one from the macro column names")
    meta = md.rename(columns={v: k for k, v in cmap.items()})
    meta["mnemonic"] = meta["mnemonic"].astype(str).str.strip()
    filled = []
    if "description" not in meta:
        meta["description"] = meta["mnemonic"]
        filled.append("description")
    text = (meta["description"].astype(str) + " " + meta["mnemonic"]).str.lower()
    if "geography" not in meta:
        meta["geography"] = "All"
        filled.append("geography")
    if "theme" not in meta:
        meta["theme"] = [guess_theme(t) for t in text]
        filled.append("theme")
    if "transform" not in meta:
        meta["transform"] = [guess_transform(base[m] if m in base else pd.Series(dtype=float), t)
                             for m, t in zip(meta["mnemonic"], text)]
        filled.append("transform")
    if "expected_sign" not in meta:
        meta["expected_sign"] = [guess_sign(t) for t in text]
        filled.append("expected_sign")
    if not filled:
        return meta_path, cmap, meta
    path = os.path.join(HERE, "data", f"{name}_metadata_draft.csv")
    cols = ["mnemonic", "description", "geography", "theme", "transform", "expected_sign"]
    out = meta[cols].copy()
    out["filled_by_setup"] = ", ".join(filled)
    out.to_csv(path, index=False)
    log(f"  filled {filled} with suggestions -> {os.path.relpath(path, HERE)} (review it, especially expected_sign)")
    return path, {c: c for c in cols}, meta[cols]


def match_concepts(meta, names, primary, log):
    """Suggest the series for each literature concept by keyword score, preferring the primary geography.

    Args:
        meta: metadata (model names). names: macro columns available. primary: preferred geography. log: messages.

    Returns:
        dict concept -> (mnemonic, note).
    """
    meta = meta[meta["mnemonic"].isin(names)]
    text = (meta["description"].astype(str) + " | " + meta["mnemonic"]).str.lower()
    out = {}
    for cid, rule in CONCEPTS.items():
        best = (0.0, None, "")
        for (_, r), t in zip(meta.iterrows(), text):
            sc = sum(2.0 for k in rule["kw"] if k in t)
            if not sc:
                continue
            sc -= sum(1.5 for a in rule["avoid"] if a in t)
            sc += sum(1.5 for p in rule.get("prefer", []) if p in t)
            if primary and str(r["geography"]) == primary:
                sc += 1.0
            if re.search(r"change|growth|y/y|q/q|%ch|percent change", t):
                sc -= 1.0
            if sc > best[0]:
                best = (sc, r["mnemonic"], r["description"])
        out[cid] = (best[1], f"matched '{best[2]}'" if best[1] else "no match: fill in series_name")
    for cid, (m, note) in out.items():
        log(f"  {cid:<18s} -> {m or '(none)':<16s} {note}")
    return out


def find_by(meta, names, kws, primary):
    """First series whose description or mnemonic contains one of kws (primary geography preferred).

    Args:
        meta: metadata. names: available columns. kws: keywords. primary: preferred geography.

    Returns:
        mnemonic or None.
    """
    m = meta[meta["mnemonic"].isin(names)]
    t = (m["description"].astype(str) + " " + m["mnemonic"]).str.lower()
    hit = m[[any(k in x for k in kws) for x in t]]
    if hit.empty:
        return None
    pri = hit[hit["geography"].astype(str) == str(primary)]
    return (pri if not pri.empty else hit)["mnemonic"].iloc[0]


# ------------------------------------------------------------------------------------------- YAML editing
def set_key(lines, path, value):
    """Replace the value of a key in YAML text, keeping its comment and the rest of the file.

    Args:
        lines: list of lines (changed in place). path: dotted key, e.g. "data.bank.file". value: YAML text.

    Returns:
        True if the key was found.
    """
    keys, depth, start = path.split("."), 0, 0
    indent = -1
    for k in keys:
        found = False
        for i in range(start, len(lines)):
            ln = lines[i]
            ind = len(ln) - len(ln.lstrip(" "))
            if ln.strip() and not ln.lstrip().startswith("#") and ind <= indent:
                return False                                   # left the parent block
            if ln.lstrip().startswith(f"{k}:") and ind > indent and (indent >= 0 or ind == 0):
                indent, start, found = ind, i + 1, True
                break
        if not found:
            return False
    i = start - 1
    ln = lines[i]
    m = re.match(r"^(\s*[^:#]+:)(\s*)([^#]*?)(\s*)(#.*)?$", ln)
    pre, comment = m.group(1), m.group(5) or ""
    new = f"{pre} {value}"
    lines[i] = (new.ljust(max(len(new) + 1, 36)) + comment).rstrip() if comment else new
    return True


def replace_block(lines, path, block):
    """Replace a key's nested block (the lines indented below it) with new lines.

    Args:
        lines: list of lines (changed in place). path: dotted key. block: list of new lines (already indented).
    """
    keys, indent, start = path.split("."), -1, 0
    for k in keys:
        for i in range(start, len(lines)):
            ln = lines[i]
            ind = len(ln) - len(ln.lstrip(" "))
            if ln.lstrip().startswith(f"{k}:") and ind > indent and (indent >= 0 or ind == 0):
                indent, start = ind, i + 1
                break
    end = start
    while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip(" ")) > indent
                                or lines[end].lstrip().startswith("#") and len(lines[end]) - len(lines[end].lstrip(" ")) > indent):
        end += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    lines[start:end] = block


def single_bank(P):
    """The one bank chosen from a panel file, or None (no bank column, or several banks chosen).

    One credit union is not a panel: it is run as a single series (the full project), from its own rows."""
    if not (P.get("bank") or {}).get("panel_column"):
        return None
    chosen = P.get("panel_banks") or []
    return chosen[0] if len(chosen) == 1 else None


def extract_single_bank(P, bank, name):
    """Write the chosen bank's rows (and its own events) to their own files for a single-series run.

    Returns:
        (target file, events file or None).
    """
    import re as _re
    bk = P["bank"]
    df = read_table(bk["file"], bk["sheet"])
    pc = bk["panel_column"]
    rows = df[df[pc].astype(str).str.strip() == str(bank).strip()]
    folder = os.path.join(os.path.dirname(os.path.abspath(bk["file"])), "single_bank")
    os.makedirs(folder, exist_ok=True)
    tag = _re.sub(r"[^A-Za-z0-9]+", "_", f"{name}_{bank}").strip("_")
    f = os.path.join(folder, f"{tag}.csv")
    rows[[bk["date_column"], bk["target_column"]]].to_csv(f, index=False)
    ef = None
    if P.get("events"):
        from src.panel.data import bank_events
        c = {"data": {"events": {"file": P["events"], "sheet": None, "date_column": "date", "series_column": "series",
                                 "amount_column": "amount", "bank_column": _events_bank_column(P)},
                      "bank": {"target_column": bk["target_column"], "panel_column": pc}}}
        try:
            ev = bank_events(c, str(bank).strip())
        except Exception:
            ev = {}
        if ev:
            ef = os.path.join(folder, f"{tag}_events.csv")
            pd.DataFrame({"date": [str(q) for q in ev], "series": bk["target_column"],
                          "amount": list(ev.values())}).to_csv(ef, index=False)
    return f, ef


def _events_bank_column(P):
    """The events file's bank column for a panel (the same name as the target file's, if it has it)."""
    pc = (P.get("bank") or {}).get("panel_column")
    if not pc or not P.get("events"):
        return None
    try:
        return pc if pc in read_table(P["events"]).columns else None
    except Exception:
        return None


def yq(v):
    """A value as YAML text (strings quoted when needed).

    Args:
        v: value.

    Returns:
        str.
    """
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(yq(x) for x in v) + "]"
    s = str(v)
    yaml_words = ("yes", "no", "y", "n", "true", "false", "on", "off", "null", "~")      # YAML would read these as booleans
    return f'"{s}"' if re.search(r"[:#\[\]{},&*!|>'\"%@`]|^\s|\s$", s) or s.lower() in yaml_words else s


def rel(p):
    """Path relative to the project folder when inside it.

    Args:
        p: path.

    Returns:
        str.
    """
    a = os.path.abspath(p)
    return os.path.relpath(a, HERE) if a.startswith(HERE) else a


# ------------------------------------------------------------------------------------------- propose / write
def propose(bank, macro_files, metadata=None, events=None, sheet=None, target=None, primary=None, name="new", log=print,
            literature=None, panel=None):
    """Read the files and propose every data-dependent setting (used by this command and by the app).

    Args:
        bank: target file. macro_files: dict scenario name -> file. metadata: metadata file or None.
        events: events file or None. sheet: Excel sheet for all files. target: balance column (None = detect).
        primary: main geography (None = detect). name: set-up name (for the draft metadata file). log: messages.
        literature: literature workbook (.xlsx with sheets models and concepts); None = the example's models and table.

    Returns:
        dict with the proposal (files, columns, scenarios, metadata mapping, concepts, dates, ...).
    """
    log("1. Loan balance")
    b = read_bank(bank, sheet, target, log, panel)
    s = b["series"]
    raw = read_table(bank, sheet)
    log(f"  '{b['target_column']}' by '{b['date_column']}', {'monthly' if b['frequency'] == 'M' else 'quarterly'}, "
        f"{s.index[0]} to {s.index[-1]} ({len(s)} quarters)")
    log("2. Scenarios")
    macro, mdc, base_name = read_macro(macro_files, sheet, log)
    base = macro[base_name]
    for n, f in macro_files.items():
        log(f"  {n:<14s} {rel(f)}  ({macro[n].shape[1]} variables, {macro[n].index[0]} to {macro[n].index[-1]})"
            + ("  <- base" if n == base_name else ""))
    log("3. Metadata")
    meta_path, cmap, meta = build_metadata(metadata, sheet, base, name, log)
    names = set(base.columns)
    geo = meta["geography"].astype(str)
    log("4. Literature concepts")
    if not primary:
        cand = match_concepts(meta, names, None, lambda *_: None)
        g = [geo[meta["mnemonic"] == m].iloc[0] for m, _ in cand.values() if m and (meta["mnemonic"] == m).any()]
        primary = pd.Series(g).mode().iloc[0] if g else (geo.mode().iloc[0] if len(geo) else None)
    concepts = match_concepts(meta, names, primary, log)
    allowed = list(dict.fromkeys(str(geo[meta["mnemonic"] == m].iloc[0]) for m, _ in concepts.values()
                                 if m and (meta["mnemonic"] == m).any()))
    log(f"  geographies kept: {allowed or 'all'}; primary: {primary}")
    mort, pol, cpi = concepts["mortgage_rate"][0], concepts["policy_rate"][0], concepts["inflation"][0]
    y5, y10 = find_by(meta, names, YIELD5, primary), find_by(meta, names, YIELD10, primary)
    constructed = []
    if mort and y5:
        constructed.append(("XMORTSPRD", f"`{mort}` - `{y5}`", "Mortgage spread, mortgage rate minus 5Y government yield (pp)", "Credit spreads", -1))
    if mort and cpi:
        constructed.append(("XRMORTREAL", f"`{mort}` - 100 * (log(`{cpi}`) - log(`{cpi}`.shift(4)))",
                            "Real mortgage rate (mortgage rate minus y/y CPI inflation)", "Interest rates", -1))
    if y10 and pol:
        constructed.append(("XTERM", f"`{y10}` - `{pol}`", "Term spread, 10Y government yield minus policy rate (pp)", "Yield curve", 0))
    log("  constructed: " + (", ".join(f"{c[0]} = {c[1]}" for c in constructed) or "none (components not found)"))
    extra = {"real_mortgage_rate": "XRMORTREAL" if any(c[0] == "XRMORTREAL" for c in constructed) else None,
             "term_spread": "XTERM" if any(c[0] == "XTERM" for c in constructed) else None}
    from src.models.literature import read_workbook
    wb = read_workbook(literature or os.path.join(HERE, "config", "literature_workbook.xlsx"), full=True)   # yours or the example's
    lit_models, lit, lit_settings, lit_refs = wb["models"], wb["concepts"], wb["settings"], wb["references"]
    log(f"  literature workbook: {len(lit_models)} model(s), {len(lit)} concept(s)")
    given = lit["series_name"].astype(object).where(lit["series_name"].notna() & (lit["series_name"].astype(str).str.strip() != ""))
    given = given.where(given.astype(str).isin(names | {c[0] for c in constructed}))     # keep names that exist in YOUR data
    lit["series_name"] = [g if isinstance(g, str) else (concepts.get(c, (None,))[0] or extra.get(c))
                          for g, c in zip(given, lit["concept_id"])]
    lit["setup_note"] = [("from your workbook" if isinstance(g, str) else concepts[c][1] if c in concepts else
                          ("constructed by the set-up" if extra.get(c) else "fill in if you have it"))
                         for g, c in zip(given, lit["concept_id"])]
    log("5. Dates")
    first = max(s.index[0], base.index[0])
    he = s.index[-1]
    es = first + 9
    be = he - 4
    bs = max(es + 20, be - 27)
    fo = max(es + 20, min(es + 40, he - 16))
    dummies = [q for q in COVID if es <= pd.Period(q, freq="Q") <= he]
    log(f"  hist_end {he}; estimation_start {es}; backtest {bs} to {be}; out-of-sample from {fo}; dummies {dummies or 'none'}")
    if he.ordinal - first.ordinal < 60:
        log("  fewer than 15 years of data: expect few backtest origins; consider sample.horizon 8")
    themes = sorted(set(meta["theme"].astype(str)))
    return {"bank": {"file": bank, "sheet": sheet, "date_column": b["date_column"], "target_column": b["target_column"],
                     "frequency": b["frequency"], "columns": [str(c) for c in raw.columns],
                     "panel_column": b["panel_column"]}, "banks": b["banks"],
            "series": s, "events": events,
            "macro": {"files": dict(macro_files), "date_column": mdc, "base": base_name, "sheet": sheet}, "base": base,
            "metadata": {"file": meta_path, "columns": cmap, "sheet": sheet if meta_path == metadata else None},
            "meta": meta, "primary": primary, "allowed": allowed, "geographies": sorted(set(geo)), "themes": themes,
            "concepts": concepts, "constructed": constructed, "lit": lit, "lit_models": lit_models, "lit_settings": lit_settings, "lit_refs": lit_refs, "deflator": cpi,
            "dates": {"hist_end": str(he), "estimation_start": str(es), "backtest_start": str(bs), "backtest_end": str(be),
                      "oos_first": str(fo), "dummies": dummies, "first": str(first)},
            "real_themes": [t for t in ["Income", "Consumption", "Money & credit"] if t in themes],
            "excluded_themes": [t for t in ["Other", "Output (by industry)", "Labour (by industry)"] if t in themes]}


def write_config(P, name, overrides=None, fast=False, log=print):
    """Write config/<name>.yaml (a copy of the example config, comments kept) and its literature table.

    Args:
        P: proposal from propose(), possibly edited. name: set-up name. overrides: dict dotted key -> YAML text for any
        other setting (e.g. {"final_model.max_vars": "3"}). fast: switch off the deep-learning models. log: messages.

    Returns:
        (config path, literature table path).
    """
    cfg_path = os.path.join(HERE, "config", f"{name}.yaml")
    lit_path = os.path.join(HERE, "config", f"{name}_literature.xlsx")
    from src.models.literature import write_workbook
    write_workbook(lit_path, P["lit_models"], P["lit"].drop(columns=["setup_note"], errors="ignore"),
                   P.get("lit_settings"), P.get("lit_refs"))
    with open(os.path.join(HERE, "config", "example_config.yaml"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    d, bk, mc, md = P["dates"], P["bank"], P["macro"], P["metadata"]
    sets = {
        "run_name": name,
        # the example's target description (a loan balance in $ millions) does not carry over to new data
        "target.label": yq(str(bk["target_column"]).replace("_", " ")), "target.units": "null",
        "target.display_scale": "null", "target.display_units": "null",
        "data.bank.file": yq(rel(bk["file"])), "data.bank.sheet": yq(bk["sheet"]), "data.bank.date_column": yq(bk["date_column"]),
        "data.bank.target_column": yq(bk["target_column"]), "data.bank.frequency": bk["frequency"],
        "data.bank.panel_column": yq(bk.get("panel_column")),
        "data.events.file": yq(rel(P["events"]) if P.get("events") else None),
        "data.events.bank_column": yq(_events_bank_column(P)),
        "data.macro.date_column": yq(mc["date_column"]), "data.macro.sheet": yq(mc["sheet"]), "data.macro.base_scenario": yq(mc["base"]),
        "data.metadata.file": yq(rel(md["file"])), "data.metadata.sheet": yq(md["sheet"]),
        "sample.hist_end": d["hist_end"], "sample.estimation_start": d["estimation_start"], "sample.backtest_start": d["backtest_start"],
        "sample.backtest_end": d["backtest_end"] or "null", "sample.dummy_quarters": yq(d["dummies"]),
        "features.deflator": yq(P["deflator"]), "features.allowed_geographies": yq(P["allowed"]),
        "features.real_themes": yq(P["real_themes"]), "features.excluded_themes": yq(P["excluded_themes"]),
        "literature.workbook": yq(rel(lit_path)),
        "screen_first.primary_geography": yq(P["primary"]), "en_first.primary_geography": yq(P["primary"]),
        "oos.first_origin": d["oos_first"],
    }
    one = single_bank(P)
    if one is not None:                                   # one credit union: a single-series run, not a panel
        bf, ef = extract_single_bank(P, one, name)
        sets.update({"data.bank.file": yq(rel(bf)), "data.bank.sheet": "null", "data.bank.panel_column": "null",
                     "data.events.file": yq(rel(ef) if ef else None), "data.events.bank_column": "null"})
        overrides = {k: v for k, v in (overrides or {}).items() if not k.startswith("panel.")}
    sets.update(overrides or {})
    for k, v in sets.items():
        if not set_key(lines, k, v):
            log(f"  (could not set {k} in the config: set it by hand)")
    for k, v in md["columns"].items():
        set_key(lines, f"data.metadata.columns.{k}", yq(v))
    replace_block(lines, "data.macro.scenarios", [f"      {yq(n)}: {yq(rel(f))}" for n, f in mc["files"].items()])
    block = []
    for nm, ex, desc, theme, sign in P["constructed"]:
        block += [f"    - name: {nm}", f"      expression: \"{ex}\"", f"      description: {yq(desc)}",
                  f"      geography: {yq(P['primary'] or 'All')}", f"      theme: {yq(theme)}", "      transform: diff",
                  f"      expected_sign: {sign}"]
    replace_block(lines, "data.constructed", block)
    if not block:
        for i, ln in enumerate(lines):
            if ln.startswith("  constructed:"):
                lines[i] = "  constructed: []"
    if fast and not set_key(lines, "models.deep.enabled", "false"):
        log("  (could not switch off the deep models: set models.deep.enabled: false by hand)")
    lines[0:0] = [f"# Written by the set-up from {rel(bk['file'])} and {len(mc['files'])} scenario file(s). Review the",
                  "# literature table and metadata it points to, then: python run_pipeline.py --config " + rel(cfg_path), ""]
    if one is not None:
        lines[0:0] = [f"# One credit union ('{one}') chosen from the panel file: it runs as a single series (the full",
                      "# project); choose two or more in the app to run the panel instead."]
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return cfg_path, lit_path


def main():
    """Command line entry: detect, suggest, write the config and run the pre-flight check."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="name of the set-up: config/<name>.yaml, outputs/<name>/")
    ap.add_argument("--bank", required=True, help="target file (.csv or .xlsx)")
    ap.add_argument("--target", help="balance column (default: detected)")
    ap.add_argument("--events", help="inorganic events file (optional)")
    ap.add_argument("--macro-dir", help="folder with one file per scenario")
    ap.add_argument("--macro", nargs="*", default=[], help="or scenario files as Name=path (repeatable)")
    ap.add_argument("--metadata", help="variable metadata file (optional: drafted if missing)")
    ap.add_argument("--literature", help="literature workbook (sheets models, concepts); default: the example's")
    ap.add_argument("--sheet", default=None, help="Excel sheet name used for all files")
    ap.add_argument("--primary-geography", help="your main region (default: detected)")
    ap.add_argument("--panel", default=None, help="column naming the bank when the file holds several (default: detected)")
    ap.add_argument("--fast", action="store_true", help="switch off the slow deep-learning models for a first run")
    ap.add_argument("--force", action="store_true", help="overwrite an existing config/<name>.yaml")
    a = ap.parse_args()
    cfg_path = os.path.join(HERE, "config", f"{a.name}.yaml")
    if os.path.exists(cfg_path) and not a.force:
        raise SystemExit(f"{rel(cfg_path)} exists: use --force to overwrite, or another --name")
    files = {}
    if a.macro_dir:
        for f in sorted(glob.glob(os.path.join(a.macro_dir, "*.csv")) + glob.glob(os.path.join(a.macro_dir, "*.xls*"))):
            files[scenario_name(f)] = f
    for item in a.macro:
        n, p = item.split("=", 1)
        files[n] = p
    if not files:
        raise SystemExit("no scenario files: use --macro-dir or --macro Name=path")
    P = propose(a.bank, files, a.metadata, a.events, a.sheet, a.target, a.primary_geography, a.name, literature=a.literature,
                panel=a.panel)
    cfg_path, lit_path = write_config(P, a.name, fast=a.fast)
    print(f"\n6. Wrote {rel(cfg_path)} and {rel(lit_path)}")
    core = P["lit"][P["lit"]["core"].astype(str).str.lower() == "yes"]
    miss = list(core[core["series_name"].isna()]["concept_id"])
    if miss:
        print(f"  core concepts without a match: {miss}: put your series names in {rel(lit_path)} (series_name)")
    from src.utils.config import load_config
    from src.utils.preflight import check
    rep = check(load_config(cfg_path))
    print("\nPre-flight check:" + rep.text(show_ok=False))
    if not rep.errors:
        print(f"\nNext: python run_pipeline.py --config {rel(cfg_path)}")


if __name__ == "__main__":
    main()
