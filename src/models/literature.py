"""The literature workbook (config: literature.workbook, e.g. config/literature_workbook.xlsx) and its concept table.

One row per economic concept: what the literature says drives household/mortgage credit, the expected sign of
the effect, the evidence, and the name of the matching series in YOUR macro file (column `series_name`).

  - Rename only here: models in the config refer to concepts (house_prices|dlog|L1), not to series names,
    so a different vendor name or vintage means editing one cell.
  - Filter here: set include = no for a concept you do not believe; it is dropped from every literature and
    hybrid model, from the error-correction long run and from the Chronos-2 covariates.
  - With literature.apply_signs_to_metadata: true, the table also sets the expected sign of the mapped series
    for the data-driven approaches, and include = no gives sign 0 (so, with features.require_sign_prior,
    the series cannot be selected by the data either).

Columns used by the code: concept_id, concept, expected_sign, core (yes/no), long_run_form (log | log_real |
level | real_rate, or empty), include (yes/no), series_name. The other columns are documentation.
"""
import os

import pandas as pd

YES = {"yes", "y", "true", "1", "x"}


def _yes(v):
    """True for yes/true/1/y (any case), as used in the include and core columns.

    Args:
        v: cell value.

    Returns:
        bool.
    """
    return str(v).strip().lower() in YES


MODEL_COLS = ["model", "source", "variable", "transform", "lag", "expected_sign", "core"]
REQUIRED_MODEL_COLS = ["model", "source", "variable", "transform", "lag", "core"]

# sheet "settings": everything else about the literature models (setting, default, meaning)
SETTINGS = [
    ("apply_signs_to_metadata", "yes", "yes = the concepts' expected signs (and include = no -> sign 0) also apply to the "
                                       "data-driven screen of the mapped series"),
    ("ecm_enabled", "yes", "yes = estimate the error-correction model (and the 'theory long run + data short run' hybrid)"),
    ("ecm_name", "Literature: error-correction model", "name shown in the results"),
    ("ecm_source", "Hofmann (2004); Calza, Gartner & Sousa (2003); Muellbauer, St-Amant & Williams (2015)", "source shown in the results"),
    ("ecm_deflator", "inflation", "concept (or series) used to deflate the target and prices in the long-run relation"),
    ("ecm_long_run", "table", "table = the concepts with a long_run_form in sheet 'concepts'; or a list like "
                              "'house_prices:log_real, real_gdp:log'"),
    ("ecm_short_run", "core", "core = the drivers of the core model; or a list of variable|transform|Llag"),
    ("channels", "table", "table = the core concepts are the channels of the 'theory channels, data measures' hybrid"),
    ("regime_threshold", "house_prices|dlog|L1", "variable|transform|Llag whose level switches the threshold and "
                                                  "smooth-transition models between regimes"),
]


def _settings_value(v):
    """A settings cell as text ('' for empty).

    Args:
        v: cell value.

    Returns:
        str.
    """
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def read_workbook(path, full=False):
    """Read the literature workbook.

    Sheets: "models" (one row per driver of each published model, with its expected sign), "concepts" (the
    literature table), and optionally "settings" (error-correction model, channels, regime threshold, sign rule)
    and "references".

    Args:
        path: .xlsx file. full: also return the settings and references.

    Returns:
        (models, concepts) or, with full=True, a dict with models, concepts, settings (dict) and references.

    Raises:
        ValueError with a readable message when a sheet or column is missing.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    low = {k.strip().lower(): v for k, v in sheets.items()}
    if "models" not in low or "concepts" not in low:
        raise ValueError(f"{os.path.basename(path)} needs the sheets 'models' and 'concepts' (it has {list(sheets)})")
    c = low["concepts"].copy()
    c.columns = [str(x).strip() for x in c.columns]
    c = legacy_columns(c)
    csign = {}
    if "concept_id" in c and "expected_sign" in c:
        csign = {str(k).strip(): v for k, v in zip(c["concept_id"], pd.to_numeric(c["expected_sign"], errors="coerce"))}
    m = low["models"].copy()
    m.columns = [str(x).strip().lower() for x in m.columns]
    miss = [x for x in REQUIRED_MODEL_COLS if x not in m.columns]
    if miss:
        raise ValueError(f"sheet 'models' is missing column(s) {miss}; expected {MODEL_COLS}")
    m = m.dropna(subset=["model", "variable"])
    models = []
    for name, g in m.groupby("model", sort=False):
        feats, signs = [], {}
        for r in g.itertuples():
            f = f"{str(r.variable).strip()}|{str(r.transform).strip()}|L{int(r.lag)}"
            feats.append(f)
            sg = pd.to_numeric(getattr(r, "expected_sign", None), errors="coerce") if "expected_sign" in m else None
            if sg is None or pd.isna(sg):
                sg = csign.get(str(r.variable).strip())
            if sg is not None and not pd.isna(sg):
                signs[f] = int(sg)
        spec = {"name": str(name), "source": str(g["source"].dropna().iloc[0]) if g["source"].notna().any() else "",
                "features": feats, "signs": signs}
        if any(_yes(v) for v in g["core"]):
            spec["core"] = True
        models.append(spec)
    if not full:
        return models, c
    settings = {k: d for k, d, _ in SETTINGS}
    if "settings" in low:
        st = low["settings"]
        st.columns = [str(x).strip().lower() for x in st.columns]
        if {"setting", "value"} <= set(st.columns):
            for k, v in zip(st["setting"], st["value"]):
                if isinstance(k, str) and k.strip():
                    settings[k.strip()] = _settings_value(v)
    refs = low.get("references", pd.DataFrame())
    return {"models": models, "concepts": c, "settings": settings, "references": refs}


NO = {"no", "n", "false", "0", "off", "none"}


def clean_settings(settings):
    """Tidy the settings sheet: text values, and Excel's TRUE/FALSE or yes/no read the way each setting means them.

    Args:
        settings: dict setting -> cell value (from read_workbook(full=True)); may hold booleans, numbers or NaN.

    Returns:
        dict setting -> cleaned value: apply_signs_to_metadata / ecm_enabled as bool; channels as "table", "off" or a
        list of concept ids; ecm_long_run as "table" or a list of "name:form"; ecm_short_run as "core" or a list;
        regime_threshold as "variable|transform|Llag" or "" (= the default); names and sources as text.
    """
    def txt(v):
        if v is None or (isinstance(v, float) and v != v):          # empty cell (NaN)
            return ""
        return str(v).strip()

    def items(v):
        return [x.strip() for x in txt(v).replace(";", ",").split(",") if x.strip()]

    out = {k: txt(v) for k, v in (settings or {}).items()}
    for k in ("apply_signs_to_metadata", "ecm_enabled"):
        out[k] = txt(settings.get(k, "yes")).lower() not in NO      # empty = the default (yes)
    ch = out.get("channels", "").lower()
    out["channels"] = "table" if ch in YES | {"", "table"} else ("off" if ch in NO else items(out["channels"]))
    lr = out.get("ecm_long_run", "").lower()
    out["ecm_long_run"] = "table" if lr in YES | {"", "table"} else items(out["ecm_long_run"])
    sr = out.get("ecm_short_run", "").lower()
    out["ecm_short_run"] = "core" if sr in YES | {"", "core", "table"} else items(out["ecm_short_run"])
    thr = out.get("regime_threshold", "")
    out["regime_threshold"] = thr if thr.count("|") == 2 else ""
    if out.get("ecm_deflator", "").lower() in YES | NO:
        out["ecm_deflator"] = ""
    return out


def apply_settings(cfg, settings):
    """Put the workbook's settings into the configuration (literature.ecm, channels, sign rule, regime threshold).

    Args:
        cfg: configuration (changed in place). settings: dict from read_workbook(full=True).
    """
    st_ = clean_settings(settings)
    lit = cfg["literature"]
    lit["apply_signs_to_metadata"] = st_["apply_signs_to_metadata"]
    lit["channels"] = st_["channels"]
    if st_["ecm_enabled"]:
        lr = st_["ecm_long_run"]
        if lr != "table":
            lr = [{"mnemonic": x.split(":")[0].strip(), "form": (x.split(":") + ["log"])[1].strip() or "log"} for x in lr]
        lit["ecm"] = {"name": st_.get("ecm_name") or "Literature: error-correction model",
                      "source": st_.get("ecm_source", ""), "deflator": st_.get("ecm_deflator") or "inflation",
                      "long_run": lr, "short_run": st_["ecm_short_run"]}
    else:
        lit["ecm"] = None
    if st_["regime_threshold"]:
        cfg.setdefault("models", {}).setdefault("regime", {})["threshold"] = st_["regime_threshold"]


def write_workbook(path, models, concepts, settings=None, references=None):
    """Write the literature workbook (sheets models, concepts, settings, references, how to fill).

    Args:
        path: .xlsx file. models: model specs (config format; "signs" optional). concepts: the literature table.
        settings: dict setting -> value (None = defaults). references: references table (None = the example's).
    """
    from ..features.transforms import parse_feature
    csign = {}
    if concepts is not None and "concept_id" in concepts and "expected_sign" in concepts:
        csign = dict(zip(concepts["concept_id"].astype(str), pd.to_numeric(concepts["expected_sign"], errors="coerce")))
    rows = []
    for spec in models:
        for f in spec["features"]:
            v, tr, lag = parse_feature(f)
            sg = (spec.get("signs") or {}).get(f, csign.get(v))
            rows.append({"model": spec["name"], "source": spec.get("source", ""), "variable": v, "transform": tr,
                         "lag": lag, "expected_sign": None if sg is None or pd.isna(sg) else int(sg),
                         "core": "yes" if spec.get("core") else "no"})
    vals = {k: d for k, d, _ in SETTINGS}
    vals.update(settings or {})
    st = pd.DataFrame([(k, vals.get(k, d), mng) for k, d, mng in SETTINGS], columns=["setting", "value", "meaning"])
    if references is None or len(references) == 0:
        rp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config",
                          "literature_references.csv")
        references = pd.read_csv(rp) if os.path.exists(rp) else pd.DataFrame()
    help_ = pd.DataFrame({"How to fill this workbook": [
        "Everything about the literature models is in this workbook; the config only points to it (literature.workbook).",
        "Sheet 'models': one row per driver of each published model. Rows with the same 'model' name form one model.",
        "variable = a concept_id from sheet 'concepts' (e.g. house_prices) or one of your series names directly.",
        "transform = dlog (q/q log growth), yoy, dlog2, acc, gap, diff (q/q change), diff2, diff4, lvl, rgap; lag = quarters (0-4).",
        "expected_sign = +1 or -1: the sign the literature expects for this driver in this model; empty = the concept's sign. "
        "Drivers whose estimate has the other sign are flagged in the diagnostics ('wrong-sign drivers').",
        "core = yes on the rows of ONE model: the theory core used by the hybrids, the regime models, ARIMAX and the networks.",
        "Sheet 'concepts': map each concept to your series in series_name; include = no leaves a concept out of every model; "
        "expected_sign there is the default for the models and, with apply_signs_to_metadata, for the data-driven screen.",
        "long_run_form (log, log_real, level, real_rate) puts a concept in the error-correction model's long-run relation.",
        "Sheet 'settings': the error-correction model, the channels hybrid, the regime threshold and the sign rule.",
        "Sheet 'references': the studies behind the models and concepts (documentation only)."]})
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(rows, columns=MODEL_COLS).to_excel(w, sheet_name="models", index=False)
        concepts.drop(columns=[c for c in ("active", "mnemonic", "include_flag", "core_flag", "setup_note") if c in concepts],
                      errors="ignore").to_excel(w, sheet_name="concepts", index=False)
        st.to_excel(w, sheet_name="settings", index=False)
        references.to_excel(w, sheet_name="references", index=False)
        help_.to_excel(w, sheet_name="how to fill", index=False)
        for ws in w.sheets.values():
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(80, max(10, max(len(str(c.value or "")) for c in col) + 2))


def legacy_columns(t):
    """Accept workbooks made with older versions, where the series column was called 'moodys_name'."""
    if "moodys_name" in t.columns and "series_name" not in t.columns:
        t = t.rename(columns={"moodys_name": "series_name"})
    return t


def load(cfg):
    """The table with two helper columns: active (included and mapped) and mnemonic."""
    path = (cfg.get("literature") or {}).get("table")
    if not path or not os.path.exists(path):
        return None
    if path.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None)
        t = next((v for k, v in sheets.items() if k.strip().lower() == "concepts"), next(iter(sheets.values())))
    else:
        t = pd.read_csv(path)
    t.columns = [c.strip() for c in t.columns]
    t = legacy_columns(t)
    t["concept_id"] = t["concept_id"].astype(str).str.strip()
    t["mnemonic"] = t["series_name"].fillna("").astype(str).str.strip()
    t["include_flag"] = t["include"].map(_yes)
    t["core_flag"] = t.get("core", pd.Series("no", index=t.index)).map(_yes)
    t["expected_sign"] = pd.to_numeric(t["expected_sign"], errors="coerce").fillna(0).astype(int)
    t["active"] = t["include_flag"] & (t["mnemonic"] != "")
    return t.set_index("concept_id", drop=False)


def resolve(token, table):
    """concept_id -> series name (None if switched off or not mapped); anything else is returned unchanged.

    Args:
        token: concept_id or a series name. table: the literature table.
    """
    if table is None or token not in table.index:
        return token
    r = table.loc[token]
    return r["mnemonic"] if r["active"] else None


def resolve_feature(name, table):
    """'house_prices|dlog|L1' -> 'FHPI.ION|dlog|L1' (or None); plain feature names pass through.

    Args:
        table: the literature table.
    """
    if "|" not in name:
        m = resolve(name, table)
        return m
    code, rest = name.split("|", 1)
    m = resolve(code, table)
    return f"{m}|{rest}" if m else None


def apply_signs(meta, table, log=print):
    """Set the metadata expected sign from the table for mapped series; excluded concepts get sign 0.

    Args:
        table: the literature table.
    """
    if table is None:
        return meta
    meta = meta.copy()
    changed = []
    for _, r in table.iterrows():
        if r["mnemonic"] and r["mnemonic"] in meta.index:
            new = int(r["expected_sign"]) if r["include_flag"] else 0
            if int(meta.loc[r["mnemonic"], "expected_sign"]) != new:
                changed.append(f"{r['mnemonic']} {int(meta.loc[r['mnemonic'], 'expected_sign']):+d} -> {new:+d}")
                meta.loc[r["mnemonic"], "expected_sign"] = new
    if changed:
        log("  literature table changed expected signs: " + "; ".join(changed))
    return meta


def channels(table, meta, only=None):
    """Concepts as channels for the 'theory channels, data measures' hybrid (theme from your metadata).

    Args:
        table: the literature table. meta: metadata indexed by series name.
        only: concept ids to use (None = the core concepts).
    """
    out = []
    if table is None:
        return out
    rows = table[table["core_flag"] & table["active"]] if only is None else \
        table[table.index.astype(str).isin([str(c) for c in only]) & table["active"]]
    for _, r in rows.iterrows():
        if r["mnemonic"] in meta.index and r["expected_sign"] != 0:
            out.append({"concept": r["concept"], "theme": meta.loc[r["mnemonic"], "theme"], "sign": int(r["expected_sign"])})
    return out


def long_run(table):
    """Long-run (error-correction) regressors: active concepts with a long_run_form.

    Args:
        table: the literature table.
    """
    if table is None or "long_run_form" not in table:
        return []
    t = table[table["active"] & table["long_run_form"].fillna("").astype(str).str.strip().ne("")]
    return [{"mnemonic": r["mnemonic"], "form": str(r["long_run_form"]).strip(), "concept": r["concept"]}
            for _, r in t.iterrows()]


def status(table, ctx_hist_columns, used):
    """The table as a report: is each concept switched on, mapped, found in the data, and where it is used.

    Args:
        table: the literature table. ctx_hist_columns: mnemonics present in the macro data. used: concept_id
        -> models using it.
    """
    if table is None:
        return pd.DataFrame()
    s = table[["concept_id", "concept", "effect_on_loan_growth", "evidence", "include", "series_name"]].copy()
    s["found_in_data"] = table["mnemonic"].map(lambda m: bool(m) and m in ctx_hist_columns)
    s["status"] = [("switched off" if not r["include_flag"] else "not mapped" if not r["mnemonic"] else
                    "mapped, but not in the macro data" if r["mnemonic"] not in ctx_hist_columns else
                    "in use" if used.get(r["concept_id"]) else "available (not used by a model)")
                   for _, r in table.iterrows()]
    s["used_in"] = table["concept_id"].map(lambda c: ", ".join(used.get(c, [])))
    return s.reset_index(drop=True)
