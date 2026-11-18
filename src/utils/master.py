"""One master file for every target's literature workbook and macro metadata (literature_master.xlsx).

The master holds, for many targets, what the project otherwise reads from two files per target:

    sheet        columns                                                           becomes
    models       target + model, source, variable, transform, lag, expected_sign, core     the workbook's 'models'
    concepts     target + concept_id, concept, ..., moodys_name, notes                    the workbook's 'concepts'
    settings     target + setting, value, meaning                                       the workbook's 'settings'
    references   short, reference, link, used_by (targets, comma separated)             the workbook's 'references'
    how_to_fill  target + line                                                          the workbook's 'how to fill'
    metadata     target + mnemonic, description, geography, theme, transform,           macro_metadata_<target>.csv
                 expected_sign, sign_reason, ...
    targets, sign_matrix, README: summaries, ignored here.

Point the config at it in either way:

    literature:
      master: config/literature_master.xlsx     # workbook AND metadata from the master
      target: mortgage_loans                    # which target's rows (default: run_name)

or give the master as literature.workbook and/or data.metadata.file (each is then taken from the master).
The target's rows are written as ordinary files to outputs/<run>/inputs/master/ (only when they changed), and
the rest of the project reads those, so every model, the pre-flight check and the cache work unchanged.
"""
import os

import pandas as pd

WB_SHEETS = {"models": "models", "concepts": "concepts", "settings": "settings", "how_to_fill": "how to fill"}


def _sheets(path):
    try:
        return {str(k).strip().lower(): v for k, v in pd.read_excel(path, sheet_name=None).items()}
    except Exception:
        return {}


def _norm(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_master(path):
    """True when path is an Excel file whose models or metadata sheet has a 'target' column."""
    if not path or not str(path).lower().endswith((".xlsx", ".xlsm", ".xls")) or not os.path.exists(path):
        return False
    s = _sheets(path)
    return any(n in s and "target" in [str(c).strip().lower() for c in s[n].columns] for n in ("models", "metadata"))


def targets(path):
    """The targets in a master file, with what each one has."""
    s = _sheets(path)
    rows = {}
    for n in ("models", "concepts", "metadata"):
        if n in s:
            d = _norm(s[n])
            if "target" in d:
                for t, g in d.groupby(d["target"].astype(str).str.strip()):
                    rows.setdefault(t, {"target": t})[n] = len(g)
    return pd.DataFrame(list(rows.values())).fillna(0)


def _match(names, target):
    """The master's spelling of target (case, spaces, hyphens and underscores ignored)."""
    key = lambda x: str(x).strip().lower().replace(" ", "_").replace("-", "_")
    m = {key(n): n for n in names}
    return m.get(key(target))


def extract(path, target, folder):
    """Write one target's workbook and metadata from the master.

    Args:
        path: the master .xlsx. target: target name. folder: where to write.

    Returns:
        dict workbook=<xlsx or None>, metadata=<csv or None>, target=<the master's name for it>.

    Raises:
        ValueError naming the targets the master has when target is not one of them.
    """
    s = _sheets(path)
    names = set()
    for n in ("models", "concepts", "metadata"):
        if n in s and "target" in _norm(s[n]):
            names |= set(_norm(s[n])["target"].dropna().astype(str).str.strip())
    if not target:
        raise ValueError(f"{os.path.basename(path)} is a master file: say which target with literature.target "
                         f"(it has: {', '.join(sorted(names))})")
    t = _match(names, target)
    if t is None:
        raise ValueError(f"target '{target}' is not in {os.path.basename(path)}; set literature.target to one of: "
                         f"{', '.join(sorted(names))}")
    os.makedirs(folder, exist_ok=True)

    def rows(name):
        d = _norm(s[name])
        d = d[d["target"].astype(str).str.strip() == t].drop(columns="target")
        return d.reset_index(drop=True)

    out = {"workbook": None, "metadata": None, "target": t}
    if "models" in s and "concepts" in s and len(rows("models")):
        wb = {WB_SHEETS[k]: rows(k) for k in WB_SHEETS if k in s and "target" in _norm(s[k])}
        if "how to fill" in wb and "line" in wb["how to fill"]:
            wb["how to fill"] = wb["how to fill"][["line"]].rename(columns={"line": "How to fill this workbook"})
        if "references" in s:
            r = _norm(s["references"])
            if "used_by" in r:
                used = r["used_by"].fillna("").astype(str).str.split(",").map(lambda xs: {x.strip() for x in xs})
                r = r[used.map(lambda u: t in u)].drop(columns="used_by")
            wb["references"] = r.reset_index(drop=True)
        f = os.path.join(folder, f"{t}_literature_workbook.xlsx")
        _write_if_changed(f, wb)
        out["workbook"] = f
    if "metadata" in s and "target" in _norm(s["metadata"]):
        m = rows("metadata")
        m = m[[c for c in m.columns if c in ("mnemonic", "expected_sign") or m[c].notna().any()]]   # other targets' columns
        if len(m):
            f = os.path.join(folder, f"macro_metadata_{t}.csv")
            text = m.to_csv(index=False)
            if not (os.path.exists(f) and open(f, encoding="utf-8").read() == text):
                with open(f, "w", encoding="utf-8", newline="") as fh:
                    fh.write(text)
            out["metadata"] = f
    return out


def _write_if_changed(f, sheets):
    """Write the workbook only when its content changed (keeps the file hash, and so the cache, stable)."""
    if os.path.exists(f):
        try:
            old = pd.read_excel(f, sheet_name=None)
            if set(old) == set(sheets) and all(
                    old[k].fillna("").astype(str).values.tolist() == sheets[k].fillna("").astype(str).values.tolist()
                    and list(old[k].columns) == list(sheets[k].columns) for k in sheets):
                return
        except Exception:
            pass
    with pd.ExcelWriter(f) as w:
        for name, d in sheets.items():
            d.to_excel(w, sheet_name=name, index=False)


def resolve(path, kind, target, folder):
    """path if it is an ordinary file; the target's extracted file if path is a master.

    Args:
        kind: "workbook" or "metadata". target: target name. folder: where extracted files go.
    """
    if not is_master(path):
        return path
    f = extract(path, target, folder)[kind]
    if f is None:
        raise ValueError(f"{os.path.basename(path)} has no {kind} rows for target '{target}'")
    return f


def apply(cfg, folder):
    """Resolve literature.master / a master given as literature.workbook or data.metadata.file, in place.

    Args:
        cfg: configuration (paths already absolute). folder: where extracted files go.

    Returns:
        the target name used, or None when no master is involved.
    """
    lit = cfg.setdefault("literature", {})
    md = cfg["data"].setdefault("metadata", {})
    master = lit.get("master")
    if master:
        lit["workbook"] = lit.get("workbook") or master
        md["file"] = md.get("file") or master
    if not (is_master(lit.get("workbook")) or is_master(md.get("file"))):
        return None
    target = lit.get("target") or cfg.get("run_name")
    used = None
    if is_master(lit.get("workbook")):
        x = extract(lit["workbook"], target, folder)
        lit["workbook"], used = x["workbook"], x["target"]
        if lit["workbook"] is None:                  # metadata only: the theory-based models are off
            m = cfg.setdefault("models", {})
            m["literature"], m["hybrids"], m["same_growth"] = False, False, []
            aa = m.setdefault("auto_ardl", {})
            aa["on"] = [k for k in aa.get("on", ["core", "screen_first"]) if k != "core"] or ["screen_first"]
    if lit["workbook"]:                              # same-growth models: keep the concepts this target has
        ids = set(pd.read_excel(lit["workbook"], sheet_name="concepts")["concept_id"].astype(str).str.strip())
        m = cfg.setdefault("models", {})
        m["same_growth"] = [v for v in (m.get("same_growth") or [])
                            if not isinstance(v, str) or v in ids or "." in v]
    if is_master(md.get("file")):
        x = extract(md["file"], target, folder)
        if x["metadata"] is None:
            raise ValueError(f"{os.path.basename(md['file'])} has no metadata rows for target '{x['target']}': "
                             f"give its metadata file as data.metadata.file (the literature still comes from the master)")
        md["file"], used = x["metadata"], x["target"]
        md["sheet"] = None
    return used
