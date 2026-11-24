"""Build, list or split a literature master file (every target's literature workbook and metadata in one .xlsx).

    python tools/literature_master.py list  config/literature_master.xlsx
    python tools/literature_master.py split config/literature_master.xlsx OUTDIR [--targets mortgage,deposits]
    python tools/literature_master.py build config/literature_master.xlsx \\
        mortgage=config/literature_workbook.xlsx,data/macro_metadata.csv \\
        deposits=,data/macro_metadata_deposits.csv          # no workbook: leave the first part empty
    python tools/literature_master.py build OUT.xlsx --folder FOLDER   # <target>_literature_workbook.xlsx and
                                                                        # macro_metadata_<target>.csv in FOLDER
    python tools/literature_master.py merge config/literature_master.xlsx --folder FOLDER [--out NEW.xlsx]
        # add the targets in FOLDER to an existing master (a target already there is replaced); default: in place
    python tools/literature_master.py names config/literature_master.xlsx --macro data/my/baseline.csv \
        [--macro-metadata data/my/metadata.csv] [--accept 0.6] [--out NEW.xlsx]
        # add or refresh the macro_names sheet: every macro series the master uses, whether your file has it,
        # and a suggested match from your metadata's descriptions. Fill your_name there: the ONLY place to edit
        # when your macro data come from another provider or use other names.

The project reads the master directly (literature.master + literature.target, see src/utils/master.py); split is
only needed to look at one target's files or to use them elsewhere. The layout is the same as literature_tool.py's
literature_master.xlsx, so either tool can read the other's file.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.utils import master as ms  # noqa: E402


def build(out, pairs):
    """pairs: {target: (workbook or None, metadata or None)}. Writes the master."""
    S = {k: [] for k in ("models", "concepts", "settings", "how_to_fill", "metadata")}
    refs = {}
    for t, (wb, md) in pairs.items():
        if wb:
            x = {k.strip().lower(): v for k, v in pd.read_excel(wb, sheet_name=None).items()}
            for k in ("models", "concepts", "settings"):
                if k in x:
                    S[k].append(x[k].assign(target=t))
            if "how to fill" in x:
                h = x["how to fill"]
                S["how_to_fill"].append(pd.DataFrame({"target": t, "line": h.iloc[:, 0].values}))
            if "references" in x:
                for r in x["references"].fillna("").itertuples(index=False):
                    refs.setdefault(tuple(r), set()).add(t)
        if md:
            S["metadata"].append(pd.read_csv(md).assign(target=t))
    sheets = {}
    tg = []
    for t, (wb, md) in pairs.items():
        mo = [d for d in S["models"] if d["target"].iloc[0] == t]
        core = sorted(set(mo[0].loc[mo[0]["core"].astype(str).str.lower() == "yes", "model"])) if mo else []
        tg.append({"target": t, "has_workbook": bool(wb), "has_metadata": bool(md),
                   "models": int(mo[0]["model"].nunique()) if mo else 0, "core_model": "; ".join(core)})
    sheets["targets"] = pd.DataFrame(tg)
    for k in ("concepts", "models", "settings"):
        if S[k]:
            d = pd.concat(S[k], ignore_index=True)
            sheets[k] = d[["target"] + [c for c in d.columns if c != "target"]]
    if refs:
        ref_cols = ["short", "reference", "link"]
        sheets["references"] = pd.DataFrame([dict(zip(ref_cols, k), used_by=", ".join(sorted(v)))
                                             for k, v in refs.items()])
    if S["metadata"]:
        d = pd.concat(S["metadata"], ignore_index=True)
        sheets["metadata"] = d[["target"] + [c for c in d.columns if c != "target"]]
        sm = d.pivot_table(index=["mnemonic"], columns="target", values="expected_sign", aggfunc="first")
        desc = d.drop_duplicates("mnemonic").set_index("mnemonic")["description"]
        sm.insert(0, "description", desc.reindex(sm.index))
        sheets["sign_matrix"] = sm.reset_index()
    if S["how_to_fill"]:
        sheets["how_to_fill"] = pd.concat(S["how_to_fill"], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with pd.ExcelWriter(out) as w:
        for k, d in sheets.items():
            d.to_excel(w, sheet_name=k, index=False)
    return out


def merge(base, pairs, out):
    """Add per-target files to an existing master; a target already in it is replaced. Returns (out, added, replaced)."""
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "new.xlsx")
    build(tmp, pairs)
    old = pd.read_excel(base, sheet_name=None)
    new = pd.read_excel(tmp, sheet_name=None)
    names = set(pairs)
    key = lambda x: str(x).strip().lower().replace(" ", "_").replace("-", "_")
    replaced = sorted(t for t in names if any(key(t) in set(old[k]["target"].map(key))
                                             for k in ("models", "metadata", "concepts") if k in old and "target" in old[k]))
    sheets = {}
    for k in list(old) + [k for k in new if k not in old]:
        if k in ("sign_matrix",):
            continue
        o, n = old.get(k), new.get(k)
        if k == "references":
            r = pd.concat([x for x in (o, n) if x is not None], ignore_index=True).fillna("")
            r["used_by"] = r["used_by"].astype(str)
            keep = [c for c in ("short", "reference", "link") if c in r]
            r = r.groupby(keep, sort=False, as_index=False)["used_by"].agg(
                lambda v: ", ".join(sorted({t.strip() for x in v for t in x.split(",") if t.strip()})))
            sheets[k] = r
        elif o is not None and "target" in o:
            o = o[~o["target"].map(key).isin({key(t) for t in names})]
            sheets[k] = pd.concat([o, n], ignore_index=True) if n is not None else o
        else:
            sheets[k] = o if o is not None else n
    if "metadata" in sheets:
        d = sheets["metadata"]
        sm = d.pivot_table(index="mnemonic", columns="target", values="expected_sign", aggfunc="first")
        desc = d.drop_duplicates("mnemonic").set_index("mnemonic")["description"]
        sm.insert(0, "description", desc.reindex(sm.index))
        sheets["sign_matrix"] = sm.reset_index()
    order = [k for k in ("README", "targets", "concepts", "models", "settings", "references", "metadata", "sign_matrix",
                         "how_to_fill") if k in sheets] + [k for k in sheets if k not in (
        "README", "targets", "concepts", "models", "settings", "references", "metadata", "sign_matrix", "how_to_fill")]
    with pd.ExcelWriter(out) as w:
        for k in order:
            sheets[k].to_excel(w, sheet_name=k, index=False)
    return out, sorted(names - set(replaced)), replaced


def _tokens(x):
    import re as _re
    stop = {"the", "of", "and", "in", "for", "a", "index", "total", "rate", "sa", "saar", "nsa", "ratio", "level"}
    return {w for w in _re.findall(r"[a-z0-9+&]+", str(x).lower()) if w not in stop and (len(w) > 1 or w.isdigit())}


def names_sheet(master, macro=None, macro_meta=None, out=None, log=print, accept=None):
    """Add or refresh the macro_names sheet (keeps every your_name already filled in). Returns the sheet."""
    sh = pd.read_excel(master, sheet_name=None)
    md = sh["metadata"]
    md = md[md["mnemonic"].notna()]
    g = md.groupby("mnemonic")
    rows = pd.DataFrame({"code": list(g.groups)})
    first = md.drop_duplicates("mnemonic").set_index("mnemonic")
    for c in ("description", "geography", "theme", "transform"):
        if c in first:
            rows[c] = rows["code"].map(first[c])
    sign = md[pd.to_numeric(md["expected_sign"], errors="coerce").fillna(0) != 0].groupby("mnemonic")["target"].nunique()
    rows["targets_with_a_sign"] = rows["code"].map(sign).fillna(0).astype(int)
    con = sh.get("concepts")
    lit = {}
    scol = next((c for c in ("series_name", "moodys_name") if con is not None and c in con), None)
    if scol:
        for t, cid, v in zip(con["target"], con["concept_id"], con[scol]):
            if isinstance(v, str) and v.strip():
                lit.setdefault(v.strip(), set()).add(f"{t}:{cid}")
        extra = sorted(set(lit) - set(rows["code"]))
        if extra:                                   # named by a concept but not in any metadata
            rows = pd.concat([rows, pd.DataFrame({"code": extra})], ignore_index=True)
    rows["literature_uses"] = rows["code"].map(lambda c: len(lit.get(c, ()))).astype(int)
    rows["source"] = "master"
    old = sh.get("macro_names")
    keep = dict(zip(old["code"].astype(str), old["your_name"])) if old is not None and "your_name" in old else {}
    mine = pd.DataFrame()
    if old is not None:                              # series you added (source empty): kept as they are
        o = old.copy()
        src_ = o["source"].fillna("").astype(str).str.strip() if "source" in o else pd.Series("master", index=o.index)
        mine = o[(src_ != "master") & ~o["code"].astype(str).isin(set(rows["code"]))
                 & (o["code"].notna() | o["your_name"].notna())]
        mine = mine.drop(columns=[c for c in ("in_your_data", "suggestion", "suggestion_score") if c in mine])
    rows["your_name"] = rows["code"].map(lambda c: keep.get(c, ""))
    rows["your_name"] = rows["your_name"].fillna("")
    if macro:
        from src.utils.io import read_table
        cols = [str(c) for c in read_table(macro, None).columns]
        low = {c.lower(): c for c in cols}
        name = rows.apply(lambda r: r["your_name"] or r["code"], axis=1)
        rows["in_your_data"] = name.map(lambda n: "yes" if n in cols else ("case differs: " + low[n.lower()]
                                                                           if n.lower() in low else "no"))
        if macro_meta:
            from src.utils.io import read_table as rt
            mm = rt(macro_meta, None)
            cc = {c.lower(): c for c in mm.columns}
            mcol = cc.get("mnemonic") or cc.get("code") or cc.get("name") or mm.columns[0]
            dcol = cc.get("description") or cc.get("label") or mm.columns[1]
            cand = [(str(a), _tokens(b), str(b)) for a, b in zip(mm[mcol], mm[dcol])]
            sug, score = [], []
            for r in rows.itertuples():
                if r.in_your_data == "yes" or not isinstance(r.description, str):
                    sug.append(""), score.append(None)
                    continue
                t = _tokens(str(r.description) + " " + str(getattr(r, "geography", "")))
                best = max(cand, key=lambda x: len(t & x[1]) / (len(t | x[1]) or 1), default=None)
                sc = len(t & best[1]) / (len(t | best[1]) or 1) if best else 0
                sug.append(f"{best[0]} ({best[2]})" if best and sc >= 0.3 else ""), score.append(round(sc, 2) if best else None)
            rows["suggestion"], rows["suggestion_score"] = sug, score
            if accept is not None:                      # fill your_name from good suggestions (review them!)
                ok = (rows["your_name"] == "") & (rows["suggestion_score"].fillna(0) >= accept) & (rows["suggestion"] != "")
                rows.loc[ok, "your_name"] = rows.loc[ok, "suggestion"].str.split(" ").str[0]
                log(f"  your_name filled from {int(ok.sum())} suggestions with a score of at least {accept}: review them")
                name = rows.apply(lambda r: r["your_name"] or r["code"], axis=1)
                rows["in_your_data"] = name.map(lambda n: "yes" if n in cols else "no")
    rows = rows.sort_values(["literature_uses", "targets_with_a_sign", "code"], ascending=[False, False, True])
    for c in ("expected_sign", "signs_by_target"):
        rows[c] = ""
    if len(mine):
        rows = pd.concat([rows, mine], ignore_index=True)
    lead = ["code", "your_name"] + [c for c in ("in_your_data", "suggestion", "suggestion_score") if c in rows]
    tail = ["expected_sign", "signs_by_target", "source"]
    rows = rows[lead + [c for c in rows.columns if c not in lead + tail] + tail]
    sh["macro_names"] = rows
    sh["target_names"] = target_names_sheet(sh)
    sh["README"] = readme(sh.get("README"))
    order = ["README", "target_names", "macro_names"] + [k for k in sh if k not in ("README", "target_names", "macro_names")]
    with pd.ExcelWriter(out or master) as w:
        for k in order:
            if k in sh:
                sh[k].to_excel(w, sheet_name=k, index=False)
    add_views(out or master)
    style_inputs(out or master)
    if macro:
        miss = rows[(rows["in_your_data"] != "yes")]
        lm = miss[miss["literature_uses"] > 0]
        log(f"macro_names: {len(rows)} series; {len(rows) - len(miss)} found in your data; "
            f"{len(miss)} not found ({len(lm)} of them used by literature models)")
    else:
        log(f"macro_names: {len(rows)} series ({int((rows['literature_uses'] > 0).sum())} used by literature models)")
    return rows


def target_names_sheet(sh, data=None):
    """The target_names sheet: one row per target of the master (source = master), your_name kept from the old
    sheet, and the rows you added (source empty) kept at the bottom."""
    import re as _re
    tg = sh.get("targets")
    if tg is None:
        tg = pd.DataFrame({"target": sorted(set(sh["metadata"]["target"].dropna().astype(str)))})
    rows = pd.DataFrame({"target": tg["target"].astype(str)})
    lab = tg["label"].astype(str) if "label" in tg else pd.Series("", index=tg.index)
    rows["label"] = lab.str.replace(r",\s*quarterly.*$", "", regex=True).str.replace(r"\s*\(workbook not.*$", "", regex=True)
    rows["form"] = lab.map(lambda x: "diff" if "(diff)" in x else "dlog" if "(dlog)" in x else "")
    rows["units"] = lab.map(lambda x: (_re.search(r"\(([^()]*%[^()]*|pp)\)", x) or [None, ""])[1])
    rows["literature"] = (tg["has_workbook"].astype(str) if "has_workbook" in tg else "").map(
        lambda v: "yes" if v == "yes" else "no")
    rows["like"] = ""
    rows["source"] = "master"
    old = sh.get("target_names")
    if old is not None and "your_name" in old:
        o = old.copy()
        o["target"] = o["target"].fillna("").astype(str).str.strip()
        keep = {t: v for t, v in zip(o["target"], o["your_name"]) if isinstance(v, str) and v.strip()}
        rows["your_name"] = rows["target"].map(keep).fillna("")
        for c in ("form", "units", "label"):          # your own entries win over the defaults
            if c in o:
                mine = {t: v for t, v in zip(o["target"], o[c]) if isinstance(v, str) and v.strip()}
                rows[c] = [mine.get(t, v) for t, v in zip(rows["target"], rows[c])]
        src_ = o["source"].fillna("").astype(str).str.strip() if "source" in o else pd.Series("master", index=o.index)
        added = o[(src_ != "master") & ~o["target"].isin(set(rows["target"])) & (o["target"] != "")]
        if len(added):
            rows = pd.concat([rows, added[[c for c in added.columns if c in
                                           ("target", "your_name", "label", "form", "units", "like", "notes")]]],
                             ignore_index=True)
    else:
        rows["your_name"] = ""
    for c in ("your_name", "like", "source", "notes"):
        if c not in rows:
            rows[c] = ""
    return rows[["target", "your_name", "form", "units", "label", "literature", "like", "notes", "source"]].fillna("")


README_TOP = [
    "YOUR DATA: fill two sheets, nothing else",
    "target_names : your financial data. One row per target of the master. In your_name write the column name of that "
    "series in YOUR data file. Blank = you do not have it: it is skipped. Fill only the ones you have.",
    "macro_names  : your macro data. One row per macro series the master uses (code = the master's code, e.g. the "
    "Moody's mnemonic). In your_name write the name of that series in YOUR macro files. Blank = the same as the code; "
    "a series your files do not have is simply not used.",
    "Adding a target the master does not have: a new row at the bottom of target_names with target (your own name), "
    "your_name, form (dlog for balances and income, diff for rates and ratios), units, label, and like = an existing "
    "target whose literature and signs it borrows (blank = no literature; every macro series is tried with a free sign).",
    "Adding a macro series the master does not have: a new row at the bottom of macro_names with your_name, "
    "description, geography, theme, transform (dlog or diff) and expected_sign (+1/-1/0 for every target); "
    "signs_by_target overrides it per target, e.g. 'mortgage_loans:+1; deposits:-1'.",
    "Leave 'source' empty on rows you add (source = master marks the master's own rows). The yellow columns are the "
    "ones you fill; hover over a header for a note. Every other sheet is the literature and is read only for you.",
    "The 'view: your name' columns in concepts, metadata and sign_matrix show your names live (Excel formulas).",
    "",
]


def readme(old):
    """The README sheet: the 'your data' instructions first, then the master's own description."""
    col = "literature_master.xlsx - how to use"
    lines = [] if old is None else [str(x) for x in old.iloc[:, 0].tolist()]
    if old is not None:
        col = str(old.columns[0])
    if README_TOP[0] in lines:                       # refresh: drop the old block
        i = lines.index(README_TOP[0])
        lines = lines[:i] + lines[i + len(README_TOP):]
    lines = [x for x in lines if not x.strip().lower().startswith("macro_names") or "your_name" not in x]
    return pd.DataFrame({col: README_TOP + lines})


def style_inputs(path):
    """Colour the columns you fill (your_name, and the inputs of new rows), add notes on the headers, freeze panes."""
    from openpyxl import load_workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Font, PatternFill
    wb = load_workbook(path)
    fill = PatternFill("solid", fgColor="FFF4D6")
    tips = {
        "target_names": {
            "your_name": "The column of this series in YOUR data file. Blank = you do not have it (skipped).",
            "form": "dlog = log growth (balances, income); diff = change (rates, ratios, shares).",
            "like": "For a target you add: an existing target whose literature and signs it borrows. "
                    "Blank = no literature, every macro series is a candidate with a free sign.",
            "source": "master = from the master. Leave empty on rows you add.",
            "target": "Add a new target in an empty row at the bottom: a name of your own."},
        "macro_names": {
            "your_name": "The name of this series in YOUR macro files. Blank = the same as the code; a series your "
                         "files do not have is not used.",
            "expected_sign": "Rows you add: the sign of the effect for every target (+1, -1, 0 = free).",
            "signs_by_target": "Rows you add: target:sign pairs that override expected_sign, e.g. "
                               "mortgage_loans:+1; deposits:-1",
            "source": "master = from the master. Leave empty on rows you add."}}
    for sheet, notes in tips.items():
        if sheet not in wb.sheetnames:
            continue
        ws = wb[sheet]
        hdr = [c.value for c in ws[1]]
        for i, h in enumerate(hdr, 1):
            c = ws.cell(row=1, column=i)
            c.font = Font(bold=True)
            if h in notes:
                c.comment = Comment(notes[h], "literature_master")
            if h in ("your_name",) or (h in ("form", "units", "label", "like", "expected_sign", "signs_by_target")):
                for r in range(1, ws.max_row + 30):
                    ws.cell(row=r, column=i).fill = fill
        ws.freeze_panes = "C2"
        for i, h in enumerate(hdr, 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
                {"your_name": 24, "label": 48, "description": 44, "notes": 30, "signs_by_target": 30}.get(h, 16)
    wb.save(path)


def add_views(path):
    """Next to every master code (concepts.series_name, metadata.mnemonic, sign_matrix.mnemonic), a formula column
    'view: your name' that looks the code up in macro_names: Excel shows your names everywhere, live, while the
    only place to edit stays macro_names. The code ignores these columns."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = load_workbook(path)
    if "macro_names" not in wb.sheetnames:
        return
    mn = wb["macro_names"]
    hdr = [c.value for c in mn[1]]
    ci, ni = hdr.index("code") + 1, hdr.index("your_name") + 1
    rng_c = f"macro_names!${get_column_letter(ci)}:${get_column_letter(ci)}"
    rng_n = f"macro_names!${get_column_letter(ni)}:${get_column_letter(ni)}"
    grey = PatternFill("solid", fgColor="EEF3F6")
    for sheet, col in (("concepts", ("series_name", "moodys_name")), ("metadata", ("mnemonic",)),
                       ("sign_matrix", ("mnemonic",))):
        if sheet not in wb.sheetnames:
            continue
        ws = wb[sheet]
        h = [c.value for c in ws[1]]
        if "view: your name" in h:                     # refresh: drop the old view column
            ws.delete_cols(h.index("view: your name") + 1)
            h = [c.value for c in ws[1]]
        src = next((c for c in col if c in h), None)
        if not src:
            continue
        si = h.index(src) + 1
        ws.insert_cols(si + 1)
        L = get_column_letter(si)
        cell = ws.cell(row=1, column=si + 1, value="view: your name")
        cell.font, cell.fill = Font(bold=True, italic=True, color="1F6F8B"), grey
        for r in range(2, ws.max_row + 1):
            if ws.cell(row=r, column=si).value in (None, ""):
                continue
            f = (f'=IF({L}{r}="","",IFERROR(IF(INDEX({rng_n},MATCH({L}{r},{rng_c},0))="",{L}{r},'
                 f'INDEX({rng_n},MATCH({L}{r},{rng_c},0))),{L}{r}))')
            c = ws.cell(row=r, column=si + 1, value=f)
            c.font, c.fill = Font(italic=True, color="1F6F8B"), grey
    wb.save(path)


def from_folder(folder):
    pairs = {}
    for f in glob.glob(os.path.join(folder, "**", "*_literature_workbook.xlsx"), recursive=True):
        pairs.setdefault(os.path.basename(f)[:-len("_literature_workbook.xlsx")], [None, None])[0] = f
    for f in glob.glob(os.path.join(folder, "**", "macro_metadata_*.csv"), recursive=True):
        pairs.setdefault(re.sub(r"^macro_metadata_|\.csv$", "", os.path.basename(f)), [None, None])[1] = f
    return {k: tuple(v) for k, v in sorted(pairs.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "split", "build", "merge", "names"])
    ap.add_argument("master")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--targets", default="")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--out", default=None, help="merge / names: write here instead of over the master")
    ap.add_argument("--macro", default=None, help="names: one of your scenario files (its columns are your names)")
    ap.add_argument("--macro-metadata", default=None, help="names: your metadata (descriptions) for suggestions")
    ap.add_argument("--accept", type=float, default=None, metavar="SCORE",
                    help="names: copy suggestions with at least this score (0-1, e.g. 0.6) into your_name")
    a = ap.parse_args()
    if a.command == "list":
        print(ms.targets(a.master).to_string(index=False))
    elif a.command == "split":
        if not a.rest:
            raise SystemExit("split needs an output folder")
        names = [t.strip() for t in a.targets.split(",") if t.strip()] or list(ms.targets(a.master)["target"])
        for t in names:
            x = ms.extract(a.master, t, a.rest[0])
            print(f"{x['target']}: {x['workbook'] or '(no workbook)'}  {x['metadata'] or '(no metadata)'}")
    elif a.command == "names":
        names_sheet(a.master, a.macro, a.macro_metadata, a.out, accept=a.accept)
    elif a.command == "merge":
        pairs = from_folder(a.folder) if a.folder else {}
        for p in a.rest:
            t, files = p.split("=", 1)
            wb, md = (files.split(",") + [""])[:2]
            pairs[t] = (wb or None, md or None)
        out, added, rep = merge(a.master, pairs, a.out or a.master)
        print(f"wrote {out}: {len(added)} added ({', '.join(added)}), {len(rep)} replaced ({', '.join(rep) or 'none'})")
        names_sheet(out)                                # new targets' series join the names map
    else:
        pairs = from_folder(a.folder) if a.folder else {}
        for p in a.rest:
            t, files = p.split("=", 1)
            wb, md = (files.split(",") + [""])[:2]
            pairs[t] = (wb or None, md or None)
        print("wrote", build(a.master, pairs), f"({len(pairs)} targets)")


if __name__ == "__main__":
    main()
