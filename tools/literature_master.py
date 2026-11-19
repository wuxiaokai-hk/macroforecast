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
    if con is not None and "moodys_name" in con:
        for r in con.dropna(subset=["moodys_name"]).itertuples():
            lit.setdefault(str(r.moodys_name).strip(), set()).add(f"{r.target}:{r.concept_id}")
        extra = sorted(set(lit) - set(rows["code"]))
        if extra:                                   # named by a concept but not in any metadata
            rows = pd.concat([rows, pd.DataFrame({"code": extra})], ignore_index=True)
    rows["literature_uses"] = rows["code"].map(lambda c: len(lit.get(c, ()))).astype(int)
    old = sh.get("macro_names")
    keep = dict(zip(old["code"].astype(str), old["your_name"])) if old is not None and "your_name" in old else {}
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
    lead = ["code", "your_name"] + [c for c in ("in_your_data", "suggestion", "suggestion_score") if c in rows]
    rows = rows[lead + [c for c in rows.columns if c not in lead]]
    sh["macro_names"] = rows
    order = ["README", "macro_names"] + [k for k in sh if k not in ("README", "macro_names")]
    with pd.ExcelWriter(out or master) as w:
        for k in order:
            if k in sh:
                sh[k].to_excel(w, sheet_name=k, index=False)
    if macro:
        miss = rows[(rows["in_your_data"] != "yes")]
        lm = miss[miss["literature_uses"] > 0]
        log(f"macro_names: {len(rows)} series; {len(rows) - len(miss)} found in your data; "
            f"{len(miss)} not found ({len(lm)} of them used by literature models)")
    else:
        log(f"macro_names: {len(rows)} series ({int((rows['literature_uses'] > 0).sum())} used by literature models)")
    return rows


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
