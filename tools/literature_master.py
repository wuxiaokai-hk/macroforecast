"""Build, list or split a literature master file (every target's literature workbook and metadata in one .xlsx).

    python tools/literature_master.py list  config/literature_master.xlsx
    python tools/literature_master.py split config/literature_master.xlsx OUTDIR [--targets mortgage,deposits]
    python tools/literature_master.py build config/literature_master.xlsx \\
        mortgage=config/literature_workbook.xlsx,data/macro_metadata.csv \\
        deposits=,data/macro_metadata_deposits.csv          # no workbook: leave the first part empty
    python tools/literature_master.py build OUT.xlsx --folder FOLDER   # <target>_literature_workbook.xlsx and
                                                                        # macro_metadata_<target>.csv in FOLDER

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


def from_folder(folder):
    pairs = {}
    for f in glob.glob(os.path.join(folder, "**", "*_literature_workbook.xlsx"), recursive=True):
        pairs.setdefault(os.path.basename(f)[:-len("_literature_workbook.xlsx")], [None, None])[0] = f
    for f in glob.glob(os.path.join(folder, "**", "macro_metadata_*.csv"), recursive=True):
        pairs.setdefault(re.sub(r"^macro_metadata_|\.csv$", "", os.path.basename(f)), [None, None])[1] = f
    return {k: tuple(v) for k, v in sorted(pairs.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "split", "build"])
    ap.add_argument("master")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--targets", default="")
    ap.add_argument("--folder", default=None)
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
    else:
        pairs = from_folder(a.folder) if a.folder else {}
        for p in a.rest:
            t, files = p.split("=", 1)
            wb, md = (files.split(",") + [""])[:2]
            pairs[t] = (wb or None, md or None)
        print("wrote", build(a.master, pairs), f"({len(pairs)} targets)")


if __name__ == "__main__":
    main()
