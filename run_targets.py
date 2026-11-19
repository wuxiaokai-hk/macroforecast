"""
Several targets from one data file: list them, choose, and run each one.

    python run_targets.py --config config/multi_target_example.yaml --list
    python run_targets.py --config config/multi_target_example.yaml --targets mortgage_balance,deposit_rate
    python run_targets.py --config config/multi_target_example.yaml --all --oos --story

The config's data file (data.bank.file) holds one column per target. Each column is matched to the literature
master (literature.master): the column name, or targets.map {column: master target}. Each chosen target runs as its
own run, outputs/<run_name>_<column>/, with its own literature models, expected signs and target form
(growth for balances, change for rates and ratios; targets.forms overrides). --oos adds the out-of-sample test,
--story the process story. A summary of all targets: outputs/<run_name>_targets/summary.csv and summary.md.

Optional config block:

    targets:
      map: {mortgage_balance: mortgage}          # column -> master target, when the names differ
      forms: {stage2_share: diff}                # override the suggested form
      labels: {cre_loans: CRE loans}             # names in titles
      units: {deposit_rate: '%'}                 # units in titles (rates)
      run: [mortgage_balance, deposit_rate]      # default selection when neither --targets nor --all is given
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.utils import multi  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def choose(cat, targets=None, everything=False, default=None):
    """Rows of the catalog to run: --targets (columns or master names), --all (every ready column), or targets.run."""
    if everything:
        return cat[cat["ready"]]
    want = [t.strip() for t in (targets or "").split(",") if t.strip()] or list(default or [])
    if not want:
        return cat.iloc[0:0]
    k = multi._key
    pick = cat[cat["column"].map(k).isin(map(k, want)) | cat["master_target"].map(k).isin(map(k, want))]
    missing = [w for w in want if k(w) not in set(pick["column"].map(k)) | set(pick["master_target"].map(k))]
    if missing:
        raise SystemExit(f"not in the data file: {missing}; columns: {list(cat['column'])}")
    return pick


def run_one(cfg_file, oos, story, log=print):
    """Pipeline (+ out-of-sample, + process story) for one target config. Returns the exit code of the pipeline."""
    py = sys.executable
    r = subprocess.run([py, os.path.join(HERE, "run_pipeline.py"), "--config", cfg_file])
    if r.returncode == 0 and oos:
        subprocess.run([py, os.path.join(HERE, "run_oos.py"), "--config", cfg_file])
    if r.returncode == 0 and story:
        subprocess.run([py, os.path.join(HERE, "tools", "process_story.py"), "--config", cfg_file])
    return r.returncode


def summary_row(cfg_file, row):
    """One line of the cross-target summary from a finished run."""
    c = load_config(cfg_file)
    out = c["out_dir"]
    rec = {}
    f = os.path.join(out, "formula", "recommended.json")
    if os.path.exists(f):
        rec = json.load(open(f, encoding="utf-8"))
    comp = os.path.join(out, "comparison", "1_model_comparison.csv")
    T = pd.read_csv(comp) if os.path.exists(comp) else pd.DataFrame()
    ar = T.set_index("model")[[x for x in T.columns if x.startswith("avg_1_")][-1]].get("AR(1) benchmark") if len(T) else None
    return {"column": row["column"], "master_target": row["master_target"], "form": row["form"],
            "best_model": rec.get("best_model"), "best_error": rec.get("best_error_pct"),
            "formula_model": rec.get("formula_model"), "formula_error": rec.get("formula_error_pct"),
            "AR1_error_backtest": ar, "basis": rec.get("basis"), "error_units": "%" if row["form"] == "dlog" else "units",
            "folder": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--list", action="store_true", help="show the columns, their master target and form, and stop")
    ap.add_argument("--targets", default=None, help="columns (or master target names) to run, comma separated")
    ap.add_argument("--all", action="store_true", help="run every column that is ready")
    ap.add_argument("--oos", action="store_true", help="also run the out-of-sample test")
    ap.add_argument("--story", action="store_true", help="also build the process story of each target")
    ap.add_argument("--write-only", action="store_true", help="write the per-target configs, do not run")
    a = ap.parse_args()

    raw = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg = load_config(a.config)
    cat = multi.catalog(cfg)
    pd.set_option("display.width", 200)
    print(f"\n{len(cat)} columns in {cfg['data']['bank']['file']}:\n")
    print(cat.drop(columns=[]).to_string(index=False))
    if a.list:
        return
    pick = choose(cat, a.targets, a.all, (raw.get("targets") or {}).get("run"))
    if pick.empty:
        raise SystemExit("\nChoose targets with --targets a,b or --all (or targets.run in the config).")
    folder = os.path.join(os.path.dirname(cfg["out_dir"]), f"{cfg['run_name']}_targets")
    os.makedirs(os.path.join(folder, "configs"), exist_ok=True)
    cfg_dir = os.path.dirname(os.path.abspath(a.config))
    rows, files = [], []
    for _, r in pick.iterrows():
        c = multi.target_config(raw, r, cfg["run_name"])
        for sec, key in (("data", "bank"), ("data", "events"), ("data", "metadata")):     # paths stay valid
            d = (c.get(sec) or {}).get(key) or {}
            if d.get("file") and not os.path.isabs(d["file"]) and not os.path.exists(d["file"]):
                d["file"] = os.path.normpath(os.path.join(cfg_dir, d["file"]))
        f = os.path.join(folder, "configs", f"{multi._key(r['column'])}.yaml")
        yaml.safe_dump(c, open(f, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
        files.append((f, r))
    print(f"\n{len(files)} target configs in {os.path.join(folder, 'configs')}")
    if a.write_only:
        return
    for i, (f, r) in enumerate(files, 1):
        print(f"\n===== [{i}/{len(files)}] {r['column']} ({r['form']}; literature: {r['master_target'] or 'none'}) =====")
        code = run_one(f, a.oos, a.story)
        rows.append(summary_row(f, r) if code == 0 else {"column": r["column"], "best_model": f"failed (exit {code})"})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(folder, "summary.csv"), index=False)
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Targets of {cfg['run_name']}\n\n" + S.drop(columns=["folder"], errors="ignore").to_markdown(index=False) + "\n")
    print("\n" + S.drop(columns=["folder"], errors="ignore").to_string(index=False))
    print(f"\nSummary: {os.path.join(folder, 'summary.md')}")


if __name__ == "__main__":
    main()
