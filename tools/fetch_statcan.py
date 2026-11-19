"""
Real example data from Statistics Canada: download, build the files the project reads, write the config.

    python tools/fetch_statcan.py                              # uses config/statcan_series.yaml
    python tools/fetch_statcan.py --spec my_series.yaml
    python tools/fetch_statcan.py --describe 14100287          # a table's dimensions and members
    python run_targets.py --config config/statcan.yaml --list  # then run, e.g. --all

Statistics Canada's Web Data Service (https://www150.statcan.gc.ca/t1/wds/rest/) is free and needs no key. For each
series in the spec, the tool reads the table's metadata (getCubeMetadata), turns the member names into the
coordinate, and downloads the series (getDataFromCubePidCoordAndLatestNPeriods). Monthly data become quarterly.
Downloads are cached in data/<name>/raw/, so a re-run works offline; --refresh downloads again.

Writes, in data/<name>/:
    targets.csv             date + one column per target (household mortgage debt, consumer credit, ...)
    scenarios/<name>.csv    one file per scenario: the actual history, then the scenario path
    macro_metadata.csv      description, theme, geography, transform and expected sign of every series
    sources.csv             table, coordinate, vector and title of every series, with the download date
and config/<name>.yaml, a run_targets.py-ready config (from config/example_config.yaml, with the dates set from the
data). The scenarios are illustrative, built from simple rules (see the spec); replace them with a provider's or your
own for real use. Statistics Canada data: Statistics Canada Open Licence (https://www.statcan.gc.ca/en/terms-conditions/open-licence).

Needs internet access to www150.statcan.gc.ca (behind a corporate proxy, set HTTPS_PROXY); no extra packages.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

WDS = "https://www150.statcan.gc.ca/t1/wds/rest/"
SCALE = {0: 1, 1: 10, 2: 100, 3: 1e3, 4: 1e4, 5: 1e5, 6: 1e6, 7: 1e7, 8: 1e8, 9: 1e9}


# ------------------------------------------------------------------------------------------------- web service
def post(method, body, tries=4):
    """POST to the Web Data Service; returns the parsed JSON (retries on network errors and rate limits).
    Standard library only; the HTTPS_PROXY environment variable is honoured."""
    import urllib.error
    import urllib.request
    data = json.dumps(body).encode("utf-8")
    for i in range(tries):
        try:
            req = urllib.request.Request(WDS + method, data=data, method="POST",
                                         headers={"Content-Type": "application/json", "User-Agent": "macroforecast"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(2 + 2 * i)
                continue
            if i == tries - 1:
                raise SystemExit(f"Statistics Canada answered {e.code} to {method}: {e.reason}")
        except Exception as e:
            if i == tries - 1:
                raise SystemExit(f"Statistics Canada did not answer {method}: {e}\n"
                                 "Check the internet connection (and HTTPS_PROXY behind a corporate proxy).")
        time.sleep(1 + i)


def cached(path, refresh, fetch):
    if os.path.exists(path) and not refresh:
        return json.load(open(path, encoding="utf-8"))
    out = fetch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"))
    return out


def metadata(pid, raw, refresh=False):
    """A table's metadata: title, frequency and dimensions with their members."""
    res = cached(os.path.join(raw, f"meta_{pid}.json"), refresh, lambda: post("getCubeMetadata", [{"productId": int(pid)}]))
    r = res[0] if isinstance(res, list) else res
    if r.get("status") != "SUCCESS":
        raise SystemExit(f"table {pid}: {r.get('object') or r}")
    return r["object"]


# ------------------------------------------------------------------------------------------------- matching names
def _words(x):
    return re.sub(r"[^a-z0-9%]+", " ", str(x).lower()).strip()


def find_member(dim, want):
    """The member of a dimension named by `want` (exact name, else every word of a string or list contained)."""
    mem = dim["member"]
    if want is None:
        return mem[0], f"first member '{mem[0]['memberNameEn']}'"
    if isinstance(want, int):
        hit = [m for m in mem if m["memberId"] == want]
        if hit:
            return hit[0], None
    names = [m["memberNameEn"] for m in mem]
    exact = [m for m in mem if _words(m["memberNameEn"]) == _words(want)] if isinstance(want, str) else []
    if exact:
        return exact[0], None
    words = [_words(w) for w in (want if isinstance(want, list) else [want])]
    hit = [m for m in mem if all(w in _words(m["memberNameEn"]) for w in words)]
    if not hit:
        raise SystemExit(f"'{dim['dimensionNameEn']}': no member matches {want}. Members:\n  - " + "\n  - ".join(names[:80])
                         + ("\n  ..." if len(names) > 80 else ""))
    hit.sort(key=lambda m: len(m["memberNameEn"]))
    note = (f"'{want}' matched {len(hit)} members; using the shortest, '{hit[0]['memberNameEn']}'" if len(hit) > 1 else None)
    return hit[0], note


def coordinate(meta, members):
    """Coordinate string ('1.2.0...') from member names per dimension; notes on the choices made."""
    dims = sorted(meta["dimension"], key=lambda d: d["dimensionPositionId"])
    given = {_words(k): v for k, v in (members or {}).items()}
    unknown = set(given) - {_words(d["dimensionNameEn"]) for d in dims}
    if unknown:
        raise SystemExit(f"table {meta['productId']}: no dimension named {sorted(unknown)}; dimensions: "
                         f"{[d['dimensionNameEn'] for d in dims]}")
    ids, notes, labels = [], [], []
    for d in dims:
        m, note = find_member(d, given.get(_words(d["dimensionNameEn"])))
        ids.append(str(m["memberId"]))
        labels.append(m["memberNameEn"])
        if note:
            notes.append(f"{d['dimensionNameEn']}: {note}")
    return ".".join(ids + ["0"] * (10 - len(ids))), notes, labels


def download(pid, coord, raw, refresh=False, n=800):
    res = cached(os.path.join(raw, f"data_{pid}_{coord.replace('.', '-')}.json"), refresh,
                 lambda: post("getDataFromCubePidCoordAndLatestNPeriods",
                              [{"productId": int(pid), "coordinate": coord, "latestN": n}]))
    r = res[0] if isinstance(res, list) else res
    if r.get("status") != "SUCCESS":
        raise SystemExit(f"table {pid}, coordinate {coord}: {r.get('object') or r}")
    o = r["object"]
    pts = o.get("vectorDataPoint") or []
    s = pd.Series({p["refPer"]: (float(p["value"]) if p.get("value") not in (None, "") else np.nan) for p in pts},
                  dtype=float)
    s.index = pd.to_datetime(s.index)
    sf = SCALE.get(int(pts[0].get("scalarFactorCode", 0) or 0), 1) if pts else 1
    return s.sort_index(), o.get("vectorId"), sf


def quarterly(s, agg):
    q = s.groupby(s.index.to_period("Q"))
    out = q.last() if agg == "last" else q.mean()
    n = q.size()
    full = n >= n.mode().iloc[0] if len(n) else n                 # drop a quarter with missing months at the end
    return out[full]


# ------------------------------------------------------------------------------------------------- scenarios
def scenario_paths(H, spec, codes, transforms, hist_end, horizon):
    """Each scenario: history, then the baseline path plus the scenario's shock (see the spec)."""
    fut = pd.period_range(hist_end + 1, periods=horizon, freq="Q")
    base = {}
    for c in codes:
        x = H[c].dropna()
        if transforms[c] == "dlog":
            g = np.log(x).diff().tail(20).mean()
            base[c] = x.iloc[-1] * np.exp(g * np.arange(1, horizon + 1))
        else:
            base[c] = np.repeat(x.iloc[-1], horizon)
    out = {}
    for name, sc in (spec or {"Baseline": {}}).items():
        sc = sc or {}
        peak = int(sc.get("peak", 4))
        t = np.arange(1, horizon + 1)
        shape = np.where(t <= peak, t / peak, 1 - 0.5 * (t - peak) / max(horizon - peak, 1))
        P = {}
        for c in codes:
            s = float((sc.get("shocks") or {}).get(c, 0.0))
            P[c] = base[c] * (1 + s / 100 * shape) if transforms[c] == "dlog" else base[c] + s * shape
        F = pd.DataFrame(P, index=fut)
        out[name] = pd.concat([H[codes].loc[:hist_end], F])
    return out


# ------------------------------------------------------------------------------------------------- main
def build(spec_file, refresh=False, log=print):
    spec = yaml.safe_load(open(spec_file, encoding="utf-8"))
    name = spec.get("name", "statcan")
    folder = os.path.join(HERE, "data", name)
    raw = os.path.join(folder, "raw")
    start = pd.Period(spec.get("start", "1995Q1"), freq="Q")
    series, src = {}, []
    for role in ("targets", "macro"):
        for it in spec.get(role) or []:
            meta = metadata(it["table"], raw, refresh)
            coord, notes, labels = coordinate(meta, it.get("members"))
            s, vec, sf = download(it["table"], coord, raw, refresh)
            if s.dropna().empty:
                raise SystemExit(f"{it['code']}: no data at coordinate {coord} of table {it['table']} ({labels})")
            freq = "monthly" if s.index.to_series().diff().dt.days.median() < 40 else "quarterly"
            q = quarterly(s, it.get("agg", "last" if role == "targets" else "mean"))
            series[it["code"]] = q.loc[start:]
            src.append({"code": it["code"], "role": role, "table": it["table"], "title": meta.get("cubeTitleEn"),
                        "coordinate": coord, "vector": f"v{vec}" if vec else "", "members": " | ".join(labels),
                        "frequency": freq, "scalar_factor": sf, "first": str(q.index[0]), "last": str(q.index[-1]),
                        "downloaded": dt.date.today().isoformat()})
            log(f"  {it['code']:<28s} table {it['table']} v{vec}  {q.index[0]}-{q.index[-1]}  " + "; ".join(notes))
    H = pd.DataFrame(series).sort_index()
    tcodes = [t["code"] for t in spec["targets"]]
    mcodes = [m["code"] for m in spec["macro"]]
    hist_end = min(H[c].last_valid_index() for c in mcodes)          # every driver observed
    t_end = min(H[c].last_valid_index() for c in tcodes)
    hist_end = min(hist_end, t_end)
    os.makedirs(os.path.join(folder, "scenarios"), exist_ok=True)
    T = H[tcodes].loc[:hist_end]
    T.insert(0, "date", [str(q) for q in T.index])
    T.to_csv(os.path.join(folder, "targets.csv"), index=False)
    M = H[mcodes].ffill(limit=1)                                    # a missing last value (late release): carried
    tr = {m["code"]: m.get("transform", "dlog") for m in spec["macro"]}
    paths = scenario_paths(M, spec.get("scenarios"), mcodes, tr, hist_end, int(spec.get("horizon", 12)))
    for nm, P in paths.items():
        P = P.copy()
        P.insert(0, "date", [str(q) for q in P.index])
        P.to_csv(os.path.join(folder, "scenarios", f"{nm}.csv"), index=False)
    md = pd.DataFrame([{"mnemonic": m["code"], "description": m["description"], "geography": m.get("geography", "Canada"),
                        "theme": m.get("theme", "Other"), "transform": tr[m["code"]], "expected_sign": int(m.get("sign", 0))}
                       for m in spec["macro"]])
    md.to_csv(os.path.join(folder, "macro_metadata.csv"), index=False)
    pd.DataFrame(src).to_csv(os.path.join(folder, "sources.csv"), index=False)
    cfg = write_config(spec, name, folder, paths, hist_end, log)
    log(f"\n{len(tcodes)} targets and {len(mcodes)} drivers, {H.index[0]} to {hist_end}; "
        f"{len(paths)} scenarios to {hist_end + int(spec.get('horizon', 12))}")
    log(f"Files in {folder}; config {cfg}")
    return cfg


def write_config(spec, name, folder, paths, hist_end, log=print):
    """config/<name>.yaml through the set-up tool (tools/new_data_setup.py): dates from the data, the literature
    concepts mapped to these series, the example's comments kept; plus a targets block for run_targets.py."""
    sys.path.insert(0, os.path.join(HERE, "tools"))
    import new_data_setup as setup
    tcodes = [t["code"] for t in spec["targets"]]
    files = {k: os.path.join(folder, "scenarios", f"{k}.csv") for k in paths}
    P = setup.propose(os.path.join(folder, "targets.csv"), files, os.path.join(folder, "macro_metadata.csv"), None,
                      None, tcodes[0], "Canada", name, log=lambda m: None)
    P["dates"]["dummies"] = list(spec.get("dummy_quarters") or [])
    cfg, lit = setup.write_config(P, name, {"target.units": setup.yq("$ millions"),
                                            "target.display_scale": "1000", "target.display_units": setup.yq("$ billions"),
                                            "features.allowed_geographies":
                                                setup.yq(sorted({m.get("geography", "Canada") for m in spec["macro"]}))},
                                  fast=True, log=log)
    block = {"targets": {"run": tcodes, "labels": {t["code"]: t.get("description", t["code"]) for t in spec["targets"]}}}
    with open(cfg, "a", encoding="utf-8") as f:
        f.write("\n# Every target in data/" + name + "/targets.csv (python run_targets.py --config " +
                os.path.relpath(cfg, HERE).replace(os.sep, "/") + " --list)\n" + yaml.safe_dump(block, sort_keys=False))
    with open(cfg, encoding="utf-8") as f:
        txt = f.read()
    with open(cfg, "w", encoding="utf-8") as f:
        f.write("# Data: Statistics Canada (Statistics Canada Open Licence), downloaded by tools/fetch_statcan.py.\n"
                "# The scenarios are illustrative rule-based paths (config/statcan_series.yaml), not forecasts.\n" + txt)
    return cfg


def describe(pid, raw):
    m = metadata(pid, raw, refresh=True)
    print(f"{pid}: {m.get('cubeTitleEn')}  ({m.get('cubeStartDate')} to {m.get('cubeEndDate')})")
    for d in sorted(m["dimension"], key=lambda d: d["dimensionPositionId"]):
        names = [x["memberNameEn"] for x in d["member"]]
        print(f"\n  {d['dimensionNameEn']} ({len(names)} members)")
        for x in names[:60]:
            print(f"    - {x}")
        if len(names) > 60:
            print("    ...")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join(HERE, "config", "statcan_series.yaml"))
    ap.add_argument("--describe", metavar="TABLE", help="print a table's dimensions and members, e.g. 36100104")
    ap.add_argument("--refresh", action="store_true", help="download again instead of using data/<name>/raw/")
    a = ap.parse_args()
    if a.describe:
        describe(re.sub(r"\D", "", a.describe)[:8], os.path.join(HERE, "data", "statcan", "raw"))
        return
    build(a.spec, a.refresh)


if __name__ == "__main__":
    main()
