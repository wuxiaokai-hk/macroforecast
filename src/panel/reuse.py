"""Reusing earlier panel results when nothing they depend on has changed.

A panel run pools the chosen banks, so its results depend on every one of them: it is reused only when the
data of every chosen bank, the macro and metadata files, the literature, the settings and the code are all
unchanged. A bank's own full single-series run (panel.per_bank_full) depends only on that bank, so it is reused
bank by bank.

Each run writes its fingerprint (one hash per component) next to its results; the next run compares and says
what changed. Settings that do not change results (run name, folders, trace, docs) are left out.
"""
import copy
import hashlib
import json
import os

from ..utils.cache import SRC, _code_only, hash_frame, hash_obj

IGNORE = {"run_name", "output_dir", "out_dir", "config_file", "trace", "docs", "cache", "save"}


def _file_hash(path, memo={}):
    st = os.stat(path)
    key = (path, st.st_mtime, st.st_size)
    if key not in memo:
        with open(path, "rb") as f:
            memo[key] = hashlib.sha256(f.read()).hexdigest()
    return memo[key]


def _canon(o):
    """Settings with every file path replaced by the hash of the file's content (so moving a file changes nothing,
    editing it does)."""
    if isinstance(o, dict):
        return {k: _canon(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    if isinstance(o, str) and len(o) < 1000 and os.path.isfile(o):
        return "file:" + _file_hash(o)[:16]
    return o


def code_hash():
    """Hash of every .py file under src/ (comments and docstrings ignored; this file left out, as it does not
    change results)."""
    m = hashlib.sha256()
    for root, _, files in sorted(os.walk(SRC)):
        if "__pycache__" in root:
            continue
        for fn in sorted(files):
            if fn.endswith(".py") and not (fn == "reuse.py" and root.endswith("panel")):
                p = os.path.join(root, fn)
                with open(p, encoding="utf-8") as f:
                    m.update(os.path.relpath(p, SRC).encode() + _code_only(f.read()).encode())
    return m.hexdigest()


def _settings(cfg):
    """The settings that can change results, with the bank file and events file left out (their content enters
    bank by bank instead, so editing a bank that is not chosen changes nothing)."""
    c = {k: v for k, v in copy.deepcopy(cfg).items() if k not in IGNORE}
    d = c.get("data") or {}
    if d.get("bank"):
        d["bank"] = {k: v for k, v in d["bank"].items() if k != "file"}
    if d.get("events"):
        d["events"] = {k: v for k, v in d["events"].items() if k != "file"}
    from .run import PANEL_DEFAULTS
    p = dict(copy.deepcopy(PANEL_DEFAULTS), **(c.get("panel") or {}))     # defaults filled in, as the run does
    for k in ("per_bank_full", "banks", "evaluation"):     # evaluation is redone from the saved errors
        p.pop(k, None)
    c["panel"] = p
    return c


def _bank_data(levels, events):
    return {b: {"data": hash_frame(s), "events": hash_obj({str(q): a for q, a in (events.get(b) or {}).items()})}
            for b, s in sorted(levels.items())}


def panel_components(cfg, levels, events):
    """Fingerprint components of a panel run: one entry per chosen bank, plus settings, files and code."""
    s = _canon(_settings(cfg))
    comps = {f"bank {b}": hash_obj(v) for b, v in _bank_data(levels, events).items()}
    comps["macro and metadata files"] = hash_obj(s.get("data"))
    comps["literature"] = hash_obj(s.get("literature"))
    comps["settings"] = hash_obj({k: v for k, v in s.items() if k not in ("data", "literature")})
    comps["code"] = code_hash()
    return comps


def bank_components(cfg, bank, series, events):
    """Fingerprint components of one bank's own full single-series run."""
    s = _canon(_settings(cfg))
    s.pop("panel", None)
    return {"bank data": hash_frame(series),
            "events": hash_obj({str(q): a for q, a in (events or {}).items()}),
            "macro and metadata files": hash_obj(s.get("data")),
            "literature": hash_obj(s.get("literature")),
            "settings": hash_obj({k: v for k, v in s.items() if k not in ("data", "literature")}),
            "code": code_hash()}


def compare(new, path, need_oos=False):
    """(reusable, reason) against the fingerprint saved at `path`."""
    if not os.path.exists(path):
        return False, "no earlier run"
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    oc = old.get("components") or {}
    if need_oos and not old.get("oos"):
        return False, "earlier run had no out-of-sample test"
    changed = [k for k in new if oc.get(k) != new[k]] + [k for k in oc if k not in new]
    if changed:
        added = [k[5:] for k in new if k.startswith("bank ") and k not in oc]
        removed = [k[5:] for k in oc if k.startswith("bank ") and k not in new]
        bits = []
        if added:
            bits.append("added " + ", ".join(added))
        if removed:
            bits.append("removed " + ", ".join(removed))
        other = [k for k in changed if not (k.startswith("bank ") and (k[5:] in added or k[5:] in removed))]
        if other:
            bits.append("changed: " + ", ".join(other))
        return False, "; ".join(bits)
    return True, f"unchanged since {old.get('time', 'the last run')}"


def save(path, components, oos, extra=None):
    import datetime as dt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"time": dt.datetime.now().isoformat(timespec="seconds"), "oos": bool(oos),
                   "components": components, **(extra or {})}, f, indent=1)


def panel_path(out_dir):
    return os.path.join(out_dir, "panel", "fitted", "fingerprint.json")


def bank_path(out_dir, bank):
    return os.path.join(out_dir, "banks", str(bank).replace(" ", "_"), "bank_fingerprint.json")


def own_run_banks(cfg, available):
    """Banks that get their own full single-series run: panel.per_bank_full = true (all) or a list of names."""
    v = (cfg.get("panel") or {}).get("per_bank_full")
    if v is True or str(v).lower() == "true":
        return list(available)
    if isinstance(v, (list, tuple)):
        return [b for b in available if b in {str(x) for x in v}]
    return []


def status(cfg, oos=False):
    """What a run would reuse, bank by bank: a DataFrame for the app (and the log)."""
    import pandas as pd
    from .data import bank_events, load_long
    levels = load_long(cfg)
    events = {b: bank_events(cfg, b) for b in levels}
    out = cfg["out_dir"]
    ok, why = compare(panel_components(cfg, levels, events), panel_path(out), need_oos=oos)
    own = set(own_run_banks(cfg, levels))
    rows = []
    for b in levels:
        r = {"bank": b, "panel": "reuse" if ok else "re-estimate", "panel reason": why}
        if b in own:
            ok_b, why_b = compare(bank_components(cfg, b, levels[b], events[b]), bank_path(out, b), need_oos=oos)
            r.update({"own run": "reuse" if ok_b else "run", "own run reason": why_b})
        else:
            r.update({"own run": "not requested", "own run reason": ""})
        rows.append(r)
    return pd.DataFrame(rows)
