"""Function reference ("vignette") built automatically from the code, its docstrings and the latest runs.

For every module, class, function and method in src/, the run scripts, tools/ and report/, the reference shows:
  - the signature and the docstring (summary, arguments, returns);
  - the config settings it reads (cfg["sample"]["horizon"] ...);
  - what it calls and what calls it (from the code, plus the calls actually seen in the last runs);
  - a real example from the latest run trace: the arguments it received and what it returned (with a table
    preview), and a warning if the function has changed since that run;
  - its source code, and the date it last changed.

It is rebuilt automatically:
  - at the end of run_pipeline.py and run_oos.py (only if the code or the traces changed; takes about a second);
  - by `python tools/make_docs.py --watch`, which rebuilds whenever a .py file is saved;
  - by a git pre-commit hook (`python tools/make_docs.py --install-hook`).
`python tools/make_docs.py --check` exits with code 1 if the reference is out of date (for CI).

Outputs (folder docs/ by default): reference.html (searchable page), REFERENCE.md (plain text, for GitHub),
changes.json (when each function was added or last changed) and examples.json (examples kept between runs).
"""
import ast
import datetime as dt
import glob
import hashlib
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCES = ["run_pipeline.py", "run_oos.py", "src/**/*.py", "tools/*.py", "report/*.py"]
GROUPS = [("Run scripts", ("run_",)), ("Data and configuration", ("src/utils/io", "src/utils/config", "src/utils/env")),
          ("Features", ("src/features",)), ("Selection approaches", ("src/pipelines", "src/selection")),
          ("Estimation and evaluation", ("src/modelling",)), ("Competing models", ("src/models",)),
          ("Out-of-sample test", ("src/validation",)), ("Reporting", ("src/reporting",)),
          ("Cache, trace and docs", ("src/utils",)), ("Tools", ("tools/",)), ("Presentation deck", ("report/",))]
# Meaning of the argument names used the same way throughout the project; shown (marked "standard") when a
# docstring does not describe the argument itself. Keys "prefix:name" apply only to files starting with prefix.
GLOSSARY = {
    "cfg": "the run configuration (from load_config).",
    "ctx": "the Context: loaded data, organic growth, scenarios, metadata and the estimation window.",
    "P": "the model's data panel (target y, drivers, D_ dummies), from model.panel().",
    "mdf": "macro DataFrame (quarters x macro mnemonics): the history or one scenario.",
    "F": "candidate features (DataFrame, one column per feature).",
    "H": "forecast horizon in quarters.",
    "origin": "last known quarter of a forecast (Period).",
    "origins": "positions of the forecast origins in ctx.idx.",
    "start": "first quarter of the estimation window (Period).",
    "end": "last quarter of the estimation window (Period).",
    "end_pos": "position of the last quarter the step may use.",
    "model": "a model following the model interface.",
    "log": "function used to print progress messages.",
    "meta": "variable metadata indexed by mnemonic.",
    "by_h": "errors by model and horizon (backtest output).",
    "horizons": "comparison horizons, e.g. [4, 8, 12].",
    "features": "driver feature names (\"code|transform|Llag\").",
    "name": "display name.",
    "note": "free text shown in the model list.",
    "family": "model family used in the comparison.",
    "deflator": "mnemonic of the price index.",
    "cache": "the StepCache.",
    "path": "file to write or read.",
    "folder": "output folder.",
    "approach": "selection approach: screen_first or en_first.",
    "approaches": "selection approaches to run.",
    "sel": "selection results per approach.",
    "level": "actual target series (Series).",
    "forms": "chosen transforms per series.",
    "report/:slide": "python-pptx slide.",
    "report/:shape": "python-pptx shape.",
    "report/:prs": "python-pptx Presentation.",
    "report/:x": "left position (inches).",
    "report/:y": "top position (inches).",
    "report/:w": "width (inches).",
    "report/:h": "height (inches).",
    "report/:items": "parsed markers: (star, colour, payload).",
}
SECTION = re.compile(r"^(Args|Arguments|Parameters|Returns|Return|Yields|Raises|Example|Examples|Notes?):\s*$")


# ----------------------------------------------------------------------------------------------- parsing
def source_files(root=ROOT):
    """All Python files documented, relative to the project folder, in a stable order.

    Args:
        root: project folder.

    Returns:
        list of relative paths (no __init__.py without content, no outputs or cache).
    """
    out = []
    for pat in SOURCES:
        for p in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel.endswith("__init__.py") and os.path.getsize(p) == 0:
                continue
            if rel not in out:
                out.append(rel)
    return out


def module_name(rel):
    """Dotted module name of a file ("src/models/zoo.py" -> "src.models.zoo"; scripts keep their stem).

    Args:
        rel: path relative to the project folder.

    Returns:
        str.
    """
    return rel[:-3].replace("/", ".").removesuffix(".__init__")


def parse_docstring(doc, params=()):
    """Split a Google-style docstring into summary, description and sections.

    Arguments may be written one per line or several on a line ("P: panel. start, end: window.");
    only names that are real parameters start a new argument.

    Args:
        doc: docstring text (or None). params: the function's parameter names.

    Returns:
        dict with summary, body (list of paragraphs), args (name -> text), returns, raises, other sections.
    """
    out = {"summary": "", "body": [], "args": {}, "returns": "", "raises": "", "sections": {}}
    if not doc:
        return out
    lines = doc.strip("\n").split("\n")
    sec, buf, cur = None, [], {}
    for ln in lines + [None]:                           # None flushes the last block
        m = SECTION.match(ln.strip()) if ln is not None else None
        if m or ln is None:
            cur[sec] = (cur.get(sec, "") + "\n" + "\n".join(buf)).strip()
            sec, buf = (m.group(1) if m else None), []
            continue
        buf.append(ln.strip())
    intro = cur.pop(None, "")
    paras = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", intro) if p.strip()]
    if paras:
        out["summary"], out["body"] = paras[0], paras[1:]
    argtxt = (cur.pop("Args", "") + "\n" + cur.pop("Arguments", "") + "\n" + cur.pop("Parameters", "")).strip()
    if argtxt:
        out["args"] = _split_args(argtxt, set(params))
    out["returns"] = (cur.pop("Returns", "") or cur.pop("Return", "") or cur.pop("Yields", "")).replace("\n", " ")
    out["raises"] = cur.pop("Raises", "").replace("\n", " ")
    out["sections"] = {k: v for k, v in cur.items() if v}
    return out


def standard_meaning(name, rel):
    """Meaning of a standard argument name from GLOSSARY (file-specific entries first).

    Args:
        name: argument name. rel: file the function is in.

    Returns:
        str or "".
    """
    for k, v in GLOSSARY.items():
        if ":" in k and rel.startswith(k.split(":")[0]) and k.split(":")[1] == name:
            return v
    return GLOSSARY.get(name, "")


def _split_args(text, params):
    """Assign the text of an Args section to parameter names.

    Args:
        text: section text. params: valid parameter names.

    Returns:
        dict name -> description.
    """
    text = " ".join(text.split())
    pat = re.compile(r"(?:^|(?<=\. )|(?<=; ))([A-Za-z_]\w*(?:, [A-Za-z_]\w*)*): ")
    hits = [m for m in pat.finditer(text) if all(n in params or not params for n in m.group(1).split(", "))]
    out = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        desc = text[m.end():end].strip()
        for n in m.group(1).split(", "):
            out[n] = desc
    if not hits and text:
        out["(all)"] = text
    return out


def code_hash(src):
    """Hash of a function's code without its docstring, comments or formatting, so editing the documentation
    does not mark run examples as out of date.

    Args:
        src: source text of one function or class.

    Returns:
        str (12 hex characters).
    """
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(src))
        for n in ast.walk(tree):
            b = getattr(n, "body", None)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and b and isinstance(b[0], ast.Expr) \
                    and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str):
                n.body = b[1:] or [ast.Pass()]
        text = ast.unparse(tree)
    except SyntaxError:
        text = src
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _signature(node):
    """Readable signature of a def, e.g. "fit(P, start, end, fast=False)".

    Args:
        node: ast.FunctionDef or ast.ClassDef (a class shows its __init__ arguments).

    Returns:
        (signature text, list of parameter dicts with name and default).
    """
    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:
            return f"{node.name}()", []
        s, p = _signature(init)
        return node.name + s[len("__init__"):], p
    a = node.args
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    params, parts = [], []
    for arg, d in zip(pos, defaults):
        if arg.arg in ("self", "cls"):
            continue
        dv = ast.unparse(d) if d is not None else None
        params.append({"name": arg.arg, "default": dv})
        parts.append(arg.arg + (f"={dv}" if dv is not None else ""))
    if a.vararg:
        params.append({"name": "*" + a.vararg.arg, "default": None})
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        dv = ast.unparse(d) if d is not None else None
        params.append({"name": arg.arg, "default": dv})
        parts.append(arg.arg + (f"={dv}" if dv is not None else ""))
    if a.kwarg:
        params.append({"name": "**" + a.kwarg.arg, "default": None})
        parts.append("**" + a.kwarg.arg)
    return f"{node.name}({', '.join(parts)})", params


def _config_keys(src):
    """Config settings read by a piece of code, e.g. sample.horizon (direct and through an alias like s = cfg["sample"]).

    Args:
        src: source code.

    Returns:
        sorted list of dotted keys.
    """
    keys = set()
    for m in re.finditer(r"cfg((?:\[[\"']\w+[\"']\])+)", src):
        keys.add(".".join(re.findall(r"[\"'](\w+)[\"']", m.group(1))))
    for m in re.finditer(r"(\w+) = (?:ctx\.)?cfg((?:\[[\"']\w+[\"']\])+)", src):
        alias, base = m.group(1), ".".join(re.findall(r"[\"'](\w+)[\"']", m.group(2)))
        sub = re.findall(rf"\b{alias}(?:\[[\"'](\w+)[\"']\]|\.get\([\"'](\w+)[\"'])", src)
        keys |= {f"{base}.{a or b}" for a, b in sub} or {base}
    for m in re.finditer(r"cfg\.get\([\"'](\w+)[\"']", src):
        keys.add(m.group(1))
    prefixes = {k.rsplit(".", 1)[0] for k in keys if "." in k}
    return sorted(k for k in keys if k not in prefixes)


def _imports(tree, mod):
    """Names imported by a module, mapped to the dotted object or module they refer to.

    Args:
        tree: module AST. mod: dotted module name (for relative imports).

    Returns:
        dict local name -> dotted target.
    """
    out = {}
    pkg = mod.split(".")
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            base = pkg[:len(pkg) - n.level] if n.level else []
            target = ".".join(base + (n.module.split(".") if n.module else []))
            for a in n.names:
                out[a.asname or a.name] = f"{target}.{a.name}" if target else a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name
    return out


def scan(root=ROOT):
    """Parse every documented file into a list of items (modules, classes, functions, methods).

    Args:
        root: project folder.

    Returns:
        (items dict id -> item, modules list in file order).
    """
    items, modules = {}, []
    for rel in source_files(root):
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            src = f.read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            modules.append({"id": module_name(rel), "file": rel, "doc": f"(could not parse: {e})", "items": []})
            continue
        lines = src.splitlines()
        mod = module_name(rel)
        imp = _imports(tree, mod)
        m = {"id": mod, "file": rel, "doc": ast.get_docstring(tree) or "", "items": [], "imports": imp}
        modules.append(m)

        def visit(node, prefix, cls=None):
            for n in node.body:
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                q = prefix + n.name
                code = "\n".join(lines[(n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1:n.end_lineno])
                sig, params = _signature(n)
                kind = "class" if isinstance(n, ast.ClassDef) else ("method" if cls else "function")
                it = {"id": f"{mod}.{q}", "module": mod, "file": rel, "qualname": q, "name": n.name, "kind": kind,
                      "line": n.lineno, "end": n.end_lineno, "signature": sig, "params": params,
                      "doc_raw": ast.get_docstring(n) or "", "source": code, "class": cls,
                      "hash": hashlib.sha1(code.encode()).hexdigest()[:12], "code_hash": code_hash(code), "config": _config_keys(code),
                      "calls_static": [] if isinstance(n, ast.ClassDef) else _calls(n, mod, imp, cls, prefix)}
                it["doc"] = parse_docstring(it["doc_raw"], [p["name"].lstrip("*") for p in params])
                if kind == "class":
                    it["methods"] = [f"{mod}.{q}.{b.name}" for b in n.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    it["bases"] = [ast.unparse(b) for b in n.bases]
                items[it["id"]] = it
                m["items"].append(it["id"])
                if isinstance(n, ast.ClassDef):
                    visit(n, q + ".", cls=q)
        visit(tree, "")
    for it in items.values():                               # a class's arguments are described on __init__
        init = items.get(it["id"] + ".__init__") if it["kind"] == "class" else None
        if init:
            it["doc"]["args"] = {**init["doc"]["args"], **it["doc"]["args"]}
    _resolve_calls(items, modules)
    return items, modules


def _calls(node, mod, imp, cls, prefix):
    """Raw call targets in a function body (resolved later against the project's items).

    Args:
        node: function AST. mod: module name. imp: imports of the module. cls: enclosing class or None.
        prefix: qualname prefix of the enclosing class.

    Returns:
        list of candidate dotted names (and "*.name" for method calls on other objects).
    """
    out = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name):
            out.append(imp.get(f.id, f"{mod}.{f.id}"))
        elif isinstance(f, ast.Attribute):
            v = f.value
            if isinstance(v, ast.Name) and v.id == "self" and cls:
                out.append(f"{mod}.{cls}.{f.attr}")
            elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super" and cls:
                out.append(f"super:{mod}.{cls}.{f.attr}")
            elif isinstance(v, ast.Name) and v.id in imp:
                out.append(f"{imp[v.id]}.{f.attr}")
            else:
                out.append(f"*.{f.attr}")
    return out


def _resolve_calls(items, modules):
    """Turn raw call targets into links between documented items (calls / called_by).

    Args:
        items: from scan (updated in place). modules: from scan.
    """
    by_name = {}
    for k, it in items.items():
        by_name.setdefault(it["name"], []).append(k)
    for it in items.values():
        calls = []
        for c in it["calls_static"]:
            if c.startswith("super:"):
                c = c[6:]
                cls = items.get(c.rsplit(".", 1)[0])
                for b in (cls or {}).get("bases", []):
                    base = items.get(_lookup(items, it["module"], b, modules))
                    if base and f"{base['id']}.{c.rsplit('.', 1)[1]}" in items:
                        calls.append(f"{base['id']}.{c.rsplit('.', 1)[1]}")
                continue
            if c.startswith("*."):
                name = c[2:]
                cands = [k for k in by_name.get(name, []) if items[k]["kind"] == "method"]
                if len(cands) == 1 and name not in ("fit", "forecast", "panel", "spec", "drivers", "run", "write", "log"):
                    calls.append(cands[0])
                continue
            t = c if c in items else _lookup(items, it["module"], c, modules)
            if t:
                calls.append(t)
        it["calls"] = sorted(set(calls) - {it["id"]})
    for it in items.values():
        it.setdefault("called_by", [])
    for it in items.values():
        for c in it["calls"]:
            items[c]["called_by"].append(it["id"])


def _lookup(items, mod, name, modules):
    """Find the documented item a dotted or bare name refers to (classes resolve to themselves).

    Args:
        items: all items. mod: module where the name is used. name: dotted or bare name. modules: all modules.

    Returns:
        item id or None.
    """
    if name in items:
        return name
    if f"{mod}.{name}" in items:
        return f"{mod}.{name}"
    tail = name.split(".")
    for k in range(1, len(tail)):                          # "src.models.ardl.ARDL" style partial matches
        cand = ".".join(tail[k:])
        hits = [i for i in items if i.endswith("." + cand)]
        if len(hits) == 1:
            return hits[0]
    return None


# ----------------------------------------------------------------------------------------------- examples
def trace_files(root=ROOT):
    """All trace.json files under outputs/, newest first.

    Args:
        root: project folder.

    Returns:
        list of paths.
    """
    fs = glob.glob(os.path.join(root, "outputs", "*", "trace*", "*", "trace.json"))
    return sorted(fs, key=os.path.getmtime, reverse=True)


def collect_examples(items, store, root=ROOT, max_traces=12):
    """Update the example store with the first call of each function in the newest traces.

    A function keeps its latest example; examples of removed functions are dropped. Each example records
    the run, the date and the source hash at the time, so a changed function is flagged.

    Args:
        items: from scan. store: dict id -> example (updated). root: project folder. max_traces: traces read.

    Returns:
        (store, run summaries for the overview).
    """
    seen, runs = set(), []
    for path in trace_files(root)[:max_traces]:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        info = d.get("info", {})
        script = os.path.splitext(os.path.basename(info.get("command", "")))[0]
        rel_run = os.path.relpath(os.path.dirname(path), root).replace(os.sep, "/")
        runs.append({"path": rel_run, "started": info.get("started", ""), "command": info.get("command", ""),
                     "seconds": info.get("seconds"), "nodes": d.get("nodes", []), "functions": d.get("functions", {})})
        extra = [{**v, "fn": k, "calls": 1} for k, v in (d.get("examples") or {}).items()]
        for n in d.get("nodes", []) + extra:
            key = n.get("fn", "")
            if key.startswith("__main__.") and script:
                key = script + key[len("__main__"):]
            if key not in items or key in seen:
                continue
            seen.add(key)
            fsrc = d.get("functions", {}).get(n.get("fn"), {}).get("source", "")
            store[key] = {"run": rel_run, "started": info.get("started", ""), "command": info.get("command", ""),
                          "args": n.get("args", []), "result": n.get("result"), "tables": n.get("tables", [])[:2],
                          "calls": n.get("calls", 1), "time": n.get("time"), "label": n.get("label", ""),
                          "source_hash": code_hash(fsrc) if fsrc else ""}
    for k in [k for k in store if k not in items]:
        del store[k]
    return store, runs


def observed_edges(runs, items):
    """Caller -> callee pairs seen in the traces (catches calls the code alone cannot resolve, like model.fit).

    Args:
        runs: from collect_examples. items: from scan.

    Returns:
        set of (caller id, callee id).
    """
    edges = set()
    for r in runs:
        script = os.path.splitext(os.path.basename(r["command"]))[0]
        fix = (lambda k: script + k[len("__main__"):] if k.startswith("__main__.") else k)
        nodes = r["nodes"]
        for n in nodes:
            p = n.get("parent")
            if p is None or p >= len(nodes) or p == 0:
                continue
            a, b = fix(nodes[p]["fn"]), fix(n["fn"])
            if a in items and b in items and a != b:
                edges.add((a, b))
    return edges


# ----------------------------------------------------------------------------------------------- change log
def update_changes(items, path):
    """Record when each function was added or last changed (by hash of its code, ignoring docstrings).

    Args:
        items: from scan. path: changes.json.

    Returns:
        dict id -> {hash, added, changed}, plus "_log": recent events.
    """
    today = dt.date.today().isoformat()
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        st = {}
    first = not st
    log = st.pop("_log", [])
    for k, it in items.items():
        old = st.get(k)
        if old is None:
            st[k] = {"hash": it["code_hash"], "added": "" if first else today, "changed": "" if first else today}
            if not first:
                log.append({"date": today, "id": k, "what": "added"})
        elif old["hash"] != it["code_hash"]:                 # code changed (documentation-only edits do not count)
            old.update(hash=it["code_hash"], changed=today)
            log.append({"date": today, "id": k, "what": "changed"})
    for k in [k for k in st if k not in items]:
        log.append({"date": today, "id": k, "what": "removed"})
        del st[k]
    st["_log"] = log[-300:]
    return st


# ----------------------------------------------------------------------------------------------- build
def fingerprint(root=ROOT):
    """Hash of everything the reference depends on: the source files, this generator and the newest traces.

    Args:
        root: project folder.

    Returns:
        str.
    """
    h = hashlib.sha1()
    for rel in source_files(root):
        with open(os.path.join(root, rel), "rb") as f:
            h.update(rel.encode() + f.read())
    for p in trace_files(root)[:12]:
        h.update(p.encode() + str(os.path.getmtime(p)).encode())
    return h.hexdigest()


def is_stale(out_dir=None, root=ROOT):
    """True if the reference does not match the current code and traces.

    Args:
        out_dir: docs folder (default docs/). root: project folder.

    Returns:
        bool.
    """
    out_dir = out_dir or os.path.join(root, "docs")
    try:
        with open(os.path.join(out_dir, ".fingerprint"), encoding="utf-8") as f:
            return f.read().strip() != fingerprint(root)
    except OSError:
        return True


def build(out_dir=None, root=ROOT, examples=True):
    """Build docs/reference.html and docs/REFERENCE.md from the code and the latest traces.

    Args:
        out_dir: output folder (default docs/). root: project folder. examples: include examples from traces.

    Returns:
        dict with paths and counts (items, documented, with examples).
    """
    out_dir = out_dir or os.path.join(root, "docs")
    os.makedirs(out_dir, exist_ok=True)
    items, modules = scan(root)
    store_path = os.path.join(out_dir, "examples.json")
    try:
        with open(store_path, encoding="utf-8") as f:
            store = json.load(f)
    except Exception:
        store = {}
    runs = []
    if examples:
        store, runs = collect_examples(items, store, root)
        for a, b in observed_edges(runs, items):
            if b not in items[a]["calls"]:
                items[a]["calls"].append(b)
                items[a].setdefault("calls_seen", []).append(b)
                items[b]["called_by"].append(a)
    for it in items.values():
        it["called_by"] = sorted(set(it["called_by"]))
        it["calls"] = sorted(set(it["calls"]))
    changes = update_changes(items, os.path.join(out_dir, "changes.json"))
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=0)
    with open(os.path.join(out_dir, "changes.json"), "w", encoding="utf-8") as f:
        json.dump(changes, f, indent=1)
    flow = _flow(runs, items)
    page = render_html(items, modules, store if examples else {}, changes, flow, runs)
    with open(os.path.join(out_dir, "reference.html"), "w", encoding="utf-8") as f:
        f.write(page)
    with open(os.path.join(out_dir, "REFERENCE.md"), "w", encoding="utf-8") as f:
        f.write(render_md(items, modules, changes))
    with open(os.path.join(out_dir, ".fingerprint"), "w", encoding="utf-8") as f:
        f.write(fingerprint(root))
    n = len(items)
    doc = sum(bool(i["doc_raw"]) for i in items.values())
    ex = sum(k in store for k in items)
    return {"html": os.path.join(out_dir, "reference.html"), "md": os.path.join(out_dir, "REFERENCE.md"),
            "items": n, "documented": doc, "examples": ex}


def refresh(cfg=None, log=print, root=ROOT):
    """Rebuild the reference if the code or the traces changed (called at the end of every run).

    Args:
        cfg: configuration (uses its docs: section; None = defaults). log: message function. root: project folder.

    Returns:
        path of reference.html, or None if docs are switched off or nothing changed.
    """
    d = (cfg or {}).get("docs", {}) or {}
    if d.get("enabled", True) is False:
        return None
    out = d.get("folder") or "docs"
    out = out if os.path.isabs(out) else os.path.join(root, out)
    if not is_stale(out, root):
        return None
    try:
        r = build(out, root, examples=d.get("examples", True))
    except Exception as e:                                   # documentation must never break a run
        log(f"Function reference not updated: {type(e).__name__}: {e}")
        return None
    log(f"Function reference updated: {r['items']} items, {r['documented']} with docstrings, "
        f"{r['examples']} with run examples -> {r['html']}")
    return r["html"]


def _flow(runs, items):
    """Top-level steps of the newest pipeline and out-of-sample runs, for the overview.

    Args:
        runs: from collect_examples. items: from scan.

    Returns:
        list of {command, started, steps: [(id, calls, seconds)]}.
    """
    out, done = [], set()
    for r in runs:
        cmd = os.path.basename(r["command"])
        if cmd in done or not r["nodes"]:
            continue
        done.add(cmd)
        script = os.path.splitext(cmd)[0]
        nodes = r["nodes"]
        top = nodes[0]["children"][:1]
        steps, agg = [], {}
        for t in top:
            for c in nodes[t]["children"]:
                k = nodes[c]["fn"]
                k = script + k[len("__main__"):] if k.startswith("__main__.") else k
                if k not in items:
                    continue
                if k not in agg:
                    agg[k] = [0, 0.0]
                    steps.append(k)
                agg[k][0] += nodes[c].get("calls", 1)
                agg[k][1] += nodes[c].get("time", 0) or 0
        out.append({"command": cmd, "started": r["started"], "seconds": r["seconds"],
                    "entry": (script + nodes[top[0]]["fn"][len("__main__"):]) if top else "",
                    "steps": [(k, agg[k][0], agg[k][1]) for k in steps]})
    return out


# ----------------------------------------------------------------------------------------------- rendering
def _e(s):
    """HTML-escape a value (None -> empty).

    Args:
        s: any value.

    Returns:
        str.
    """
    return html.escape("" if s is None else str(s))


def _anchor(i):
    """HTML id of an item.

    Args:
        i: item id.

    Returns:
        str.
    """
    return "f-" + re.sub(r"[^\w]", "-", i)


def _link(i, items, short=True):
    """Link to an item's card.

    Args:
        i: item id. items: all items. short: show the qualified name only.

    Returns:
        HTML.
    """
    it = items.get(i)
    label = it["qualname"] if (it and short) else i
    return f'<a class="ref" href="#{_anchor(i)}" title="{_e(i)}">{_e(label)}</a>'


def _inline(s, items, names):
    """Escape text and turn `code` spans into code; known function names become links.

    Args:
        s: text. items: all items. names: bare name -> id for unique names.

    Returns:
        HTML.
    """
    s = _e(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def _table(t, rows=6):
    """Small HTML preview of a trace table.

    Args:
        t: table dict (label, columns, rows, n, n_cols). rows: rows shown.

    Returns:
        HTML.
    """
    cols = t.get("columns", [])[:8]
    head = "".join(f"<th>{_e(c)}</th>" for c in cols)
    body = ""
    for r in t.get("rows", [])[:rows]:
        body += "<tr>" + "".join(f"<td>{_e(_fmt(v))}</td>" for v in r[:8]) + "</tr>"
    more = []
    if t.get("n", 0) > rows:
        more.append(f"{t['n']:,} rows")
    if t.get("n_cols", 0) > 8:
        more.append(f"{t['n_cols']} columns")
    cap = f"<div class='cap'>{_e(t.get('label', ''))}" + (f" · first {rows} of {', '.join(more)}" if more else "") + "</div>"
    return f"{cap}<div class='tw'><table><tr>{head}</tr>{body}</table></div>"


def _fmt(v):
    """Short display of a cell value.

    Args:
        v: value.

    Returns:
        str.
    """
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def _example_html(ex, it):
    """The "example from the last run" block of a card.

    Args:
        ex: stored example or None. it: item.

    Returns:
        HTML.
    """
    if not ex:
        return ("<div class='noex'>No example yet: not called while a run trace was recording. It may run before the "
                "trace starts (config loading), only in tools/ or the deck, only when an optional package is installed, "
                "or its step was loaded from the cache.</div>")
    stale = ex.get("source_hash") and ex["source_hash"] != it["code_hash"]
    when = ex.get("started", "").replace("T", " ")[:16]
    head = (f"<div class='exh'>Example from <b>{_e(os.path.basename(ex.get('command', '')))}</b>, {_e(when)}"
            + (f" · called {ex['calls']}×" if ex.get("calls", 1) > 1 else "")
            + (f" · {ex['time']:.2f} s" if isinstance(ex.get("time"), (int, float)) and ex["time"] >= 0.01 else "")
            + (f" · step <code>{_e(ex['label'][:60])}</code>" if ex.get("label") else "") + "</div>")
    warn = "<div class='warn'>The function has changed since this run; the example may be out of date.</div>" if stale else ""
    rows = "".join(f"<tr><td><code>{_e(a.get('name'))}</code></td><td>{_e(a.get('type'))}</td><td>{_e(a.get('size'))}</td>"
                   f"<td>{_e(a.get('detail'))}</td></tr>" for a in ex.get("args", []) if a.get("name") != "self")
    r = ex.get("result") or {}
    res = (f"<tr class='res'><td><b>returned</b></td><td>{_e(r.get('type'))}</td><td>{_e(r.get('size'))}</td>"
           f"<td>{_e(r.get('detail'))}</td></tr>") if r else ""
    tabs = "".join(_table(t) for t in ex.get("tables", []))
    return (f"<div class='ex'>{head}{warn}<table class='kv'><tr><th>input</th><th>type</th><th>size</th><th>value</th></tr>"
            f"{rows}{res}</table>{tabs}</div>")


def _card(it, items, ex, ch, names):
    """HTML card of one function, method or class.

    Args:
        it: item. items: all items. ex: example or None. ch: change record. names: unique names.

    Returns:
        HTML.
    """
    d = it["doc"]
    kind = it["kind"]
    badges = f"<span class='b {kind}'>{kind}</span>"
    if not it["doc_raw"]:
        badges += "<span class='b nodoc'>no docstring</span>"
    if ch and ch.get("changed"):
        badges += f"<span class='b chg'>changed {_e(ch['changed'])}</span>"
    if it["qualname"].split(".")[-1].startswith("_") and not it["qualname"].endswith("__init__"):
        badges += "<span class='b priv'>internal</span>"
    body = ""
    if d["summary"]:
        body += f"<p class='sum'>{_inline(d['summary'], items, names)}</p>"
    for p in d["body"]:
        body += f"<p>{_inline(p, items, names)}</p>"
    if it["params"]:
        rows = ""
        for p in it["params"]:
            nm = p["name"].lstrip("*")
            desc = _inline(d["args"].get(nm, ""), items, names)
            if not desc:
                g = standard_meaning(nm, it["file"])
                desc = f"{_e(g)} <span class='std'>standard</span>" if g else ""
            rows += (f"<tr><td><code>{_e(p['name'])}</code></td><td>{'<code>' + _e(p['default']) + '</code>' if p['default'] is not None else ''}</td>"
                     f"<td>{desc}</td></tr>")
        if d["args"].get("(all)"):
            rows += f"<tr><td colspan=3>{_inline(d['args']['(all)'], items, names)}</td></tr>"
        body += f"<h4>Arguments</h4><table class='args'><tr><th>name</th><th>default</th><th>meaning</th></tr>{rows}</table>"
    if d["returns"]:
        body += f"<h4>Returns</h4><p>{_inline(d['returns'], items, names)}</p>"
    if d["raises"]:
        body += f"<h4>Raises</h4><p>{_inline(d['raises'], items, names)}</p>"
    for k, v in d["sections"].items():
        body += f"<h4>{_e(k)}</h4><pre class='plain'>{_e(v)}</pre>"
    if kind == "class" and it.get("methods"):
        body += "<h4>Methods</h4><p class='links'>" + " ".join(_link(m, items) for m in it["methods"] if m in items) + "</p>"
    if it["config"]:
        body += "<h4>Config settings read</h4><p class='links'>" + " ".join(f"<code>{_e(k)}</code>" for k in it["config"]) + "</p>"
    if it["calls"]:
        seen = set(it.get("calls_seen", []))
        body += ("<h4>Calls</h4><p class='links'>" + " ".join(_link(c, items) + ("<sup title='seen in a run'>run</sup>" if c in seen else "")
                                                         for c in it["calls"]) + "</p>")
    if it["called_by"]:
        body += "<h4>Called by</h4><p class='links'>" + " ".join(_link(c, items) for c in it["called_by"]) + "</p>"
    if kind != "class":
        body += "<h4>Example from the last run</h4>" + _example_html(ex, it)
    lines = it["end"] - it["line"] + 1
    body += (f"<details><summary>Source · {_e(it['file'])} lines {it['line']}–{it['end']} ({lines} lines)</summary>"
             f"<pre class='src'>{_e(it['source'])}</pre></details>")
    search = " ".join([it["id"], d["summary"], " ".join(it["config"])]).lower()
    return (f"<section class='card' id='{_anchor(it['id'])}' data-s='{_e(search)}'>"
            f"<div class='ch'><h3>{_e(it['qualname'])}</h3>{badges}<span class='loc'>{_e(it['file'])}:{it['line']}</span></div>"
            f"<pre class='sig'>{_e(it['signature'])}</pre>{body}</section>")


def _group(rel):
    """Sidebar group of a file.

    Args:
        rel: relative path.

    Returns:
        group title.
    """
    for title, prefixes in GROUPS:
        if rel.startswith(prefixes):
            return title
    return "Other"


def render_html(items, modules, store, changes, flow, runs):
    """The searchable reference page (self-contained HTML).

    Args:
        items, modules: from scan. store: examples. changes: from update_changes. flow: from _flow. runs: traces read.

    Returns:
        HTML text.
    """
    names = {}
    n = len(items)
    doc = sum(bool(i["doc_raw"]) for i in items.values())
    ex = sum(k in store for k in items if items[k]["kind"] != "class")
    nfun = sum(items[k]["kind"] != "class" for k in items)
    groups = {}
    for m in modules:
        groups.setdefault(_group(m["file"]), []).append(m)
    order = [g for g, _ in GROUPS] + ["Other"]
    nav, main = [], []
    for g in order:
        if g not in groups:
            continue
        nav.append(f"<div class='ng'>{_e(g)}</div>")
        for m in groups[g]:
            mid = "m-" + re.sub(r"[^\w]", "-", m["id"])
            nav.append(f"<a class='nm' href='#{mid}' data-s='{_e(m['id'].lower())}'>{_e(m['file'])}</a>")
            sub = "".join(f"<a class='nf {items[i]['kind']}' href='#{_anchor(i)}' data-s='{_e(i.lower())}'>{_e(items[i]['qualname'])}</a>"
                          for i in m["items"])
            nav.append(f"<div class='nfs'>{sub}</div>")
            cards = "".join(_card(items[i], items, store.get(i), changes.get(i), names) for i in m["items"])
            mdoc = "".join(f"<p>{_e(p)}</p>" for p in re.split(r"\n\s*\n", m["doc"]) if p.strip()) if m["doc"] else ""
            pre = f"<pre class='plain mdoc'>{_e(m['doc'])}</pre>" if m["doc"] and ("  " in m["doc"] or "\n-" in m["doc"]) else mdoc
            main.append(f"<section class='mod' id='{mid}'><h2>{_e(m['file'])}</h2>"
                        f"<div class='mi'>module <code>{_e(m['id'])}</code> · {len(m['items'])} items</div>{pre}{cards}</section>")
    # overview
    fl = ""
    for f in flow:
        steps = "".join(f"<li>{_link(k, items)} <span class='muted'>×{c}, {t:.1f} s</span> — {_e(items[k]['doc']['summary'][:140])}</li>"
                        for k, c, t in f["steps"])
        fl += (f"<div class='flow'><h3>{_e(f['command'])} <span class='muted'>(run {_e(f['started'].replace('T', ' ')[:16])}, "
               f"{f['seconds'] or 0:.0f} s)</span></h3><p>Entry point {_link(f['entry'], items) if f['entry'] in items else ''}; "
               f"the steps it ran, in order:</p><ol>{steps}</ol></div>")
    if not fl:
        fl = "<p class='muted'>No run traces found yet: run <code>python run_pipeline.py</code> to add the run flow and examples.</p>"
    log = [e for e in changes.get("_log", []) if e["what"] != "removed" or True][-25:][::-1]
    chg = "".join(f"<tr><td>{_e(e['date'])}</td><td>{_e(e['what'])}</td><td>{_link(e['id'], items) if e['id'] in items else _e(e['id'])}</td></tr>"
                  for e in log) or "<tr><td colspan=3 class='muted'>No changes recorded since the reference was first built.</td></tr>"
    nodoc = [i for i in items if not items[i]["doc_raw"]]
    nd = (" ".join(_link(i, items) for i in nodoc)) if nodoc else "none"
    built = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    overview = f"""
<section class='mod' id='overview'><h1>macroforecast: function reference</h1>
<p class='lead'>Every module, class and function of the project, generated from the code, its docstrings and the latest
runs. This page is rebuilt automatically when the code changes (at the end of each run, by
<code>python tools/make_docs.py --watch</code>, or by the git pre-commit hook), so it cannot drift from the code.</p>
<div class='stats'><div><b>{n}</b><span>items</span></div><div><b>{doc}</b><span>with docstrings</span></div>
<div><b>{ex}/{nfun}</b><span>functions with a run example</span></div><div><b>{len(modules)}</b><span>files</span></div></div>
<h2>How to read a card</h2>
<ul><li><b>Signature and arguments</b>: from the code; meanings from the docstring.</li>
<li><b>Config settings read</b>: the config keys the function uses, e.g. <code>sample.horizon</code>.</li>
<li><b>Calls / Called by</b>: links both ways; <sup>run</sup> marks calls seen in a run but not visible in the code (like <code>model.fit</code>).</li>
<li><b>Example from the last run</b>: what the function actually received and returned, with a table preview. Steps loaded from the
cache are not called, so run with <code>--force</code> once to collect examples for everything.</li>
<li><b>The model interface</b>: every competing model has <code>panel(ctx, mdf)</code>, <code>fit(P, start, end)</code>,
<code>forecast(P, origin, H)</code>, <code>spec()</code> and <code>drivers()</code>, plus <code>name</code>, <code>family</code> and
<code>note</code>; any class with these can be added to the zoo.</li></ul>
<h2>How a run flows</h2>{fl}
<h2>Recent changes</h2><table class='args'><tr><th>date</th><th>what</th><th>item</th></tr>{chg}</table>
<h2>Without a docstring</h2><p class='links'>{nd}</p>
<p class='muted'>Built {built} from {len(runs)} run trace(s).</p></section>"""
    return TEMPLATE.replace("%NAV%", "".join(nav)).replace("%MAIN%", overview + "".join(main))


def render_md(items, modules, changes):
    """Plain-text reference (REFERENCE.md): signatures, docstrings and links; no run examples, so it only changes
    when the code changes.

    Args:
        items, modules: from scan. changes: from update_changes.

    Returns:
        Markdown text.
    """
    L = ["# Function reference", "",
         "Generated by `python tools/make_docs.py` from the code and docstrings; do not edit by hand.",
         "The searchable version with examples from the last run is `docs/reference.html`.", ""]
    for m in modules:
        L.append(f"- [{m['file']}](#{re.sub(r'[^a-z0-9-]', '', m['file'].lower().replace('/', '').replace('.', ''))})")
    for m in modules:
        L += ["", f"## {m['file']}", ""]
        if m["doc"]:
            L += [m["doc"].split("\n\n")[0], ""]
        for i in m["items"]:
            it = items[i]
            d = it["doc"]
            L += [f"### `{it['qualname']}`", "", f"```python\n{it['signature']}\n```", ""]
            if d["summary"]:
                L += [d["summary"], ""]
            L += [p + "\n" for p in d["body"]]
            if it["params"]:
                L += ["| argument | default | meaning |", "|---|---|---|"]
                for p in it["params"]:
                    desc = (d["args"].get(p["name"].lstrip("*"), "") or standard_meaning(p["name"].lstrip("*"), it["file"])).replace("|", "\\|")
                    L.append(f"| `{p['name']}` | {('`' + p['default'] + '`') if p['default'] is not None else ''} | {desc} |")
                L.append("")
            if d["returns"]:
                L += [f"**Returns:** {d['returns']}", ""]
            if it["config"]:
                L += ["**Config:** " + ", ".join(f"`{k}`" for k in it["config"]), ""]
            if it["calls"]:
                L += ["**Calls:** " + ", ".join(f"`{items[c]['qualname']}`" for c in it["calls"]), ""]
            if it["called_by"]:
                L += ["**Called by:** " + ", ".join(f"`{items[c]['qualname']}`" for c in it["called_by"]), ""]
            L += [f"<sub>{it['file']}:{it['line']}</sub>", ""]
    return "\n".join(L) + "\n"


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Function reference</title><style>
:root{--ink:#1B2A41;--navy:#0E3B4F;--teal:#1F6F8B;--gold:#E3A72F;--green:#2E9E83;--red:#C2553F;--grey:#8796A8;--bg:#F7F9FB;--card:#fff;--line:#D9E1E8;--soft:#EEF3F6;--code:#F2F5F8;--muted:#5d6b7a}
@media (prefers-color-scheme:dark){:root{--ink:#E6EDF3;--navy:#0b2633;--teal:#5FB3CF;--bg:#0f1720;--card:#16212c;--line:#2a3a4a;--soft:#1c2a37;--code:#1a2633;--muted:#9fb0c0}}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 Calibri,'Segoe UI',system-ui,sans-serif;color:var(--ink);background:var(--bg)}
nav{position:fixed;top:0;left:0;bottom:0;width:300px;overflow:auto;background:var(--navy);color:#dfe8ee;padding:14px 10px 40px}
nav h1{font:600 16px Cambria,Georgia,serif;margin:4px 6px 10px;color:#fff}nav input{width:100%;padding:7px 9px;border-radius:6px;border:0;margin-bottom:10px;font-size:13px}
.ng{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--gold);margin:14px 6px 4px}
nav a{display:block;color:#dfe8ee;text-decoration:none;padding:2px 6px;border-radius:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
nav a:hover{background:#1A4A61}.nm{font-weight:600;font-size:13px}.nfs{margin-left:10px;font-size:12px;display:none}.nfs.open{display:block}
.nf.method{padding-left:16px;color:#b9cbd6}.nf.class{color:#fff}
main{margin-left:300px;padding:24px 34px 80px;max-width:1180px}
h1{font:600 26px Cambria,Georgia,serif;margin:0 0 8px}h2{font:600 20px Cambria,Georgia,serif;margin:28px 0 8px;color:var(--teal)}
h3{font:600 16px Cambria,Georgia,serif;margin:0}h4{margin:12px 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.lead{font-size:15px;max-width:860px}.muted{color:var(--muted)}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}.stats div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;min-width:150px}
.stats b{display:block;font:600 24px Cambria,serif;color:var(--teal)}.stats span{color:var(--muted);font-size:12px}
.mod{margin-bottom:30px}.mi{color:var(--muted);font-size:12px;margin-bottom:6px}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--teal);border-radius:8px;padding:12px 16px;margin:12px 0}
.ch{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.loc{margin-left:auto;color:var(--muted);font-size:12px;font-family:Consolas,monospace}
.b{font-size:10.5px;border-radius:10px;padding:1px 8px;background:var(--soft);color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.b.class{background:var(--gold);color:#1B2A41}.b.method{background:#d8e7ee;color:#1F6F8B}.b.function{background:#d6efe7;color:#1d6b58}
.b.nodoc{background:#f6dcd6;color:#9a3b29}.b.chg{background:#fbecc8;color:#7a5a12}.b.priv{background:var(--soft)}
pre{font:12.5px/1.45 Consolas,'SFMono-Regular',monospace;background:var(--code);border-radius:6px;padding:8px 10px;overflow:auto;margin:6px 0}
pre.sig{background:transparent;padding:4px 0;color:var(--teal);font-weight:600;white-space:pre-wrap}pre.src{max-height:460px}
pre.plain{white-space:pre-wrap}.sum{font-size:14.5px;margin:4px 0}
code{font-family:Consolas,monospace;font-size:12.5px;background:var(--code);padding:0 4px;border-radius:3px}
table{border-collapse:collapse;font-size:12.5px;margin:4px 0}th,td{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600}table.args td:first-child{white-space:nowrap}
.links a.ref{display:inline-block;margin:0 6px 4px 0;padding:1px 7px;background:var(--soft);border-radius:4px;text-decoration:none;color:var(--teal);font-family:Consolas,monospace;font-size:12px}
sup{font-size:9px;color:var(--gold);margin-left:-4px;margin-right:4px}
.ex{border:1px dashed var(--line);border-radius:6px;padding:6px 10px;background:var(--bg)}.exh{font-size:12px;color:var(--muted);margin-bottom:4px}
.std{font-size:9.5px;color:var(--muted);border:1px solid var(--line);border-radius:8px;padding:0 5px;margin-left:4px}
.warn{color:var(--red);font-size:12px;margin:2px 0 4px}.noex{color:var(--muted);font-size:12.5px;font-style:italic}
tr.res td{border-top:2px solid var(--line)}.cap{font-size:11.5px;color:var(--muted);margin-top:6px}.tw{overflow:auto}
details summary{cursor:pointer;color:var(--muted);font-size:12.5px;margin-top:8px}
.flow{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;margin:10px 0}.flow ol{margin:4px 0;padding-left:22px}
.hide{display:none!important}
@media (max-width:900px){nav{position:static;width:auto;max-height:45vh}main{margin-left:0;padding:16px}}
</style></head><body>
<nav><h1>Function reference</h1><input id="q" placeholder="Search functions, settings, text…" autocomplete="off">
<a class="nm" href="#overview">Overview and run flow</a>%NAV%</nav>
<main>%MAIN%</main>
<script>
const q=document.getElementById('q');
document.querySelectorAll('nav .nm').forEach(a=>a.addEventListener('click',()=>{const n=a.nextElementSibling;if(n&&n.classList.contains('nfs'))n.classList.toggle('open')}));
q.addEventListener('input',()=>{const v=q.value.trim().toLowerCase();
 document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hide',!!v&&!c.dataset.s.includes(v)));
 document.querySelectorAll('.mod').forEach(m=>{if(m.id==='overview'){m.classList.toggle('hide',!!v);return}
   m.classList.toggle('hide',!!v&&!m.querySelector('.card:not(.hide)'))});
 document.querySelectorAll('nav .nf').forEach(a=>a.classList.toggle('hide',!!v&&!a.dataset.s.includes(v)));
 document.querySelectorAll('nav .nfs').forEach(s=>s.classList.toggle('open',!!v&&!!s.querySelector('.nf:not(.hide)')));});
</script></body></html>"""
