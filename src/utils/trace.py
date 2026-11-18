"""Detailed run trace: records every call to a function in the project while the pipeline runs.

Nothing in the pipeline has to be marked up. At start-up `install()` wraps
  - every function and method defined under src/ (and the names the entry script imported from it);
  - the entry script's own functions (for example run() in run_pipeline.py);
  - functions defined inside other functions, and lambdas, when they are passed to a traced call
    (for example the work a cached step does);
  - a list of library routines that do the actual numerical work (statsmodels OLS and ADF, the
    scikit-learn Elastic Net, scipy clustering, numpy least squares, pandas file reading). These are
    shown as "library" calls; what happens inside them is not traced.
So a function you add or change later is traced automatically on the next run.

For each call it records:
  - the function, its file and line, its docstring and its source code as it was at run time;
  - which call it came from (the call tree), how long it took, and the log messages it printed;
  - a summary of every argument (type, shape, column names, value) and of the return value;
  - the local variables at the end of the call (what a debugger shows), with small table previews;
  - a preview of every table it returned, with the full table saved as CSV (optional);
  - for regressions, the coefficient table (coef, s.e., t, p);
  - the error, traceback and local variables if it failed.

Functions called many times from the same place (one regression per candidate, one fit per backtest
origin) are recorded in detail the first `detail_calls` times; the rest are listed with their label and
time only. Calls whose first text argument differs (a step name such as 'screen_first', a folder) count as
different calls, up to `distinct_labels` of them, so each pipeline step gets its own entry.

Output, in outputs/<run_name>/trace/<date_time>/ (one folder per run; the newest `keep_runs` are kept):
  trace.json   everything above (read by tools/trace_to_html.py)
  trace.log    the same as an indented text log
  tables/      full CSVs of returned tables
  trace.html   the clickable page (written at the end of the run)
"""
import datetime as dt
import functools
import importlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SRC)
THIS = os.path.abspath(__file__)

# Library routines traced by default ("module:attribute" or "module:Class.method").
LIBRARY = [
    "pandas:read_csv", "pandas:read_excel",
    "numpy.linalg:lstsq", "numpy.linalg:pinv", "numpy.linalg:inv",
    "statsmodels.regression.linear_model:RegressionModel.fit",
    "statsmodels.tsa.stattools:adfuller",
    "statsmodels.stats.diagnostic:acorr_ljungbox", "statsmodels.stats.diagnostic:het_breuschpagan",
    "statsmodels.stats.diagnostic:breaks_cusumolsresid",
    "statsmodels.stats.stattools:durbin_watson", "statsmodels.stats.stattools:jarque_bera",
    "sklearn.linear_model._coordinate_descent:ElasticNet.fit",
    "scipy.cluster.hierarchy:linkage", "scipy.cluster.hierarchy:fcluster", "scipy.spatial.distance:squareform",
]


# --------------------------------------------------------------------------- summaries of values
def _cell(v):
    """Make a table cell JSON-safe (rounded floats, NaN -> None, numpy scalars -> Python).

    Args:
        v: value.

    Returns:
        JSON-serialisable value.
    """
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(f"{v:.6g}")
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (int, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))[:200]
    return str(v)[:200]


def _is_fit_result(v):
    """True for a statsmodels results object (has params, tvalues and bse).

    Args:
        v: any object.

    Returns:
        bool.
    """
    return hasattr(v, "params") and hasattr(v, "tvalues") and hasattr(v, "bse")


def summarize(v):
    """One-line description of any value: type, size, and a short look inside.

    Args:
        v: any value.
    """
    try:
        if isinstance(v, pd.DataFrame):
            cols = [str(c) for c in v.columns[:6]]
            idx = f"; index {v.index[0]} … {v.index[-1]}" if len(v) and not isinstance(v.index, pd.RangeIndex) else ""
            return {"type": "DataFrame", "size": f"{v.shape[0]} × {v.shape[1]}",
                    "detail": "columns: " + ", ".join(cols) + (" …" if v.shape[1] > 6 else "") + idx}
        if isinstance(v, pd.Series):
            idx = f"{v.index[0]} … {v.index[-1]}" if len(v) else ""
            d = f"name {v.name!s}; index {idx}"
            if len(v) and pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
                d += f"; mean {v.mean():.4g}, min {v.min():.4g}, max {v.max():.4g}"
            return {"type": "Series", "size": str(len(v)), "detail": d}
        if isinstance(v, pd.Index):
            return {"type": type(v).__name__, "size": str(len(v)),
                    "detail": (f"{v[0]} … {v[-1]}" if len(v) else "")}
        if isinstance(v, np.ndarray):
            d = ""
            if v.size and np.issubdtype(v.dtype, np.number):
                d = f"min {np.nanmin(v):.4g}, max {np.nanmax(v):.4g}" if v.size > 1 else f"value {v.ravel()[0]:.6g}"
            elif v.size and v.dtype == bool:
                d = f"{int(v.sum())} True"
            return {"type": "ndarray", "size": " × ".join(map(str, v.shape)), "detail": f"{v.dtype}; {d}"}
        if isinstance(v, dict):
            keys = list(v)[:8]
            return {"type": "dict", "size": f"{len(v)} keys",
                    "detail": ", ".join(map(str, keys)) + (" …" if len(v) > 8 else "")}
        if isinstance(v, (list, tuple, set)):
            v2 = list(v)
            inner = ", ".join(summarize(x)["type"] if not isinstance(x, (str, int, float)) else repr(x)
                              for x in v2[:6])
            return {"type": type(v).__name__, "size": f"{len(v2)} items", "detail": inner[:240] + (" …" if len(v2) > 6 else "")}
        if isinstance(v, str):
            return {"type": "str", "size": "", "detail": repr(v[:200]) + ("…" if len(v) > 200 else "")}
        if isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)) or v is None:
            return {"type": type(v).__name__, "size": "", "detail": str(_cell(v))}
        if isinstance(v, pd.Period):
            return {"type": "Period", "size": "", "detail": str(v)}
        if _is_fit_result(v):                                        # statsmodels results
            m = getattr(v, "model", None)
            names = list(getattr(m, "exog_names", []) or [])
            return {"type": "regression results", "size": f"{int(getattr(v, 'nobs', 0))} obs",
                    "detail": f"{getattr(m, 'endog_names', 'y')} ~ {' + '.join(names[:8])}{' …' if len(names) > 8 else ''}; "
                              f"R² {getattr(v, 'rsquared', float('nan')):.3f}"}
        if hasattr(v, "exog_names") and hasattr(v, "endog"):          # statsmodels model before fitting
            names = list(v.exog_names or [])
            return {"type": type(v).__name__, "size": f"{len(v.endog)} obs",
                    "detail": f"{getattr(v, 'endog_names', 'y')} ~ {' + '.join(names[:8])}{' …' if len(names) > 8 else ''}"}
        if hasattr(v, "coef_") and hasattr(v, "get_params"):         # fitted scikit-learn model
            c = np.ravel(v.coef_)
            p = v.get_params()
            return {"type": type(v).__name__, "size": f"{len(c)} coefs",
                    "detail": f"{int((c != 0).sum())} non-zero; alpha {p.get('alpha', '')}, l1_ratio {p.get('l1_ratio', '')}"}
        if hasattr(v, "get_params"):
            p = v.get_params()
            return {"type": type(v).__name__, "size": "", "detail": ", ".join(f"{k}={p[k]}" for k in list(p)[:6])}
        if callable(v):
            return {"type": "function", "size": "", "detail": getattr(v, "__qualname__", str(v))[:120]}
        return {"type": type(v).__name__, "size": "", "detail": ""}
    except Exception as e:                                   # never let tracing break the run
        return {"type": type(v).__name__, "size": "", "detail": f"(could not summarise: {e})"}


def table_json(df, rows):
    """First rows of a table in the trace's JSON format.

    Args:
        df: DataFrame or Series. rows: rows to keep.

    Returns:
        dict with columns, rows, n and n_cols.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name if df.name is not None else "value")
    show_index = not isinstance(df.index, pd.RangeIndex)
    d = df.iloc[:rows, :40].copy()
    if show_index:
        nm = str(d.index.name) if d.index.name is not None else "index"
        d.index.name = nm if nm not in map(str, d.columns) else f"{nm} (index)"
        d = d.reset_index()
        d.columns = [str(c) for c in d.columns]
    return {"columns": [str(c) for c in d.columns], "rows": [[_cell(x) for x in r] for r in d.itertuples(index=False)],
            "n": int(len(df)), "n_cols": int(df.shape[1])}


def coef_table(res):
    """coef / s.e. / t / p of a statsmodels results object as a DataFrame.

    Args:
        res: statsmodels results.
    """
    return pd.DataFrame({"coef": res.params, "std_err": res.bse, "t": res.tvalues, "p": res.pvalues})


def tables_in(v, label="result", depth=0):
    """(label, table) pairs for every DataFrame/Series in a return value (looks one level into dicts/tuples).

    Args:
        v: return value. label: name of v. depth: nesting level.
    """
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return [(label, v)] if len(v) else []
    if _is_fit_result(v):
        try:
            return [(f"{label} coefficients", coef_table(v))]
        except Exception:
            return []
    out = []
    if depth < 2 and isinstance(v, dict):
        for k, x in list(v.items())[:12]:
            out += tables_in(x, f"{label}['{k}']" if label != "result" else f"['{k}']", depth + 1)
    elif depth < 2 and isinstance(v, (list, tuple)) and len(v) <= 8:
        for i, x in enumerate(v):
            out += tables_in(x, f"{label}[{i}]", depth + 1)
    return out


def compact(v):
    """Very short description of an argument, for the list of repeated calls.

    Args:
        v: any value.
    """
    try:
        if isinstance(v, pd.Series):
            return f"{v.name}" if v.name is not None else f"Series[{len(v)}]"
        if isinstance(v, pd.DataFrame):
            return f"DataFrame {v.shape[0]}×{v.shape[1]}" + (f" [{', '.join(map(str, v.columns[:3]))}{'…' if v.shape[1] > 3 else ''}]" if v.shape[1] <= 6 else "")
        if isinstance(v, np.ndarray):
            return f"array {'×'.join(map(str, v.shape))}"
        if isinstance(v, (str, int, float, np.integer, np.floating, bool)) or v is None:
            return repr(_cell(v))[:60]
        if isinstance(v, (list, tuple)):
            return f"[{', '.join(compact(x) for x in list(v)[:3])}{'…' if len(v) > 3 else ''}]"
        if hasattr(v, "exog_names") and hasattr(v, "exog"):
            n = list(v.exog_names or [])
            return f"OLS on {len(n)} regressors, last '{n[-1] if n else ''}' (mean {np.mean(v.exog[:, -1]):.4g})"
        if isinstance(v, dict):
            return f"dict({len(v)})"
        if hasattr(v, "get_params"):
            p = v.get_params()
            return f"{type(v).__name__}(alpha={_cell(p.get('alpha'))})" if "alpha" in p else type(v).__name__
        return type(v).__name__
    except Exception:
        return "?"


def _project_function(x):
    """A plain function or lambda written in this project (not a library, not this file).

    Args:
        x: any object.
    """
    if not inspect.isfunction(x) or getattr(x, "__traced__", False):
        return False
    f = os.path.abspath(x.__code__.co_filename)
    return f.startswith(ROOT + os.sep) and f != THIS and "site-packages" not in f


# --------------------------------------------------------------------------- the tracer
class Tracer:
    """Records every call of the project's functions (arguments, results, local variables, logs, time)
    and writes trace.json, trace.log and trace.html.
    """
    def __init__(self, folder, title, preview_rows=20, save_tables=True, detail_calls=1, distinct_labels=8,
                 max_table_cells=200_000, local_variables=True, local_preview_rows=8, max_repeats_listed=500):
        """Create a tracer writing to folder; the settings come from the trace: section of the config.

        Args:
            folder: output folder. title: page title. preview_rows: table rows shown. save_tables: save
            returned tables as CSV. detail_calls: repeated calls recorded in full. distinct_labels: calls with
            different text arguments shown separately. max_table_cells: larger tables are not saved.
            local_variables: record local variables. local_preview_rows: rows per local table.
            max_repeats_listed: repeated calls listed per node.
        """
        self.folder = folder
        self.preview_rows, self.save_tables, self.detail_calls = preview_rows, save_tables, max(1, int(detail_calls))
        self.distinct_labels = max(1, int(distinct_labels))
        self.max_table_cells = int(max_table_cells)
        self.local_variables, self.local_preview_rows = bool(local_variables), int(local_preview_rows)
        self.max_repeats_listed = int(max_repeats_listed)
        self.t0 = time.perf_counter()
        self.started = dt.datetime.now()
        self.nodes = [{"id": 0, "parent": None, "fn": title, "calls": 1, "time": 0.0, "start": 0.0, "args": [],
                       "logs": [], "children": [], "depth": 0}]
        self.stack, self.suppress = [0], 0
        self.seen = {}              # (parent id, function key) -> {first string argument: [node ids]}
        self.functions = {}         # function key -> file, line, doc, source
        self.wrapped = {}           # id(original) -> wrapper (long-lived functions only)
        self.examples = {}          # function key -> first call seen only inside a folded call (for docs/)
        self.story = None           # the data story (src/reporting/story.py), shown as its own view
        self.paused = False         # while True, calls are not recorded (used while the story is built)
        self.example_keys = set()   # functions that already have a recorded call
        self.n_wrapped, self.n_library = 0, 0
        os.makedirs(os.path.join(folder, "tables"), exist_ok=True)

    # ---- registration
    def _register(self, fn, library=False, display=None):
        """Add a function to the trace's function list (file, line, docstring, source).

        Args:
            fn: function. library: True for a library routine. display: name to show.

        Returns:
            the function's key.
        """
        key = ("lib:" if library else "") + f"{fn.__module__}.{fn.__qualname__}"
        if key not in self.functions:
            file, src, line = "", "", 0
            if not library:
                try:
                    lines, line = inspect.getsourcelines(fn)
                    src = "".join(lines)
                except (OSError, TypeError):
                    pass
                file = inspect.getsourcefile(fn) or ""
                file = os.path.relpath(file, ROOT) if file else ""
                if fn.__module__ == "__main__":
                    file = os.path.relpath(os.path.abspath(sys.argv[0]), ROOT)
            doc = inspect.getdoc(fn) or ""
            if library and len(doc) > 1500:
                doc = doc[:1500].rsplit("\n", 1)[0] + "\n…"
            self.functions[key] = {"name": display or fn.__qualname__,
                                   "module": file.replace(".py", "").replace(os.sep, ".") if fn.__module__ == "__main__" else fn.__module__,
                                   "file": file, "line": line, "doc": doc, "source": src, "library": library}
        return key

    def wrap(self, fn, library=False, display=None, cache=True):
        """Wrap a function so each call is recorded as a node of the call tree.

        Args:
            fn: function. library: library routine. display: name to show. cache: reuse an existing wrapper.

        Returns:
            the wrapper.
        """
        if cache and id(fn) in self.wrapped:
            return self.wrapped[id(fn)]
        key = self._register(fn, library, display)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        tr = self

        @functools.wraps(fn)
        def wrapper(*a, **k):
            if tr.paused:
                return fn(*a, **k)
            if tr.suppress:
                if library or key in tr.example_keys:
                    return fn(*a, **k)
                return tr._example_call(key, sig, fn, a, k)
            parent = tr.stack[-1]
            lab = next((x for x in list(a) + list(k.values()) if isinstance(x, str)), None)
            ent = tr.seen.setdefault((parent, key), {"labels": {}, "last": None})
            ids = ent["labels"].get(lab)
            detailed = (len(ids) < tr.detail_calls) if ids is not None else (len(ent["labels"]) < tr.distinct_labels)
            if ids is None and not detailed:
                ent["overflow"] = True
            if not detailed:                                       # repeated call: listed, timed
                node = tr.nodes[ent["last"]]
                node["calls"] += 1
                tr.suppress += 1
                t = time.perf_counter()
                err = None
                try:
                    return fn(*a, **k)
                except Exception as e:
                    node["errors"] = node.get("errors", 0) + 1
                    err = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    el = time.perf_counter() - t
                    node["time"] += el
                    tr.suppress -= 1
                    rep = node.setdefault("repeats", [])
                    if len(rep) < tr.max_repeats_listed:
                        desc = ", ".join(compact(x) for x in list(a)[:4] + list(k.values())[:2]
                                         if not (isinstance(x, dict) and "sample" in x) and type(x).__name__ != "Context")
                        rep.append([node["calls"], lab, round(t - tr.t0, 4), round(el, 5), err, desc[:200]])
            nid = len(tr.nodes)
            node = {"id": nid, "parent": parent, "fn": key, "calls": 1, "time": 0.0,
                    "start": round(time.perf_counter() - tr.t0, 4), "depth": tr.nodes[parent]["depth"] + 1,
                    "args": tr._safe(tr._args, sig, a, k) or [], "logs": [], "children": []}
            if lab is not None:
                node["label"] = lab[:60]
            tr.nodes.append(node)
            tr.example_keys.add(key)
            tr.nodes[parent]["children"].append(nid)
            ent["labels"].setdefault(lab, []).append(nid)
            ent["last"] = nid
            if library:                                            # leaf: nothing inside is traced
                tr.suppress += 1
                t = time.perf_counter()
                try:
                    r = fn(*a, **k)
                except Exception as e:
                    node["error"] = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    node["time"] += time.perf_counter() - t
                    tr.suppress -= 1
                tr._safe(tr._result, node, r)
                return r
            # pass inner functions / lambdas through the tracer too
            a = tuple(tr.wrap(x, cache=False) if _project_function(x) else x for x in a)
            k = {n: (tr.wrap(x, cache=False) if _project_function(x) else x) for n, x in k.items()}
            tr.stack.append(nid)
            box = {}
            if tr.local_variables:
                def grab(frame, event, arg):                     # first Python frame entered = fn itself
                    if event == "call":
                        box["f"] = frame
                        sys.setprofile(None)
                sys.setprofile(grab)
            t = time.perf_counter()
            try:
                r = fn(*a, **k)
            except Exception as e:
                node["error"] = f"{type(e).__name__}: {e}"
                node["traceback"] = traceback.format_exc()
                raise
            finally:
                node["time"] += time.perf_counter() - t
                if tr.local_variables:
                    sys.setprofile(None)
                tr.stack.pop()
                if "f" in box:
                    tr._safe(tr._locals, node, box.pop("f"))
            tr._safe(tr._result, node, r)
            return r

        wrapper.__traced__ = True
        if cache:
            self.wrapped[id(fn)] = wrapper
        if library:
            self.n_library += 1
        else:
            self.n_wrapped += 1
        return wrapper

    @staticmethod
    def _safe(f, *a):
        """Call f, returning None instead of raising (tracing must never break a run).

        Args:
            f: function. a: its arguments.
        """
        try:
            return f(*a)
        except Exception:                           # tracing must never break the run
            return None

    def _args(self, sig, a, k):
        """Summaries of a call's arguments (name, type, size, detail) and their table previews.

        Args:
            sig: inspect.Signature or None. a, k: positional and keyword arguments.

        Returns:
            list of {name, type, size, detail}.
        """
        try:
            b = sig.bind_partial(*a, **k) if sig else None
            items = list(b.arguments.items()) if b else [(f"arg{i}", v) for i, v in enumerate(a)] + list(k.items())
        except TypeError:
            items = [(f"arg{i}", v) for i, v in enumerate(a)] + list(k.items())
        out = []
        for name, v in items:
            if name == "cfg" and isinstance(v, dict):
                out.append({"name": name, "type": "dict", "size": f"{len(v)} keys", "detail": "the run configuration"})
                continue
            if name in ("args", "kwargs") and isinstance(v, (tuple, dict)):
                vals = v.items() if isinstance(v, dict) else enumerate(v)
                out += [{"name": f"{name}[{i}]", **summarize(x)} for i, x in list(vals)[:8]]
                continue
            out.append({"name": name, **summarize(v)})
        return out

    def _example_call(self, key, sig, fn, a, k):
        """Run a call inside a folded (repeated) call and keep its arguments and result as the function's example.

        Only the first such call of each function is kept, so functions that only ever run inside repeated calls
        (e.g. evaluate inside cached steps) still get an example in the function reference.

        Args:
            key: function key. sig: its signature. fn: the original function. a, k: the call's arguments.

        Returns:
            the function's return value.
        """
        self.example_keys.add(key)
        ex = {"args": self._safe(self._args, sig, a, k) or [], "start": round(time.perf_counter() - self.t0, 4)}
        t = time.perf_counter()
        r = fn(*a, **k)
        ex["time"] = round(time.perf_counter() - t, 4)
        tmp = {"id": 0, "fn": key}
        keep, self.save_tables = self.save_tables, False
        try:
            self._safe(self._result, tmp, r)
        finally:
            self.save_tables = keep
        ex["result"] = tmp.get("result")
        ex["tables"] = [{k2: v for k2, v in t2.items() if k2 != "file"} for t2 in tmp.get("tables", [])[:2]]
        self.examples[key] = ex
        return r

    def _locals(self, node, frame):
        """Variables in the function's frame when it returned (or failed).

        Args:
            node: call node. frame: the function's frame.
        """
        loc = frame.f_locals
        argnames = {a["name"] for a in node["args"]}
        code = frame.f_code
        params = set(code.co_varnames[:code.co_argcount + code.co_kwonlyargcount])
        out, prevs = [], []
        for name, v in loc.items():
            if name.startswith("__") or inspect.ismodule(v) or (inspect.isfunction(v) and name in ("tr", "grab")):
                continue
            if name in params and name in argnames and not isinstance(v, (int, float, str)):
                continue                                         # unchanged input, already listed
            s = summarize(v)
            out.append({"name": name, **s})
            if isinstance(v, (pd.DataFrame, pd.Series)) and len(v) and len(prevs) < 6:
                prevs.append({"label": name, **table_json(v, self.local_preview_rows)})
            elif _is_fit_result(v) and len(prevs) < 6:
                try:
                    prevs.append({"label": f"{name} coefficients", **table_json(coef_table(v), 30)})
                except Exception:
                    pass
        node["locals"] = out
        if prevs:
            node["local_tables"] = prevs

    def _result(self, node, r):
        """Store a summary and table previews of a call's return value on its node.

        Args:
            node: call node. r: return value.
        """
        node["result"] = summarize(r)
        prev = []
        for label, t in tables_in(r)[:8]:
            entry = {"label": label, **table_json(t, self.preview_rows)}
            if self.save_tables and t.size > self.max_table_cells:
                entry["file"] = None
                entry["not_saved"] = f"not saved: {t.size:,} cells is above trace.max_table_cells"
            elif self.save_tables:
                safe = "".join(c if c.isalnum() else "_" for c in f"{node['id']:04d}_{node['fn'].split('.')[-1]}_{label}")
                path = os.path.join(self.folder, "tables", safe.strip("_")[:120] + ".csv")
                try:
                    t.to_csv(path)
                    entry["file"] = os.path.relpath(path, ROOT)
                except Exception:
                    pass
            prev.append(entry)
        if prev:
            node["tables"] = prev

    def log(self, msg):
        """Attach a progress message to the call currently running.

        Args:
            msg: text.
        """
        if not self.suppress:
            self.nodes[self.stack[-1]]["logs"].append(str(msg))

    # ---- install into the code
    def install(self, extra_namespaces=(), library=None):
        """Wrap every function/method defined in src/ in every module that refers to it, the entry script's
        own functions, and the listed library routines.

        Args:
            extra_namespaces: namespaces of the entry script (globals()). library: library routines to trace
            (None = defaults).
        """
        mods = [m for n, m in list(sys.modules.items()) if (n == "src" or n.startswith("src.")) and m is not None]
        mods = [m for m in mods if m.__name__ != __name__]
        originals = {}
        for m in mods:
            for name, obj in list(vars(m).items()):
                if inspect.isfunction(obj) and (obj.__module__ or "").startswith("src.") and obj.__module__ != __name__:
                    originals[id(obj)] = obj
                elif inspect.isclass(obj) and (obj.__module__ or "").startswith("src.") and obj.__module__ == m.__name__:
                    for an, av in list(vars(obj).items()):
                        if inspect.isfunction(av) and (not an.startswith("__") or an == "__init__"):
                            setattr(obj, an, self.wrap(av))
        log_fn = next((o for o in originals.values() if o.__name__ == "log" and o.__module__ == "src.utils.io"), None)
        tr = self

        def traced_log(msg):
            log_fn(msg)
            tr.log(msg)

        # library routines: replace them where they are defined and wherever they were imported by name
        libs = {}
        for spec in (LIBRARY if library is None else library):
            try:
                modname, attr = spec.split(":")
                mod = importlib.import_module(modname)
                if "." in attr:
                    cls_name, meth = attr.split(".")
                    cls = getattr(mod, cls_name)
                    fn = vars(cls)[meth]
                    setattr(cls, meth, self.wrap(fn, library=True, display=f"{modname.split('.')[0]}: {cls_name}.{meth}"))
                else:
                    fn = getattr(mod, attr)
                    w = self.wrap(fn, library=True, display=f"{modname.split('.')[0]}: {attr}")
                    setattr(mod, attr, w)
                    libs[id(fn)] = w
            except Exception:
                continue                                         # library not installed / renamed: skip it

        spaces = [vars(m) for m in mods] + list(extra_namespaces)
        for ns in spaces:
            for name, obj in list(ns.items()):
                if id(obj) in originals:
                    o = originals[id(obj)]
                    ns[name] = traced_log if o is log_fn else self.wrap(o)
                elif id(obj) in libs:
                    ns[name] = libs[id(obj)]
        for ns in extra_namespaces:                              # the entry script's own functions
            for name, obj in list(ns.items()):
                if inspect.isfunction(obj) and obj.__module__ == "__main__" and name != "main" \
                        and not getattr(obj, "__traced__", False):
                    ns[name] = self.wrap(obj)
        return self

    # ---- write
    def _tidy(self):
        """Fold groups that ran out of distinct labels (helpers called with many different strings) into one
        entry, then renumber the nodes that are still in the tree."""
        for ent in self.seen.values():
            if not ent.get("overflow"):
                continue
            ids = sorted(i for v in ent["labels"].values() for i in v)
            keep = self.nodes[ids[0]]
            rep = keep.setdefault("repeats", [])
            for i in ids[1:]:
                n = self.nodes[i]
                keep["calls"] += n["calls"]
                keep["time"] += n["time"]
                if len(rep) < self.max_repeats_listed:
                    rep.append([None, n.get("label"), n["start"], round(n["time"], 5), n.get("error"),
                                ", ".join(f"{x['name']}={x['detail'][:40] if x['type'] == 'str' else x['type']}" for x in n["args"][:3])])
                rep.extend(n.get("repeats", [])[:max(0, self.max_repeats_listed - len(rep))])
                kids = self.nodes[n["parent"]]["children"]
                if i in kids:
                    kids.remove(i)
            rep.sort(key=lambda r: r[2])
            for j, r in enumerate(rep):
                r[0] = j + 2
        new, order = {}, []

        def walk(i):
            new[i] = len(order)
            order.append(i)
            for c in self.nodes[i]["children"]:
                walk(c)
        walk(0)
        kept_fns = {self.nodes[i]["fn"] for i in order}
        for n in self.nodes:                         # calls folded away: keep one example per function (docs/)
            if n["id"] not in new and n["fn"] not in kept_fns and n["fn"] not in self.examples and "result" in n:
                self.examples[n["fn"]] = {"args": n.get("args", []), "result": n.get("result"), "time": n.get("time"),
                                          "start": n.get("start"), "label": n.get("label", ""),
                                          "tables": [{k: v for k, v in t.items() if k != "file"}
                                                     for t in n.get("tables", [])[:2]]}
        nodes = []
        for i in order:
            n = dict(self.nodes[i])
            n["id"] = new[i]
            n["parent"] = new.get(n["parent"]) if n["parent"] is not None else None
            n["children"] = [new[c] for c in n["children"]]
            nodes.append(n)
        self.nodes = nodes

    def write(self, meta=None):
        """Write trace.json, trace.log and trace.html and remove old trace folders.

        Args:
            meta: extra run information (command, config).

        Returns:
            path of trace.html.
        """
        self.nodes[0]["time"] = time.perf_counter() - self.t0
        self._tidy()
        for n in self.nodes:
            n["time"] = round(n["time"], 4)
        git = ""
        try:
            git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True,
                                   timeout=5).stdout.strip()
            git += " (with uncommitted changes)" if git and dirty else ""
        except Exception:
            pass
        info = {"started": self.started.isoformat(timespec="seconds"), "seconds": round(self.nodes[0]["time"], 2),
                "command": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]), "python": platform.python_version(),
                "platform": platform.platform(terse=True), "git": git, "functions_traced": self.n_wrapped,
                "library_functions_traced": self.n_library, "calls_recorded": len(self.nodes) - 1,
                "calls_total": int(sum(n["calls"] for n in self.nodes[1:])), **(meta or {})}
        with open(os.path.join(self.folder, "trace.json"), "w", encoding="utf-8") as f:
            json.dump({"info": info, "nodes": self.nodes, "functions": self.functions, "examples": self.examples,
                       "story": self.story}, f, default=_cell)
        with open(os.path.join(self.folder, "trace.log"), "w", encoding="utf-8") as f:
            f.write(f"# {info['command']}  started {info['started']}  ({info['seconds']}s)\n")
            self._text(0, f)
        return info

    def _text(self, nid, f):
        """Write one call (and its children) to the plain-text trace.log.

        Args:
            nid: node id. f: open file.
        """
        n = self.nodes[nid]
        if nid:
            fn = self.functions[n["fn"]]
            pad = "  " * (n["depth"] - 1)
            rep = f" ×{n['calls']}" if n["calls"] > 1 else ""
            lab = f" '{n['label'][:40]}'" if n.get("label") else ""
            where = "library" if fn.get("library") else f"{fn['file']}:{fn['line']}"
            f.write(f"{pad}{fn['name']}(){lab}{rep}  [{n['time']:.3f}s]  {where}\n")
            for a in n["args"]:
                f.write(f"{pad}    in  {a['name']}: {a['type']} {a['size']} {a['detail']}\n")
            for a in n.get("locals", []):
                f.write(f"{pad}    var {a['name']}: {a['type']} {a['size']} {a['detail']}\n")
            if "result" in n:
                r = n["result"]
                f.write(f"{pad}    out {r['type']} {r['size']} {r['detail']}\n")
            for t in n.get("tables", []):
                f.write(f"{pad}    table {t['label']} {t['n']}×{t['n_cols']}" + (f" -> {t['file']}" if t.get("file") else "") + "\n")
            for m in n["logs"]:
                f.write(f"{pad}    log {m.strip()}\n")
            if n.get("error"):
                f.write(f"{pad}    ERROR {n['error']}\n")
        for c in n["children"]:
            self._text(c, f)


_ACTIVE = None


def add_story(builder, log=print):
    """Build the data story with the running tracer paused (its own calls are not part of the run) and keep it
    for trace.html. Does nothing when tracing is off; a failure is reported and never stops the run.

    Args:
        builder: function returning {"steps": [...], ...} (see src/reporting/story.py). log: message function.
    """
    tr = _ACTIVE
    if tr is None:
        return
    tr.paused = True
    try:
        tr.story = builder()
    except Exception as e:
        import traceback
        log(f"Data story not built: {type(e).__name__}: {e}")
        tr.story = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(), "steps": []}
    finally:
        tr.paused = False


def start(cfg, title, namespace, enabled=True):
    """Create and install a tracer from the config's `trace:` section. Returns None when tracing is off.

    Args:
        title: page title. namespace: the script's globals(). enabled: False = no trace.
    """
    t = cfg.get("trace", {})
    if not enabled or not t.get("enabled", True):
        return None
    base = os.path.join(cfg["out_dir"], t.get("folder", "trace"))
    os.makedirs(base, exist_ok=True)
    old = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d[:2] == "20")
    for d in old[:max(0, len(old) - int(t.get("keep_runs", 10)) + 1)]:      # keep the newest keep_runs traces
        shutil.rmtree(os.path.join(base, d), ignore_errors=True)
    folder = os.path.join(base, dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    tr = Tracer(folder, title, t.get("preview_rows", 20), t.get("save_tables", True), t.get("detail_calls", 1),
                t.get("distinct_labels", 8), t.get("max_table_cells", 200_000), t.get("local_variables", True),
                t.get("local_preview_rows", 8))
    lib = t.get("library_functions")
    global _ACTIVE
    _ACTIVE = tr
    return tr.install([namespace], library=None if lib in (None, "default") else (lib or []))


def finish(tracer, meta=None, log=print):
    """Write trace.json / trace.log and build trace.html.

    Args:
        tracer: from start (None does nothing).
    """
    if tracer is None:
        return
    info = tracer.write(meta)
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from trace_to_html import build
        page = build(os.path.join(tracer.folder, "trace.json"))
        log(f"Trace: {info['calls_total']} calls of {info['functions_traced']} project functions and "
            f"{info['library_functions_traced']} library routines -> {page}")
    except Exception as e:
        log(f"Trace written to {tracer.folder} (HTML not built: {e})")
