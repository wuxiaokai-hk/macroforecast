"""Build the function reference (docs/reference.html and docs/REFERENCE.md) from the code and the latest runs.

    python tools/make_docs.py                 build now (only if something changed; --force to always build)
    python tools/make_docs.py --watch         rebuild every time a .py file in the project is saved
    python tools/make_docs.py --check         exit code 1 if the reference is out of date (for CI)
    python tools/make_docs.py --install-hook  git pre-commit hook: rebuild and add docs/ to every commit

The reference is also rebuilt at the end of run_pipeline.py and run_oos.py (setting docs: in the config).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.utils import docs  # noqa: E402

HOOK = """#!/bin/sh
# added by tools/make_docs.py --install-hook: keep the function reference in step with the code
PY=$(command -v python3 || command -v python)
"$PY" tools/make_docs.py --quiet || exit 1
git add docs/reference.html docs/REFERENCE.md docs/changes.json docs/examples.json 2>/dev/null
exit 0
"""


def install_hook():
    """Write .git/hooks/pre-commit so every commit rebuilds the reference and includes it.

    Returns:
        path of the hook, or None if the project is not a git repository.
    """
    git = os.path.join(HERE, ".git")
    if not os.path.isdir(git):
        print("Not a git repository (no .git folder): run `git init` first.")
        return None
    path = os.path.join(git, "hooks", "pre-commit")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and "make_docs.py" not in open(path, encoding="utf-8").read():
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + HOOK.split("\n", 1)[1])
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(HOOK)
    os.chmod(path, 0o755)
    print(f"Installed {path}")
    return path


def watch(out, interval=2.0):
    """Rebuild whenever a documented file changes (polls file modification times; Ctrl+C to stop).

    Args:
        out: docs folder. interval: seconds between checks.
    """
    def stamp():
        """Modification times of all documented files and traces."""
        fs = [os.path.join(HERE, f) for f in docs.source_files(HERE)] + docs.trace_files(HERE)[:3]
        return tuple((f, os.path.getmtime(f)) for f in fs if os.path.exists(f))

    last = None
    print(f"Watching {len(docs.source_files(HERE))} files; the reference is rebuilt on every save (Ctrl+C to stop).")
    while True:
        s = stamp()
        if s != last:
            if last is not None or docs.is_stale(out, HERE):
                r = docs.build(out, HERE)
                print(f"{time.strftime('%H:%M:%S')} rebuilt: {r['items']} items -> {r['html']}", flush=True)
            last = s
        time.sleep(interval)


def main():
    """Command line entry."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "docs"), help="output folder (default docs/)")
    ap.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    ap.add_argument("--watch", action="store_true", help="rebuild on every save")
    ap.add_argument("--check", action="store_true", help="exit 1 if the reference is out of date")
    ap.add_argument("--install-hook", action="store_true", help="install the git pre-commit hook")
    ap.add_argument("--no-examples", action="store_true", help="leave out the examples from run traces")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.install_hook:
        install_hook()
        return
    if a.check:
        stale = docs.is_stale(a.out, HERE)
        print("The function reference is out of date: run python tools/make_docs.py" if stale else "Up to date.")
        sys.exit(1 if stale else 0)
    if a.watch:
        watch(a.out)
        return
    if not a.force and not docs.is_stale(a.out, HERE):
        if not a.quiet:
            print(f"Up to date: {os.path.join(a.out, 'reference.html')}")
        return
    r = docs.build(a.out, HERE, examples=not a.no_examples)
    if not a.quiet:
        print(f"{r['items']} items ({r['documented']} with docstrings, {r['examples']} with run examples)\n"
              f"  {r['html']}\n  {r['md']}")


if __name__ == "__main__":
    main()
