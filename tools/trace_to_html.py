"""
Turn a run trace (trace.json, written by src/utils/trace.py during a run) into a clickable HTML page.

    python tools/trace_to_html.py                                         # newest trace under outputs/
    python tools/trace_to_html.py outputs/ontario_loans_example/trace/trace.json
    python tools/trace_to_html.py <trace.json> --out my_page.html

run_pipeline.py and run_oos.py call this automatically at the end of every traced run, so you only need it
to rebuild a page (for example after editing the template) or to build one from a trace someone sent you.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def build(trace_path, out=None, body_only=False):
    """Turn a trace.json into a self-contained HTML page (or the page body for embedding)."""
    with open(trace_path, encoding="utf-8") as f:
        data = f.read()
    json.loads(data)                                              # fail early on a broken file
    with open(os.path.join(HERE, "trace_template.html"), encoding="utf-8") as f:
        tpl = f.read()
    page = tpl.replace("/*__DATA__*/null", data.replace("</", "<\\/"))
    if not body_only:
        page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>\n'
                + page + "\n</body></html>\n")
    out = out or os.path.join(os.path.dirname(trace_path), "trace.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


def main():
    """Command line: convert the newest (or a given) trace.json to HTML."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="path to trace.json (default: the newest one under outputs/)")
    ap.add_argument("--out", help="where to write the page (default: trace.html next to trace.json)")
    ap.add_argument("--body-only", action="store_true", help="leave out <html>/<head> (for embedding)")
    a = ap.parse_args()
    path = a.trace
    if not path:
        found = sorted(glob.glob(os.path.join(ROOT, "outputs", "*", "**", "trace.json"), recursive=True),
                       key=os.path.getmtime)
        if not found:
            raise SystemExit("No trace.json found under outputs/. Run run_pipeline.py first.")
        path = found[-1]
    print(build(path, a.out, a.body_only))


if __name__ == "__main__":
    main()
