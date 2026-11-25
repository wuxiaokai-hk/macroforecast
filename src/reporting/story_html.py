"""The process story as one self-contained HTML page (charts embedded), with a contents sidebar.

Every section shows all its text: the finding, the takeaway, what was done, why, what happened in this run, how to
read the chart, the key figures, the chart and the table.
"""
import base64
import html
import re

E = html.escape

from . import themes as TH

CSS = """
:root{--ink:#$ink;--title:#$title;--kicker:#$kicker;--hl:#$highlight;--mid:#$mid;--tile:#$tile;--tilev:#$tile_value;
--line:#$line;--alt:#$alt;--page:#$page;--card:#$card;--headbg:#$head_bg;--headfg:#$head_fg;--coverbg:#$cover_bg;
--coverfg:#$cover_fg;--coversub:#$cover_sub;--navbg:#$nav_bg;--navfg:#$nav_fg;--navh:#$nav_h;--fh:$css_head;--fb:$css_body}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 var(--fb)}
nav{position:fixed;top:0;left:0;bottom:0;width:290px;overflow:auto;background:var(--navbg);color:var(--navfg);
padding:22px 18px;border-right:1px solid var(--line)}
nav h1{font:700 17px/1.3 var(--fh);color:var(--navh);margin:0 0 16px}
nav .part{margin:16px 0 4px;color:var(--hl);font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-weight:700}
nav a{display:block;color:var(--navfg);text-decoration:none;font-size:13px;padding:3px 0 3px 8px;border-left:2px solid transparent}
nav a:hover{color:var(--navh);border-left-color:var(--hl)}
main{margin-left:290px;padding:32px 48px 80px;max-width:1180px}
header.hero{background:var(--coverbg);color:var(--coverfg);border-radius:$radius;padding:40px 44px;margin-bottom:28px;
border:1px solid var(--line)$hero_extra}
header.hero h1{font:700 34px/1.2 var(--fh);margin:0 0 10px}
header.hero p{color:var(--coversub);font-size:18px;margin:0 0 22px;max-width:820px}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}
.tile{background:var(--tile);border-radius:$radius_s;padding:10px 16px;min-width:150px}
.tile b{display:block;font:700 22px/1.2 var(--fh);color:var(--tilev)}
.tile span{font-size:13px;color:var(--mid)}
.hero .tile{background:$hero_tile} .hero .tile b{color:var(--hl)} .hero .tile span{color:var(--coversub)}
h2.part{font:700 26px/1.2 var(--fh);color:var(--title);margin:46px 0 10px;padding-bottom:6px;border-bottom:3px solid var(--hl)}
section{background:var(--card);border:1px solid var(--line);border-radius:$radius;padding:26px 30px;margin:18px 0$card_extra}
section h3{font:700 22px/1.3 var(--fh);color:var(--title);margin:0 0 8px}
.take{border-left:4px solid var(--hl);padding:4px 0 4px 14px;color:var(--mid);font-size:17px;margin:6px 0 16px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:26px}
.block h4{margin:14px 0 4px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--kicker)}
.block ul{margin:0;padding-left:20px} .block li{margin:3px 0}
.read{background:var(--tile);border-radius:$radius_s;padding:8px 12px;font-size:14px;color:var(--mid);margin-top:10px}
figure{margin:16px 0 0} figure img{max-width:100%;border:1px solid var(--line);border-radius:$radius_s}
table{border-collapse:collapse;width:100%;font-size:14px;margin-top:14px}
th{background:var(--headbg);color:var(--headfg);text-align:left;padding:6px 8px;font-weight:600}
td{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tr:nth-child(even) td{background:var(--alt)}
.note{font-size:13px;color:var(--mid);margin-top:4px}
.idx{color:var(--hl);font-weight:700;margin-right:6px}
@media (max-width:900px){nav{display:none}main{margin-left:0;padding:16px}.cols{grid-template-columns:1fr}}
@media print{nav{display:none}main{margin-left:0}section{break-inside:avoid}body{background:#fff}}
"""

STYLE = {     # shape details per theme
    "classic": dict(radius="10px", radius_s="8px"),
    "corporate": dict(radius="4px", radius_s="3px", card_extra=";box-shadow:0 1px 3px rgba(0,0,0,.06)"),
    "midnight": dict(radius="12px", radius_s="8px"),
    "minimal": dict(radius="0", radius_s="0", card_extra=";border-width:0 0 1px 0;padding-left:0;padding-right:0",
                    hero_extra=";border-width:0 0 4px 0;border-color:#111;padding-left:0"),
    "forest": dict(radius="14px", radius_s="10px"),
    "boardroom": dict(radius="2px", radius_s="2px", hero_extra=";border-bottom:4px solid #B08D57"),
    "ocean": dict(radius="16px", radius_s="10px", card_extra=";box-shadow:0 2px 10px rgba(11,110,121,.07)"),
}


def css(theme=None):
    """The page's style sheet for a theme."""
    from string import Template
    t = TH.get(theme)
    light_nav = not TH.is_dark(dict(bg=t["nav_bg"]))
    v = {k: x for k, x in t.items() if isinstance(x, str)}
    v.update(nav_h=t["title"] if light_nav else t["cover_fg"],
             hero_tile="var(--tile)" if t["cover_bg"].upper() in ("FFFFFF", t["page"].upper()) else "rgba(255,255,255,.08)",
             radius="10px", radius_s="8px", card_extra="", hero_extra="")
    v.update(STYLE.get(t["name"], {}))
    return Template(CSS).substitute(v)


def _img(p):
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _s(v):
    return "" if v is None or (isinstance(v, float) and v != v) else str(v)


def _slug(s, i):
    return f"s{i}-" + re.sub(r"[^a-z0-9]+", "-", s.lower())[:40].strip("-")


def _list(items):
    items = [t for t in items if t]
    return "<ul>" + "".join(f"<li>{E(t)}</li>" for t in items) + "</ul>" if items else ""


def _table(df, note):
    if df is None or not len(df):
        return ""
    h = "<table><thead><tr>" + "".join(f"<th>{E(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for r in df.values:
        h += "<tr>" + "".join(f"<td>{E(_s(v))}</td>" for v in r) + "</tr>"
    return h + "</tbody></table>" + (f'<div class="note">{E(note)}</div>' if note else "")


def _tiles(nums, cls=""):
    if not nums:
        return ""
    return '<div class="tiles">' + "".join(f'<div class="tile"><b>{E(v)}</b><span>{E(k)}</span></div>' for k, v in nums) + "</div>"


def build(sections, path, title, theme=None):
    nav, body, part, k = [], [], None, 0
    for i, s in enumerate(sections):
        if s["kind"] == "title":
            body.append(f'<header class="hero"><h1>{E(s["title"])}</h1><p>{E(s["takeaway"])}</p>{_tiles(s["numbers"])}</header>')
            continue
        sid = _slug(s["title"], i)
        if s["part"] != part:
            part = s["part"]
            nav.append(f'<div class="part">{E(part)}</div>')
            body.append(f'<h2 class="part">{E(part)}</h2>')
        k += 1
        nav.append(f'<a href="#{sid}"><span class="idx">{k}</span>{E(s["title"][:70])}</a>')
        what = f'<div class="block"><h4>What we did</h4>{_list(s["what"])}</div>' if any(s["what"]) else ""
        why = f'<div class="block"><h4>Why</h4>{_list(s["why"])}</div>' if any(s["why"]) else ""
        res = f'<div class="block"><h4>What happened in this run</h4>{_list(s["result"])}</div>' if any(s["result"]) else ""
        left, right = what + why, res
        cols = f'<div class="cols"><div>{left}</div><div>{right}</div></div>' if left and right else (left or right)
        fig = f'<figure><img src="{_img(s["chart"])}" alt="{E(s["title"])}"></figure>' if s["chart"] else ""
        rd = f'<div class="read"><b>How to read it.</b> {E(s["read"])}</div>' if s["read"] else ""
        tb = _table(s["table"], s["table_note"])
        body.append(f'<section id="{sid}"><h3>{E(s["title"])}</h3>'
                    + (f'<div class="take">{E(s["takeaway"])}</div>' if s["takeaway"] else "")
                    + _tiles(s["numbers"]) + cols + fig + rd + tb + "</section>")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{E(title)}</title><style>{css(theme)}</style></head><body>'
            f'<nav><h1>{E(title)}</h1>{"".join(nav)}</nav><main>{"".join(body)}</main></body></html>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
