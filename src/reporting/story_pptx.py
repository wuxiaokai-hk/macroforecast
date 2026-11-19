"""The process story as a PowerPoint deck (python-pptx only; no Quarto).

One slide per section: the finding as the title, the takeaway under it, the key points and figures on the left,
the chart or table on the right. The full text of the section (what was done, why, what happened, how to read the
chart) goes in the speaker notes, so the presenter has every detail and the slide stays readable.
"""
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

C = dict(ink="1B2A41", navy="0E3B4F", teal="1F6F8B", gold="E3A72F", mid="4A5866", light="EEF3F6", line="D9E1E8",
         white="FFFFFF", sky="C9DCE6", green="2E9E83")
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5


def rgb(k):
    return RGBColor.from_string(C.get(k, k))


def box(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, paras, size=12, color="ink", font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space=4):
    """paras: list of strings or (string, dict(size, color, bold, font, bullet))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        s, o = (p, {}) if isinstance(p, str) else p
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.space_after = Pt(o.get("space", space))
        r = par.add_run()
        r.text = (("•  " if o.get("bullet") else "") + s)
        f = r.font
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.name = o.get("font", font)
        f.color.rgb = rgb(o.get("color", color))
    return tb


def _fit(lines, w, h, start=15, low=9.5):
    """Largest font size (pt) at which the lines roughly fit a w x h inch box."""
    s = start
    while s > low:
        cpl = w * 72 / (s * 0.52)            # characters per line
        n = sum(max(1, -(-len(t) // int(cpl))) for t in lines)
        if n * s * 1.35 / 72 + len(lines) * 4 / 72 <= h:
            return s
        s -= 0.5
    return low


def table(slide, df, x, y, w, h, note=""):
    df = df.copy().map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v))
    cols = list(df.columns)
    longest = max([sum(len(v) for v in r) for r in df.values] or [10])
    size = 11 if longest < 70 and len(df) <= 8 else 10 if longest < 110 else 9
    rh = size * 1.9 / 72 + 0.06                       # one-line row height (inches)
    nr = max(1, min(len(df), int((h - (0.35 if note else 0.05)) / rh) - 1, 16))
    df = df.head(nr)
    cpi = 72 / (size * 0.5)                           # characters per inch
    need = [max(len(str(c)), max([len(v) for v in df[c]] or [1])) + 2 for c in cols]
    mins = [min(n, 9) / cpi + 0.12 for n in need]      # short columns never wrap
    flex = [max(0.0, min(n, 70) / cpi + 0.12 - m) for n, m in zip(need, mins)]
    room = max(0.0, w - sum(mins))
    widths = [m + (room * f / sum(flex) if sum(flex) else room / len(cols)) for m, f in zip(mins, flex)]
    shp = slide.shapes.add_table(nr + 1, len(cols), Inches(x), Inches(y), Inches(w), Inches(rh * (nr + 1)))
    tbl = shp.table
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Inches(cw)
    for i in range(nr + 1):
        tbl.rows[i].height = Inches(rh)
    for j, c in enumerate(cols):
        cell = tbl.cell(0, j)
        cell.text = c
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb("navy")
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.bold, r.font.name = Pt(size), True, BODY
                r.font.color.rgb = rgb("white")
    for i, row in enumerate(df.values, start=1):
        for j, v in enumerate(row):
            cell = tbl.cell(i, j)
            cap = int(widths[j] * cpi * 2) - 1             # at most about two lines
            cell.text = v if len(v) <= cap else v[:max(cap - 1, 4)] + "…"
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb("light" if i % 2 else "white")
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.name = Pt(size), BODY
                    r.font.color.rgb = rgb("ink")
                    if v in ("yes", "pass"):
                        r.font.bold = True
    est = rh * (nr + 1) + sum(0.12 for r in df.values if any(len(v) > widths[j] * cpi for j, v in enumerate(r)))
    if note:
        text(slide, x, min(y + est + 0.05, y + h - 0.3), w, 0.35, [note], size=9, color="mid")


def notes(slide, s):
    parts = []
    for k, lab in (("what", "WHAT WE DID"), ("why", "WHY"), ("result", "WHAT HAPPENED IN THIS RUN")):
        v = [t for t in s[k] if t]
        if v:
            parts.append(lab + "\n" + "\n".join("- " + t for t in v))
    if s["read"]:
        parts.append("HOW TO READ IT\n" + s["read"])
    if s["table_note"]:
        parts.append("NOTE\n" + s["table_note"])
    slide.notes_slide.notes_text_frame.text = "\n\n".join(parts)


def chrome(slide, s, n, footer):
    box(slide, 0, 0, W, 0.09, "navy")
    if s["part"]:
        text(slide, 0.55, 0.28, 9, 0.3, [s["part"].upper()], size=10, color="teal", bold=True)
    tsize = 24 if len(s["title"]) < 70 else 21 if len(s["title"]) < 95 else 19
    text(slide, 0.55, 0.55, 12.2, 0.9, [s["title"]], size=tsize, color="navy", font=HEAD, bold=True)
    if s["takeaway"]:
        box(slide, 0.55, 1.42, 0.07, 0.5, "gold")
        text(slide, 0.72, 1.4, 12, 0.6, [s["takeaway"]], size=13, color="mid", anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 0.55, 7.08, 9, 0.3, [footer], size=9, color="mid")
    text(slide, 12.2, 7.08, 0.6, 0.3, [str(n)], size=9, color="mid", align=PP_ALIGN.RIGHT)


def tiles(slide, nums, x, y, w):
    nums = nums[:4]
    if not nums:
        return 0
    tw = (w - 0.15 * (len(nums) - 1)) / len(nums)
    for i, (k, v) in enumerate(nums):
        xx = x + i * (tw + 0.15)
        box(slide, xx, y, tw, 0.95, "light")
        vs = 20 if len(v) <= 10 else 15 if len(v) <= 18 else 12
        text(slide, xx + 0.1, y + 0.06, tw - 0.2, 0.5, [v], size=vs, color="navy", font=HEAD, bold=True)
        text(slide, xx + 0.1, y + 0.6, tw - 0.2, 0.3, [k], size=9.5, color="mid")
    return 1.1


def points(s, n=5):
    """The slide's bullet points: what happened first, then what was done."""
    pts = [t for t in s["result"] if t][:n]
    if len(pts) < 3:
        pts += [t for t in s["what"] if t][: n - len(pts)]
    if len(pts) < 2:
        pts += [t for t in s["why"] if t][: n - len(pts)]
    return pts


def build(sections, path, footer):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    part, n = None, 0
    for s in sections:
        if s["kind"] == "title":
            sl = prs.slides.add_slide(blank)
            box(sl, 0, 0, W, H, "navy")
            box(sl, 0.8, 2.3, 0.1, 1.6, "gold")
            text(sl, 1.1, 2.15, 11, 1.2, [s["title"]], size=38, color="white", font=HEAD, bold=True)
            text(sl, 1.1, 3.35, 10.5, 1, [s["takeaway"]], size=16, color="sky")
            for i, (k, v) in enumerate(s["numbers"]):
                text(sl, 1.1 + i * 2.9, 5.2, 2.8, 0.5, [v], size=16, color="gold", font=HEAD, bold=True)
                text(sl, 1.1 + i * 2.9, 5.7, 2.8, 0.4, [k], size=11, color="sky")
            n += 1
            continue
        if s["part"] != part and s["part"]:
            part = s["part"]
            sl = prs.slides.add_slide(blank)
            n += 1
            box(sl, 0, 0, W, H, "navy")
            text(sl, 1.1, 3.0, 11, 1.2, [part], size=34, color="white", font=HEAD, bold=True)
            heads = [x["title"] for x in sections if x["part"] == part][:8]
            text(sl, 1.1, 4.1, 11, 3, [(h, {"size": 13, "color": "sky", "bullet": True, "space": 3}) for h in heads],
                 size=13, color="sky")
        sl = prs.slides.add_slide(blank)
        n += 1
        chrome(sl, s, n, footer)
        notes(sl, s)
        top = 2.15 if s["takeaway"] else 1.6
        body_h = 6.95 - top
        vis = s["chart"] or (s["table"] is not None and len(s["table"]))
        if s["kind"] == "closing" or not vis:
            used = tiles(sl, s["numbers"], 0.55, top, 12.2)
            pts = [t for t in s["what"] if t] + [t for t in s["result"] if t]
            why = [t for t in s["why"] if t]
            if why and s["kind"] != "closing":
                fs = _fit(pts, 6.6, body_h - used)
                text(sl, 0.55, top + used, 6.6, body_h - used, [(t, {"bullet": True}) for t in pts], size=fs)
                box(sl, 7.5, top + used, 5.25, body_h - used - 0.1, "light")
                text(sl, 7.7, top + used + 0.12, 4.9, 0.3, ["Why"], size=12, color="teal", bold=True)
                text(sl, 7.7, top + used + 0.45, 4.9, body_h - used - 0.6, why,
                     size=_fit(why, 4.9, body_h - used - 0.6, 12))
            else:
                fs = _fit(pts, 12.2, body_h - used, 15)
                text(sl, 0.55, top + used, 12.2, body_h - used, [(t, {"bullet": True}) for t in pts], size=fs)
            continue
        lw = 4.55
        used = 0
        if s["numbers"]:
            used = tiles(sl, s["numbers"][:2] if len(s["numbers"]) > 2 else s["numbers"], 0.55, top, lw)
        pts = points(s)
        rd = s["read"]
        avail = body_h - used - (0.75 if rd else 0)
        if pts:
            fs = _fit(pts, lw, avail)
            text(sl, 0.55, top + used, lw, avail, [(t, {"bullet": True}) for t in pts], size=fs)
        if rd:
            text(sl, 0.55, 6.95 - 0.7, lw, 0.65, [("How to read: " + rd, {"size": 9.5, "color": "mid"})], size=9.5)
        rx, rw = 5.35, 7.45
        if s["chart"]:
            from PIL import Image
            im = Image.open(s["chart"])
            ar = im.height / im.width
            ch_h = min(body_h, rw * ar)
            tab = s["table"] is not None and len(s["table"]) and ar <= 0.6      # a tall chart takes the slide
            if tab:
                need = 0.37 * (min(len(s["table"]), 5) + 1) + 0.2
                ch_h = min(ch_h, max(body_h - need, body_h * 0.55))
            cw = ch_h / ar
            sl.shapes.add_picture(s["chart"], Inches(rx + (rw - cw) / 2), Inches(top), Inches(cw), Inches(ch_h))
            if tab and body_h - ch_h > 1.0:
                table(sl, s["table"], rx, top + ch_h + 0.1, rw, body_h - ch_h - 0.15, "")
        elif not pts:
            table(sl, s["table"], 0.55, top + used, 12.2, body_h - used, s["table_note"])
        else:
            table(sl, s["table"], rx, top, rw, body_h, s["table_note"])
    prs.save(path)
    return path
