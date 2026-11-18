"""
Design pass for the Quarto deck: turns the plain PowerPoint that `quarto render` writes into the designed deck
(cards, icon grids, stat tiles, process steps, native PowerPoint charts, styled tables, footer).

It runs automatically after `quarto render report/forecast_deck.qmd` (see report/_quarto.yml), or by hand:

    python report/style_deck.py report/forecast_deck.pptx

The .qmd stays plain Markdown. Special paragraphs ("markers") in it are replaced by designed elements:

  [[tile: 1.37% | best within the limit | Screen first, 1-12 quarters]]     stat tile   ([[tile*: ...]] = gold number)
  [[icon: home | House prices | Collateral value drives demand]]           icon card   ([[icon@red: ...]] other colour)
  [[step: filter | 3. Screen | One-at-a-time test]]                        process step (a row, with chevrons)
  [[card: Levels | q/q growth; y/y growth; gap]]                           card with a list ([[card*: ...]] = dark card)
  [[band: Also shaping the design | stress tests now project balance-sheet volumes]]  dark band across the bottom
  [[chart: oos_bars]]                                                      native chart from report/_charts/oos_bars.json
  [[split: 60]]                                                            left column 60% of the width
  [[dark-slide]]                                                           dark background for the whole slide

Other rules: a column whose first line starts with "★" becomes a dark card, other columns become light cards;
tables get a dark header, banded rows and green "Pass"; every content slide gets the footer, the slide number
and (for simulated data) a note. Icons are PNG files in report/icons/ (names: see that folder).
"""
import json
import os
import re
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
C = dict(ink="1B2A41", navy="0E3B4F", navy2="1A4A61", light="EEF3F6", teal="1F6F8B", gold="E3A72F", green="2E9E83",
         red="C2553F", grey="8796A8", purple="5B4B8A", orange="C77B30", plum="8E4A6E", sea="0F8A8A", mid="4A5866", white="FFFFFF", sky="C9DCE6", line="D9E1E8")
HEAD, BODY = "Cambria", "Calibri"
MARK = re.compile(r"^\s*\[\[\s*([a-z\-]+)(\*)?(?:@(\w+))?\s*:?\s*(.*?)\s*\]\]\s*$", re.S)
IN = 914400
L, R, T, B = 0.5, 9.5, 1.22, 5.08          # content box (inches) on a 10 x 5.625 slide
FOOT = "Macro-driver forecasting model"


def rgb(k):
    """Colour from a palette name (navy, teal, gold ...) or a hex string."""
    return RGBColor.from_string(C.get(k, k))


def emu(x):
    """Inches to EMU (PowerPoint units)."""
    return int(round(x * IN))


def inch(v):
    """EMU to inches."""
    return Emu(v).inches


# ---------------------------------------------------------------- low-level helpers
def send_to_back(slide, shape):
    """Move a shape behind all others on its slide."""
    tree = slide.shapes._spTree
    tree.remove(shape._element)
    tree.insert(2, shape._element)


def remove(shape):
    """Delete a shape from its slide."""
    shape._element.getparent().remove(shape._element)


def rect(slide, x, y, w, h, fill="light", rounded=True, back=False):
    """Add a filled (rounded) rectangle; back=True sends it behind the text."""
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                               emu(x), emu(y), emu(w), emu(h))
    if rounded:
        s.adjustments[0] = min(0.12, 0.08 / max(min(w, h), 0.1) * 1.0)
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if back:
        send_to_back(slide, s)
    return s


def text(slide, x, y, w, h, runs, size=11, color="ink", font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False):
    """runs: a string, or a list of paragraphs; each paragraph a string or a list of (text, {overrides})."""
    tb = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        for t, o in (p if isinstance(p, list) else [(p, {})]):
            r = para.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.name = o.get("font", font)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def circle_icon(slide, x, y, d, name, bg="teal"):
    """Add a coloured circle with a white icon from report/icons."""
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, emu(x), emu(y), emu(d), emu(d))
    c.fill.solid()
    c.fill.fore_color.rgb = rgb(bg)
    c.line.fill.background()
    c.shadow.inherit = False
    path = os.path.join(HERE, "icons", f"{name}.png")
    if os.path.exists(path):
        pad = d * 0.24
        slide.shapes.add_picture(path, emu(x + pad), emu(y + pad), emu(d - 2 * pad), emu(d - 2 * pad))
    else:
        text(slide, x, y, d, d, name[:2], size=d * 22, color="white", bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)


def parts(s, n):
    """Split a marker payload 'a | b | c' into n stripped parts (padded with empty strings)."""
    p = [x.strip() for x in s.split("|")]
    return (p + [""] * n)[:n]


def geom(shape):
    """Position and size of a shape in inches: [x, y, w, h]."""
    return [inch(shape.left), inch(shape.top), inch(shape.width), inch(shape.height)]


def set_geom(shape, x, y, w, h):
    """Move and resize a shape (tables keep their relative column widths)."""
    old_w = shape.width
    shape.left, shape.top, shape.width, shape.height = emu(x), emu(y), emu(w), emu(h)
    if shape.has_table and old_w:                    # keep the table's relative column widths
        cols = shape.table.columns
        tot = sum(c.width for c in cols)
        for c in cols:
            c.width = int(c.width * emu(w) / tot)


# ---------------------------------------------------------------- components
def tiles(slide, items, x, y, w, h):
    """Draw stat tiles (big number + label) from [[tile: ...]] markers."""
    n = len(items)
    row = w / max(h, 0.1) > 1.6
    gap = 0.18
    for i, (star, color, body) in enumerate(items):
        big, label, sub = parts(body, 3)
        if row:
            tw, th = (w - gap * (n - 1)) / n, h
            tx, ty = x + i * (tw + gap), y
        else:
            tw, th = w, (h - gap * (n - 1)) / n
            tx, ty = x, y + i * (th + gap)
        rect(slide, tx, ty, tw, th, "light")
        big_size = 30 if th > 1.2 else 24 if th > 0.95 else 20
        text(slide, tx + 0.18, ty + 0.12, tw - 0.36, big_size / 60, big, size=big_size, bold=True, font=HEAD,
             color="gold" if star else (color or "teal"))
        yy = ty + 0.14 + big_size / 60
        text(slide, tx + 0.18, yy, tw - 0.36, th - (yy - ty) - 0.08,
             [[(label, {"bold": True, "size": 11.5})], [(sub, {"size": 9.5, "color": "mid"})]] if sub else
             [[(label, {"bold": True, "size": 11.5})]])


def icon_grid(slide, items, x, y, w, h):
    """Draw a grid of icon cards from [[icon: name | title | text]] markers."""
    n = len(items)
    cols = 1 if w < 4.2 else (2 if (w < 6.5 or n <= 4) else (4 if (w >= 8.5 and n > 9) else 3))
    rows = -(-n // cols)
    gap = 0.16
    cw, ch = (w - gap * (cols - 1)) / cols, min(1.45, (h - gap * (rows - 1)) / rows)
    d = min(0.52, ch * 0.55)
    for i, (star, color, body) in enumerate(items):
        icon, head, sub = parts(body, 3)
        cx, cy = x + (i % cols) * (cw + gap), y + (i // cols) * (ch + gap)
        rect(slide, cx, cy, cw, ch, "light")
        circle_icon(slide, cx + 0.16, cy + (ch - d) / 2 if ch < 0.95 else cy + 0.16, d, icon, color or "teal")
        tx = cx + 0.16 + d + 0.16
        text(slide, tx, cy + 0.12, cw - (tx - cx) - 0.12, ch - 0.2,
             [[(head, {"bold": True, "size": 11.5, "font": HEAD})], [(sub, {"size": 9.5, "color": "mid"})]] if sub else
             [[(head, {"bold": True, "size": 11.5, "font": HEAD})]], anchor=MSO_ANCHOR.MIDDLE if ch < 0.95 else MSO_ANCHOR.TOP)


def steps(slide, items, x, y, w, h):
    """Draw numbered process steps from [[step: ...]] markers."""
    n = len(items)
    gap = 0.22
    cw = (w - gap * (n - 1)) / n
    for i, (star, color, body) in enumerate(items):
        icon, head, sub = parts(body, 3)
        cx = x + i * (cw + gap)
        rect(slide, cx, y, cw, h, "light")
        d = min(0.62, cw * 0.45)
        circle_icon(slide, cx + (cw - d) / 2, y + 0.2, d, icon, color or "teal")
        text(slide, cx + 0.08, y + 0.32 + d, cw - 0.16, h - d - 0.4,
             [[(head, {"bold": True, "size": 11, "font": HEAD})], [(sub, {"size": 9, "color": "mid"})]],
             align=PP_ALIGN.CENTER)
        if i < n - 1:
            text(slide, cx + cw, y + h / 2 - 0.15, gap, 0.3, "›", size=20, color="grey", bold=True, align=PP_ALIGN.CENTER)


def cards(slide, items, x, y, w, h):
    """Draw text cards from [[card: title | text]] markers."""
    n = len(items)
    gap = 0.18
    cw = (w - gap * (n - 1)) / n
    for i, (star, color, body) in enumerate(items):
        head, rest = (body.split("|", 1) + [""])[:2]
        dark = bool(star)
        cx = x + i * (cw + gap)
        rect(slide, cx, y, cw, h, color or ("navy" if dark else "light"))
        lines = [ln.strip() for ln in rest.split(";") if ln.strip()]
        paras = [[(head.strip(), {"bold": True, "size": 12, "font": HEAD, "color": "gold" if dark else "teal"})]]
        paras += [[("• " + ln if len(lines) > 1 else ln, {"size": 10, "color": "white" if dark else "ink"})] for ln in lines]
        tb = text(slide, cx + 0.16, y + 0.14, cw - 0.32, h - 0.24, paras)
        for p in tb.text_frame.paragraphs[1:]:
            p.space_before = Pt(3)


def band(slide, body, dark_slide=False):
    """Draw the dark take-away band at the bottom of a slide."""
    lead, rest = (body.split("|", 1) + [""])[:2]
    rect(slide, L, B - 0.55, R - L, 0.55, "navy2" if dark_slide else "navy", rounded=True)
    text(slide, L + 0.2, B - 0.5, R - L - 0.4, 0.45,
         [[(lead.strip() + (": " if rest else ""), {"bold": True, "color": "gold", "size": 10.5}),
           (rest.strip(), {"color": "white", "size": 10.5})]], anchor=MSO_ANCHOR.MIDDLE)


def chart(slide, cid, x, y, w, h):
    """Insert a native PowerPoint chart from report/_charts/<cid>.json."""
    path = os.path.join(HERE, "_charts", f"{cid}.json")
    if not os.path.exists(path):
        text(slide, x, y, w, h, f"(chart {cid} not found)", color="red")
        return
    spec = json.load(open(path, encoding="utf-8"))
    kind = spec["type"]
    fmt = spec.get("number_format", "0.00")
    if kind == "scatter":
        data = XyChartData()
        for s in spec["series"]:
            ser = data.add_series(s["name"])
            for px, py in zip(s["x"], s["y"]):
                ser.add_data_point(px, py)
        gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, emu(x), emu(y), emu(w), emu(h), data)
    else:
        data = CategoryChartData(number_format=fmt)
        data.categories = spec["categories"]
        for s in spec["series"]:
            data.add_series(s["name"], [None if v is None else v for v in s["values"]])
        ctype = {"barh": XL_CHART_TYPE.BAR_CLUSTERED, "bar": XL_CHART_TYPE.COLUMN_CLUSTERED,
                 "line": XL_CHART_TYPE.LINE_MARKERS if spec.get("markers", True) else XL_CHART_TYPE.LINE}[kind]
        gf = slide.shapes.add_chart(ctype, emu(x), emu(y), emu(w), emu(h), data)
    ch = gf.chart
    ch.font.name, ch.font.size = BODY, Pt(spec.get("font_size", 9))
    ch.font.color.rgb = rgb("ink")
    if spec.get("title"):
        ch.has_title = True
        ch.chart_title.text_frame.text = spec["title"]
        tp = ch.chart_title.text_frame.paragraphs[0]
        tp.runs[0].font.size, tp.runs[0].font.bold = Pt(10.5), False
        tp.runs[0].font.color.rgb = rgb("mid")
    else:
        ch.has_title = False
    ch.has_legend = spec.get("legend", len(spec["series"]) > 1)
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM if spec.get("legend_pos", "bottom") == "bottom" else XL_LEGEND_POSITION.RIGHT
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(8.5)
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("line")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(8.5)
    va.tick_labels.number_format = spec.get("axis_format", fmt)
    va.tick_labels.number_format_is_linked = False
    if spec.get("y_min") is not None:
        va.minimum_scale = spec["y_min"]
    elif kind == "line":                     # lines: start the axis near the data, not at zero
        vals = [v for s_ in spec["series"] for v in s_["values"] if v is not None]
        if vals and min(vals) > 0:
            lo, hi = min(vals), max(vals)
            step = 10 ** len(str(int(max(1, (hi - lo) / 5)))) / 10 or 1
            va.minimum_scale = max(0, (lo - (hi - lo) * 0.08) // step * step)
    if spec.get("y_max") is not None:
        va.maximum_scale = spec["y_max"]
    if spec.get("y_title"):
        va.has_title = True
        va.axis_title.text_frame.text = spec["y_title"]
        va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(8.5)
        va.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(spec.get("cat_font_size", 8.5))
    ca.format.line.color.rgb = rgb("line")
    ca.has_major_gridlines = False
    if kind == "barh":
        ca.reverse_order = True
    if spec.get("label_skip"):
        _set_skip(ca._element, int(spec["label_skip"]))
    if kind == "scatter":
        ca.tick_labels.number_format, ca.tick_labels.number_format_is_linked = "0", False
        if spec.get("x_min") is not None:
            ca.minimum_scale = spec["x_min"]
        if spec.get("x_max") is not None:
            ca.maximum_scale = spec["x_max"]
        if spec.get("x_title"):
            ca.has_title = True
            ca.axis_title.text_frame.text = spec["x_title"]
            ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(8.5)
            ca.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    if kind in ("barh", "bar"):
        ch.plots[0].gap_width = spec.get("gap_width", 45)
        ch.plots[0].overlap = 0
    for s, ser in zip(spec["series"], ch.series):
        col = s.get("color", "teal")
        if kind in ("barh", "bar"):
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = rgb(col)
            for i, pc in enumerate(s.get("colors") or []):
                pt = ser.points[i]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = rgb(pc)
            if spec.get("labels", True):
                ser.data_labels.show_value = True
                ser.data_labels.number_format, ser.data_labels.number_format_is_linked = fmt, False
                ser.data_labels.font.size = Pt(8)
                ser.data_labels.font.color.rgb = rgb("mid")
                ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        else:
            ln = ser.format.line
            if kind == "scatter":
                ln.fill.background()
                ser.marker.style = XL_MARKER_STYLE.CIRCLE
                ser.marker.size = s.get("marker_size", 9)
            else:
                ln.color.rgb = rgb(col)
                ln.width = Pt(s.get("width", 2.25))
                ser.smooth = False
                ser.marker.style = XL_MARKER_STYLE.CIRCLE if spec.get("markers", True) else XL_MARKER_STYLE.NONE
                ser.marker.size = 5
                if s.get("dash"):
                    ln.dash_style = MSO_LINE_DASH_STYLE.DASH
            ser.marker.format.fill.solid()
            ser.marker.format.fill.fore_color.rgb = rgb(col)
            ser.marker.format.line.fill.background()


def _set_skip(cat_ax, n):
    """Show every n-th category label (c:tickLblSkip, inserted where the schema expects it)."""
    from pptx.oxml.ns import qn
    for tag in ("c:tickLblSkip", "c:tickMarkSkip"):
        old = cat_ax.find(qn(tag))
        if old is not None:
            cat_ax.remove(old)
    el = etree.Element(qn("c:tickLblSkip"))
    el.set("val", str(n))
    anchor = cat_ax.find(qn("c:noMultiLvlLbl"))
    if anchor is None:
        anchor = cat_ax.find(qn("c:extLst"))
    if anchor is not None:
        anchor.addprevious(el)
    else:
        cat_ax.append(el)


# ---------------------------------------------------------------- slide-level passes
def style_table(gf, dark_slide=False):
    """Apply the deck's table style (header colour, banding, fonts, borders)."""
    tbl = gf.table
    for r, row in enumerate(tbl.rows):
        for c, cell in enumerate(row.cells):
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb("navy" if r == 0 else ("light" if r % 2 else "white"))
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name = BODY
                    run.font.size = Pt(9.5 if len(tbl.rows) < 9 else 8.5)
                    run.font.color.rgb = rgb("white" if r == 0 else "ink")
                    run.font.bold = r == 0 or (c == 0 and len(tbl.columns) > 2) or run.font.bold
                    if r > 0 and run.text.strip() in ("Pass", "yes", "in use"):
                        run.font.color.rgb, run.font.bold = rgb("green"), True
                    if r > 0 and run.text.strip() in ("Review", "no", "switched off"):
                        run.font.color.rgb, run.font.bold = rgb("red"), True
                    if r > 0 and run.text.strip() in ("Positive", "+"):
                        run.font.color.rgb, run.font.bold = rgb("green"), True
                    if r > 0 and run.text.strip() in ("Negative", "−", "-"):
                        run.font.color.rgb, run.font.bold = rgb("red"), True
    # first row height minimal
    tblPr = gf._element.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}tblPr")
    if tblPr is not None:
        for k in ("bandRow", "firstRow"):
            tblPr.set(k, "0")


def style_title(shape, dark=False):
    """Centre and style a slide title."""
    tf = shape.text_frame
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.name, r.font.bold, r.font.size = HEAD, True, Pt(20)
            r.font.color.rgb = rgb("white" if dark else "ink")
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def card_text(shape, dark):
    """Light or dark card behind a text column; first bold line becomes the card heading."""
    tf = shape.text_frame
    tf.margin_left = tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.14)
    tf.margin_bottom = Inches(0.1)
    tf.word_wrap = True
    for i, p in enumerate(tf.paragraphs):
        for r in p.runs:
            if r.text.startswith("★"):
                r.text = r.text.lstrip("★ ").lstrip()
            r.font.name = BODY
            r.font.size = Pt(11.5)
            heading = i == 0 and (r.font.bold or len(tf.paragraphs) > 1 and all(rr.font.bold for rr in p.runs))
            if heading:
                r.font.name, r.font.size = HEAD, Pt(13)
                r.font.color.rgb = rgb("gold" if dark else "teal")
            else:
                r.font.color.rgb = rgb("white" if dark else "ink")
                if r.font.bold and dark:
                    r.font.color.rgb = rgb("white")


def paragraphs_markers(shape):
    """Find the [[kind: ...]] marker paragraphs in a text shape."""
    out = []
    for p in shape.text_frame.paragraphs:
        t = "".join(r.text for r in p.runs).strip()
        m = MARK.match(t) if t.startswith("[[") else None
        out.append((p, m))
    return out


def process_slide(slide, idx, meta):
    """Apply the design to one content slide: layout, markers, tables, charts and footer."""
    layout = slide.slide_layout.name
    shapes = list(slide.shapes)
    for sh in shapes:                       # write inherited positions explicitly before moving anything
        if sh.is_placeholder:
            x, y, w, h = sh.left, sh.top, sh.width, sh.height
            sh.left, sh.top, sh.width, sh.height = x, y, w, h
    title = next((s for s in shapes if s.is_placeholder and s.placeholder_format.idx == 0), None)
    body = [s for s in shapes if s is not title]
    if layout in ("Title Slide", "Section Header"):
        return
    # ---- markers
    found = []          # (shape, kind, star, color, payload)
    flags = {}
    for s in body:
        if not s.has_text_frame:
            continue
        for p, m in paragraphs_markers(s):
            if not m:
                continue
            kind, star, color, payload = m.group(1), bool(m.group(2)), m.group(3), m.group(4)
            if kind in ("split", "dark-slide"):
                flags[kind] = payload or True
            else:
                found.append((s, kind, star, color, payload))
            p._p.getparent().remove(p._p)
    dark = "dark-slide" in flags
    if dark:
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb("navy")
    if title is not None:
        style_title(title, dark)
        title.left, title.top, title.width, title.height = emu(0.4), emu(0.18), emu(9.2), emu(0.9)

    def has_text(s):
        return s.has_text_frame and any("".join(r.text for r in p.runs).strip() for p in s.text_frame.paragraphs)

    # ---- columns: re-split the two body shapes if asked
    cols = sorted([s for s in body if s.width and inch(s.width) < 6.5], key=lambda s: s.left)
    if len(cols) == 2 and abs(cols[0].top - cols[1].top) < Inches(0.3):
        ratio = float(flags.get("split", 50)) / 100
        gap = 0.25
        lw = (R - L - gap) * ratio
        for s, (x, w) in zip(cols, [(L, lw), (L + lw + gap, R - L - gap - lw)]):
            set_geom(s, x, T, w, B - T)
    elif len(body) == 1 and not body[0].has_text_frame:
        x, y, w, h = geom(body[0])
        if body[0].shape_type == 13:              # picture: keep aspect, centre
            pass
        else:
            set_geom(body[0], L, T, R - L, B - T)

    # ---- full-width tiles row on top (tiles in a column shape, at least 3, other content present)
    groups = {}
    for s, kind, star, color, payload in found:
        groups.setdefault((id(s), kind), {"shape": s, "kind": kind, "items": []})["items"].append(
            (star, {"red": "red", "green": "green", "gold": "gold", "purple": "purple", "navy": "navy", "grey": "grey", "orange": "orange", "plum": "plum", "sea": "sea"}.get(color, None), payload))
    bands = [g for g in groups.values() if g["kind"] == "band"]
    bottom = B - (0.7 if bands else 0)
    if bands:
        for s in body:
            if s.top + s.height > emu(bottom):
                s.height = max(emu(0.3), emu(bottom) - s.top)
    to_remove = {}
    for g in groups.values():
        s, kind, items = g["shape"], g["kind"], g["items"]
        x, y, w, h = geom(s)
        h = min(h, bottom - y)
        rest = has_text(s)
        if kind == "tile" and rest and len(items) >= 3:            # tiles row across, content pushed down
            th = 1.3
            tiles(slide, [i for i in items], L, T, R - L, th)
            for o in body:
                if o.top < emu(T + 0.1):
                    o.top = emu(T + th + 0.2)
                    o.height = max(emu(0.5), emu(bottom - (T + th + 0.2)))
            continue
        if rest:                                                    # marker block under the text of the shape
            n_lines = sum(1 for p in s.text_frame.paragraphs if "".join(r.text for r in p.runs).strip())
            text_h = min(h * 0.55, 0.32 * n_lines + 0.25)
            region = (x, y + text_h, w, h - text_h)
            s.height = emu(text_h)
        else:
            region = (x, y, w, h)
            to_remove[id(s._element)] = s
        rx, ry, rw, rh = region
        if kind == "tile":
            tiles(slide, items, rx, ry, rw, rh)
        elif kind == "icon":
            icon_grid(slide, items, rx, ry, rw, rh)
        elif kind == "step":
            steps(slide, items, rx, ry, rw, min(rh, 2.2))
        elif kind == "card":
            cards(slide, items, rx, ry, rw, min(rh, 2.2))
        elif kind == "chart":
            chart(slide, items[0][2], rx, ry, rw, rh)
        elif kind == "band":
            band(slide, items[0][2], dark)

    for s in to_remove.values():
        remove(s)

    # ---- cards behind remaining text columns, styled tables
    for s in list(slide.shapes):
        if (s.is_placeholder and s.placeholder_format.idx == 0) or s.shape_type in (13,):
            continue
        if s.has_table:
            style_table(s, dark)
        elif s.is_placeholder and has_text(s) and inch(s.width) < 6.5:
            first = "".join(r.text for r in s.text_frame.paragraphs[0].runs).strip()
            is_dark = first.startswith("★") or dark
            x, y, w, h = geom(s)
            card = rect(slide, x, y, w, h, "navy2" if dark else ("navy" if is_dark else "light"))
            send_to_back(slide, card)
            card_text(s, is_dark)
        elif s.is_placeholder and has_text(s):
            for p in s.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = BODY
                    r.font.color.rgb = rgb("white" if dark else "ink")

    # ---- footer
    text(slide, L, 5.3, 4.5, 0.2, FOOT, size=8, color="sky" if dark else "grey")
    text(slide, 9.0, 5.3, 0.5, 0.2, str(idx), size=8, color="sky" if dark else "grey", align=PP_ALIGN.RIGHT)
    if meta.get("simulated"):
        text(slide, 4.6, 5.3, 4.3, 0.2, meta.get("note", "Results on simulated data"), size=8, italic=True,
             color="gold" if dark else "red", align=PP_ALIGN.RIGHT)


def process_title(slide, meta):
    """Style the title slide."""
    for s in slide.shapes:
        if not s.has_text_frame:
            continue
        idx = s.placeholder_format.idx if s.is_placeholder else None
        for p in s.text_frame.paragraphs:
            for r in p.runs:
                r.font.name = HEAD if idx == 0 else BODY
                if idx == 0:
                    r.font.bold, r.font.size = True, Pt(34)
    if meta.get("simulated"):
        text(slide, 5.2, 4.95, 4.3, 0.3, "Results shown are on simulated data", size=11, color="gold", align=PP_ALIGN.RIGHT)


def process_section(slide):
    """Style a section divider slide."""
    for s in slide.shapes:
        if s.has_text_frame:
            for p in s.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name, r.font.bold, r.font.size = HEAD, True, Pt(30)


def fix_empty_text(prs):
    """Make every text box valid for PowerPoint: a text body must hold at least one paragraph.

    When a slide's content is left out (e.g. no out-of-sample results) the design markers are removed and a placeholder
    can end up with an empty <p:txBody>, which PowerPoint "repairs" by dropping shapes. Empty placeholders are removed;
    any other empty text body gets an empty paragraph.

    Args:
        prs: Presentation (changed in place).

    Returns:
        number of shapes fixed.
    """
    from pptx.oxml.ns import qn
    n = 0
    for slide in prs.slides:
        for shape in list(slide.shapes):
            for tb in shape._element.iter(qn("p:txBody"), qn("a:txBody")):
                if tb.find(qn("a:p")) is not None:
                    continue
                n += 1
                is_ph = shape.is_placeholder and shape.placeholder_format.type not in (1, 3)
                if is_ph and tb.getparent() is shape._element:
                    shape._element.getparent().remove(shape._element)
                    break
                tb.append(tb.makeelement(qn("a:p"), {}))
    return n


def check_package(path):
    """Look for the faults PowerPoint refuses (it then asks to repair the file and drops content).

    Checks every slide and chart: text bodies without a paragraph, duplicate shape ids on a slide, relationships to
    missing parts, and 'nan' / 'inf' numbers in chart data.

    Args:
        path: .pptx file.

    Returns:
        list of problem descriptions (empty = clean).
    """
    import re
    import zipfile
    from lxml import etree
    out = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for n in sorted(names):
            if n.endswith(".rels"):
                base = os.path.dirname(os.path.dirname(n))
                for rel in etree.fromstring(z.read(n)).iter():
                    t, mode = rel.get("Target"), rel.get("TargetMode")
                    if t and mode != "External":
                        tgt = os.path.normpath(os.path.join(base, t)).replace("\\", "/").lstrip("/") if not t.startswith("/") \
                            else t.lstrip("/")
                        if tgt not in names:
                            out.append(f"{n}: points to a missing part {t}")
            if re.match(r"ppt/slides/slide\d+\.xml$", n):
                x = etree.fromstring(z.read(n))
                ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                for tb in x.iterfind(".//p:txBody", ns):
                    if tb.find("a:p", ns) is None:
                        out.append(f"{n}: a text box with no paragraph")
                ids = [e.get("id") for e in x.iterfind(".//p:cNvPr", ns)]
                dup = sorted({i for i in ids if ids.count(i) > 1})
                if dup:
                    out.append(f"{n}: duplicate shape ids {dup}")
            if re.match(r"ppt/charts/chart\d+\.xml$", n):
                txt = z.read(n).decode("utf-8", "ignore")
                if re.search(r"<c:v>\s*(nan|NaN|inf|-inf)\s*</c:v>", txt):
                    out.append(f"{n}: chart data contains nan / inf")
    return out


def drop_empty_slides(prs):
    """Remove content slides that have only a title (e.g. out-of-sample slides of a run without run_oos.py)."""
    ids = prs.slides._sldIdLst
    for sid, slide in list(zip(list(ids), prs.slides)):
        if slide.slide_layout.name in ("Title Slide", "Section Header"):
            continue
        body = [s for s in slide.shapes if not (s.is_placeholder and s.placeholder_format.type in (1, 3))]  # not the title
        filled = [s for s in body if (s.has_text_frame and s.text_frame.text.strip()) or not s.has_text_frame]
        if not filled:
            prs.part.drop_rel(sid.rId)
            ids.remove(sid)


def main(path):
    """Post-render step: restyle every slide of the rendered deck and save it in place."""
    meta_path = os.path.join(HERE, "_charts", "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    global FOOT
    FOOT = meta.get("footer", FOOT)
    prs = Presentation(path)
    drop_empty_slides(prs)
    for i, slide in enumerate(prs.slides, start=1):
        name = slide.slide_layout.name
        if name == "Title Slide":
            process_title(slide, meta)
        elif name == "Section Header":
            process_section(slide)
        else:
            process_slide(slide, i, meta)
    fix_empty_text(prs)
    prs.save(path)
    problems = check_package(path)
    for p_ in problems:
        print(f"WARNING (PowerPoint may ask to repair the file): {p_}")
    print(f"styled {path}")


if __name__ == "__main__":
    if os.environ.get("STYLE_DECK_SKIP") == "1":           # for debugging: keep Quarto's plain output
        sys.exit(0)
    out = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "").split("\n")
    targets = sys.argv[1:] or [f for f in out if f.endswith(".pptx")] or [os.path.join(HERE, "forecast_deck.pptx")]
    for t in targets:
        main(t if os.path.isabs(t) else os.path.join(os.getcwd(), t))
