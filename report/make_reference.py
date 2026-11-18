"""
Build report/reference.pptx, the PowerPoint template Quarto uses for forecast_deck.qmd.

    python report/make_reference.py

Starts from Quarto's default template (`quarto pandoc -o default_reference.pptx --print-default-data-file
reference.pptx`) and applies the deck's look: deep-teal title and section slides, serif headings (Georgia),
sans-serif body text (Calibri), the accent colours used in the charts, and a footer line. To use your
organisation's template instead, point `reference-doc:` in the .qmd at it.
"""
import copy
import os
import sys

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
A = "{%s}" % NS["a"]
NAVY, INK, INK2, LIGHT = "0E3B4F", "1B2A41", "8796A8", "EEF3F6"
THEME = {"dk1": INK, "lt1": "FFFFFF", "dk2": NAVY, "lt2": LIGHT, "accent1": "1F6F8B", "accent2": "E3A72F",
         "accent3": "2E9E83", "accent4": "C2553F", "accent5": "8796A8", "accent6": "5B4B8A",
         "hlink": "1F6F8B", "folHlink": "5B4B8A"}
FOOTER = "Macro-driver forecasting model"


def set_theme(prs):
    """Set the theme colours and fonts of the reference deck."""
    part = prs.slide_master.part.part_related_by(
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme")
    root = etree.fromstring(part.blob)
    cs = root.find(".//a:clrScheme", NS)
    for name, val in THEME.items():
        el = cs.find(f"a:{name}", NS)
        for c in list(el):
            el.remove(c)
        etree.SubElement(el, A + "srgbClr", val=val)
    fs = root.find(".//a:fontScheme", NS)
    fs.find("a:majorFont/a:latin", NS).set("typeface", "Cambria")
    fs.find("a:minorFont/a:latin", NS).set("typeface", "Calibri")
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def run_props(parent, size=None, color=None, bold=None):
    """Set a:defRPr inside a lstStyle level (creating elements as needed)."""
    d = parent.find("a:defRPr", NS)
    if d is None:
        d = etree.SubElement(parent, A + "defRPr")
    if size:
        d.set("sz", str(int(size * 100)))
    if bold is not None:
        d.set("b", "1" if bold else "0")
    d.set("cap", "none")
    if color:
        for c in d.findall("a:solidFill", NS):
            d.remove(c)
        f = etree.Element(A + "solidFill")
        etree.SubElement(f, A + "srgbClr", val=color)
        d.insert(0, f)


def style_placeholder(ph, size=None, color=None, bold=None, align=None):
    """Set font size, colour, weight and alignment of a placeholder's text."""
    tx = ph._element.find(".//p:txBody", NS)
    lst = tx.find("a:lstStyle", NS)
    if lst is None:
        lst = etree.SubElement(tx, A + "lstStyle")
        tx.remove(lst)
        tx.insert(1, lst)
    lvl = lst.find("a:lvl1pPr", NS)
    if lvl is None:
        lvl = etree.SubElement(lst, A + "lvl1pPr")
    if align:
        lvl.set("algn", align)
    run_props(lvl, size, color, bold)


def master_text(prs):
    """Set the master title and body text styles."""
    m = prs.slide_master._element
    ts = m.find(".//p:titleStyle/a:lvl1pPr", NS)
    ts.set("algn", "l")
    run_props(ts, size=20, color=INK, bold=True)
    bs = m.find(".//p:bodyStyle", NS)
    for i, sz in enumerate([14, 13, 12, 11, 11], start=1):
        lvl = bs.find(f"a:lvl{i}pPr", NS)
        if lvl is not None:
            run_props(lvl, size=sz, color=INK)


def footer(prs):
    """Footer text on the master. Built on a temporary slide, then copied (masters cannot add text boxes)."""
    tmp = prs.slides.add_slide(prs.slide_layouts[6])
    tb = tmp.shapes.add_textbox(Emu(457200), prs.slide_height - Emu(330000), Emu(5000000), Emu(250000))
    p = tb.text_frame.paragraphs[0]
    p.text = FOOTER
    p.runs[0].font.size = Pt(8)
    p.runs[0].font.color.rgb = RGBColor.from_string(INK2)
    prs.slide_master.shapes._spTree.append(copy.deepcopy(tb._element))
    sid = prs.slides._sldIdLst[-1]
    prs.part.drop_rel(sid.rId)
    prs.slides._sldIdLst.remove(sid)


def dark_layout(layout):
    """Turn a layout into a dark (navy) layout for title and section slides."""
    bg = layout.background.fill
    bg.solid()
    bg.fore_color.rgb = RGBColor.from_string(NAVY)
    layout._element.set("showMasterSp", "0")
    for ph in layout.placeholders:
        t = ph.placeholder_format.type
        idx = ph.placeholder_format.idx
        if idx in (10, 11, 12):
            continue
        is_title = idx == 0
        style_placeholder(ph, size=34 if is_title else 16, color="FFFFFF" if is_title else "C9DCE6",
                          bold=is_title, align="l")


def main(default_path, out_path):
    """Build report/reference.pptx from Quarto's default reference deck."""
    prs = Presentation(default_path)
    set_theme(prs)
    master_text(prs)
    # footer, slide number and data note are added by style_deck.py
    for layout in prs.slide_layouts:
        if layout.name in ("Title Slide", "Section Header"):
            dark_layout(layout)
    prs.save(out_path)
    print(out_path)


def default_reference(path):
    """Write Pandoc's default PowerPoint reference deck to path (through Quarto's bundled Pandoc, or Pandoc).

    Args:
        path: where to write default_reference.pptx.

    Returns:
        path.

    Raises:
        RuntimeError if neither quarto nor pandoc is installed.
    """
    import shutil
    import subprocess
    for cmd in (["quarto", "pandoc"], ["pandoc"]):
        if shutil.which(cmd[0]):
            with open(path, "wb") as f:
                r = subprocess.run(cmd + ["--print-default-data-file", "reference.pptx"], stdout=f)
            if r.returncode == 0 and os.path.getsize(path) > 0:
                print(f"Created {path} with {' '.join(cmd)}")
                return path
            os.remove(path)                                  # do not leave an empty file behind
    raise RuntimeError("Neither quarto nor pandoc was found: install Quarto, or pass a reference .pptx as argument")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "default_reference.pptx")
    if not os.path.exists(src) or os.path.getsize(src) == 0:
        default_reference(src)
    main(src, os.path.join(HERE, "reference.pptx"))
