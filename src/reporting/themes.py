"""Visual themes for the process story (PowerPoint and HTML) and its charts.

    python tools/process_story.py --config <cfg> --theme corporate        # one theme
    python tools/process_story.py --config <cfg> --theme all              # every theme, side by side
    python tools/process_story.py --list-themes

A theme is a palette (hex colours without '#'), two fonts for the deck and two font stacks for the page.
Colour roles:
    bg         slide / page background        card     section card (HTML), chart background
    ink        body text                      mid      secondary text (labels, footers, notes)
    title      slide and section titles       kicker   small labels above titles, "Why", "What we did"
    bar        thin bar at the top of slides  highlight  the takeaway bar, key numbers on the cover
    tile       background of number tiles     tile_value  the number in a tile
    line       borders and grid lines         alt      alternate table rows
    head_bg    table header background        head_fg  table header text
    cover_bg   cover and divider background   cover_fg / cover_sub  their title and text
    nav_bg     the HTML contents sidebar      nav_fg   its links
    chart      the categorical chart colours, in order (models, scenarios, series)
"""

THEMES = {
    "classic": dict(
        label="Classic navy", note="Navy, teal and gold with serif titles (the original look).",
        head="Cambria", body="Calibri", css_head='Cambria,Georgia,serif',
        css_body='-apple-system,"Segoe UI",Calibri,Roboto,sans-serif',
        bg="FFFFFF", card="FFFFFF", page="F6F8FA", ink="1B2A41", mid="4A5866", title="0E3B4F", kicker="1F6F8B",
        bar="0E3B4F", highlight="E3A72F", tile="EEF3F6", tile_value="0E3B4F", line="D9E1E8", alt="F7FAFB",
        head_bg="0E3B4F", head_fg="FFFFFF", cover_bg="0E3B4F", cover_fg="FFFFFF", cover_sub="C9DCE6",
        nav_bg="0E3B4F", nav_fg="C9DCE6",
        chart=["1F6F8B", "E3A72F", "2E9E83", "C2553F", "5B4B8A", "C77B30", "8796A8", "0F8A8A"]),
    "corporate": dict(
        label="Corporate blue", note="White slides, strong blue headings and an orange accent; Arial throughout.",
        head="Arial", body="Arial", css_head='Arial,"Helvetica Neue",Helvetica,sans-serif',
        css_body='Arial,"Helvetica Neue",Helvetica,sans-serif',
        bg="FFFFFF", card="FFFFFF", page="F3F5F8", ink="222B36", mid="5F6B7A", title="1F4E79", kicker="2E75B6",
        bar="1F4E79", highlight="ED7D31", tile="EAF1F8", tile_value="1F4E79", line="D6DEE8", alt="F6F9FC",
        head_bg="1F4E79", head_fg="FFFFFF", cover_bg="1F4E79", cover_fg="FFFFFF", cover_sub="BDD7EE",
        nav_bg="FFFFFF", nav_fg="1F4E79",
        chart=["2E75B6", "ED7D31", "A5A5A5", "FFC000", "5B9BD5", "70AD47", "264478", "9E480E"]),
    "midnight": dict(
        label="Midnight (dark)", note="Dark slides with cyan and amber accents; good on screens, not for print.",
        head="Calibri Light", body="Calibri", css_head='"Segoe UI Semibold","Segoe UI",Calibri,sans-serif',
        css_body='"Segoe UI",Calibri,Roboto,sans-serif',
        bg="0F172A", card="1E293B", page="0B1120", ink="E2E8F0", mid="94A3B8", title="F8FAFC", kicker="22D3EE",
        bar="22D3EE", highlight="F59E0B", tile="1E293B", tile_value="22D3EE", line="334155", alt="172033",
        head_bg="0E7490", head_fg="F8FAFC", cover_bg="020617", cover_fg="F8FAFC", cover_sub="94A3B8",
        nav_bg="020617", nav_fg="94A3B8", box2="155E75",
        chart=["22D3EE", "F59E0B", "34D399", "F87171", "A78BFA", "FB923C", "94A3B8", "2DD4BF"]),
    "minimal": dict(
        label="Minimal mono", note="Black, white and grey with a single red accent; quiet and print-friendly.",
        head="Helvetica", body="Helvetica", css_head='"Helvetica Neue",Helvetica,Arial,sans-serif',
        css_body='"Helvetica Neue",Helvetica,Arial,sans-serif',
        bg="FFFFFF", card="FFFFFF", page="FFFFFF", ink="111111", mid="6B6B6B", title="111111", kicker="D7263D",
        bar="111111", highlight="D7263D", tile="F2F2F2", tile_value="111111", line="DDDDDD", alt="FAFAFA",
        head_bg="111111", head_fg="FFFFFF", cover_bg="FFFFFF", cover_fg="111111", cover_sub="6B6B6B",
        nav_bg="FAFAFA", nav_fg="333333",
        chart=["111111", "D7263D", "8C8C8C", "3D3D3D", "BDBDBD", "E07A5F", "5C5C5C", "A8A8A8"]),
    "forest": dict(
        label="Forest", note="Deep greens with an ochre accent and Georgia titles; calm and warm.",
        head="Georgia", body="Calibri", css_head='Georgia,"Times New Roman",serif',
        css_body='Calibri,"Segoe UI",Roboto,sans-serif',
        bg="FFFFFF", card="FFFFFF", page="F4F6F1", ink="1F2A1F", mid="56614F", title="1E4D2B", kicker="4F7942",
        bar="1E4D2B", highlight="C9A227", tile="EDF2E8", tile_value="1E4D2B", line="D7DFD0", alt="F7F9F4",
        head_bg="1E4D2B", head_fg="FFFFFF", cover_bg="1E4D2B", cover_fg="FFFFFF", cover_sub="CFE0C3",
        nav_bg="1E4D2B", nav_fg="CFE0C3",
        chart=["4F7942", "C9A227", "8C5E3C", "2F6F73", "A3B18A", "B5651D", "6B705C", "344E41"]),
    "boardroom": dict(
        label="Boardroom", note="Burgundy and charcoal with a brass accent and Garamond titles; formal.",
        head="Garamond", body="Calibri", css_head='Garamond,"EB Garamond","Times New Roman",serif',
        css_body='Calibri,"Segoe UI",Roboto,sans-serif',
        bg="FFFFFF", card="FFFFFF", page="F7F4F2", ink="2B2A2A", mid="62595A", title="5B1A2E", kicker="8C2F39",
        bar="5B1A2E", highlight="B08D57", tile="F4ECEA", tile_value="5B1A2E", line="E2D6D3", alt="FAF6F5",
        head_bg="3A3A3A", head_fg="FFFFFF", cover_bg="3A1220", cover_fg="FFFFFF", cover_sub="E2CDB0",
        nav_bg="3A1220", nav_fg="E2CDB0",
        chart=["8C2F39", "B08D57", "3A3A3A", "5E7C8B", "C97C5D", "6B4E71", "9A9A9A", "2F5D50"]),
    "ocean": dict(
        label="Ocean", note="Teal and turquoise with a coral accent; bright and modern.",
        head="Segoe UI Semibold", body="Segoe UI", css_head='"Segoe UI Semibold","Segoe UI",Roboto,sans-serif',
        css_body='"Segoe UI",Roboto,Helvetica,sans-serif',
        bg="FFFFFF", card="FFFFFF", page="F0F7F8", ink="16303A", mid="4E6A73", title="0B6E79", kicker="13A89E",
        bar="13A89E", highlight="F2715A", tile="E3F4F4", tile_value="0B6E79", line="CFE5E7", alt="F5FBFB",
        head_bg="0B6E79", head_fg="FFFFFF", cover_bg="0B6E79", cover_fg="FFFFFF", cover_sub="BFE8E6",
        nav_bg="0B6E79", nav_fg="BFE8E6",
        chart=["13A89E", "F2715A", "0B6E79", "F2A541", "6C8EAD", "8FC1A9", "E85D75", "3D5A80"]),
}
DEFAULT = "classic"


def get(name=None):
    """The theme dict (unknown names raise with the list)."""
    name = (name or DEFAULT).lower()
    if name not in THEMES:
        raise ValueError(f"unknown theme '{name}'; choose from: {', '.join(THEMES)}")
    return dict(THEMES[name], name=name)


def describe():
    return "\n".join(f"  {k:10s} {v['label']}: {v['note']}" for k, v in THEMES.items())


def is_dark(t):
    h = t["bg"]
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) < 128
