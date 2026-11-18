
# Windows consoles default to a code page (cp1252, "charmap") that cannot show characters such as ⑤, →, ≤ or ×.
# Print UTF-8 instead, and replace anything the console still cannot show rather than stopping the run.
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:          # not a normal console (Jupyter, Streamlit, a pipe that cannot be reconfigured)
        pass
