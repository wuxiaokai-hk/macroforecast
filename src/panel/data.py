"""Panel data: the long target file (date, bank, balance), each bank's organic change, and the stacked arrays
every panel estimator works on.

One row per bank and quarter. Banks may start at different dates (a bank founded later, a merger that ends
one): each bank uses the quarters it has. One-off events (acquisitions, portfolio sales) are removed per bank
when the events file has the bank column.
"""
import copy
import os

import numpy as np
import pandas as pd

from ..utils import target as tg
from ..utils.io import Context, read_table, to_quarters


def load_long(cfg):
    """The panel target file as one level series per bank.

    Args:
        cfg: configuration (data.bank.file, date_column, target_column, panel_column, sample.hist_end,
            panel.banks to keep a subset).

    Returns:
        dict bank name -> Series of levels indexed by quarter (sorted, up to hist_end, no missing values).
    """
    b = cfg["data"]["bank"]
    pc = b.get("panel_column")
    df = read_table(b["file"], b.get("sheet"))
    for c in (b["date_column"], b["target_column"], pc):
        if c not in df.columns:
            raise KeyError(f"panel file {b['file']}: column '{c}' not found; columns are {list(df.columns)}")
    df = df[[b["date_column"], pc, b["target_column"]]].copy()
    df["q"] = to_quarters(df[b["date_column"]])
    df["v"] = pd.to_numeric(df[b["target_column"]], errors="coerce")
    df[pc] = df[pc].astype(str).str.strip()
    dup = df.duplicated(["q", pc], keep=False)
    if dup.any():
        if str(b.get("frequency", "Q")).upper() == "M":       # monthly: the last month of each quarter
            df = df.sort_values(b["date_column"]).groupby([pc, "q"], as_index=False).last()
        else:
            ex = df.loc[dup, [pc, "q"]].head(3).astype(str).values.tolist()
            raise ValueError(f"panel file has several rows for the same bank and quarter, e.g. {ex}")
    keep = [str(x) for x in ((cfg.get("panel") or {}).get("banks") or [])]
    end = pd.Period(cfg["sample"]["hist_end"], freq="Q")
    out = {}
    for name, g in df.groupby(pc, sort=True):
        if keep and name not in keep:
            continue
        s = g.set_index("q")["v"].sort_index()
        s = s[~s.index.duplicated(keep="last")].loc[:end].dropna()
        if len(s) >= 2:
            out[name] = s
    if not out:
        raise ValueError("no bank with at least two quarters of data")
    return out


def bank_events(cfg, bank):
    """One-off events of one bank: {quarter: amount}. The events file may have the panel column (one row per
    bank) or not (then rows whose series is '<bank>' or '<target>|<bank>' are used)."""
    e = cfg["data"]["events"]
    if not e.get("file") or not os.path.exists(e["file"]):
        return {}
    ev = read_table(e["file"], e.get("sheet"))
    tcol = cfg["data"]["bank"]["target_column"]
    bcol = e.get("bank_column") or cfg["data"]["bank"].get("panel_column")
    if bcol in ev.columns:
        ev = ev[(ev[bcol].astype(str).str.strip() == bank) &
                ((ev[e["series_column"]].astype(str) == tcol) if e["series_column"] in ev.columns else True)]
    elif e["series_column"] in ev.columns:
        ev = ev[ev[e["series_column"]].astype(str).isin([bank, f"{tcol}|{bank}"])]
    else:
        return {}
    out = {}
    for q, a in zip(to_quarters(ev[e["date_column"]]), pd.to_numeric(ev[e["amount_column"]], errors="coerce")):
        if pd.notna(a):
            out[q] = out.get(q, 0.0) + float(a)
    return out


def organic(level, events):
    """Organic change of one bank: tg.change((L_t - A_t), L_t-1)."""
    y = tg.growth(level)
    for q, a in events.items():
        if q in level.index and q != level.index[0]:
            y.loc[q] = tg.change(level.loc[q] - a, level.shift(1).loc[q])
    return y


def aggregate(levels, ys):
    """The total of all banks and its organic change, computed only over banks present in both quarters
    (so a bank entering or leaving the panel is not a jump in the total).

    Returns:
        (total level Series, organic change Series).
    """
    idx = sorted(set().union(*[s.index for s in levels.values()]))
    idx = pd.PeriodIndex(idx, freq="Q")
    L = pd.DataFrame({k: v.reindex(idx) for k, v in levels.items()})
    Y = pd.DataFrame({k: v.reindex(idx) for k, v in ys.items()})
    # balances add up; for a rate the "total" is the average across banks
    total = L.sum(axis=1, min_count=1) if tg.is_log() else L.mean(axis=1)
    g = pd.Series(np.nan, index=idx)
    for t in range(1, len(idx)):
        both = L.iloc[t].notna() & L.iloc[t - 1].notna()
        if not both.any():
            continue
        prev = L.iloc[t - 1][both].sum()
        if tg.is_log():
            # organic level now = previous level grown by each bank's organic change
            now = (L.iloc[t - 1][both] * np.exp(Y.iloc[t][both].fillna(0))).sum()
            g.iloc[t] = np.log(now / prev)
        else:
            g.iloc[t] = float(Y.iloc[t][both].fillna(0).mean())      # rates: average change across banks
    return total, g


class Panel:
    """Everything the panel estimators need.

    Attributes:
        cfg, banks (names), levels / ys (per bank), base (Context of the total: macro data, metadata, dates,
        dummies, scenarios), ctx (per-bank Context copies), dates (all quarters), and the stacked arrays
        B (bank code), T (position in dates), Y, YL (last quarter's change) for every bank-quarter that has both.
    """

    def __init__(self, cfg, out_dir, log=print):
        self.cfg = cfg
        levels = load_long(cfg)
        # the target form (log growth or change) is decided on all banks together
        allv = pd.concat(list(levels.values()))
        tcfg = cfg.setdefault("target", {})
        tcfg["form"] = tg.resolve(tcfg.get("transform", "auto"), allv)
        tg.set_form(tcfg["form"])
        self.levels = levels
        self.ys = {k: organic(v, bank_events(cfg, k)) for k, v in levels.items()}
        total, gagg = aggregate(levels, self.ys)
        # a Context for the total: macro data, metadata, literature, dates and dummies come from it
        inp = os.path.join(out_dir, "inputs")
        os.makedirs(inp, exist_ok=True)
        f = os.path.join(inp, "panel_total.csv")
        pd.DataFrame({"date": [str(q) for q in total.index], "total": total.values}).to_csv(f, index=False)
        c2 = copy.deepcopy(cfg)
        c2["data"]["bank"].update(file=f, target_column="total", date_column="date", sheet=None, frequency="Q")
        c2["data"]["bank"].pop("panel_column", None)
        c2["data"]["events"]["file"] = None
        c2.setdefault("target", {})["transform"] = tcfg["form"]
        base = Context(c2)
        base.y = gagg.reindex(base.idx)
        base.logL = tg.state(total.iloc[0]) + base.y.fillna(0).cumsum()
        base.cfg = cfg
        self.base = base
        self.total = total
        self.dates = base.idx
        pos = {q: i for i, q in enumerate(self.dates)}
        self.banks = sorted(levels)
        self.code = {b: i for i, b in enumerate(self.banks)}
        self.ctx = {b: self._bank_ctx(b) for b in self.banks}
        B, T, Y, YL = [], [], [], []
        for b in self.banks:
            y = self.ys[b]
            for q in y.index:
                p = pos.get(q)
                if p is None or pd.isna(y.loc[q]):
                    continue
                prev = y.get(q - 1, np.nan)
                if pd.isna(prev):
                    continue
                B.append(self.code[b])
                T.append(p)
                Y.append(float(y.loc[q]))
                YL.append(float(prev))
        self.B, self.T = np.array(B, int), np.array(T, int)
        self.Y, self.YL = np.array(Y, float), np.array(YL, float)
        dq = [d for d in base.dummies.columns if base.dummies[d].std() > 0]
        self.dummy_names = dq
        self.Dm = base.dummies[dq].values if dq else np.zeros((len(self.dates), 0))
        log(f"  panel: {len(self.banks)} banks, {len(self.Y)} bank-quarters "
            f"({', '.join(f'{b} {self.levels[b].index[0]}-{self.levels[b].index[-1]}' for b in self.banks[:6])}"
            f"{', ...' if len(self.banks) > 6 else ''}); target form {tg.form()}")

    def _bank_ctx(self, b):
        """A Context for one bank: the total's macro data and settings, the bank's own level and change."""
        c = copy.copy(self.base)
        lvl, y = self.levels[b], self.ys[b]
        c.level, c.y = lvl, y
        c.logL = tg.state(lvl.iloc[0]) + y.fillna(0).cumsum()
        c.idx = y.index
        c.hist = self.base.base.reindex(c.idx)
        est = pd.Period(self.cfg["sample"]["estimation_start"], freq="Q")
        c.est0 = int(np.searchsorted(c.idx.asi8, max(est, c.idx[1]).ordinal))
        hist_end = pd.Period(self.cfg["sample"]["hist_end"], freq="Q")
        c.end = int(np.where(c.idx == hist_end)[0][0]) if hist_end in c.idx else len(c.idx) - 1
        c.dummies = self.base.dummies.reindex(c.idx).fillna(0.0)
        c.bank = b
        return c

    def pos(self, label):
        """Position of a quarter in self.dates."""
        return int(self.dates.get_loc(pd.Period(label, freq="Q")))

    def window(self, start, end, X=None):
        """Mask of stacked rows inside [start, end] (labels or positions) with complete features X."""
        a = self.pos(start) if not isinstance(start, (int, np.integer)) else int(start)
        z = self.pos(end) if not isinstance(end, (int, np.integer)) else int(end)
        m = (self.T >= a) & (self.T <= z)
        if X is not None and X.shape[1]:
            m &= ~np.isnan(X).any(axis=1)
        return m

    def active(self, label):
        """Banks with data at a quarter (to forecast from it)."""
        q = pd.Period(label, freq="Q")
        return [b for b in self.banks if q in self.levels[b].index]
