"""Panel analysis: one target (e.g. a loan balance) for several banks, with common macro drivers.

    data.py      read the long file (date, bank, balance), per-bank organic change, the stacked panel
    estimate.py  pooled fixed effects, separate per-bank regressions, partial pooling (shrinkage), top-down share
    select.py    driver selection on the pooled panel (Driscoll-Kraay screen, then the same search rules)
    run.py       backtest, out-of-sample test with re-selection, scenarios, tables, charts, saved models
"""
