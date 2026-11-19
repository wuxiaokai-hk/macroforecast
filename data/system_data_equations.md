# Simulated system (data/system_data.csv): the equations that generated it

Quarterly, 2000Q1-2025Q4, from the history of the example scenario baseline (data/scenario_baseline.csv).
e = independent normal noise (std. dev. in brackets). Δ = change from the previous quarter. g = log growth.

| Target | Equation |
|---|---|
| mortgage_rate | mr_t = mr_t-1 + 0.5 [(1.4 + 0.9 GoC5Y_t) - mr_t-1] + e (0.18) |
| deposit_rate | dr_t = max(0.05, dr_t-1 + 0.4 [(0.1 + 0.55 BoC_t-1) - dr_t-1] + e (0.12)) |
| mortgage | g_t = 0.003 + 0.30 g_t-1 + 0.12 g(house prices)_t-1 + 0.25 g(disposable income)_t-1 - 0.010 Δmr_t-1 + e (0.003) |
| commercial | g_t = 0.002 + 0.25 g_t-1 + 0.40 g(real GDP)_t + 0.15 g(business investment)_t-1 - 0.003 Δprime_t-1 + e (0.004) |
| deposits | g_t = 0.0005 + 0.25 g_t-1 + 0.30 g(disposable income)_t-1 + 0.010 Δ(dr - BoC)_t-1 + 0.35 g(mortgage)_t-1 + e (0.003) |

Links between targets: the mortgage rate lowers mortgage growth (-1% a quarter per 1 pp rise, a quarter later);
the deposit spread over the policy rate raises deposit growth; mortgage growth raises deposit growth a quarter
later (loans create deposits). Identity: total loans = mortgage + commercial.
