# macroforecast: explainable forecasts of bank financial performance from macro drivers

A complete, reusable method for forecasting any quarterly financial measure of a bank, a credit union or the banking sector from the economy, under scenarios, with models you can explain driver by driver. Bring your own target series and public macro data; the method does the rest:

1. **Prepares the target:** organic quarterly growth (or change, for rates and ratios), one-off events removed.
2. **Builds candidates:** every macro series in economically sensible forms (growth, annual change, acceleration, gap from trend, ...) and lags, each with the sign economic theory expects.
3. **Chooses a few drivers** (at most 3 by default) in two independent ways, screening one at a time and an Elastic Net with stability selection, followed by the same final search: significant, right-signed, not collinear, and ranked by forecast accuracy, not fit.
4. **Competes:** the chosen models against benchmarks, published specifications from the literature, theory-data hybrids, regime-switching, time-series and machine-learning models, about 30 models in all.
5. **Tests honestly:** a rolling backtest 1–12 quarters ahead, an out-of-sample test that re-chooses the drivers at every origin, and a holdout of the latest quarters.
6. **Forecasts every scenario** and explains each forecast quarter as a sum of named pieces, with a deck, a step-by-step process story, an Excel calculator and the full tables.

It also handles **many banks at once** (panel models that pool across banks), **several related targets as one system** (loans, deposits and their rates, with SUR, VECM and BVARX), and **many targets from one file**.

## What you can forecast

Anything observed quarterly (or monthly) for which economic drivers make sense. Balances and volumes are modelled as log growth; rates, ratios and shares as quarterly changes (`target.transform: diff`, suggested automatically).

| Area | Examples | Form |
|---|---|---|
| Lending | residential mortgages, HELOCs, personal, credit card, commercial, CRE, agricultural loans; undrawn commitments | growth |
| Funding and liquidity | demand and term deposits, wholesale funding, liquid assets, securities | growth |
| Pricing | mortgage, loan, deposit and GIC rates; spreads; net interest margin | change |
| Income and expense | interest income and expense, fee income, wealth and insurance income, salaries | growth |
| Credit quality | delinquency rates, Stage 2 share, provision rate, net write-off rate, allowance coverage | change |
| Capital | RWA, CET1 / Tier 1 / Tier 2 capital, capital ratios | growth or change |

A literature master (`config/literature_master_example.xlsx`, or your own built with `tools/literature_master.py`) holds, for each target, the economic concepts the research identifies with their evidence and expected effect, the published model specifications, and the expected sign of every macro series, so a new target starts from theory, not from a blank page.

## What you need

| Input | What it is | Typical source |
|---|---|---|
| **Target series** | date + one or more columns (one row per bank and quarter for several banks) | your institution's general ledger or regulatory returns; public regulatory data (below) |
| **Macro history and scenarios** | one file per scenario: the history of each macro series followed by the scenario path, same columns in every file | official statistics for the history; a scenario provider, a supervisor's published scenarios or your own (below) |
| **Metadata** | one row per macro series: code, description, geography, theme, natural transform, expected sign | written once for your macro data; `tools/new_data_setup.py` drafts it |
| **Literature workbook or master** (recommended) | the published models and economic concepts for the target, mapped to your series | `config/literature_master_example.xlsx`, or your own master |
| **One-off events** (optional) | acquisitions, portfolio sales, reclassifications to take out of the growth | your institution |

## Where to find the data

The method needs no proprietary data. These public sources cover the targets and the drivers; check each source's terms of use before redistributing what you download.

**Targets: bank and banking-sector financials**

- **Canada, by institution:** OSFI's financial data on the Open Government portal (FINDAT) has the regulatory returns banks file, among them the M4 consolidated balance sheet, the P3 income statement and BCAR capital adequacy; monthly data are published on the 15th of each month and quarterly data in late May, August, November and March ([OSFI](https://www.osfi-bsif.gc.ca/en/data-forms/open-government-osfi-financial-data/working-open-government-data)). The balance sheet return is published by institution and, as a total for all banks, in the Bank of Canada's Banking and Financial Statistics ([OSFI, M4](https://osfi-bsif.gc.ca/en/node/1153)).
- **Canada, banking sector:** the Bank of Canada's Banking and Financial Statistics tables (chartered bank assets and liabilities, regional distribution of loans and deposits, credit aggregates); many are also Statistics Canada tables, e.g. *Chartered banks, assets and liabilities, month-end* ([Open Government](https://open.canada.ca/data/en/dataset/53a00090-9ad7-4915-bf99-5e63489596ac)). Federally regulated credit unions are excluded from these chartered-bank tables ([Bank of Canada](https://www.bankofcanada.ca/?p=194601)).
- **Credit unions (Canada):** your provincial regulator's published system data, or your own returns.
- **United States:** quarterly Call Reports from the FFIEC Central Data Repository (bulk files for all filers about 45 days after each quarter end) and the FDIC BankFind Suite and its API ([FDIC bank data guide](https://www.fdic.gov/bank-data-guide/data-downloads); [FFIEC CDR](https://cdr.ffiec.gov/Public/HelpFiles/WelcomeAdditionalInfo.htm)); NCUA 5300 call reports for credit unions.
- **Your own institution:** the general ledger, the data warehouse or the returns you file are the best target data, because they match exactly what you report.

**Drivers: macro and financial history**

- **Statistics Canada** tables, by download (CSV) or through the Web Data Service API ([WDS](https://statcan.gc.ca/eng/developers/wds)). Useful starting points: GDP by expenditure, quarterly (36-10-0104); household current and capital accounts, including disposable income (36-10-0112); household credit market debt, mortgage and consumer credit (38-10-0238); Labour Force Survey by province (14-10-0287); CPI (18-10-0004); plus housing, retail, building permits and population tables, at national, provincial and CMA level.
- **Bank of Canada:** the policy rate, prime, Government of Canada yields, exchange rates and thousands of other series through the free Valet API (CSV, JSON or XML), including the staff economic projections ([Bank of Canada](https://www.bankofcanada.ca/rates/staff-economic-projections/)).
- **United States and international:** FRED (Federal Reserve Bank of St. Louis), the BEA and BLS.

**Scenarios: where the macro paths come from**

- **Supervisory scenarios:** the Federal Reserve publishes its stress-test scenarios each year as CSV files, a baseline and a severely adverse scenario for domestic and international variables ([Federal Reserve, 2026 stress tests](https://www.federalreserve.gov/supervisionreg/dfa-stress-tests-2026.htm)); the 2026 scenarios run from the first quarter of 2026 to the first quarter of 2029 with 28 variables ([OMB filing](https://omb.report/icr/202603-1557-002/doc/original/166956500.pdf)). Use them directly for US targets, or as the shape of a stress for others.
- **Commercial providers** (Moody's Analytics, Oxford Economics, S&P Global and others) sell consistent baseline and alternative scenarios for thousands of national and regional series, under licence. The example files in `data/` are laid out like such a vendor file (one column per series, Moody's-style codes).
- **Your own:** the baseline from your economics team or the Bank of Canada's projections, and alternatives built by shifting the key paths (rates up 200 bp, unemployment +3 points, house prices −20%). Each scenario file is the same history followed by its own path.

Whatever the source, the history in every scenario file must be the same actual data, and every series the metadata marks as a possible driver must have a path in every scenario.

## From downloads to the files the project reads

1. **Target:** a CSV or Excel file with a date column (`2020Q1` or `2020-03-31`) and one column per target series. Monthly data are turned into quarter ends. Several banks: add a bank column.
2. **Scenarios:** one file per scenario with a date column and one column per macro series (use short codes; the metadata describes them). History and scenario path in the same column, one after the other.
3. **Metadata:** one row per series with its code, description, geography, theme, natural transform (`dlog` for levels, `diff` for rates) and expected sign (+1, −1, or 0 to keep it out of the selection).
4. Run `python tools/new_data_setup.py` (or the app, `streamlit run app.py`): it reads the files, detects the columns, drafts the missing metadata fields, maps the literature concepts to your series, sets every date from the data and writes the config. `python tools/check_config.py` checks it before a run.
5. Run (below). The process story (`tools/process_story.py`) writes up what happened for your audience.

For a real data set to start from, `tools/fetch_statcan.py` (next section) does all of this for public Statistics Canada series.

## Real example data from Statistics Canada (`python tools/fetch_statcan.py`)

One command downloads a real example data set from Statistics Canada and writes everything a run needs:

```
python tools/fetch_statcan.py                                   # download, build the files, write config/statcan.yaml
python run_targets.py --config config/statcan.yaml --list       # the targets it built
python run_targets.py --config config/statcan.yaml --all --oos --story
python tools/fetch_statcan.py --describe 14100287               # a table's dimensions and members
```

- **Targets:** household mortgage debt and consumer credit outstanding (table 38-10-0238, quarterly).
- **Drivers:** real GDP (36-10-0104), household disposable income (36-10-0112), unemployment rate and employment for Canada and Ontario (14-10-0287), CPI (18-10-0004), new housing price index for Canada and Ontario (18-10-0205), retail sales (20-10-0008), the prime rate, the bank rate, the 5-year conventional mortgage rate and the 5-year Government of Canada yield (10-10-0122), and Ontario population (17-10-0009).
- **How:** the tool uses Statistics Canada's free Web Data Service (no key). For each series it reads the table's dimensions, turns the member names in `config/statcan_series.yaml` into the series' coordinate and downloads it; monthly series become quarterly (average, or quarter end for stocks). Downloads are cached in `data/statcan/raw/`, so a re-run works offline (`--refresh` downloads again).
- **Files written:** `data/statcan/targets.csv`, one scenario file per scenario in `data/statcan/scenarios/`, `data/statcan/macro_metadata.csv` (with expected signs), `data/statcan/sources.csv` (table, coordinate, vector and title of every series, with the download date) and `config/statcan.yaml` with every date set from the data and the literature concepts mapped to these series, through the same set-up as `tools/new_data_setup.py`.
- **Scenarios:** Statistics Canada publishes history, not scenarios, so the scenario paths are illustrative and rule-based: the baseline continues each series' average growth of the last 20 quarters (rates stay flat), and Upside, Downside and Severe add shocks that build up over 4–5 quarters and half fade by the end (Severe: real GDP −5%, Ontario house prices −22%, unemployment +4 points, rates −1 to −2 points). Edit them in the spec, or replace the files with a provider's scenarios for real use.
- **Changing the series:** add or edit entries in `config/statcan_series.yaml` (table number, member names per dimension, transform, expected sign). Member names can be exact or a list of words that must appear; a name that matches nothing stops the tool with the list of members of that dimension. `--describe` prints a table's dimensions and members.
- Needs internet access to `www150.statcan.gc.ca` (behind a corporate proxy, set `HTTPS_PROXY`) and no extra packages. Statistics Canada data are used under the Statistics Canada Open Licence.

**The small example files in `data/` that ship with the project are simulated**, so the project runs without internet and its results can be checked against the equations that generated them (`data/*_equations.md`, `tools/simulate_data.py`). For real data, run `tools/fetch_statcan.py` or bring your own as above.

## Run

```
python run_pipeline.py                                    # example data and config
python run_pipeline.py --config config/my_portfolio.yaml  # your data
python run_pipeline.py --config config/my_portfolio.yaml --approach en_first   # one approach only
```

Missing Python packages are installed automatically the first time. A full run takes about a minute (the machine-learning models take most of it); re-runs with nothing changed take seconds.

## As a package (`macroforecast/`): fit, leaderboard, predict

Install once from the project folder (editable, so it keeps using this folder's `data/`, `config/` and `outputs/`):

```
pip install -e .
```

Then, from any folder or notebook, in the style of AutoGluon:

```python
from macroforecast import Forecaster

fc = Forecaster(target="loan_balance", path="outputs/mortgages")      # any column: deposits, investment, a rate ...
fc.fit(data="data/bank_data.csv",                                       # date column + target (csv/xlsx), or a Series
       metadata="data/macro_metadata.csv",
       literature="config/literature_workbook.xlsx",
       presets="medium",                                                # fast | medium | best | full
       hyperparameters={"sample.horizon": 8, "en_first": {"target_n": 8}})   # any config setting

fc.leaderboard()                       # every model ranked (out of sample when run, else backtest)
fc.predict()                           # scenario forecasts of the best model (level and quarterly change)
fc.predict(macro="new_scenario.csv")   # a NEW scenario with the fitted model, nothing re-estimated
fc.coefficients()                      # every coefficient with standard error, t and p value
fc.fitted_values()                     # in-sample fit and residuals of every model
fc.model("Screen first")               # a fitted model object; fc.models() for all of them
fc.backtest_models("Screen first")     # {origin: model} at every backtest re-estimation
fc.oos_models()                        # {origin: {model: model}} at every out-of-sample origin
fc = Forecaster.load("outputs/mortgages")   # later, without running anything
```

**`hyperparameters`** (alias `settings`) changes any YAML setting without editing the YAML: a dict with nested or dotted keys, a list of `"key=value"` strings, or a YAML file (`config/options_example.yaml`). It is applied on top of the preset.

**Presets** choose how many model families compete:

| Preset | Models |
|---|---|
| `fast` | benchmarks, data-driven, literature, hybrids, same-growth, auto-ARDL (no levels ARDL) |
| `medium` (default) | `fast` plus ridge, random forest, gradient boosting, ARIMAX, Theta, Markov and threshold regimes |
| `best` | every family, plus deep learning and pretrained models when installed, and moving-average and volatility transforms |
| `full` | exactly the example config |

**Any quarterly target.** `target_form="dlog"` (log growth) suits balances, deposits, investment and prices, i.e. anything that is always positive. `target_form="diff"` (quarterly change) suits interest rates, spreads and ratios. With `"auto"`, log growth is used when every value is positive, so a rate needs `"diff"` set explicitly. With `diff`, the long-run models work in levels without a price deflator, the Excel calculator adds changes instead of compounding them, and every `*_pct` error column is in the target's own units (percentage points for a rate). `label=` and `units=` name the target in titles and the deck. `data/example_deposit_rate.csv` with `data/macro_metadata_deposit_rate.csv` is a worked rate example:

```python
fc = Forecaster(target="deposit_rate", target_form="diff", label="deposit rate", units="%")
fc.fit(data="data/example_deposit_rate.csv", metadata="data/macro_metadata_deposit_rate.csv", presets="fast")
```

Each target needs metadata whose expected signs fit it. A higher policy rate lowers loan growth but raises a deposit rate.

What the defaults do:

- **Sample dates** not set in `hyperparameters` are taken from the data. `hist_end` is the last quarter with data. `estimation_start` is the first quarter with both target and macro data, plus room for lags. `backtest_start` is 8 years before the end.
- **Scenario files** default to the example (simulated) files in `data/`; pass `macro={"Baseline": f1, "S4 Severe": f2}` for your own.
- **No literature workbook** (`literature=None`) switches off the literature, hybrid and same-growth models; the data-driven approaches still run.
- **Same-growth models** keep only the concepts the workbook has, and **constructed series** of the example config are dropped when your macro files lack their inputs.
- **Config record:** each run writes the full config it used to `<path>/run_config.yaml`. `python run_pipeline.py --config <path>/run_config.yaml` repeats it exactly.

The one-call function and the terminal do the same:

```python
import macroforecast as mf
res = mf.run(target="data/bank_data.csv", target_column="loan_balance", metadata=..., literature=...,
             options={"sample.horizon": 8})
```
```
macroforecast --target data/example_deposit_rate.csv --metadata data/macro_metadata_deposit_rate.csv \
              --target-form diff --presets fast --set sample.horizon=8
python -m macroforecast ...        # without installing the command
```

## Several banks at once: panel analysis (`run_panel.py`, `src/panel/`)

When the target file has one row per bank and quarter (`date, bank_name, loan_balance`), the drivers are chosen on all banks together and each bank is forecast by panel models that share information across banks:

```python
fc = Forecaster(target="loan_balance", panel="bank_name")
fc.fit(data="data/panel_bank_data.csv", metadata="data/macro_metadata.csv",
       literature="config/literature_workbook.xlsx", events="data/panel_bank_events.csv")
fc.leaderboard()                    # every model: mean error across banks, banks won, better than AR(1)
fc.leaderboard(bank="Bank E")       # one bank's ranking
fc.best_by_bank()                   # {bank: its best model}
fc.predict(bank="Bank A")           # scenario forecasts (each bank's best model by default)
fc.coefficients(bank="(common)")    # pooled coefficients with Driscoll-Kraay standard errors
fc.shrinkage("Bank E")              # how much each bank keeps its own coefficient
```
or `python run_panel.py --config config/panel_example.yaml` (`data.bank.panel_column: bank_name`).

| Model | Each bank's equation |
|---|---|
| Separate (each bank alone) | estimated on the bank's own history only |
| Pooled FE | one effect of each driver for all banks, a bank intercept (fixed effects) |
| Partial pooling | each bank's coefficients pulled toward the pooled ones by w = τ²/(τ² + se²): a short or noisy history borrows from the others, a long clean one keeps its own; τ² (how much banks really differ) is estimated from the data |
| Top-down (total × share) | the total of all banks from the drivers, times each bank's share, whose change follows its own AR(1) (balances only) |
| AR(1) benchmark | each bank's own past growth |

Each model runs with two driver sets: the **panel drivers** and the **literature core**.

**How the panel drivers are chosen.** The same steps as screen first, run on all banks together:

1. A pooled screen: each candidate alone in a fixed-effects regression with every bank's lagged growth and the dummies.
2. Grouping of correlated candidates, one kept per group.
3. A search with the `final_model` rules (t, signs, VIF, one per theme).
4. Backtest of the combinations with the lowest pooled BIC; the simplest one within the tolerance wins.

All t statistics use **Driscoll–Kraay** standard errors (they match statsmodels' `hac-groupsum`). Every bank is hit by the same shock in the same quarter, so ordinary panel t statistics would overstate significance.

- **Uneven data.** Banks may start or stop at different dates; each uses the quarters it has. The total is computed only over banks present in both quarters, so a bank entering is not a jump. `panel.min_obs` (20) is the number of quarters a bank needs for its own equation; with fewer it is fully pooled.
- **One-off events** are removed per bank when the events file has the bank column (`data.events.bank_column`).
- **Rates.** With `target_form="diff"` the panel works on changes, errors are in the rate's units, and top-down is left out.
- **Out of sample.** The drivers are re-selected on all banks at every origin, as in the single-series test.
- **Banks that stopped reporting.** `panel.drop_exited: true` leaves out every bank whose data end at least `panel.exited_after_missing` quarters (2) before the latest quarter (failed, acquired or exited banks). The run log, `summary.md` and `fitted/manifest.json` list them with their last quarter; the app has the same switch in ③ and shows which banks it would drop. Default `false`: keeping them uses their history, often their downturns, in the estimation, and estimating on surviving banks only tends to understate losses in a severe scenario. In a panel system, set it under `hyperparameters`.
- **Choosing banks.** `panel.banks` lists the banks to estimate (`[]` = all). The panel pools them, so leaving a bank out changes the estimates of the others. In the app, tick them in ② in the table under the bank column ("Estimate").
- **Full single-series run per bank.** `panel.per_bank_full: true` (every bank) or a list of banks, e.g. `[Bank A, Bank C]`, also runs the complete project (every model family) for those banks in `<run>/banks/<bank>/`, to compare with the panel models. In the app, tick "Own full run" in ②.
- **Reusing results.** Each run saves a fingerprint of everything it depends on. The panel result is reused when the data of every chosen bank, the macro, metadata and literature files, the settings and the code are all unchanged (it pools the banks, so one changed bank re-estimates it); editing a bank that is not chosen changes nothing. A bank's own full run is reused when that bank is unchanged, whatever happens to the others. With `--no-oos`, a run reuses an earlier run with or without the out-of-sample test; with it, only one that had it. `python run_panel.py --config <cfg> --status` shows, bank by bank, what would be reused or re-estimated and why; `--force` re-estimates everything. The app shows the same table in ④, with a "re-estimate everything" box.

Results go to `outputs/<run>/panel/`. They include driver selection (`P1`–`P3`), leaderboards overall and by bank, backtest by horizon and every forecast path, every bank's coefficients, the shrinkage table, scenario forecasts and fitted values. Also: the coefficients at every backtest origin, `oos/`, charts, `summary.md`, and `fitted/` with every model on all history and at every backtest and out-of-sample origin.

## Several related targets: the system (`run_system.py`, `src/system/`)

For targets that affect each other, such as two loans, deposits and the bank's own rates, list them in one config (`config/system_example.yaml`). For each target, say what it may use:

```python
from macroforecast import System
sy = System("config/system_example.yaml").fit()     # or: python run_system.py --config config/system_example.yaml
sy.chain()                 # order, form, what each target uses, chosen model and drivers
sy.leaderboard()           # backtest; sy.leaderboard("oos") for the out-of-sample test
sy.links()                 # how each target responds to the earlier ones in its chosen model
sy.scenarios("deposits")   # joint scenario paths: chain, BVARX, reconciled
sy.ratios(outside_only=True)   # e.g. loan-to-deposit quarters outside the historical range
sy.target("mortgage")      # one target's complete single-target results (a Forecaster)
sy.sur_coefficients(); sy.sur_correlation(); sy.fan("deposits", "S4 Severe")
sy.vecm_coefficients(); sy.cointegration()
```

**How it works:**

1. **Order.** Targets are sorted so that each comes after the ones it `uses`: rates, then loans, then deposits.
2. **Chain, with the full project per target.** Each target runs through the complete single-target project: both selection approaches, every model family, formula and calculator, each with its own metadata and literature workbook. The earlier targets are added to its candidate drivers as `SYS.<NAME>`, or as a spread over a market rate (`SYS.<NAME>_SPREAD`), with the sign given in the config. Selection decides whether they enter. In scenarios, the earlier targets' scenario forecasts are the inputs.
3. **Joint backtest.** A target's own backtest uses the actual values of the earlier targets, which nobody has at forecast time. The joint backtest forecasts the whole chain from each origin, so the earlier targets' forecasts feed the later ones. The leaderboard shows both: `Chain` (the honest number) and `Chain, actual inputs`.
4. **Out of sample** (`system.oos`). At each origin, every target's drivers are re-selected with data up to the origin, then the chain forecasts itself.
5. **BVARX challenger.** All targets' changes go in one VAR(1) with the chain's macro drivers as inputs from outside. Cross-target effects are shrunk toward 0 (`bvarx.lambda`); own lags and macro effects are not shrunk.
6. **SUR estimation of the chain** (`sur`). The same equations (each target's chosen drivers, its lagged change, constant, dummies) are estimated jointly by feasible GLS (Zellner's seemingly unrelated regressions), using the correlation of the targets' surprises. That gives slightly more precise coefficients and joint uncertainty. The joint backtest scores it as `SUR chain`. The fan charts (`4b_sur_fan`, `system_fan.png`) draw shocks from the estimated covariance, so loans, deposits and rates move together as they did in history.
7. **VECM challenger** (`vecm.relations`). Every target's change, with error-correction terms for long-run relations:
   - `numerator`/`denominator`: a ratio such as loan-to-deposit, ln(loans) − ln(deposits);
   - `target` + `against`: an own rate on market rates in levels, estimated as an Engle–Granger regression.

   Cross-target lags are shrunk as in the BVARX. Each relation gets an ADF test (Engle–Granger critical values when estimated), and the targets' levels get a Johansen trace test (`7e_cointegration`). A relation that fails the test is not a reliable anchor; remove it from the config.
8. **Reconciliation** (`identities`). For an identity such as total loans = mortgage + commercial, the total is also forecast on its own. All forecasts are then combined so the identity holds, weighting each by its backtest mean squared error (MinT with a diagonal error matrix).
9. **Ratios** (`ratios`). For example loan-to-deposit, under every scenario, flagged when outside its historical range.

Balances use log growth and rates use quarterly change. Errors are % for balances and percentage points for rates.

`data/system_data.csv` is simulated from known equations, written out in `data/system_data_equations.md`, so you can check what the framework recovers. Results go to `outputs/<run>/system/`:

- the chain (`0_chain`);
- the leaderboard and errors by horizon;
- every forecast path and every scenario path;
- the ratios and the links;
- the BVARX coefficients and the reconciliation weights;
- out-of-sample choices, charts, `summary.md`;
- `fitted/`: the chain's models, the models at every backtest origin, and the BVARX.

Each target's complete results are in `outputs/<run>/targets/<target>/`.

### The system with several banks (`system.panel_column`)

When the system's data file has a bank column, every bank gets the same chain of targets and each link is estimated across banks:

```
python run_system.py --config config/panel_system_example.yaml     # system.panel_column: bank_name
```

```python
sy = System("config/panel_system_example.yaml").fit()
sy.leaderboard()                     # mean across banks: Panel chain vs Panel chain, actual inputs
sy.by_bank("deposits", "Bank E")     # one bank
sy.links()                           # each link: pooled, partial pooling and every bank's own estimate
sy.target_leaderboard("mortgage")    # the panel models for one target
sy.ratios(outside_only=True)         # loan-to-deposit per bank, outside its own history
```

- **Each target is a panel run.** Drivers are chosen on all banks together (as in `run_panel.py`). A bank's own earlier targets are candidates (`SYS.MORTGAGE_RATE`, `SYS.DEPOSIT_RATE_SPREAD`): bank A's deposits are driven by bank A's mortgages and bank A's deposit spread, not by the industry's.
- **Models per target:** separate (one equation per bank), pooled fixed effects (common slopes, own intercepts), partial pooling (each bank's slope shrunk toward the pooled one), AR(1). The best by rank across banks drives the chain.
- **Chain backtest per bank.** Mortgage rate → mortgages → deposits, with each bank's forecasts feeding its own later targets; scored against the same chain fed the actual earlier values.
- **Scenarios and ratios per bank**, with the ratio flagged when outside that bank's own history.
- **Links table** (`6_links`): for every system driver, the pooled estimate with Driscoll–Kraay t, every bank's separate estimate, and the partial-pooling weight on the bank's own estimate. A weight of 0 means the banks' own estimates differ from each other by less than their sampling error, so the data cannot tell the banks apart and the pooled slope is used for all.

Not yet in the panel system: SUR, VECM, BVARX and reconciliation (they run in the single-series system). Results: `outputs/<run>/system_panel/`; each target's panel tables in `targets/<target>/panel/`. `data/panel_system_data.csv` is simulated for 5 banks with different true links (`data/panel_system_equations.md`).

## The app (`streamlit run app.py`)

A point-and-click front end for everything above. It runs on your computer and opens in your browser; nothing leaves your machine.

```
pip install streamlit
streamlit run app.py
```

| Tab | What you do |
|---|---|
| ① Files | Upload the loan balance, the scenario files, the **literature workbook (required; a filled-in template can be downloaded here)**, the metadata (recommended) and the events (optional). The app reads them and proposes every setting |
| ② Data settings | **The dependent variable**: pick the balance column, with charts of its level and quarterly growth. Then the scenario names and the baseline, the metadata column mapping, the geographies and themes, the deflator, which of your series stands for each literature concept, and the published models to compare with their expected signs (the workbook's models sheet, editable: add or delete rows) and the literature settings (error-correction model, channels, regime threshold) |
| ③ Sample and models | Dates (estimation start, backtest window, out-of-sample origins, one-off quarters), **the transforms tried for each candidate (level forms, rate forms), the maximum lag and the ADF threshold**, the horizon and re-estimation schedule, the selection rules (most drivers, smallest \|t\|, VIF), which model families compete, **the auto-ARDL settings (criterion, most lags, driver sets, levels + bounds test), the horizon-combination blocks, the same-growth variables, and which models to run** |
| ④ Check and run | Save the configuration (`config/<name>.yaml`), see the pre-flight check, run the pipeline or the out-of-sample test with a live log |
| ⑤ Results | **Results to show**: the latest run or any saved earlier run. **The model to use by hand**: the recommended formula, the next 4 quarters term by term and an Excel calculator (plus the most accurate model overall if it has no formula). The out-of-sample ranking (the one to use) and the backtest, the charts, **a detail page for every model**, downloads (the Excel tables, the summary and the run trace with its data story), and **the presentation**: one button builds the styled PowerPoint deck for this run (title and region editable) and offers it for download |

**The model detail page** (⑤, pick a model):
- What the model is, its source and drivers.
- Its driver table, which depends on the model type:
  - regressions (ARDL, ECM, ARIMAX): coefficients with HAC standard errors, t, p and a plain-language reading, including long-run effects;
  - random forest and gradient boosting: importance;
  - TFT: the weight its variable-selection networks give each input, looking back and looking ahead;
  - combinations: member weights;
  - regime models: the coefficients in each regime.
- Two scenario-effect columns for every model, the networks included: how much the driver's baseline path moves growth (pp a year) and the balance after 8 quarters (%), compared with holding the driver at its last actual value. This makes black-box and regression models comparable.
- **Forecasts in the test samples**: for the backtest or the out-of-sample test, pick the quarter the forecast was made at and see each quarter's forecast against the actual balance (chart and table).
- The residual checks, the balance error by horizon (backtest against out of sample), the forecast under each scenario, the coefficients at each backtest re-estimation (a stability check), the regime history, and the drivers chosen at each out-of-sample origin. The same tables are in `comparison/3_drivers.csv` and its neighbours.

**Panel data in the app.** In ① a bank column (text column with several names repeating over the same dates) is detected and proposed; it can be changed or set to none. ② then shows each bank's history (start, end, quarters, last balance) and a chart per bank; ③ adds the panel settings (models, driver sets, minimum quarters per bank, banks to include, full run per bank); ④ runs `run_panel.py`; ⑤ shows the panel results: leaderboard overall and by bank, the pooled and per-bank coefficients, shrinkage, and each bank's forecasts and scenarios.

"Existing configuration" in the sidebar opens a config you already have, to run it, see its results, or edit any setting in the file directly. Whatever the app saves is an ordinary config file, so `python run_pipeline.py --config config/<name>.yaml` repeats the same run from the command line. The deep-learning and foundation models are off by default in the app (they add several minutes and need extra packages): tick them in ③.

## Using a new data set (`tools/new_data_setup.py`, `tools/check_config.py`)

No code changes are needed. One command writes the config from your files:

```
python tools/new_data_setup.py --name my_portfolio --bank data/my/loans.xlsx \
    --macro-dir data/my/scenarios --metadata data/my/metadata.csv --fast
```

It finds the date and balance columns, takes the scenario names from the file names, and matches your metadata headers to the six fields the model needs. Missing metadata fields (transform, theme, expected sign) are filled with suggestions in `data/<name>_metadata_draft.csv`. It maps the literature concepts to your series by keywords (keeping any series your workbook already names; `--literature my_workbook.xlsx`, otherwise the example's) into `config/<name>_literature.xlsx`, sets every date from the data, and writes `config/<name>.yaml` with the example's comments kept. Review the metadata draft and the workbook it names, then run.

`python tools/check_config.py --config config/<name>.yaml` checks names, files and dates in a few seconds and prints a fix for each problem. The same check runs at the start of `run_pipeline.py` and `run_oos.py` and stops on errors (`--skip-check` overrides it).

"Where to find the data" and "From downloads to the files the project reads" above cover the sources and file formats.

## One master file for all targets (`literature.master`)

Instead of one literature workbook and one metadata file per target, all of them can sit in one Excel file (the `literature_master.xlsx` layout: sheets `models`, `concepts`, `settings`, `how_to_fill` and `metadata` each with a `target` column, `references` with a `used_by` column; `targets`, `sign_matrix` and `README` are summaries and are ignored):

```yaml
literature:
  master: config/literature_master.xlsx   # workbook and metadata both come from here
  target: mortgage                         # the rows to use (default: run_name; case and _/-/space ignored)
```

- The master can also be given as `literature.workbook` and/or `data.metadata.file`; each is then taken from it.
- Package: `fc.fit(data=..., metadata=M, literature=M, hyperparameters={"literature.target": "mortgage"})`.
- System: `system.master: config/literature_master.xlsx`; each target uses the rows named like it, or `master_target: commercial_loans`. A target's own `metadata:` / `literature:` still wins.
- App ①: upload the master as the literature workbook and pick the target.
- The target's rows are written as ordinary files to `outputs/<run>/inputs/master/` (rewritten only when they change, so the cache still works) and everything else reads those. A target with metadata but no literature rows runs without the literature models. A target not in the master stops with the list of targets it has.
- `python tools/literature_master.py list|split|build` shows the targets, writes one target's files, or builds a master from per-target files. `config/literature_master_example.xlsx` holds the project's five example targets.

### Your macro names: one place (`macro_names` sheet)

The master names macro series by its own codes (e.g. `FHPI.ION`, vendor-style mnemonics), in every target's metadata and in every literature concept. When your scenario files come from another provider, or use other names, there is **one place to edit**: the master's `macro_names` sheet.

| code | your_name | in_your_data | suggestion | description | literature_uses | ... |
|---|---|---|---|---|---|---|
| FHPI.ION | ON_HOUSE_PRICES | yes | | House price index, Ontario | 14 | |

- One row per macro series the master uses (400 in the uploaded master), sorted by how many literature models use it. Write your column name in `your_name`; blank means your files use the code itself.
- The map is applied everywhere a series is named: each target's metadata (so expected signs, themes and transforms carry over), each literature concept's `series_name`, and the config's deflator, constructed series (spreads, real rates), same-growth drivers and `trace.follow`. Nothing else needs editing.
- Help to fill it: `python tools/literature_master.py names config/literature_master.xlsx --macro data/my/baseline.csv --macro-metadata data/my/metadata.csv` adds `in_your_data` (is each name in your file?) and a `suggestion` matched from your metadata's descriptions with its score; `--accept 0.6` copies the suggestions with at least that score into `your_name` for you to review. Re-running keeps what you wrote. `merge` refreshes the sheet when targets are added.
- **Where the codes still appear, and why that is fine.** The codes stay as keys in `concepts.series_name`, `metadata.mnemonic` and `sign_matrix.mnemonic`: they are how each row finds its line in `macro_names`. Do not edit them there. Next to each of these columns, a formula column `view: your name` shows the name that will be used, live, as soon as you fill `macro_names` (read only; the code ignores it).
- **Configs too.** Codes written in a config (the deflator, constructed series such as spreads and real rates, same-growth drivers, and in a system config `spread_over:` and `against:`) are translated by the same map; a system takes it from `system.master` or `system.names: my_names.csv`. The example configs can stay as they are.
- A series you do not have can stay unmapped: the pre-flight check lists it, and the literature concepts that use it are skipped with a message.
- Instead of the sheet, `literature.names: my_names.csv` (columns `code`, `your_name`) can hold the map, e.g. one file per provider; it wins over the sheet. It also works without a master, on a single literature workbook and metadata file.
- Tested by renaming all 400 example series and editing only the map: all 51 targets of the master load and pass the pre-flight check (apart from three targets the master has incomplete), and two full runs give exactly the same results as with the original names.

## Many targets in one data file (`python run_targets.py`)

A target file can hold many series side by side (`date, cre_loans, heloc_balances, stage2_share, ...`). With a literature master, every column gets its own literature models, expected signs and metadata, so any of them can be chosen and run:

```
python run_targets.py --config config/multi_target_example.yaml --list                        # what can be forecast
python run_targets.py --config config/multi_target_example.yaml --targets mortgage_balance,deposit_rate
python run_targets.py --config config/multi_target_example.yaml --all --oos --story            # every ready column
```

- `--list` shows each column, the master target it matches (same name, case and `_`/`-`/space ignored, or `targets.map`), how many literature and metadata rows it has, its history, the suggested form and whether it is ready.
- The form is suggested per column: quarterly change (`diff`) for rates, ratios, shares, margins, delinquency and anything zero or negative; log growth otherwise. `targets.forms` overrides it.
- Each chosen column runs as its own run, `outputs/<run_name>_<column>/`, with all the usual results (and the out-of-sample test with `--oos`, the process story with `--story`). The per-target configs are kept in `outputs/<run_name>_targets/configs/`, so any target can be re-run alone with `run_pipeline.py --config ...`.
- `outputs/<run_name>_targets/summary.md` lists, for every target, the best model, the best model with a formula and their errors.
- A column with no master target runs with the config's own `data.metadata.file` and no literature models; without either it is not ready.
- In one run, `literature.target` now defaults to the target column's name when the master has it (then the run name).
- App ①: a file with several series offers a "Target to forecast" choice, and the master's target follows it; ③ suggests the form.
- New targets are added to a master with `python tools/literature_master.py merge config/literature_master.xlsx --folder FOLDER` (per-target `<target>_literature_workbook.xlsx` and `macro_metadata_<target>.csv`; a target already in the master is replaced).

Optional config block (`config/multi_target_example.yaml` uses `data/system_data.csv`, five simulated series, and `config/literature_master_example.xlsx`):

```yaml
targets:
  map: {mortgage_balance: mortgage, commercial_balance: commercial_loans}   # column -> master target
  forms: {stage2_share: diff}
  labels: {cre_loans: CRE loans}
  units: {deposit_rate: '%'}
  run: [mortgage_balance, deposit_rate]          # default choice when neither --targets nor --all is given
```

## Using your own files: only the config changes

Copy `config/example_config.yaml`, rename it, and edit it. Everything that depends on a data set lives there:

| What | Where in the config |
|---|---|
| Bank file, sheet, date column, balance column, monthly or quarterly | `data.bank` |
| Inorganic events file and its columns (optional) | `data.events` |
| One file per scenario (any provider), any names, plus the date column | `data.macro.scenarios`, `data.macro.date_column` |
| Metadata file and **your** column names for mnemonic, description, geography, theme, transform, expected sign | `data.metadata.columns` |
| Extra variables built from macro columns (spreads, real rates) | `data.constructed` |
| Dates: last actual quarter, estimation start, backtest start, dummy quarters | `sample` |
| Selection settings for each approach and the final model | `features`, `screen_first`, `en_first`, `final_model` |

A few details on the inputs:

- Files can be `.csv` or `.xlsx`; give `sheet:` for Excel.
- Dates can be quarter labels (`2020Q1`) or real dates (`2020-03-31`).
- Monthly balances are converted to quarter-end values.
- `run_name` sets the output folder, so runs on different portfolios never overwrite each other.

## Skipping unchanged steps (hash-based cache)

Each step gets a fingerprint, a SHA-256 hash, built from everything it depends on:

| Step | Its fingerprint includes |
|---|---|
| `data` | the loaded values of the target, balance, metadata, macro history and every scenario |
| `features` | macro history, metadata, the `features` and `sample` settings, the code in `features/` |
| `screen_first` / `en_first` | the `features` fingerprint, the target, that approach's settings, `final_model`, `sample`, its code |
| `evaluate:<model>` | the chosen drivers, the target, the scenarios, `sample`, the evaluation code |

If a step's fingerprint matches a result already in the cache, the result is loaded instead of recomputed. A full rerun with nothing changed takes about 4 seconds instead of 30. Some examples from testing:

- **New scenario path only:** only the `evaluate` steps re-run.
- **One revised history value:** everything downstream re-runs.
- **Settings change that leaves the drivers the same:** the evaluation is reused, because it depends on the drivers, not on how they were found.
- **Same values under new file or column names:** everything is loaded from the cache. Data are hashed by value, not by file name.
- **Edited code:** only the steps that use that code re-run.

Each run writes `outputs/<run_name>/run_manifest.json` and appends to `run_history.csv`. These record, for every step, whether it was cached or computed and which inputs changed since the previous run, for example `changed: config.en_first` or `changed: scenarios`.

Options:

```
python run_pipeline.py --force                      # recompute everything
python run_pipeline.py --force-step en_first        # recompute one step (also: features, screen_first, evaluate)
python run_pipeline.py --no-cache                   # neither read nor write the cache
```

The cache lives in `cache/` (set by `cache.dir` in the config) and is shared by all configs. Delete the folder to clear it.

## Backtest window and re-estimation schedule

All set in `sample:` in the config. They apply to the rolling backtest in `comparison/` and to the backtest the final search uses to rank driver combinations, so selection and evaluation stay consistent.

| Setting | Default | What it does |
|---|---|---|
| `backtest_start` | 2018Q1 | First forecast origin |
| `backtest_end` | 2024Q4 | Last forecast origin (null = the last quarter with an actual after it) |
| `backtest_step` | 1 | Quarters between origins: 2 = every other quarter |
| `reestimate_every` | 4 | Quarters between re-estimations. 4 = once a year: the coefficients are estimated at the first origin of each year and kept for the next three quarters, and only the forecast is redone from the latest actual data, as in production |
| `reestimate_quarter` | 1 | With `reestimate_every: 4`, the quarter in which the yearly re-estimation happens |

**A short window.** The default window 2018Q1–2024Q4 has 28 origins, but the COVID quarters (2020Q1–2021Q2) are left out of the error metrics, and with data to 2025Q4 only origins up to 2022Q4 have a 12-quarter-ahead actual. So the 12-quarter errors rest on about 7 origins (22 at 1 quarter, 19 at 4, 11 at 8). The ranking of driver combinations uses the same window, so the chosen drivers can differ from a longer window. For a more robust ranking, widen it (`backtest_start: 2010Q4`, `backtest_end: null`).

Every estimation uses data up to its own origin only. Machine-learning models are re-estimated at least every 4 quarters (`refit_every`) to save time; this is aligned to the same calendar.

On the example data, with the earlier window 2010Q4–2022Q4 (average error over 1–12 quarters):

| Schedule | Origins | Re-estimations | Run time | AR(1) | Screen first | EN first | Hybrid: theory long run + data short run |
|---|---|---|---|---|---|---|---|
| Every quarter, re-estimate each | 60 | 60 | 79 s | 1.87% | 0.71% | 0.93% | 0.83% |
| Every 2nd quarter | 30 | 30 | 64 s | 1.90% | 0.72% | 0.92% | 0.83% |
| Every quarter, re-estimate yearly | 60 | 16 | 65 s | 1.89% | 0.70% | 0.95% | 0.88% |
| Every 2nd quarter, re-estimate yearly | 30 | 16 | 55 s | 1.89% | 0.77% | 0.92% | 0.81% |

Yearly re-estimation costs little accuracy. The run-time saving is modest because driver selection, not the backtest, takes most of the time. With every 2nd quarter and yearly re-estimation, screen first chooses Toronto single-detached prices and income instead of mortgage credit, income and permits.

The out-of-sample test has its own spacing: `oos.first_origin` and `oos.step` (default every 4 quarters).

## Out-of-sample test (`python run_oos.py`)

The backtest in the main pipeline re-estimates the coefficients at every origin, but it chooses the drivers once using all the data. That makes it optimistic. `run_oos.py` runs a true out-of-sample test instead:

1. **Re-select at every origin.** At each forecast origin (every `oos.step` quarters from `oos.first_origin`), the whole selection is re-run with data up to that origin only: stationarity tests, candidate features, both approaches, the final search, the hybrid searches and the construction of every competing model.
2. **Forecast.** Every model forecasts the next 1–12 quarters (`sample.horizon`), using realised macro values.
3. **Compare** against:
   - the same approach with the drivers chosen on the full sample (shows how much the full-sample backtest flatters the model);
   - the AR(1) benchmark.
4. **Report.** Outputs go to `outputs/<run_name>/oos/`:
   - MAPE, RMSE, bias and Theil's U against the benchmark, by horizon;
   - Diebold-Mariano tests against the benchmark, with the small-sample correction;
   - the drivers chosen at each origin, how often each driver was chosen, and the average overlap between consecutive origins;
   - a holdout: select and estimate on data up to `hist_end - holdout_quarters`, then forecast the held-out quarters.

Set `oos.window: rolling` (with `window_length`) for a fixed-length estimation window instead of an expanding one.

Each origin's selection is cached, so the first run takes a few minutes (about 4 minutes for 12 origins on the example data) and re-runs take seconds. `step: 1` gives more origins and more powerful tests, but takes about four times longer the first time.

Results on the example (simulated) data are in **Example results** below. The look-ahead check for the two data-driven approaches (average balance error, 1–12 quarters):

| Approach | Re-selected at each origin | Drivers fixed from the full sample |
|---|---|---|
| Screen first | 1.37% | 0.72% |
| Elastic Net first | 1.61% | 1.05% |

Elastic Net first chose more stable drivers across origins (average overlap with the previous origin 0.59, against 0.18 for screen first).

## Competing models (`src/models/`)

After the two approaches have chosen their drivers, `src/models/zoo.py` builds every competing model. All of them use the same information (the macro history, and the scenario path in scenario forecasts), the same rolling backtest and the same out-of-sample test.

| Family | Models | How they are specified |
|---|---|---|
| Benchmark | AR(1); mean growth | Loan growth on its own lag; average growth (a random walk with drift in the balance) |
| Data-driven | Screen first; Elastic Net first | The two selection approaches |
| Literature | Classic ARDL (Hofmann; Calza et al.); ECB BEAST loan demand (Budnik et al. 2023); four-channel core (house prices, income, mortgage rate, unemployment); error-correction model | Fixed specifications from published studies, in the literature workbook |
| Same growth as a driver | Same growth as disposable income; same growth as house prices (each exact, and plus the average gap) | Loan growth = growth of the chosen variable, nothing estimated (or plus its average historical gap); `models.same_growth` in the config |
| Auto ARDL | Auto-ARDL on the literature core; auto-ARDL on the screen-first drivers; ARDL in levels with the bounds test | EViews-style: lags chosen by BIC/AIC/HQ, again at every re-estimation (`models.auto_ardl`) |
| Horizon combination | Best per horizon; weighted per horizon | A different model for each block of quarters ahead (1-4, 5-8, 9-12), chosen from errors already known at each origin (`models.horizon_combination`) |
| Hybrid | Theory core + data add-ons; theory long run + data short run; theory channels, data measures; 50/50 data + theory; performance-weighted | Theory sets the structure and data fills it in, or both are averaged |
| Machine learning | Ridge; random forest; gradient boosting; neural network (5 small nets, averaged) | Lagged growth, dummies and the pooled drivers (literature core + data-chosen); forecasts iterated with the macro path |
| Regime switching | Markov switching (Hamilton); threshold ARDL; smooth transition (LSTAR); time-varying parameters (Kalman); break-adaptive ARDL | The literature core drivers, with relations that change with the state of the economy (see below) |
| Statistical time series | ARIMAX (literature core drivers with ARMA(1,1) errors); Theta (M3 winner, own history) | statsmodels; coefficients kept between re-estimations |
| Deep learning | TiDE (Google, 2023); N-HiTS (2023); TFT, temporal fusion transformer (2021); optional N-BEATS, TSMixer | Small networks trained on this data (darts), average of several random seeds; TiDE and TFT take the drivers' scenario paths as known future inputs |
| AI foundation model | Chronos-Bolt (own history); Chronos-2 (with the literature drivers and their scenario paths) | Pretrained, zero-shot |

**Choosing which models run.** `models.only` in the config (or "Run only some models" in the app, ③) limits the run to the listed families or models, e.g. `only: [Literature, "Screen first", "Same growth as a driver"]`. An entry matches a family, a model name or the start of one; the AR(1) benchmark is always kept, because every comparison and test is made against it. Leave it empty (`[]`) to run everything.

**Same growth as another variable.** `models.same_growth: [disposable_income, house_prices]` adds, for each variable, a model where loans grow exactly as fast as the variable (a constant loans-to-variable ratio) and one where they grow as fast plus the average historical gap. Each entry is a literature concept or one of your series; the long form `{variable: FYPD.ION, lag: 1, gap: yes}` sets a lag or only one variant. Rates are skipped (their growth has no meaning). They go through the same backtest, out-of-sample test and scenarios as every other model.

**Auto-ARDL, as in EViews (`src/models/autoardl.py`).** The ARDL models above use one lag per driver, fixed by the selection. The auto-ARDL instead chooses the lags the way EViews' ARDL estimator does: every combination of 1 to `max_p` lags of loan growth and 0 to `max_q` lags of each driver is estimated on the same sample, and the one with the lowest criterion wins (`criterion: bic`, or `aic`, `hq`). With 4 drivers and 4 lags that is 2,500 regressions, about a second. The choice is made again at every re-estimation, so the backtest and the out-of-sample test only use lags that could have been chosen at the time. The detail page shows the chosen lags in EViews notation, e.g. ARDL(1, 1, 1, 0, 0), every lag's coefficient, and each driver's long-run effect: the sum of its lag coefficients divided by (1 − the sum of the growth lags). The sign check uses that long-run effect.

**ARDL in levels and the bounds test.** "Auto-ARDL in levels (bounds test)" models ln(real organic loans) on the long-run variables of the literature error-correction model (log real house prices, log real GDP, the real mortgage rate, …), with lags chosen the same way. The Pesaran, Shin & Smith (2001) bounds test then checks whether a long-run relation exists, as EViews does: the model is rewritten in conditional error-correction form (dummies kept), and the F-statistic that all lagged levels are zero, and the t-statistic on lagged real loans, are compared with the I(0) and I(1) critical bounds for case III (unrestricted constant, no trend). F above the upper bound means a long-run relation; below the lower bound, none; in between, inconclusive. The F, the bounds, the p-values and the decision are in `comparison/4_diagnostics.csv` and on the detail page, with the long-run coefficients and the adjustment speed. (The critical values are taken by the number of regressors k; statsmodels' own `bounds_test` uses k + 1 and drops the dummies, so it is not used directly.) On the simulated example, F = 2.0 is below the 5% lower bound (3.23): no long-run relation.

**Horizon combinations (`src/models/horizon.py`).** Some models are best for the next few quarters, others for the medium term. "Horizon combination: best per horizon" splices them: quarters 1-4 from the model best at 1-4 quarters ahead, quarters 5-8 from the best at 5-8, and so on (`buckets`). "Weighted per horizon" averages the best `top_k` (3) in each block, weighted by 1/error. There is no look-ahead: at each origin, the choice uses only forecasts made earlier whose quarters had already happened, with at least `min_history` (3) earlier origins per block; before that it uses the model best over all horizons so far (or the AR(1)), and the choices table says so. "Best" is judged by the error of the growth in those quarters (`criterion: growth`, since that is what gets spliced) or by the balance error at those horizons (`criterion: balance`). The combinations go through the same backtest and out-of-sample test as every other model; their scenario forecasts use the choice made with all the test errors. Each origin's choices are in `comparison/13_horizon_choices.csv` and `oos/8_horizon_choices.csv`, and on the detail page. They have no single formula, so the formula output reports the best formula model next to them.

**Keeping the run time down.** Machine-learning hyperparameters are fixed (no tuning loop), and in the rolling backtest these models are re-estimated every 4th origin (`refit_every`), always with data up to that origin only. Foundation models take under a second per forecast. On the example data the 17 models without foundation models take about a minute; the out-of-sample test with re-selection takes about 5 minutes the first time and is cached after that.

**Regime-switching models (`src/models/regime.py`).** All five use the four literature core drivers. Their only difference from "Literature: four-channel core" is that the relation is allowed to change, so the comparison shows directly whether regimes help.

| Model | How the regime is set | In scenarios |
|---|---|---|
| Markov switching (Hamilton 1989) | Two unobserved regimes with low and high mean growth. The regime probability is inferred from the data and carried forward with the estimated transition matrix | The drivers respond to the scenario; the regime probability evolves on its own |
| Threshold ARDL (Hansen 2000) | Every coefficient differs below and above an estimated threshold of the regime variable (default: house price growth, `models.regime.threshold`) | The scenario's house price path decides the regime in each future quarter |
| Smooth transition, LSTAR (Teräsvirta 1994) | As the threshold model, but the switch is gradual (a logistic weight) | Same |
| Time-varying parameters | Every coefficient drifts slowly as a random walk (Kalman filter); the speed of drift is estimated, and 0 means no drift | Latest coefficients |
| Break-adaptive ARDL (Andrews 1993) | Tests for a structural break (sup-F) at each re-estimation and, if one is significant, estimates only on the data after it | Post-break coefficients |

- `comparison/3_drivers.csv` shows each model's regimes: the threshold and the share of quarters in each regime, the regime means and expected durations, how the coefficients drifted, and the break date.
- `comparison/11_regimes.csv` and `regimes.png` show the regime indicator over history.
- The Markov model is estimated with growth in percent and standardised inputs, because the likelihood search needs this with about 100 quarters. Each regime having its own volatility is off by default: with this little data, one regime tends to collapse onto a few outliers. Turn it on with `settings: {markov: {switching_variance: true}}`.
- If the likelihood search fails, the model falls back to one regime and the driver table says so.
- On the simulated data, which was generated without regimes, the tests find weak evidence for them, as they should. Real data, with policy changes such as the 2017 Canadian mortgage stress test and the 2022 rate shock, is where they can show their value.

**Deep-learning models need `pip install darts`** (it installs PyTorch, about 1 GB, more on Linux with GPU libraries). Without it they are skipped with a message. They are deliberately small, because there are only about 100 quarters to learn from: 1 hidden layer of 16–32 units, 8 quarters of input, 4 quarters predicted per step (the 12-quarter path is built by repeating that), 80 training epochs, refitted once a year. Settings are in `models.deep.settings`, with per-model overrides (TFT uses 2 seeds and 50 epochs because it is the slowest). Approximate run time on 4 CPU cores for the rolling backtest plus scenarios: N-HiTS 50 s, TiDE 80 s, TFT 4 min.

**Foundation models need two extra things.** First, `pip install chronos-forecasting`, which also installs PyTorch. Second, the weights must be downloaded from Hugging Face on first use: about 190 MB for chronos-bolt-small and about 480 MB for chronos-2. Without them the two models are skipped with a message, and everything else runs as usual. Offline, download the model folders once and put their paths in `models.foundation.models[].id`. Their code was tested here with small randomly initialised networks, because Hugging Face could not be reached from the test environment; their accuracy has not been measured yet.

**Comparing models at 4, 8 and 12 quarters.** `sample.horizon: 12` sets the forecast length; `sample.report_horizons: [4, 8, 12]` sets the comparison points. For each model, `comparison/1_model_comparison.csv` and `oos/1_oos_summary.csv` give:

- `avg_1_h_pct`: the average balance error over 1..h quarters;
- `at_hq_pct`: the error at exactly h quarters;
- `vs_AR_1_h_pct`: the improvement on AR(1);
- the rank over 1–12 quarters.

The drivers of the data-driven approaches are ranked over the full 12 quarters.

Other outputs in `comparison/`:
- `model_comparison.png`: all models side by side at 4, 8 and 12 quarters, coloured by family;
- `9_model_list.csv`: what each model is, its drivers and its sources;
- `10_hybrid_*.csv`: the hybrid searches.

To switch families on or off, or change settings, edit the `models:` section of the config.

## The recommended model as a formula (`outputs/<run>/formula/`)

Every run writes the most accurate model **that is a formula you can write down** (ARDL regressions, the error-correction model, the same-growth rules), with everything needed to use it by hand for the next quarters:

| File | What it holds |
|---|---|
| `formula.md` | The equation with its coefficients, e.g. `g(t) = 0.0045 + 0.20·g(t-1) + 0.14·x1(t) + …` and `L(t) = L(t-1) × exp(g(t))`, and what each symbol is as a formula in the raw series (`x1(t) = ln(FHPI.ION(t-1) / FHPI.ION(t-2))`) |
| `1_terms.csv` | Every coefficient, its definition and meaning |
| `2_next_quarters.csv` | The next 4 quarters (`save.formula_quarters`) under each scenario, term by term: each driver's value, its contribution, growth, the balance. It equals the model's own forecast exactly (checked on every run) |
| `recommended_model_calculator.xlsx` | The same calculation with live Excel formulas: one sheet per scenario, the raw series (history and future values), each driver rebuilt from them, growth and the balance. Change a yellow cell (a future value of a driver) and the forecast updates |
| `3_best_overall_next_quarters.csv` | When the most accurate model overall has no formula (machine learning, networks, regime-switching, combinations), its forecast for the same quarters, so both are on the table |

`run_pipeline.py` ranks by the backtest; `run_oos.py` redoes the choice by the out-of-sample error (the one to use). The app shows it at the top of ⑤ Results, and the deck has a slide for it.

## Saved results: open any run again without re-running (`fitted/`, `saved_runs/`)

Everything estimated in a run is kept:

| File | What it holds |
|---|---|
| `fitted/models.pkl` | every model estimated on all history (`load_models`) |
| `fitted/coefficients.csv` | every coefficient with standard error, t and p value; weights or parameters for other model types |
| `fitted/fit_statistics.csv` | observations, R², adjusted R², AIC, BIC, residual standard deviation |
| `fitted/fitted_values.csv` | in-sample one-step fit of every model: actual, fitted, residual, levels (dummy quarters flagged) |
| `fitted/panels/<model>.csv` | the data each model was estimated on (history and the baseline path) |
| `fitted/backtest_models/<model>.pkl` | the model as re-estimated at every backtest origin (`load_origin_models`) |
| `oos/fitted_models/<origin>.pkl` | the re-selected models at every out-of-sample origin |
| `oos/fitted_coefficients_by_origin.csv` | their coefficients, origin by origin |
| `fitted/data/target.csv` | the target level and its organic change |
| `fitted/manifest.json` | what was saved for each model, and the target form |

`save.origin_models`, `save.fitted_values` and `save.panels` (all on by default) switch the larger parts off.

- **`fitted/models.pkl`**: every model as estimated on all history. `from src.utils.saved import load_models; models = load_models("outputs/<run>")` gives them back, ready to forecast. **`fitted/formula_models.json`** holds the regression and same-growth models as plain numbers, readable on any computer even when package versions differ.
- **`saved_runs/<date_time>_<pipeline|oos>/`**: after every run, a complete copy of the results (tables, charts, formula, fitted models) plus the config and literature workbook used (`inputs/`). In the app, ⑤ Results → "Results to show" lists them with their best model; picking one shows that run, and the presentation can be built from it. The newest `save.keep_runs` (20) are kept.
- The step cache (below) also means an unchanged re-run finishes in seconds.

## Forecasts in the test samples

Every forecast made in the tests, quarter by quarter, next to what happened:

- `comparison/12_backtest_forecasts.csv` (rolling backtest) and `oos/6_all_forecast_errors.csv` (out of sample, drivers re-selected at each origin): model, origin, quarter, quarters ahead, forecast and actual growth (% q/q), balance at the origin, forecast balance, actual organic and reported balance, error (%), whether the quarter falls in the COVID window.
- In the app, each model's detail page (⑤) has "Forecasts in the test samples": pick the test and the origin, and see the forecast path against the actual balance.

## Literature workbook (`config/literature_workbook.xlsx`)

Everything about the literature lives in one Excel workbook: the published models, their expected signs, the concept table, the error-correction and channel settings, the regime threshold and the references. The config only points to it:

```yaml
literature:
  workbook: config/literature_workbook.xlsx
```

The app requires one; `python tools/literature_template.py` writes a template filled with the example's content (the app offers the same file for download).

| Sheet | One row per | Columns |
|---|---|---|
| `models` | driver of a published model (rows with the same `model` name form one model) | `model`, `source`, `variable` (a concept from `concepts`, or one of your series directly), `transform` (dlog, yoy, dlog2, acc, gap, diff, diff2, diff4, lvl, rgap, and windows such as ma8, vol4, dma8, dvol4), `lag` (quarters), **`expected_sign`** (+1 / −1 / 0: the sign this study expects for this driver; empty = the concept's sign), `core` (yes on the rows of the one model that is the theory core) |
| `concepts` | economic concept | the concept table described below |
| `settings` | setting | `apply_signs_to_metadata`, `ecm_enabled`, `ecm_name`, `ecm_source`, `ecm_deflator`, `ecm_long_run` (`table` = the concepts with a `long_run_form`, or e.g. `house_prices:log, disposable_income:log_real`), `ecm_short_run` (`core` or a list), `channels` (`table` or a list), `regime_threshold` (e.g. `house_prices\|dlog\|L1`) |
| `references` | study | the full references |
| `how to fill` | — | these instructions |

**Expected signs:** each literature model's coefficients are checked against the signs in its `models` rows. The model detail page (⑤) shows an "Expected sign" and a "Sign as expected" column, and a wrong sign is listed in the diagnostics. In the concepts sheet, `series_name` cells that do not match your data are re-matched automatically by the set-up; the matches, the models (signs included) and the settings can all be changed in the app (②), and the app saves them into `config/<name>_literature.xlsx`. The pre-flight check validates every sheet: variables that exist, valid transforms and signs, a deflator and a regime threshold that resolve.

### The concepts sheet

One row per economic concept that the literature links to household and mortgage credit.

| Column | What it holds |
|---|---|
| `concept_id` | Used in the models sheet: `house_prices`, `disposable_income`, `real_gdp`, `mortgage_rate`, `policy_rate`, `real_mortgage_rate`, `unemployment`, `inflation`, `term_spread`, `affordability`, `credit_conditions`, `deposits` |
| `expected_sign`, `effect_on_loan_growth` | +1 / −1 / 0, and Positive / Negative / Ambiguous |
| `why`, `evidence` | The mechanism, and the studies that support it |
| `transform`, `theory_lags` | Suggested form and lag window |
| `core` | Part of the four-channel theory core (house prices, income, mortgage rate, unemployment) |
| `long_run_form` | Enters the error-correction long run (`log`, `log_real`, `real_rate`) |
| `include` | `yes` / `no`: your filter |
| `series_name` | **The name of the series in your macro file.** This is the only place names are matched. |

How the pipeline uses it:
- **Models refer to concepts, not series codes.** For example, `house_prices|dlog|L1` in the models sheet is resolved through `series_name`. If your vendor or vintage uses different names, edit that one column.
- **`include: no` removes a concept everywhere:** from every literature and hybrid model, the error-correction long run, the theory channels and the Chronos-2 covariates.
- **With `apply_signs_to_metadata = yes` in the settings sheet (the default),** the table's signs also apply to the data-driven screen. A concept set to `include: no` gets sign 0, so the data cannot select it either.
- **A concept with no `series_name`** (affordability and credit conditions in the example) is skipped with a message.
- **Each run writes `comparison/10_literature_variables.csv`,** showing for every concept whether it is in use, available, not mapped or switched off, and which models use it.

## Presentation deck (`report/forecast_deck.qmd`)

```
python run_pipeline.py
python run_oos.py
python tools/make_deck.py --config config/example_config.yaml   # writes outputs/<run>/<run>_deck.pptx
```

`tools/make_deck.py` (also the "Build the presentation" button in the app) renders the deck for any config: the title, the region in the text and the footer come from the config (`screen_first.primary_geography`) or `--title` / `--region`; `--simulated` adds the simulated-data notes. `quarto render report/forecast_deck.qmd` still works for the example (writes `report/forecast_deck.pptx`).

Every number, table and chart in the deck is read from `outputs/`, so after a re-run the deck updates itself. The slides cover:
- the objective and requirements;
- the methodology: target, transformations, the two selection approaches, the selection rules, the literature, the model families, and machine learning and AI;
- the rolling backtest and the out-of-sample test at 4, 8 and 12 quarters;
- the best model of each family;
- the trade-off between accuracy and the number of variables;
- the look-ahead check, driver stability and the holdout;
- the recommended model with its diagnostics and scenarios;
- limitations and next steps.

The recommended model is the most accurate out-of-sample model with at most `max_variables` (3) distinct macro variables, long-run variables included. The most accurate model overall is shown next to it.

- **Requirements:** Quarto (quarto.org), plus `pip install python-pptx jupyter` in the same Python that runs the app or `tools/make_deck.py` (both are in requirements.txt). If one is missing, the build stops with the exact install command.
- **Another run:** `python tools/make_deck.py --config config/my_portfolio.yaml` (the deck's parameters are read from `DECK_CONFIG`, `DECK_OUTPUTS`, `DECK_SIMULATED`, `DECK_REGION` environment variables, so papermill is not needed).
- **Look:** two layers.
  - `report/reference.pptx` sets the theme: Cambria headings, Calibri body, the colour palette, and dark title and section slides. Build it with `python report/make_reference.py`.
  - `report/style_deck.py` runs automatically after `quarto render` (set in `report/_quarto.yml`) and turns simple markers in the `.qmd` into designed elements:

    | Marker | Becomes |
    |---|---|
    | `[[tile: 1.37% \| label \| note]]` | stat tile |
    | `[[icon: home \| heading \| text]]` | icon card (icons in `report/icons/`) |
    | `[[step: ...]]` | process steps |
    | `[[card: heading \| line; line]]` | card with a list |
    | `[[band: lead \| text]]` | dark band across the bottom |
    | `[[chart: id]]` | native, editable PowerPoint chart (specs written by the `.qmd` to `report/_charts/`) |
    | `[[split: 60]]` | left column 60% of the width |
    | `[[dark-slide]]` | dark background for the whole slide |

    A column starting with ★ becomes a dark card. Tables get a dark header, banded rows, and green Pass / red Review.
  - To keep Quarto's plain output for debugging, set `STYLE_DECK_SKIP=1`.

## Process story: what happened at every step (`python tools/process_story.py`)

A detailed, non-technical account of a run for an audience: not the code, but every decision the process made and why, with the run's own numbers and charts.

```
python tools/process_story.py --config config/example_config.yaml            # after run_pipeline.py (and run_oos.py)
python tools/process_story.py --config config/my.yaml --format pptx          # html, pptx or both (default)
python tools/process_story.py --config config/my.yaml --run-dir outputs/my_run/saved_runs/<copy>
fc.process_story()                                                           # from the package
```

Writes `outputs/<run>/process_story/process_story.html` (one long page with a contents sidebar, charts embedded) and `process_story.pptx` (about 50 slides). Both come from the same 44 sections, in 11 chapters: the question; the data (target, one-off events, macro series, expected signs, literature); building candidates (forms, stationarity, lags, the funnel); screen first (screen, best form, clusters, shortlist); Elastic Net first (selection, stability); the final search (each rule's removals, accuracy ranking, the simplest near-best model); the competition; testing (backtest design, one origin up close, every backtest path, leaderboard, errors by horizon, out of sample vs backtest, driver stability, holdout); the recommended model (equation, effects in words, drivers vs target, coefficient stability, diagnostics); the forecast (scenario inputs, forecasts, each quarter taken apart driver by driver, scenario ordering); limitations, recommendation, glossary and settings.

Each section states the finding as its title, a one-line takeaway, what was done, why, what happened in this run and how to read the chart. On the slides the key points and figures sit beside the chart or table, and the full text is in the speaker notes. Titles and takeaways follow the results (e.g. a flagged diagnostic is named, not glossed over). Balance targets are described in growth and percent, rate targets (`target.transform: diff`) in changes and their own units. Needs only python-pptx and matplotlib (no Quarto); the app has a "Build the process story" button under ⑤ Results.

## Function reference (`docs/reference.html`, automatic)

A searchable page with a card for every module, class, function and method in the project (about 280), built from the code itself, so it cannot drift from it. Each card shows:

- the signature, and what each argument means and what is returned (from the docstring);
- the config settings the function reads, e.g. `sample.horizon`;
- what it calls and what calls it, with links. Calls that only show up at run time, like `model.fit` inside the backtest, are taken from the run trace and marked *run*;
- **an example from the latest run**: the inputs it actually received, what it returned and a preview of the returned table. If the function has changed since that run, a warning says the example may be out of date;
- its source code, and the date it was added or last changed.

The page also has a short run flow for each script, listing the steps it ran in order with their run times, plus a list of recent changes. `docs/REFERENCE.md` holds the same reference as plain text without the examples, for reading on GitHub.

**It updates itself.** Three ways, use any of them:

| When | How |
|---|---|
| After every run | `run_pipeline.py` and `run_oos.py` rebuild it at the end if any code or trace changed (about 2 s; `docs:` in the config, `enabled: false` to switch off) |
| While editing | `python tools/make_docs.py --watch` rebuilds every time a `.py` file is saved |
| On every commit | `python tools/make_docs.py --install-hook` adds a git pre-commit hook that rebuilds it and adds `docs/` to the commit |

`python tools/make_docs.py --check` exits with code 1 if the reference is out of date (for CI); `--force` rebuilds now.

Examples are kept in `docs/examples.json` between runs, so a step loaded from the cache keeps the example from the last time it ran. To give a new function a description, write a docstring in the same style as the others:

```python
def refit_flags(ctx, origins, every=None):
    """True where the coefficients are re-estimated.

    Args:
        ctx: Context. origins: positions of the forecast origins. every: quarters between re-estimations.

    Returns:
        bool array, one per origin.
    """
```

Functions without a docstring are listed on the overview page.

## Run trace (automatic, every run)

**Two views.** `trace.html` opens on the **Data story**, written for someone who knows econometrics but not this model. It walks the data through the pipeline in 14 steps (4 for `run_oos.py`: re-selection, accuracy, Diebold-Mariano tests, holdout):

1. data in
2. organic growth (events removed)
3. relevance filter
4. transforms
5. ADF stationarity tests
6. lags
7. univariate screen
8. correlation clusters
9. the combination search
10. Elastic Net with stability selection
11. the final estimate
12. one backtest origin worked through, quarter by quarter
13. the baseline forecast taken apart, driver by driver
14. the competing models

Each step gives:
- what goes in, and the operation as an equation (e.g. `y_t = a + ρ y_{t−1} + Σ δ_q D_q + β x_t + e_t`, HAC s.e.);
- the decision rule, and this run's own counts (e.g. 638 of 738 forms stationary, 2,625 combinations and what each rule removed);
- tables of the actual data, including the regression's design matrix: the exact rows OLS sees;
- links to the exact calls in the **Call tree** view.

One variable is followed through every step: the first chosen driver, or `trace.follow` in the config. The story is built from each stage's outputs, so it is complete even when steps come from the cache. The **Call tree** view is the debugger described below.


Every run of `run_pipeline.py` or `run_oos.py` writes a detailed trace. Nothing in the code needs marking up. At start-up the tracer wraps:
- every function and method under `src/`;
- the entry script's own functions, such as `run()`;
- functions defined inside other functions, and lambdas, when they are passed to a traced call (such as the work inside each cached step);
- the library routines that do the numerical work, marked **lib**: statsmodels OLS and the ADF test, the residual tests, scikit-learn's ElasticNet, scipy clustering, numpy least squares, and pandas file reading. What happens inside a library routine is not traced.

A function you add or change is traced on the next run.

For each call the trace records:
- the function, its file and line, its docstring, and its source code as it was when the run happened;
- which call it came from, in execution order, and how long it took;
- every argument and the return value: type, size, column names, values;
- **the local variables when the call ended**, as a debugger shows them, with small previews of tables. For a loop, these are the values from the last pass;
- for every regression, the coefficient table (coef, s.e., t, p);
- a preview of every table it returned, with the full table saved as CSV;
- the messages it printed;
- the error, traceback and local variables, if it failed.

A function called many times from the same place, such as one regression per candidate, is shown once with ×N. Its page lists the other calls (up to 500) with a short description of their arguments, start time and duration. Calls with a different text argument, such as the step names `screen_first` and `en_first`, get their own entry.

Output goes to `outputs/<run_name>/trace/<date_time>/` (`trace_oos/` for `run_oos.py`), one folder per run. The newest 10 are kept.

| File | What it is |
|---|---|
| `trace.html` | the clickable page: call tree on the left; **Step into / Step over / Step out / Back** (arrow keys) |
| `trace.log` | the same as an indented text log |
| `trace.json` | everything, for `tools/trace_to_html.py` |
| `tables/` | the full tables |

The page is built automatically at the end of the run. To rebuild one (for example from a trace a colleague sent you):

```
python tools/trace_to_html.py                                   # newest trace under outputs/
python tools/trace_to_html.py outputs/<run>/trace/<date_time>/trace.json
```

Settings are in the `trace:` section of the config:

| Setting | Default | What it does |
|---|---|---|
| `enabled` | `true` | `--no-trace` also switches it off for one run |
| `preview_rows` | 20 | rows shown per table in the page |
| `save_tables` | `true` | save returned tables as CSV |
| `max_table_cells` | 200000 | larger tables are previewed but not saved |
| `detail_calls` | 1 | how many repeated calls to record in full |
| `local_variables` | `true` | record each call's local variables |
| `local_preview_rows` | 8 | rows shown per local table |
| `library_functions` | `default` | `default` for the list in `src/utils/trace.py`, `[]` for none, or your own list such as `["statsmodels.tsa.stattools:adfuller"]` |
| `distinct_labels` | 8 | how many differently labelled calls get their own entry |
| `keep_runs` | 10 | how many trace folders to keep |

A step loaded from the cache has no calls inside it in the trace. Run with `--force` to trace the full computation. Tracing adds about 1–2 seconds to a 25-second full run.

## Step-by-step walkthrough (`python tools/walkthrough.py`)

A guided tour rather than a trace: it repeats the pipeline's steps by hand with an explanation for each, so it does not follow your code changes automatically (the run trace does).

Runs the whole pipeline one function at a time on a small example (`config/mini_config.yaml`: 12 macro variables in `data/mini/`, 2 scenarios, fewer transforms and lags), like stepping through it in a debugger. Takes about 20 seconds.

- Each step's result is saved as `outputs/mini_walkthrough/walkthrough/<NN>_<name>.csv` (or `.txt` for regression output).
- `walkthrough.html` in the same folder lets you click through the 25 steps. For each step it shows the call, the file and line of the function, its source code, the result tables and charts, the file the result was saved to, and which objects are now in memory (like RStudio's Environment pane).
- The last step re-runs `screen_first.run()` and `en_first.run()` and checks they give the same results as the step-by-step run.

To walk through your own data, point it at a small config: `python tools/walkthrough.py --config config/my_small_config.yaml` (`--skip-oos` leaves out the slower out-of-sample step).

## Folders

```
run_pipeline.py          entry point (selection, evaluation, scenarios)
run_oos.py               out-of-sample test with re-selection at each origin
app.py                   the app: upload files, review settings, run, see results (streamlit run app.py)
config/                  one YAML per data set
data/                    example (simulated) inputs
tools/simulate_data.py   rebuilds the example data
tools/walkthrough.py     step-by-step walkthrough (HTML) on the small example
tools/trace_to_html.py   turns a run trace into a clickable page
tools/new_data_setup.py  writes a config for a new data set from its files
tools/check_config.py    checks a config and its data before a run
tools/make_docs.py       builds the function reference (--watch, --check, --install-hook)
docs/                    reference.html and REFERENCE.md (generated; do not edit)
src/utils/trace.py       records every call during a run
src/utils/               config loading, file reading, package set-up, cache.py (hashing), docs.py (reference)
src/features/            transformations, stationarity tests, candidate features
src/selection/           screening.py, clustering.py, elastic_net.py (incl. stability selection)
src/modelling/           ardl.py (final model), search.py (shared final search), evaluate.py
src/pipelines/           screen_first.py (A), en_first.py (B)
src/validation/oos.py    out-of-sample test
src/models/              competing models: zoo.py (builds them all), ecm.py, ml.py, combination.py, foundation.py,
                         hybrid.py, deep.py (TiDE, N-HiTS, TFT, ARIMAX, Theta), regime.py (regime switching)
report/                  forecast_deck.qmd (Quarto deck), style_deck.py (design pass), reference.pptx, icons/
config/literature_workbook.xlsx  everything on the literature: models with expected signs, concepts, settings,
                         references (literature_references.csv is the source of the template's references sheet)
tools/make_deck.py       builds the styled deck for any run (also the button in the app's Results tab)
src/reporting/formula.py the recommended model as a formula, its next quarters and the Excel calculator
src/utils/saved.py       fitted models (save / load) and the saved copies of each run
src/models/simple.py     same-growth-as-a-driver models
src/models/autoardl.py   auto-ARDL (lags by information criterion), ARDL in levels, bounds test
src/models/horizon.py    horizon combinations (a different model for each block of quarters ahead)
src/reporting/           tables, charts, summary
outputs/<run_name>/      common/, screen_first/, en_first/, comparison/, summary.md
```

## What each approach does

**Shared start**
- Relevance filter, transformations (q/q, y/y, 2-quarter, acceleration, gap, real growth; changes and levels for rates).
  Moving windows are optional: `features.moving_average: [8, 12]` adds the average q/q growth over the last k quarters (`ma8`; `dma8` for rates), and `features.volatility: [4, 8]` adds the rolling standard deviation of q/q growth (`vol4`; `dvol4` for rates). Volatility gets its own expected sign, `features.volatility_sign` (default −1: more uncertainty, less growth), because a variable's own sign says nothing about the effect of its volatility. The 4- and 2-quarter averages are not offered: `ma4` = yoy/4 and `ma2` = dlog2/2, so they would duplicate existing transforms exactly. The Excel calculator rebuilds both kinds with plain formulas.
- ADF stationarity test, lags 0–4, giving about 2,300 candidate features on the example data.

**A: screen first** (`src/pipelines/screen_first.py`)
1. Regress loan growth on lagged growth and the dummies plus each candidate on its own. A candidate passes if p < 0.05 (Newey-West) and has the expected sign.
2. Keep the best form and lag per variable.
3. Cluster at |correlation| > 0.75 and keep one per cluster (an Ontario series is preferred if nearly as strong).
4. The top 25 go to the final search.

**B: Elastic Net first** (`src/pipelines/en_first.py`, `src/selection/elastic_net.py`)
1. Lagged loan growth and the dummies are left unpenalized: they are partialled out first, which is the same as glmnet's `penalty.factor = 0`.
2. X and y are standardized as glmnet does.
3. Coefficients are constrained to their expected sign (optional).
4. Lambda is the first point on the path where `target_n` features are selected. There is no cross-validation, so the result is deterministic for given data.
5. **Stability selection** (optional, on by default): repeat on 100 subsamples built from blocks of consecutive quarters, and keep features selected in at least 60% of them.
6. Group the picks at |correlation| > 0.75 and keep one per group.

**Shared final step** (`src/modelling/search.py`)
1. Try every combination of 1–3 candidates. A combination is kept only if:
   - every driver is significant;
   - every sign is as expected;
   - the VIF is below 5;
   - there is at most one driver per theme.
2. Rank the survivors by out-of-sample balance error in the rolling backtest.
3. Choose the simplest model within 5% of the best.

**Evaluation** (`src/modelling/evaluate.py`)
- Rolling backtest 1–9 quarters ahead, with dummy quarters excluded from the error.
- Residual diagnostics.
- A plain-language driver table.
- Forecasts under every scenario.
- An AR(1) benchmark, run alongside for comparison.

## Example results (simulated data, for checking the method only)

Average balance error (MAPE), dummy quarters excluded. "Rolling backtest" uses drivers chosen with all the data. "Out of sample" re-selects everything at each of 12 origins.

| Model | Family | Backtest 1–4Q | Backtest 1–12Q | Out of sample 1–4Q | 1–8Q | 1–12Q |
|---|---|---|---|---|---|---|
| AR(1) benchmark | Benchmark | 0.82% | 1.87% | 0.83% | 1.45% | 2.09% |
| Screen first | Data-driven | 0.43% | 0.71% | 0.60% | 0.91% | 1.37% |
| Elastic Net first | Data-driven | 0.43% | 0.93% | 0.53% | 1.05% | 1.61% |
| Literature: four-channel core | Literature | 0.46% | 1.02% | 0.47% | 0.77% | 1.11% |
| Literature: error-correction model | Literature | 0.55% | 0.94% | 0.59% | 0.90% | 1.10% |
| Hybrid: theory long run + data short run | Hybrid | 0.50% | 0.83% | 0.65% | 0.88% | 1.02% |
| Hybrid: 50/50 data + theory | Hybrid | 0.36% | 0.70% | 0.52% | 0.78% | 1.12% |
| Gradient boosting | Machine learning | 0.41% | 0.81% | 0.52% | 0.83% | 1.10% |
| Neural network (MLP) | Machine learning | 0.40% | 0.75% | 0.60% | 0.86% | 1.19% |

What this shows:
- With drivers chosen on all the data, the 50/50 hybrid and screen first look best.
- Out of sample, with everything re-selected, the models anchored in theory hold up best at 8–12 quarters. The best is the hybrid with a theory long run and data-chosen short-run drivers.
- Within the three-variable limit, screen first is best (35% below AR(1) over 1–12 quarters).
- The ECB BEAST specification does worse than AR(1) on this data.

With 12-quarter ranking, screen first chose Canadian mortgage-credit growth, Ontario income and residential building permits. Mortgage credit is close to the target itself; review whether credit aggregates should be allowed as drivers.

At `target_n = 10`, the full-sample Elastic Net spent 9 of its 10 picks on city house-price series. Stability selection and grouping reduced these to one house-price driver plus income.

## License

MIT, see [LICENSE](LICENSE). Free to use, change and share. Data downloaded from public sources stay under their providers' terms. Independent open-source project, not affiliated with or endorsed by any employer, regulator or data vendor.
