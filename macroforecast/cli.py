"""Command line for macroforecast (the same as Forecaster(...).fit(...)).

    macroforecast --target data/bank_data.csv --metadata data/macro_metadata.csv \\
                  --literature config/literature_workbook.xlsx
    macroforecast ... --set sample.horizon=8 --set en_first.target_n=8 --no-oos
    macroforecast ... --options my_options.yaml
    python -m macroforecast ...            (same thing, without installing the command)
"""
import argparse
import sys

from .api import run


def main(argv=None):
    a = argparse.ArgumentParser(prog="macroforecast", description="Run the loan balance forecasting project")
    a.add_argument("--target", required=True, help="balance file (csv/xlsx) with a date column and the balance")
    a.add_argument("--metadata", required=True, help="macro metadata file")
    a.add_argument("--literature", default=None, help="literature workbook (.xlsx); omit to switch those models off")
    a.add_argument("--target-column", default=None)
    a.add_argument("--panel", default=None, metavar="COLUMN",
                   help="column naming the bank when the file holds several banks (panel run)")
    a.add_argument("--target-form", choices=["auto", "dlog", "diff"], default="auto",
                   help="dlog = log growth (balances, volumes); diff = quarterly change (interest rates, spreads)")
    a.add_argument("--label", default=None, help="name used in titles, e.g. 'Ontario deposit balance'")
    a.add_argument("--units", default=None, help="units of the target, e.g. '$ millions' or '%%'")
    a.add_argument("--presets", choices=["fast", "medium", "best", "full"], default="full",
                   help="which model families compete (full = the example config as it is)")
    a.add_argument("--date-column", default=None)
    a.add_argument("--macro", action="append", default=[], metavar="NAME=FILE",
                   help="a scenario file, repeatable: --macro Baseline=base.csv --macro 'S4 Severe'=s4.csv")
    a.add_argument("--base-scenario", default=None)
    a.add_argument("--events", default=None, help="inorganic events file")
    a.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="any config setting, repeatable: --set sample.horizon=8")
    a.add_argument("--options", default=None, help="YAML file of settings (applied before --set)")
    a.add_argument("--base-config", default=None, help="start from this YAML config instead of the example")
    a.add_argument("--run-name", default=None)
    a.add_argument("--output-dir", default=None)
    a.add_argument("--approach", choices=["screen_first", "en_first", "bayes_first"], default=None)
    a.add_argument("--no-oos", action="store_true", help="skip the out-of-sample test")
    a.add_argument("--deck", action="store_true", help="also build the PowerPoint deck (needs Quarto)")
    a.add_argument("--force", action="store_true", help="recompute every step")
    a.add_argument("--no-cache", action="store_true")
    a.add_argument("--trace", action="store_true", help="write the detailed function trace")
    a.add_argument("--skip-check", action="store_true", help="do not stop on pre-flight errors")
    x = a.parse_args(argv)

    macro = None
    if x.macro:
        macro = {}
        for item in x.macro:
            name, _, path = item.partition("=")
            if not path:
                name, path = "Baseline", name
            macro[name.strip()] = path.strip()
    from .forecaster import PRESETS
    import copy
    options = copy.deepcopy(PRESETS[x.presets])
    options.setdefault("target", {})["transform"] = x.target_form
    if x.label:
        options["target"]["label"] = x.label
    if x.units:
        options["target"]["units"] = x.units
    if x.units or x.target_form == "diff":
        options["target"].update(display_scale=None, display_units=None)   # not the example's "$ billions"
    if x.options:
        from .api import parse_options, _merge
        options = _merge(options, parse_options(x.options))
    if x.set:
        from .api import parse_options, _merge
        options = _merge(options, parse_options(x.set))
    res = run(x.target, x.metadata, x.literature, target_column=x.target_column, date_column=x.date_column,
              macro=macro, base_scenario=x.base_scenario, events=x.events, options=options or None,
              base_config=x.base_config, run_name=x.run_name, output_dir=x.output_dir, oos=not x.no_oos,
              deck=x.deck, approach=x.approach, force=x.force, cache=not x.no_cache, trace=x.trace,
              check=not x.skip_check, panel=x.panel)
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
