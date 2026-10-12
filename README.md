# CALPHAD Alloy Virtual Data Generator

This project generates virtual aluminum-alloy composition and heat-treatment samples, evaluates phase equilibria with pycalphad, and writes the results to CSV.

## Requirements

- Python 3.10 or newer
- The dependencies listed in `requirements.txt`

Install the dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the sample

The repository includes a small, synthetic workbook and a thermodynamic database. Run a 100-point smoke test with one worker:

```powershell
.\.venv\Scripts\python.exe main.py --test --workers 1 --source-data examples/sample_input.xlsx --output output/sample_virtual_data.csv
```

The generated CSV and validation report are written under `output/`. To generate a larger dataset, omit `--test` and set `--max-points`, for example:

```powershell
.\.venv\Scripts\python.exe main.py --max-points 1000 --workers 4 --source-data examples/sample_input.xlsx
```

## Input workbook

`examples/sample_input.xlsx` contains synthetic values only. A custom workbook can be supplied with `--source-data`. Composition columns use weight percent; process temperatures use degrees Celsius. Missing optional composition or process columns fall back to the built-in parameter ranges.

## Thermodynamic database

`data/COST507.pycalphad.tdb` is included for the example calculation. The database is a separate project asset and is covered by the software MIT license.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```
