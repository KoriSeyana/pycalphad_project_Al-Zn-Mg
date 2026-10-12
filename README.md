# Al-Zn-Mg Alloy Modeling Workflows

This repository contains reusable code for four connected tasks:

1. Generate virtual CALPHAD data and add CALPHAD phase features to alloy/process data.
2. Add composition-derived and physicochemical features, then train and evaluate property-prediction models.
3. Train a LightGBM phase-fraction surrogate to replace repeated equilibrium calculations.
4. Run multi-objective optimization with a trained property model and phase-fraction surrogate.

The repository includes the COST507 thermodynamic database and a small synthetic workbook for the virtual-data example. It does not include experimental spreadsheets, literature records, fitted models, optimization results, or paper materials. Supply your own authorized data locally. Keep credentials and private configuration files out of Git.

## Setup

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Workflow 1: Virtual data and CALPHAD features

Run a small virtual-data example:

```powershell
.\.venv\Scripts\python.exe main.py --test --workers 1 --source-data examples/sample_input.xlsx --output output/sample_virtual_data.csv
```

Add CALPHAD phase and process-stage features to a local workbook with alloy composition and heat-treatment columns:

```powershell
.\.venv\Scripts\python.exe scripts/enhance_features.py --input data/my_alloy_data.xlsx --sheet Sheet1 --output output/enhanced_alloy_data.csv --workers 1
```

`src/ml/data_processor.py` contains additional composition, physicochemical, indicator, and interaction feature generation used during supervised training.

## Workflow 2: Property model training and prediction

Set `DATA_CONFIG['data_path']` and `DATA_CONFIG['target_cols']` in `config.py` to match your local enhanced dataset and measured property columns. The input dataset must contain the target measurements; they are intentionally not distributed here.

```powershell
.\.venv\Scripts\python.exe pipelines/pipeline_2_train_optimize.py
```

The pipeline splits the dataset, trains and evaluates a selected regressor, runs PIMO hyperparameter optimization, and saves the resulting model under `results/`. Keep generated models and predictions local.

Predict measured properties for a separate input CSV with the same raw feature columns:

```powershell
.\.venv\Scripts\python.exe scripts/predict_properties.py --input data/new_alloys.csv --model results/best_model/optimized_model.pkl --output output/predictions.csv
```

Prediction runs the same physicochemical feature generation as training and uses the saved feature-column order.

## Workflow 3: Phase-fraction surrogate training

Generate or prepare a CSV containing the nine composition inputs, `solution_temp`, `aging_temp`, and CALPHAD phase-fraction targets. Then run:

```powershell
.\.venv\Scripts\python.exe scripts/train_surrogate_model.py --input output/my_calphad_data.csv --output-dir output/surrogate_models
```

The script writes a fitted model and its feature metadata under the selected output directory. These are generated artifacts and should not be committed.

## Workflow 4: Multi-objective optimization

After training a property model and phase-fraction surrogate, configure the model paths and decision-variable bounds in `pipelines/pipeline_3_multi_objective.py` and `config.py`, then run:

```powershell
.\.venv\Scripts\python.exe pipelines/pipeline_3_multi_objective.py
```

The optimization implementation uses `pymoo` NSGA-II and writes the Pareto solutions under `results/`. The property model must use features that the optimization feature assembler can construct from its decision variables and surrogate outputs; inspect the assembled feature names when adapting the workflow to a different dataset.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Data and licensing

Composition values in the examples are weight percent. Temperatures supplied by spreadsheets are generally Celsius; CALPHAD equilibrium calculations use Kelvin internally. The COST507 database is included for the example calculation and is covered by this repository's MIT license.
