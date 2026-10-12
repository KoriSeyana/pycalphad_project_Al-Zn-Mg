"""Train a LightGBM surrogate for CALPHAD phase fractions."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

INPUT_FEATURES = [
    "ZN", "MG", "CU", "CR", "MN", "ZR", "TI", "FE", "SI",
    "solution_temp", "aging_temp",
]
PHASE_NAMES = [
    "FCC_A1", "ETA", "T_PHASE", "S_PHASE", "AL3ZR", "MG2SI", "AL_FE_SI", "HCP_A3",
]
OUTPUT_FEATURES = [
    f"{stage}_{phase}_fraction"
    for stage in ("sol", "age")
    for phase in PHASE_NAMES
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CALPHAD feature CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/surrogate_models")
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = INPUT_FEATURES + OUTPUT_FEATURES
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    X = df[INPUT_FEATURES].to_numpy()
    y = df[OUTPUT_FEATURES].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=64,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.random_state,
            verbose=-1,
        ),
        n_jobs=-1,
    )
    started = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - started

    predictions = model.predict(X_val)
    val_r2 = float(r2_score(y_val, predictions, multioutput="uniform_average"))
    phase_fraction_sums = np.vstack(
        [predictions[:, :8].sum(axis=1), predictions[:, 8:].sum(axis=1)]
    )
    balance_rate = float(
        np.all((phase_fraction_sums >= 0.99) & (phase_fraction_sums <= 1.01), axis=0).mean()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "lightgbm_surrogate_model.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream)

    info = {
        "input_features": INPUT_FEATURES,
        "output_features": OUTPUT_FEATURES,
        "phase_names": PHASE_NAMES,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "validation_r2": val_r2,
        "validation_phase_balance_rate": balance_rate,
        "training_seconds": elapsed,
        "validation_mape_by_output": {},
    }
    for index, name in enumerate(OUTPUT_FEATURES):
        nonzero = y_val[:, index] > 1e-6
        info["validation_mape_by_output"][name] = (
            float(mean_absolute_percentage_error(y_val[nonzero, index], predictions[nonzero, index]))
            if nonzero.any()
            else None
        )
    (args.output_dir / "model_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(f"Saved model: {model_path}")
    print(f"Validation R2: {val_r2:.4f}")
    print(f"Phase balance within 0.99-1.01: {balance_rate:.1%}")


if __name__ == "__main__":
    main()
