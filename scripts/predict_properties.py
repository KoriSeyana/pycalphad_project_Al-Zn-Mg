"""Predict alloy properties with a model trained by pipeline_2_train_optimize."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from src.ml.data_processor import DataProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input alloy CSV or Excel file")
    parser.add_argument(
        "--model", type=Path, default=Path("results/best_model/optimized_model.pkl")
    )
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=Path("results/best_model/feature_columns.txt"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    processor = DataProcessor(config["data"])
    data = processor.load_data(str(args.input))
    features = args.feature_columns.read_text(encoding="utf-8").splitlines()
    missing = [name for name in features if name not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing model features: {missing}")

    with args.model.open("rb") as stream:
        model = pickle.load(stream)
    predictions = model.predict(data[features].to_numpy())
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    result = pd.read_csv(args.input) if args.input.suffix.lower() == ".csv" else pd.read_excel(args.input)
    for index, target in enumerate(config["data"]["target_cols"]):
        if index < predictions.shape[1]:
            result[f"predicted_{target}"] = predictions[:, index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main()
