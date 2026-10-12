from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_enhancer import ProcessFeatureEnhancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CALPHAD-enhanced dataset from raw alloy/process Excel data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/data.xlsx",
        help="Input Excel file containing raw composition + heat-treatment features",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Sheet2",
        help="Excel sheet name or index string, e.g. Sheet1 / Sheet2 / 0 / 1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/enhanced_dataset.csv",
        help="Output CSV path for enhanced dataset",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/COST507.pycalphad.tdb",
        help="TDB database path",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only process first N rows for quick validation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker processes for parallel enhancement",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Rows per worker chunk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    enhancer = ProcessFeatureEnhancer(args.db_path)
    enhanced = enhancer.enhance_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        sheet_name=(int(args.sheet) if args.sheet.isdigit() else args.sheet),
        max_rows=args.max_rows,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    logger.info("增强完成: %s 行 x %s 列", len(enhanced), len(enhanced.columns))


if __name__ == "__main__":
    main()
