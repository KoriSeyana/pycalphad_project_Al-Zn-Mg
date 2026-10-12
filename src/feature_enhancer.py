from __future__ import annotations

import os
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.calculator import CALPHADCalculator
from src.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageDef:
    prefix: str
    temp_col: str
    time_col: str


def _enhance_chunk_worker(
    db_path: str,
    records: list[dict[str, Any]],
    start_idx: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Process a chunk in a separate process and return original order index."""
    enhancer = ProcessFeatureEnhancer(db_path)
    df_chunk = pd.DataFrame.from_records(records)
    enhanced = enhancer.enhance_dataframe(df_chunk)
    return start_idx, enhanced.to_dict(orient="records")


class ProcessFeatureEnhancer:
    """Build CALPHAD-enhanced features from raw alloy/process data."""

    ELEMENTS = ["ZN", "MG", "CU", "CR", "MN", "ZR", "TI", "FE", "SI"]

    STAGES = [
        StageDef("sol1", "primary_solution_temp", "primary_solution_time"),
        StageDef("sol2", "secondary_solution_temp", "secondary_solution_time"),
        StageDef("sol3", "tertiary_solution_temp", "tertiary_solution_time"),
        StageDef("age1", "primary_aging_temp", "primary_aging_time"),
        StageDef("age2", "secondary_aging_temp", "secondary_aging_time"),
        StageDef("age3", "tertiary_aging_temp", "tertiary_aging_time"),
    ]

    def __init__(self, db_path: str | Path):
        self.calculator = CALPHADCalculator(db_path)
        self.extractor = FeatureExtractor()
        # cache by (composition tuple, temperature_c)
        self._cache: dict[tuple[tuple[float, ...], float], dict[str, float]] = {}
        self.calculation_phases = self._build_relevant_phase_list()

    def _build_relevant_phase_list(self) -> list[str]:
        """Limit equilibrium search space to phases used by feature engineering."""
        candidates = set()
        for aliases in self.extractor.PRECIPITATION_PHASE_MAP.values():
            candidates.update(aliases)
        candidates.update(self.extractor.MATRIX_PHASES.keys())

        available = set(self.calculator.get_phases())
        selected = [str(p) for p in sorted(candidates & available)]

        # Fallback to all phases if intersection is unexpectedly empty.
        if not selected:
            selected = [str(p) for p in sorted(available)]
        logger.info("用于特征增强的相数量: %s", len(selected))
        return selected

    @staticmethod
    def _norm_name(col: str) -> str:
        s = str(col).strip().lower()
        s = s.replace("℃", "c").replace("��", "c")
        s = s.replace("％", "%")
        s = re.sub(r"\s+", " ", s)
        return s

    def _build_column_map(self, df: pd.DataFrame) -> dict[str, str]:
        norm_to_original = {self._norm_name(c): c for c in df.columns}
        col_map: dict[str, str] = {}

        # composition columns: robust against mojibake in percent symbol
        for elem in self.ELEMENTS:
            elem_key = elem.lower()
            match = None
            for n, original in norm_to_original.items():
                if re.match(rf"^{elem_key}\s*/", n):
                    match = original
                    break
                if n == elem_key:
                    match = original
                    break
            if match:
                col_map[elem_key] = match

        # process columns
        pattern_map = {
            "primary_solution_temp": "primary solution temperature",
            "primary_solution_time": "primary solution time",
            "secondary_solution_temp": "secondary solution temperature",
            "secondary_solution_time": "secondary solution time",
            "tertiary_solution_temp": "tertiary solution temperature",
            "tertiary_solution_time": "tertiary solution time",
            "primary_aging_temp": "primary aging temperature",
            "primary_aging_time": "primary aging time",
            "secondary_aging_temp": "secondary aging temperature",
            "secondary_aging_time": "secondary aging time",
            "tertiary_aging_temp": "tertiary aging temperature",
            "tertiary_aging_time": "tertiary aging time",
        }

        for canonical, pattern in pattern_map.items():
            for n, original in norm_to_original.items():
                if pattern in n:
                    col_map[canonical] = original
                    break

        return col_map

    def _phase_at_temp(
        self, composition_frac: dict[str, float], temperature_c: float
    ) -> dict[str, float]:
        comp_tuple = tuple(round(composition_frac[e], 8) for e in self.ELEMENTS)
        key = (comp_tuple, round(float(temperature_c), 3))
        if key not in self._cache:
            # Remove strictly zero components to avoid over-constraining
            # equilibrium with many X(element)=0 conditions.
            active_composition = {
                e: x for e, x in composition_frac.items() if x > 1e-12
            }

            phase_fractions = self.calculator.calculate_equilibrium(
                active_composition,
                float(temperature_c) + 273.15,
                phases=self.calculation_phases,
            )
            # Some compositions/temperatures have no feasible solution under the
            # restricted phase set. Fall back to all phases to avoid false zeros.
            if not phase_fractions:
                try:
                    phase_fractions = self.calculator.calculate_equilibrium(
                        active_composition,
                        float(temperature_c) + 273.15,
                        phases=None,
                    )
                except RuntimeError:
                    # Keep empty dict; caller will write zero features for this stage.
                    phase_fractions = {}
            self._cache[key] = phase_fractions
        return self._cache[key]

    def enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = self._build_column_map(df)

        missing_elems = [e for e in self.ELEMENTS if e.lower() not in col_map]
        if missing_elems:
            raise ValueError(f"输入数据缺少成分列: {missing_elems}")

        expected_feature_keys = list(self.extractor.extract_phase_features({}).keys())
        out_rows: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            raw = row.to_dict()
            out: dict[str, Any] = dict(raw)

            composition_pct = {
                e: float(row[col_map[e.lower()]]) if pd.notna(row[col_map[e.lower()]]) else 0.0
                for e in self.ELEMENTS
            }
            composition_frac = {k: max(v, 0.0) / 100.0 for k, v in composition_pct.items()}

            stage_features: list[dict[str, float]] = []
            stage_times: list[float] = []
            stage_count = 0
            errors: list[str] = []

            for stage in self.STAGES:
                temp_col = col_map.get(stage.temp_col)
                time_col = col_map.get(stage.time_col)
                if temp_col is None:
                    continue

                temp_val = row[temp_col]
                if not pd.notna(temp_val) or float(temp_val) <= 0:
                    continue

                stage_count += 1
                time_val = float(row[time_col]) if time_col and pd.notna(row[time_col]) else 0.0
                out[f"{stage.prefix}_temperature_c"] = float(temp_val)
                out[f"{stage.prefix}_time_min"] = time_val

                try:
                    phase = self._phase_at_temp(composition_frac, float(temp_val))
                    feat = self.extractor.extract_phase_features(phase)
                    stage_features.append(feat)
                    stage_times.append(max(time_val, 0.0))
                    for k, v in feat.items():
                        out[f"{stage.prefix}_{k}"] = v
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{stage.prefix}: {exc}")
                    for k in expected_feature_keys:
                        out[f"{stage.prefix}_{k}"] = 0.0

            # Aggregate: final stage features
            if stage_features:
                final_feat = stage_features[-1]
                max_feat = {
                    k: max(f.get(k, 0.0) for f in stage_features)
                    for k in expected_feature_keys
                }
            else:
                final_feat = {k: 0.0 for k in expected_feature_keys}
                max_feat = {k: 0.0 for k in expected_feature_keys}

            total_time = sum(stage_times)
            if stage_features and total_time > 0:
                weighted_feat = {
                    k: sum(
                        feat.get(k, 0.0) * t
                        for feat, t in zip(stage_features, stage_times, strict=False)
                    )
                    / total_time
                    for k in expected_feature_keys
                }
            else:
                weighted_feat = {k: 0.0 for k in expected_feature_keys}

            for k, v in final_feat.items():
                out[k] = v
            for k, v in max_feat.items():
                out[f"max_{k}"] = v
            for k, v in weighted_feat.items():
                out[f"time_weighted_{k}"] = v

            out["calphad_stage_count"] = stage_count
            out["calphad_status"] = "error" if errors else "success"
            out["calphad_error"] = " | ".join(errors)

            out_rows.append(out)
            if (idx + 1) % 100 == 0:
                logger.info("特征增强进度: %s/%s", idx + 1, len(df))

        return pd.DataFrame(out_rows)

    def enhance_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sheet_name: str | int = 0,
        max_rows: int | None = None,
        workers: int | None = None,
        chunk_size: int = 50,
    ) -> pd.DataFrame:
        input_path = Path(input_path)
        output_path = Path(output_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        df = pd.read_excel(input_path, sheet_name=sheet_name)
        if max_rows:
            df = df.head(max_rows).copy()

        logger.info("读取输入数据: %s 行 x %s 列", len(df), len(df.columns))
        if workers is None:
            workers = max(1, min(61 if os.name == "nt" else os.cpu_count() or 1, (os.cpu_count() or 2) - 2))

        if workers <= 1:
            enhanced = self.enhance_dataframe(df)
        else:
            logger.info("并行增强: workers=%s, chunk_size=%s", workers, chunk_size)
            records = df.to_dict(orient="records")
            ordered_results: dict[int, list[dict[str, Any]]] = {}

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for start in range(0, len(records), chunk_size):
                    batch = records[start : start + chunk_size]
                    futures.append(
                        executor.submit(
                            _enhance_chunk_worker,
                            str(self.calculator.db_path),
                            batch,
                            start,
                        )
                    )

                done = 0
                for future in as_completed(futures):
                    start_idx, chunk_rows = future.result()
                    ordered_results[start_idx] = chunk_rows
                    done += 1
                    if done % 5 == 0 or done == len(futures):
                        logger.info("并行进度: %s/%s 个分块完成", done, len(futures))

            merged: list[dict[str, Any]] = []
            for start_idx in sorted(ordered_results.keys()):
                merged.extend(ordered_results[start_idx])
            enhanced = pd.DataFrame.from_records(merged)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.to_csv(output_path, index=False)
        logger.info("增强数据已保存: %s (%s 行, %s 列)", output_path, len(enhanced), len(enhanced.columns))
        return enhanced
