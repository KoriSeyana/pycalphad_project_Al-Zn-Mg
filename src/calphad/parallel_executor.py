"""高性能并行计算执行器，针对多核工作站优化（支持双路128核心）"""
from __future__ import annotations

import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# 每个子进程缓存一个计算器，避免每个点重复加载数据库
_WORKER_CALC_CACHE: dict[str, Any] = {}
PHASE_FAMILY_ALIASES: dict[str, list[str]] = {
    "FCC_A1": ["FCC_A1"],
    # COST507 在 Al-Zn-Mg 成分窗口下常以 MG2ZN11/TAU 形式返回 η 类析出。
    # 将其归并到 ETA，避免时效态 η 被系统性低估。
    "ETA": ["ETA", "ALCU_ETA", "MGZN2", "MG2ZN11", "MGZN", "MG2ZN3", "TAU"],
    "T_PHASE": ["T_PHASE", "QPHASE", "VPHASE", "ALMG_GAMMA"],
    "S_PHASE": ["S_PHASE", "SPHASE"],
    "AL3ZR": ["AL3ZR", "AL3ZR2", "AL3ZR5"],
    "MG2SI": ["MG2SI"],
    "AL_FE_SI": [
        "AL13FE4",
        "AL2FE",
        "AL5FE2",
        "ALFESI_ALPHA",
        "ALFESI_BETA",
        "ALFESI_GAMMA",
        "ALFESI_DELTA",
    ],
    "HCP_A3": ["HCP_A3"],
}

TARGET_PHASES = list(PHASE_FAMILY_ALIASES.keys())

# Stable phase subset for the actual equilibrium solve. A wider phase set can
# trigger pycalphad solver failures or collapse to empty results on this TDB.
CALCULATION_PHASES = [
    "FCC_A1",
    "ALCU_ETA",
    "MGZN2",
    "MG2ZN11",
    "MG2ZN3",
    "TAU",
    "SPHASE",
    "AL3ZR",
    "HCP_A3",
]


class HPCConfig:
    """高性能计算配置"""

    # 预设配置
    PRESETS = {
        "workstation_128c": {
            "description": "双路128核心工作站",
            "max_workers": 120,  # 留8个核心给系统
            "batch_size": 5000,
            "chunk_size": 100,
            "prefetch_factor": 4,
        },
        "workstation_64c": {
            "description": "64核心工作站",
            "max_workers": 60,
            "batch_size": 3000,
            "chunk_size": 50,
            "prefetch_factor": 3,
        },
        "workstation_32c": {
            "description": "32核心工作站",
            "max_workers": 30,
            "batch_size": 2000,
            "chunk_size": 30,
            "prefetch_factor": 2,
        },
        "auto": {
            "description": "自动检测",
            "max_workers": None,  # 自动计算
            "batch_size": None,
            "chunk_size": None,
            "prefetch_factor": 2,
        },
    }

    @classmethod
    def get_config(cls, preset: str = "auto") -> dict[str, Any]:
        """获取配置预设

        Args:
            preset: 配置预设名称

        Returns:
            dict: 配置字典
        """
        if preset in cls.PRESETS:
            return cls.PRESETS[preset].copy()
        return cls.PRESETS["auto"].copy()

    @classmethod
    def auto_detect(cls) -> dict[str, Any]:
        """自动检测系统配置并返回优化参数

        Returns:
            dict: 优化后的配置
        """
        total_cpus = cpu_count()
        logger.info(f"检测到 {total_cpus} 个CPU核心")

        # 根据核心数选择配置
        if total_cpus >= 100:  # 128核心系统
            config = cls.get_config("workstation_128c")
        elif total_cpus >= 50:  # 64核心系统
            config = cls.get_config("workstation_64c")
        elif total_cpus >= 24:  # 32核心系统
            config = cls.get_config("workstation_32c")
        else:
            # 通用配置
            config = {
                "description": f"自动检测 ({total_cpus}核心)",
                "max_workers": max(1, total_cpus - 2),
                "batch_size": max(500, total_cpus * 50),
                "chunk_size": max(10, total_cpus // 2),
                "prefetch_factor": 2,
            }

        config["detected_cpus"] = total_cpus
        return config


def _calculate_single_point(
    db_path: str, point: dict[str, float]
) -> dict[str, Any]:
    """单个计算点的执行函数（用于并行）

    Args:
        db_path: 数据库文件路径
        point: 包含成分和温度的参数点

    Returns:
        dict: 计算结果
    """
    from src.calphad.calculator import CALPHADCalculator

    try:
        if db_path not in _WORKER_CALC_CACHE:
            warnings.filterwarnings(
                "ignore",
                message="The type definition character .*",
                module="pycalphad.io.tdb",
            )
            _WORKER_CALC_CACHE[db_path] = CALPHADCalculator(db_path)
        calc = _WORKER_CALC_CACHE[db_path]

        composition = {
            k: v / 100.0
            for k, v in point.items()
            if k
            in {
                "ZN",
                "MG",
                "CU",
                "CR",
                "MN",
                "ZR",
                "TI",
                "FE",
                "SI",
            }
            and abs(v) > 1e-12
        }

        # 温度口径：摄氏输入，内部统一转开尔文。
        sol_temp_k = point.get("solution_temp", 460.0) + 273.15
        age_temp_k = point.get("aging_temp", 140.0) + 273.15

        sol_fractions = calc.calculate_equilibrium(
            composition,
            sol_temp_k,
            phases=CALCULATION_PHASES,
        )
        age_fractions = calc.calculate_equilibrium(
            composition,
            age_temp_k,
            phases=CALCULATION_PHASES,
        )

        result = dict(point)
        for family in TARGET_PHASES:
            aliases = PHASE_FAMILY_ALIASES[family]
            result[f"sol_{family}_fraction"] = float(
                sum(sol_fractions.get(alias, 0.0) for alias in aliases)
            )
            result[f"age_{family}_fraction"] = float(
                sum(age_fractions.get(alias, 0.0) for alias in aliases)
            )

        sol_sum = float(
            sum(result[f"sol_{family}_fraction"] for family in TARGET_PHASES)
        )
        age_sum = float(
            sum(result[f"age_{family}_fraction"] for family in TARGET_PHASES)
        )
        if not (0.99 <= sol_sum <= 1.01 and 0.99 <= age_sum <= 1.01):
            return {
                **point,
                "status": "error",
                "error_msg": (
                    f"phase fraction sum out of range: "
                    f"sol_sum={sol_sum:.6f}, age_sum={age_sum:.6f}"
                ),
            }

        result["sol_total_strengthening"] = (
            result.get("sol_ETA_fraction", 0.0)
            + result.get("sol_T_PHASE_fraction", 0.0)
            + result.get("sol_S_PHASE_fraction", 0.0)
        )
        result["age_total_strengthening"] = (
            result.get("age_ETA_fraction", 0.0)
            + result.get("age_T_PHASE_fraction", 0.0)
            + result.get("age_S_PHASE_fraction", 0.0)
        )
        result["status"] = "success"
        return result
    except Exception as e:
        return {**point, "status": "error", "error_msg": str(e)}


class ParallelExecutor:
    """高性能并行计算执行器"""

    def __init__(
        self,
        db_path: str | Path,
        max_workers: int | None = None,
        checkpoint_dir: str | Path = "output/checkpoints",
        hpc_preset: str = "auto",
    ):
        """初始化并行执行器

        Args:
            db_path: 数据库文件路径
            max_workers: 并行进程数，None则自动检测
            checkpoint_dir: 检查点保存目录
            hpc_preset: HPC配置预设 ("auto", "workstation_128c", "workstation_64c", "workstation_32c")
        """
        self.db_path = Path(db_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 获取HPC配置
        self.hpc_config = HPCConfig.auto_detect()
        if hpc_preset != "auto":
            self.hpc_config.update(HPCConfig.get_config(hpc_preset))

        # 设置worker数量
        self.max_workers = max_workers or self.hpc_config["max_workers"]
        # Windows 的 ProcessPoolExecutor 要求 max_workers <= 61
        if os.name == "nt":
            self.max_workers = min(self.max_workers, 61)
        self.batch_size = self.hpc_config["batch_size"]

        logger.info(f"HPC配置: {self.hpc_config['description']}")
        logger.info(f"  核心数: {self.hpc_config.get('detected_cpus', 'unknown')}")
        logger.info(f"  工作进程: {self.max_workers}")
        logger.info(f"  批次大小: {self.batch_size}")

    def execute(
        self,
        grid_points: list[dict[str, float]],
        batch_size: int | None = None,
        save_checkpoints: bool = True,
        existing_results: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """并行执行所有网格点计算

        Args:
            grid_points: 参数网格点列表
            batch_size: 每批次处理的点数，None则使用HPC配置
            save_checkpoints: 是否保存检查点
            existing_results: 用于续跑的已有结果

        Returns:
            pd.DataFrame: 计算结果
        """
        batch_size = batch_size or self.batch_size
        total_points = len(grid_points)

        logger.info("=" * 60)
        logger.info("开始高性并行计算")
        logger.info(f"  总计算点数: {total_points}")
        logger.info(f"  并行进程数: {self.max_workers}")
        logger.info(f"  批次大小: {batch_size}")
        logger.info("=" * 60)

        all_results = []
        start_idx = 0
        if existing_results is not None and not existing_results.empty:
            all_results.extend(existing_results.to_dict(orient="records"))
            start_idx = len(existing_results)
            logger.info(f"检测到续跑数据: {start_idx} 条")

        pending_points = grid_points[start_idx:]
        if not pending_points:
            logger.info("所有样本均已完成，无需继续计算")
            return pd.DataFrame(all_results)

        num_batches = (total_points + batch_size - 1) // batch_size
        pending_batches = (len(pending_points) + batch_size - 1) // batch_size

        for batch_idx in range(pending_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(pending_points))
            batch_points = pending_points[start_idx:end_idx]

            progress = (batch_idx + 1) / pending_batches * 100
            logger.info(
                f"处理批次 {batch_idx + 1}/{pending_batches} "
                f"[{progress:.1f}%] ({len(batch_points)} 点)"
            )

            batch_results = self._process_batch(batch_points)
            all_results.extend(batch_results)

            if save_checkpoints and (batch_idx % 5 == 0 or batch_idx == pending_batches - 1):
                self._save_checkpoint(all_results, batch_idx)

        df = pd.DataFrame(all_results)
        logger.info(f"计算完成: {len(df)} 条记录")
        return df

    def _process_batch(self, points: list[dict[str, float]]) -> list[dict]:
        """处理单个批次的计算

        Args:
            points: 本批次的参数点列表

        Returns:
            list: 计算结果列表
        """
        if self.max_workers <= 1:
            return [
                _calculate_single_point(str(self.db_path), point)
                for point in points
            ]

        results = []
        chunk_size = self.hpc_config.get("chunk_size", 50)

        # 使用max_tasks_per_child限制每个worker的任务数，防止内存泄漏
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            max_tasks_per_child=chunk_size * 10,
        ) as executor:
            # 分块提交任务以优化调度
            futures = {}
            for i, point in enumerate(points):
                future = executor.submit(
                    _calculate_single_point, str(self.db_path), point
                )
                futures[future] = i

            # 收集结果
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 1000 == 0:
                    logger.info(f"  完成: {completed}/{len(points)}")

                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    results.append(result)
                except Exception as e:
                    logger.error(f"计算错误: {e}")
                    point_idx = futures[future]
                    results.append(
                        {**points[point_idx], "status": "error", "error_msg": str(e)}
                    )

        return results

    def _save_checkpoint(self, results: list[dict], batch_num: int) -> None:
        """保存检查点

        Args:
            results: 当前所有结果
            batch_num: 批次编号
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{batch_num:04d}.csv"
        df = pd.DataFrame(results)
        df.to_csv(checkpoint_path, index=False)
        logger.info(f"检查点已保存: {checkpoint_path} ({len(results)} 条)")

    def load_checkpoint(self, batch_num: int) -> pd.DataFrame | None:
        """加载检查点

        Args:
            batch_num: 批次编号

        Returns:
            pd.DataFrame | None: 检查点数据，不存在返回 None
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{batch_num:04d}.csv"
        if checkpoint_path.exists():
            return pd.read_csv(checkpoint_path)
        return None

    def resume_from_checkpoint(self, grid_points: list[dict[str, float]]) -> pd.DataFrame | None:
        """从最近的检查点恢复计算

        Args:
            grid_points: 完整的参数网格点列表

        Returns:
            pd.DataFrame | None: 已完成的结果，如果没有检查点则返回None
        """
        # 查找最新的检查点
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.csv"))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        logger.info(f"从检查点恢复: {latest}")
        return pd.read_csv(latest)
