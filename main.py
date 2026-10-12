"""CALPHAD 虚拟高通量实验室 - 主入口"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.calculator import CALPHADCalculator
from src.data_generator import DataGenerator, ParameterSpace
from src.data_loader import DataLoader
from src.parallel_executor import HPCConfig, ParallelExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CALPHAD 虚拟高通量实验室 - 生成 7000 系铝合金虚拟数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试模式（100个点）
  python main.py --test

  # 使用128核心工作站配置
  python main.py --hpc workstation_128c

  # 自定义参数
  python main.py --workers 100 --batch-size 5000 --max-points 50000
        """,
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/COST507.pycalphad.tdb",
        help="COST507 TDB 数据库文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/virtual_bds.csv",
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--source-data",
        type=str,
        default="examples/sample_input.xlsx",
        help="Optional workbook used to infer composition and temperature ranges",
    )
    parser.add_argument(
        "--top-temp-levels",
        type=int,
        default=5,
        help="从原始数据中选择的温度高频档位数量",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="最大计算点数（用于测试）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数（默认自动检测）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每批次处理的点数（默认自动优化）",
    )
    parser.add_argument(
        "--hpc",
        type=str,
        choices=["auto", "workstation_128c", "workstation_64c", "workstation_32c"],
        default="auto",
        help="HPC配置预设（针对不同工作站优化）",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：仅计算100个点",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从最近的检查点恢复计算",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260402,
        help="LHS 采样随机种子",
    )
    parser.add_argument(
        "--target-success",
        type=int,
        default=None,
        help="目标成功样本数（达到后停止，优先于 --max-points 语义）",
    )
    parser.add_argument(
        "--round-points",
        type=int,
        default=None,
        help="目标成功模式下每轮采样点数（默认自动估算）",
    )
    return parser.parse_args()


def _recover_and_filter_success(
    results_df: pd.DataFrame,
    executor: ParallelExecutor,
) -> tuple[pd.DataFrame, int, int]:
    """对失败点重算并筛选最终成功样本。

    Returns:
        tuple[pd.DataFrame, int, int]:
            - 最终成功样本
            - 失败样本数（重算后仍失败）
            - 相平衡异常剔除数
    """
    sol_cols = [f"sol_{p}_fraction" for p in DataGenerator.TARGET_PHASES]
    age_cols = [f"age_{p}_fraction" for p in DataGenerator.TARGET_PHASES]

    invalid_balance_mask = pd.Series(False, index=results_df.index)
    if set(sol_cols).issubset(results_df.columns) and set(age_cols).issubset(
        results_df.columns
    ):
        sol_sum = results_df[sol_cols].sum(axis=1)
        age_sum = results_df[age_cols].sum(axis=1)
        invalid_balance_mask = (
            (sol_sum < 0.99) | (sol_sum > 1.01) | (age_sum < 0.99) | (age_sum > 1.01)
        )

    if "status" in results_df.columns:
        failed_mask = results_df["status"] != "success"
        retry_mask = failed_mask | invalid_balance_mask
        failed_count = int(retry_mask.sum())
    else:
        retry_mask = invalid_balance_mask.copy()
        failed_count = 0

    if failed_count > 0:
        logger.warning(f"检测到并行失败样本: {failed_count}，开始串行重算...")
        retry_points: list[dict[str, float]] = []
        point_keys = {
            "idx",
            "ZN",
            "MG",
            "CU",
            "CR",
            "MN",
            "ZR",
            "TI",
            "FE",
            "SI",
            "solution_temp",
            "aging_temp",
        }
        for record in results_df.loc[retry_mask].to_dict(orient="records"):
            retry_point = {
                k: float(v)
                for k, v in record.items()
                if k in point_keys and pd.notna(v)
            }
            retry_points.append(retry_point)

        retry_results = executor.process_points(retry_points)
        retry_df = pd.DataFrame(retry_results)
        retry_success_df = (
            retry_df[retry_df["status"] == "success"]
            if "status" in retry_df.columns
            else pd.DataFrame()
        )

        if (
            not retry_success_df.empty
            and "idx" in results_df.columns
            and "idx" in retry_success_df.columns
        ):
            remaining_df = results_df.loc[~retry_mask]
            merged_df = pd.concat([remaining_df, retry_success_df], ignore_index=True)
            results_df = (
                merged_df.sort_values("idx")
                .drop_duplicates(subset=["idx"], keep="last")
                .reset_index(drop=True)
            )
            logger.info(f"串行重算成功恢复样本: {len(retry_success_df)}")
        else:
            logger.warning("串行重算未恢复成功样本，失败点将从最终数据集中剔除")

    if "status" in results_df.columns:
        final_success_df = results_df[results_df["status"] == "success"].copy()
        final_failed_count = int((results_df["status"] != "success").sum())
    else:
        final_success_df = results_df.copy()
        final_failed_count = 0

    if set(sol_cols).issubset(final_success_df.columns) and set(age_cols).issubset(
        final_success_df.columns
    ):
        final_sol_sum = final_success_df[sol_cols].sum(axis=1)
        final_age_sum = final_success_df[age_cols].sum(axis=1)
        final_balance_ok = (
            (final_sol_sum >= 0.99)
            & (final_sol_sum <= 1.01)
            & (final_age_sum >= 0.99)
            & (final_age_sum <= 1.01)
        )
        dropped_unbalanced = int((~final_balance_ok).sum())
        if dropped_unbalanced > 0:
            logger.warning(f"剔除相平衡异常样本: {dropped_unbalanced}")
        final_success_df = final_success_df.loc[final_balance_ok].copy()
    else:
        dropped_unbalanced = 0

    return final_success_df, final_failed_count, dropped_unbalanced


def run_generation(args: argparse.Namespace) -> None:
    """执行数据生成流程"""
    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"数据库文件不存在: {db_path}")
        logger.error("请确认热力学数据库文件位于 data/ 目录")
        sys.exit(1)

    max_points = 100 if args.test else args.max_points

    # 显示HPC配置
    hpc_config = HPCConfig.get_config(args.hpc)
    if args.hpc == "auto":
        hpc_config = HPCConfig.auto_detect()

    logger.info("=" * 70)
    logger.info("  CALPHAD 虚拟高通量实验室")
    logger.info("  7000系铝合金虚拟数据生成器")
    logger.info("=" * 70)
    logger.info(f"数据库: {db_path}")
    logger.info(f"最大点数: {max_points or '不限制'}")
    logger.info(f"目标成功样本数: {args.target_success or '未启用'}")
    logger.info(f"HPC配置: {hpc_config.get('description', args.hpc)}")
    logger.info(f"并行进程数: {args.workers or hpc_config.get('max_workers', '自动')}")
    logger.info(f"批次大小: {args.batch_size or hpc_config.get('batch_size', '自动')}")

    logger.info("\n步骤 1/5: 验证数据库...")
    try:
        calc = CALPHADCalculator(db_path)
        elements = calc.get_elements()
        phases = calc.get_phases()
        logger.info(f"  可用元素 ({len(elements)}): {', '.join(elements)}")
        logger.info(f"  可用相 ({len(phases)}): {', '.join(phases[:8])}...")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"数据库验证失败: {e}")
        sys.exit(1)

    logger.info("\n步骤 2/5: 生成 LHS 参数样本...")
    generator = DataGenerator()
    source_data_path = Path(args.source_data)
    if source_data_path.exists():
        try:
            loader = DataLoader(source_data_path)
            param_space_cfg = loader.generate_param_space_from_data(
                top_k_temp_levels=max(1, args.top_temp_levels)
            )

            default_space = ParameterSpace()
            composition_ranges = (
                param_space_cfg.get("composition") or default_space.composition_ranges
            )
            temperature_ranges = (
                param_space_cfg.get("temperature") or default_space.temperature_ranges
            )

            level_with_counts = param_space_cfg.get("temperature_levels", {})
            temp_levels = {
                key: [float(v) for v, _ in values]
                for key, values in level_with_counts.items()
                if values
            }
            temp_weights = {}
            for key, values in level_with_counts.items():
                if not values:
                    continue
                counts = [float(count) for _, count in values]
                count_sum = sum(counts)
                if count_sum <= 0:
                    continue
                temp_weights[key] = [count / count_sum for count in counts]

            generator = DataGenerator(
                ParameterSpace(
                    composition_ranges=composition_ranges,
                    temperature_ranges=temperature_ranges,
                    temperature_levels=temp_levels,
                    temperature_level_weights=temp_weights,
                )
            )

            if temp_levels:
                logger.info("  已启用原始数据高频温度档位采样:")
                for temp_key, values in temp_levels.items():
                    logger.info(f"    {temp_key}: {values}")
            else:
                logger.warning("  未从原始数据提取到高频温度档位，温度按连续区间采样")
        except Exception as exc:
            logger.warning(f"  读取原始数据失败，回退默认参数空间: {exc}")
    else:
        logger.warning(f"  原始数据文件不存在，回退默认参数空间: {source_data_path}")

    logger.info("\n步骤 3/5: 执行并行计算...")
    executor = ParallelExecutor(
        db_path=db_path,
        max_workers=args.workers,
        hpc_preset=args.hpc,
    )
    total_attempted = 0
    total_failed = 0
    total_dropped_unbalanced = 0

    if args.target_success is not None:
        if args.target_success <= 0:
            logger.error("--target-success 必须为正整数")
            sys.exit(1)
        if args.resume:
            logger.error("按成功样本模式暂不支持 --resume，请先关闭 --resume 后运行")
            sys.exit(1)

        round_points = args.round_points or max(1000, args.target_success * 2)
        success_chunks: list[pd.DataFrame] = []
        current_success = 0
        round_idx = 0
        seed_cursor = args.seed

        while current_success < args.target_success:
            round_idx += 1
            logger.info(
                f"  成功样本模式 - 第 {round_idx} 轮: "
                f"采样 {round_points} 点 (seed={seed_cursor})"
            )

            grid_points = generator.generate_lhs_samples(
                n_samples=round_points,
                seed=seed_cursor,
            )
            for point in grid_points:
                point["idx"] = int(point["idx"]) + total_attempted
            total_attempted += len(grid_points)

            results_df = executor.execute(
                grid_points,
                batch_size=args.batch_size,
                save_checkpoints=not args.test,
                existing_results=None,
            )
            round_success_df, round_failed, round_dropped = _recover_and_filter_success(
                results_df,
                executor,
            )
            success_chunks.append(round_success_df)
            total_failed += round_failed
            total_dropped_unbalanced += round_dropped

            current_success = int(sum(len(chunk) for chunk in success_chunks))
            logger.info(
                f"  第 {round_idx} 轮完成: 成功 {len(round_success_df)}，"
                f"累计成功 {current_success}/{args.target_success}"
            )
            seed_cursor += 1

        final_success_df = pd.concat(success_chunks, ignore_index=True)
        if "idx" in final_success_df.columns:
            final_success_df = final_success_df.sort_values("idx").reset_index(drop=True)
        final_success_df = final_success_df.iloc[: args.target_success].copy()
    else:
        sample_count = max_points or 50000
        grid_points = generator.generate_lhs_samples(
            n_samples=sample_count,
            seed=args.seed,
        )
        logger.info(f"  生成 {len(grid_points):,} 个参数点 (seed={args.seed})")

        # 检查是否从检查点恢复
        existing = None
        if args.resume:
            existing = executor.resume_from_checkpoint(grid_points)
            if existing is not None:
                logger.info(f"  从检查点恢复了 {len(existing)} 条记录")

        results_df = executor.execute(
            grid_points,
            batch_size=args.batch_size,
            save_checkpoints=not args.test,
            existing_results=existing,
        )
        final_success_df, total_failed, total_dropped_unbalanced = (
            _recover_and_filter_success(results_df, executor)
        )

    logger.info("\n步骤 4/5: 格式化与质量校验...")
    output_df = generator.format_output(final_success_df)

    report_text = generator.build_validation_report(output_df)
    report_text += (
        "\n\n计算执行统计:\n"
        f"- 最终成功样本: {len(final_success_df)}\n"
        f"- 失败样本(已剔除): {total_failed}\n"
        f"- 相平衡异常样本(已剔除): {total_dropped_unbalanced}\n"
        f"- 总尝试样本: {total_attempted if total_attempted > 0 else len(final_success_df) + total_failed}"
    )
    report_path = Path("output/validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    logger.info("\n步骤 5/5: 保存结果...")
    output_path = Path(args.output)
    generator.save(output_df, output_path)

    # 显示统计信息
    success_count = len(final_success_df)
    error_count = total_failed

    logger.info("\n" + "=" * 70)
    logger.info("  计算完成！")
    logger.info("=" * 70)
    logger.info(f"输出文件: {output_path}")
    logger.info(f"校验报告: {report_path}")
    logger.info(f"总记录数: {len(output_df):,}")
    logger.info(f"成功计算: {success_count:,}")
    if error_count > 0:
        logger.info(f"计算错误: {error_count:,}")
    logger.info("=" * 70)


def main() -> None:
    """主函数"""
    args = parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
