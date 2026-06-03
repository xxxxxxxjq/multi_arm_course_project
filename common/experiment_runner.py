# -*- coding: utf-8 -*-
"""四个实验部分共用的运行器。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Sequence

from common.config import OUTPUT_DIR
from common.instance_generator import generate_instance
from common.mode_builder import build_modes
from common.utils import ensure_dirs, print_schedule, save_csv, save_json, schedule_to_rows
from common.visualization import plot_gantt, plot_instance, plot_metrics

TYPE_IDS = (1, 2, 3, 4)


def input_non_negative_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("输入无效：请输入非负整数，例如 0、1、2、3。")
            continue
        if value < 0:
            print("输入无效：数量不能为负。")
            continue
        return value


def prompt_counts() -> Dict[int, int]:
    print("\n请依次输入四种方块的数量。")
    print("说明：类型1/类型2为单臂任务，类型3/类型4为双臂协同任务。")
    while True:
        counts = {i: input_non_negative_int(f"类型{i}数量 = ") for i in TYPE_IDS}
        if sum(counts.values()) <= 0:
            print("四类方块总数不能为 0，请重新输入。")
            continue
        return counts


def parse_counts(text: str) -> Dict[int, int]:
    parts = text.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("--counts 后必须给 4 个整数，例如：--counts 1,2,1,1")
    values = [int(x) for x in parts]
    if any(x < 0 for x in values):
        raise ValueError("--counts 中的数量不能为负。")
    if sum(values) <= 0:
        raise ValueError("四类方块总数不能为 0。")
    return dict(zip(TYPE_IDS, values))


def counts_to_block_types(counts: Dict[int, int]) -> list[int]:
    block_types: list[int] = []
    for type_id in TYPE_IDS:
        block_types.extend([type_id] * counts.get(type_id, 0))
    return block_types


def make_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return random.SystemRandom().randint(1, 2_147_483_647)


def make_prefix(scenario_id: str, seed: int, task_count: int, counts: Dict[int, int]) -> str:
    counts_tag = "-".join(str(counts[i]) for i in TYPE_IDS)
    return f"{scenario_id}_seed{seed}_tasks{task_count}_types{counts_tag}"


def get_solver(algorithm: str):
    if algorithm == "basic":
        from common.solver_basic import solve_schedule
        return solve_schedule
    if algorithm == "optimized":
        from common.solver_optimized import solve_schedule
        return solve_schedule
    raise ValueError(f"未知算法：{algorithm}")


def run_scenario(
    scenario_id: str,
    scenario_name: str,
    arm_count: int,
    algorithm: str,
    counts: Dict[int, int] | None = None,
    seed: int | None = None,
    max_retries: int = 30,
) -> dict:
    """运行某一个独立实验部分。"""
    if counts is None:
        counts = prompt_counts()

    scenario_dir = OUTPUT_DIR / scenario_id
    instance_dir = scenario_dir / "instances"
    result_dir = scenario_dir / "results"
    ensure_dirs(instance_dir, result_dir)

    attempts = 1 if seed is not None else max(1, int(max_retries))
    solver = get_solver(algorithm)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        actual_seed = make_seed(seed)
        try:
            block_types = counts_to_block_types(counts)
            instance = generate_instance(seed=actual_seed, block_types=block_types, arm_count=arm_count)
            modes = build_modes(instance)
            result = solver(instance, modes)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if seed is not None:
                raise
    else:
        raise RuntimeError(f"连续尝试 {attempts} 次仍未得到可行解。") from last_error

    prefix = make_prefix(scenario_id, actual_seed, len(instance.tasks), counts)
    result["scenario_id"] = scenario_id
    result["scenario_name"] = scenario_name
    result["algorithm_key"] = algorithm
    result["arm_count"] = arm_count
    result["case_prefix"] = prefix
    result["type_counts"] = {str(k): int(v) for k, v in counts.items()}

    save_json(instance.to_dict(), instance_dir / f"instance_{prefix}.json")
    save_json(result, result_dir / f"result_{prefix}.json")
    save_csv(schedule_to_rows(result), result_dir / f"schedule_{prefix}.csv")

    fig_workspace = plot_instance(instance, result, result_dir / f"workspace_{prefix}.png")
    fig_gantt = plot_gantt(result, result_dir / f"gantt_{prefix}.png")
    fig_metrics = plot_metrics(result, result_dir / f"metrics_{prefix}.png")

    print_schedule(result)
    print("\n运行完成，已输出：")
    print(f"  实验部分：{scenario_name}")
    print(f"  文件前缀：{prefix}")
    print(f"  {fig_workspace}")
    print(f"  {fig_gantt}")
    print(f"  {fig_metrics}")
    print(f"  {result_dir / f'result_{prefix}.json'}")
    return result


def main_for_scenario(scenario_id: str, scenario_name: str, arm_count: int, algorithm: str, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=scenario_name)
    parser.add_argument("--counts", type=str, default=None, help="四类方块数量，例如：--counts 1,2,1,1")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；不填则每次随机生成")
    parser.add_argument("--max-retries", type=int, default=30, help="未指定 seed 时的最大重试次数")
    args = parser.parse_args(argv)
    counts = parse_counts(args.counts) if args.counts else prompt_counts()
    run_scenario(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        arm_count=arm_count,
        algorithm=algorithm,
        counts=counts,
        seed=args.seed,
        max_retries=args.max_retries,
    )
