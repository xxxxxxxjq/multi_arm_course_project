# -*- coding: utf-8 -*-
"""批量生成“运筹学方法 + 启发式算法”下二臂/三臂的时间和能量参数 CSV。

作用：
1. 默认运行 16 种 Type 输入：
   1111/1112/1121/1211/2111/1122/1212/2112/
   1221/2121/2211/1222/2122/2212/2221/2222

2. 每种 Type 输入默认运行 3 次随机位置：
   seed = 0, 1, 2

3. 每次分别运行：
   双臂 运筹学方法 + 启发式算法
   三臂 运筹学方法 + 启发式算法

4. 只输出一个 CSV：
   outputs/four_case_framework/optimized_warmstart_2B3B_time_energy.csv

"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import OUTPUT_DIR  # noqa: E402
from common.experiment_runner import counts_to_block_types, parse_counts  # noqa: E402
from common.instance_generator import generate_instance  # noqa: E402
from common.mode_builder import build_modes  # noqa: E402
from common.utils import ensure_dirs  # noqa: E402
from scripts.run_optimized_case import (  # noqa: E402
    DEFAULT_COUNT_CODES,
    parse_count_code,
    parse_count_codes,
    parse_seeds,
    to_float_or_blank,
)


METHOD = "optimized_warmstart"


CSV_FIELDS = [
    "n1",
    "n2",
    "n3",
    "n4",
    "total_tasks",
    "seed",
    "cmax_2B_optimized_warmstart",
    "energy_2B_optimized_warmstart",
    "heuristic_initial_cmax_2B",
    "heuristic_initial_energy_2B",
    "nodes_visited_2B",
    "nodes_pruned_2B",
    "cmax_3B_optimized_warmstart",
    "energy_3B_optimized_warmstart",
    "heuristic_initial_cmax_3B",
    "heuristic_initial_energy_3B",
    "nodes_visited_3B",
    "nodes_pruned_3B",
    "calc_time_optimized_warmstart_s",
]


def save_csv(rows: list[dict], path: Path) -> None:
    ensure_dirs(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gp_value(result: dict, key: str) -> Any:
    return result.get("goal_programming", {}).get(key, "")


def solve_optimized_warmstart_case(counts_text: str, seed: int, arm_count: int) -> dict:
    from common.solver_optimized_warmstart import solve_schedule

    counts = parse_counts(counts_text)
    block_types = counts_to_block_types(counts)
    instance = generate_instance(
        seed=seed,
        block_types=block_types,
        arm_count=arm_count,
    )
    modes = build_modes(instance)
    return solve_schedule(instance, modes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run optimized exact branch-and-bound with heuristic initial incumbent."
    )
    parser.add_argument(
        "--counts-list",
        default="default",
        help=(
            "Counts list. Use default/all, 1111/1112/1121, 1111,1112,1121, "
            "or one case like 1,1,1,2."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated random seeds. Default: 0,1,2.",
    )
    args = parser.parse_args()

    count_codes = parse_count_codes(args.counts_list)
    seeds = parse_seeds(args.seeds)

    out_dir = OUTPUT_DIR / "four_case_framework"
    ensure_dirs(out_dir)
    output_path = out_dir / "optimized_warmstart_2B3B_time_energy.csv"

    rows = []
    total_runs = len(count_codes) * len(seeds)
    run_index = 0

    for raw_code in count_codes:
        counts_code, counts, counts_text = parse_count_code(raw_code)
        total_tasks = sum(counts.values())

        for seed in seeds:
            run_index += 1
            print(
                f"\n[{METHOD} {run_index}/{total_runs}] "
                f"counts={counts_code}, seed={seed}, total_tasks={total_tasks}",
                flush=True,
            )

            t0 = time.perf_counter()

            print(f"[{METHOD} {run_index}/{total_runs}] start 2B", flush=True)
            result_2b = solve_optimized_warmstart_case(counts_text, seed, 2)
            print(
                f"[{METHOD} {run_index}/{total_runs}] finish 2B: "
                f"status={result_2b.get('status', '')}, "
                f"cmax={result_2b.get('cmax', '')}, "
                f"energy={result_2b.get('total_energy', '')}",
                flush=True,
            )

            print(f"[{METHOD} {run_index}/{total_runs}] start 3B", flush=True)
            result_3b = solve_optimized_warmstart_case(counts_text, seed, 3)
            print(
                f"[{METHOD} {run_index}/{total_runs}] finish 3B: "
                f"status={result_3b.get('status', '')}, "
                f"cmax={result_3b.get('cmax', '')}, "
                f"energy={result_3b.get('total_energy', '')}",
                flush=True,
            )

            calc_time = time.perf_counter() - t0
            rows.append(
                {
                    "n1": counts[1],
                    "n2": counts[2],
                    "n3": counts[3],
                    "n4": counts[4],
                    "total_tasks": total_tasks,
                    "seed": seed,
                    "cmax_2B_optimized_warmstart": to_float_or_blank(result_2b.get("cmax")),
                    "energy_2B_optimized_warmstart": to_float_or_blank(result_2b.get("total_energy")),
                    "heuristic_initial_cmax_2B": to_float_or_blank(gp_value(result_2b, "heuristic_initial_cmax")),
                    "heuristic_initial_energy_2B": to_float_or_blank(gp_value(result_2b, "heuristic_initial_energy")),
                    "nodes_visited_2B": gp_value(result_2b, "nodes_visited"),
                    "nodes_pruned_2B": gp_value(result_2b, "nodes_pruned"),
                    "cmax_3B_optimized_warmstart": to_float_or_blank(result_3b.get("cmax")),
                    "energy_3B_optimized_warmstart": to_float_or_blank(result_3b.get("total_energy")),
                    "heuristic_initial_cmax_3B": to_float_or_blank(gp_value(result_3b, "heuristic_initial_cmax")),
                    "heuristic_initial_energy_3B": to_float_or_blank(gp_value(result_3b, "heuristic_initial_energy")),
                    "nodes_visited_3B": gp_value(result_3b, "nodes_visited"),
                    "nodes_pruned_3B": gp_value(result_3b, "nodes_pruned"),
                    "calc_time_optimized_warmstart_s": round(calc_time, 6),
                }
            )
            save_csv(rows, output_path)
            print(f"[{METHOD} {run_index}/{total_runs}] saved partial CSV: {output_path}", flush=True)
            print(f"[{METHOD} {run_index}/{total_runs}] calc_time={calc_time:.6f}s", flush=True)

    save_csv(rows, output_path)
    print("\nCSV generated.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
