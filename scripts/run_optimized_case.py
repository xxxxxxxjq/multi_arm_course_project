# -*- coding: utf-8 -*-
"""批量生成优化方法下二臂/三臂的时间和能量参数 CSV。

作用：
1. 默认运行 16 种 Type 输入：
   1111/1112/1121/1211/2111/1122/1212/2112/
   1221/2121/2211/1222/2122/2212/2221/2222

2. 每种 Type 输入默认运行 3 次随机位置：
   seed = 0, 1, 2

3. 每次分别运行：
   双臂优化方法
   三臂优化方法

4. 只输出一个 CSV：
   outputs/four_case_framework/optimized_2B3B_time_energy.csv
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

from common.experiment_runner import parse_counts, counts_to_block_types  # noqa: E402
from common.config import OUTPUT_DIR  # noqa: E402
from common.instance_generator import generate_instance  # noqa: E402
from common.mode_builder import build_modes  # noqa: E402
from common.utils import ensure_dirs  # noqa: E402


DEFAULT_COUNT_CODES = [
    "1111",
    "1112",
    "1121",
    "1211",
    "2111",
    "1122",
    "1212",
    "2112",
    "1221",
    "2121",
    "2211",
    "1222",
    "2122",
    "2212",
    "2221",
    "2222",
]


CSV_FIELDS = [
    "n1",
    "n2",
    "n3",
    "n4",
    "total_tasks",
    "seed",
    "cmax_2B_optimized",
    "energy_2B_optimized",
    "cmax_3B_optimized",
    "energy_3B_optimized",
    "calc_time_optimized_s",
]


def parse_count_code(code: str) -> tuple[str, dict[int, int], str]:
    """把 1112 或 1,1,1,2 转成 counts 字典和 counts_text。"""
    code = code.strip()

    if "," in code:
        counts = parse_counts(code)
        normalized_code = "".join(str(counts[i]) for i in range(1, 5))
        counts_text = ",".join(str(counts[i]) for i in range(1, 5))
        return normalized_code, counts, counts_text

    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"counts 输入必须类似 1112 或 1,1,1,2，当前为：{code}")

    counts = {
        1: int(code[0]),
        2: int(code[1]),
        3: int(code[2]),
        4: int(code[3]),
    }
    counts_text = ",".join(str(counts[i]) for i in range(1, 5))

    return code, counts, counts_text


def parse_count_codes(text: str) -> list[str]:
    """解析多个 counts 输入。

    支持：
    default
    all
    1111/1112/1121
    1111,1112,1121
    1,1,1,2
    """
    text = text.strip()

    if text.lower() in {"default", "all", ""}:
        return DEFAULT_COUNT_CODES

    if "/" in text:
        return [x.strip() for x in text.split("/") if x.strip()]

    parts = [x.strip() for x in text.split(",") if x.strip()]

    # 单组输入：1,1,1,2
    if len(parts) == 4 and all(len(x) == 1 and x.isdigit() for x in parts):
        return [",".join(parts)]

    # 多组输入：1111,1112,1121
    return parts


def parse_seeds(text: str) -> list[int]:
    """解析随机种子列表，例如 0,1,2。"""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def to_float_or_blank(value: Any) -> Any:
    """把数值转成四位小数；如果为空或无法转换，则输出空白。"""
    if value is None:
        return ""

    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return ""


def save_csv(rows: list[dict], path: Path) -> None:
    """保存为 Excel 友好的 UTF-8-SIG CSV。"""
    ensure_dirs(path.parent)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def solve_optimized_case(counts_text: str, seed: int, arm_count: int) -> dict:
    """运行指定机械臂数量下的优化方法。"""
    from common.solver_optimized import solve_schedule

    counts = parse_counts(counts_text)
    block_types = counts_to_block_types(counts)

    instance = generate_instance(
        seed=seed,
        block_types=block_types,
        arm_count=arm_count,
    )

    modes = build_modes(instance)
    result = solve_schedule(instance, modes)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量生成优化方法下二臂/三臂的 Cmax 和能耗 CSV"
    )

    parser.add_argument(
        "--counts-list",
        default="default",
        help=(
            "多个 counts 输入。默认 default 表示运行 "
            "1111/1112/1121/1211/2111/1122/1212/2112/"
            "1221/2121/2211/1222/2122/2212/2221/2222。"
            "也可写成 1111/1112/1121，或 1111,1112,1121。"
        ),
    )

    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="随机种子列表。默认 0,1,2，即每种 Type 输入生成 3 次随机位置。",
    )

    args = parser.parse_args()

    count_codes = parse_count_codes(args.counts_list)
    seeds = parse_seeds(args.seeds)

    out_dir = OUTPUT_DIR / "four_case_framework"
    ensure_dirs(out_dir)

    output_path = out_dir / "optimized_2B3B_time_energy.csv"

    rows = []
    total_runs = len(count_codes) * len(seeds)
    run_index = 0

    for raw_code in count_codes:
        counts_code, counts, counts_text = parse_count_code(raw_code)
        total_tasks = sum(counts.values())

        for seed in seeds:
            run_index += 1

            print(
                f"\n[optimized {run_index}/{total_runs}] "
                f"counts={counts_code}, seed={seed}, total_tasks={total_tasks}",
                flush=True,
            )

            # 记录本行实验开始时间。
            # 这里统计的是：
            # 同一个 counts 和 seed 下，
            # 2B 优化方法 + 3B 优化方法 两次求解的总耗时。
            t0 = time.perf_counter()

            print(
                f"[optimized {run_index}/{total_runs}] start 2B optimized",
                flush=True,
            )

            result_2b = solve_optimized_case(
                counts_text=counts_text,
                seed=seed,
                arm_count=2,
            )

            print(
                f"[optimized {run_index}/{total_runs}] finish 2B: "
                f"status={result_2b.get('status', '')}, "
                f"cmax={result_2b.get('cmax', '')}, "
                f"energy={result_2b.get('total_energy', '')}",
                flush=True,
            )

            print(
                f"[optimized {run_index}/{total_runs}] start 3B optimized",
                flush=True,
            )

            result_3b = solve_optimized_case(
                counts_text=counts_text,
                seed=seed,
                arm_count=3,
            )

            print(
                f"[optimized {run_index}/{total_runs}] finish 3B: "
                f"status={result_3b.get('status', '')}, "
                f"cmax={result_3b.get('cmax', '')}, "
                f"energy={result_3b.get('total_energy', '')}",
                flush=True,
            )

            # 记录本行实验结束时间。
            t1 = time.perf_counter()
            calc_time_optimized = t1 - t0

            rows.append({
                "n1": counts[1],
                "n2": counts[2],
                "n3": counts[3],
                "n4": counts[4],
                "total_tasks": total_tasks,
                "seed": seed,
                "cmax_2B_optimized": to_float_or_blank(result_2b.get("cmax")),
                "energy_2B_optimized": to_float_or_blank(result_2b.get("total_energy")),
                "cmax_3B_optimized": to_float_or_blank(result_3b.get("cmax")),
                "energy_3B_optimized": to_float_or_blank(result_3b.get("total_energy")),
                "calc_time_optimized_s": round(calc_time_optimized, 6),
            })

            save_csv(rows, output_path)

            print(
                f"[optimized {run_index}/{total_runs}] saved partial CSV: {output_path}",
                flush=True,
            )

            print(
                f"[optimized {run_index}/{total_runs}] "
                f"calc_time={calc_time_optimized:.6f}s",
                flush=True,
            )

    save_csv(rows, output_path)

    print("\nCSV 生成完成。")
    print(f"输出文件：{output_path}")
    print("说明：该 CSV 只包含优化方法下二臂和三臂的 Cmax、总能耗和每行计算耗时，不包含分析结论。")


if __name__ == "__main__":
    main()