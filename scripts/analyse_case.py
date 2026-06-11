# -*- coding: utf-8 -*-
"""二臂/三臂运行模式选择分析：按 counts_code 汇总 seed 平均，并进行多组时间-能耗偏好对比。

上游输入：
    outputs/four_case_framework/optimized_2B3B_time_energy.csv

核心指标：
    eta_T = (Cmax_2B - Cmax_3B) / Cmax_2B * 100
        表示三臂相对二臂的时间节省率，越大越支持三臂。

    eta_E = (Energy_3B - Energy_2B) / Energy_2B * 100
        表示三臂相对二臂的能耗增加率，越大越不支持三臂。

    S_lambda = eta_T - lambda * eta_E
        若 S_lambda > 0，说明在该偏好下三臂的时间收益足以抵消能耗代价，推荐三臂；
        若 S_lambda < 0，说明能耗代价更重要，推荐二臂。

lambda 的含义：
    lambda 越小，越重视时间；
    lambda 越大，越重视能耗。

输出：
    outputs/mode_decision/mode_decision_summary_optimized.csv

运行方式：
    python scripts/analyse_case.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "four_case_framework"
    / "optimized_2B3B_time_energy.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "outputs"
    / "mode_decision"
    / "mode_decision_summary_optimized.csv"
)


# 10 组时间-能耗偏好参数。
# lambda 越小越偏时间，lambda 越大越偏能耗。
# 输出顺序为：只考虑时间 → 多组 lambda 过渡 → 只考虑能耗。
LAMBDA_VALUES = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]


def lambda_label(lam: float) -> str:
    """把 lambda 转成适合 CSV 字段名的短标签。"""
    return str(lam).replace(".", "_")


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为浮点数。"""
    if value is None or value == "":
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为整数。"""
    if value is None or value == "":
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def round2(value: Any) -> Any:
    """保留两位小数；无穷大写为 inf。"""
    if isinstance(value, str):
        return value

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return ""

    if math.isinf(value_float):
        return "inf"

    if math.isnan(value_float):
        return ""

    return round(value_float, 2)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def read_csv(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_scenario(n1: int, n2: int, n3: int, n4: int) -> str:
    """根据任务结构给出简单标签，方便报告解释。"""
    single_arm_tasks = n1 + n2
    dual_arm_tasks = n3 + n4

    if n1 == n2 == n3 == n4:
        return "balanced"

    if n4 > max(n1, n2, n3):
        return "type4_dominant"

    if single_arm_tasks > dual_arm_tasks:
        return "single_arm_dominant"

    if dual_arm_tasks > single_arm_tasks:
        return "dual_arm_dominant"

    return "mixed_balanced"


def get_counts_code(row: dict, n1: int, n2: int, n3: int, n4: int) -> str:
    """从 CSV 中读取 counts_code；如果没有该列，则由 n1-n4 自动生成。"""
    counts_code = row.get("counts_code", "")

    if counts_code:
        return str(counts_code)

    return f"{n1}{n2}{n3}{n4}"


def calc_one_seed_metrics(row: dict) -> dict | None:
    """计算单个 seed 下的二/三臂对比指标。

    适配输入文件：
        outputs/four_case_framework/optimized_2B3B_time_energy.csv

    需要字段：
        n1, n2, n3, n4, total_tasks, seed
        cmax_2B_optimized
        energy_2B_optimized
        cmax_3B_optimized
        energy_3B_optimized
    """
    n1 = safe_int(row.get("n1"))
    n2 = safe_int(row.get("n2"))
    n3 = safe_int(row.get("n3"))
    n4 = safe_int(row.get("n4"))

    counts_code = get_counts_code(row, n1, n2, n3, n4)

    cmax_2 = safe_float(row.get("cmax_2B_optimized"))
    cmax_3 = safe_float(row.get("cmax_3B_optimized"))
    energy_2 = safe_float(row.get("energy_2B_optimized"))
    energy_3 = safe_float(row.get("energy_3B_optimized"))

    # optimized_2B3B_time_energy.csv 没有 status 字段，
    # 所以这里直接用数值是否有效判断该行是否可用。
    if cmax_2 <= 0 or cmax_3 <= 0 or energy_2 <= 0 or energy_3 <= 0:
        return None

    eta_t = (cmax_2 - cmax_3) / cmax_2 * 100
    eta_e = (energy_3 - energy_2) / energy_2 * 100

    if eta_e > 0:
        lambda_star = eta_t / eta_e
    elif eta_t > 0 and eta_e <= 0:
        lambda_star = math.inf
    else:
        lambda_star = 0.0

    return {
        "counts_code": counts_code,
        "seed": safe_int(row.get("seed")),
        "n1": n1,
        "n2": n2,
        "n3": n3,
        "n4": n4,
        "total_tasks": safe_int(row.get("total_tasks"), n1 + n2 + n3 + n4),
        "cmax_2B": cmax_2,
        "cmax_3B": cmax_3,
        "energy_2B": energy_2,
        "energy_3B": energy_3,
        "eta_T": eta_t,
        "eta_E": eta_e,
        "lambda_star": lambda_star,
    }


def recommend_time_only(mean_cmax_2b: float, mean_cmax_3b: float) -> str:
    """只考虑时间：谁 Cmax 更小选谁。"""
    if mean_cmax_3b < mean_cmax_2b:
        return "recommend_3arm"

    if mean_cmax_3b > mean_cmax_2b:
        return "recommend_2arm"

    return "similar_prefer_2arm"


def recommend_energy_only(mean_energy_2b: float, mean_energy_3b: float) -> str:
    """只考虑能耗：谁能耗更低选谁。"""
    if mean_energy_3b < mean_energy_2b:
        return "recommend_3arm"

    if mean_energy_3b > mean_energy_2b:
        return "recommend_2arm"

    return "similar_prefer_2arm"


def recommend_by_score(score: float) -> str:
    """根据综合收益判断推荐结果。"""
    if score > 0:
        return "recommend_3arm"

    if score < 0:
        return "recommend_2arm"

    return "similar_prefer_2arm"


def make_summary_row(group_rows: list[dict]) -> dict:
    """对同一个 counts_code 的多个 seed 取平均，生成一行汇总结果。"""
    first = group_rows[0]

    cmax_2_values = [r["cmax_2B"] for r in group_rows]
    cmax_3_values = [r["cmax_3B"] for r in group_rows]
    energy_2_values = [r["energy_2B"] for r in group_rows]
    energy_3_values = [r["energy_3B"] for r in group_rows]
    eta_t_values = [r["eta_T"] for r in group_rows]
    eta_e_values = [r["eta_E"] for r in group_rows]

    finite_lambda_values = [
        r["lambda_star"]
        for r in group_rows
        if not math.isinf(r["lambda_star"])
    ]

    mean_cmax_2 = mean(cmax_2_values)
    mean_cmax_3 = mean(cmax_3_values)
    mean_energy_2 = mean(energy_2_values)
    mean_energy_3 = mean(energy_3_values)
    mean_eta_t = mean(eta_t_values)
    mean_eta_e = mean(eta_e_values)

    # 这里的 lambda_star 是每个 seed 先算临界值，
    # 再对同一 counts_code 内取平均。
    # 如果所有 seed 都是三臂时间更短且能耗不增加，则 lambda_star 记为 inf。
    if finite_lambda_values:
        mean_lambda_star = mean(finite_lambda_values)
    else:
        mean_lambda_star = math.inf

    row = {
        "counts_code": first["counts_code"],
        "n1": first["n1"],
        "n2": first["n2"],
        "n3": first["n3"],
        "n4": first["n4"],
        "total_tasks": first["total_tasks"],
        "scenario_type": classify_scenario(
            first["n1"],
            first["n2"],
            first["n3"],
            first["n4"],
        ),
        "seed_count": len(group_rows),
        "seeds_used": ",".join(
            str(r["seed"])
            for r in sorted(group_rows, key=lambda x: x["seed"])
        ),

        "mean_cmax_2B": round2(mean_cmax_2),
        "mean_cmax_3B": round2(mean_cmax_3),
        "mean_energy_2B": round2(mean_energy_2),
        "mean_energy_3B": round2(mean_energy_3),

        "mean_eta_T_percent": round2(mean_eta_t),
        "mean_eta_E_percent": round2(mean_eta_e),
        "mean_lambda_star": round2(mean_lambda_star),

        "recommend_time_only": recommend_time_only(mean_cmax_2, mean_cmax_3),
    }

    # 动态加入 10 组 lambda 的综合收益和推荐结果。
    for lam in LAMBDA_VALUES:
        label = lambda_label(lam)
        score = mean_eta_t - lam * mean_eta_e

        row[f"S_lambda_{label}"] = round2(score)
        row[f"recommend_lambda_{label}"] = recommend_by_score(score)

    # “只考虑能耗”放在最后，作为能耗极限情况。
    row["recommend_energy_only"] = recommend_energy_only(
        mean_energy_2,
        mean_energy_3,
    )

    return row


def build_summary_rows(raw_rows: list[dict]) -> tuple[list[dict], int]:
    """按 counts_code 分组，只在同一种输入内部对 seed 取平均。"""
    groups: dict[tuple, list[dict]] = {}
    skipped = 0

    for row in raw_rows:
        metrics = calc_one_seed_metrics(row)

        if metrics is None:
            skipped += 1
            continue

        key = (
            metrics["counts_code"],
            metrics["n1"],
            metrics["n2"],
            metrics["n3"],
            metrics["n4"],
        )

        groups.setdefault(key, []).append(metrics)

    summary_rows = []

    for key in sorted(groups.keys()):
        group_rows = sorted(groups[key], key=lambda r: r["seed"])
        summary_rows.append(make_summary_row(group_rows))

    return summary_rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize optimized 2B/3B mode decision by counts_code "
            "with seed averaging and multiple lambda values."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="input CSV generated by optimized 2B/3B batch script",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="output summary CSV path",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)
    summary_rows, skipped = build_summary_rows(raw_rows)

    save_csv(summary_rows, output_path)

    print("Optimized mode decision summary finished.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Raw rows: {len(raw_rows)}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Skipped rows because of invalid numeric values: {skipped}")
    print("Lambda values used:", ", ".join(str(x) for x in LAMBDA_VALUES))
    print("Note: each counts_code is summarized independently; no overall averaging is performed.")


if __name__ == "__main__":
    main()
