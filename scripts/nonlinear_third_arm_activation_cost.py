# -*- coding: utf-8 -*-
"""非线性规划拓展 3：考虑第三臂固定启用成本的模式切换优化。

输入：
    outputs/mode_decision/mode_decision_summary_basic.csv

输出：
    outputs/nonlinear_programming/third_arm_activation_cost/
        third_arm_activation_cost_parameters.csv
        third_arm_activation_cost_threshold.csv
        third_arm_activation_cost_switching_result.csv
        third_arm_activation_cost_summary.csv

研究目标：
    在原有三臂综合收益 S = eta_T - lambda * eta_E 的基础上，
    加入第三臂固定启用成本 K * Phi(p,r)，分析不同任务结构和不同能耗权重下，
    二臂/三臂模式的切换边界。

运行：
    python scripts/nonlinear_third_arm_activation_cost.py
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
    / "mode_decision"
    / "mode_decision_summary_basic.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "nonlinear_programming"
    / "third_arm_activation_cost"
)

# 和当前 mode_decision_summary_basic.csv 使用的 lambda 保持一致
LAMBDA_VALUES = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]

# 第三臂固定启用成本 K，单位可以理解为“综合收益百分点评价惩罚”
ACTIVATION_COST_LEVELS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25]

# 任务规模归一化基准
N_MAX = 8.0

# 结构成本因子参数
SIZE_WEIGHT = 0.20
IMBALANCE_WEIGHT = 0.25
TYPE4_WEIGHT = 0.15

# 四类任务比例不均衡度的归一化上界
# 当某一类比例为 1，其他为 0 时：
# sum((p_i-0.25)^2)=0.75
IMBALANCE_MAX = 0.75


def lambda_label(lam: float) -> str:
    if abs(lam - int(lam)) < 1e-9:
        return str(int(lam))
    return str(lam).replace(".", "_")


def lambda_display(lam: float) -> str:
    if abs(lam - int(lam)) < 1e-9:
        return str(int(lam))
    return str(lam)


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def round2(value: float) -> float:
    return round(float(value), 2)


def round4(value: float) -> float:
    return round(float(value), 4)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def get_n(row: dict) -> int:
    if "total_tasks" in row:
        return safe_int(row["total_tasks"])
    if "n" in row:
        return safe_int(row["n"])
    return (
        safe_int(row["n1"])
        + safe_int(row["n2"])
        + safe_int(row["n3"])
        + safe_int(row["n4"])
    )


def get_eta_t(row: dict) -> float:
    if "mean_eta_T_percent" in row:
        return safe_float(row["mean_eta_T_percent"])
    return safe_float(row.get("mean_eta_T", 0.0))


def get_eta_e(row: dict) -> float:
    if "mean_eta_E_percent" in row:
        return safe_float(row["mean_eta_E_percent"])
    return safe_float(row.get("mean_eta_E", 0.0))


def get_proportions(row: dict) -> tuple[float, float, float, float, float]:
    n1 = safe_float(row["n1"])
    n2 = safe_float(row["n2"])
    n3 = safe_float(row["n3"])
    n4 = safe_float(row["n4"])

    total = n1 + n2 + n3 + n4

    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    p1 = n1 / total
    p2 = n2 / total
    p3 = n3 / total
    p4 = n4 / total
    r = total / N_MAX

    return p1, p2, p3, p4, r


def imbalance_index(p1: float, p2: float, p3: float, p4: float) -> float:
    """任务结构不均衡度，归一化到大约 0~1。"""
    raw = (
        (p1 - 0.25) ** 2
        + (p2 - 0.25) ** 2
        + (p3 - 0.25) ** 2
        + (p4 - 0.25) ** 2
    )
    return raw / IMBALANCE_MAX


def activation_cost_factor(p1: float, p2: float, p3: float, p4: float, r: float) -> float:
    """第三臂固定启用成本的任务结构修正因子。

    Phi(p,r)=1 + a*r^2 + b*I(p) + c*p4^2

    含义：
        r^2：任务规模越大，第三臂调度和维护复杂度越高；
        I(p)：任务结构越不均衡，调度适配成本越高；
        p4^2：第4类任务占比越高，可能带来更高抓取/搬运复杂度。
    """
    imb = imbalance_index(p1, p2, p3, p4)

    return (
        1.0
        + SIZE_WEIGHT * r * r
        + IMBALANCE_WEIGHT * imb
        + TYPE4_WEIGHT * p4 * p4
    )


def recommendation(score: float) -> str:
    if score > 0:
        return "recommend_3arm"
    if score < 0:
        return "recommend_2arm"
    return "similar_prefer_2arm"


def build_threshold_rows(rows: list[dict]) -> list[dict]:
    """计算每个任务结构在每个 lambda 下可承受的最大固定启用成本 K*。"""
    result_rows = []

    for row in rows:
        p1, p2, p3, p4, r = get_proportions(row)
        phi = activation_cost_factor(p1, p2, p3, p4, r)

        eta_t = get_eta_t(row)
        eta_e = get_eta_e(row)
        n = get_n(row)

        for lam in LAMBDA_VALUES:
            base_score = eta_t - lam * eta_e

            if base_score > 0 and phi > 0:
                max_affordable_k = base_score / phi
            else:
                max_affordable_k = 0.0

            result_rows.append(
                {
                    "counts_code": row["counts_code"],
                    "scenario_type": row.get("scenario_type", ""),
                    "total_tasks": n,
                    "n1": row["n1"],
                    "n2": row["n2"],
                    "n3": row["n3"],
                    "n4": row["n4"],
                    "p1": round4(p1),
                    "p2": round4(p2),
                    "p3": round4(p3),
                    "p4": round4(p4),
                    "r": round4(r),
                    "lambda": lambda_display(lam),
                    "lambda_label": lambda_label(lam),
                    "eta_T_percent": round2(eta_t),
                    "eta_E_percent": round2(eta_e),
                    "base_score_without_cost": round2(base_score),
                    "activation_cost_factor_phi": round4(phi),
                    "max_affordable_activation_cost_K": round2(max_affordable_k),
                    "zero_cost_recommendation": recommendation(base_score),
                    "threshold_interpretation": (
                        "3-arm is recommended when actual K is smaller than this threshold."
                    ),
                }
            )

    return result_rows


def build_switching_rows(rows: list[dict]) -> list[dict]:
    """枚举不同 K 下的模式切换结果。"""
    result_rows = []

    for row in rows:
        p1, p2, p3, p4, r = get_proportions(row)
        phi = activation_cost_factor(p1, p2, p3, p4, r)

        eta_t = get_eta_t(row)
        eta_e = get_eta_e(row)
        n = get_n(row)

        for lam in LAMBDA_VALUES:
            base_score = eta_t - lam * eta_e

            for k in ACTIVATION_COST_LEVELS:
                activation_penalty = k * phi
                adjusted_score = base_score - activation_penalty

                result_rows.append(
                    {
                        "counts_code": row["counts_code"],
                        "scenario_type": row.get("scenario_type", ""),
                        "total_tasks": n,
                        "n1": row["n1"],
                        "n2": row["n2"],
                        "n3": row["n3"],
                        "n4": row["n4"],
                        "lambda": lambda_display(lam),
                        "lambda_label": lambda_label(lam),
                        "activation_cost_K": k,
                        "eta_T_percent": round2(eta_t),
                        "eta_E_percent": round2(eta_e),
                        "base_score_without_cost": round2(base_score),
                        "activation_cost_factor_phi": round4(phi),
                        "activation_penalty": round2(activation_penalty),
                        "adjusted_score_with_cost": round2(adjusted_score),
                        "recommendation_with_cost": recommendation(adjusted_score),
                    }
                )

    return result_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary_rows(switching_rows: list[dict]) -> list[dict]:
    """按 lambda 和 K 汇总推荐三臂比例。"""
    groups: dict[tuple, list[dict]] = {}

    for row in switching_rows:
        key = (row["lambda"], row["lambda_label"], row["activation_cost_K"])
        groups.setdefault(key, []).append(row)

    summary_rows = []

    def sort_key(item):
        lam, label, k = item[0]
        return (safe_float(lam), safe_float(k))

    for key, rows in sorted(groups.items(), key=sort_key):
        lam, label, k = key

        case_count = len(rows)
        rec_3arm_count = sum(
            1 for r in rows if r["recommendation_with_cost"] == "recommend_3arm"
        )
        rec_2arm_count = sum(
            1 for r in rows if r["recommendation_with_cost"] == "recommend_2arm"
        )

        adjusted_scores = [safe_float(r["adjusted_score_with_cost"]) for r in rows]

        summary_rows.append(
            {
                "lambda": lam,
                "lambda_label": label,
                "activation_cost_K": k,
                "case_count": case_count,
                "recommend_3arm_count": rec_3arm_count,
                "recommend_2arm_count": rec_2arm_count,
                "recommend_3arm_ratio_percent": round2(rec_3arm_count / case_count * 100),
                "mean_adjusted_score": round2(mean(adjusted_scores)),
                "min_adjusted_score": round2(min(adjusted_scores)),
                "max_adjusted_score": round2(max(adjusted_scores)),
                "note": "As K or lambda increases, 3-arm recommendation ratio should decrease.",
            }
        )

    return summary_rows


def build_parameter_rows() -> list[dict]:
    rows = []

    rows.append(
        {
            "parameter": "activation_cost_K",
            "value": ",".join(str(k) for k in ACTIVATION_COST_LEVELS),
            "meaning": "normalized fixed activation cost levels for enabling the third arm",
        }
    )

    rows.append(
        {
            "parameter": "lambda_values",
            "value": ",".join(lambda_display(lam) for lam in LAMBDA_VALUES),
            "meaning": "energy penalty weights",
        }
    )

    rows.append(
        {
            "parameter": "activation_cost_factor",
            "value": "Phi(p,r)=1+0.20*r^2+0.25*I(p)+0.15*p4^2",
            "meaning": "nonlinear task-structure-dependent cost correction factor",
        }
    )

    rows.append(
        {
            "parameter": "I(p)",
            "value": "sum((p_i-0.25)^2)/0.75",
            "meaning": "normalized task-structure imbalance index",
        }
    )

    rows.append(
        {
            "parameter": "decision_rule",
            "value": "recommend_3arm if eta_T-lambda*eta_E-K*Phi(p,r)>0 else recommend_2arm",
            "meaning": "mode switching rule with fixed third-arm activation cost",
        }
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Third-arm activation cost switching optimization."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="input mode_decision_summary_basic.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    rows = read_csv(input_path)

    threshold_rows = build_threshold_rows(rows)
    switching_rows = build_switching_rows(rows)
    summary_rows = build_summary_rows(switching_rows)
    parameter_rows = build_parameter_rows()

    save_csv(
        parameter_rows,
        output_dir / "third_arm_activation_cost_parameters.csv",
    )

    save_csv(
        threshold_rows,
        output_dir / "third_arm_activation_cost_threshold.csv",
    )

    save_csv(
        switching_rows,
        output_dir / "third_arm_activation_cost_switching_result.csv",
    )

    save_csv(
        summary_rows,
        output_dir / "third_arm_activation_cost_summary.csv",
    )

    print("Third-arm activation cost switching optimization finished.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Input rows: {len(rows)}")
    print(f"Threshold rows: {len(threshold_rows)}")
    print(f"Switching rows: {len(switching_rows)}")
    print("Generated:")
    print("  third_arm_activation_cost_parameters.csv")
    print("  third_arm_activation_cost_threshold.csv")
    print("  third_arm_activation_cost_switching_result.csv")
    print("  third_arm_activation_cost_summary.csv")
    print()
    print("Model:")
    print("  S = eta_T - lambda * eta_E - K * Phi(p,r)")
    print("  Phi(p,r)=1+0.20*r^2+0.25*I(p)+0.15*p4^2")
    print("  recommend_3arm if S>0 else recommend_2arm")


if __name__ == "__main__":
    main()