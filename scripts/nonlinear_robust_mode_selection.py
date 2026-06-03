# -*- coding: utf-8 -*-
"""非线性规划拓展 4：随机扰动下的鲁棒模式选择优化。

输入优先级：
    1. outputs/four_case_framework/basic_2B3B_time_energy.csv
       该文件包含不同 seed / instance 下的二臂和三臂结果，适合计算波动性。
    2. 若原始文件不存在，则退化使用：
       outputs/mode_decision/mode_decision_summary_basic.csv
       但这种情况下标准差只能取 0，鲁棒性分析会变弱。

输出：
    outputs/nonlinear_programming/robust_mode_selection/
        robust_mode_selection_parameters.csv
        robust_mode_selection_result.csv
        robust_mode_selection_summary.csv
        robust_reliable_structures.csv

研究目标：
    原模型只看平均综合收益：
        S_lambda = eta_T - lambda * eta_E

    本模型进一步考虑随机任务位置或不同 seed 导致的波动：
        S_robust = mean(S_lambda) - gamma * std(S_lambda)

    其中 gamma 是风险厌恶系数。
    gamma 越大，决策越保守。

运行：
    python scripts/nonlinear_robust_mode_selection.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_RAW_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "four_case_framework"
    / "basic_2B3B_time_energy.csv"
)

DEFAULT_SUMMARY_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "mode_decision"
    / "mode_decision_summary_basic.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "nonlinear_programming"
    / "robust_mode_selection"
)

LAMBDA_VALUES = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]

# gamma 越大，鲁棒性越保守
GAMMA_VALUES = [0, 0.5, 1, 1.5, 2]


def lambda_label(lam: float) -> str:
    if abs(lam - int(lam)) < 1e-9:
        return str(int(lam))
    return str(lam).replace(".", "_")


def lambda_display(lam: float) -> str:
    if abs(lam - int(lam)) < 1e-9:
        return str(int(lam))
    return str(lam)


def gamma_display(gamma: float) -> str:
    if abs(gamma - int(gamma)) < 1e-9:
        return str(int(gamma))
    return str(gamma)


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


def find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    fieldname_set = set(fieldnames)

    for c in candidates:
        if c in fieldname_set:
            return c

    lower_map = {name.lower(): name for name in fieldnames}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def classify_scenario(n1: int, n2: int, n3: int, n4: int) -> str:
    if n1 == n2 == n3 == n4:
        return "balanced"

    if n4 > max(n1, n2, n3):
        return "type4_dominant"

    if n1 > max(n2, n3, n4) or n2 > max(n1, n3, n4):
        return "single_arm_dominant"

    if n3 > max(n1, n2, n4):
        return "dual_arm_dominant"

    small = n1 + n2
    large = n3 + n4

    if small == large:
        return "mixed_balanced"

    if large > small:
        return "dual_arm_dominant"

    return "single_arm_dominant"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std_sample(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def recommendation(score: float) -> str:
    if score > 0:
        return "recommend_3arm"
    if score < 0:
        return "recommend_2arm"
    return "similar_prefer_2arm"


def load_instance_rows_from_raw(raw_path: Path) -> list[dict] | None:
    """从原始 seed 级别结果中读取二臂/三臂 Cmax 和 Energy。"""
    if not raw_path.exists():
        return None

    rows = read_csv(raw_path)

    if not rows:
        return None

    fieldnames = list(rows[0].keys())

    cmax_2_col = find_column(
        fieldnames,
        [
            "cmax_2B_basic",
            "cmax_2arm",
            "cmax_2B",
            "Cmax_2B_basic",
        ],
    )

    cmax_3_col = find_column(
        fieldnames,
        [
            "cmax_3B_basic",
            "cmax_3arm",
            "cmax_3B",
            "Cmax_3B_basic",
        ],
    )

    energy_2_col = find_column(
        fieldnames,
        [
            "energy_2B_basic",
            "energy_2arm",
            "energy_2B",
            "Energy_2B_basic",
        ],
    )

    energy_3_col = find_column(
        fieldnames,
        [
            "energy_3B_basic",
            "energy_3arm",
            "energy_3B",
            "Energy_3B_basic",
        ],
    )

    status_2_col = find_column(
        fieldnames,
        [
            "status_2B_basic",
            "status_2arm",
            "status_2B",
        ],
    )

    status_3_col = find_column(
        fieldnames,
        [
            "status_3B_basic",
            "status_3arm",
            "status_3B",
        ],
    )

    required = [cmax_2_col, cmax_3_col, energy_2_col, energy_3_col]

    if not all(required):
        return None

    valid_status = {"OPTIMAL", "FEASIBLE", "SUCCESS", "success", "optimal", "feasible"}

    instance_rows = []

    for row in rows:
        if status_2_col and status_3_col:
            status_2 = row.get(status_2_col, "")
            status_3 = row.get(status_3_col, "")

            if status_2 not in valid_status or status_3 not in valid_status:
                continue

        n1 = safe_int(row.get("n1", 0))
        n2 = safe_int(row.get("n2", 0))
        n3 = safe_int(row.get("n3", 0))
        n4 = safe_int(row.get("n4", 0))

        total_tasks = safe_int(row.get("total_tasks", n1 + n2 + n3 + n4))

        cmax_2 = safe_float(row[cmax_2_col])
        cmax_3 = safe_float(row[cmax_3_col])
        energy_2 = safe_float(row[energy_2_col])
        energy_3 = safe_float(row[energy_3_col])

        if cmax_2 <= 0 or energy_2 <= 0:
            continue

        eta_t = (cmax_2 - cmax_3) / cmax_2 * 100.0
        eta_e = (energy_3 - energy_2) / energy_2 * 100.0

        counts_code = row.get("counts_code", f"{n1}{n2}{n3}{n4}")
        scenario_type = row.get("scenario_type", classify_scenario(n1, n2, n3, n4))

        instance_id = (
            row.get("scenario_id")
            or row.get("instance_id")
            or row.get("task_id")
            or row.get("seed")
            or ""
        )

        instance_rows.append(
            {
                "counts_code": counts_code,
                "scenario_type": scenario_type,
                "total_tasks": total_tasks,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "instance_id": instance_id,
                "eta_T_percent": eta_t,
                "eta_E_percent": eta_e,
                "data_source": "basic_2B3B_time_energy.csv",
            }
        )

    return instance_rows


def get_eta_t(row: dict) -> float:
    if "mean_eta_T_percent" in row:
        return safe_float(row["mean_eta_T_percent"])
    if "mean_eta_T" in row:
        return safe_float(row["mean_eta_T"])
    if "eta_T_percent" in row:
        return safe_float(row["eta_T_percent"])
    return 0.0


def get_eta_e(row: dict) -> float:
    if "mean_eta_E_percent" in row:
        return safe_float(row["mean_eta_E_percent"])
    if "mean_eta_E" in row:
        return safe_float(row["mean_eta_E"])
    if "eta_E_percent" in row:
        return safe_float(row["eta_E_percent"])
    return 0.0


def load_instance_rows_from_summary(summary_path: Path) -> list[dict] | None:
    """兜底：从均值文件读取。此时没有 seed 波动，标准差会退化为 0。"""
    if not summary_path.exists():
        return None

    rows = read_csv(summary_path)

    if not rows:
        return None

    instance_rows = []

    for row in rows:
        n1 = safe_int(row.get("n1", 0))
        n2 = safe_int(row.get("n2", 0))
        n3 = safe_int(row.get("n3", 0))
        n4 = safe_int(row.get("n4", 0))

        total_tasks = safe_int(row.get("total_tasks", row.get("n", n1 + n2 + n3 + n4)))
        counts_code = row.get("counts_code", f"{n1}{n2}{n3}{n4}")
        scenario_type = row.get("scenario_type", classify_scenario(n1, n2, n3, n4))

        instance_rows.append(
            {
                "counts_code": counts_code,
                "scenario_type": scenario_type,
                "total_tasks": total_tasks,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "instance_id": "summary_mean_only",
                "eta_T_percent": get_eta_t(row),
                "eta_E_percent": get_eta_e(row),
                "data_source": "mode_decision_summary_basic.csv",
            }
        )

    return instance_rows


def load_instance_rows(raw_path: Path, summary_path: Path) -> list[dict]:
    instance_rows = load_instance_rows_from_raw(raw_path)

    if instance_rows:
        return instance_rows

    print("Warning: raw seed-level file not available, fallback to summary file.")
    print("Warning: robustness standard deviation will be zero in fallback mode.")

    instance_rows = load_instance_rows_from_summary(summary_path)

    if instance_rows:
        return instance_rows

    raise FileNotFoundError("No usable input file found for robust optimization.")


def group_by_structure(instance_rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}

    for row in instance_rows:
        key = (
            row["counts_code"],
            row["scenario_type"],
            row["total_tasks"],
            row["n1"],
            row["n2"],
            row["n3"],
            row["n4"],
        )
        groups.setdefault(key, []).append(row)

    return groups


def build_result_rows(instance_rows: list[dict]) -> list[dict]:
    groups = group_by_structure(instance_rows)

    result_rows = []

    for key, rows in sorted(groups.items()):
        counts_code, scenario_type, total_tasks, n1, n2, n3, n4 = key

        eta_t_values = [safe_float(r["eta_T_percent"]) for r in rows]
        eta_e_values = [safe_float(r["eta_E_percent"]) for r in rows]

        mean_eta_t = mean(eta_t_values)
        mean_eta_e = mean(eta_e_values)
        std_eta_t = std_sample(eta_t_values)
        std_eta_e = std_sample(eta_e_values)

        for lam in LAMBDA_VALUES:
            score_values = [
                safe_float(r["eta_T_percent"]) - lam * safe_float(r["eta_E_percent"])
                for r in rows
            ]

            mean_score = mean(score_values)
            std_score = std_sample(score_values)

            for gamma in GAMMA_VALUES:
                robust_score = mean_score - gamma * std_score

                result_rows.append(
                    {
                        "counts_code": counts_code,
                        "scenario_type": scenario_type,
                        "total_tasks": total_tasks,
                        "n1": n1,
                        "n2": n2,
                        "n3": n3,
                        "n4": n4,
                        "lambda": lambda_display(lam),
                        "lambda_label": lambda_label(lam),
                        "gamma": gamma_display(gamma),
                        "sample_count": len(rows),
                        "mean_eta_T_percent": round2(mean_eta_t),
                        "std_eta_T_percent": round2(std_eta_t),
                        "mean_eta_E_percent": round2(mean_eta_e),
                        "std_eta_E_percent": round2(std_eta_e),
                        "mean_score": round2(mean_score),
                        "std_score": round2(std_score),
                        "robust_score": round2(robust_score),
                        "robust_recommendation": recommendation(robust_score),
                        "model_note": "robust_score = mean(S_lambda) - gamma * std(S_lambda)",
                    }
                )

    return result_rows


def build_summary_rows(result_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}

    for row in result_rows:
        key = (row["lambda"], row["lambda_label"], row["gamma"])
        groups.setdefault(key, []).append(row)

    def sort_key(item):
        lam, label, gamma = item[0]
        return (safe_float(lam), safe_float(gamma))

    summary_rows = []

    for key, rows in sorted(groups.items(), key=sort_key):
        lam, label, gamma = key
        case_count = len(rows)

        rec_3 = sum(1 for r in rows if r["robust_recommendation"] == "recommend_3arm")
        rec_2 = sum(1 for r in rows if r["robust_recommendation"] == "recommend_2arm")

        robust_scores = [safe_float(r["robust_score"]) for r in rows]
        mean_scores = [safe_float(r["mean_score"]) for r in rows]
        std_scores = [safe_float(r["std_score"]) for r in rows]

        summary_rows.append(
            {
                "lambda": lam,
                "lambda_label": label,
                "gamma": gamma,
                "case_count": case_count,
                "recommend_3arm_count": rec_3,
                "recommend_2arm_count": rec_2,
                "recommend_3arm_ratio_percent": round2(rec_3 / case_count * 100),
                "mean_of_mean_score": round2(mean(mean_scores)),
                "mean_std_score": round2(mean(std_scores)),
                "mean_robust_score": round2(mean(robust_scores)),
                "min_robust_score": round2(min(robust_scores)),
                "max_robust_score": round2(max(robust_scores)),
                "note": "As gamma increases, the decision becomes more conservative.",
            }
        )

    return summary_rows


def build_reliable_structure_rows(result_rows: list[dict]) -> list[dict]:
    """汇总每个结构在不同 gamma 下可稳定承受的最大 lambda。"""
    groups: dict[tuple, list[dict]] = {}

    for row in result_rows:
        key = (
            row["counts_code"],
            row["scenario_type"],
            row["total_tasks"],
            row["n1"],
            row["n2"],
            row["n3"],
            row["n4"],
            row["gamma"],
        )
        groups.setdefault(key, []).append(row)

    reliable_rows = []

    def sort_key(item):
        counts_code, scenario_type, total_tasks, n1, n2, n3, n4, gamma = item[0]
        return (safe_float(gamma), str(counts_code))

    for key, rows in sorted(groups.items(), key=sort_key):
        counts_code, scenario_type, total_tasks, n1, n2, n3, n4, gamma = key

        rows_sorted = sorted(rows, key=lambda r: safe_float(r["lambda"]))

        positive_lambdas = [
            safe_float(r["lambda"])
            for r in rows_sorted
            if r["robust_recommendation"] == "recommend_3arm"
        ]

        if positive_lambdas:
            max_lambda_still_3arm = max(positive_lambdas)
        else:
            max_lambda_still_3arm = 0.0

        robust_scores = [safe_float(r["robust_score"]) for r in rows_sorted]

        # 记录几个常用 lambda 下的鲁棒得分，方便报告直接引用
        score_map = {
            str(r["lambda"]): safe_float(r["robust_score"])
            for r in rows_sorted
        }

        reliable_rows.append(
            {
                "counts_code": counts_code,
                "scenario_type": scenario_type,
                "total_tasks": total_tasks,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "gamma": gamma,
                "recommend_3arm_lambda_count": len(positive_lambdas),
                "max_lambda_still_recommend_3arm": lambda_display(max_lambda_still_3arm),
                "min_robust_score_over_all_lambda": round2(min(robust_scores)),
                "max_robust_score_over_all_lambda": round2(max(robust_scores)),
                "robust_score_lambda_2": round2(score_map.get("2", 0.0)),
                "robust_score_lambda_5": round2(score_map.get("5", 0.0)),
                "robust_score_lambda_8": round2(score_map.get("8", 0.0)),
                "structure_note": (
                    "Larger max_lambda_still_recommend_3arm means more robust 3-arm value."
                ),
            }
        )

    return reliable_rows


def build_parameter_rows() -> list[dict]:
    return [
        {
            "parameter": "lambda_values",
            "value": ",".join(lambda_display(lam) for lam in LAMBDA_VALUES),
            "meaning": "energy penalty weights",
        },
        {
            "parameter": "gamma_values",
            "value": ",".join(gamma_display(g) for g in GAMMA_VALUES),
            "meaning": "risk-aversion coefficients for robust optimization",
        },
        {
            "parameter": "base_score",
            "value": "S_lambda = eta_T - lambda * eta_E",
            "meaning": "original mean benefit of enabling the third arm",
        },
        {
            "parameter": "robust_score",
            "value": "S_robust = mean(S_lambda) - gamma * std(S_lambda)",
            "meaning": "lower-confidence robust benefit considering random variation",
        },
        {
            "parameter": "decision_rule",
            "value": "recommend_3arm if S_robust > 0 else recommend_2arm",
            "meaning": "robust two-arm / three-arm mode selection rule",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust mode selection under random instance uncertainty."
    )
    parser.add_argument(
        "--raw-input",
        default=str(DEFAULT_RAW_INPUT),
        help="seed-level raw input basic_2B3B_time_energy.csv",
    )
    parser.add_argument(
        "--summary-input",
        default=str(DEFAULT_SUMMARY_INPUT),
        help="fallback summary input mode_decision_summary_basic.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory",
    )

    args = parser.parse_args()

    raw_path = Path(args.raw_input)
    summary_path = Path(args.summary_input)
    output_dir = Path(args.output_dir)

    instance_rows = load_instance_rows(raw_path, summary_path)

    parameter_rows = build_parameter_rows()
    result_rows = build_result_rows(instance_rows)
    summary_rows = build_summary_rows(result_rows)
    reliable_rows = build_reliable_structure_rows(result_rows)

    save_csv(
        parameter_rows,
        output_dir / "robust_mode_selection_parameters.csv",
    )

    save_csv(
        result_rows,
        output_dir / "robust_mode_selection_result.csv",
    )

    save_csv(
        summary_rows,
        output_dir / "robust_mode_selection_summary.csv",
    )

    save_csv(
        reliable_rows,
        output_dir / "robust_reliable_structures.csv",
    )

    print("Robust mode selection optimization finished.")
    print(f"Raw input: {raw_path}")
    print(f"Summary input: {summary_path}")
    print(f"Output directory: {output_dir}")
    print(f"Instance rows: {len(instance_rows)}")
    print(f"Result rows: {len(result_rows)}")
    print("Generated:")
    print("  robust_mode_selection_parameters.csv")
    print("  robust_mode_selection_result.csv")
    print("  robust_mode_selection_summary.csv")
    print("  robust_reliable_structures.csv")
    print()
    print("Model:")
    print("  S_lambda = eta_T - lambda * eta_E")
    print("  S_robust = mean(S_lambda) - gamma * std(S_lambda)")
    print("  recommend_3arm if S_robust > 0 else recommend_2arm")


if __name__ == "__main__":
    main()