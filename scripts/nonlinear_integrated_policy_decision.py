# -*- coding: utf-8 -*-
"""非线性规划拓展 5：综合鲁棒启用策略与最终决策规则。

输入：
    1. outputs/mode_decision/mode_decision_summary_basic.csv
    2. outputs/nonlinear_programming/robust_mode_selection/robust_mode_selection_result.csv
    3. outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization_result.csv

输出：
    outputs/nonlinear_programming/integrated_policy_decision/
        integrated_policy_parameters.csv
        integrated_policy_decision_result.csv
        integrated_policy_summary.csv
        integrated_policy_key_scenarios.csv

研究目标：
    将前四个子问题合并：
        任务结构收益
        速度-能耗优化
        第三臂固定启用成本
        随机扰动鲁棒性

    构造最终综合得分：
        S_final = mean(S_lambda) - gamma*std(S_lambda) - K*Phi(p,r)

    决策规则：
        S_final > 0 推荐三臂，否则推荐二臂。

运行：
    python scripts/nonlinear_integrated_policy_decision.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SUMMARY_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "mode_decision"
    / "mode_decision_summary_basic.csv"
)

DEFAULT_ROBUST_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "nonlinear_programming"
    / "robust_mode_selection"
    / "robust_mode_selection_result.csv"
)

DEFAULT_SPEED_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "nonlinear_programming"
    / "speed_energy_optimization"
    / "speed_energy_optimization_result.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "nonlinear_programming"
    / "integrated_policy_decision"
)

LAMBDA_VALUES = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]
GAMMA_VALUES = [0, 0.5, 1, 1.5, 2]
ACTIVATION_COST_LEVELS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25]

N_MAX = 8.0

SIZE_WEIGHT = 0.20
IMBALANCE_WEIGHT = 0.25
TYPE4_WEIGHT = 0.15
IMBALANCE_MAX = 0.75

KEY_POLICY_SCENARIOS = [
    {
        "policy_name": "efficiency_oriented",
        "lambda": 2,
        "gamma": 0.5,
        "activation_cost_K": 4,
        "meaning": "time efficiency is important and third-arm activation cost is low",
    },
    {
        "policy_name": "balanced_operation",
        "lambda": 5,
        "gamma": 1,
        "activation_cost_K": 8,
        "meaning": "balanced time-energy preference with moderate robustness requirement",
    },
    {
        "policy_name": "conservative_energy",
        "lambda": 8,
        "gamma": 1.5,
        "activation_cost_K": 10,
        "meaning": "energy saving and robustness are both emphasized",
    },
    {
        "policy_name": "strict_energy_saving",
        "lambda": 10,
        "gamma": 2,
        "activation_cost_K": 12,
        "meaning": "strong energy saving preference and strict robustness requirement",
    },
]


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


def get_eta_t(row: dict) -> float:
    if "mean_eta_T_percent" in row:
        return safe_float(row["mean_eta_T_percent"])
    if "mean_eta_T" in row:
        return safe_float(row["mean_eta_T"])
    return 0.0


def get_eta_e(row: dict) -> float:
    if "mean_eta_E_percent" in row:
        return safe_float(row["mean_eta_E_percent"])
    if "mean_eta_E" in row:
        return safe_float(row["mean_eta_E"])
    return 0.0


def get_total_tasks(row: dict) -> int:
    if "total_tasks" in row:
        return safe_int(row["total_tasks"])
    if "n" in row:
        return safe_int(row["n"])
    return (
        safe_int(row.get("n1", 0))
        + safe_int(row.get("n2", 0))
        + safe_int(row.get("n3", 0))
        + safe_int(row.get("n4", 0))
    )


def get_proportions(row: dict) -> tuple[float, float, float, float, float]:
    n1 = safe_float(row.get("n1", 0))
    n2 = safe_float(row.get("n2", 0))
    n3 = safe_float(row.get("n3", 0))
    n4 = safe_float(row.get("n4", 0))

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
    raw = (
        (p1 - 0.25) ** 2
        + (p2 - 0.25) ** 2
        + (p3 - 0.25) ** 2
        + (p4 - 0.25) ** 2
    )
    return raw / IMBALANCE_MAX


def activation_cost_factor(p1: float, p2: float, p3: float, p4: float, r: float) -> float:
    imbalance = imbalance_index(p1, p2, p3, p4)

    return (
        1.0
        + SIZE_WEIGHT * r * r
        + IMBALANCE_WEIGHT * imbalance
        + TYPE4_WEIGHT * p4 * p4
    )


def recommendation(score: float) -> str:
    if score > 0:
        return "recommend_3arm"
    if score < 0:
        return "recommend_2arm"
    return "similar_prefer_2arm"


def speed_preference_by_lambda(lam: float) -> str:
    """把 lambda 映射到子问题二的速度偏好。"""
    if lam <= 2:
        return "efficiency_first"

    if lam <= 6:
        return "balanced"

    return "energy_first"


def load_structure_rows(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary input not found: {summary_path}")

    rows = read_csv(summary_path)

    structure_rows = []

    for row in rows:
        n1 = safe_int(row.get("n1", 0))
        n2 = safe_int(row.get("n2", 0))
        n3 = safe_int(row.get("n3", 0))
        n4 = safe_int(row.get("n4", 0))
        total_tasks = get_total_tasks(row)

        structure_rows.append(
            {
                "counts_code": row["counts_code"],
                "scenario_type": row.get("scenario_type", ""),
                "total_tasks": total_tasks,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "eta_T_percent": get_eta_t(row),
                "eta_E_percent": get_eta_e(row),
            }
        )

    return structure_rows


def load_robust_score_map(robust_path: Path) -> dict[tuple[str, str], dict]:
    robust_map: dict[tuple[str, str], dict] = {}

    if not robust_path.exists():
        return robust_map

    rows = read_csv(robust_path)

    for row in rows:
        # mean_score 和 std_score 与 gamma 无关，重复行覆盖也没关系
        key = (row["counts_code"], row["lambda_label"])

        robust_map[key] = {
            "mean_score": safe_float(row["mean_score"]),
            "std_score": safe_float(row["std_score"]),
            "sample_count": safe_int(row.get("sample_count", 0)),
            "data_source": "robust_mode_selection_result.csv",
        }

    return robust_map


def load_speed_map(speed_path: Path) -> dict[tuple[str, str], dict]:
    speed_map: dict[tuple[str, str], dict] = {}

    if not speed_path.exists():
        return speed_map

    rows = read_csv(speed_path)

    for row in rows:
        key = (row["counts_code"], row["preference"])

        speed_map[key] = {
            "opt_speed_2arm": safe_float(row.get("opt_speed_2arm", 0)),
            "opt_speed_3arm": safe_float(row.get("opt_speed_3arm", 0)),
            "opt_obj_2arm": safe_float(row.get("opt_obj_2arm", 0)),
            "opt_obj_3arm": safe_float(row.get("opt_obj_3arm", 0)),
            "speed_objective_gain_3arm_vs_2arm_percent": safe_float(
                row.get("objective_gain_3arm_vs_2arm_percent", 0)
            ),
            "data_source": "speed_energy_optimization_result.csv",
        }

    return speed_map


def build_decision_rows(
    structure_rows: list[dict],
    robust_map: dict[tuple[str, str], dict],
    speed_map: dict[tuple[str, str], dict],
) -> list[dict]:
    result_rows = []

    for row in structure_rows:
        p1, p2, p3, p4, r = get_proportions(row)
        phi = activation_cost_factor(p1, p2, p3, p4, r)

        eta_t = safe_float(row["eta_T_percent"])
        eta_e = safe_float(row["eta_E_percent"])

        for lam in LAMBDA_VALUES:
            label = lambda_label(lam)
            lam_text = lambda_display(lam)

            robust_info = robust_map.get((row["counts_code"], label))

            if robust_info:
                mean_score = safe_float(robust_info["mean_score"])
                std_score = safe_float(robust_info["std_score"])
                sample_count = safe_int(robust_info["sample_count"])
                robust_source = robust_info["data_source"]
            else:
                mean_score = eta_t - lam * eta_e
                std_score = 0.0
                sample_count = 1
                robust_source = "mode_decision_summary_basic.csv"

            speed_pref = speed_preference_by_lambda(lam)
            speed_info = speed_map.get((row["counts_code"], speed_pref), {})

            opt_speed_2arm = safe_float(speed_info.get("opt_speed_2arm", 0))
            opt_speed_3arm = safe_float(speed_info.get("opt_speed_3arm", 0))
            opt_obj_2arm = safe_float(speed_info.get("opt_obj_2arm", 0))
            opt_obj_3arm = safe_float(speed_info.get("opt_obj_3arm", 0))
            speed_gain = safe_float(
                speed_info.get("speed_objective_gain_3arm_vs_2arm_percent", 0)
            )

            for gamma in GAMMA_VALUES:
                risk_penalty = gamma * std_score

                for k in ACTIVATION_COST_LEVELS:
                    activation_penalty = k * phi
                    final_score = mean_score - risk_penalty - activation_penalty
                    final_rec = recommendation(final_score)

                    if final_rec == "recommend_3arm":
                        selected_mode = "3arm"
                        selected_speed = opt_speed_3arm
                    else:
                        selected_mode = "2arm"
                        selected_speed = opt_speed_2arm

                    result_rows.append(
                        {
                            "counts_code": row["counts_code"],
                            "scenario_type": row["scenario_type"],
                            "total_tasks": row["total_tasks"],
                            "n1": row["n1"],
                            "n2": row["n2"],
                            "n3": row["n3"],
                            "n4": row["n4"],
                            "p1": round4(p1),
                            "p2": round4(p2),
                            "p3": round4(p3),
                            "p4": round4(p4),
                            "r": round4(r),
                            "lambda": lam_text,
                            "lambda_label": label,
                            "gamma": gamma_display(gamma),
                            "activation_cost_K": k,
                            "speed_preference": speed_pref,
                            "eta_T_percent": round2(eta_t),
                            "eta_E_percent": round2(eta_e),
                            "mean_score": round2(mean_score),
                            "std_score": round2(std_score),
                            "risk_penalty": round2(risk_penalty),
                            "activation_cost_factor_phi": round4(phi),
                            "activation_penalty": round2(activation_penalty),
                            "final_integrated_score": round2(final_score),
                            "final_recommendation": final_rec,
                            "selected_mode": selected_mode,
                            "recommended_speed_multiplier": round2(selected_speed),
                            "opt_speed_2arm": round2(opt_speed_2arm),
                            "opt_speed_3arm": round2(opt_speed_3arm),
                            "opt_obj_2arm": round4(opt_obj_2arm),
                            "opt_obj_3arm": round4(opt_obj_3arm),
                            "speed_objective_gain_3arm_vs_2arm_percent": round2(speed_gain),
                            "sample_count": sample_count,
                            "robust_data_source": robust_source,
                            "model_note": "S_final=mean(S_lambda)-gamma*std(S_lambda)-K*Phi(p,r)",
                        }
                    )

    return result_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary_rows(result_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}

    for row in result_rows:
        key = (row["lambda"], row["lambda_label"], row["gamma"], row["activation_cost_K"])
        groups.setdefault(key, []).append(row)

    def sort_key(item):
        lam, label, gamma, k = item[0]
        return (safe_float(lam), safe_float(gamma), safe_float(k))

    summary_rows = []

    for key, rows in sorted(groups.items(), key=sort_key):
        lam, label, gamma, k = key

        case_count = len(rows)
        rec_3arm = sum(1 for r in rows if r["final_recommendation"] == "recommend_3arm")
        rec_2arm = sum(1 for r in rows if r["final_recommendation"] == "recommend_2arm")

        final_scores = [safe_float(r["final_integrated_score"]) for r in rows]
        selected_speeds = [safe_float(r["recommended_speed_multiplier"]) for r in rows]

        summary_rows.append(
            {
                "lambda": lam,
                "lambda_label": label,
                "gamma": gamma,
                "activation_cost_K": k,
                "case_count": case_count,
                "recommend_3arm_count": rec_3arm,
                "recommend_2arm_count": rec_2arm,
                "recommend_3arm_ratio_percent": round2(rec_3arm / case_count * 100),
                "mean_final_integrated_score": round2(mean(final_scores)),
                "min_final_integrated_score": round2(min(final_scores)),
                "max_final_integrated_score": round2(max(final_scores)),
                "mean_recommended_speed_multiplier": round2(mean(selected_speeds)),
                "note": "final decision combines mean benefit, robustness penalty, activation cost and speed policy",
            }
        )

    return summary_rows


def build_key_policy_rows(result_rows: list[dict]) -> list[dict]:
    key_rows = []

    for policy in KEY_POLICY_SCENARIOS:
        target_lam = safe_float(policy["lambda"])
        target_gamma = safe_float(policy["gamma"])
        target_k = safe_float(policy["activation_cost_K"])

        selected_rows = [
            row
            for row in result_rows
            if abs(safe_float(row["lambda"]) - target_lam) < 1e-9
            and abs(safe_float(row["gamma"]) - target_gamma) < 1e-9
            and abs(safe_float(row["activation_cost_K"]) - target_k) < 1e-9
        ]

        selected_rows = sorted(
            selected_rows,
            key=lambda r: safe_float(r["final_integrated_score"]),
            reverse=True,
        )

        for rank, row in enumerate(selected_rows, start=1):
            out = {
                "policy_name": policy["policy_name"],
                "policy_meaning": policy["meaning"],
                "lambda": lambda_display(target_lam),
                "gamma": gamma_display(target_gamma),
                "activation_cost_K": int(target_k),
                "rank_by_final_score": rank,
            }

            out.update(row)
            key_rows.append(out)

    return key_rows


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
            "meaning": "risk-aversion coefficients",
        },
        {
            "parameter": "activation_cost_K",
            "value": ",".join(str(k) for k in ACTIVATION_COST_LEVELS),
            "meaning": "fixed activation cost levels of enabling the third arm",
        },
        {
            "parameter": "final_score",
            "value": "S_final=mean(S_lambda)-gamma*std(S_lambda)-K*Phi(p,r)",
            "meaning": "integrated score combining benefit, uncertainty and fixed activation cost",
        },
        {
            "parameter": "speed_policy",
            "value": "lambda<=2:efficiency_first; 3<=lambda<=6:balanced; lambda>=8:energy_first",
            "meaning": "speed preference selected according to energy penalty weight",
        },
        {
            "parameter": "decision_rule",
            "value": "recommend_3arm if S_final>0 else recommend_2arm",
            "meaning": "final integrated two-arm / three-arm decision rule",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated robust policy decision for two-arm / three-arm scheduling."
    )
    parser.add_argument(
        "--summary-input",
        default=str(DEFAULT_SUMMARY_INPUT),
        help="mode_decision_summary_basic.csv",
    )
    parser.add_argument(
        "--robust-input",
        default=str(DEFAULT_ROBUST_INPUT),
        help="robust_mode_selection_result.csv",
    )
    parser.add_argument(
        "--speed-input",
        default=str(DEFAULT_SPEED_INPUT),
        help="speed_energy_optimization_result.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_input)
    robust_path = Path(args.robust_input)
    speed_path = Path(args.speed_input)
    output_dir = Path(args.output_dir)

    structure_rows = load_structure_rows(summary_path)
    robust_map = load_robust_score_map(robust_path)
    speed_map = load_speed_map(speed_path)

    parameter_rows = build_parameter_rows()
    result_rows = build_decision_rows(structure_rows, robust_map, speed_map)
    summary_rows = build_summary_rows(result_rows)
    key_policy_rows = build_key_policy_rows(result_rows)

    save_csv(
        parameter_rows,
        output_dir / "integrated_policy_parameters.csv",
    )

    save_csv(
        result_rows,
        output_dir / "integrated_policy_decision_result.csv",
    )

    save_csv(
        summary_rows,
        output_dir / "integrated_policy_summary.csv",
    )

    save_csv(
        key_policy_rows,
        output_dir / "integrated_policy_key_scenarios.csv",
    )

    print("Integrated robust policy decision finished.")
    print(f"Summary input: {summary_path}")
    print(f"Robust input: {robust_path}")
    print(f"Speed input: {speed_path}")
    print(f"Output directory: {output_dir}")
    print(f"Structure rows: {len(structure_rows)}")
    print(f"Robust score entries: {len(robust_map)}")
    print(f"Speed entries: {len(speed_map)}")
    print(f"Decision rows: {len(result_rows)}")
    print("Generated:")
    print("  integrated_policy_parameters.csv")
    print("  integrated_policy_decision_result.csv")
    print("  integrated_policy_summary.csv")
    print("  integrated_policy_key_scenarios.csv")
    print()
    print("Model:")
    print("  S_final = mean(S_lambda) - gamma*std(S_lambda) - K*Phi(p,r)")
    print("  speed policy is selected according to lambda")
    print("  recommend_3arm if S_final > 0 else recommend_2arm")


if __name__ == "__main__":
    main()