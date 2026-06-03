# -*- coding: utf-8 -*-
"""非线性规划拓展 2：机械臂速度-能耗联合优化。

输入优先级：
    1. outputs/mode_decision/mode_decision_summary_basic.csv
    2. 如果上述文件缺少 Cmax / Energy 字段，则自动读取：
       outputs/four_case_framework/basic_2B3B_time_energy.csv

输出：
    outputs/nonlinear_programming/speed_energy_optimization/
        speed_energy_optimization_result.csv
        speed_energy_summary_by_preference.csv
        speed_energy_model_parameters.csv

研究目标：
    在二臂/三臂调度结果已确定的基础上，引入速度倍率 s，
    分析不同时间-能耗偏好下的最优速度和最终二臂/三臂推荐结果。

运行：
    python scripts/nonlinear_speed_energy_optimization.py
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

DEFAULT_RAW_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "four_case_framework"
    / "basic_2B3B_time_energy.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "nonlinear_programming"
    / "speed_energy_optimization"
)

# 速度倍率约束
SPEED_MIN = 0.6
SPEED_MAX = 1.4
SPEED_STEP = 0.01

# 动态能耗占比
DYNAMIC_ENERGY_RATIO = 0.7

# 三种时间-能耗偏好
PREFERENCES = [
    {
        "preference": "efficiency_first",
        "alpha_time": 0.8,
        "beta_energy": 0.2,
    },
    {
        "preference": "balanced",
        "alpha_time": 0.5,
        "beta_energy": 0.5,
    },
    {
        "preference": "energy_first",
        "alpha_time": 0.2,
        "beta_energy": 0.8,
    },
]


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
    """在 CSV 字段中寻找候选列名。"""
    fieldname_set = set(fieldnames)

    for c in candidates:
        if c in fieldname_set:
            return c

    # 再做一次不区分大小写的匹配
    lower_map = {name.lower(): name for name in fieldnames}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def classify_scenario(n1: int, n2: int, n3: int, n4: int) -> str:
    """当原始文件中没有 scenario_type 时，自动分类。"""
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


def load_base_rows_from_summary(summary_path: Path) -> list[dict] | None:
    """优先从 mode_decision_summary_basic.csv 中读取均值结果。

    如果该文件中没有 Cmax / Energy 字段，则返回 None。
    """
    if not summary_path.exists():
        return None

    rows = read_csv(summary_path)

    if not rows:
        return None

    fieldnames = list(rows[0].keys())

    cmax_2_col = find_column(
        fieldnames,
        [
            "mean_cmax_2arm",
            "mean_cmax_2B",
            "mean_cmax_2B_basic",
            "cmax_2arm",
            "cmax_2B_basic",
        ],
    )

    cmax_3_col = find_column(
        fieldnames,
        [
            "mean_cmax_3arm",
            "mean_cmax_3B",
            "mean_cmax_3B_basic",
            "cmax_3arm",
            "cmax_3B_basic",
        ],
    )

    energy_2_col = find_column(
        fieldnames,
        [
            "mean_energy_2arm",
            "mean_energy_2B",
            "mean_energy_2B_basic",
            "energy_2arm",
            "energy_2B_basic",
        ],
    )

    energy_3_col = find_column(
        fieldnames,
        [
            "mean_energy_3arm",
            "mean_energy_3B",
            "mean_energy_3B_basic",
            "energy_3arm",
            "energy_3B_basic",
        ],
    )

    if not all([cmax_2_col, cmax_3_col, energy_2_col, energy_3_col]):
        return None

    base_rows = []

    for row in rows:
        n1 = safe_int(row.get("n1", 0))
        n2 = safe_int(row.get("n2", 0))
        n3 = safe_int(row.get("n3", 0))
        n4 = safe_int(row.get("n4", 0))

        total_tasks = safe_int(row.get("total_tasks", row.get("n", n1 + n2 + n3 + n4)))

        scenario_type = row.get(
            "scenario_type",
            classify_scenario(n1, n2, n3, n4),
        )

        base_rows.append(
            {
                "counts_code": row["counts_code"],
                "scenario_type": scenario_type,
                "total_tasks": total_tasks,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "base_cmax_2arm": safe_float(row[cmax_2_col]),
                "base_cmax_3arm": safe_float(row[cmax_3_col]),
                "base_energy_2arm": safe_float(row[energy_2_col]),
                "base_energy_3arm": safe_float(row[energy_3_col]),
                "data_source": "mode_decision_summary_basic.csv",
            }
        )

    return base_rows


def load_base_rows_from_raw(raw_path: Path) -> list[dict]:
    """从 basic_2B3B_time_energy.csv 中按 counts_code 聚合。"""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw input not found: {raw_path}")

    rows = read_csv(raw_path)

    groups: dict[tuple, list[dict]] = {}

    valid_status = {"OPTIMAL", "FEASIBLE", "success", "SUCCESS"}

    for row in rows:
        status_2 = row.get("status_2B_basic", "")
        status_3 = row.get("status_3B_basic", "")

        if status_2 not in valid_status or status_3 not in valid_status:
            continue

        n1 = safe_int(row["n1"])
        n2 = safe_int(row["n2"])
        n3 = safe_int(row["n3"])
        n4 = safe_int(row["n4"])
        total_tasks = safe_int(row.get("total_tasks", n1 + n2 + n3 + n4))
        counts_code = row["counts_code"]
        scenario_type = classify_scenario(n1, n2, n3, n4)

        key = (counts_code, scenario_type, total_tasks, n1, n2, n3, n4)
        groups.setdefault(key, []).append(row)

    base_rows = []

    for key, group_rows in sorted(groups.items()):
        counts_code, scenario_type, total_tasks, n1, n2, n3, n4 = key

        cmax_2_values = [safe_float(r["cmax_2B_basic"]) for r in group_rows]
        cmax_3_values = [safe_float(r["cmax_3B_basic"]) for r in group_rows]
        energy_2_values = [safe_float(r["energy_2B_basic"]) for r in group_rows]
        energy_3_values = [safe_float(r["energy_3B_basic"]) for r in group_rows]

        base_rows.append(
            {
                "counts_code": counts_code,
                "scenario_type": scenario_type,
                "total_tasks": total_tasks,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "base_cmax_2arm": sum(cmax_2_values) / len(cmax_2_values),
                "base_cmax_3arm": sum(cmax_3_values) / len(cmax_3_values),
                "base_energy_2arm": sum(energy_2_values) / len(energy_2_values),
                "base_energy_3arm": sum(energy_3_values) / len(energy_3_values),
                "data_source": "basic_2B3B_time_energy.csv",
            }
        )

    return base_rows


def load_base_rows(summary_path: Path, raw_path: Path) -> list[dict]:
    base_rows = load_base_rows_from_summary(summary_path)

    if base_rows is not None:
        return base_rows

    print("Warning: summary file has no Cmax/Energy columns, fallback to raw time-energy file.")
    return load_base_rows_from_raw(raw_path)


def cmax_after_speed(base_cmax: float, speed: float) -> float:
    return base_cmax / speed


def energy_after_speed(base_energy: float, speed: float, rho: float) -> float:
    return base_energy * ((1.0 - rho) + rho * speed * speed)


def objective_value(
    cmax: float,
    energy: float,
    cmax_ref: float,
    energy_ref: float,
    alpha_time: float,
    beta_energy: float,
) -> float:
    return alpha_time * cmax / cmax_ref + beta_energy * energy / energy_ref


def optimize_speed_for_mode(
    base_cmax: float,
    base_energy: float,
    cmax_ref: float,
    energy_ref: float,
    alpha_time: float,
    beta_energy: float,
    rho: float,
    speed_min: float,
    speed_max: float,
    speed_step: float,
) -> dict:
    """网格搜索求单个模式下的最优速度。"""
    best_speed = None
    best_cmax = None
    best_energy = None
    best_obj = float("inf")

    steps = int(round((speed_max - speed_min) / speed_step))

    for k in range(steps + 1):
        speed = speed_min + k * speed_step
        speed = round(speed, 6)

        cmax = cmax_after_speed(base_cmax, speed)
        energy = energy_after_speed(base_energy, speed, rho)

        obj = objective_value(
            cmax=cmax,
            energy=energy,
            cmax_ref=cmax_ref,
            energy_ref=energy_ref,
            alpha_time=alpha_time,
            beta_energy=beta_energy,
        )

        if obj < best_obj:
            best_speed = speed
            best_cmax = cmax
            best_energy = energy
            best_obj = obj

    return {
        "speed": best_speed,
        "cmax": best_cmax,
        "energy": best_energy,
        "objective": best_obj,
    }


def recommend_mode(obj_2arm: float, obj_3arm: float) -> str:
    if obj_3arm < obj_2arm:
        return "recommend_3arm"

    if obj_3arm > obj_2arm:
        return "recommend_2arm"

    return "similar_prefer_2arm"


def build_optimization_rows(
    base_rows: list[dict],
    rho: float,
    speed_min: float,
    speed_max: float,
    speed_step: float,
) -> list[dict]:
    result_rows = []

    for row in base_rows:
        cmax_2 = safe_float(row["base_cmax_2arm"])
        cmax_3 = safe_float(row["base_cmax_3arm"])
        energy_2 = safe_float(row["base_energy_2arm"])
        energy_3 = safe_float(row["base_energy_3arm"])

        if cmax_2 <= 0 or energy_2 <= 0:
            continue

        # 用二臂默认速度下的 Cmax/Energy 作为归一化基准
        cmax_ref = cmax_2
        energy_ref = energy_2

        for pref in PREFERENCES:
            preference = pref["preference"]
            alpha_time = pref["alpha_time"]
            beta_energy = pref["beta_energy"]

            base_obj_2 = objective_value(
                cmax_2,
                energy_2,
                cmax_ref,
                energy_ref,
                alpha_time,
                beta_energy,
            )

            base_obj_3 = objective_value(
                cmax_3,
                energy_3,
                cmax_ref,
                energy_ref,
                alpha_time,
                beta_energy,
            )

            base_recommendation = recommend_mode(base_obj_2, base_obj_3)

            opt_2 = optimize_speed_for_mode(
                base_cmax=cmax_2,
                base_energy=energy_2,
                cmax_ref=cmax_ref,
                energy_ref=energy_ref,
                alpha_time=alpha_time,
                beta_energy=beta_energy,
                rho=rho,
                speed_min=speed_min,
                speed_max=speed_max,
                speed_step=speed_step,
            )

            opt_3 = optimize_speed_for_mode(
                base_cmax=cmax_3,
                base_energy=energy_3,
                cmax_ref=cmax_ref,
                energy_ref=energy_ref,
                alpha_time=alpha_time,
                beta_energy=beta_energy,
                rho=rho,
                speed_min=speed_min,
                speed_max=speed_max,
                speed_step=speed_step,
            )

            optimized_recommendation = recommend_mode(
                opt_2["objective"],
                opt_3["objective"],
            )

            obj_gain_3_vs_2 = (
                (opt_2["objective"] - opt_3["objective"]) / opt_2["objective"] * 100
                if opt_2["objective"] > 0
                else 0.0
            )

            result_rows.append(
                {
                    "counts_code": row["counts_code"],
                    "scenario_type": row["scenario_type"],
                    "total_tasks": row["total_tasks"],
                    "n1": row["n1"],
                    "n2": row["n2"],
                    "n3": row["n3"],
                    "n4": row["n4"],
                    "preference": preference,
                    "alpha_time": alpha_time,
                    "beta_energy": beta_energy,
                    "rho_dynamic_energy": rho,
                    "base_cmax_2arm": round2(cmax_2),
                    "base_cmax_3arm": round2(cmax_3),
                    "base_energy_2arm": round2(energy_2),
                    "base_energy_3arm": round2(energy_3),
                    "base_obj_2arm": round4(base_obj_2),
                    "base_obj_3arm": round4(base_obj_3),
                    "base_recommendation": base_recommendation,
                    "opt_speed_2arm": round2(opt_2["speed"]),
                    "opt_cmax_2arm": round2(opt_2["cmax"]),
                    "opt_energy_2arm": round2(opt_2["energy"]),
                    "opt_obj_2arm": round4(opt_2["objective"]),
                    "opt_speed_3arm": round2(opt_3["speed"]),
                    "opt_cmax_3arm": round2(opt_3["cmax"]),
                    "opt_energy_3arm": round2(opt_3["energy"]),
                    "opt_obj_3arm": round4(opt_3["objective"]),
                    "optimized_recommendation": optimized_recommendation,
                    "objective_gain_3arm_vs_2arm_percent": round2(obj_gain_3_vs_2),
                    "data_source": row["data_source"],
                    "model_note": "Cmax=Cmax0/s, Energy=Energy0*((1-rho)+rho*s^2)",
                }
            )

    return result_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary_rows(result_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}

    for row in result_rows:
        groups.setdefault(row["preference"], []).append(row)

    summary_rows = []

    for preference, rows in groups.items():
        case_count = len(rows)

        rec_3arm = sum(1 for r in rows if r["optimized_recommendation"] == "recommend_3arm")
        rec_2arm = sum(1 for r in rows if r["optimized_recommendation"] == "recommend_2arm")

        base_rec_3arm = sum(1 for r in rows if r["base_recommendation"] == "recommend_3arm")
        base_rec_2arm = sum(1 for r in rows if r["base_recommendation"] == "recommend_2arm")

        summary_rows.append(
            {
                "preference": preference,
                "case_count": case_count,
                "base_recommend_3arm_count": base_rec_3arm,
                "base_recommend_2arm_count": base_rec_2arm,
                "optimized_recommend_3arm_count": rec_3arm,
                "optimized_recommend_2arm_count": rec_2arm,
                "optimized_recommend_3arm_ratio_percent": round2(rec_3arm / case_count * 100),
                "mean_opt_speed_2arm": round2(mean([safe_float(r["opt_speed_2arm"]) for r in rows])),
                "mean_opt_speed_3arm": round2(mean([safe_float(r["opt_speed_3arm"]) for r in rows])),
                "mean_objective_gain_3arm_vs_2arm_percent": round2(
                    mean([safe_float(r["objective_gain_3arm_vs_2arm_percent"]) for r in rows])
                ),
                "note": "positive gain means optimized 3-arm objective is lower than optimized 2-arm objective",
            }
        )

    return summary_rows


def build_parameter_rows(
    rho: float,
    speed_min: float,
    speed_max: float,
    speed_step: float,
) -> list[dict]:
    rows = []

    for pref in PREFERENCES:
        rows.append(
            {
                "preference": pref["preference"],
                "alpha_time": pref["alpha_time"],
                "beta_energy": pref["beta_energy"],
                "speed_min": speed_min,
                "speed_max": speed_max,
                "speed_step": speed_step,
                "rho_dynamic_energy": rho,
                "cmax_model": "Cmax(s)=Cmax0/s",
                "energy_model": "E(s)=E0*((1-rho)+rho*s^2)",
                "objective": "min alpha*Cmax/Cmax_2arm0 + beta*Energy/Energy_2arm0",
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nonlinear speed-energy optimization for multi-arm scheduling."
    )
    parser.add_argument(
        "--summary-input",
        default=str(DEFAULT_SUMMARY_INPUT),
        help="mode_decision_summary_basic.csv",
    )
    parser.add_argument(
        "--raw-input",
        default=str(DEFAULT_RAW_INPUT),
        help="basic_2B3B_time_energy.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory",
    )
    parser.add_argument(
        "--speed-min",
        type=float,
        default=SPEED_MIN,
        help="minimum speed multiplier",
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=SPEED_MAX,
        help="maximum speed multiplier",
    )
    parser.add_argument(
        "--speed-step",
        type=float,
        default=SPEED_STEP,
        help="speed grid step",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=DYNAMIC_ENERGY_RATIO,
        help="dynamic energy ratio",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_input)
    raw_path = Path(args.raw_input)
    output_dir = Path(args.output_dir)

    base_rows = load_base_rows(summary_path, raw_path)

    result_rows = build_optimization_rows(
        base_rows=base_rows,
        rho=args.rho,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        speed_step=args.speed_step,
    )

    summary_rows = build_summary_rows(result_rows)

    parameter_rows = build_parameter_rows(
        rho=args.rho,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        speed_step=args.speed_step,
    )

    save_csv(
        result_rows,
        output_dir / "speed_energy_optimization_result.csv",
    )

    save_csv(
        summary_rows,
        output_dir / "speed_energy_summary_by_preference.csv",
    )

    save_csv(
        parameter_rows,
        output_dir / "speed_energy_model_parameters.csv",
    )

    print("Speed-energy nonlinear optimization finished.")
    print(f"Summary input: {summary_path}")
    print(f"Raw input: {raw_path}")
    print(f"Output directory: {output_dir}")
    print(f"Base rows: {len(base_rows)}")
    print(f"Optimization rows: {len(result_rows)}")
    print("Generated:")
    print("  speed_energy_optimization_result.csv")
    print("  speed_energy_summary_by_preference.csv")
    print("  speed_energy_model_parameters.csv")
    print()
    print("Model:")
    print("  Cmax(s)=Cmax0/s")
    print("  Energy(s)=Energy0*((1-rho)+rho*s^2)")
    print("  Objective=min alpha*Cmax/Cmax_2arm0 + beta*Energy/Energy_2arm0")
    print()
    print("Preferences:")
    print("  efficiency_first: alpha=0.8, beta=0.2")
    print("  balanced: alpha=0.5, beta=0.5")
    print("  energy_first: alpha=0.2, beta=0.8")


if __name__ == "__main__":
    main()