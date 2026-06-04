# -*- coding: utf-8 -*-
"""基于截止时间约束的速度-能耗非线性优化。

本脚本只输出一个结果文件：
    outputs/nonlinear_programming/speed_energy_optimization/
        speed_energy_deadline_optimization.csv

核心思想：
    前面的整数规划/调度模型已经得到二臂和三臂在默认速度下的：
        Cmax0
        Energy0

    本脚本不重新改变任务分配和任务顺序，只引入速度倍率 s。

非线性规划模型：
    对二臂和三臂分别求解：

        min Energy(s)
        s.t. Cmax(s) <= D
             speed_min <= s <= speed_max

    其中：
        Cmax(s) = Cmax0 / s
        Energy(s) = Energy0 * ((1-rho) + rho*s^2)

    D 为截止时间要求。

数值解法：
    使用课件中的“内点法 + 0.618法”。

    约束写成严格内点形式：
        g1(s) = s - speed_min > 0
        g2(s) = speed_max - s > 0
        g3(s) = D - Cmax0/s > 0

    构造障碍函数：
        B(s,r) = Energy(s)
                 - r * [log(g1(s)) + log(g2(s)) + log(g3(s))]

    每个 r 下用 0.618 法求 B(s,r) 的一维极小值；
    然后逐步减小 r，使结果逼近原约束问题最优解。

默认截止时间状态：
    loose_deadline  : D = 1.50 * Cmax_2B
    normal_deadline : D = 1.00 * Cmax_2B
    tight_deadline  : D = 0.80 * Cmax_2B
    urgent_deadline : D = 0.70 * Cmax_2B

运行方式：
    python scripts/nonlinear_speed_energy_optimization.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable


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

OUTPUT_FILENAME = "speed_energy_deadline_optimization.csv"


# 速度倍率约束。
SPEED_MIN = 0.6
SPEED_MAX = 1.4

# 动态能耗占比。
# rho 越大，速度平方项对能耗影响越明显。
DYNAMIC_ENERGY_RATIO = 0.7

# 内点法参数。
BARRIER_INITIAL_R = 1.0
BARRIER_FACTOR = 10.0
BARRIER_MIN_R = 1e-7
OUTER_TOL = 1e-7
INNER_TOL = 1e-8
MAX_OUTER_ITER = 30
MAX_INNER_ITER = 300

# 0.618 法中的黄金比例。
GOLDEN_RATIO = 0.6180339887498949

# 截止时间状态。
# multiplier 表示 D = multiplier * Cmax_2B。
DEADLINE_SCENARIOS = [
    {
        "deadline_type": "loose_deadline",
        "deadline_multiplier_of_2B_cmax": 1.50,
        "deadline_note": "time requirement is loose",
    },
    {
        "deadline_type": "normal_deadline",
        "deadline_multiplier_of_2B_cmax": 1.00,
        "deadline_note": "time requirement equals original 2-arm Cmax",
    },
    {
        "deadline_type": "tight_deadline",
        "deadline_multiplier_of_2B_cmax": 0.80,
        "deadline_note": "time requirement is tight",
    },
    {
        "deadline_type": "urgent_deadline",
        "deadline_multiplier_of_2B_cmax": 0.70,
        "deadline_note": "time requirement is very urgent",
    },
]


INVALID_STATUS = {
    "INFEASIBLE",
    "NO_SOLUTION",
    "FAILED",
    "FAIL",
    "ERROR",
    "TIMEOUT",
}


def normalize_status(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


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


def round2(value: Any) -> float | str:
    if value is None:
        return ""
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return ""


def round4(value: Any) -> float | str:
    if value is None:
        return ""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return ""


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

    single_arm_tasks = n1 + n2
    dual_arm_tasks = n3 + n4

    if single_arm_tasks > dual_arm_tasks:
        return "single_arm_dominant"

    if dual_arm_tasks > single_arm_tasks:
        return "dual_arm_dominant"

    return "mixed_balanced"


def load_base_rows_from_summary(summary_path: Path) -> list[dict] | None:
    """优先从 mode_decision_summary_basic.csv 中读取均值结果。"""
    if not summary_path.exists():
        return None

    rows = read_csv(summary_path)

    if not rows:
        return None

    fieldnames = list(rows[0].keys())

    cmax_2_col = find_column(
        fieldnames,
        [
            "mean_cmax_2B",
            "mean_cmax_2arm",
            "mean_cmax_2B_basic",
            "cmax_2B_basic",
            "cmax_2arm",
        ],
    )
    cmax_3_col = find_column(
        fieldnames,
        [
            "mean_cmax_3B",
            "mean_cmax_3arm",
            "mean_cmax_3B_basic",
            "cmax_3B_basic",
            "cmax_3arm",
        ],
    )
    energy_2_col = find_column(
        fieldnames,
        [
            "mean_energy_2B",
            "mean_energy_2arm",
            "mean_energy_2B_basic",
            "energy_2B_basic",
            "energy_2arm",
        ],
    )
    energy_3_col = find_column(
        fieldnames,
        [
            "mean_energy_3B",
            "mean_energy_3arm",
            "mean_energy_3B_basic",
            "energy_3B_basic",
            "energy_3arm",
        ],
    )

    if not all([cmax_2_col, cmax_3_col, energy_2_col, energy_3_col]):
        return None

    base_rows = []

    for row in rows:
        n1 = safe_int(row.get("n1"))
        n2 = safe_int(row.get("n2"))
        n3 = safe_int(row.get("n3"))
        n4 = safe_int(row.get("n4"))
        total_tasks = safe_int(row.get("total_tasks", n1 + n2 + n3 + n4))

        cmax_2 = safe_float(row.get(cmax_2_col))
        cmax_3 = safe_float(row.get(cmax_3_col))
        energy_2 = safe_float(row.get(energy_2_col))
        energy_3 = safe_float(row.get(energy_3_col))

        if cmax_2 <= 0 or cmax_3 <= 0 or energy_2 <= 0 or energy_3 <= 0:
            continue

        base_rows.append(
            {
                "counts_code": row.get("counts_code", ""),
                "scenario_type": row.get("scenario_type") or classify_scenario(n1, n2, n3, n4),
                "total_tasks": total_tasks,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "base_cmax_2arm": cmax_2,
                "base_cmax_3arm": cmax_3,
                "base_energy_2arm": energy_2,
                "base_energy_3arm": energy_3,
                "data_source": "mode_decision_summary_basic.csv",
            }
        )

    return base_rows if base_rows else None


def load_base_rows_from_raw(raw_path: Path) -> list[dict]:
    """从 basic_2B3B_time_energy.csv 中按 counts_code 聚合。"""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw input not found: {raw_path}")

    rows = read_csv(raw_path)

    groups: dict[tuple, list[dict]] = {}
    group_order: list[tuple] = []

    for row in rows:
        status_2 = normalize_status(row.get("status_2B_basic", ""))
        status_3 = normalize_status(row.get("status_3B_basic", ""))

        if status_2 in INVALID_STATUS or status_3 in INVALID_STATUS:
            continue

        cmax_2 = safe_float(row.get("cmax_2B_basic"))
        cmax_3 = safe_float(row.get("cmax_3B_basic"))
        energy_2 = safe_float(row.get("energy_2B_basic"))
        energy_3 = safe_float(row.get("energy_3B_basic"))

        if cmax_2 <= 0 or cmax_3 <= 0 or energy_2 <= 0 or energy_3 <= 0:
            continue

        n1 = safe_int(row.get("n1"))
        n2 = safe_int(row.get("n2"))
        n3 = safe_int(row.get("n3"))
        n4 = safe_int(row.get("n4"))
        total_tasks = safe_int(row.get("total_tasks", n1 + n2 + n3 + n4))
        counts_code = row.get("counts_code", "")
        scenario_type = classify_scenario(n1, n2, n3, n4)

        key = (counts_code, scenario_type, total_tasks, n1, n2, n3, n4)

        if key not in groups:
            groups[key] = []
            group_order.append(key)

        groups[key].append(row)

    base_rows = []

    for key in group_order:
        counts_code, scenario_type, total_tasks, n1, n2, n3, n4 = key
        group_rows = groups[key]

        cmax_2_values = [safe_float(r.get("cmax_2B_basic")) for r in group_rows]
        cmax_3_values = [safe_float(r.get("cmax_3B_basic")) for r in group_rows]
        energy_2_values = [safe_float(r.get("energy_2B_basic")) for r in group_rows]
        energy_3_values = [safe_float(r.get("energy_3B_basic")) for r in group_rows]

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

    print("Warning: summary file unavailable or has no valid Cmax/Energy columns.")
    print("Fallback to raw time-energy file.")
    return load_base_rows_from_raw(raw_path)


def cmax_after_speed(base_cmax: float, speed: float) -> float:
    return base_cmax / speed


def energy_after_speed(base_energy: float, speed: float, rho: float) -> float:
    return base_energy * ((1.0 - rho) + rho * speed * speed)


def energy_objective_for_speed(
    speed: float,
    base_energy: float,
    rho: float,
) -> float:
    """非线性目标函数 Energy(s)。"""
    return energy_after_speed(base_energy, speed, rho)


def barrier_energy_objective(
    speed: float,
    r: float,
    base_cmax: float,
    base_energy: float,
    deadline: float,
    rho: float,
    speed_min: float,
    speed_max: float,
) -> float:
    """内点法障碍函数。

    原问题：
        min Energy(s)
        s.t. Cmax0/s <= D
             speed_min <= s <= speed_max

    严格内点约束：
        g1 = s - speed_min > 0
        g2 = speed_max - s > 0
        g3 = deadline - base_cmax / s > 0
    """
    g1 = speed - speed_min
    g2 = speed_max - speed
    g3 = deadline - base_cmax / speed

    if g1 <= 0 or g2 <= 0 or g3 <= 0:
        return float("inf")

    original_energy = energy_objective_for_speed(
        speed=speed,
        base_energy=base_energy,
        rho=rho,
    )

    return original_energy - r * (math.log(g1) + math.log(g2) + math.log(g3))


def golden_section_search(
    func: Callable[[float], float],
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> dict:
    """0.618 法求一维极小值。"""
    if left >= right:
        raise ValueError("golden_section_search requires left < right")

    a = left
    b = right

    x1 = b - GOLDEN_RATIO * (b - a)
    x2 = a + GOLDEN_RATIO * (b - a)
    f1 = func(x1)
    f2 = func(x2)

    iterations = 0

    while abs(b - a) > tol and iterations < max_iter:
        iterations += 1

        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - GOLDEN_RATIO * (b - a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + GOLDEN_RATIO * (b - a)
            f2 = func(x2)

    best_x = (a + b) / 2.0
    best_f = func(best_x)

    return {
        "x": best_x,
        "f": best_f,
        "iterations": iterations,
        "final_interval_length": abs(b - a),
    }


def optimize_energy_under_deadline(
    base_cmax: float,
    base_energy: float,
    deadline: float,
    rho: float,
    speed_min: float,
    speed_max: float,
    barrier_initial_r: float,
    barrier_factor: float,
    barrier_min_r: float,
    outer_tol: float,
    inner_tol: float,
    max_outer_iter: int,
    max_inner_iter: int,
) -> dict:
    """用内点法 + 0.618 法求满足截止时间约束下的最低能耗速度。"""

    if deadline <= 0 or base_cmax <= 0 or base_energy <= 0:
        return {
            "status": "INFEASIBLE",
            "reason": "invalid deadline, cmax, or energy",
        }

    required_speed = base_cmax / deadline

    if required_speed > speed_max + 1e-10:
        return {
            "status": "INFEASIBLE",
            "reason": "required speed exceeds speed_max",
            "required_speed": required_speed,
        }

    # 如果理论最优点正好在上边界附近，直接给边界可行近似。
    if required_speed >= speed_max - 1e-10:
        speed = speed_max
        cmax = cmax_after_speed(base_cmax, speed)
        energy = energy_after_speed(base_energy, speed, rho)

        return {
            "status": "BOUNDARY_FEASIBLE",
            "reason": "optimum is near speed_max boundary",
            "required_speed": required_speed,
            "speed": speed,
            "cmax": cmax,
            "energy": energy,
            "outer_iterations": 0,
            "inner_iterations_total": 0,
            "barrier_r_final": "",
            "final_interval_length": "",
        }

    lower_bound = max(speed_min, required_speed)
    upper_bound = speed_max

    eps = max(1e-12, (speed_max - speed_min) * 1e-10)

    left = lower_bound + eps
    right = upper_bound - eps

    if left >= right:
        return {
            "status": "INFEASIBLE",
            "reason": "no strict interior point exists",
            "required_speed": required_speed,
        }

    r = barrier_initial_r
    previous_speed: float | None = None
    final_search: dict | None = None
    outer_iterations = 0
    total_inner_iterations = 0

    for outer_idx in range(1, max_outer_iter + 1):
        outer_iterations = outer_idx

        def current_barrier_func(speed: float) -> float:
            return barrier_energy_objective(
                speed=speed,
                r=r,
                base_cmax=base_cmax,
                base_energy=base_energy,
                deadline=deadline,
                rho=rho,
                speed_min=speed_min,
                speed_max=speed_max,
            )

        search_result = golden_section_search(
            func=current_barrier_func,
            left=left,
            right=right,
            tol=inner_tol,
            max_iter=max_inner_iter,
        )

        speed_star = float(search_result["x"])
        total_inner_iterations += int(search_result["iterations"])
        final_search = search_result

        speed_change = (
            abs(speed_star - previous_speed)
            if previous_speed is not None
            else float("inf")
        )

        previous_speed = speed_star

        if r <= barrier_min_r and speed_change <= outer_tol:
            break

        if r <= barrier_min_r:
            break

        r = r / barrier_factor

    if previous_speed is None or final_search is None:
        return {
            "status": "INFEASIBLE",
            "reason": "optimization failed",
            "required_speed": required_speed,
        }

    final_speed = previous_speed
    final_cmax = cmax_after_speed(base_cmax, final_speed)
    final_energy = energy_after_speed(base_energy, final_speed, rho)

    return {
        "status": "OPTIMAL",
        "reason": "solved by interior point method and 0.618 search",
        "required_speed": required_speed,
        "speed": final_speed,
        "cmax": final_cmax,
        "energy": final_energy,
        "outer_iterations": outer_iterations,
        "inner_iterations_total": total_inner_iterations,
        "barrier_r_final": r,
        "final_interval_length": final_search["final_interval_length"],
    }


def choose_mode(result_2arm: dict, result_3arm: dict) -> tuple[str, str]:
    """根据可行性和能耗选择二臂或三臂。"""
    status_2 = result_2arm.get("status", "")
    status_3 = result_3arm.get("status", "")

    feasible_2 = status_2 in {"OPTIMAL", "BOUNDARY_FEASIBLE"}
    feasible_3 = status_3 in {"OPTIMAL", "BOUNDARY_FEASIBLE"}

    if not feasible_2 and not feasible_3:
        return "no_feasible_mode", "both 2arm and 3arm cannot meet deadline"

    if feasible_2 and not feasible_3:
        return "recommend_2arm", "only 2arm can meet deadline"

    if feasible_3 and not feasible_2:
        return "recommend_3arm", "only 3arm can meet deadline"

    energy_2 = safe_float(result_2arm.get("energy"))
    energy_3 = safe_float(result_3arm.get("energy"))

    if energy_2 < energy_3:
        return "recommend_2arm", "both feasible, 2arm has lower energy"

    if energy_3 < energy_2:
        return "recommend_3arm", "both feasible, 3arm has lower energy"

    return "similar_prefer_2arm", "both feasible and energy is similar"


def build_result_rows(
    base_rows: list[dict],
    rho: float,
    speed_min: float,
    speed_max: float,
    barrier_initial_r: float,
    barrier_factor: float,
    barrier_min_r: float,
    outer_tol: float,
    inner_tol: float,
    max_outer_iter: int,
    max_inner_iter: int,
) -> list[dict]:
    result_rows = []

    for base in base_rows:
        cmax_2 = safe_float(base["base_cmax_2arm"])
        cmax_3 = safe_float(base["base_cmax_3arm"])
        energy_2 = safe_float(base["base_energy_2arm"])
        energy_3 = safe_float(base["base_energy_3arm"])

        if cmax_2 <= 0 or cmax_3 <= 0 or energy_2 <= 0 or energy_3 <= 0:
            continue

        for scenario in DEADLINE_SCENARIOS:
            deadline_type = scenario["deadline_type"]
            multiplier = safe_float(scenario["deadline_multiplier_of_2B_cmax"])
            deadline = multiplier * cmax_2

            opt_2 = optimize_energy_under_deadline(
                base_cmax=cmax_2,
                base_energy=energy_2,
                deadline=deadline,
                rho=rho,
                speed_min=speed_min,
                speed_max=speed_max,
                barrier_initial_r=barrier_initial_r,
                barrier_factor=barrier_factor,
                barrier_min_r=barrier_min_r,
                outer_tol=outer_tol,
                inner_tol=inner_tol,
                max_outer_iter=max_outer_iter,
                max_inner_iter=max_inner_iter,
            )

            opt_3 = optimize_energy_under_deadline(
                base_cmax=cmax_3,
                base_energy=energy_3,
                deadline=deadline,
                rho=rho,
                speed_min=speed_min,
                speed_max=speed_max,
                barrier_initial_r=barrier_initial_r,
                barrier_factor=barrier_factor,
                barrier_min_r=barrier_min_r,
                outer_tol=outer_tol,
                inner_tol=inner_tol,
                max_outer_iter=max_outer_iter,
                max_inner_iter=max_inner_iter,
            )

            recommendation, reason = choose_mode(opt_2, opt_3)

            energy_2_opt = safe_float(opt_2.get("energy"))
            energy_3_opt = safe_float(opt_3.get("energy"))

            if energy_2_opt > 0 and energy_3_opt > 0:
                if recommendation == "recommend_2arm":
                    selected_energy = energy_2_opt
                    other_energy = energy_3_opt
                elif recommendation == "recommend_3arm":
                    selected_energy = energy_3_opt
                    other_energy = energy_2_opt
                else:
                    selected_energy = min(energy_2_opt, energy_3_opt)
                    other_energy = max(energy_2_opt, energy_3_opt)

                energy_advantage = (
                    (other_energy - selected_energy) / other_energy * 100
                    if other_energy > 0
                    else 0.0
                )
            else:
                energy_advantage = ""

            result_rows.append(
                {
                    "deadline_type": deadline_type,
                    "deadline_multiplier_of_2B_cmax": multiplier,
                    "deadline_value": round2(deadline),
                    "deadline_note": scenario["deadline_note"],

                    "counts_code": base["counts_code"],
                    "scenario_type": base["scenario_type"],
                    "total_tasks": base["total_tasks"],
                    "n1": base["n1"],
                    "n2": base["n2"],
                    "n3": base["n3"],
                    "n4": base["n4"],

                    "base_cmax_2arm": round2(cmax_2),
                    "base_cmax_3arm": round2(cmax_3),
                    "base_energy_2arm": round2(energy_2),
                    "base_energy_3arm": round2(energy_3),

                    "status_2arm": opt_2.get("status", ""),
                    "reason_2arm": opt_2.get("reason", ""),
                    "required_speed_2arm": round4(opt_2.get("required_speed")),
                    "opt_speed_2arm": round4(opt_2.get("speed")),
                    "opt_cmax_2arm": round2(opt_2.get("cmax")),
                    "opt_energy_2arm": round2(opt_2.get("energy")),

                    "status_3arm": opt_3.get("status", ""),
                    "reason_3arm": opt_3.get("reason", ""),
                    "required_speed_3arm": round4(opt_3.get("required_speed")),
                    "opt_speed_3arm": round4(opt_3.get("speed")),
                    "opt_cmax_3arm": round2(opt_3.get("cmax")),
                    "opt_energy_3arm": round2(opt_3.get("energy")),

                    "recommendation": recommendation,
                    "recommendation_reason": reason,
                    "energy_advantage_selected_vs_other_percent": round2(energy_advantage),

                    "solve_method": "interior_point_method + golden_section_0_618",
                    "model": "min Energy(s), s.t. Cmax0/s <= D and speed_min <= s <= speed_max",
                    "energy_model": "Energy(s)=Energy0*((1-rho)+rho*s^2)",
                    "rho_dynamic_energy": rho,
                    "speed_min": speed_min,
                    "speed_max": speed_max,
                    "data_source": base["data_source"],
                }
            )

    return result_rows


def print_summary(result_rows: list[dict]) -> None:
    """在终端打印按 deadline_type 的简要统计，不额外输出 CSV。"""
    groups: dict[str, list[dict]] = {}

    for row in result_rows:
        groups.setdefault(row["deadline_type"], []).append(row)

    print()
    print("Deadline summary:")

    for deadline_type, rows in groups.items():
        total = len(rows)
        rec_2 = sum(1 for r in rows if r["recommendation"] == "recommend_2arm")
        rec_3 = sum(1 for r in rows if r["recommendation"] == "recommend_3arm")
        no_feasible = sum(1 for r in rows if r["recommendation"] == "no_feasible_mode")

        print(
            f"  {deadline_type}: total={total}, "
            f"recommend_2arm={rec_2}, recommend_3arm={rec_3}, "
            f"no_feasible={no_feasible}"
        )


def remove_old_files(output_dir: Path) -> None:
    """删除旧版本可能生成的不需要的文件，保证最终只留一个核心输出。"""
    stale_files = [
        output_dir / "speed_energy_optimization_result.csv",
        output_dir / "speed_energy_summary_by_preference.csv",
        output_dir / "speed_energy_model_parameters.csv",
    ]

    for path in stale_files:
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deadline-constrained nonlinear speed-energy optimization by interior point method and 0.618 search."
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
        "--rho",
        type=float,
        default=DYNAMIC_ENERGY_RATIO,
        help="dynamic energy ratio",
    )
    parser.add_argument(
        "--barrier-initial-r",
        type=float,
        default=BARRIER_INITIAL_R,
        help="initial barrier factor r",
    )
    parser.add_argument(
        "--barrier-factor",
        type=float,
        default=BARRIER_FACTOR,
        help="barrier reduction factor; r <- r / factor",
    )
    parser.add_argument(
        "--barrier-min-r",
        type=float,
        default=BARRIER_MIN_R,
        help="minimum barrier factor used as stopping threshold",
    )
    parser.add_argument(
        "--outer-tol",
        type=float,
        default=OUTER_TOL,
        help="outer iteration speed-change tolerance",
    )
    parser.add_argument(
        "--inner-tol",
        type=float,
        default=INNER_TOL,
        help="golden section interval length tolerance",
    )
    parser.add_argument(
        "--max-outer-iter",
        type=int,
        default=MAX_OUTER_ITER,
        help="maximum number of barrier outer iterations",
    )
    parser.add_argument(
        "--max-inner-iter",
        type=int,
        default=MAX_INNER_ITER,
        help="maximum number of golden section iterations per barrier subproblem",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_input)
    raw_path = Path(args.raw_input)
    output_dir = Path(args.output_dir)
    output_path = output_dir / OUTPUT_FILENAME

    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_base_rows(summary_path, raw_path)

    result_rows = build_result_rows(
        base_rows=base_rows,
        rho=args.rho,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        barrier_initial_r=args.barrier_initial_r,
        barrier_factor=args.barrier_factor,
        barrier_min_r=args.barrier_min_r,
        outer_tol=args.outer_tol,
        inner_tol=args.inner_tol,
        max_outer_iter=args.max_outer_iter,
        max_inner_iter=args.max_inner_iter,
    )

    save_csv(result_rows, output_path)
    remove_old_files(output_dir)

    print("Deadline-constrained speed-energy nonlinear optimization finished.")
    print(f"Summary input: {summary_path}")
    print(f"Raw input: {raw_path}")
    print(f"Output file: {output_path}")
    print(f"Base rows: {len(base_rows)}")
    print(f"Result rows: {len(result_rows)}")
    print()
    print("Generated only:")
    print(f"  {OUTPUT_FILENAME}")
    print()
    print("Constrained nonlinear model:")
    print("  min Energy(s)")
    print("  s.t. Cmax0/s <= D")
    print("       speed_min <= s <= speed_max")
    print("  Energy(s)=Energy0*((1-rho)+rho*s^2)")
    print()
    print("Numerical method:")
    print("  Interior point barrier method")
    print("  B(s,r)=Energy(s)-r*(log(s-speed_min)+log(speed_max-s)+log(D-Cmax0/s))")
    print("  Each barrier subproblem is solved by golden section 0.618 search")
    print()
    print("Deadline scenarios:")
    for scenario in DEADLINE_SCENARIOS:
        print(
            f"  {scenario['deadline_type']}: "
            f"D = {scenario['deadline_multiplier_of_2B_cmax']} * Cmax_2B"
        )

    print_summary(result_rows)

    if len(base_rows) > 0 and len(result_rows) == 0:
        print()
        print("Warning: base rows were loaded, but no result rows were generated.")
        print("Please check whether Cmax/Energy values are positive.")


if __name__ == "__main__":
    main()