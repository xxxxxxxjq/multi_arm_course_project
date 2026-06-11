# -*- coding: utf-8 -*-
"""非线性规划 2：随机 seed 扰动下的双速度变量鲁棒速度-能耗优化。

本脚本只读取文件夹中已有的不同 seed 下 2B/3B 完工时间与能耗结果，不重新运行主调度，
不重新设置启动成本、固定能耗、服务时间或物块能耗参数。与旧版本相比，本版本同步采用双速度变量与 U 形经济速度能耗函数：

    s_single,m：模式 m 的单臂任务统一速度倍率
    s_dual,m  ：模式 m 的双臂协同任务统一速度倍率

鲁棒模型：
    对 m in {2B, 3B}，在同一个 counts_code 下的所有 seed 使用同一组速度变量：

       min z_m
       s.t. E_fixed,m,j + E_single,m,j(s_single,m) + E_dual,m,j(s_dual,m) <= z_m, for all seed j
            C_fixed,m,j + C_single,m,j/s_single,m + C_dual,m,j/s_dual,m <= D,   for all seed j
            s_min <= s_single,m, s_dual,m <= s_max

能耗函数：
       E_k(s_k)=E_k0*[lambda_k/s_k + (1-lambda_k)*((1-rho_k)+rho_k*s_k^2)]

该函数具有 U 形经济速度特征：速度过慢会增加保持/待机/夹持能耗，速度过快会增加
动态损耗、同步协调和冲击代价。

默认输入：
       outputs/four_case_framework/optimized_heuristic_2B3B_time_energy.csv
默认输出：
       outputs/nonlinear_programming/robust_mode_selection/robust_speed_mode_selection.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import OUTPUT_DIR  # noqa: E402
from common.utils import ensure_dirs  # noqa: E402


# ============================================================
# 默认参数
# ============================================================

DEFAULT_METHOD = "optimized_heuristic"
DEFAULT_SPEED_MIN = 0.60
DEFAULT_SPEED_MAX = 1.80
DEFAULT_LAMBDA_SINGLE = 0.35
DEFAULT_LAMBDA_DUAL = 0.45
DEFAULT_RHO_SINGLE = 0.35
DEFAULT_RHO_DUAL = 0.45

DEFAULT_DEADLINE_RATIOS = [
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.00, 1.05, 1.10, 1.15, 1.20, 1.25,
    1.30, 1.35, 1.40, 1.50,
]

# 截止时间基准：mean_2b / max_2b / min_2b。
DEFAULT_DEADLINE_BASE = "mean_2b"

GRID_COARSE_POINTS = 81
GRID_REFINE_POINTS = 41
GRID_REFINE_ROUNDS = 6
EPS = 1e-9


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SeedCase:
    counts_code: str
    n1: int
    n2: int
    n3: int
    n4: int
    total_tasks: int
    seed: int
    cmax_2b: float
    energy_2b: float
    cmax_3b: float
    energy_3b: float


@dataclass
class ModeFeature:
    base_cmax: float
    base_energy: float
    single_time: float
    dual_time: float
    fixed_time: float
    single_energy: float
    dual_energy: float
    fixed_energy: float


@dataclass
class RobustOptResult:
    feasible: bool
    required_speed_single: float
    required_speed_dual: float
    opt_speed_single: float | None
    opt_speed_dual: float | None
    worst_energy_z: float | None
    mean_energy_after_speed: float | None
    std_energy_after_speed: float | None
    max_cmax_after_speed: float | None
    mean_cmax_after_speed: float | None
    worst_energy_seed: int | None
    binding_time_seed: int | None
    infeasible_reason: str
    search_iterations: int


# ============================================================
# 工具函数
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def round_or_blank(value: Any, digits: int = 6) -> Any:
    if value is None:
        return ""
    try:
        value_float = float(value)
        if math.isinf(value_float):
            return "inf"
        if math.isnan(value_float):
            return ""
        return round(value_float, digits)
    except (TypeError, ValueError):
        return ""


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def counts_code_from_row(row: dict) -> str:
    if row.get("counts_code"):
        return str(row["counts_code"])
    n1 = safe_int(row.get("n1"))
    n2 = safe_int(row.get("n2"))
    n3 = safe_int(row.get("n3"))
    n4 = safe_int(row.get("n4"))
    return f"{n1}{n2}{n3}{n4}"


def get_method_columns(method: str) -> dict[str, str]:
    return {
        "cmax_2b": f"cmax_2B_{method}",
        "energy_2b": f"energy_2B_{method}",
        "cmax_3b": f"cmax_3B_{method}",
        "energy_3b": f"energy_3B_{method}",
    }


def scenario_type(n1: int, n2: int, n3: int, n4: int) -> str:
    single = n1 + n2
    dual = n3 + n4
    if n1 == n2 == n3 == n4:
        return "balanced"
    if n4 > max(n1, n2, n3):
        return "type4_dominant"
    if single > dual:
        return "single_arm_dominant"
    if dual > single:
        return "dual_arm_dominant"
    return "mixed_balanced"


# ============================================================
# 读取并分组 seed 数据
# ============================================================

def build_seed_cases(raw_rows: list[dict], method: str) -> list[SeedCase]:
    cols = get_method_columns(method)
    cases: list[SeedCase] = []
    for row in raw_rows:
        counts_code = counts_code_from_row(row)
        n1 = safe_int(row.get("n1"))
        n2 = safe_int(row.get("n2"))
        n3 = safe_int(row.get("n3"))
        n4 = safe_int(row.get("n4"))
        total_tasks = safe_int(row.get("total_tasks"), n1 + n2 + n3 + n4)
        seed = safe_int(row.get("seed"))
        cmax_2b = safe_float(row.get(cols["cmax_2b"]))
        energy_2b = safe_float(row.get(cols["energy_2b"]))
        cmax_3b = safe_float(row.get(cols["cmax_3b"]))
        energy_3b = safe_float(row.get(cols["energy_3b"]))
        if cmax_2b <= 0 or energy_2b <= 0 or cmax_3b <= 0 or energy_3b <= 0:
            continue
        cases.append(SeedCase(counts_code, n1, n2, n3, n4, total_tasks, seed, cmax_2b, energy_2b, cmax_3b, energy_3b))
    return cases


def group_seed_cases(cases: list[SeedCase]) -> dict[tuple, list[SeedCase]]:
    groups: dict[tuple, list[SeedCase]] = defaultdict(list)
    for case in cases:
        key = (case.counts_code, case.n1, case.n2, case.n3, case.n4, case.total_tasks)
        groups[key].append(case)
    return groups


def get_deadline_reference(cases: list[SeedCase], deadline_base: str) -> float:
    cmax_2b_values = [c.cmax_2b for c in cases]
    if deadline_base == "mean_2b":
        return mean(cmax_2b_values)
    if deadline_base == "max_2b":
        return max(cmax_2b_values)
    if deadline_base == "min_2b":
        return min(cmax_2b_values)
    raise ValueError("Unsupported deadline_base")


# ============================================================
# 任务特征拆分与 U 形能耗函数
# ============================================================

def build_mode_feature(base_cmax: float, base_energy: float, n1: int, n2: int, n3: int, n4: int, mode: str) -> ModeFeature:
    """从已有 seed 运行结果构造鲁棒速度层特征。

    严格原则：不重新设定启动成本、固定能耗、服务时间、物块能耗参数，
    也不重新运行主调度。每个 seed 只使用输入 CSV 中已有的：

        base_cmax、base_energy、n1、n2、n3、n4

    为了形成 s_single 与 s_dual 两个连续速度变量，仅按已有任务结构中
    单臂任务数量和双臂任务数量占比，将已有总时间/总能耗分摊到两类任务。
    这样 s_single=s_dual=1 时仍严格回到该 seed 的原始 Cmax/Energy 结果。
    """
    single_units = max(float(n1 + n2), 0.0)
    dual_units = max(float(n3 + n4), 0.0)
    total_units = single_units + dual_units

    if total_units <= EPS:
        single_share = 0.5
        dual_share = 0.5
    else:
        single_share = single_units / total_units
        dual_share = dual_units / total_units

    fixed_time = 0.0
    fixed_energy = 0.0
    single_time = base_cmax * single_share
    dual_time = base_cmax * dual_share
    single_energy = base_energy * single_share
    dual_energy = base_energy * dual_share

    return ModeFeature(
        base_cmax=base_cmax,
        base_energy=base_energy,
        single_time=single_time,
        dual_time=dual_time,
        fixed_time=fixed_time,
        single_energy=single_energy,
        dual_energy=dual_energy,
        fixed_energy=fixed_energy,
    )

def economic_energy_component(base_energy: float, speed: float, lambda_keep: float, rho: float) -> float:
    return base_energy * (
        lambda_keep / speed
        + (1.0 - lambda_keep) * ((1.0 - rho) + rho * speed * speed)
    )


def economic_speed_unconstrained(lambda_keep: float, rho: float, speed_min: float, speed_max: float) -> float:
    denom = 2.0 * (1.0 - lambda_keep) * rho
    if denom <= 0:
        return speed_min
    s = (lambda_keep / denom) ** (1.0 / 3.0)
    return min(speed_max, max(speed_min, s))


def cmax_after_speed(feature: ModeFeature, speed_single: float, speed_dual: float) -> float:
    return feature.fixed_time + feature.single_time / speed_single + feature.dual_time / speed_dual


def energy_after_speed(
    feature: ModeFeature,
    speed_single: float,
    speed_dual: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> float:
    return (
        feature.fixed_energy
        + economic_energy_component(feature.single_energy, speed_single, lambda_single, rho_single)
        + economic_energy_component(feature.dual_energy, speed_dual, lambda_dual, rho_dual)
    )


def linspace(left: float, right: float, n: int) -> list[float]:
    if n <= 1:
        return [0.5 * (left + right)]
    step = (right - left) / (n - 1)
    return [left + i * step for i in range(n)]


# ============================================================
# 鲁棒非线性优化
# ============================================================

def robust_objective_and_stats(
    features: list[ModeFeature],
    seeds: list[int],
    deadline: float,
    speed_single: float,
    speed_dual: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> tuple[bool, float, float, float, float, float, int | None, int | None]:
    energies: list[float] = []
    cmax_values: list[float] = []
    for f in features:
        c = cmax_after_speed(f, speed_single, speed_dual)
        if c > deadline + 1e-7:
            return False, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), None, None
        cmax_values.append(c)
        energies.append(energy_after_speed(f, speed_single, speed_dual, lambda_single, lambda_dual, rho_single, rho_dual))

    worst_energy = max(energies)
    worst_index = energies.index(worst_energy)
    max_cmax = max(cmax_values)
    binding_index = cmax_values.index(max_cmax)
    return (
        True,
        worst_energy,
        mean(energies),
        sample_std(energies),
        max_cmax,
        mean(cmax_values),
        seeds[worst_index] if worst_index < len(seeds) else None,
        seeds[binding_index] if binding_index < len(seeds) else None,
    )


def solve_robust_speed_problem(
    features: list[ModeFeature],
    seeds: list[int],
    deadline: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> RobustOptResult:
    if not features or deadline <= 0:
        return RobustOptResult(False, float("inf"), float("inf"), None, None, None, None, None, None, None, None, None, "invalid input", 0)

    min_times = [cmax_after_speed(f, speed_max, speed_max) for f in features]
    if max(min_times) > deadline + EPS:
        return RobustOptResult(False, float("inf"), float("inf"), None, None, None, None, None, None, None, None, None, "deadline cannot be met even at speed_max for both speed variables", 0)

    # 这两个字段用于说明单独变量的紧迫程度：固定另一类速度为 speed_max 时，该变量至少需要多快。
    required_single = max((f.single_time / max(deadline - f.fixed_time - f.dual_time / speed_max, EPS)) for f in features)
    required_dual = max((f.dual_time / max(deadline - f.fixed_time - f.single_time / speed_max, EPS)) for f in features)

    s_single_econ = economic_speed_unconstrained(lambda_single, rho_single, speed_min, speed_max)
    s_dual_econ = economic_speed_unconstrained(lambda_dual, rho_dual, speed_min, speed_max)
    feasible, worst, mean_e, std_e, max_c, mean_c, worst_seed, binding_seed = robust_objective_and_stats(
        features, seeds, deadline, s_single_econ, s_dual_econ, lambda_single, lambda_dual, rho_single, rho_dual
    )
    if feasible:
        return RobustOptResult(True, required_single, required_dual, s_single_econ, s_dual_econ, worst, mean_e, std_e, max_c, mean_c, worst_seed, binding_seed, "", 0)

    best_s1 = best_s2 = None
    best_worst = float("inf")
    best_stats = None
    iterations = 0
    left1, right1 = speed_min, speed_max
    left2, right2 = speed_min, speed_max
    points = GRID_COARSE_POINTS

    for _round in range(GRID_REFINE_ROUNDS + 1):
        for s1 in linspace(left1, right1, points):
            for s2 in linspace(left2, right2, points):
                iterations += 1
                feasible, worst, mean_e, std_e, max_c, mean_c, worst_seed, binding_seed = robust_objective_and_stats(
                    features, seeds, deadline, s1, s2, lambda_single, lambda_dual, rho_single, rho_dual
                )
                if not feasible:
                    continue
                if worst < best_worst:
                    best_worst = worst
                    best_s1 = s1
                    best_s2 = s2
                    best_stats = (mean_e, std_e, max_c, mean_c, worst_seed, binding_seed)

        if best_s1 is None or best_s2 is None or best_stats is None:
            return RobustOptResult(False, required_single, required_dual, None, None, None, None, None, None, None, None, None, "no feasible robust speed found", iterations)

        width1 = (right1 - left1) / max(points - 1, 1)
        width2 = (right2 - left2) / max(points - 1, 1)
        left1 = max(speed_min, best_s1 - 2.0 * width1)
        right1 = min(speed_max, best_s1 + 2.0 * width1)
        left2 = max(speed_min, best_s2 - 2.0 * width2)
        right2 = min(speed_max, best_s2 + 2.0 * width2)
        points = GRID_REFINE_POINTS

    mean_e, std_e, max_c, mean_c, worst_seed, binding_seed = best_stats
    return RobustOptResult(True, required_single, required_dual, best_s1, best_s2, best_worst, mean_e, std_e, max_c, mean_c, worst_seed, binding_seed, "", iterations)


# ============================================================
# 2B/3B 鲁棒推荐
# ============================================================

def compare_robust_modes(robust_2b: RobustOptResult, robust_3b: RobustOptResult) -> tuple[str, str, str, float | None, float | None]:
    if not robust_2b.feasible and not robust_3b.feasible:
        return "none", "infeasible_both", "neither 2B nor 3B can satisfy the deadline for all seeds", None, None
    if robust_2b.feasible and not robust_3b.feasible:
        return "2B", "recommend_2arm_only_robust_feasible", "only 2B is robustly feasible for all seeds", None, None
    if robust_3b.feasible and not robust_2b.feasible:
        return "3B", "recommend_3arm_only_robust_feasible", "only 3B is robustly feasible for all seeds", None, None
    assert robust_2b.worst_energy_z is not None and robust_3b.worst_energy_z is not None
    gap = robust_2b.worst_energy_z - robust_3b.worst_energy_z
    gap_percent = gap / robust_2b.worst_energy_z * 100.0 if robust_2b.worst_energy_z > 0 else None
    if gap > EPS:
        return "3B", "recommend_3arm_lower_worst_case_energy", "3B has lower worst-case energy under seed perturbations", gap, gap_percent
    if gap < -EPS:
        return "2B", "recommend_2arm_lower_worst_case_energy", "2B has lower worst-case energy under seed perturbations", gap, gap_percent
    return "2B", "similar_prefer_2arm", "worst-case energies are nearly equal, prefer simpler 2B", gap, gap_percent


# ============================================================
# 输出构造
# ============================================================

def build_output_rows(
    grouped_cases: dict[tuple, list[SeedCase]],
    method: str,
    deadline_ratios: list[float],
    deadline_base: str,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> list[dict]:
    rows: list[dict] = []

    for key, cases in grouped_cases.items():
        counts_code, n1, n2, n3, n4, total_tasks = key
        cases_sorted = sorted(cases, key=lambda c: c.seed)
        seeds = [c.seed for c in cases_sorted]
        seeds_used = ",".join(str(s) for s in seeds)

        cmax_2b_values = [c.cmax_2b for c in cases_sorted]
        energy_2b_values = [c.energy_2b for c in cases_sorted]
        cmax_3b_values = [c.cmax_3b for c in cases_sorted]
        energy_3b_values = [c.energy_3b for c in cases_sorted]
        deadline_reference = get_deadline_reference(cases_sorted, deadline_base)
        scen_type = scenario_type(n1, n2, n3, n4)

        features_2b = [build_mode_feature(c.cmax_2b, c.energy_2b, n1, n2, n3, n4, "2B") for c in cases_sorted]
        features_3b = [build_mode_feature(c.cmax_3b, c.energy_3b, n1, n2, n3, n4, "3B") for c in cases_sorted]

        for ratio in deadline_ratios:
            deadline = ratio * deadline_reference
            robust_2b = solve_robust_speed_problem(features_2b, seeds, deadline, speed_min, speed_max, lambda_single, lambda_dual, rho_single, rho_dual)
            robust_3b = solve_robust_speed_problem(features_3b, seeds, deadline, speed_min, speed_max, lambda_single, lambda_dual, rho_single, rho_dual)
            recommended_mode, recommendation, recommendation_reason, gap, gap_percent = compare_robust_modes(robust_2b, robust_3b)

            rows.append({
                "counts_code": counts_code,
                "n1": n1, "n2": n2, "n3": n3, "n4": n4,
                "total_tasks": total_tasks,
                "scenario_type": scen_type,
                "method": method,
                "seed_count": len(cases_sorted),
                "seeds_used": seeds_used,
                "deadline_base": deadline_base,
                "deadline_reference_cmax_2B": round_or_blank(deadline_reference, 6),
                "deadline_ratio": round_or_blank(ratio, 4),
                "deadline_value": round_or_blank(deadline, 6),

                "mean_cmax_2B_original": round_or_blank(mean(cmax_2b_values), 6),
                "std_cmax_2B_original": round_or_blank(sample_std(cmax_2b_values), 6),
                "max_cmax_2B_original": round_or_blank(max(cmax_2b_values), 6),
                "mean_energy_2B_original": round_or_blank(mean(energy_2b_values), 6),
                "std_energy_2B_original": round_or_blank(sample_std(energy_2b_values), 6),
                "max_energy_2B_original": round_or_blank(max(energy_2b_values), 6),
                "mean_cmax_3B_original": round_or_blank(mean(cmax_3b_values), 6),
                "std_cmax_3B_original": round_or_blank(sample_std(cmax_3b_values), 6),
                "max_cmax_3B_original": round_or_blank(max(cmax_3b_values), 6),
                "mean_energy_3B_original": round_or_blank(mean(energy_3b_values), 6),
                "std_energy_3B_original": round_or_blank(sample_std(energy_3b_values), 6),
                "max_energy_3B_original": round_or_blank(max(energy_3b_values), 6),

                "lambda_single": round_or_blank(lambda_single, 4),
                "lambda_dual": round_or_blank(lambda_dual, 4),
                "rho_single": round_or_blank(rho_single, 4),
                "rho_dual": round_or_blank(rho_dual, 4),
                "speed_min": round_or_blank(speed_min, 4),
                "speed_max": round_or_blank(speed_max, 4),

                "robust_feasible_2B": int(robust_2b.feasible),
                "required_speed_single_2B": round_or_blank(robust_2b.required_speed_single, 6),
                "required_speed_dual_2B": round_or_blank(robust_2b.required_speed_dual, 6),
                "opt_speed_single_2B": round_or_blank(robust_2b.opt_speed_single, 6),
                "opt_speed_dual_2B": round_or_blank(robust_2b.opt_speed_dual, 6),
                "worst_energy_z_2B": round_or_blank(robust_2b.worst_energy_z, 6),
                "mean_energy_after_speed_2B": round_or_blank(robust_2b.mean_energy_after_speed, 6),
                "std_energy_after_speed_2B": round_or_blank(robust_2b.std_energy_after_speed, 6),
                "max_cmax_after_speed_2B": round_or_blank(robust_2b.max_cmax_after_speed, 6),
                "mean_cmax_after_speed_2B": round_or_blank(robust_2b.mean_cmax_after_speed, 6),
                "worst_energy_seed_2B": robust_2b.worst_energy_seed if robust_2b.worst_energy_seed is not None else "",
                "binding_time_seed_2B": robust_2b.binding_time_seed if robust_2b.binding_time_seed is not None else "",
                "search_iterations_2B": robust_2b.search_iterations,
                "infeasible_reason_2B": robust_2b.infeasible_reason,

                "robust_feasible_3B": int(robust_3b.feasible),
                "required_speed_single_3B": round_or_blank(robust_3b.required_speed_single, 6),
                "required_speed_dual_3B": round_or_blank(robust_3b.required_speed_dual, 6),
                "opt_speed_single_3B": round_or_blank(robust_3b.opt_speed_single, 6),
                "opt_speed_dual_3B": round_or_blank(robust_3b.opt_speed_dual, 6),
                "worst_energy_z_3B": round_or_blank(robust_3b.worst_energy_z, 6),
                "mean_energy_after_speed_3B": round_or_blank(robust_3b.mean_energy_after_speed, 6),
                "std_energy_after_speed_3B": round_or_blank(robust_3b.std_energy_after_speed, 6),
                "max_cmax_after_speed_3B": round_or_blank(robust_3b.max_cmax_after_speed, 6),
                "mean_cmax_after_speed_3B": round_or_blank(robust_3b.mean_cmax_after_speed, 6),
                "worst_energy_seed_3B": robust_3b.worst_energy_seed if robust_3b.worst_energy_seed is not None else "",
                "binding_time_seed_3B": robust_3b.binding_time_seed if robust_3b.binding_time_seed is not None else "",
                "search_iterations_3B": robust_3b.search_iterations,
                "infeasible_reason_3B": robust_3b.infeasible_reason,

                "worst_energy_gap_2B_minus_3B": round_or_blank(gap, 6),
                "worst_energy_gap_percent_vs_2B": round_or_blank(gap_percent, 4),
                "recommended_mode": recommended_mode,
                "recommendation": recommendation,
                "recommendation_reason": recommendation_reason,
                "robust_model_formula": "min z, s.t. E_fixed_j+E_single_j(s_single)+E_dual_j(s_dual)<=z and C_fixed_j+C_single_j/s_single+C_dual_j/s_dual<=D for all seeds",
                "energy_formula": "E_k(s)=E_k0*(lambda_k/s+(1-lambda_k)*((1-rho_k)+rho_k*s^2))",
                "model_meaning": "choose one pair of single/dual speeds for all seed perturbations and minimize worst-case energy",
            })

    return sorted(rows, key=lambda r: (r["counts_code"], safe_float(r["deadline_ratio"])))


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Robust two-variable U-shaped nonlinear speed-energy optimization under seed perturbations.")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--input", default="")
    parser.add_argument("--deadline-ratios", default=",".join(str(x) for x in DEFAULT_DEADLINE_RATIOS))
    parser.add_argument("--deadline-base", default=DEFAULT_DEADLINE_BASE, choices=["mean_2b", "max_2b", "min_2b"])
    parser.add_argument("--speed-min", type=float, default=DEFAULT_SPEED_MIN)
    parser.add_argument("--speed-max", type=float, default=DEFAULT_SPEED_MAX)
    parser.add_argument("--lambda-single", type=float, default=DEFAULT_LAMBDA_SINGLE)
    parser.add_argument("--lambda-dual", type=float, default=DEFAULT_LAMBDA_DUAL)
    parser.add_argument("--rho-single", type=float, default=DEFAULT_RHO_SINGLE)
    parser.add_argument("--rho-dual", type=float, default=DEFAULT_RHO_DUAL)
    args = parser.parse_args()

    method = args.method.strip()
    input_path = Path(args.input) if args.input.strip() else OUTPUT_DIR / "four_case_framework" / f"{method}_2B3B_time_energy.csv"
    output_dir = OUTPUT_DIR / "nonlinear_programming" / "robust_mode_selection"
    output_path = output_dir / "robust_speed_mode_selection.csv"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)
    seed_cases = build_seed_cases(raw_rows, method)
    grouped_cases = group_seed_cases(seed_cases)
    deadline_ratios = parse_float_list(args.deadline_ratios)

    rows = build_output_rows(
        grouped_cases, method, deadline_ratios, args.deadline_base,
        args.speed_min, args.speed_max,
        args.lambda_single, args.lambda_dual, args.rho_single, args.rho_dual,
    )

    fieldnames = [
        "counts_code", "n1", "n2", "n3", "n4", "total_tasks", "scenario_type", "method", "seed_count", "seeds_used",
        "deadline_base", "deadline_reference_cmax_2B", "deadline_ratio", "deadline_value",
        "mean_cmax_2B_original", "std_cmax_2B_original", "max_cmax_2B_original", "mean_energy_2B_original", "std_energy_2B_original", "max_energy_2B_original",
        "mean_cmax_3B_original", "std_cmax_3B_original", "max_cmax_3B_original", "mean_energy_3B_original", "std_energy_3B_original", "max_energy_3B_original",
        "lambda_single", "lambda_dual", "rho_single", "rho_dual", "speed_min", "speed_max",
        "robust_feasible_2B", "required_speed_single_2B", "required_speed_dual_2B", "opt_speed_single_2B", "opt_speed_dual_2B", "worst_energy_z_2B",
        "mean_energy_after_speed_2B", "std_energy_after_speed_2B", "max_cmax_after_speed_2B", "mean_cmax_after_speed_2B", "worst_energy_seed_2B", "binding_time_seed_2B", "search_iterations_2B", "infeasible_reason_2B",
        "robust_feasible_3B", "required_speed_single_3B", "required_speed_dual_3B", "opt_speed_single_3B", "opt_speed_dual_3B", "worst_energy_z_3B",
        "mean_energy_after_speed_3B", "std_energy_after_speed_3B", "max_cmax_after_speed_3B", "mean_cmax_after_speed_3B", "worst_energy_seed_3B", "binding_time_seed_3B", "search_iterations_3B", "infeasible_reason_3B",
        "worst_energy_gap_2B_minus_3B", "worst_energy_gap_percent_vs_2B", "recommended_mode", "recommendation", "recommendation_reason",
        "robust_model_formula", "energy_formula", "model_meaning",
    ]

    save_csv(rows, output_path, fieldnames)

    print("Robust two-variable nonlinear speed-energy optimization finished.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Grouped counts_code cases: {len(grouped_cases)}")
    print(f"Rows: {len(rows)}")
    print("Model: min z with single-arm speed and dual-arm speed under U-shaped economic speed energy.")


if __name__ == "__main__":
    main()
