# -*- coding: utf-8 -*-
"""非线性规划 2：随机 seed 扰动下的鲁棒速度-能耗优化与二臂/三臂选择。

新版鲁棒性分析文件

核心：
        对同一个 counts_code 下的 seed=0,1,2 不再取平均；
        而是选择一个统一速度倍率 s，使该速度对所有 seed 都满足截止时间；
        并最小化所有 seed 中的最坏能耗。

鲁棒速度-能耗非线性规划模型：
    对某个模式 m in {2B, 3B}，seed 记为 j=0,1,2。

    已知：
        C^0_{m,j}：模式 m 在 seed j 下的默认完工时间；
        E^0_{m,j}：模式 m 在 seed j 下的默认能耗。

    决策变量：
        s_m：模式 m 采用的统一速度倍率；
        z_m：模式 m 在所有 seed 中的最坏能耗上界。

    模型：
        min     z_m

        s.t.    E^0_{m,j} * [(1-rho) + rho*s_m^2] <= z_m,   for all seed j
                C^0_{m,j} / s_m <= D,                       for all seed j
                s_min <= s_m <= s_max

    其中 D 是统一截止时间：
        D = deadline_ratio * reference_cmax_2B

    默认 reference_cmax_2B 取同一 counts_code 下 2B 的 seed 平均 Cmax。

解释：
    该模型回答的问题是：
        如果任务位置存在随机扰动，而且不能针对每个 seed 单独调速，
        那么应该给 2B 或 3B 设置怎样的统一速度，才能保证所有 seed 都按时完成，
        并且最坏情况下的能耗最低？

默认输入：
    outputs/four_case_framework/optimized_2B3B_time_energy.csv

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

DEFAULT_METHOD = "optimized"

DEFAULT_SPEED_MIN = 0.60
DEFAULT_SPEED_MAX = 1.60
DEFAULT_RHO = 0.35

DEFAULT_DEADLINE_RATIOS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.35,
    1.40,
    1.50,
]

# 截止时间基准：
# mean_2b：同一 counts_code 下 2B 的平均 Cmax，默认；
# max_2b ：同一 counts_code 下 2B 的最大 Cmax，更保守；
# min_2b ：同一 counts_code 下 2B 的最小 Cmax，更激进。
DEFAULT_DEADLINE_BASE = "mean_2b"

EPS = 1e-9


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SeedCase:
    """同一个 counts_code 下某个 seed 的 2B/3B 结果。"""

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
class RobustOptResult:
    """某个模式在多个 seed 下的鲁棒速度优化结果。"""

    feasible: bool

    required_uniform_speed: float
    opt_uniform_speed: float | None

    worst_energy_z: float | None
    mean_energy_after_speed: float | None
    std_energy_after_speed: float | None

    max_cmax_after_speed: float | None
    mean_cmax_after_speed: float | None

    worst_energy_seed: int | None
    binding_time_seed: int | None

    infeasible_reason: str


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
    """读取原始 CSV，不对 seed 取平均。"""
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

        cases.append(
            SeedCase(
                counts_code=counts_code,
                n1=n1,
                n2=n2,
                n3=n3,
                n4=n4,
                total_tasks=total_tasks,
                seed=seed,
                cmax_2b=cmax_2b,
                energy_2b=energy_2b,
                cmax_3b=cmax_3b,
                energy_3b=energy_3b,
            )
        )

    return cases


def group_seed_cases(cases: list[SeedCase]) -> dict[tuple, list[SeedCase]]:
    """按 counts_code 分组。"""
    groups: dict[tuple, list[SeedCase]] = defaultdict(list)

    for case in cases:
        key = (
            case.counts_code,
            case.n1,
            case.n2,
            case.n3,
            case.n4,
            case.total_tasks,
        )
        groups[key].append(case)

    return groups


def get_deadline_reference(cases: list[SeedCase], deadline_base: str) -> float:
    """根据指定方式计算截止时间基准。"""
    cmax_2b_values = [c.cmax_2b for c in cases]

    if deadline_base == "mean_2b":
        return mean(cmax_2b_values)

    if deadline_base == "max_2b":
        return max(cmax_2b_values)

    if deadline_base == "min_2b":
        return min(cmax_2b_values)

    raise ValueError(
        f"Unsupported deadline_base: {deadline_base}. "
        "Use mean_2b, max_2b or min_2b."
    )


# ============================================================
# 鲁棒非线性优化
# ============================================================

def energy_after_speed(base_energy: float, speed: float, rho: float) -> float:
    return base_energy * ((1.0 - rho) + rho * speed * speed)


def cmax_after_speed(base_cmax: float, speed: float) -> float:
    return base_cmax / speed


def solve_robust_speed_problem(
    cmax_values: list[float],
    energy_values: list[float],
    seeds: list[int],
    deadline: float,
    speed_min: float,
    speed_max: float,
    rho: float,
) -> RobustOptResult:
    """求解鲁棒速度优化问题。

    模型：
        min z
        s.t. E_j0*((1-rho)+rho*s^2) <= z, for all j
             C_j0/s <= D, for all j
             s_min <= s <= s_max

    因为能耗关于 s 单调递增，所以最优统一速度为：
        s* = max(s_min, max_j C_j0/D)

    然后：
        z* = max_j E_j0*((1-rho)+rho*s*^2)
    """
    if not cmax_values or not energy_values or len(cmax_values) != len(energy_values):
        return RobustOptResult(
            feasible=False,
            required_uniform_speed=float("inf"),
            opt_uniform_speed=None,
            worst_energy_z=None,
            mean_energy_after_speed=None,
            std_energy_after_speed=None,
            max_cmax_after_speed=None,
            mean_cmax_after_speed=None,
            worst_energy_seed=None,
            binding_time_seed=None,
            infeasible_reason="invalid empty input",
        )

    if deadline <= 0:
        return RobustOptResult(
            feasible=False,
            required_uniform_speed=float("inf"),
            opt_uniform_speed=None,
            worst_energy_z=None,
            mean_energy_after_speed=None,
            std_energy_after_speed=None,
            max_cmax_after_speed=None,
            mean_cmax_after_speed=None,
            worst_energy_seed=None,
            binding_time_seed=None,
            infeasible_reason="deadline is not positive",
        )

    required_speeds = [c / deadline for c in cmax_values]
    required_uniform_speed = max(required_speeds)
    binding_index = required_speeds.index(required_uniform_speed)

    opt_speed = max(speed_min, required_uniform_speed)

    if opt_speed > speed_max + EPS:
        return RobustOptResult(
            feasible=False,
            required_uniform_speed=required_uniform_speed,
            opt_uniform_speed=None,
            worst_energy_z=None,
            mean_energy_after_speed=None,
            std_energy_after_speed=None,
            max_cmax_after_speed=None,
            mean_cmax_after_speed=None,
            worst_energy_seed=None,
            binding_time_seed=seeds[binding_index] if binding_index < len(seeds) else None,
            infeasible_reason="required uniform speed exceeds speed_max",
        )

    opt_speed = min(opt_speed, speed_max)

    energies_after = [
        energy_after_speed(e0, opt_speed, rho)
        for e0 in energy_values
    ]

    cmax_after = [
        cmax_after_speed(c0, opt_speed)
        for c0 in cmax_values
    ]

    worst_energy = max(energies_after)
    worst_index = energies_after.index(worst_energy)

    return RobustOptResult(
        feasible=True,
        required_uniform_speed=required_uniform_speed,
        opt_uniform_speed=opt_speed,
        worst_energy_z=worst_energy,
        mean_energy_after_speed=mean(energies_after),
        std_energy_after_speed=sample_std(energies_after),
        max_cmax_after_speed=max(cmax_after),
        mean_cmax_after_speed=mean(cmax_after),
        worst_energy_seed=seeds[worst_index] if worst_index < len(seeds) else None,
        binding_time_seed=seeds[binding_index] if binding_index < len(seeds) else None,
        infeasible_reason="",
    )


# ============================================================
# 2B/3B 鲁棒推荐
# ============================================================

def compare_robust_modes(
    robust_2b: RobustOptResult,
    robust_3b: RobustOptResult,
) -> tuple[str, str, str, float | None, float | None]:
    """比较 2B 和 3B 的鲁棒最坏能耗。"""
    if not robust_2b.feasible and not robust_3b.feasible:
        return (
            "none",
            "infeasible_both",
            "neither 2B nor 3B can satisfy the deadline for all seeds",
            None,
            None,
        )

    if robust_2b.feasible and not robust_3b.feasible:
        return (
            "2B",
            "recommend_2arm_only_robust_feasible",
            "only 2B is robustly feasible for all seeds",
            None,
            None,
        )

    if robust_3b.feasible and not robust_2b.feasible:
        return (
            "3B",
            "recommend_3arm_only_robust_feasible",
            "only 3B is robustly feasible for all seeds",
            None,
            None,
        )

    assert robust_2b.worst_energy_z is not None
    assert robust_3b.worst_energy_z is not None

    gap = robust_2b.worst_energy_z - robust_3b.worst_energy_z

    gap_percent = (
        gap / robust_2b.worst_energy_z * 100.0
        if robust_2b.worst_energy_z > 0
        else None
    )

    if gap > EPS:
        return (
            "3B",
            "recommend_3arm_lower_worst_case_energy",
            "3B has lower worst-case energy under seed perturbations",
            gap,
            gap_percent,
        )

    if gap < -EPS:
        return (
            "2B",
            "recommend_2arm_lower_worst_case_energy",
            "2B has lower worst-case energy under seed perturbations",
            gap,
            gap_percent,
        )

    return (
        "2B",
        "similar_prefer_2arm",
        "worst-case energies are nearly equal, prefer simpler 2B",
        gap,
        gap_percent,
    )


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
    rho: float,
) -> list[dict]:
    """生成唯一输出表。"""
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

        for ratio in deadline_ratios:
            deadline = ratio * deadline_reference

            robust_2b = solve_robust_speed_problem(
                cmax_values=cmax_2b_values,
                energy_values=energy_2b_values,
                seeds=seeds,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                rho=rho,
            )

            robust_3b = solve_robust_speed_problem(
                cmax_values=cmax_3b_values,
                energy_values=energy_3b_values,
                seeds=seeds,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                rho=rho,
            )

            (
                recommended_mode,
                recommendation,
                recommendation_reason,
                worst_energy_gap_2b_minus_3b,
                worst_energy_gap_percent,
            ) = compare_robust_modes(robust_2b, robust_3b)

            rows.append(
                {
                    "counts_code": counts_code,
                    "n1": n1,
                    "n2": n2,
                    "n3": n3,
                    "n4": n4,
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

                    "rho_speed_sensitive_energy": round_or_blank(rho, 4),
                    "speed_min": round_or_blank(speed_min, 4),
                    "speed_max": round_or_blank(speed_max, 4),

                    "robust_feasible_2B": int(robust_2b.feasible),
                    "required_uniform_speed_2B": round_or_blank(robust_2b.required_uniform_speed, 6),
                    "opt_uniform_speed_2B": round_or_blank(robust_2b.opt_uniform_speed, 6),
                    "worst_energy_z_2B": round_or_blank(robust_2b.worst_energy_z, 6),
                    "mean_energy_after_speed_2B": round_or_blank(robust_2b.mean_energy_after_speed, 6),
                    "std_energy_after_speed_2B": round_or_blank(robust_2b.std_energy_after_speed, 6),
                    "max_cmax_after_speed_2B": round_or_blank(robust_2b.max_cmax_after_speed, 6),
                    "mean_cmax_after_speed_2B": round_or_blank(robust_2b.mean_cmax_after_speed, 6),
                    "worst_energy_seed_2B": robust_2b.worst_energy_seed if robust_2b.worst_energy_seed is not None else "",
                    "binding_time_seed_2B": robust_2b.binding_time_seed if robust_2b.binding_time_seed is not None else "",
                    "infeasible_reason_2B": robust_2b.infeasible_reason,

                    "robust_feasible_3B": int(robust_3b.feasible),
                    "required_uniform_speed_3B": round_or_blank(robust_3b.required_uniform_speed, 6),
                    "opt_uniform_speed_3B": round_or_blank(robust_3b.opt_uniform_speed, 6),
                    "worst_energy_z_3B": round_or_blank(robust_3b.worst_energy_z, 6),
                    "mean_energy_after_speed_3B": round_or_blank(robust_3b.mean_energy_after_speed, 6),
                    "std_energy_after_speed_3B": round_or_blank(robust_3b.std_energy_after_speed, 6),
                    "max_cmax_after_speed_3B": round_or_blank(robust_3b.max_cmax_after_speed, 6),
                    "mean_cmax_after_speed_3B": round_or_blank(robust_3b.mean_cmax_after_speed, 6),
                    "worst_energy_seed_3B": robust_3b.worst_energy_seed if robust_3b.worst_energy_seed is not None else "",
                    "binding_time_seed_3B": robust_3b.binding_time_seed if robust_3b.binding_time_seed is not None else "",
                    "infeasible_reason_3B": robust_3b.infeasible_reason,

                    "worst_energy_gap_2B_minus_3B": round_or_blank(worst_energy_gap_2b_minus_3b, 6),
                    "worst_energy_gap_percent_vs_2B": round_or_blank(worst_energy_gap_percent, 4),
                    "recommended_mode": recommended_mode,
                    "recommendation": recommendation,
                    "recommendation_reason": recommendation_reason,

                    "robust_model_formula": (
                        "min z, s.t. E_j0*((1-rho)+rho*s^2)<=z, "
                        "C_j0/s<=D for all seeds, s_min<=s<=s_max"
                    ),
                    "model_meaning": (
                        "choose one uniform speed for all seed perturbations and minimize worst-case energy"
                    ),
                }
            )

    return sorted(
        rows,
        key=lambda r: (r["counts_code"], safe_float(r["deadline_ratio"])),
    )


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust nonlinear speed-energy optimization under seed perturbations."
    )

    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="method suffix in input CSV, e.g. optimized or optimized_heuristic",
    )

    parser.add_argument(
        "--input",
        default="",
        help="input CSV path. If empty, use outputs/four_case_framework/{method}_2B3B_time_energy.csv",
    )

    parser.add_argument(
        "--deadline-ratios",
        default=",".join(str(x) for x in DEFAULT_DEADLINE_RATIOS),
        help="comma separated deadline ratios",
    )

    parser.add_argument(
        "--deadline-base",
        default=DEFAULT_DEADLINE_BASE,
        choices=["mean_2b", "max_2b", "min_2b"],
        help="reference Cmax used to define deadline D",
    )

    parser.add_argument(
        "--speed-min",
        type=float,
        default=DEFAULT_SPEED_MIN,
        help="minimum speed multiplier",
    )

    parser.add_argument(
        "--speed-max",
        type=float,
        default=DEFAULT_SPEED_MAX,
        help="maximum speed multiplier",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=DEFAULT_RHO,
        help="speed-sensitive energy ratio",
    )

    args = parser.parse_args()

    method = args.method.strip()

    input_path = (
        Path(args.input)
        if args.input.strip()
        else OUTPUT_DIR / "four_case_framework" / f"{method}_2B3B_time_energy.csv"
    )

    output_dir = OUTPUT_DIR / "nonlinear_programming" / "robust_mode_selection"
    output_path = output_dir / "robust_speed_mode_selection.csv"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)

    seed_cases = build_seed_cases(
        raw_rows=raw_rows,
        method=method,
    )

    grouped_cases = group_seed_cases(seed_cases)

    deadline_ratios = parse_float_list(args.deadline_ratios)

    rows = build_output_rows(
        grouped_cases=grouped_cases,
        method=method,
        deadline_ratios=deadline_ratios,
        deadline_base=args.deadline_base,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        rho=args.rho,
    )

    fieldnames = [
        "counts_code",
        "n1",
        "n2",
        "n3",
        "n4",
        "total_tasks",
        "scenario_type",
        "method",
        "seed_count",
        "seeds_used",

        "deadline_base",
        "deadline_reference_cmax_2B",
        "deadline_ratio",
        "deadline_value",

        "mean_cmax_2B_original",
        "std_cmax_2B_original",
        "max_cmax_2B_original",
        "mean_energy_2B_original",
        "std_energy_2B_original",
        "max_energy_2B_original",

        "mean_cmax_3B_original",
        "std_cmax_3B_original",
        "max_cmax_3B_original",
        "mean_energy_3B_original",
        "std_energy_3B_original",
        "max_energy_3B_original",

        "rho_speed_sensitive_energy",
        "speed_min",
        "speed_max",

        "robust_feasible_2B",
        "required_uniform_speed_2B",
        "opt_uniform_speed_2B",
        "worst_energy_z_2B",
        "mean_energy_after_speed_2B",
        "std_energy_after_speed_2B",
        "max_cmax_after_speed_2B",
        "mean_cmax_after_speed_2B",
        "worst_energy_seed_2B",
        "binding_time_seed_2B",
        "infeasible_reason_2B",

        "robust_feasible_3B",
        "required_uniform_speed_3B",
        "opt_uniform_speed_3B",
        "worst_energy_z_3B",
        "mean_energy_after_speed_3B",
        "std_energy_after_speed_3B",
        "max_cmax_after_speed_3B",
        "mean_cmax_after_speed_3B",
        "worst_energy_seed_3B",
        "binding_time_seed_3B",
        "infeasible_reason_3B",

        "worst_energy_gap_2B_minus_3B",
        "worst_energy_gap_percent_vs_2B",
        "recommended_mode",
        "recommendation",
        "recommendation_reason",

        "robust_model_formula",
        "model_meaning",
    ]

    save_csv(rows, output_path, fieldnames)

    print("Robust nonlinear speed-energy optimization finished.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Grouped counts_code cases: {len(grouped_cases)}")
    print(f"Rows: {len(rows)}")
    print("Model: min z, s.t. E_j0*((1-rho)+rho*s^2)<=z and C_j0/s<=D for all seeds")
    print("Meaning: one uniform speed is selected for all seed perturbations, minimizing worst-case energy.")


if __name__ == "__main__":
    main()
