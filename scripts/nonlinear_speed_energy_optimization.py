# -*- coding: utf-8 -*-
"""非线性规划：双速度变量的速度-能耗优化、KKT 验证与小型敏感性分析。

本脚本只读取文件夹中已有的 2B/3B 完工时间与能耗运行结果，不重新运行主调度，
不重新设置启动成本、固定能耗、服务时间或物块能耗参数。与旧版本相比，本版本完成两点核心修改：

1. 将原来的单一速度倍率 s 扩展为两个连续速度变量：
       s_single：单臂任务速度倍率
       s_dual  ：双臂协同任务速度倍率

2. 将原来的单调递增能耗函数改为经济速度型 U 形函数：
       E_k(s_k)=E_k0*[lambda_k/s_k + (1-lambda_k)*((1-rho_k)+rho_k*s_k^2)]
   其中 1/s 项表示速度过慢导致的保持/待机/夹持能耗增加，s^2 项表示速度过快
   导致的电机动态损耗、冲击和协同控制代价增加。

确定性模型：
       min  E_fixed + E_single(s_single) + E_dual(s_dual)
       s.t. C_fixed + C_single/s_single + C_dual/s_dual <= D
            s_min <= s_single, s_dual <= s_max

默认输入：
       outputs/four_case_framework/optimized_heuristic_2B3B_time_energy.csv
默认输出：
       outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization.csv
       outputs/nonlinear_programming/speed_energy_optimization/kkt_verification.csv
       outputs/nonlinear_programming/speed_energy_optimization/sensitivity_analysis.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
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

# 默认参数

DEFAULT_METHOD = "optimized_heuristic"

DEFAULT_DEADLINE_RATIOS = [
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.00, 1.05, 1.10, 1.15, 1.20, 1.25,
    1.30, 1.35, 1.40, 1.50,
]

DEFAULT_SPEED_MIN = 0.60
DEFAULT_SPEED_MAX = 1.80

# U 形经济速度能耗参数。lambda 越大，速度过慢惩罚越强；rho 越大，速度过快惩罚越强。
DEFAULT_LAMBDA_SINGLE = 0.35
DEFAULT_LAMBDA_DUAL = 0.45
DEFAULT_RHO_SINGLE = 0.35
DEFAULT_RHO_DUAL = 0.45

# 二维非线性搜索参数：先粗网格，再围绕最优点逐步细化。
GRID_COARSE_POINTS = 81
GRID_REFINE_POINTS = 41
GRID_REFINE_ROUNDS = 6

EPS = 1e-9
KKT_TOL = 2e-3

# 数据结构

@dataclass
class AveragedCase:
    counts_code: str
    n1: int
    n2: int
    n3: int
    n4: int
    total_tasks: int
    seed_count: int
    seeds_used: str
    mean_cmax_2b: float
    mean_energy_2b: float
    mean_cmax_3b: float
    mean_energy_3b: float
    std_cmax_2b: float
    std_energy_2b: float
    std_cmax_3b: float
    std_energy_3b: float


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
    single_task_share: float
    dual_task_share: float


@dataclass
class SpeedOptResult:
    feasible: bool
    feature: ModeFeature
    deadline: float
    required_speed_if_equal: float
    opt_speed_single: float | None
    opt_speed_dual: float | None
    opt_energy: float | None
    opt_cmax: float | None
    active_constraints: str
    search_iterations: int
    infeasible_reason: str


@dataclass
class KKTCheck:
    feasible: bool
    g_single_lower: float | None
    g_single_upper: float | None
    g_dual_lower: float | None
    g_dual_upper: float | None
    g_deadline: float | None
    mu_single_lower: float | None
    mu_single_upper: float | None
    mu_dual_lower: float | None
    mu_dual_upper: float | None
    mu_deadline: float | None
    stationarity_single_residual: float | None
    stationarity_dual_residual: float | None
    max_complementarity_error: float | None
    primal_min: float | None
    dual_min: float | None
    kkt_pass: bool
    kkt_note: str

# 基础工具函数

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

# 数据读取与 seed 平均

def build_averaged_cases(raw_rows: list[dict], method: str) -> list[AveragedCase]:
    cols = get_method_columns(method)
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for row in raw_rows:
        counts_code = counts_code_from_row(row)
        n1 = safe_int(row.get("n1"))
        n2 = safe_int(row.get("n2"))
        n3 = safe_int(row.get("n3"))
        n4 = safe_int(row.get("n4"))
        total_tasks = safe_int(row.get("total_tasks"), n1 + n2 + n3 + n4)

        cmax_2b = safe_float(row.get(cols["cmax_2b"]))
        energy_2b = safe_float(row.get(cols["energy_2b"]))
        cmax_3b = safe_float(row.get(cols["cmax_3b"]))
        energy_3b = safe_float(row.get(cols["energy_3b"]))
        if cmax_2b <= 0 or energy_2b <= 0 or cmax_3b <= 0 or energy_3b <= 0:
            continue

        groups[(counts_code, n1, n2, n3, n4, total_tasks)].append(row)

    averaged_cases: list[AveragedCase] = []
    for key, rows in groups.items():
        counts_code, n1, n2, n3, n4, total_tasks = key
        seeds = [safe_int(r.get("seed")) for r in rows]
        cmax_2b_values = [safe_float(r.get(cols["cmax_2b"])) for r in rows]
        energy_2b_values = [safe_float(r.get(cols["energy_2b"])) for r in rows]
        cmax_3b_values = [safe_float(r.get(cols["cmax_3b"])) for r in rows]
        energy_3b_values = [safe_float(r.get(cols["energy_3b"])) for r in rows]
        averaged_cases.append(
            AveragedCase(
                counts_code=counts_code,
                n1=n1, n2=n2, n3=n3, n4=n4,
                total_tasks=total_tasks,
                seed_count=len(rows),
                seeds_used=",".join(str(x) for x in sorted(seeds)),
                mean_cmax_2b=mean(cmax_2b_values),
                mean_energy_2b=mean(energy_2b_values),
                mean_cmax_3b=mean(cmax_3b_values),
                mean_energy_3b=mean(energy_3b_values),
                std_cmax_2b=sample_std(cmax_2b_values),
                std_energy_2b=sample_std(energy_2b_values),
                std_cmax_3b=sample_std(cmax_3b_values),
                std_energy_3b=sample_std(energy_3b_values),
            )
        )
    return sorted(averaged_cases, key=lambda x: x.counts_code)

# 任务类型特征拆分：只用主结果，不改主调度

def build_mode_feature(
    base_cmax: float,
    base_energy: float,
    n1: int,
    n2: int,
    n3: int,
    n4: int,
    mode: str,
) -> ModeFeature:
    """从已有运行结果构造单臂/双臂速度层特征。

    严格原则：本函数不重新设定任何启动成本、固定能耗、服务时间或物块能耗参数，
    也不调用求解器重新计算调度结果。它只使用输入 CSV 中已经存在的运行结果字段：

        base_cmax  = 已有 2B/3B 完工时间结果
        base_energy= 已有 2B/3B 能耗结果
        n1,n2,n3,n4= 已有任务结构计数字段

    由于当前提交包内没有逐任务 schedule 明细，本脚本不额外伪造 setup、center-zone、
    startup 等成本。为了得到两个速度变量，只按照任务类别数量将已有总时间/总能耗
    分摊到单臂任务和双臂任务两类：

        single_share = (n1+n2)/(n1+n2+n3+n4)
        dual_share   = (n3+n4)/(n1+n2+n3+n4)

    因此 s_single=s_dual=1 时：

        C_single + C_dual = base_cmax
        E_single + E_dual = base_energy

    即速度优化层严格锚定已有运行结果，不引入与主模型冲突的新成本数据。
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

    # 不重新设定固定时间或固定能耗；只保留已有总结果的单臂/双臂分摊。
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
        single_task_share=single_share,
        dual_task_share=dual_share,
    )

# 双变量 U 形速度-能耗模型

def economic_energy_component(base_energy: float, speed: float, lambda_keep: float, rho: float) -> float:
    return base_energy * (
        lambda_keep / speed
        + (1.0 - lambda_keep) * ((1.0 - rho) + rho * speed * speed)
    )


def economic_energy_derivative(base_energy: float, speed: float, lambda_keep: float, rho: float) -> float:
    return base_energy * (
        -lambda_keep / (speed * speed)
        + 2.0 * (1.0 - lambda_keep) * rho * speed
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


def is_feasible(feature: ModeFeature, deadline: float, speed_single: float, speed_dual: float) -> bool:
    return cmax_after_speed(feature, speed_single, speed_dual) <= deadline + 1e-7


def linspace(left: float, right: float, n: int) -> list[float]:
    if n <= 1:
        return [0.5 * (left + right)]
    step = (right - left) / (n - 1)
    return [left + i * step for i in range(n)]


def two_dimensional_grid_refine(
    feature: ModeFeature,
    deadline: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> tuple[float | None, float | None, float | None, float | None, int]:
    """二维非线性搜索。返回 s_single, s_dual, energy, cmax, iterations。"""
    min_time = cmax_after_speed(feature, speed_max, speed_max)
    if min_time > deadline + EPS:
        return None, None, None, None, 0

    # 若经济速度已满足截止时间，它就是无时间压力下的 U 形能耗最小点。
    s_single_econ = economic_speed_unconstrained(lambda_single, rho_single, speed_min, speed_max)
    s_dual_econ = economic_speed_unconstrained(lambda_dual, rho_dual, speed_min, speed_max)
    if is_feasible(feature, deadline, s_single_econ, s_dual_econ):
        e = energy_after_speed(feature, s_single_econ, s_dual_econ, lambda_single, lambda_dual, rho_single, rho_dual)
        c = cmax_after_speed(feature, s_single_econ, s_dual_econ)
        return s_single_econ, s_dual_econ, e, c, 0

    best_s1 = None
    best_s2 = None
    best_energy = float("inf")
    best_cmax = None
    iterations = 0

    left1, right1 = speed_min, speed_max
    left2, right2 = speed_min, speed_max
    points = GRID_COARSE_POINTS

    for round_id in range(GRID_REFINE_ROUNDS + 1):
        improved = False
        for s1 in linspace(left1, right1, points):
            for s2 in linspace(left2, right2, points):
                iterations += 1
                c = cmax_after_speed(feature, s1, s2)
                if c > deadline + 1e-7:
                    continue
                e = energy_after_speed(feature, s1, s2, lambda_single, lambda_dual, rho_single, rho_dual)
                if e < best_energy:
                    best_energy = e
                    best_cmax = c
                    best_s1 = s1
                    best_s2 = s2
                    improved = True

        if best_s1 is None or best_s2 is None:
            return None, None, None, None, iterations

        width1 = (right1 - left1) / max(points - 1, 1)
        width2 = (right2 - left2) / max(points - 1, 1)
        left1 = max(speed_min, best_s1 - 2.0 * width1)
        right1 = min(speed_max, best_s1 + 2.0 * width1)
        left2 = max(speed_min, best_s2 - 2.0 * width2)
        right2 = min(speed_max, best_s2 + 2.0 * width2)
        points = GRID_REFINE_POINTS

        if not improved and round_id > 1:
            break

    return best_s1, best_s2, best_energy, best_cmax, iterations


def active_constraint_text(
    feature: ModeFeature,
    deadline: float,
    speed_single: float,
    speed_dual: float,
    speed_min: float,
    speed_max: float,
) -> str:
    g = constraint_values(feature, deadline, speed_single, speed_dual, speed_min, speed_max)
    names = []
    tol_time = max(1e-5, 1e-5 * max(1.0, deadline))
    if abs(g["single_lower"]) <= 1e-5:
        names.append("single_lower_speed")
    if abs(g["single_upper"]) <= 1e-5:
        names.append("single_upper_speed")
    if abs(g["dual_lower"]) <= 1e-5:
        names.append("dual_lower_speed")
    if abs(g["dual_upper"]) <= 1e-5:
        names.append("dual_upper_speed")
    if abs(g["deadline"]) <= tol_time:
        names.append("deadline")
    return "+".join(names) if names else "none"


def solve_speed_problem(
    feature: ModeFeature,
    deadline: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> SpeedOptResult:
    if feature.base_cmax <= 0 or feature.base_energy <= 0 or deadline <= 0:
        return SpeedOptResult(False, feature, deadline, float("inf"), None, None, None, None, "", 0, "invalid input")

    required_speed_if_equal = (feature.single_time + feature.dual_time) / max(deadline - feature.fixed_time, EPS)

    s1, s2, energy, cmax, iterations = two_dimensional_grid_refine(
        feature, deadline, speed_min, speed_max,
        lambda_single, lambda_dual, rho_single, rho_dual,
    )

    if s1 is None or s2 is None:
        return SpeedOptResult(
            False, feature, deadline, required_speed_if_equal,
            None, None, None, None, "", iterations,
            "deadline cannot be met even at speed_max for both speed variables",
        )

    active = active_constraint_text(feature, deadline, s1, s2, speed_min, speed_max)
    return SpeedOptResult(
        True, feature, deadline, required_speed_if_equal,
        s1, s2, energy, cmax, active, iterations, "",
    )

# KKT 验证

def constraint_values(
    feature: ModeFeature,
    deadline: float,
    speed_single: float,
    speed_dual: float,
    speed_min: float,
    speed_max: float,
) -> dict[str, float]:
    return {
        "single_lower": speed_single - speed_min,
        "single_upper": speed_max - speed_single,
        "dual_lower": speed_dual - speed_min,
        "dual_upper": speed_max - speed_dual,
        "deadline": deadline - cmax_after_speed(feature, speed_single, speed_dual),
    }


def solve_nonnegative_stationarity(
    f_grad: tuple[float, float],
    active_grads: list[tuple[str, tuple[float, float]]],
) -> tuple[dict[str, float], tuple[float, float]]:
    """枚举活跃约束子集，寻找非负乘子使 ||grad f - sum mu grad g|| 最小。"""
    best_mu: dict[str, float] = {name: 0.0 for name, _g in active_grads}
    best_residual = f_grad
    best_norm = math.hypot(*best_residual)

    candidates: list[tuple[dict[str, float], tuple[float, float]]] = []
    candidates.append(({name: 0.0 for name, _g in active_grads}, f_grad))

    for name, g in active_grads:
        denom = g[0] * g[0] + g[1] * g[1]
        if denom <= EPS:
            continue
        mu = (f_grad[0] * g[0] + f_grad[1] * g[1]) / denom
        if mu >= -1e-10:
            mu = max(0.0, mu)
            residual = (f_grad[0] - mu * g[0], f_grad[1] - mu * g[1])
            candidates.append(({name: mu}, residual))

    for (name1, g1), (name2, g2) in itertools.combinations(active_grads, 2):
        det = g1[0] * g2[1] - g2[0] * g1[1]
        if abs(det) <= EPS:
            continue
        # [g1 g2] [mu1,mu2]^T = f_grad
        mu1 = (f_grad[0] * g2[1] - g2[0] * f_grad[1]) / det
        mu2 = (g1[0] * f_grad[1] - f_grad[0] * g1[1]) / det
        if mu1 >= -1e-10 and mu2 >= -1e-10:
            mu1 = max(0.0, mu1)
            mu2 = max(0.0, mu2)
            residual = (
                f_grad[0] - mu1 * g1[0] - mu2 * g2[0],
                f_grad[1] - mu1 * g1[1] - mu2 * g2[1],
            )
            candidates.append(({name1: mu1, name2: mu2}, residual))

    for partial_mu, residual in candidates:
        norm = math.hypot(*residual)
        if norm < best_norm:
            best_norm = norm
            best_residual = residual
            best_mu = {name: 0.0 for name, _g in active_grads}
            best_mu.update(partial_mu)

    return best_mu, best_residual


def verify_kkt(
    result: SpeedOptResult,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> KKTCheck:
    if not result.feasible or result.opt_speed_single is None or result.opt_speed_dual is None:
        return KKTCheck(False, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, False, result.infeasible_reason)

    f = result.feature
    s1 = result.opt_speed_single
    s2 = result.opt_speed_dual
    d = result.deadline

    gvals = constraint_values(f, d, s1, s2, speed_min, speed_max)
    grad_f = (
        economic_energy_derivative(f.single_energy, s1, lambda_single, rho_single),
        economic_energy_derivative(f.dual_energy, s2, lambda_dual, rho_dual),
    )

    tol_time = max(1e-4, 1e-4 * max(1.0, d))
    active_grads: list[tuple[str, tuple[float, float]]] = []
    if abs(gvals["single_lower"]) <= 1e-5:
        active_grads.append(("single_lower", (1.0, 0.0)))
    if abs(gvals["single_upper"]) <= 1e-5:
        active_grads.append(("single_upper", (-1.0, 0.0)))
    if abs(gvals["dual_lower"]) <= 1e-5:
        active_grads.append(("dual_lower", (0.0, 1.0)))
    if abs(gvals["dual_upper"]) <= 1e-5:
        active_grads.append(("dual_upper", (0.0, -1.0)))
    if abs(gvals["deadline"]) <= tol_time:
        active_grads.append(("deadline", (f.single_time / (s1 * s1), f.dual_time / (s2 * s2))))

    mu_active, residual = solve_nonnegative_stationarity(grad_f, active_grads)
    mu = {"single_lower": 0.0, "single_upper": 0.0, "dual_lower": 0.0, "dual_upper": 0.0, "deadline": 0.0}
    mu.update(mu_active)

    comp = [abs(mu[name] * gvals[name]) for name in mu]
    max_comp = max(comp) if comp else 0.0
    primal_min = min(gvals.values())
    dual_min = min(mu.values())
    scale = max(1.0, math.hypot(*grad_f))
    res_norm = math.hypot(*residual)

    # 网格法是数值求解，允许较小的相对残差；该字段用于报告“数值上满足/近似满足”。
    kkt_pass = (
        primal_min >= -KKT_TOL * max(1.0, d)
        and dual_min >= -KKT_TOL
        and max_comp <= KKT_TOL * max(1.0, abs(result.opt_energy or 1.0))
        and res_norm <= 0.08 * scale
    )
    note = "KKT conditions numerically satisfied for two-variable nonlinear model" if kkt_pass else "KKT residual is reported for numerical inspection"

    return KKTCheck(
        True,
        gvals["single_lower"], gvals["single_upper"],
        gvals["dual_lower"], gvals["dual_upper"], gvals["deadline"],
        mu["single_lower"], mu["single_upper"], mu["dual_lower"], mu["dual_upper"], mu["deadline"],
        residual[0], residual[1], max_comp, primal_min, dual_min, kkt_pass, note,
    )

# 结果生成

def compare_2b_3b(opt_2b: SpeedOptResult, opt_3b: SpeedOptResult) -> tuple[str, str, str, float | None]:
    if not opt_2b.feasible and not opt_3b.feasible:
        return "none", "infeasible_both", "neither 2B nor 3B can meet the deadline", None
    if opt_2b.feasible and not opt_3b.feasible:
        return "2B", "recommend_2arm_only_feasible", "only 2B is feasible", None
    if opt_3b.feasible and not opt_2b.feasible:
        return "3B", "recommend_3arm_only_feasible", "only 3B is feasible", None
    assert opt_2b.opt_energy is not None and opt_3b.opt_energy is not None
    advantage = opt_2b.opt_energy - opt_3b.opt_energy
    if advantage > EPS:
        return "3B", "recommend_3arm_lower_energy_under_deadline", "3B has lower optimized energy under the same deadline", advantage
    if advantage < -EPS:
        return "2B", "recommend_2arm_lower_energy_under_deadline", "2B has lower optimized energy under the same deadline", advantage
    return "2B", "similar_prefer_2arm", "optimized energies are nearly equal, prefer simpler 2B", advantage


def build_output_rows(
    averaged_cases: list[AveragedCase],
    method: str,
    deadline_ratios: list[float],
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> tuple[list[dict], list[dict]]:
    result_rows: list[dict] = []
    kkt_rows: list[dict] = []

    for case in averaged_cases:
        scen_type = scenario_type(case.n1, case.n2, case.n3, case.n4)
        feature_2b = build_mode_feature(case.mean_cmax_2b, case.mean_energy_2b, case.n1, case.n2, case.n3, case.n4, "2B")
        feature_3b = build_mode_feature(case.mean_cmax_3b, case.mean_energy_3b, case.n1, case.n2, case.n3, case.n4, "3B")

        for ratio in deadline_ratios:
            deadline = ratio * case.mean_cmax_2b
            opt_2b = solve_speed_problem(feature_2b, deadline, speed_min, speed_max, lambda_single, lambda_dual, rho_single, rho_dual)
            opt_3b = solve_speed_problem(feature_3b, deadline, speed_min, speed_max, lambda_single, lambda_dual, rho_single, rho_dual)
            kkt_2b = verify_kkt(opt_2b, speed_min, speed_max, lambda_single, lambda_dual, rho_single, rho_dual)
            kkt_3b = verify_kkt(opt_3b, speed_min, speed_max, lambda_single, lambda_dual, rho_single, rho_dual)
            recommended_mode, recommendation, reason, advantage = compare_2b_3b(opt_2b, opt_3b)

            common = {
                "counts_code": case.counts_code,
                "n1": case.n1, "n2": case.n2, "n3": case.n3, "n4": case.n4,
                "total_tasks": case.total_tasks,
                "scenario_type": scen_type,
                "seed_count": case.seed_count,
                "seeds_used": case.seeds_used,
                "method": method,
                "deadline_ratio": round_or_blank(ratio, 4),
                "deadline_value": round_or_blank(deadline, 6),
                "lambda_single": round_or_blank(lambda_single, 4),
                "lambda_dual": round_or_blank(lambda_dual, 4),
                "rho_single": round_or_blank(rho_single, 4),
                "rho_dual": round_or_blank(rho_dual, 4),
                "speed_min": round_or_blank(speed_min, 4),
                "speed_max": round_or_blank(speed_max, 4),
            }

            row = {
                **common,
                "mean_cmax_2B_original": round_or_blank(case.mean_cmax_2b, 6),
                "mean_energy_2B_original": round_or_blank(case.mean_energy_2b, 6),
                "std_cmax_2B_original": round_or_blank(case.std_cmax_2b, 6),
                "std_energy_2B_original": round_or_blank(case.std_energy_2b, 6),
                "mean_single_time_2B": round_or_blank(feature_2b.single_time, 6),
                "mean_dual_time_2B": round_or_blank(feature_2b.dual_time, 6),
                "mean_fixed_time_2B": round_or_blank(feature_2b.fixed_time, 6),
                "mean_single_energy_2B": round_or_blank(feature_2b.single_energy, 6),
                "mean_dual_energy_2B": round_or_blank(feature_2b.dual_energy, 6),
                "mean_fixed_energy_2B": round_or_blank(feature_2b.fixed_energy, 6),
                "feasible_2B": int(opt_2b.feasible),
                "required_speed_if_equal_2B": round_or_blank(opt_2b.required_speed_if_equal, 6),
                "opt_speed_single_2B": round_or_blank(opt_2b.opt_speed_single, 6),
                "opt_speed_dual_2B": round_or_blank(opt_2b.opt_speed_dual, 6),
                "opt_energy_2B": round_or_blank(opt_2b.opt_energy, 6),
                "opt_cmax_2B": round_or_blank(opt_2b.opt_cmax, 6),
                "active_constraints_2B": opt_2b.active_constraints,
                "search_iterations_2B": opt_2b.search_iterations,
                "infeasible_reason_2B": opt_2b.infeasible_reason,
                "kkt_pass_2B": int(kkt_2b.kkt_pass),

                "mean_cmax_3B_original": round_or_blank(case.mean_cmax_3b, 6),
                "mean_energy_3B_original": round_or_blank(case.mean_energy_3b, 6),
                "std_cmax_3B_original": round_or_blank(case.std_cmax_3b, 6),
                "std_energy_3B_original": round_or_blank(case.std_energy_3b, 6),
                "mean_single_time_3B": round_or_blank(feature_3b.single_time, 6),
                "mean_dual_time_3B": round_or_blank(feature_3b.dual_time, 6),
                "mean_fixed_time_3B": round_or_blank(feature_3b.fixed_time, 6),
                "mean_single_energy_3B": round_or_blank(feature_3b.single_energy, 6),
                "mean_dual_energy_3B": round_or_blank(feature_3b.dual_energy, 6),
                "mean_fixed_energy_3B": round_or_blank(feature_3b.fixed_energy, 6),
                "feasible_3B": int(opt_3b.feasible),
                "required_speed_if_equal_3B": round_or_blank(opt_3b.required_speed_if_equal, 6),
                "opt_speed_single_3B": round_or_blank(opt_3b.opt_speed_single, 6),
                "opt_speed_dual_3B": round_or_blank(opt_3b.opt_speed_dual, 6),
                "opt_energy_3B": round_or_blank(opt_3b.opt_energy, 6),
                "opt_cmax_3B": round_or_blank(opt_3b.opt_cmax, 6),
                "active_constraints_3B": opt_3b.active_constraints,
                "search_iterations_3B": opt_3b.search_iterations,
                "infeasible_reason_3B": opt_3b.infeasible_reason,
                "kkt_pass_3B": int(kkt_3b.kkt_pass),

                "energy_advantage_2B_minus_3B": round_or_blank(advantage, 6),
                "recommended_mode": recommended_mode,
                "recommendation": recommendation,
                "recommendation_reason": reason,
                "model_formula": "min E_fixed+E_single(s_single)+E_dual(s_dual), s.t. C_fixed+C_single/s_single+C_dual/s_dual<=D",
                "energy_formula": "E_k(s)=E_k0*(lambda_k/s+(1-lambda_k)*((1-rho_k)+rho_k*s^2))",
                "course_method": "two_variable_nonlinear_search + KKT_verification + sensitivity_analysis",
            }
            result_rows.append(row)

            for mode_name, opt, kkt in [("2B", opt_2b, kkt_2b), ("3B", opt_3b, kkt_3b)]:
                kkt_rows.append({
                    **common,
                    "mode": mode_name,
                    "base_cmax": round_or_blank(opt.feature.base_cmax, 6),
                    "base_energy": round_or_blank(opt.feature.base_energy, 6),
                    "single_time": round_or_blank(opt.feature.single_time, 6),
                    "dual_time": round_or_blank(opt.feature.dual_time, 6),
                    "fixed_time": round_or_blank(opt.feature.fixed_time, 6),
                    "single_energy": round_or_blank(opt.feature.single_energy, 6),
                    "dual_energy": round_or_blank(opt.feature.dual_energy, 6),
                    "fixed_energy": round_or_blank(opt.feature.fixed_energy, 6),
                    "feasible": int(kkt.feasible),
                    "opt_speed_single": round_or_blank(opt.opt_speed_single, 6),
                    "opt_speed_dual": round_or_blank(opt.opt_speed_dual, 6),
                    "opt_energy": round_or_blank(opt.opt_energy, 6),
                    "opt_cmax": round_or_blank(opt.opt_cmax, 6),
                    "active_constraints": opt.active_constraints,
                    "g_single_lower": round_or_blank(kkt.g_single_lower, 10),
                    "g_single_upper": round_or_blank(kkt.g_single_upper, 10),
                    "g_dual_lower": round_or_blank(kkt.g_dual_lower, 10),
                    "g_dual_upper": round_or_blank(kkt.g_dual_upper, 10),
                    "g_deadline": round_or_blank(kkt.g_deadline, 10),
                    "mu_single_lower": round_or_blank(kkt.mu_single_lower, 10),
                    "mu_single_upper": round_or_blank(kkt.mu_single_upper, 10),
                    "mu_dual_lower": round_or_blank(kkt.mu_dual_lower, 10),
                    "mu_dual_upper": round_or_blank(kkt.mu_dual_upper, 10),
                    "mu_deadline": round_or_blank(kkt.mu_deadline, 10),
                    "stationarity_single_residual": round_or_blank(kkt.stationarity_single_residual, 10),
                    "stationarity_dual_residual": round_or_blank(kkt.stationarity_dual_residual, 10),
                    "max_complementarity_error": round_or_blank(kkt.max_complementarity_error, 10),
                    "primal_min": round_or_blank(kkt.primal_min, 10),
                    "dual_min": round_or_blank(kkt.dual_min, 10),
                    "kkt_pass": int(kkt.kkt_pass),
                    "kkt_note": kkt.kkt_note,
                })
    return result_rows, kkt_rows


def build_sensitivity_rows(
    averaged_cases: list[AveragedCase],
    method: str,
    base_deadline_ratios: list[float],
    speed_min: float,
    speed_max: float,
) -> list[dict]:
    """小型敏感性分析：围绕 U 形能耗参数改变 lambda/rho，观察推荐模式变化。"""
    parameter_sets = [
        ("low_slow_penalty", 0.25, 0.35, 0.35, 0.45),
        ("baseline", DEFAULT_LAMBDA_SINGLE, DEFAULT_LAMBDA_DUAL, DEFAULT_RHO_SINGLE, DEFAULT_RHO_DUAL),
        ("high_slow_penalty", 0.45, 0.55, 0.35, 0.45),
        ("high_fast_penalty", 0.35, 0.45, 0.50, 0.60),
    ]
    selected_ratios = [r for r in base_deadline_ratios if r in {0.85, 1.00, 1.20}]
    if not selected_ratios:
        selected_ratios = base_deadline_ratios[:3]

    rows: list[dict] = []
    for name, lam_s, lam_d, rho_s, rho_d in parameter_sets:
        for ratio in selected_ratios:
            total = feasible_total = recommend_3b = recommend_2b = infeasible = 0
            gaps: list[float] = []
            for case in averaged_cases:
                deadline = ratio * case.mean_cmax_2b
                f2 = build_mode_feature(case.mean_cmax_2b, case.mean_energy_2b, case.n1, case.n2, case.n3, case.n4, "2B")
                f3 = build_mode_feature(case.mean_cmax_3b, case.mean_energy_3b, case.n1, case.n2, case.n3, case.n4, "3B")
                o2 = solve_speed_problem(f2, deadline, speed_min, speed_max, lam_s, lam_d, rho_s, rho_d)
                o3 = solve_speed_problem(f3, deadline, speed_min, speed_max, lam_s, lam_d, rho_s, rho_d)
                rec, _tag, _reason, gap = compare_2b_3b(o2, o3)
                total += 1
                if rec == "3B":
                    recommend_3b += 1
                elif rec == "2B":
                    recommend_2b += 1
                else:
                    infeasible += 1
                if o2.feasible or o3.feasible:
                    feasible_total += 1
                if gap is not None:
                    gaps.append(gap)
            rows.append({
                "parameter_set": name,
                "method": method,
                "deadline_ratio": ratio,
                "lambda_single": lam_s,
                "lambda_dual": lam_d,
                "rho_single": rho_s,
                "rho_dual": rho_d,
                "case_count": total,
                "feasible_case_count": feasible_total,
                "recommend_3B_count": recommend_3b,
                "recommend_2B_count": recommend_2b,
                "infeasible_count": infeasible,
                "recommend_3B_ratio_percent": round(recommend_3b / total * 100.0, 4) if total else 0.0,
                "mean_energy_advantage_2B_minus_3B": round_or_blank(mean(gaps), 6),
                "sensitivity_meaning": "small sensitivity analysis for U-shaped speed-energy parameters",
            })
    return rows

# 主函数

def main() -> None:
    parser = argparse.ArgumentParser(description="Two-variable U-shaped nonlinear speed-energy optimization with KKT and sensitivity analysis.")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--input", default="")
    parser.add_argument("--deadline-ratios", default=",".join(str(x) for x in DEFAULT_DEADLINE_RATIOS))
    parser.add_argument("--speed-min", type=float, default=DEFAULT_SPEED_MIN)
    parser.add_argument("--speed-max", type=float, default=DEFAULT_SPEED_MAX)
    parser.add_argument("--lambda-single", type=float, default=DEFAULT_LAMBDA_SINGLE)
    parser.add_argument("--lambda-dual", type=float, default=DEFAULT_LAMBDA_DUAL)
    parser.add_argument("--rho-single", type=float, default=DEFAULT_RHO_SINGLE)
    parser.add_argument("--rho-dual", type=float, default=DEFAULT_RHO_DUAL)
    args = parser.parse_args()

    method = args.method.strip()
    input_path = Path(args.input) if args.input.strip() else OUTPUT_DIR / "four_case_framework" / f"{method}_2B3B_time_energy.csv"
    output_dir = OUTPUT_DIR / "nonlinear_programming" / "speed_energy_optimization"
    result_path = output_dir / "speed_energy_optimization.csv"
    kkt_path = output_dir / "kkt_verification.csv"
    sensitivity_path = output_dir / "sensitivity_analysis.csv"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)
    deadline_ratios = parse_float_list(args.deadline_ratios)
    averaged_cases = build_averaged_cases(raw_rows, method)

    result_rows, kkt_rows = build_output_rows(
        averaged_cases, method, deadline_ratios,
        args.speed_min, args.speed_max,
        args.lambda_single, args.lambda_dual, args.rho_single, args.rho_dual,
    )
    sensitivity_rows = build_sensitivity_rows(averaged_cases, method, deadline_ratios, args.speed_min, args.speed_max)

    result_fields = [
        "counts_code", "n1", "n2", "n3", "n4", "total_tasks", "scenario_type", "seed_count", "seeds_used", "method",
        "deadline_ratio", "deadline_value", "lambda_single", "lambda_dual", "rho_single", "rho_dual", "speed_min", "speed_max",
        "mean_cmax_2B_original", "mean_energy_2B_original", "std_cmax_2B_original", "std_energy_2B_original",
        "mean_single_time_2B", "mean_dual_time_2B", "mean_fixed_time_2B", "mean_single_energy_2B", "mean_dual_energy_2B", "mean_fixed_energy_2B",
        "feasible_2B", "required_speed_if_equal_2B", "opt_speed_single_2B", "opt_speed_dual_2B", "opt_energy_2B", "opt_cmax_2B", "active_constraints_2B", "search_iterations_2B", "infeasible_reason_2B", "kkt_pass_2B",
        "mean_cmax_3B_original", "mean_energy_3B_original", "std_cmax_3B_original", "std_energy_3B_original",
        "mean_single_time_3B", "mean_dual_time_3B", "mean_fixed_time_3B", "mean_single_energy_3B", "mean_dual_energy_3B", "mean_fixed_energy_3B",
        "feasible_3B", "required_speed_if_equal_3B", "opt_speed_single_3B", "opt_speed_dual_3B", "opt_energy_3B", "opt_cmax_3B", "active_constraints_3B", "search_iterations_3B", "infeasible_reason_3B", "kkt_pass_3B",
        "energy_advantage_2B_minus_3B", "recommended_mode", "recommendation", "recommendation_reason", "model_formula", "energy_formula", "course_method",
    ]
    kkt_fields = [
        "counts_code", "n1", "n2", "n3", "n4", "total_tasks", "scenario_type", "seed_count", "seeds_used", "method", "deadline_ratio", "deadline_value",
        "lambda_single", "lambda_dual", "rho_single", "rho_dual", "speed_min", "speed_max", "mode",
        "base_cmax", "base_energy", "single_time", "dual_time", "fixed_time", "single_energy", "dual_energy", "fixed_energy",
        "feasible", "opt_speed_single", "opt_speed_dual", "opt_energy", "opt_cmax", "active_constraints",
        "g_single_lower", "g_single_upper", "g_dual_lower", "g_dual_upper", "g_deadline",
        "mu_single_lower", "mu_single_upper", "mu_dual_lower", "mu_dual_upper", "mu_deadline",
        "stationarity_single_residual", "stationarity_dual_residual", "max_complementarity_error", "primal_min", "dual_min", "kkt_pass", "kkt_note",
    ]
    sensitivity_fields = [
        "parameter_set", "method", "deadline_ratio", "lambda_single", "lambda_dual", "rho_single", "rho_dual",
        "case_count", "feasible_case_count", "recommend_3B_count", "recommend_2B_count", "infeasible_count",
        "recommend_3B_ratio_percent", "mean_energy_advantage_2B_minus_3B", "sensitivity_meaning",
    ]

    save_csv(result_rows, result_path, result_fields)
    save_csv(kkt_rows, kkt_path, kkt_fields)
    save_csv(sensitivity_rows, sensitivity_path, sensitivity_fields)

    print("Two-variable nonlinear speed-energy optimization finished.")
    print(f"Input: {input_path}")
    print(f"Output 1: {result_path}")
    print(f"Output 2: {kkt_path}")
    print(f"Output 3: {sensitivity_path}")
    print(f"Averaged cases: {len(averaged_cases)}")
    print(f"Optimization rows: {len(result_rows)}")
    print(f"KKT rows: {len(kkt_rows)}")
    print(f"Sensitivity rows: {len(sensitivity_rows)}")
    print("Model: min E_fixed+E_single(s_single)+E_dual(s_dual), with U-shaped economic speed energy.")


if __name__ == "__main__":
    main()
