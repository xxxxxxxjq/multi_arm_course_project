# -*- coding: utf-8 -*-
"""非线性规划

只读取已有 2B/3B 完工时间与能耗结果

1. 新建模：
       s_single：单臂任务速度倍率
       s_dual  ：双臂协同任务速度倍率

2. 能耗函数保留 U 形经济速度模型：
       E_k(s_k)=E_k0*[lambda_k/s_k + (1-lambda_k)*((1-rho_k)+rho_k*s_k^2)]

3. 约束模型：
       min  E_fixed + E_single(s_single) + E_dual(s_dual)

       s.t. C_fixed + C_single/s_single + C_dual/s_dual <= D
            s_min <= s_single <= s_max
            s_min <= s_dual   <= s_max

4. 求解方法：
       有约束问题
       -> 构造内点障碍函数
       -> 每个障碍因子 mu 下，用最速下降方向迭代
       -> 每次沿下降方向用 0.618 黄金分割法求一维最优步长
       -> mu 逐步减小，逼近原约束问题解
       -> 单独输出 KKT 验证表

5. 敏感性分析：
       不再单独输出 sensitivity_analysis.csv；
       而是把不同参数组作为 sensitivity_case 合并到主结果 speed_energy_optimization.csv。

默认输入：
       outputs/four_case_framework/optimized_2B3B_time_energy.csv

默认输出：
       outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization.csv
       outputs/nonlinear_programming/speed_energy_optimization/kkt_verification.csv
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
from typing import Any, Callable


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import OUTPUT_DIR  # noqa: E402
from common.utils import ensure_dirs  # noqa: E402


# ============================================================
# 默认参数
# ============================================================

DEFAULT_METHOD = "optimized"

DEFAULT_DEADLINE_RATIOS = [
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.00, 1.05, 1.10, 1.15, 1.20, 1.25,
    1.30, 1.35, 1.40, 1.50,
]

DEFAULT_SPEED_MIN = 0.60
DEFAULT_SPEED_MAX = 1.80

# U 形经济速度能耗参数。
DEFAULT_LAMBDA_SINGLE = 0.35
DEFAULT_LAMBDA_DUAL = 0.45
DEFAULT_RHO_SINGLE = 0.35
DEFAULT_RHO_DUAL = 0.45

# 0.618 黄金分割法参数。
GOLDEN_RATIO = 0.6180339887498949
GOLDEN_TOL = 1e-8
GOLDEN_MAX_ITER = 120

# 内点制约函数参数。
# mu 越小，障碍函数解越接近原约束问题边界解。
BARRIER_MU_VALUES = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

# 每个 mu 下的最速下降迭代参数。
DESCENT_MAX_ITER = 120
GRAD_TOL = 1e-6
STEP_TOL = 1e-9

EPS = 1e-10
KKT_TOL = 5e-3


# ============================================================
# 数据结构
# ============================================================

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

    barrier_objective_value: float | None
    barrier_outer_iterations: int
    descent_iterations: int
    golden_iterations: int

    active_constraints: str
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
    stationarity_norm: float | None

    max_complementarity_error: float | None
    primal_min: float | None
    dual_min: float | None

    kkt_pass: bool
    kkt_note: str


# ============================================================
# 基础工具函数
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
# seed 平均
# ============================================================

def build_averaged_cases(raw_rows: list[dict], method: str) -> list[AveragedCase]:
    """按 counts_code 分组，对 seed 结果取平均。"""
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

        key = (counts_code, n1, n2, n3, n4, total_tasks)
        groups[key].append(row)

    averaged_cases: list[AveragedCase] = []

    for key, rows in groups.items():
        counts_code, n1, n2, n3, n4, total_tasks = key

        seeds = [safe_int(r.get("seed")) for r in rows]
        seeds_used = ",".join(str(x) for x in sorted(seeds))

        cmax_2b_values = [safe_float(r.get(cols["cmax_2b"])) for r in rows]
        energy_2b_values = [safe_float(r.get(cols["energy_2b"])) for r in rows]
        cmax_3b_values = [safe_float(r.get(cols["cmax_3b"])) for r in rows]
        energy_3b_values = [safe_float(r.get(cols["energy_3b"])) for r in rows]

        averaged_cases.append(
            AveragedCase(
                counts_code=counts_code,
                n1=n1,
                n2=n2,
                n3=n3,
                n4=n4,
                total_tasks=total_tasks,
                seed_count=len(rows),
                seeds_used=seeds_used,

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


# ============================================================
# 任务类型特征拆分：只用主结果，不改主调度
# ============================================================

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

    严格原则：
    1. 不重新设定启动成本、固定能耗、服务时间或物块能耗参数；
    2. 不调用求解器重新计算调度；
    3. 只把已有 base_cmax/base_energy 按任务结构拆分为单臂任务和双臂任务两类。

    s_single=s_dual=1 时：
        C_single + C_dual + C_fixed = base_cmax
        E_single + E_dual + E_fixed = base_energy

    因此速度优化层严格锚定已有主问题结果。
    """
    _ = mode  # 保留 mode 参数，便于以后扩展 2B/3B 差异。
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

        single_task_share=single_share,
        dual_task_share=dual_share,
    )


# ============================================================
# 双变量 U 形速度-能耗模型
# ============================================================

def economic_energy_component(
    base_energy: float,
    speed: float,
    lambda_keep: float,
    rho: float,
) -> float:
    """U 形经济速度能耗分量。"""
    return base_energy * (
        lambda_keep / speed
        + (1.0 - lambda_keep) * ((1.0 - rho) + rho * speed * speed)
    )


def economic_energy_derivative(
    base_energy: float,
    speed: float,
    lambda_keep: float,
    rho: float,
) -> float:
    """U 形能耗分量的一阶导数。"""
    return base_energy * (
        -lambda_keep / (speed * speed)
        + 2.0 * (1.0 - lambda_keep) * rho * speed
    )


def economic_speed_unconstrained(
    lambda_keep: float,
    rho: float,
    speed_min: float,
    speed_max: float,
) -> float:
    """单个 U 形能耗项的无约束经济速度。"""
    denom = 2.0 * (1.0 - lambda_keep) * rho
    if denom <= 0:
        return speed_min
    s = (lambda_keep / denom) ** (1.0 / 3.0)
    return min(speed_max, max(speed_min, s))


def cmax_after_speed(
    feature: ModeFeature,
    speed_single: float,
    speed_dual: float,
) -> float:
    return (
        feature.fixed_time
        + feature.single_time / speed_single
        + feature.dual_time / speed_dual
    )


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


def energy_gradient(
    feature: ModeFeature,
    speed_single: float,
    speed_dual: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> tuple[float, float]:
    return (
        economic_energy_derivative(feature.single_energy, speed_single, lambda_single, rho_single),
        economic_energy_derivative(feature.dual_energy, speed_dual, lambda_dual, rho_dual),
    )


# ============================================================
# 约束、障碍函数和梯度
# ============================================================

def constraint_values(
    feature: ModeFeature,
    deadline: float,
    speed_single: float,
    speed_dual: float,
    speed_min: float,
    speed_max: float,
) -> dict[str, float]:
    """所有约束统一写成 g_i(x)>=0。"""
    return {
        "single_lower": speed_single - speed_min,
        "single_upper": speed_max - speed_single,
        "dual_lower": speed_dual - speed_min,
        "dual_upper": speed_max - speed_dual,
        "deadline": deadline - cmax_after_speed(feature, speed_single, speed_dual),
    }


def constraint_gradients(
    feature: ModeFeature,
    speed_single: float,
    speed_dual: float,
) -> dict[str, tuple[float, float]]:
    return {
        "single_lower": (1.0, 0.0),
        "single_upper": (-1.0, 0.0),
        "dual_lower": (0.0, 1.0),
        "dual_upper": (0.0, -1.0),
        "deadline": (
            feature.single_time / (speed_single * speed_single),
            feature.dual_time / (speed_dual * speed_dual),
        ),
    }


def is_strictly_feasible(
    feature: ModeFeature,
    deadline: float,
    speed_single: float,
    speed_dual: float,
    speed_min: float,
    speed_max: float,
    margin: float = 1e-10,
) -> bool:
    g = constraint_values(feature, deadline, speed_single, speed_dual, speed_min, speed_max)
    return min(g.values()) > margin


def is_feasible(
    feature: ModeFeature,
    deadline: float,
    speed_single: float,
    speed_dual: float,
    speed_min: float,
    speed_max: float,
    tol: float = 1e-8,
) -> bool:
    g = constraint_values(feature, deadline, speed_single, speed_dual, speed_min, speed_max)
    return min(g.values()) >= -tol


def barrier_objective(
    feature: ModeFeature,
    deadline: float,
    speed_single: float,
    speed_dual: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
    mu: float,
) -> float:
    """内点对数障碍函数 B(x,mu)=F(x)-mu*sum(log(g_i(x)))。"""
    g = constraint_values(feature, deadline, speed_single, speed_dual, speed_min, speed_max)

    if min(g.values()) <= 0:
        return float("inf")

    original = energy_after_speed(
        feature, speed_single, speed_dual,
        lambda_single, lambda_dual, rho_single, rho_dual,
    )
    barrier = -mu * sum(math.log(v) for v in g.values())

    return original + barrier


def barrier_gradient(
    feature: ModeFeature,
    deadline: float,
    speed_single: float,
    speed_dual: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
    mu: float,
) -> tuple[float, float]:
    """障碍函数梯度。"""
    grad_f = energy_gradient(
        feature, speed_single, speed_dual,
        lambda_single, lambda_dual, rho_single, rho_dual,
    )
    g = constraint_values(feature, deadline, speed_single, speed_dual, speed_min, speed_max)
    grad_g = constraint_gradients(feature, speed_single, speed_dual)

    grad_1 = grad_f[0]
    grad_2 = grad_f[1]

    for name, gv in g.items():
        if gv <= 0:
            return float("inf"), float("inf")
        gg = grad_g[name]
        grad_1 -= mu * gg[0] / gv
        grad_2 -= mu * gg[1] / gv

    return grad_1, grad_2


# ============================================================
# 0.618 黄金分割法
# ============================================================

def golden_section_minimize(
    func: Callable[[float], float],
    left: float,
    right: float,
    tol: float = GOLDEN_TOL,
    max_iter: int = GOLDEN_MAX_ITER,
) -> tuple[float, float, int]:
    """用 0.618 黄金分割法求一维函数在 [left,right] 上的极小值。"""
    if not left < right:
        return left, func(left), 0

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

    x_star = 0.5 * (a + b)
    return x_star, func(x_star), iterations


# ============================================================
# 内点法 + 最速下降 + 0.618 一维搜索
# ============================================================

def find_strict_feasible_start(
    feature: ModeFeature,
    deadline: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> tuple[float | None, float | None, str]:
    """寻找严格内点初始解。

    优先使用无约束经济速度；若不能满足截止时间，则逐步向 speed_max 靠近。
    """
    min_time = cmax_after_speed(feature, speed_max, speed_max)
    if min_time > deadline + 1e-9:
        return None, None, "deadline cannot be met even at speed_max for both speed variables"

    width = speed_max - speed_min
    margin = max(1e-8, 1e-8 * width)
    low = speed_min + margin
    high = speed_max - margin

    if not low < high:
        return None, None, "speed interval has no strict interior"

    s1_econ = economic_speed_unconstrained(lambda_single, rho_single, speed_min, speed_max)
    s2_econ = economic_speed_unconstrained(lambda_dual, rho_dual, speed_min, speed_max)

    s1_econ = min(high, max(low, s1_econ))
    s2_econ = min(high, max(low, s2_econ))

    if is_strictly_feasible(feature, deadline, s1_econ, s2_econ, speed_min, speed_max):
        return s1_econ, s2_econ, ""

    # 等速估计：C/s <= D。
    denominator = max(deadline - feature.fixed_time, EPS)
    equal_required = (feature.single_time + feature.dual_time) / denominator
    base_s = min(high, max(low, equal_required * 1.03))

    if is_strictly_feasible(feature, deadline, base_s, base_s, speed_min, speed_max):
        return base_s, base_s, ""

    # 从经济速度逐步向接近 speed_max 的点移动。
    for alpha_i in range(1, 101):
        alpha = alpha_i / 100.0
        s1 = (1.0 - alpha) * s1_econ + alpha * high
        s2 = (1.0 - alpha) * s2_econ + alpha * high

        if is_strictly_feasible(feature, deadline, s1, s2, speed_min, speed_max):
            return s1, s2, ""

    # 再试等速点向 high 移动。
    for alpha_i in range(1, 101):
        alpha = alpha_i / 100.0
        s = (1.0 - alpha) * base_s + alpha * high

        if is_strictly_feasible(feature, deadline, s, s, speed_min, speed_max):
            return s, s, ""

    return None, None, "no strictly feasible interior point found for barrier method"


def max_feasible_step_along_direction(
    feature: ModeFeature,
    deadline: float,
    x: tuple[float, float],
    p: tuple[float, float],
    speed_min: float,
    speed_max: float,
) -> float:
    """计算沿方向 p 能保持严格可行的大致最大步长。"""
    s1, s2 = x
    p1, p2 = p

    width = speed_max - speed_min
    margin = max(1e-8, 1e-8 * width)

    alpha_max = float("inf")

    if p1 > 0:
        alpha_max = min(alpha_max, (speed_max - margin - s1) / p1)
    elif p1 < 0:
        alpha_max = min(alpha_max, (speed_min + margin - s1) / p1)

    if p2 > 0:
        alpha_max = min(alpha_max, (speed_max - margin - s2) / p2)
    elif p2 < 0:
        alpha_max = min(alpha_max, (speed_min + margin - s2) / p2)

    if not math.isfinite(alpha_max):
        alpha_max = 1.0

    alpha_max = max(0.0, 0.99 * alpha_max)

    # 截止时间约束不是线性的，直接试探缩小。
    while alpha_max > 1e-14:
        candidate_s1 = s1 + alpha_max * p1
        candidate_s2 = s2 + alpha_max * p2

        if is_strictly_feasible(
            feature, deadline, candidate_s1, candidate_s2,
            speed_min, speed_max,
            margin=1e-12,
        ):
            return alpha_max

        alpha_max *= 0.5

    return 0.0


def minimize_barrier_by_descent_and_golden(
    feature: ModeFeature,
    deadline: float,
    start: tuple[float, float],
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
    mu: float,
) -> tuple[tuple[float, float], float, int, int]:
    """在固定 mu 下，使用最速下降方向 + 0.618 法求障碍函数极小。"""
    s1, s2 = start
    descent_iter = 0
    golden_iter_total = 0

    current_value = barrier_objective(
        feature, deadline, s1, s2, speed_min, speed_max,
        lambda_single, lambda_dual, rho_single, rho_dual, mu,
    )

    for _ in range(DESCENT_MAX_ITER):
        descent_iter += 1

        grad = barrier_gradient(
            feature, deadline, s1, s2, speed_min, speed_max,
            lambda_single, lambda_dual, rho_single, rho_dual, mu,
        )

        if not math.isfinite(grad[0]) or not math.isfinite(grad[1]):
            break

        grad_norm = math.hypot(grad[0], grad[1])
        if grad_norm <= GRAD_TOL * max(1.0, abs(current_value)):
            break

        # 最速下降方向。归一化避免步长区间数值过大。
        direction = (-grad[0] / grad_norm, -grad[1] / grad_norm)

        alpha_right = max_feasible_step_along_direction(
            feature=feature,
            deadline=deadline,
            x=(s1, s2),
            p=direction,
            speed_min=speed_min,
            speed_max=speed_max,
        )

        if alpha_right <= STEP_TOL:
            break

        def phi(alpha: float) -> float:
            return barrier_objective(
                feature, deadline,
                s1 + alpha * direction[0],
                s2 + alpha * direction[1],
                speed_min, speed_max,
                lambda_single, lambda_dual, rho_single, rho_dual,
                mu,
            )

        alpha_star, phi_star, golden_iter = golden_section_minimize(
            func=phi,
            left=0.0,
            right=alpha_right,
        )
        golden_iter_total += golden_iter

        if not math.isfinite(phi_star):
            break

        new_s1 = s1 + alpha_star * direction[0]
        new_s2 = s2 + alpha_star * direction[1]

        if abs(alpha_star) <= STEP_TOL:
            break

        # 若数值上没有下降，停止当前 mu 的内层迭代。
        if phi_star > current_value + 1e-9 * max(1.0, abs(current_value)):
            break

        s1, s2 = new_s1, new_s2
        old_value = current_value
        current_value = phi_star

        if abs(old_value - current_value) <= 1e-9 * max(1.0, abs(old_value)):
            break

    return (s1, s2), current_value, descent_iter, golden_iter_total


def solve_speed_problem_by_barrier(
    feature: ModeFeature,
    deadline: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> SpeedOptResult:
    """双变量速度-能耗模型的内点法求解。"""
    if feature.base_cmax <= 0 or feature.base_energy <= 0 or deadline <= 0:
        return SpeedOptResult(
            feasible=False,
            feature=feature,
            deadline=deadline,
            required_speed_if_equal=float("inf"),
            opt_speed_single=None,
            opt_speed_dual=None,
            opt_energy=None,
            opt_cmax=None,
            barrier_objective_value=None,
            barrier_outer_iterations=0,
            descent_iterations=0,
            golden_iterations=0,
            active_constraints="",
            infeasible_reason="invalid input",
        )

    required_speed_if_equal = (
        (feature.single_time + feature.dual_time)
        / max(deadline - feature.fixed_time, EPS)
    )

    start_s1, start_s2, infeasible_reason = find_strict_feasible_start(
        feature=feature,
        deadline=deadline,
        speed_min=speed_min,
        speed_max=speed_max,
        lambda_single=lambda_single,
        lambda_dual=lambda_dual,
        rho_single=rho_single,
        rho_dual=rho_dual,
    )

    if start_s1 is None or start_s2 is None:
        return SpeedOptResult(
            feasible=False,
            feature=feature,
            deadline=deadline,
            required_speed_if_equal=required_speed_if_equal,
            opt_speed_single=None,
            opt_speed_dual=None,
            opt_energy=None,
            opt_cmax=None,
            barrier_objective_value=None,
            barrier_outer_iterations=0,
            descent_iterations=0,
            golden_iterations=0,
            active_constraints="",
            infeasible_reason=infeasible_reason,
        )

    x = (start_s1, start_s2)
    final_barrier_value = None
    total_descent_iter = 0
    total_golden_iter = 0

    for mu in BARRIER_MU_VALUES:
        x, final_barrier_value, descent_iter, golden_iter = minimize_barrier_by_descent_and_golden(
            feature=feature,
            deadline=deadline,
            start=x,
            speed_min=speed_min,
            speed_max=speed_max,
            lambda_single=lambda_single,
            lambda_dual=lambda_dual,
            rho_single=rho_single,
            rho_dual=rho_dual,
            mu=mu,
        )
        total_descent_iter += descent_iter
        total_golden_iter += golden_iter

    s1, s2 = x

    if not is_feasible(feature, deadline, s1, s2, speed_min, speed_max):
        return SpeedOptResult(
            feasible=False,
            feature=feature,
            deadline=deadline,
            required_speed_if_equal=required_speed_if_equal,
            opt_speed_single=None,
            opt_speed_dual=None,
            opt_energy=None,
            opt_cmax=None,
            barrier_objective_value=final_barrier_value,
            barrier_outer_iterations=len(BARRIER_MU_VALUES),
            descent_iterations=total_descent_iter,
            golden_iterations=total_golden_iter,
            active_constraints="",
            infeasible_reason="barrier iteration ended at an infeasible point",
        )

    opt_energy = energy_after_speed(
        feature, s1, s2,
        lambda_single, lambda_dual, rho_single, rho_dual,
    )
    opt_cmax = cmax_after_speed(feature, s1, s2)

    active = active_constraint_text(
        feature, deadline, s1, s2, speed_min, speed_max
    )

    return SpeedOptResult(
        feasible=True,
        feature=feature,
        deadline=deadline,
        required_speed_if_equal=required_speed_if_equal,
        opt_speed_single=s1,
        opt_speed_dual=s2,
        opt_energy=opt_energy,
        opt_cmax=opt_cmax,
        barrier_objective_value=final_barrier_value,
        barrier_outer_iterations=len(BARRIER_MU_VALUES),
        descent_iterations=total_descent_iter,
        golden_iterations=total_golden_iter,
        active_constraints=active,
        infeasible_reason="",
    )


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
    speed_tol = 1e-4
    time_tol = max(1e-4, 1e-4 * max(1.0, deadline))

    if abs(g["single_lower"]) <= speed_tol:
        names.append("single_lower_speed")
    if abs(g["single_upper"]) <= speed_tol:
        names.append("single_upper_speed")
    if abs(g["dual_lower"]) <= speed_tol:
        names.append("dual_lower_speed")
    if abs(g["dual_upper"]) <= speed_tol:
        names.append("dual_upper_speed")
    if abs(g["deadline"]) <= time_tol:
        names.append("deadline")

    return "+".join(names) if names else "none"


# ============================================================
# KKT 验证
# ============================================================

def solve_nonnegative_stationarity(
    f_grad: tuple[float, float],
    active_grads: list[tuple[str, tuple[float, float]]],
) -> tuple[dict[str, float], tuple[float, float]]:
    """枚举活跃约束子集，寻找非负乘子使 ||grad f - sum mu_i grad g_i|| 最小。

    二维问题中独立活跃约束通常不超过两个，因此枚举 0/1/2 个约束即可。
    """
    all_names = [name for name, _ in active_grads]
    best_mu: dict[str, float] = {name: 0.0 for name in all_names}
    best_residual = f_grad
    best_norm = math.hypot(*best_residual)

    candidates: list[tuple[dict[str, float], tuple[float, float]]] = []
    candidates.append(({name: 0.0 for name in all_names}, f_grad))

    # 单个活跃约束。
    for name, g in active_grads:
        denom = g[0] * g[0] + g[1] * g[1]
        if denom <= EPS:
            continue

        mu = (f_grad[0] * g[0] + f_grad[1] * g[1]) / denom

        if mu >= -1e-10:
            mu = max(0.0, mu)
            residual = (f_grad[0] - mu * g[0], f_grad[1] - mu * g[1])
            candidates.append(({name: mu}, residual))

    # 两个活跃约束。
    for (name1, g1), (name2, g2) in itertools.combinations(active_grads, 2):
        det = g1[0] * g2[1] - g2[0] * g1[1]

        if abs(det) <= EPS:
            continue

        # [g1 g2] [mu1, mu2]^T = f_grad
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
            best_mu = {name: 0.0 for name in all_names}
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
    """原约束问题 KKT 条件验证。"""
    if (
        not result.feasible
        or result.opt_speed_single is None
        or result.opt_speed_dual is None
    ):
        return KKTCheck(
            feasible=False,
            g_single_lower=None,
            g_single_upper=None,
            g_dual_lower=None,
            g_dual_upper=None,
            g_deadline=None,
            mu_single_lower=None,
            mu_single_upper=None,
            mu_dual_lower=None,
            mu_dual_upper=None,
            mu_deadline=None,
            stationarity_single_residual=None,
            stationarity_dual_residual=None,
            stationarity_norm=None,
            max_complementarity_error=None,
            primal_min=None,
            dual_min=None,
            kkt_pass=False,
            kkt_note=result.infeasible_reason,
        )

    f = result.feature
    s1 = result.opt_speed_single
    s2 = result.opt_speed_dual
    d = result.deadline

    gvals = constraint_values(f, d, s1, s2, speed_min, speed_max)
    grad_f = energy_gradient(f, s1, s2, lambda_single, lambda_dual, rho_single, rho_dual)
    grad_g = constraint_gradients(f, s1, s2)

    speed_tol = 1e-4
    time_tol = max(1e-4, 1e-4 * max(1.0, d))

    active_grads: list[tuple[str, tuple[float, float]]] = []

    if abs(gvals["single_lower"]) <= speed_tol:
        active_grads.append(("single_lower", grad_g["single_lower"]))
    if abs(gvals["single_upper"]) <= speed_tol:
        active_grads.append(("single_upper", grad_g["single_upper"]))
    if abs(gvals["dual_lower"]) <= speed_tol:
        active_grads.append(("dual_lower", grad_g["dual_lower"]))
    if abs(gvals["dual_upper"]) <= speed_tol:
        active_grads.append(("dual_upper", grad_g["dual_upper"]))
    if abs(gvals["deadline"]) <= time_tol:
        active_grads.append(("deadline", grad_g["deadline"]))

    mu_active, residual = solve_nonnegative_stationarity(grad_f, active_grads)

    mu = {
        "single_lower": 0.0,
        "single_upper": 0.0,
        "dual_lower": 0.0,
        "dual_upper": 0.0,
        "deadline": 0.0,
    }
    mu.update(mu_active)

    comp = [abs(mu[name] * gvals[name]) for name in mu]
    max_comp = max(comp) if comp else 0.0

    primal_min = min(gvals.values())
    dual_min = min(mu.values())
    res_norm = math.hypot(*residual)
    scale = max(1.0, math.hypot(*grad_f), abs(result.opt_energy or 1.0))

    kkt_pass = (
        primal_min >= -KKT_TOL * max(1.0, d)
        and dual_min >= -KKT_TOL
        and max_comp <= KKT_TOL * scale
        and res_norm <= 0.08 * max(1.0, math.hypot(*grad_f))
    )

    note = (
        "KKT conditions numerically satisfied for barrier-golden solution"
        if kkt_pass
        else "KKT residual is reported for numerical inspection"
    )

    return KKTCheck(
        feasible=True,
        g_single_lower=gvals["single_lower"],
        g_single_upper=gvals["single_upper"],
        g_dual_lower=gvals["dual_lower"],
        g_dual_upper=gvals["dual_upper"],
        g_deadline=gvals["deadline"],
        mu_single_lower=mu["single_lower"],
        mu_single_upper=mu["single_upper"],
        mu_dual_lower=mu["dual_lower"],
        mu_dual_upper=mu["dual_upper"],
        mu_deadline=mu["deadline"],
        stationarity_single_residual=residual[0],
        stationarity_dual_residual=residual[1],
        stationarity_norm=res_norm,
        max_complementarity_error=max_comp,
        primal_min=primal_min,
        dual_min=dual_min,
        kkt_pass=kkt_pass,
        kkt_note=note,
    )


# ============================================================
# 2B/3B 比较
# ============================================================

def compare_2b_3b(
    opt_2b: SpeedOptResult,
    opt_3b: SpeedOptResult,
) -> tuple[str, str, str, float | None]:
    if not opt_2b.feasible and not opt_3b.feasible:
        return "none", "infeasible_both", "neither 2B nor 3B can meet the deadline", None

    if opt_2b.feasible and not opt_3b.feasible:
        return "2B", "recommend_2arm_only_feasible", "only 2B is feasible", None

    if opt_3b.feasible and not opt_2b.feasible:
        return "3B", "recommend_3arm_only_feasible", "only 3B is feasible", None

    assert opt_2b.opt_energy is not None
    assert opt_3b.opt_energy is not None

    advantage = opt_2b.opt_energy - opt_3b.opt_energy

    if advantage > EPS:
        return (
            "3B",
            "recommend_3arm_lower_energy_under_deadline",
            "3B has lower optimized energy under the same deadline",
            advantage,
        )

    if advantage < -EPS:
        return (
            "2B",
            "recommend_2arm_lower_energy_under_deadline",
            "2B has lower optimized energy under the same deadline",
            advantage,
        )

    return (
        "2B",
        "similar_prefer_2arm",
        "optimized energies are nearly equal, prefer simpler 2B",
        advantage,
    )


# ============================================================
# 敏感性参数组
# ============================================================

def get_parameter_sets(
    include_sensitivity: bool,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> list[tuple[str, float, float, float, float]]:
    baseline = ("baseline", lambda_single, lambda_dual, rho_single, rho_dual)

    if not include_sensitivity:
        return [baseline]

    return [
        ("baseline", lambda_single, lambda_dual, rho_single, rho_dual),
        ("low_slow_penalty", 0.25, 0.35, rho_single, rho_dual),
        ("high_slow_penalty", 0.45, 0.55, rho_single, rho_dual),
        ("high_fast_penalty", lambda_single, lambda_dual, 0.50, 0.60),
    ]


# ============================================================
# 输出构造
# ============================================================

def build_output_rows(
    averaged_cases: list[AveragedCase],
    method: str,
    deadline_ratios: list[float],
    speed_min: float,
    speed_max: float,
    parameter_sets: list[tuple[str, float, float, float, float]],
) -> tuple[list[dict], list[dict]]:
    result_rows: list[dict] = []
    kkt_rows: list[dict] = []

    for sensitivity_case, lam_s, lam_d, rho_s, rho_d in parameter_sets:
        for case in averaged_cases:
            scen_type = scenario_type(case.n1, case.n2, case.n3, case.n4)

            feature_2b = build_mode_feature(
                base_cmax=case.mean_cmax_2b,
                base_energy=case.mean_energy_2b,
                n1=case.n1,
                n2=case.n2,
                n3=case.n3,
                n4=case.n4,
                mode="2B",
            )
            feature_3b = build_mode_feature(
                base_cmax=case.mean_cmax_3b,
                base_energy=case.mean_energy_3b,
                n1=case.n1,
                n2=case.n2,
                n3=case.n3,
                n4=case.n4,
                mode="3B",
            )

            for ratio in deadline_ratios:
                deadline = ratio * case.mean_cmax_2b

                opt_2b = solve_speed_problem_by_barrier(
                    feature=feature_2b,
                    deadline=deadline,
                    speed_min=speed_min,
                    speed_max=speed_max,
                    lambda_single=lam_s,
                    lambda_dual=lam_d,
                    rho_single=rho_s,
                    rho_dual=rho_d,
                )
                opt_3b = solve_speed_problem_by_barrier(
                    feature=feature_3b,
                    deadline=deadline,
                    speed_min=speed_min,
                    speed_max=speed_max,
                    lambda_single=lam_s,
                    lambda_dual=lam_d,
                    rho_single=rho_s,
                    rho_dual=rho_d,
                )

                kkt_2b = verify_kkt(opt_2b, speed_min, speed_max, lam_s, lam_d, rho_s, rho_d)
                kkt_3b = verify_kkt(opt_3b, speed_min, speed_max, lam_s, lam_d, rho_s, rho_d)

                recommended_mode, recommendation, reason, advantage = compare_2b_3b(opt_2b, opt_3b)

                common = {
                    "sensitivity_case": sensitivity_case,
                    "counts_code": case.counts_code,
                    "n1": case.n1,
                    "n2": case.n2,
                    "n3": case.n3,
                    "n4": case.n4,
                    "total_tasks": case.total_tasks,
                    "scenario_type": scen_type,
                    "seed_count": case.seed_count,
                    "seeds_used": case.seeds_used,
                    "method": method,
                    "deadline_ratio": round_or_blank(ratio, 4),
                    "deadline_value": round_or_blank(deadline, 6),
                    "lambda_single": round_or_blank(lam_s, 4),
                    "lambda_dual": round_or_blank(lam_d, 4),
                    "rho_single": round_or_blank(rho_s, 4),
                    "rho_dual": round_or_blank(rho_d, 4),
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
                    "barrier_outer_iterations_2B": opt_2b.barrier_outer_iterations,
                    "descent_iterations_2B": opt_2b.descent_iterations,
                    "golden_iterations_2B": opt_2b.golden_iterations,
                    "active_constraints_2B": opt_2b.active_constraints,
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
                    "barrier_outer_iterations_3B": opt_3b.barrier_outer_iterations,
                    "descent_iterations_3B": opt_3b.descent_iterations,
                    "golden_iterations_3B": opt_3b.golden_iterations,
                    "active_constraints_3B": opt_3b.active_constraints,
                    "infeasible_reason_3B": opt_3b.infeasible_reason,
                    "kkt_pass_3B": int(kkt_3b.kkt_pass),

                    "energy_advantage_2B_minus_3B": round_or_blank(advantage, 6),
                    "recommended_mode": recommended_mode,
                    "recommendation": recommendation,
                    "recommendation_reason": reason,

                    "model_formula": (
                        "min E_fixed+E_single(s_single)+E_dual(s_dual), "
                        "s.t. C_fixed+C_single/s_single+C_dual/s_dual<=D"
                    ),
                    "energy_formula": (
                        "E_k(s)=E_k0*(lambda_k/s+(1-lambda_k)*((1-rho_k)+rho_k*s^2))"
                    ),
                    "course_method": (
                        "interior_barrier_method + steepest_descent_direction + golden_section_0_618 + KKT_verification"
                    ),
                }
                result_rows.append(row)

                for mode_name, opt, kkt in [
                    ("2B", opt_2b, kkt_2b),
                    ("3B", opt_3b, kkt_3b),
                ]:
                    kkt_rows.append(
                        {
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
                            "barrier_outer_iterations": opt.barrier_outer_iterations,
                            "descent_iterations": opt.descent_iterations,
                            "golden_iterations": opt.golden_iterations,
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
                            "stationarity_norm": round_or_blank(kkt.stationarity_norm, 10),
                            "max_complementarity_error": round_or_blank(kkt.max_complementarity_error, 10),
                            "primal_min": round_or_blank(kkt.primal_min, 10),
                            "dual_min": round_or_blank(kkt.dual_min, 10),
                            "kkt_pass": int(kkt.kkt_pass),
                            "kkt_note": kkt.kkt_note,
                        }
                    )

    return result_rows, kkt_rows


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-variable nonlinear speed-energy optimization using interior barrier method and golden section line search."
    )

    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="method suffix in input CSV, e.g. optimized_heuristic or optimized",
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
        "--lambda-single",
        type=float,
        default=DEFAULT_LAMBDA_SINGLE,
        help="slow-speed penalty weight for single-arm tasks",
    )

    parser.add_argument(
        "--lambda-dual",
        type=float,
        default=DEFAULT_LAMBDA_DUAL,
        help="slow-speed penalty weight for dual-arm cooperative tasks",
    )

    parser.add_argument(
        "--rho-single",
        type=float,
        default=DEFAULT_RHO_SINGLE,
        help="fast-speed penalty weight for single-arm tasks",
    )

    parser.add_argument(
        "--rho-dual",
        type=float,
        default=DEFAULT_RHO_DUAL,
        help="fast-speed penalty weight for dual-arm cooperative tasks",
    )

    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="only output baseline parameter set; by default sensitivity cases are merged into main CSV",
    )

    args = parser.parse_args()

    method = args.method.strip()

    input_path = (
        Path(args.input)
        if args.input.strip()
        else OUTPUT_DIR / "four_case_framework" / f"{method}_2B3B_time_energy.csv"
    )

    output_dir = OUTPUT_DIR / "nonlinear_programming" / "speed_energy_optimization"
    result_path = output_dir / "speed_energy_optimization.csv"
    kkt_path = output_dir / "kkt_verification.csv"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)
    deadline_ratios = parse_float_list(args.deadline_ratios)

    averaged_cases = build_averaged_cases(raw_rows=raw_rows, method=method)

    parameter_sets = get_parameter_sets(
        include_sensitivity=not args.no_sensitivity,
        lambda_single=args.lambda_single,
        lambda_dual=args.lambda_dual,
        rho_single=args.rho_single,
        rho_dual=args.rho_dual,
    )

    result_rows, kkt_rows = build_output_rows(
        averaged_cases=averaged_cases,
        method=method,
        deadline_ratios=deadline_ratios,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        parameter_sets=parameter_sets,
    )

    if not result_rows:
        print("Warning: no result rows generated. Please check input CSV columns and method suffix.")
        return

    result_fieldnames = list(result_rows[0].keys())
    kkt_fieldnames = list(kkt_rows[0].keys()) if kkt_rows else []

    save_csv(result_rows, result_path, result_fieldnames)

    if kkt_rows:
        save_csv(kkt_rows, kkt_path, kkt_fieldnames)

    print("Two-variable nonlinear speed-energy optimization completed.")
    print(f"Input: {input_path}")
    print(f"Output result: {result_path}")
    print(f"Output KKT: {kkt_path}")
    print(f"Cases: {len(averaged_cases)}")
    print(f"Deadline ratios: {len(deadline_ratios)}")
    print(f"Parameter sets merged in main CSV: {len(parameter_sets)}")
    print("Method: interior barrier method + steepest descent + golden section 0.618 + KKT verification")


if __name__ == "__main__":
    main()
