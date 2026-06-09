# -*- coding: utf-8 -*-
"""非线性规划：seed 平均下的速度-能耗优化、KKT 验证与二臂/三臂模式比较。


解决的问题：
    先通过主问题得到默认速度下的二臂/三臂调度结果：
        Cmax_2B, Energy_2B, Cmax_3B, Energy_3B

    对同一 counts_code 下 seed=0,1,2 的结果取平均，
    得到代表性平均场景：
        mean_cmax_2B, mean_energy_2B
        mean_cmax_3B, mean_energy_3B

    在固定离散调度结果的基础上，只引入一个连续变量：
        s = 速度倍率

连续非线性规划模型：
    对 m in {2B, 3B} 分别求解：

        min  E_m(s) = E_m0 * [(1-rho) + rho*s^2]

        s.t. C_m0 / s <= D
             s_min <= s <= s_max

其中：
    C_m0：seed 平均后的默认完工时间；
    E_m0：seed 平均后的默认总能耗；
    D：截止时间；
    rho：速度敏感能耗比例；
    s：连续速度倍率。

    g1(s) = s - s_min >= 0
    g2(s) = s_max - s >= 0
    g3(s) = D - C0/s >= 0

方法：
    1. 有约束非线性规划建模；
    2. 内点制约函数法：
           Phi(s, mu) = E(s) - mu * sum(log(g_i(s)))
    3. 0.618 黄金分割法求一维极小；
    4. KKT 条件验证：
           stationarity
           primal feasibility
           dual feasibility
           complementary slackness
    5. 二分法求二臂/三臂能耗相等的切换边界：
           A(r) = E*_2B(r) - E*_3B(r) = 0

默认输入：
    outputs/four_case_framework/optimized_2B3B_time_energy.csv

默认输出：
    outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization.csv
    outputs/nonlinear_programming/speed_energy_optimization/kkt_verification.csv

"""

from __future__ import annotations

import argparse
import csv
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

# deadline_ratio = 1.00 表示 D = 平均二臂 Cmax。
# 小于 1 表示任务更紧急，大于 1 表示截止时间更宽松。
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

DEFAULT_SPEED_MIN = 0.60
DEFAULT_SPEED_MAX = 1.60

# 速度敏感能耗比例。
DEFAULT_RHO = 0.35

# 0.618 黄金分割法参数。
GOLDEN_RATIO = 0.6180339887498949
GOLDEN_TOL = 1e-8
GOLDEN_MAX_ITER = 200

# 内点制约函数参数。
BARRIER_MU_VALUES = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

# 二分法参数，用于寻找 E2*=E3* 的切换点。
BISECTION_TOL = 1e-6
BISECTION_MAX_ITER = 100

EPS = 1e-9
KKT_TOL = 1e-5


# ============================================================
# 数据结构
# ============================================================

@dataclass
class AveragedCase:
    """同一个 counts_code 下 seed 平均后的代表性场景。"""

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
class SpeedOptResult:
    """某个模式在某个截止时间下的速度优化结果。"""

    feasible: bool
    base_cmax: float
    base_energy: float
    deadline: float
    required_speed: float

    opt_speed_kkt: float | None
    opt_energy_kkt: float | None
    opt_cmax_after_speed: float | None

    barrier_speed: float | None
    barrier_energy: float | None
    barrier_iterations: int

    active_constraints: str
    infeasible_reason: str


@dataclass
class KKTCheck:
    """KKT 条件检验结果。"""

    feasible: bool

    g_lower: float | None
    g_upper: float | None
    g_deadline: float | None

    mu_lower: float | None
    mu_upper: float | None
    mu_deadline: float | None

    stationarity_residual: float | None
    max_complementarity_error: float | None
    primal_min: float | None
    dual_min: float | None

    kkt_pass: bool
    kkt_note: str


@dataclass
class BoundaryInfo:
    """二臂/三臂切换边界信息。"""

    boundary_type: str
    switch_boundary_ratio_estimate: float | None
    boundary_interval_low: float | None
    boundary_interval_high: float | None
    advantage_at_boundary: float | None
    boundary_method: str
    boundary_explanation: str


# ============================================================
# 基础工具函数
# ============================================================

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


def round_or_blank(value: Any, digits: int = 6) -> Any:
    """数值保留小数；None 输出空白。"""
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
    """均值。"""
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    """样本标准差。"""
    if len(values) <= 1:
        return 0.0

    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def parse_float_list(text: str) -> list[float]:
    """解析逗号分隔浮点数列表。"""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def read_csv(path: Path) -> list[dict]:
    """读取 CSV。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """保存 CSV。"""
    ensure_dirs(path.parent)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def counts_code_from_row(row: dict) -> str:
    """从行中读取或生成 counts_code。"""
    if row.get("counts_code"):
        return str(row["counts_code"])

    n1 = safe_int(row.get("n1"))
    n2 = safe_int(row.get("n2"))
    n3 = safe_int(row.get("n3"))
    n4 = safe_int(row.get("n4"))

    return f"{n1}{n2}{n3}{n4}"


def get_method_columns(method: str) -> dict[str, str]:
    """根据 method 返回输入 CSV 需要读取的列名。"""
    return {
        "cmax_2b": f"cmax_2B_{method}",
        "energy_2b": f"energy_2B_{method}",
        "cmax_3b": f"cmax_3B_{method}",
        "energy_3b": f"energy_3B_{method}",
    }


def scenario_type(n1: int, n2: int, n3: int, n4: int) -> str:
    """简单任务结构分类，方便后续报告解释。"""
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
    """按 counts_code 分组，对 seed=0,1,2 的结果取平均。"""
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
# 连续调速模型
# ============================================================

def cmax_after_speed(base_cmax: float, speed: float) -> float:
    """速度倍率为 speed 时的完工时间。"""
    return base_cmax / speed


def energy_after_speed(base_energy: float, speed: float, rho: float) -> float:
    """速度倍率为 speed 时的能耗。"""
    return base_energy * ((1.0 - rho) + rho * speed * speed)


def energy_derivative(base_energy: float, speed: float, rho: float) -> float:
    """能耗函数对速度倍率的一阶导数。"""
    return 2.0 * base_energy * rho * speed


def constraint_values(
    base_cmax: float,
    deadline: float,
    speed: float,
    speed_min: float,
    speed_max: float,
) -> tuple[float, float, float]:
    """返回 g1, g2, g3，均要求 >= 0。"""
    g_lower = speed - speed_min
    g_upper = speed_max - speed
    g_deadline = deadline - base_cmax / speed

    return g_lower, g_upper, g_deadline


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
    """用 0.618 黄金分割法求一维函数在 [left, right] 上的极小值。"""
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
# 内点制约函数法
# ============================================================

def barrier_objective(
    speed: float,
    base_cmax: float,
    base_energy: float,
    deadline: float,
    speed_min: float,
    speed_max: float,
    rho: float,
    mu: float,
) -> float:
    """内点对数制约函数。"""
    g_lower, g_upper, g_deadline = constraint_values(
        base_cmax=base_cmax,
        deadline=deadline,
        speed=speed,
        speed_min=speed_min,
        speed_max=speed_max,
    )

    # 内点法要求严格可行。
    if g_lower <= 0 or g_upper <= 0 or g_deadline <= 0:
        return float("inf")

    original = energy_after_speed(base_energy, speed, rho)
    barrier = -mu * (
        math.log(g_lower)
        + math.log(g_upper)
        + math.log(g_deadline)
    )

    return original + barrier


def solve_by_barrier_and_golden(
    base_cmax: float,
    base_energy: float,
    deadline: float,
    speed_min: float,
    speed_max: float,
    rho: float,
) -> tuple[float | None, float | None, int]:
    """用内点制约函数法 + 0.618 法求近似解。"""
    required_speed = base_cmax / deadline
    lower = max(speed_min, required_speed)
    upper = speed_max

    if lower > upper + EPS:
        return None, None, 0

    width = upper - lower

    if width <= 1e-10:
        speed = lower
        return speed, energy_after_speed(base_energy, speed, rho), 0

    inner_left = lower + max(1e-8, 1e-8 * width)
    inner_right = upper - max(1e-8, 1e-8 * width)

    if inner_left >= inner_right:
        speed = 0.5 * (lower + upper)
        return speed, energy_after_speed(base_energy, speed, rho), 0

    total_iter = 0
    best_speed = None
    best_energy = None

    for mu in BARRIER_MU_VALUES:
        func = lambda s, m=mu: barrier_objective(
            speed=s,
            base_cmax=base_cmax,
            base_energy=base_energy,
            deadline=deadline,
            speed_min=speed_min,
            speed_max=speed_max,
            rho=rho,
            mu=m,
        )

        speed_mu, _phi_mu, iterations = golden_section_minimize(
            func=func,
            left=inner_left,
            right=inner_right,
        )

        total_iter += iterations
        best_speed = speed_mu
        best_energy = energy_after_speed(base_energy, speed_mu, rho)

    return best_speed, best_energy, total_iter


# ============================================================
# KKT 解析解与验证
# ============================================================

def active_constraint_text(
    speed: float,
    base_cmax: float,
    deadline: float,
    speed_min: float,
    speed_max: float,
) -> str:
    """判断哪些约束起作用。"""
    g_lower, g_upper, g_deadline = constraint_values(
        base_cmax=base_cmax,
        deadline=deadline,
        speed=speed,
        speed_min=speed_min,
        speed_max=speed_max,
    )

    active = []

    if abs(g_lower) <= 1e-6:
        active.append("lower_speed")

    if abs(g_upper) <= 1e-6:
        active.append("upper_speed")

    if abs(g_deadline) <= 1e-6:
        active.append("deadline")

    return "+".join(active) if active else "none"


def solve_speed_by_kkt_boundary(
    base_cmax: float,
    base_energy: float,
    deadline: float,
    speed_min: float,
    speed_max: float,
    rho: float,
) -> SpeedOptResult:
    """根据单调性和 KKT 边界条件求连续调速子问题的最优解。

    因为 E(s)=E0*((1-rho)+rho*s^2) 在 rho>0 且 s>0 时单调递增，
    所以最优速度一定是满足所有约束的最小速度：

        s* = max(s_min, C0/D)

    若 s* > s_max，则该模式在该截止时间下不可行。
    """
    if base_cmax <= 0 or base_energy <= 0 or deadline <= 0:
        return SpeedOptResult(
            feasible=False,
            base_cmax=base_cmax,
            base_energy=base_energy,
            deadline=deadline,
            required_speed=float("inf"),
            opt_speed_kkt=None,
            opt_energy_kkt=None,
            opt_cmax_after_speed=None,
            barrier_speed=None,
            barrier_energy=None,
            barrier_iterations=0,
            active_constraints="",
            infeasible_reason="invalid input: Cmax, Energy or deadline is not positive",
        )

    required_speed = base_cmax / deadline
    opt_speed = max(speed_min, required_speed)

    if opt_speed > speed_max + EPS:
        return SpeedOptResult(
            feasible=False,
            base_cmax=base_cmax,
            base_energy=base_energy,
            deadline=deadline,
            required_speed=required_speed,
            opt_speed_kkt=None,
            opt_energy_kkt=None,
            opt_cmax_after_speed=None,
            barrier_speed=None,
            barrier_energy=None,
            barrier_iterations=0,
            active_constraints="",
            infeasible_reason="required speed exceeds speed_max",
        )

    opt_speed = min(opt_speed, speed_max)
    opt_energy = energy_after_speed(base_energy, opt_speed, rho)
    opt_cmax = cmax_after_speed(base_cmax, opt_speed)

    barrier_speed, barrier_energy, barrier_iterations = solve_by_barrier_and_golden(
        base_cmax=base_cmax,
        base_energy=base_energy,
        deadline=deadline,
        speed_min=speed_min,
        speed_max=speed_max,
        rho=rho,
    )

    active = active_constraint_text(
        speed=opt_speed,
        base_cmax=base_cmax,
        deadline=deadline,
        speed_min=speed_min,
        speed_max=speed_max,
    )

    return SpeedOptResult(
        feasible=True,
        base_cmax=base_cmax,
        base_energy=base_energy,
        deadline=deadline,
        required_speed=required_speed,
        opt_speed_kkt=opt_speed,
        opt_energy_kkt=opt_energy,
        opt_cmax_after_speed=opt_cmax,
        barrier_speed=barrier_speed,
        barrier_energy=barrier_energy,
        barrier_iterations=barrier_iterations,
        active_constraints=active,
        infeasible_reason="",
    )


def verify_kkt(
    result: SpeedOptResult,
    speed_min: float,
    speed_max: float,
    rho: float,
) -> KKTCheck:
    """验证 KKT 条件。"""
    if not result.feasible or result.opt_speed_kkt is None:
        return KKTCheck(
            feasible=False,
            g_lower=None,
            g_upper=None,
            g_deadline=None,
            mu_lower=None,
            mu_upper=None,
            mu_deadline=None,
            stationarity_residual=None,
            max_complementarity_error=None,
            primal_min=None,
            dual_min=None,
            kkt_pass=False,
            kkt_note=result.infeasible_reason,
        )

    s = result.opt_speed_kkt
    c0 = result.base_cmax
    e0 = result.base_energy
    d = result.deadline

    g_lower, g_upper, g_deadline = constraint_values(
        base_cmax=c0,
        deadline=d,
        speed=s,
        speed_min=speed_min,
        speed_max=speed_max,
    )

    f_prime = energy_derivative(e0, s, rho)

    # g1 = s - s_min,     g1' = 1
    # g2 = s_max - s,     g2' = -1
    # g3 = D - C0 / s,    g3' = C0 / s^2
    dg_lower = 1.0
    dg_upper = -1.0
    dg_deadline = c0 / (s * s)

    mu_lower = 0.0
    mu_upper = 0.0
    mu_deadline = 0.0

    active_lower = abs(g_lower) <= 1e-6
    active_deadline = abs(g_deadline) <= 1e-6

    # 目标函数随 s 单调递增，最优点通常被 lower_speed 或 deadline 约束顶住。
    if active_lower:
        mu_lower = f_prime
    elif active_deadline:
        mu_deadline = f_prime / dg_deadline

    stationarity = (
        f_prime
        - mu_lower * dg_lower
        - mu_upper * dg_upper
        - mu_deadline * dg_deadline
    )

    comp_lower = abs(mu_lower * g_lower)
    comp_upper = abs(mu_upper * g_upper)
    comp_deadline = abs(mu_deadline * g_deadline)

    max_comp = max(comp_lower, comp_upper, comp_deadline)
    primal_min = min(g_lower, g_upper, g_deadline)
    dual_min = min(mu_lower, mu_upper, mu_deadline)

    scale = max(1.0, abs(f_prime))

    stationarity_ok = abs(stationarity) <= KKT_TOL * scale
    complementarity_ok = max_comp <= KKT_TOL * scale
    primal_ok = primal_min >= -KKT_TOL
    dual_ok = dual_min >= -KKT_TOL

    kkt_pass = (
        stationarity_ok
        and complementarity_ok
        and primal_ok
        and dual_ok
    )

    note = "KKT conditions satisfied" if kkt_pass else "KKT residual exceeds tolerance"

    return KKTCheck(
        feasible=True,
        g_lower=g_lower,
        g_upper=g_upper,
        g_deadline=g_deadline,
        mu_lower=mu_lower,
        mu_upper=mu_upper,
        mu_deadline=mu_deadline,
        stationarity_residual=stationarity,
        max_complementarity_error=max_comp,
        primal_min=primal_min,
        dual_min=dual_min,
        kkt_pass=kkt_pass,
        kkt_note=note,
    )


# ============================================================
# 二臂/三臂比较与切换边界
# ============================================================

def compare_2b_3b(
    opt_2b: SpeedOptResult,
    opt_3b: SpeedOptResult,
) -> tuple[str, str, str, float | None]:
    """比较同一 deadline 下 2B 和 3B 的优化后能耗。"""
    if not opt_2b.feasible and not opt_3b.feasible:
        return "none", "infeasible_both", "neither 2B nor 3B can meet the deadline", None

    if opt_2b.feasible and not opt_3b.feasible:
        return "2B", "recommend_2arm_only_feasible", "only 2B is feasible", None

    if opt_3b.feasible and not opt_2b.feasible:
        return "3B", "recommend_3arm_only_feasible", "only 3B is feasible", None

    assert opt_2b.opt_energy_kkt is not None
    assert opt_3b.opt_energy_kkt is not None

    advantage = opt_2b.opt_energy_kkt - opt_3b.opt_energy_kkt

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


def advantage_at_ratio(
    case: AveragedCase,
    ratio: float,
    speed_min: float,
    speed_max: float,
    rho: float,
) -> float | None:
    """计算 A(r)=E*_2B(r)-E*_3B(r)。

    A(r)>0 表示 3B 优化后能耗更低；
    A(r)<0 表示 2B 优化后能耗更低。
    """
    deadline = ratio * case.mean_cmax_2b

    opt_2b = solve_speed_by_kkt_boundary(
        base_cmax=case.mean_cmax_2b,
        base_energy=case.mean_energy_2b,
        deadline=deadline,
        speed_min=speed_min,
        speed_max=speed_max,
        rho=rho,
    )

    opt_3b = solve_speed_by_kkt_boundary(
        base_cmax=case.mean_cmax_3b,
        base_energy=case.mean_energy_3b,
        deadline=deadline,
        speed_min=speed_min,
        speed_max=speed_max,
        rho=rho,
    )

    if not opt_2b.feasible or not opt_3b.feasible:
        return None

    if opt_2b.opt_energy_kkt is None or opt_3b.opt_energy_kkt is None:
        return None

    return opt_2b.opt_energy_kkt - opt_3b.opt_energy_kkt


def bisection_root_for_boundary(
    case: AveragedCase,
    left: float,
    right: float,
    speed_min: float,
    speed_max: float,
    rho: float,
) -> tuple[float | None, float | None, int]:
    """用二分法求 A(r)=0 的根。"""
    f_left = advantage_at_ratio(case, left, speed_min, speed_max, rho)
    f_right = advantage_at_ratio(case, right, speed_min, speed_max, rho)

    if f_left is None or f_right is None:
        return None, None, 0

    if abs(f_left) <= BISECTION_TOL:
        return left, f_left, 0

    if abs(f_right) <= BISECTION_TOL:
        return right, f_right, 0

    if f_left * f_right > 0:
        return None, None, 0

    a = left
    b = right
    fa = f_left

    iterations = 0

    for _ in range(BISECTION_MAX_ITER):
        iterations += 1
        mid = 0.5 * (a + b)
        fm = advantage_at_ratio(case, mid, speed_min, speed_max, rho)

        if fm is None:
            return None, None, iterations

        if abs(fm) <= BISECTION_TOL or abs(b - a) <= BISECTION_TOL:
            return mid, fm, iterations

        if fa * fm <= 0:
            b = mid
        else:
            a = mid
            fa = fm

    mid = 0.5 * (a + b)
    fm = advantage_at_ratio(case, mid, speed_min, speed_max, rho)
    return mid, fm, iterations


def find_switch_boundary(
    case: AveragedCase,
    deadline_ratios: list[float],
    speed_min: float,
    speed_max: float,
    rho: float,
) -> BoundaryInfo:
    """先扫描 deadline_ratio，再用二分法寻找 2B/3B 能耗相等点。"""
    ordered_ratios = sorted(deadline_ratios)

    scanned: list[tuple[float, float]] = []

    for ratio in ordered_ratios:
        advantage = advantage_at_ratio(
            case=case,
            ratio=ratio,
            speed_min=speed_min,
            speed_max=speed_max,
            rho=rho,
        )

        if advantage is not None:
            scanned.append((ratio, advantage))

    if not scanned:
        return BoundaryInfo(
            boundary_type="no_comparable_feasible_point",
            switch_boundary_ratio_estimate=None,
            boundary_interval_low=None,
            boundary_interval_high=None,
            advantage_at_boundary=None,
            boundary_method="scan_only",
            boundary_explanation="No deadline ratio has both 2B and 3B feasible, so no energy boundary can be computed.",
        )

    positive_ratios = [r for r, a in scanned if a > EPS]
    negative_ratios = [r for r, a in scanned if a < -EPS]
    zero_ratios = [r for r, a in scanned if abs(a) <= EPS]

    if zero_ratios:
        r0 = zero_ratios[0]
        return BoundaryInfo(
            boundary_type="exact_boundary_on_scanned_ratio",
            switch_boundary_ratio_estimate=r0,
            boundary_interval_low=r0,
            boundary_interval_high=r0,
            advantage_at_boundary=0.0,
            boundary_method="direct_scan",
            boundary_explanation=f"Energy equality appears directly at deadline_ratio={r0:.6f}.",
        )

    # 找相邻点的符号变化区间。
    for i in range(len(scanned) - 1):
        r_left, a_left = scanned[i]
        r_right, a_right = scanned[i + 1]

        if a_left * a_right < 0:
            root, adv_root, _iters = bisection_root_for_boundary(
                case=case,
                left=r_left,
                right=r_right,
                speed_min=speed_min,
                speed_max=speed_max,
                rho=rho,
            )

            if root is not None:
                return BoundaryInfo(
                    boundary_type="switching_boundary_found",
                    switch_boundary_ratio_estimate=root,
                    boundary_interval_low=r_left,
                    boundary_interval_high=r_right,
                    advantage_at_boundary=adv_root,
                    boundary_method="scan_plus_bisection",
                    boundary_explanation=(
                        f"A sign change is detected between {r_left:.4f} and {r_right:.4f}; "
                        f"bisection gives switch boundary r*≈{root:.6f}."
                    ),
                )

    if positive_ratios and not negative_ratios:
        return BoundaryInfo(
            boundary_type="all_3arm_lower_energy_in_scan",
            switch_boundary_ratio_estimate=None,
            boundary_interval_low=min(positive_ratios),
            boundary_interval_high=max(positive_ratios),
            advantage_at_boundary=None,
            boundary_method="scan_only",
            boundary_explanation="In all comparable scanned deadline ratios, A(r)>0, so 3B has lower optimized energy.",
        )

    if negative_ratios and not positive_ratios:
        return BoundaryInfo(
            boundary_type="all_2arm_lower_energy_in_scan",
            switch_boundary_ratio_estimate=None,
            boundary_interval_low=min(negative_ratios),
            boundary_interval_high=max(negative_ratios),
            advantage_at_boundary=None,
            boundary_method="scan_only",
            boundary_explanation="In all comparable scanned deadline ratios, A(r)<0, so 2B has lower optimized energy.",
        )

    return BoundaryInfo(
        boundary_type="no_clear_switching_boundary",
        switch_boundary_ratio_estimate=None,
        boundary_interval_low=None,
        boundary_interval_high=None,
        advantage_at_boundary=None,
        boundary_method="scan_only",
        boundary_explanation="No clear switching boundary is detected in the scanned deadline range.",
    )


# ============================================================
# 结果生成
# ============================================================

def build_output_rows(
    averaged_cases: list[AveragedCase],
    method: str,
    deadline_ratios: list[float],
    speed_min: float,
    speed_max: float,
    rho: float,
) -> tuple[list[dict], list[dict]]:
    """生成主结果表和 KKT 验证表。"""
    result_rows: list[dict] = []
    kkt_rows: list[dict] = []

    for case in averaged_cases:
        scen_type = scenario_type(case.n1, case.n2, case.n3, case.n4)

        boundary = find_switch_boundary(
            case=case,
            deadline_ratios=deadline_ratios,
            speed_min=speed_min,
            speed_max=speed_max,
            rho=rho,
        )

        for ratio in deadline_ratios:
            deadline = ratio * case.mean_cmax_2b

            opt_2b = solve_speed_by_kkt_boundary(
                base_cmax=case.mean_cmax_2b,
                base_energy=case.mean_energy_2b,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                rho=rho,
            )

            opt_3b = solve_speed_by_kkt_boundary(
                base_cmax=case.mean_cmax_3b,
                base_energy=case.mean_energy_3b,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                rho=rho,
            )

            kkt_2b = verify_kkt(opt_2b, speed_min, speed_max, rho)
            kkt_3b = verify_kkt(opt_3b, speed_min, speed_max, rho)

            recommended_mode, recommendation, reason, advantage = compare_2b_3b(
                opt_2b=opt_2b,
                opt_3b=opt_3b,
            )

            common = {
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

                "mean_cmax_2B": round_or_blank(case.mean_cmax_2b, 6),
                "mean_energy_2B": round_or_blank(case.mean_energy_2b, 6),
                "mean_cmax_3B": round_or_blank(case.mean_cmax_3b, 6),
                "mean_energy_3B": round_or_blank(case.mean_energy_3b, 6),

                "std_cmax_2B": round_or_blank(case.std_cmax_2b, 6),
                "std_energy_2B": round_or_blank(case.std_energy_2b, 6),
                "std_cmax_3B": round_or_blank(case.std_cmax_3b, 6),
                "std_energy_3B": round_or_blank(case.std_energy_3b, 6),

                "rho_speed_sensitive_energy": round_or_blank(rho, 4),
                "speed_min": round_or_blank(speed_min, 4),
                "speed_max": round_or_blank(speed_max, 4),
            }

            result_rows.append(
                {
                    **common,

                    "feasible_2B": int(opt_2b.feasible),
                    "required_speed_2B": round_or_blank(opt_2b.required_speed, 6),
                    "opt_speed_2B_kkt": round_or_blank(opt_2b.opt_speed_kkt, 6),
                    "opt_energy_2B_kkt": round_or_blank(opt_2b.opt_energy_kkt, 6),
                    "opt_cmax_2B_after_speed": round_or_blank(opt_2b.opt_cmax_after_speed, 6),
                    "barrier_speed_2B_0_618": round_or_blank(opt_2b.barrier_speed, 6),
                    "barrier_energy_2B_0_618": round_or_blank(opt_2b.barrier_energy, 6),
                    "active_constraints_2B": opt_2b.active_constraints,
                    "infeasible_reason_2B": opt_2b.infeasible_reason,

                    "feasible_3B": int(opt_3b.feasible),
                    "required_speed_3B": round_or_blank(opt_3b.required_speed, 6),
                    "opt_speed_3B_kkt": round_or_blank(opt_3b.opt_speed_kkt, 6),
                    "opt_energy_3B_kkt": round_or_blank(opt_3b.opt_energy_kkt, 6),
                    "opt_cmax_3B_after_speed": round_or_blank(opt_3b.opt_cmax_after_speed, 6),
                    "barrier_speed_3B_0_618": round_or_blank(opt_3b.barrier_speed, 6),
                    "barrier_energy_3B_0_618": round_or_blank(opt_3b.barrier_energy, 6),
                    "active_constraints_3B": opt_3b.active_constraints,
                    "infeasible_reason_3B": opt_3b.infeasible_reason,

                    "energy_advantage_2B_minus_3B": round_or_blank(advantage, 6),
                    "recommended_mode": recommended_mode,
                    "recommendation": recommendation,
                    "recommendation_reason": reason,

                    "kkt_pass_2B": int(kkt_2b.kkt_pass),
                    "kkt_pass_3B": int(kkt_3b.kkt_pass),

                    "boundary_type": boundary.boundary_type,
                    "switch_boundary_ratio_estimate": round_or_blank(
                        boundary.switch_boundary_ratio_estimate, 6
                    ),
                    "boundary_interval_low": round_or_blank(boundary.boundary_interval_low, 6),
                    "boundary_interval_high": round_or_blank(boundary.boundary_interval_high, 6),
                    "advantage_at_boundary": round_or_blank(boundary.advantage_at_boundary, 10),
                    "boundary_method": boundary.boundary_method,
                    "boundary_explanation": boundary.boundary_explanation,

                    "model_formula": "min E0*((1-rho)+rho*s^2), s.t. C0/s<=D, s_min<=s<=s_max",
                    "course_method": "seed_average + interior_barrier_method + golden_section_0_618 + KKT + bisection_boundary",
                }
            )

            for mode_name, opt, kkt in [
                ("2B", opt_2b, kkt_2b),
                ("3B", opt_3b, kkt_3b),
            ]:
                kkt_rows.append(
                    {
                        **common,
                        "mode": mode_name,
                        "base_cmax": round_or_blank(opt.base_cmax, 6),
                        "base_energy": round_or_blank(opt.base_energy, 6),
                        "feasible": int(kkt.feasible),
                        "opt_speed_kkt": round_or_blank(opt.opt_speed_kkt, 6),
                        "opt_energy_kkt": round_or_blank(opt.opt_energy_kkt, 6),
                        "active_constraints": opt.active_constraints,
                        "g_lower_speed": round_or_blank(kkt.g_lower, 10),
                        "g_upper_speed": round_or_blank(kkt.g_upper, 10),
                        "g_deadline": round_or_blank(kkt.g_deadline, 10),
                        "mu_lower_speed": round_or_blank(kkt.mu_lower, 10),
                        "mu_upper_speed": round_or_blank(kkt.mu_upper, 10),
                        "mu_deadline": round_or_blank(kkt.mu_deadline, 10),
                        "stationarity_residual": round_or_blank(kkt.stationarity_residual, 10),
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
        description="Merged seed-averaged nonlinear speed-energy optimization, KKT verification and 2B/3B comparison."
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

    output_dir = OUTPUT_DIR / "nonlinear_programming" / "speed_energy_optimization"
    result_path = output_dir / "speed_energy_optimization.csv"
    kkt_path = output_dir / "kkt_verification.csv"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)
    deadline_ratios = parse_float_list(args.deadline_ratios)

    averaged_cases = build_averaged_cases(
        raw_rows=raw_rows,
        method=method,
    )

    result_rows, kkt_rows = build_output_rows(
        averaged_cases=averaged_cases,
        method=method,
        deadline_ratios=deadline_ratios,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        rho=args.rho,
    )

    result_fields = [
        "counts_code",
        "n1",
        "n2",
        "n3",
        "n4",
        "total_tasks",
        "scenario_type",
        "seed_count",
        "seeds_used",
        "method",
        "deadline_ratio",
        "deadline_value",

        "mean_cmax_2B",
        "mean_energy_2B",
        "mean_cmax_3B",
        "mean_energy_3B",

        "std_cmax_2B",
        "std_energy_2B",
        "std_cmax_3B",
        "std_energy_3B",

        "rho_speed_sensitive_energy",
        "speed_min",
        "speed_max",

        "feasible_2B",
        "required_speed_2B",
        "opt_speed_2B_kkt",
        "opt_energy_2B_kkt",
        "opt_cmax_2B_after_speed",
        "barrier_speed_2B_0_618",
        "barrier_energy_2B_0_618",
        "active_constraints_2B",
        "infeasible_reason_2B",

        "feasible_3B",
        "required_speed_3B",
        "opt_speed_3B_kkt",
        "opt_energy_3B_kkt",
        "opt_cmax_3B_after_speed",
        "barrier_speed_3B_0_618",
        "barrier_energy_3B_0_618",
        "active_constraints_3B",
        "infeasible_reason_3B",

        "energy_advantage_2B_minus_3B",
        "recommended_mode",
        "recommendation",
        "recommendation_reason",

        "kkt_pass_2B",
        "kkt_pass_3B",

        "boundary_type",
        "switch_boundary_ratio_estimate",
        "boundary_interval_low",
        "boundary_interval_high",
        "advantage_at_boundary",
        "boundary_method",
        "boundary_explanation",

        "model_formula",
        "course_method",
    ]

    kkt_fields = [
        "counts_code",
        "n1",
        "n2",
        "n3",
        "n4",
        "total_tasks",
        "scenario_type",
        "seed_count",
        "seeds_used",
        "method",
        "deadline_ratio",
        "deadline_value",

        "mean_cmax_2B",
        "mean_energy_2B",
        "mean_cmax_3B",
        "mean_energy_3B",

        "std_cmax_2B",
        "std_energy_2B",
        "std_cmax_3B",
        "std_energy_3B",

        "rho_speed_sensitive_energy",
        "speed_min",
        "speed_max",

        "mode",
        "base_cmax",
        "base_energy",
        "feasible",
        "opt_speed_kkt",
        "opt_energy_kkt",
        "active_constraints",

        "g_lower_speed",
        "g_upper_speed",
        "g_deadline",
        "mu_lower_speed",
        "mu_upper_speed",
        "mu_deadline",
        "stationarity_residual",
        "max_complementarity_error",
        "primal_min",
        "dual_min",
        "kkt_pass",
        "kkt_note",
    ]

    save_csv(result_rows, result_path, result_fields)
    save_csv(kkt_rows, kkt_path, kkt_fields)

    print("Merged nonlinear speed-energy optimization finished.")
    print(f"Input: {input_path}")
    print(f"Output 1: {result_path}")
    print(f"Output 2: {kkt_path}")
    print(f"Averaged cases: {len(averaged_cases)}")
    print(f"Optimization/comparison rows: {len(result_rows)}")
    print(f"KKT rows: {len(kkt_rows)}")
    print("Main model: min E0*((1-rho)+rho*s^2), s.t. C0/s<=D, s_min<=s<=s_max")
    print("Methods: seed averaging + interior barrier method + 0.618 golden section + KKT verification + bisection switching boundary")


if __name__ == "__main__":
    main()