# -*- coding: utf-8 -*-
"""鲁棒非线性规划：双速度变量 + SLP 逐次线性规划 + KKT 验证。

本脚本只读取已有的 2B/3B 完工时间与能耗结果，不重新运行主调度，
不重新设置启动成本、固定能耗、服务时间或物块能耗参数。

本版本用于替换原来的二维网格搜索鲁棒文件。它保留新版建模：

    s_single,m：模式 m 的单臂任务统一速度倍率
    s_dual,m  ：模式 m 的双臂协同任务统一速度倍率
    z_m        ：模式 m 在所有 seed 中的最坏能耗上界

鲁棒模型：
    对 m in {2B, 3B}，同一个 counts_code 下所有 seed 使用同一组速度：

       min z_m

       s.t. E_fixed,m,j + E_single,m,j(s_single,m) + E_dual,m,j(s_dual,m) <= z_m, for all seed j
            C_fixed,m,j + C_single,m,j/s_single,m + C_dual,m,j/s_dual,m <= D,   for all seed j
            s_min <= s_single,m <= s_max
            s_min <= s_dual,m   <= s_max

能耗函数：
       E_k(s_k)=E_k0*[lambda_k/s_k + (1-lambda_k)*((1-rho_k)+rho_k*s_k^2)]

求解方法：
    逐次线性规划法 SLP：
        1. 从一个可行速度点出发；
        2. 在当前点对每个 seed 的时间约束和能耗上界约束做一阶 Taylor 线性化；
        3. 加入步长限制 |s - s(k)| <= delta；
        4. 求一个小型线性规划子问题；
        5. 用原非线性约束检查候选点，若可行且改善则接受，否则缩小步长；
        6. 迭代至步长或目标改进足够小。

    本脚本不调用 scipy 或外部 LP 求解器。
    因为 SLP 子问题只有 3 个变量 (s_single, s_dual, z)，
    所以采用“顶点枚举法”求解线性规划子问题。

输出：
    outputs/nonlinear_programming/robust_mode_selection/robust_speed_mode_selection.csv
    outputs/nonlinear_programming/robust_mode_selection/robust_kkt_verification.csv
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

try:
    from scipy.optimize import linprog
except Exception:
    linprog = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import OUTPUT_DIR  # noqa: E402
from common.utils import ensure_dirs  # noqa: E402


# ============================================================
# 默认参数
# ============================================================

DEFAULT_METHOD = "optimized"

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

# 截止时间基准：
# mean_2b：同一 counts_code 下 2B 的平均 Cmax，默认；
# max_2b ：同一 counts_code 下 2B 的最大 Cmax，更保守；
# min_2b ：同一 counts_code 下 2B 的最小 Cmax，更激进。
DEFAULT_DEADLINE_BASE = "mean_2b"

# SLP 迭代参数。
SLP_MAX_ITER = 80
SLP_DELTA_INIT_RATIO = 0.25
SLP_DELTA_MIN = 1e-6
SLP_DELTA_EXPAND = 1.20
SLP_DELTA_SHRINK = 0.50
SLP_OBJ_TOL = 1e-7
SLP_STEP_TOL = 1e-7

EPS = 1e-10
LP_FEAS_TOL = 1e-8
KKT_TOL = 5e-3


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

    slp_iterations: int
    lp_subproblem_count: int
    final_delta: float

    infeasible_reason: str


@dataclass
class LinearConstraint:
    name: str
    a: tuple[float, float, float]
    b: float


@dataclass
class KKTCheck:
    feasible: bool

    primal_min: float | None
    dual_min: float | None

    stationarity_s_single: float | None
    stationarity_s_dual: float | None
    stationarity_z: float | None
    stationarity_norm: float | None

    max_complementarity_error: float | None

    active_constraints: str
    active_time_seeds: str
    active_energy_seeds: str
    selected_multipliers: str

    kkt_pass: bool
    kkt_note: str


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


# ============================================================
# CSV 输出字段筛选
# 只控制写出的 CSV 列，不参与任何建模、求解、KKT 计算。
# ============================================================

ROBUST_RESULT_FIELDNAMES = [
    "counts_code",
    "n1",
    "n2",
    "n3",
    "n4",
    "total_tasks",
    "scenario_type",
    "method",
    "seed_count",
    "deadline_base",
    "deadline_reference_cmax_2B",
    "deadline_ratio",
    "deadline_value",
    "lambda_single",
    "lambda_dual",
    "rho_single",
    "rho_dual",
    "speed_min",
    "speed_max",

    "mean_cmax_2B_original",
    "max_cmax_2B_original",
    "mean_energy_2B_original",
    "max_energy_2B_original",
    "robust_feasible_2B",
    "opt_speed_single_2B",
    "opt_speed_dual_2B",
    "worst_energy_z_2B",
    "mean_energy_after_speed_2B",
    "max_cmax_after_speed_2B",
    "worst_energy_seed_2B",
    "binding_time_seed_2B",
    "kkt_pass_2B",
    "infeasible_reason_2B",

    "mean_cmax_3B_original",
    "max_cmax_3B_original",
    "mean_energy_3B_original",
    "max_energy_3B_original",
    "robust_feasible_3B",
    "opt_speed_single_3B",
    "opt_speed_dual_3B",
    "worst_energy_z_3B",
    "mean_energy_after_speed_3B",
    "max_cmax_after_speed_3B",
    "worst_energy_seed_3B",
    "binding_time_seed_3B",
    "kkt_pass_3B",
    "infeasible_reason_3B",

    "worst_energy_gap_2B_minus_3B",
    "worst_energy_gap_percent_vs_2B",
    "recommended_mode",
    "recommendation",
    "recommendation_reason",
]

ROBUST_KKT_FIELDNAMES = [
    "counts_code",
    "deadline_base",
    "deadline_ratio",
    "deadline_value",
    "mode",

    "feasible",
    "opt_speed_single",
    "opt_speed_dual",
    "worst_energy_z",
    "max_cmax_after_speed",
    "worst_energy_seed",
    "binding_time_seed",

    "primal_min",
    "dual_min",
    "stationarity_s_single",
    "stationarity_s_dual",
    "stationarity_z",
    "stationarity_norm",
    "max_complementarity_error",
    "active_constraints",
    "active_time_seeds",
    "active_energy_seeds",
    "selected_multipliers",
    "kkt_pass",
    "kkt_note",
]


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


def dot3(a: tuple[float, float, float], x: tuple[float, float, float]) -> float:
    return a[0] * x[0] + a[1] * x[1] + a[2] * x[2]


# ============================================================
# 读取 seed 数据
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
# 任务特征拆分和 U 形能耗函数
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
    """从已有 seed 运行结果构造速度层特征。

    严格原则：
        不重新设定启动成本、固定能耗、服务时间或物块能耗参数；
        不重新运行主调度；
        每个 seed 只使用输入 CSV 中已有的 base_cmax/base_energy/n1/n2/n3/n4。

    为形成 s_single 与 s_dual 两个连续速度变量，
    仅按单臂任务数量与双臂任务数量占比拆分已有总时间/总能耗。
    这样 s_single=s_dual=1 时严格回到该 seed 原始 Cmax/Energy。
    """
    _ = mode

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


def economic_energy_component(
    base_energy: float,
    speed: float,
    lambda_keep: float,
    rho: float,
) -> float:
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
# 鲁棒目标统计
# ============================================================

def robust_stats(
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
    """返回鲁棒可行性和统计量。

    返回：
        feasible,
        worst_energy,
        mean_energy,
        std_energy,
        max_cmax,
        mean_cmax,
        worst_energy_seed,
        binding_time_seed
    """
    energies: list[float] = []
    cmax_values: list[float] = []

    for feature in features:
        c = cmax_after_speed(feature, speed_single, speed_dual)
        if c > deadline + 1e-7:
            return False, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), None, None

        cmax_values.append(c)
        energies.append(
            energy_after_speed(
                feature, speed_single, speed_dual,
                lambda_single, lambda_dual, rho_single, rho_dual,
            )
        )

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


def is_robust_time_feasible(
    features: list[ModeFeature],
    deadline: float,
    speed_single: float,
    speed_dual: float,
) -> bool:
    return all(
        cmax_after_speed(feature, speed_single, speed_dual) <= deadline + 1e-7
        for feature in features
    )


def find_initial_feasible_speed(
    features: list[ModeFeature],
    deadline: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> tuple[float | None, float | None, str]:
    """寻找一个原非线性约束下的可行初始速度点。"""
    if not is_robust_time_feasible(features, deadline, speed_max, speed_max):
        return None, None, "deadline cannot be met even at speed_max for both speed variables"

    s1_econ = economic_speed_unconstrained(lambda_single, rho_single, speed_min, speed_max)
    s2_econ = economic_speed_unconstrained(lambda_dual, rho_dual, speed_min, speed_max)

    if is_robust_time_feasible(features, deadline, s1_econ, s2_econ):
        return s1_econ, s2_econ, ""

    # 从经济速度向 speed_max 移动，寻找第一个可行点。
    for i in range(1, 101):
        alpha = i / 100.0
        s1 = (1.0 - alpha) * s1_econ + alpha * speed_max
        s2 = (1.0 - alpha) * s2_econ + alpha * speed_max

        if is_robust_time_feasible(features, deadline, s1, s2):
            return s1, s2, ""

    # 兜底：speed_max 一定可行，因为上面已检查。
    return speed_max, speed_max, ""


# ============================================================
# 小型 LP 子问题：顶点枚举
# ============================================================

def solve_linear_system_3(
    rows: list[tuple[float, float, float]],
    rhs: list[float],
) -> tuple[float, float, float] | None:
    """求解 3x3 线性方程组。"""
    a = [[float(rows[i][j]) for j in range(3)] + [float(rhs[i])] for i in range(3)]

    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))

        if abs(a[pivot][col]) <= 1e-12:
            return None

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]

        div = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= div

        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]

    return a[0][3], a[1][3], a[2][3]


def solve_lp_by_vertex_enumeration(
    constraints: list[LinearConstraint],
) -> tuple[float, float, float] | None:
    """求解 3 变量 LP：min z, s.t. A x >= b。

    优先使用 scipy.optimize.linprog 求解线性规划子问题，速度更快；
    如果环境中没有 scipy，则退回到顶点枚举法。
    这里的整体算法仍然是课内的 SLP 逐次线性规划法，
    linprog 只负责求每一步的一次线性规划子问题。
    """
    if linprog is not None:
        c = [0.0, 0.0, 1.0]
        a_ub = [[-v for v in cons.a] for cons in constraints]
        b_ub = [-cons.b for cons in constraints]
        res = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (None, None)], method="highs")
        if res.success and res.x is not None:
            x = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
            if all(dot3(cons.a, x) >= cons.b - 1e-7 for cons in constraints):
                return x

    # scipy 不可用时，退回顶点枚举。
    best_x: tuple[float, float, float] | None = None
    best_obj = float("inf")

    for combo in itertools.combinations(constraints, 3):
        rows = [c.a for c in combo]
        rhs = [c.b for c in combo]

        x = solve_linear_system_3(rows, rhs)
        if x is None:
            continue

        if any(not math.isfinite(v) for v in x):
            continue

        feasible = True
        for constraint in constraints:
            if dot3(constraint.a, x) < constraint.b - LP_FEAS_TOL:
                feasible = False
                break

        if not feasible:
            continue

        obj = x[2]

        if obj < best_obj:
            best_obj = obj
            best_x = x

    return best_x

def add_constraint(
    constraints: list[LinearConstraint],
    name: str,
    a: tuple[float, float, float],
    b: float,
) -> None:
    constraints.append(LinearConstraint(name=name, a=a, b=b))


def build_slp_linear_constraints(
    features: list[ModeFeature],
    seeds: list[int],
    deadline: float,
    current_x: tuple[float, float, float],
    delta_speed: float,
    delta_z: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> list[LinearConstraint]:
    """构造 SLP 的线性化约束。"""
    s1, s2, z = current_x
    constraints: list[LinearConstraint] = []

    # 原始速度上下界。
    add_constraint(constraints, "s_single_lower", (1.0, 0.0, 0.0), speed_min)
    add_constraint(constraints, "s_single_upper", (-1.0, 0.0, 0.0), -speed_max)
    add_constraint(constraints, "s_dual_lower", (0.0, 1.0, 0.0), speed_min)
    add_constraint(constraints, "s_dual_upper", (0.0, -1.0, 0.0), -speed_max)

    # z 非负。理论上能耗约束会保证 z 为正，这里加上用于数值稳定。
    add_constraint(constraints, "z_lower", (0.0, 0.0, 1.0), 0.0)

    # SLP 步长限制，贴合课内“逐次逼近法”的局部近似思想。
    add_constraint(constraints, "trust_s_single_lower", (1.0, 0.0, 0.0), s1 - delta_speed)
    add_constraint(constraints, "trust_s_single_upper", (-1.0, 0.0, 0.0), -(s1 + delta_speed))
    add_constraint(constraints, "trust_s_dual_lower", (0.0, 1.0, 0.0), s2 - delta_speed)
    add_constraint(constraints, "trust_s_dual_upper", (0.0, -1.0, 0.0), -(s2 + delta_speed))
    add_constraint(constraints, "trust_z_lower", (0.0, 0.0, 1.0), max(0.0, z - delta_z))
    add_constraint(constraints, "trust_z_upper", (0.0, 0.0, -1.0), -(z + delta_z))

    for feature, seed in zip(features, seeds):
        # 时间约束：
        # g_t = D - C(s1,s2) >= 0
        # linearized: g_t(xk)+grad_g_t^T(x-xk)>=0
        c = cmax_after_speed(feature, s1, s2)
        g_time = deadline - c
        grad_time = (
            feature.single_time / (s1 * s1),
            feature.dual_time / (s2 * s2),
            0.0,
        )
        rhs_time = dot3(grad_time, current_x) - g_time
        add_constraint(
            constraints,
            f"time_seed_{seed}",
            grad_time,
            rhs_time,
        )

        # 能耗上界约束：
        # g_e = z - E(s1,s2) >= 0
        e = energy_after_speed(
            feature, s1, s2,
            lambda_single, lambda_dual, rho_single, rho_dual,
        )
        de1, de2 = energy_gradient(
            feature, s1, s2,
            lambda_single, lambda_dual, rho_single, rho_dual,
        )

        g_energy = z - e
        grad_energy = (-de1, -de2, 1.0)
        rhs_energy = dot3(grad_energy, current_x) - g_energy
        add_constraint(
            constraints,
            f"energy_seed_{seed}",
            grad_energy,
            rhs_energy,
        )

    return constraints


# ============================================================
# SLP 求解鲁棒速度问题
# ============================================================

def solve_robust_speed_problem_slp(
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
    """使用 SLP 求解鲁棒双速度模型。"""
    if not features or deadline <= 0:
        return RobustOptResult(
            feasible=False,
            required_speed_single=float("inf"),
            required_speed_dual=float("inf"),
            opt_speed_single=None,
            opt_speed_dual=None,
            worst_energy_z=None,
            mean_energy_after_speed=None,
            std_energy_after_speed=None,
            max_cmax_after_speed=None,
            mean_cmax_after_speed=None,
            worst_energy_seed=None,
            binding_time_seed=None,
            slp_iterations=0,
            lp_subproblem_count=0,
            final_delta=0.0,
            infeasible_reason="invalid input",
        )

    # 先检查极限速度是否能让所有 seed 按时完成。
    if not is_robust_time_feasible(features, deadline, speed_max, speed_max):
        return RobustOptResult(
            feasible=False,
            required_speed_single=float("inf"),
            required_speed_dual=float("inf"),
            opt_speed_single=None,
            opt_speed_dual=None,
            worst_energy_z=None,
            mean_energy_after_speed=None,
            std_energy_after_speed=None,
            max_cmax_after_speed=None,
            mean_cmax_after_speed=None,
            worst_energy_seed=None,
            binding_time_seed=None,
            slp_iterations=0,
            lp_subproblem_count=0,
            final_delta=0.0,
            infeasible_reason="deadline cannot be met even at speed_max for both speed variables",
        )

    required_single = max(
        (
            feature.single_time
            / max(deadline - feature.fixed_time - feature.dual_time / speed_max, EPS)
        )
        for feature in features
    )
    required_dual = max(
        (
            feature.dual_time
            / max(deadline - feature.fixed_time - feature.single_time / speed_max, EPS)
        )
        for feature in features
    )

    start_s1, start_s2, infeasible_reason = find_initial_feasible_speed(
        features=features,
        deadline=deadline,
        speed_min=speed_min,
        speed_max=speed_max,
        lambda_single=lambda_single,
        lambda_dual=lambda_dual,
        rho_single=rho_single,
        rho_dual=rho_dual,
    )

    if start_s1 is None or start_s2 is None:
        return RobustOptResult(
            feasible=False,
            required_speed_single=required_single,
            required_speed_dual=required_dual,
            opt_speed_single=None,
            opt_speed_dual=None,
            worst_energy_z=None,
            mean_energy_after_speed=None,
            std_energy_after_speed=None,
            max_cmax_after_speed=None,
            mean_cmax_after_speed=None,
            worst_energy_seed=None,
            binding_time_seed=None,
            slp_iterations=0,
            lp_subproblem_count=0,
            final_delta=0.0,
            infeasible_reason=infeasible_reason,
        )

    feasible, z, mean_e, std_e, max_c, mean_c, worst_seed, binding_seed = robust_stats(
        features=features,
        seeds=seeds,
        deadline=deadline,
        speed_single=start_s1,
        speed_dual=start_s2,
        lambda_single=lambda_single,
        lambda_dual=lambda_dual,
        rho_single=rho_single,
        rho_dual=rho_dual,
    )

    if not feasible:
        return RobustOptResult(
            feasible=False,
            required_speed_single=required_single,
            required_speed_dual=required_dual,
            opt_speed_single=None,
            opt_speed_dual=None,
            worst_energy_z=None,
            mean_energy_after_speed=None,
            std_energy_after_speed=None,
            max_cmax_after_speed=None,
            mean_cmax_after_speed=None,
            worst_energy_seed=None,
            binding_time_seed=None,
            slp_iterations=0,
            lp_subproblem_count=0,
            final_delta=0.0,
            infeasible_reason="failed to construct initial robust feasible point",
        )

    x = (start_s1, start_s2, z)

    speed_range = max(speed_max - speed_min, 1e-6)
    delta_speed = SLP_DELTA_INIT_RATIO * speed_range
    delta_z = max(1.0, 0.30 * abs(z))

    slp_iterations = 0
    lp_subproblem_count = 0
    no_improve_count = 0

    best_x = x
    best_stats = (z, mean_e, std_e, max_c, mean_c, worst_seed, binding_seed)

    for _ in range(SLP_MAX_ITER):
        slp_iterations += 1

        constraints = build_slp_linear_constraints(
            features=features,
            seeds=seeds,
            deadline=deadline,
            current_x=x,
            delta_speed=delta_speed,
            delta_z=delta_z,
            speed_min=speed_min,
            speed_max=speed_max,
            lambda_single=lambda_single,
            lambda_dual=lambda_dual,
            rho_single=rho_single,
            rho_dual=rho_dual,
        )

        lp_subproblem_count += 1
        candidate = solve_lp_by_vertex_enumeration(constraints)

        if candidate is None:
            delta_speed *= SLP_DELTA_SHRINK
            delta_z *= SLP_DELTA_SHRINK
            if delta_speed < SLP_DELTA_MIN:
                break
            continue

        cand_s1 = min(speed_max, max(speed_min, candidate[0]))
        cand_s2 = min(speed_max, max(speed_min, candidate[1]))

        feasible, cand_z_true, cand_mean_e, cand_std_e, cand_max_c, cand_mean_c, cand_worst_seed, cand_binding_seed = robust_stats(
            features=features,
            seeds=seeds,
            deadline=deadline,
            speed_single=cand_s1,
            speed_dual=cand_s2,
            lambda_single=lambda_single,
            lambda_dual=lambda_dual,
            rho_single=rho_single,
            rho_dual=rho_dual,
        )

        if not feasible:
            # 线性化点在原问题中不可行，说明步长过大。
            delta_speed *= SLP_DELTA_SHRINK
            delta_z *= SLP_DELTA_SHRINK
            if delta_speed < SLP_DELTA_MIN:
                break
            continue

        old_z = best_stats[0]
        improvement = old_z - cand_z_true
        step_norm = math.hypot(cand_s1 - x[0], cand_s2 - x[1])

        if improvement > SLP_OBJ_TOL * max(1.0, abs(old_z)):
            # 接受候选点，并用原非线性能耗重新修正 z，保证原问题真实可行。
            x = (cand_s1, cand_s2, cand_z_true)
            best_x = x
            best_stats = (
                cand_z_true,
                cand_mean_e,
                cand_std_e,
                cand_max_c,
                cand_mean_c,
                cand_worst_seed,
                cand_binding_seed,
            )

            delta_speed = min(speed_range, delta_speed * SLP_DELTA_EXPAND)
            delta_z = max(1.0, 0.30 * abs(cand_z_true))
            no_improve_count = 0

            if step_norm <= SLP_STEP_TOL:
                break

        else:
            # 没有明显改善，缩小局部近似范围。
            no_improve_count += 1
            delta_speed *= SLP_DELTA_SHRINK
            delta_z *= SLP_DELTA_SHRINK

            if delta_speed < SLP_DELTA_MIN or no_improve_count >= 10:
                break

    z, mean_e, std_e, max_c, mean_c, worst_seed, binding_seed = best_stats

    return RobustOptResult(
        feasible=True,
        required_speed_single=required_single,
        required_speed_dual=required_dual,
        opt_speed_single=best_x[0],
        opt_speed_dual=best_x[1],
        worst_energy_z=z,
        mean_energy_after_speed=mean_e,
        std_energy_after_speed=std_e,
        max_cmax_after_speed=max_c,
        mean_cmax_after_speed=mean_c,
        worst_energy_seed=worst_seed,
        binding_time_seed=binding_seed,
        slp_iterations=slp_iterations,
        lp_subproblem_count=lp_subproblem_count,
        final_delta=delta_speed,
        infeasible_reason="",
    )


# ============================================================
# 鲁棒 KKT 验证
# ============================================================

def robust_constraint_values_and_gradients(
    features: list[ModeFeature],
    seeds: list[int],
    deadline: float,
    speed_single: float,
    speed_dual: float,
    z: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> list[tuple[str, float, tuple[float, float, float]]]:
    """返回所有约束 g_i(x)>=0 及其梯度。

    x=(s_single,s_dual,z)
    """
    constraints: list[tuple[str, float, tuple[float, float, float]]] = []

    constraints.append(("s_single_lower", speed_single - speed_min, (1.0, 0.0, 0.0)))
    constraints.append(("s_single_upper", speed_max - speed_single, (-1.0, 0.0, 0.0)))
    constraints.append(("s_dual_lower", speed_dual - speed_min, (0.0, 1.0, 0.0)))
    constraints.append(("s_dual_upper", speed_max - speed_dual, (0.0, -1.0, 0.0)))

    for feature, seed in zip(features, seeds):
        c = cmax_after_speed(feature, speed_single, speed_dual)
        g_time = deadline - c
        grad_time = (
            feature.single_time / (speed_single * speed_single),
            feature.dual_time / (speed_dual * speed_dual),
            0.0,
        )
        constraints.append((f"time_seed_{seed}", g_time, grad_time))

        e = energy_after_speed(
            feature, speed_single, speed_dual,
            lambda_single, lambda_dual, rho_single, rho_dual,
        )
        de1, de2 = energy_gradient(
            feature, speed_single, speed_dual,
            lambda_single, lambda_dual, rho_single, rho_dual,
        )
        g_energy = z - e
        grad_energy = (-de1, -de2, 1.0)
        constraints.append((f"energy_seed_{seed}", g_energy, grad_energy))

    return constraints


def solve_small_linear_system(
    matrix: list[list[float]],
    rhs: list[float],
) -> list[float] | None:
    """解小型线性方程组，用于 KKT 乘子最小二乘。"""
    n = len(rhs)
    if n == 0:
        return []

    a = [list(map(float, matrix[i])) + [float(rhs[i])] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))

        if abs(a[pivot][col]) <= 1e-12:
            return None

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]

        div = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= div

        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]

    return [a[i][n] for i in range(n)]


def solve_nonnegative_stationarity_multipliers(
    objective_grad: tuple[float, float, float],
    active_grads: list[tuple[str, tuple[float, float, float]]],
) -> tuple[dict[str, float], tuple[float, float, float]]:
    """枚举活跃约束子集，寻找非负 KKT 乘子。

    变量维数为 3，驻点条件最多需要 3 个独立活跃约束即可表达。
    若存在多个活跃约束，枚举 0/1/2/3 个约束组合，用最小残差选择。
    """
    names = [name for name, _g in active_grads]
    best_mu = {name: 0.0 for name in names}
    best_residual = objective_grad
    best_norm = math.sqrt(sum(v * v for v in best_residual))

    candidate_sizes = [0, 1, 2, 3]

    for size in candidate_sizes:
        for combo in itertools.combinations(active_grads, size):
            if size == 0:
                residual = objective_grad
                norm = math.sqrt(sum(v * v for v in residual))
                if norm < best_norm:
                    best_norm = norm
                    best_residual = residual
                    best_mu = {name: 0.0 for name in names}
                continue

            combo_names = [name for name, _g in combo]
            grads = [g for _name, g in combo]

            # 解最小二乘：G mu ≈ objective_grad
            # 正规方程：(G^T G)mu = G^T f
            normal_matrix: list[list[float]] = []
            normal_rhs: list[float] = []

            for i in range(size):
                row = []
                for j in range(size):
                    row.append(
                        grads[i][0] * grads[j][0]
                        + grads[i][1] * grads[j][1]
                        + grads[i][2] * grads[j][2]
                    )
                normal_matrix.append(row)
                normal_rhs.append(
                    grads[i][0] * objective_grad[0]
                    + grads[i][1] * objective_grad[1]
                    + grads[i][2] * objective_grad[2]
                )

            mu_values = solve_small_linear_system(normal_matrix, normal_rhs)
            if mu_values is None:
                continue

            if any(mu < -1e-9 for mu in mu_values):
                continue

            mu_values = [max(0.0, mu) for mu in mu_values]

            approx = [0.0, 0.0, 0.0]
            for mu, g in zip(mu_values, grads):
                approx[0] += mu * g[0]
                approx[1] += mu * g[1]
                approx[2] += mu * g[2]

            residual = (
                objective_grad[0] - approx[0],
                objective_grad[1] - approx[1],
                objective_grad[2] - approx[2],
            )
            norm = math.sqrt(sum(v * v for v in residual))

            if norm < best_norm:
                best_norm = norm
                best_residual = residual
                best_mu = {name: 0.0 for name in names}
                for name, mu in zip(combo_names, mu_values):
                    best_mu[name] = mu

    return best_mu, best_residual


def verify_robust_kkt(
    result: RobustOptResult,
    features: list[ModeFeature],
    seeds: list[int],
    deadline: float,
    speed_min: float,
    speed_max: float,
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> KKTCheck:
    if (
        not result.feasible
        or result.opt_speed_single is None
        or result.opt_speed_dual is None
        or result.worst_energy_z is None
    ):
        return KKTCheck(
            feasible=False,
            primal_min=None,
            dual_min=None,
            stationarity_s_single=None,
            stationarity_s_dual=None,
            stationarity_z=None,
            stationarity_norm=None,
            max_complementarity_error=None,
            active_constraints="",
            active_time_seeds="",
            active_energy_seeds="",
            selected_multipliers="",
            kkt_pass=False,
            kkt_note=result.infeasible_reason,
        )

    s1 = result.opt_speed_single
    s2 = result.opt_speed_dual
    z = result.worst_energy_z

    constraints = robust_constraint_values_and_gradients(
        features=features,
        seeds=seeds,
        deadline=deadline,
        speed_single=s1,
        speed_dual=s2,
        z=z,
        speed_min=speed_min,
        speed_max=speed_max,
        lambda_single=lambda_single,
        lambda_dual=lambda_dual,
        rho_single=rho_single,
        rho_dual=rho_dual,
    )

    speed_tol = 1e-4
    time_tol = max(1e-4, 1e-4 * max(1.0, deadline))
    energy_tol = max(1e-4, 1e-4 * max(1.0, abs(z)))

    active: list[tuple[str, tuple[float, float, float]]] = []
    active_names: list[str] = []
    active_time_seeds: list[str] = []
    active_energy_seeds: list[str] = []

    for name, value, grad in constraints:
        is_active = False

        if name.startswith("s_"):
            is_active = abs(value) <= speed_tol
        elif name.startswith("time_seed_"):
            is_active = abs(value) <= time_tol
        elif name.startswith("energy_seed_"):
            is_active = abs(value) <= energy_tol

        if is_active:
            active.append((name, grad))
            active_names.append(name)

            if name.startswith("time_seed_"):
                active_time_seeds.append(name.replace("time_seed_", ""))

            if name.startswith("energy_seed_"):
                active_energy_seeds.append(name.replace("energy_seed_", ""))

    # 目标函数 f(s1,s2,z)=z，所以 grad f=(0,0,1)。
    objective_grad = (0.0, 0.0, 1.0)
    mu_active, residual = solve_nonnegative_stationarity_multipliers(objective_grad, active)

    mu_all: dict[str, float] = {name: 0.0 for name, _value, _grad in constraints}
    mu_all.update(mu_active)

    primal_min = min(value for _name, value, _grad in constraints)
    dual_min = min(mu_all.values()) if mu_all else 0.0

    max_comp = 0.0
    for name, value, _grad in constraints:
        max_comp = max(max_comp, abs(mu_all.get(name, 0.0) * value))

    stationarity_norm = math.sqrt(sum(v * v for v in residual))

    scale = max(1.0, abs(z))
    kkt_pass = (
        primal_min >= -KKT_TOL * max(1.0, deadline)
        and dual_min >= -KKT_TOL
        and max_comp <= KKT_TOL * scale
        and stationarity_norm <= 0.10
    )

    selected_multipliers = ";".join(
        f"{name}:{mu:.6g}"
        for name, mu in sorted(mu_all.items())
        if abs(mu) > 1e-10
    )

    note = (
        "KKT conditions numerically satisfied for robust SLP solution"
        if kkt_pass
        else "KKT residual is reported for numerical inspection"
    )

    return KKTCheck(
        feasible=True,
        primal_min=primal_min,
        dual_min=dual_min,
        stationarity_s_single=residual[0],
        stationarity_s_dual=residual[1],
        stationarity_z=residual[2],
        stationarity_norm=stationarity_norm,
        max_complementarity_error=max_comp,
        active_constraints="+".join(active_names) if active_names else "none",
        active_time_seeds=",".join(active_time_seeds),
        active_energy_seeds=",".join(active_energy_seeds),
        selected_multipliers=selected_multipliers,
        kkt_pass=kkt_pass,
        kkt_note=note,
    )


# ============================================================
# 2B/3B 鲁棒推荐
# ============================================================

def compare_robust_modes(
    robust_2b: RobustOptResult,
    robust_3b: RobustOptResult,
) -> tuple[str, str, str, float | None, float | None]:
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
    lambda_single: float,
    lambda_dual: float,
    rho_single: float,
    rho_dual: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    kkt_rows: list[dict] = []

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

        features_2b = [
            build_mode_feature(c.cmax_2b, c.energy_2b, n1, n2, n3, n4, "2B")
            for c in cases_sorted
        ]
        features_3b = [
            build_mode_feature(c.cmax_3b, c.energy_3b, n1, n2, n3, n4, "3B")
            for c in cases_sorted
        ]

        for ratio in deadline_ratios:
            deadline = ratio * deadline_reference

            robust_2b = solve_robust_speed_problem_slp(
                features=features_2b,
                seeds=seeds,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                lambda_single=lambda_single,
                lambda_dual=lambda_dual,
                rho_single=rho_single,
                rho_dual=rho_dual,
            )
            robust_3b = solve_robust_speed_problem_slp(
                features=features_3b,
                seeds=seeds,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                lambda_single=lambda_single,
                lambda_dual=lambda_dual,
                rho_single=rho_single,
                rho_dual=rho_dual,
            )

            kkt_2b = verify_robust_kkt(
                result=robust_2b,
                features=features_2b,
                seeds=seeds,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                lambda_single=lambda_single,
                lambda_dual=lambda_dual,
                rho_single=rho_single,
                rho_dual=rho_dual,
            )
            kkt_3b = verify_robust_kkt(
                result=robust_3b,
                features=features_3b,
                seeds=seeds,
                deadline=deadline,
                speed_min=speed_min,
                speed_max=speed_max,
                lambda_single=lambda_single,
                lambda_dual=lambda_dual,
                rho_single=rho_single,
                rho_dual=rho_dual,
            )

            recommended_mode, recommendation, recommendation_reason, gap, gap_percent = compare_robust_modes(
                robust_2b, robust_3b
            )

            common = {
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

                "lambda_single": round_or_blank(lambda_single, 4),
                "lambda_dual": round_or_blank(lambda_dual, 4),
                "rho_single": round_or_blank(rho_single, 4),
                "rho_dual": round_or_blank(rho_dual, 4),
                "speed_min": round_or_blank(speed_min, 4),
                "speed_max": round_or_blank(speed_max, 4),
            }

            rows.append(
                {
                    **common,

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
                    "slp_iterations_2B": robust_2b.slp_iterations,
                    "lp_subproblem_count_2B": robust_2b.lp_subproblem_count,
                    "final_delta_2B": round_or_blank(robust_2b.final_delta, 10),
                    "infeasible_reason_2B": robust_2b.infeasible_reason,
                    "kkt_pass_2B": int(kkt_2b.kkt_pass),

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
                    "slp_iterations_3B": robust_3b.slp_iterations,
                    "lp_subproblem_count_3B": robust_3b.lp_subproblem_count,
                    "final_delta_3B": round_or_blank(robust_3b.final_delta, 10),
                    "infeasible_reason_3B": robust_3b.infeasible_reason,
                    "kkt_pass_3B": int(kkt_3b.kkt_pass),

                    "worst_energy_gap_2B_minus_3B": round_or_blank(gap, 6),
                    "worst_energy_gap_percent_vs_2B": round_or_blank(gap_percent, 4),
                    "recommended_mode": recommended_mode,
                    "recommendation": recommendation,
                    "recommendation_reason": recommendation_reason,

                    "robust_model_formula": (
                        "min z, s.t. E_j(s_single,s_dual)<=z and "
                        "C_j(s_single,s_dual)<=D for all seeds"
                    ),
                    "energy_formula": (
                        "E_k(s)=E_k0*(lambda_k/s+(1-lambda_k)*((1-rho_k)+rho_k*s^2))"
                    ),
                    "course_method": (
                        "SLP_sequential_linear_programming + linear_programming_subproblem + KKT_verification"
                    ),
                    "model_meaning": (
                        "choose one pair of single/dual speeds for all seed perturbations and minimize worst-case energy"
                    ),
                }
            )

            for mode_name, robust, kkt in [
                ("2B", robust_2b, kkt_2b),
                ("3B", robust_3b, kkt_3b),
            ]:
                kkt_rows.append(
                    {
                        **common,
                        "mode": mode_name,

                        "feasible": int(kkt.feasible),
                        "opt_speed_single": round_or_blank(robust.opt_speed_single, 6),
                        "opt_speed_dual": round_or_blank(robust.opt_speed_dual, 6),
                        "worst_energy_z": round_or_blank(robust.worst_energy_z, 6),
                        "max_cmax_after_speed": round_or_blank(robust.max_cmax_after_speed, 6),
                        "worst_energy_seed": robust.worst_energy_seed if robust.worst_energy_seed is not None else "",
                        "binding_time_seed": robust.binding_time_seed if robust.binding_time_seed is not None else "",

                        "primal_min": round_or_blank(kkt.primal_min, 10),
                        "dual_min": round_or_blank(kkt.dual_min, 10),
                        "stationarity_s_single": round_or_blank(kkt.stationarity_s_single, 10),
                        "stationarity_s_dual": round_or_blank(kkt.stationarity_s_dual, 10),
                        "stationarity_z": round_or_blank(kkt.stationarity_z, 10),
                        "stationarity_norm": round_or_blank(kkt.stationarity_norm, 10),
                        "max_complementarity_error": round_or_blank(kkt.max_complementarity_error, 10),

                        "active_constraints": kkt.active_constraints,
                        "active_time_seeds": kkt.active_time_seeds,
                        "active_energy_seeds": kkt.active_energy_seeds,
                        "selected_multipliers": kkt.selected_multipliers,

                        "kkt_pass": int(kkt.kkt_pass),
                        "kkt_note": kkt.kkt_note,
                    }
                )

    return (
        sorted(rows, key=lambda r: (r["counts_code"], safe_float(r["deadline_ratio"]))),
        sorted(kkt_rows, key=lambda r: (r["counts_code"], safe_float(r["deadline_ratio"]), r["mode"])),
    )


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust two-variable U-shaped nonlinear speed-energy optimization using SLP and KKT verification."
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

    args = parser.parse_args()

    method = args.method.strip()

    input_path = (
        Path(args.input)
        if args.input.strip()
        else OUTPUT_DIR / "four_case_framework" / f"{method}_2B3B_time_energy.csv"
    )

    output_dir = OUTPUT_DIR / "nonlinear_programming" / "robust_mode_selection"
    output_path = output_dir / "robust_speed_mode_selection.csv"
    kkt_path = output_dir / "robust_kkt_verification.csv"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)

    seed_cases = build_seed_cases(raw_rows=raw_rows, method=method)
    grouped_cases = group_seed_cases(seed_cases)
    deadline_ratios = parse_float_list(args.deadline_ratios)

    rows, kkt_rows = build_output_rows(
        grouped_cases=grouped_cases,
        method=method,
        deadline_ratios=deadline_ratios,
        deadline_base=args.deadline_base,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        lambda_single=args.lambda_single,
        lambda_dual=args.lambda_dual,
        rho_single=args.rho_single,
        rho_dual=args.rho_dual,
    )

    if not rows:
        print("Warning: no rows generated. Please check input CSV columns and method suffix.")
        return

    save_csv(rows, output_path, ROBUST_RESULT_FIELDNAMES)

    if kkt_rows:
        save_csv(kkt_rows, kkt_path, ROBUST_KKT_FIELDNAMES)

    print("Robust SLP nonlinear speed-energy optimization finished.")
    print(f"Input: {input_path}")
    print(f"Output result: {output_path}")
    print(f"Output KKT: {kkt_path}")
    print(f"Grouped counts_code cases: {len(grouped_cases)}")
    print(f"Rows: {len(rows)}")
    print("Method: SLP sequential linear programming + linear programming subproblem + KKT verification")
    print("Model: min z with single-arm speed and dual-arm speed under U-shaped economic speed energy.")


if __name__ == "__main__":
    main()
