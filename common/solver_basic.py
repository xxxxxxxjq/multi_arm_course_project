# -*- coding: utf-8 -*-
"""基础方法调度优化模型。

这是整个工程最核心的文件。
你可以把它理解成：把“多机械臂搬运方块”翻译成整数约束优化模型，
然后调用通用优化求解器求解。

模型考虑了 5 件事：
1. 每个任务必须且只能选择一种执行模式；
2. 同一只机械臂同一时间不能执行两个任务；
3. 任务前的空载转移时间和能耗取决于任务顺序；
4. 如果任务组经过中心安全区，则对不同任务组加中心区互斥约束；
5. 采用序贯目标规划：先时间，再能耗，再负载均衡。

"""

from __future__ import annotations

from dataclasses import asdict
from math import ceil, sqrt
from typing import Dict, List, Tuple

try:
    # cp_model 是 OR-Tools 里的整数约束建模接口。
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 OR-Tools。请先运行：pip install ortools") from exc

from common.config import (
    ALPHA_EMPTY,
    CENTER_ZONE_RADIUS,
    ENERGY_SCALE,
    ENFORCE_CENTER_ZONE_EXCLUSION,
    NUM_WORKERS,
    SOLVER_TIME_LIMIT,
    SYSTEM_OVERHEAD_PER_ARM,
    TIME_SCALE,
    USE_CENTER_ZONE,
    V_EMPTY,
    V_LOADED_DUAL,
    V_LOADED_SINGLE,
)
from common.geometry import Instance, Mode, Point, Task, distance


class ScheduleResult(dict):
    """求解结果字典。

    目前它只是 dict 的语义包装。
    这样写的好处是：以后如果想给结果对象增加方法，不需要大改其他代码。
    """


# 第三阶段是最低优先级目标：负载均衡。
# 为避免批量实验因为第三阶段在时间限制内找不到解而中断，
# 第三阶段允许在第二阶段最优能耗附近有少量容差。
# 如果第三阶段仍未找到可行解，则回退到第二阶段结果。
STAGE3_ENERGY_TOLERANCE = 10
STAGE3_D_ENERGY_TOLERANCE = 10


def _to_ticks(seconds: float) -> int:
    """把秒转换为整数 tick。"""
    return max(1, int(round(seconds * TIME_SCALE)))


def _to_energy(value: float) -> int:
    """把连续能耗值转换为整数。"""
    return max(1, int(round(value * ENERGY_SCALE)))


def _setup_ticks(p_from: Point, p_to: Point) -> int:
    """计算空载转移时间。

    空载转移指：机械臂没有拿方块时，从一个位置移动到另一个位置。
    例如：standby -> 当前方块，或 上一个目标盒 -> 下一个方块。
    """
    return _to_ticks(distance(p_from, p_to) / V_EMPTY)


def _setup_energy(p_from: Point, p_to: Point) -> int:
    """计算空载转移能耗。"""
    return _to_energy(ALPHA_EMPTY * distance(p_from, p_to))


def _segment_distance_to_origin(a: Point, b: Point) -> float:
    """计算原点到线段 ab 的最短距离。

    这个函数用于判断一段运动路径是否靠近中心危险区。
    中心危险区默认以原点为圆心。
    """
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy

    # 线段长度为 0 时，退化成点到原点距离。
    if length_sq == 0:
        return distance((0.0, 0.0), a)

    # 计算原点在线段方向上的投影参数。
    t = -(ax * vx + ay * vy) / length_sq
    t = max(0.0, min(1.0, t))
    px, py = ax + t * vx, ay + t * vy
    return distance((0.0, 0.0), (px, py))


def _segment_crosses_center_zone(a: Point, b: Point) -> bool:
    """判断线段 ab 是否穿过中心危险区。"""
    if not USE_CENTER_ZONE:
        return False
    return _segment_distance_to_origin(a, b) <= CENTER_ZONE_RADIUS


def _segment_center_window(a: Point, b: Point, travel_ticks: int) -> tuple[int, int] | None:
    """计算线段匀速运动时穿越中心危险区的局部时间窗。

    返回：
        (offset, duration)

    含义：
    - offset：从这段运动开始算起，过多少 tick 进入中心危险区；
    - duration：在中心危险区内持续多少 tick。

    如果不穿越中心危险区，返回 None。

    为什么要算局部时间窗？
    如果直接把整段任务都视为占用中心区，会过度保守，导致很多本来可以并行的任务被迫串行。
    现在只锁定真正进入中心区的那一小段时间，更符合“局部安全约束”。
    """
    if not USE_CENTER_ZONE or travel_ticks <= 0:
        return None

    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay

    # 路径点可以写成 P(t)=a+t(b-a)，t 从 0 到 1。
    # 判断 P(t) 是否在中心圆内，就是解一个二次不等式。
    A = vx * vx + vy * vy
    if A == 0:
        if distance((0.0, 0.0), a) <= CENTER_ZONE_RADIUS:
            return 0, max(1, travel_ticks)
        return None

    B = 2.0 * (ax * vx + ay * vy)
    C = ax * ax + ay * ay - CENTER_ZONE_RADIUS * CENTER_ZONE_RADIUS
    disc = B * B - 4.0 * A * C

    # 判断起点/终点是否已经在中心圆内部。
    inside_start = C <= 0.0
    inside_end = (bx * bx + by * by) <= CENTER_ZONE_RADIUS * CENTER_ZONE_RADIUS

    # 判别式小于 0：线段所在直线不与圆相交。
    if disc < 0:
        if inside_start and inside_end:
            return 0, max(1, travel_ticks)
        return None

    # 求出进入/离开中心圆的两个 t 值。
    root = sqrt(max(0.0, disc))
    t1 = (-B - root) / (2.0 * A)
    t2 = (-B + root) / (2.0 * A)
    lo = max(0.0, min(t1, t2))
    hi = min(1.0, max(t1, t2))

    # hi <= lo 表示在线段范围内没有有效穿越区间。
    if hi <= lo:
        return None

    # 把连续比例 t 转换成整数 tick。
    offset = int(round(lo * travel_ticks))
    duration = max(1, int(round((hi - lo) * travel_ticks)))
    if offset >= travel_ticks:
        return None
    duration = min(duration, travel_ticks - offset)
    return offset, duration


def _loaded_travel_ticks(task: Task, mode: Mode) -> int:
    """计算处理段中真正的负载移动时间。

    mode.duration 包含服务时间和安全时间；
    这里仅计算 pick -> target 的移动时间，用来定位中心区时间窗。
    """
    speed = V_LOADED_DUAL if len(mode.arms) > 1 else V_LOADED_SINGLE
    return _to_ticks(distance(task.pos, task.target_pos) / speed)


def _add_center_window_before_end(
    model,
    horizon: int,
    center_zone_intervals: list,
    end_var,
    presence,
    segment_ticks: int,
    offset: int,
    duration: int,
    name: str,
) -> None:
    """给“在 end_var 前完成的一段运动”添加中心区占用 interval。

    典型场景：空载 setup 运动在任务开始时刻之前完成。
    如果某段 setup 总时长是 segment_ticks，任务开始时刻是 end_var，
    那么 setup 段开始时刻就是 end_var - segment_ticks。
    中心区窗口开始时刻 = setup 段开始时刻 + offset。
    """
    c_start = model.NewIntVar(0, horizon, f"center_start_{name}")
    c_end = model.NewIntVar(0, horizon, f"center_end_{name}")
    model.Add(c_start == end_var - segment_ticks + offset).OnlyEnforceIf(presence)
    model.Add(c_end == c_start + duration).OnlyEnforceIf(presence)
    center_zone_intervals.append(
        model.NewOptionalIntervalVar(c_start, duration, c_end, presence, f"center_window_{name}")
    )


def _add_center_window_after_start(
    model,
    horizon: int,
    center_zone_intervals: list,
    start_var,
    presence,
    offset: int,
    duration: int,
    name: str,
) -> None:
    """给“从 start_var 开始的一段运动”添加中心区占用 interval。"""
    c_start = model.NewIntVar(0, horizon, f"center_start_{name}")
    c_end = model.NewIntVar(0, horizon, f"center_end_{name}")
    model.Add(c_start == start_var + offset).OnlyEnforceIf(presence)
    model.Add(c_end == c_start + duration).OnlyEnforceIf(presence)
    center_zone_intervals.append(
        model.NewOptionalIntervalVar(c_start, duration, c_end, presence, f"center_window_{name}")
    )


def _compute_horizon(instance: Instance, modes_by_task: Dict[int, List[Mode]]) -> int:
    """计算时间变量的保守上界 horizon。

    start/end 变量需要一个最大取值范围。
    这个范围不能太小，否则可行解会被排除；也不能太离谱，否则求解变慢。

    这里采用保守估计：
    所有任务最长处理时间之和 + 每个任务一次最坏空载转移 + 额外缓冲。
    """
    process_bound = sum(max(m.duration for m in modes) for modes in modes_by_task.values())

    # 收集所有可能出现在转移中的点：机械臂基座、方块点、目标盒点。
    points = [arm.pos for arm in instance.arms.values()]
    for task in instance.tasks:
        points.append(task.pos)
        points.append(task.target_pos)

    # 找到任意两点之间的最大 setup 时间。
    max_setup = 1
    for a in points:
        for b in points:
            max_setup = max(max_setup, _setup_ticks(a, b))

    return process_bound + max_setup * max(1, len(instance.tasks)) + 100


def _compute_goal_targets(instance: Instance, modes_by_task: Dict[int, List[Mode]]) -> dict:
    """自动设置目标规划中的理想目标值。

    目标规划形式：
    - 最大完工时间 + d1- - d1+ = time_goal；
    - 总能耗 + d2- - d2+ = energy_goal；
    - 负载差 + d3- - d3+ = balance_goal。

    我们主要惩罚 d+，也就是“超过理想目标”的正偏差。
    """
    # 每个任务选择最短处理时间，得到理论处理时间下界。
    min_process = [min(m.duration for m in modes) for modes in modes_by_task.values()]

    # 每个任务选择最低处理能耗，得到理论处理能耗下界。
    min_energies = [min(m.energy for m in modes) for modes in modes_by_task.values()]
    n_arms = max(1, len(instance.arms))

    min_initial_setups = []
    min_initial_energies = []
    task_by_id = {t.task_id: t for t in instance.tasks}

    # 对每个任务估计一次“从某只可用机械臂基座到方块”的最短初始接近成本。
    for modes in modes_by_task.values():
        vals_t = []
        vals_e = []
        for mode in modes:
            task = task_by_id[mode.task_id]
            for arm_name in mode.arms:
                arm = instance.arms[arm_name]
                vals_t.append(_setup_ticks(arm.pos, task.pos))
                vals_e.append(_setup_energy(arm.pos, task.pos))
        min_initial_setups.append(min(vals_t) if vals_t else 0)
        min_initial_energies.append(min(vals_e) if vals_e else 0)

    # time_goal 同时考虑单个任务下界和平均分配到多臂后的下界。
    min_total_times = [p + s for p, s in zip(min_process, min_initial_setups)]
    time_goal = max(max(min_total_times), ceil(sum(min_total_times) / n_arms))

    # energy_goal 是理想最低能耗估计。
    # mode.energy 已经由 mode_builder.py 计算，包含物块类型相关的负载、保持和协同能耗。
    # 此处再加入系统级固定开销：二臂系统计 2 份，三臂系统计 3 份。
    # 该开销代表机械臂上电、启动、伺服保持、控制通信等静态/固定消耗。
    system_overhead_energy = int(SYSTEM_OVERHEAD_PER_ARM * len(instance.arms))
    energy_goal = sum(min_energies) + sum(min_initial_energies) + system_overhead_energy

    # balance_goal=0 表示理想情况下多臂负载完全相等。
    balance_goal = 0

    return {
        "time_goal": int(time_goal),
        "energy_goal": int(energy_goal),
        "balance_goal": int(balance_goal),
    }


def _build_model(instance: Instance, modes_by_task: Dict[int, List[Mode]]):
    """建立 基础方法 模型，但不求解。

    返回：
    - model：OR-Tools CpModel 对象；
    - data：后续求解和提取结果时要用到的变量字典。
    """
    model = cp_model.CpModel()
    horizon = _compute_horizon(instance, modes_by_task)
    goals = _compute_goal_targets(instance, modes_by_task)

    # 下面几个字典用于保存每个 mode 对应的 基础方法 变量。
    all_presences = {}  # mode 是否被选择，BoolVar。
    all_starts = {}     # mode 处理段开始时间，IntVar。
    all_ends = {}       # mode 处理段结束时间，IntVar。
    all_intervals = {}  # mode 对应的 OptionalIntervalVar。

    task_by_id: Dict[int, Task] = {t.task_id: t for t in instance.tasks}
    mode_lookup: Dict[str, Mode] = {}
    for modes in modes_by_task.values():
        for m in modes:
            mode_lookup[m.mode_id] = m

    # 每只机械臂有哪些 interval，用于 NoOverlap 约束。
    arm_intervals = {arm_name: [] for arm_name in instance.arms.keys()}

    # 中心危险区 interval 列表，最后统一 AddNoOverlap。
    center_zone_intervals = []

    # 空载转移能耗项，最后加到 total_energy 中。
    # 注意：物块类型相关的负载/保持/协同能耗已经体现在 mode.energy 中。
    transition_energy_terms = []

    # 每只机械臂的负载项，包含处理时间和空载转移时间。
    load_terms = {arm: [] for arm in instance.arms.keys()}

    # 记录每个 mode 的实际前置 setup 来自哪里，方便求解后输出解释。
    incoming_setup_info: Dict[Tuple[str, str], List[tuple]] = {}

    # ============================================================
    # A. 为每个任务模式创建变量，并约束每个任务只能选一种模式
    # ============================================================
    for task in instance.tasks:
        presences_for_task = []

        for mode in modes_by_task[task.task_id]:
            # p=1 表示选择这个执行模式，p=0 表示不选择。
            p = model.NewBoolVar(f"choose_{mode.mode_id}")

            # s/e 表示该模式处理段的开始/结束时间。
            s = model.NewIntVar(0, horizon, f"start_process_{mode.mode_id}")
            e = model.NewIntVar(0, horizon, f"end_process_{mode.mode_id}")

            # OptionalIntervalVar：只有 p=1 时，这段处理任务才真正存在。
            interval = model.NewOptionalIntervalVar(
                s,
                mode.duration,
                e,
                p,
                f"process_{mode.mode_id}",
            )

            all_presences[mode.mode_id] = p
            all_starts[mode.mode_id] = s
            all_ends[mode.mode_id] = e
            all_intervals[mode.mode_id] = interval
            presences_for_task.append(p)
            mode_lookup[mode.mode_id] = mode

            # 这个 mode 占用哪些机械臂，就把 interval 加到哪些机械臂的资源列表中。
            for arm_name in mode.arms:
                arm_intervals[arm_name].append(interval)
                load_terms[arm_name].append(mode.duration * p)

            # 中心安全区互斥约束：采用“任务组级”资源建模。
            #
            # 关键点：中心区不是“单个机械臂只能进一个”的资源，而是
            # “同一时刻最多允许一个任务组占用”的共享安全资源。
            # 因此，对于 Type3/Type4 这类双臂协同任务，参与同一任务的两只机械臂
            # 可以同时进入中心区；但其他任务组必须等待该任务组释放中心区后才能进入。
            #
            # 第一版采用保守处理：只要 mode 的 loaded path 经过中心区，就把整个任务处理段
            # [start, end] 视为占用中心区。这样比精细的“进入/离开中心区时间窗”更保守，
            # 但不会把同一个双臂协同任务内部的两只机械臂错误地判为冲突。
            if ENFORCE_CENTER_ZONE_EXCLUSION and mode.uses_center_zone:
                safe_name = mode.mode_id.replace("+", "_")
                center_zone_intervals.append(
                    model.NewOptionalIntervalVar(
                        s,
                        mode.duration,
                        e,
                        p,
                        f"center_task_group_{safe_name}",
                    )
                )

        # 每个任务必须选择且只能选择一个模式。
        model.AddExactlyOne(presences_for_task)

    # ============================================================
    # B. 对每只机械臂建立 circuit，决定任务执行顺序
    # ============================================================
    for arm_name, arm in instance.arms.items():
        # 找出所有可能占用当前机械臂的 mode。
        modes_for_arm = [m for m in mode_lookup.values() if arm_name in m.arms]

        # circuit 需要节点编号。0 号节点作为 depot，表示待机位。
        node_by_mode = {m.mode_id: idx + 1 for idx, m in enumerate(modes_for_arm)}

        arcs = []

        # 如果该机械臂没有被安排任何任务，就允许 0->0 自环。
        selected_count = sum(all_presences[m.mode_id] for m in modes_for_arm)
        empty_arm = model.NewBoolVar(f"empty_sequence_{arm_name}")
        model.Add(selected_count == 0).OnlyEnforceIf(empty_arm)
        model.Add(selected_count >= 1).OnlyEnforceIf(empty_arm.Not())
        arcs.append((0, 0, empty_arm))

        # 未被选择的 mode 需要自环跳过；被选择的 mode 必须进入路径。
        for mode in modes_for_arm:
            node = node_by_mode[mode.mode_id]
            p = all_presences[mode.mode_id]
            skip = model.NewBoolVar(f"skip_{arm_name}_{mode.mode_id}")
            model.Add(skip + p == 1)
            arcs.append((node, node, skip))

        # depot -> first task：当前机械臂的第一个任务从基座/待机位出发。
        for mode in modes_for_arm:
            node = node_by_mode[mode.mode_id]
            p = all_presences[mode.mode_id]
            task = task_by_id[mode.task_id]

            first_arc = model.NewBoolVar(f"arc_{arm_name}_depot_to_{mode.mode_id}")
            last_arc = model.NewBoolVar(f"arc_{arm_name}_{mode.mode_id}_to_depot")
            arcs.append((0, node, first_arc))
            arcs.append((node, 0, last_arc))

            # 如果某条弧被选中，那么对应 mode 必须存在。
            model.AddImplication(first_arc, p)
            model.AddImplication(last_arc, p)

            # 初始 setup：机械臂基座 -> 当前任务 pick 点。
            setup = _setup_ticks(arm.pos, task.pos)
            setup_energy = _setup_energy(arm.pos, task.pos)

            # 如果这个任务是该机械臂的第一个任务，则任务开始时间必须晚于初始 setup。
            model.Add(all_starts[mode.mode_id] >= setup).OnlyEnforceIf(first_arc)
            transition_energy_terms.append(setup_energy * first_arc)
            load_terms[arm_name].append(setup * first_arc)
            incoming_setup_info.setdefault((arm_name, mode.mode_id), []).append((first_arc, setup, "standby", "depot"))

            # 不再对初始 setup 段单独添加中心区互斥 interval。
            # 原因：双臂协同任务可能需要两只机械臂同时从不同方向接近中心区，
            # 如果按“机械臂级 setup 段”互斥，会把同一任务组内部的两只机械臂误判为冲突。
            # 当前中心区约束统一在任务处理段上按“任务组级”建模。

        # task i -> task j：如果当前机械臂先做 pred，再做 succ，就加入序列相关 setup。
        for pred in modes_for_arm:
            pred_node = node_by_mode[pred.mode_id]
            pred_task = task_by_id[pred.task_id]

            for succ in modes_for_arm:
                if pred.mode_id == succ.mode_id:
                    continue

                succ_node = node_by_mode[succ.mode_id]
                succ_task = task_by_id[succ.task_id]

                arc = model.NewBoolVar(f"arc_{arm_name}_{pred.mode_id}_to_{succ.mode_id}")
                arcs.append((pred_node, succ_node, arc))

                # 只有 pred 和 succ 都被选择，这条顺序弧才有可能被选择。
                model.AddImplication(arc, all_presences[pred.mode_id])
                model.AddImplication(arc, all_presences[succ.mode_id])

                # 序列相关 setup：从 pred 的目标盒移动到 succ 的 pick 点。
                setup = _setup_ticks(pred_task.target_pos, succ_task.pos)
                setup_energy = _setup_energy(pred_task.target_pos, succ_task.pos)

                # 如果选择 pred -> succ，则 succ 开始时间 >= pred 结束时间 + setup。
                model.Add(all_starts[succ.mode_id] >= all_ends[pred.mode_id] + setup).OnlyEnforceIf(arc)
                transition_energy_terms.append(setup_energy * arc)
                load_terms[arm_name].append(setup * arc)
                incoming_setup_info.setdefault((arm_name, succ.mode_id), []).append((arc, setup, pred.mode_id, "mode"))

                # 不再对后续空载 setup 段单独添加中心区互斥 interval。
                # 当前中心区约束统一在任务处理段上按“任务组级”建模，
                # 避免同一个双臂协同任务的两只机械臂在进入中心区时互相冲突。

        # AddCircuit 负责保证当前机械臂形成一条合法任务序列。
        model.AddCircuit(arcs)

    # 处理段 NoOverlap：同一机械臂不能同时执行两个处理任务。
    for _arm_name, intervals in arm_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    # 中心安全区 NoOverlap：任务组级共享资源。
    # 含义：不同任务组不能同时占用中心区；
    # 但同一个双臂协同任务内部的两只机械臂允许同时进入中心区。
    if ENFORCE_CENTER_ZONE_EXCLUSION and center_zone_intervals:
        model.AddNoOverlap(center_zone_intervals)

    # ============================================================
    # C. 指标变量：最大完工时间、总能耗、负载差、目标规划偏差变量
    # ============================================================

    # 最大完工时间是所有已选任务结束时间的最大值。
    cmax = model.NewIntVar(0, horizon, "maximum_completion_time")
    for modes in modes_by_task.values():
        for mode in modes:
            model.Add(cmax >= all_ends[mode.mode_id]).OnlyEnforceIf(all_presences[mode.mode_id])

    # 运动能耗 = 处理能耗 + 空载转移能耗。
    # 系统总能耗 = 运动能耗 + 系统级固定开销。
    #
    # 为什么要加入系统级固定开销？
    # 原始运动能耗只反映任务路径和搬运动作。三臂系统由于多了一只机械臂，
    # 即使路径更短，也会产生额外的启动、上电、伺服保持和控制通信消耗。
    # 因此这里按启用机械臂数量加入固定能耗项，使能耗定义更接近实际系统。
    process_energy_terms = []
    for modes in modes_by_task.values():
        for mode in modes:
            p = all_presences[mode.mode_id]
            process_energy_terms.append(mode.energy * p)

    motion_energy = model.NewIntVar(0, 10**9, "motion_energy")
    model.Add(motion_energy == sum(process_energy_terms + transition_energy_terms))

    system_overhead_energy = int(SYSTEM_OVERHEAD_PER_ARM * len(instance.arms))

    total_energy = model.NewIntVar(0, 10**9, "total_energy")
    model.Add(total_energy == motion_energy + system_overhead_energy)

    # 每只机械臂的负载 = 处理时间 + 空载转移时间。
    arm_load_vars = {}
    for arm_name, terms in load_terms.items():
        v = model.NewIntVar(0, horizon, f"load_{arm_name}")
        if terms:
            model.Add(v == sum(terms))
        else:
            model.Add(v == 0)
        arm_load_vars[arm_name] = v

    # 负载差 = 最大负载 - 最小负载。
    loads = list(arm_load_vars.values())
    load_max = model.NewIntVar(0, horizon, "LoadMax")
    load_min = model.NewIntVar(0, horizon, "LoadMin")
    model.AddMaxEquality(load_max, loads)
    model.AddMinEquality(load_min, loads)
    load_imbalance = model.NewIntVar(0, horizon, "load_imbalance")
    model.Add(load_imbalance == load_max - load_min)

    # 目标规划偏差变量：d_plus 表示超过目标，d_minus 表示低于目标。
    d_time_plus = model.NewIntVar(0, horizon, "d1_plus_time_over")
    d_time_minus = model.NewIntVar(0, horizon, "d1_minus_time_under")
    model.Add(cmax + d_time_minus - d_time_plus == goals["time_goal"])

    d_energy_plus = model.NewIntVar(0, 10**9, "d2_plus_energy_over")
    d_energy_minus = model.NewIntVar(0, 10**9, "d2_minus_energy_under")
    model.Add(total_energy + d_energy_minus - d_energy_plus == goals["energy_goal"])

    d_balance_plus = model.NewIntVar(0, horizon, "d3_plus_balance_over")
    d_balance_minus = model.NewIntVar(0, horizon, "d3_minus_balance_under")
    model.Add(load_imbalance + d_balance_minus - d_balance_plus == goals["balance_goal"])

    # 把后续求解/提取结果需要的变量统一打包返回。
    data = {
        "horizon": horizon,
        "goals": goals,
        "presence": all_presences,
        "starts": all_starts,
        "ends": all_ends,
        "cmax": cmax,
        "motion_energy": motion_energy,
        "system_overhead_energy": system_overhead_energy,
        "total_energy": total_energy,
        "arm_loads": arm_load_vars,
        "load_imbalance": load_imbalance,
        "d_time_plus": d_time_plus,
        "d_time_minus": d_time_minus,
        "d_energy_plus": d_energy_plus,
        "d_energy_minus": d_energy_minus,
        "d_balance_plus": d_balance_plus,
        "d_balance_minus": d_balance_minus,
        "incoming_setup_info": incoming_setup_info,
    }
    return model, data


def _new_solver() -> cp_model.CpSolver:
    """创建一个 基础方法 求解器，并设置求解参数。"""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
    solver.parameters.num_search_workers = NUM_WORKERS
    return solver


def _is_good_status(status: int) -> bool:
    """判断求解器是否给出了可用解。"""
    return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def _check_status(status: int, message: str) -> None:
    """检查求解状态，不可行时抛出错误。

    第一阶段必须找到可行解，因为它代表任务本身在当前约束下是否能完成。
    第二、第三阶段属于更高要求的目标规划优化，后面会采用回退策略。
    """
    if not _is_good_status(status):
        raise RuntimeError(message)


def _add_solution_hint(model: cp_model.CpModel, data: dict, solver: cp_model.CpSolver) -> None:
    """把上一阶段的解作为下一阶段提示。

    CP-SAT 第三阶段有时并不是真的无解，而是在时间限制内没有重新找到
    同时满足前两级目标的解。把第二阶段解作为 hint，可以帮助第三阶段
    更快找到可行起点。
    """
    hinted = []
    hinted.extend(data["presence"].values())
    hinted.extend(data["starts"].values())
    hinted.extend(data["ends"].values())
    hinted.append(data["cmax"])
    hinted.append(data["motion_energy"])
    hinted.append(data["total_energy"])
    hinted.extend(data["arm_loads"].values())
    hinted.append(data["load_imbalance"])
    hinted.append(data["d_time_plus"])
    hinted.append(data["d_time_minus"])
    hinted.append(data["d_energy_plus"])
    hinted.append(data["d_energy_minus"])
    hinted.append(data["d_balance_plus"])
    hinted.append(data["d_balance_minus"])

    for var in hinted:
        try:
            model.AddHint(var, solver.Value(var))
        except Exception:
            pass


def solve_schedule(instance: Instance, modes_by_task: Dict[int, List[Mode]]) -> ScheduleResult:
    """序贯目标规划求解。

    本工程不用贪心算法，也不用遗传算法、模拟退火等启发式算法。
    这里使用 基础方法 整数规划，并按优先级分三阶段求解：
    1. P1：优先最小化时间目标正偏差 d1+，并尽量减小最大完工时间；
    2. P2：在不破坏 P1 结果的前提下，最小化能耗目标正偏差 d2+ 和总能耗；
    3. P3：在不破坏 P1/P2 结果的前提下，最小化负载均衡目标正偏差 d3+ 和负载差。
    """
    model, data = _build_model(instance, modes_by_task)

    # ------------------------------
    # 第一阶段：时间优先
    # ------------------------------
    model.Minimize(data["d_time_plus"] * 100000 + data["cmax"])
    solver1 = _new_solver()
    status1 = solver1.Solve(model)
    _check_status(status1, "基础方法 第一阶段未找到可行解。")
    best_d1 = solver1.Value(data["d_time_plus"])
    best_cmax = solver1.Value(data["cmax"])

    # ------------------------------
    # 第二阶段：固定第一阶段结果，再优化能耗
    # ------------------------------
    model.Add(data["d_time_plus"] == best_d1)
    model.Add(data["cmax"] == best_cmax)
    model.Minimize(data["d_energy_plus"] * 100000 + data["total_energy"])
    _add_solution_hint(model, data, solver1)
    solver2 = _new_solver()
    status2 = solver2.Solve(model)

    # 第二阶段理论上至少可以沿用第一阶段解，但在复杂场景下可能因时间限制返回 UNKNOWN。
    # 为了避免批量实验中断，如果第二阶段未找到可用解，则回退到第一阶段结果。
    if _is_good_status(status2):
        best_d2 = solver2.Value(data["d_energy_plus"])
        best_energy = solver2.Value(data["total_energy"])
        stage2_used = True
    else:
        best_d2 = solver1.Value(data["d_energy_plus"])
        best_energy = solver1.Value(data["total_energy"])
        stage2_used = False

    # ------------------------------
    # 第三阶段：在不破坏前两级主要目标的前提下优化负载均衡
    # ------------------------------
    # 这里不再用完全等式硬卡死第二阶段能耗，而是允许极小容差。
    # 这样更符合工程求解：第三阶段是最低优先级目标，不能因为它搜索困难
    # 就否定前两阶段已经得到的可行调度。
    if stage2_used:
        model.Add(data["d_energy_plus"] <= best_d2 + STAGE3_D_ENERGY_TOLERANCE)
        model.Add(data["total_energy"] <= best_energy + STAGE3_ENERGY_TOLERANCE)
        model.Minimize(data["d_balance_plus"] * 100000 + data["load_imbalance"])
        _add_solution_hint(model, data, solver2)
        solver3 = _new_solver()
        status3 = solver3.Solve(model)
    else:
        solver3 = None
        status3 = cp_model.UNKNOWN

    # 回退策略：
    # - 第三阶段成功：采用第三阶段结果；
    # - 第三阶段失败：采用第二阶段结果；
    # - 第二阶段也失败：采用第一阶段结果。
    if solver3 is not None and _is_good_status(status3):
        final_solver = solver3
        final_status = status3
        final_stage = "P3_load_balance"
        fallback_reason = ""
    elif stage2_used:
        final_solver = solver2
        final_status = status2
        final_stage = "P2_energy_fallback"
        fallback_reason = "第三阶段未在限制时间内找到满足前两级目标约束的解，已回退至第二阶段结果。"
    else:
        final_solver = solver1
        final_status = status1
        final_stage = "P1_time_fallback"
        fallback_reason = "第二、第三阶段未在限制时间内找到可用解，已回退至第一阶段结果。"

    # ============================================================
    # D. 从求解器中提取最终调度结果
    # ============================================================
    selected = []
    task_by_id = {t.task_id: t for t in instance.tasks}

    # 建立 mode_id -> Mode 对象映射，方便后续根据 mode_id 查原始信息。
    mode_lookup: Dict[str, Mode] = {}
    for modes in modes_by_task.values():
        for m in modes:
            mode_lookup[m.mode_id] = m

    # 提取每个已选 mode 在各机械臂上的实际前置 setup。
    setup_by_mode: Dict[str, List[dict]] = {m.mode_id: [] for m in mode_lookup.values()}
    for (arm_name, mode_id), setup_items in data["incoming_setup_info"].items():
        for arc, setup, pred, pred_type in setup_items:
            # 有些变量可能在不同阶段模型中被固定，try 是为了提高鲁棒性。
            try:
                if final_solver.Value(arc):
                    setup_by_mode.setdefault(mode_id, []).append(
                        {
                            "arm": arm_name,
                            "setup": int(setup),
                            "from": pred,
                            "from_type": pred_type,
                        }
                    )
            except Exception:
                pass

    # 遍历所有模式，找出最终被选择的模式。
    for mode_id, p in data["presence"].items():
        if final_solver.Value(p):
            mode = mode_lookup[mode_id]
            task = task_by_id[mode.task_id]
            setups = setup_by_mode.get(mode_id, [])
            max_setup = max((s["setup"] for s in setups), default=0)

            selected.append(
                {
                    "task_id": mode.task_id,
                    "block_type": task.block_type,
                    "task_type": task.task_type,
                    "mujoco_name": task.mujoco_name,
                    "half_size": task.half_size,
                    "pick": [task.x, task.y],
                    "target_box": task.target_box,
                    "target": list(task.target_pos),
                    "arms": list(mode.arms),
                    "mode_id": mode.mode_id,
                    "description": mode.description,
                    "start": final_solver.Value(data["starts"][mode_id]),
                    "end": final_solver.Value(data["ends"][mode_id]),
                    "duration": mode.duration,
                    "process_duration": mode.duration,
                    "setup_before_max": max_setup,
                    "setup_before_by_arm": setups,
                    "energy": mode.energy,
                    "process_energy": mode.energy,
                    "uses_center_zone": mode.uses_center_zone,
                }
            )

    # 按开始时间排序，输出表更容易读。
    selected.sort(key=lambda x: (x["start"], x["task_id"]))

    # 目标规划信息单独保存，方便报告解释模型。
    goal_programming = {
        "model_name": "序列相关转移时间目标规划模型",
        "method": "基础方法 序贯目标规划：P1 时间 -> P2 能耗 -> P3 负载均衡",
        "time_goal": data["goals"]["time_goal"],
        "energy_goal": data["goals"]["energy_goal"],
        "balance_goal": data["goals"]["balance_goal"],
        "energy_model": "总能耗 = 类型相关运动/负载/保持/协同能耗 + 序列转移能耗 + 系统级固定开销",
        "system_overhead_per_arm": int(SYSTEM_OVERHEAD_PER_ARM),
        "system_overhead_energy": int(data["system_overhead_energy"]),
        "d_time_plus": final_solver.Value(data["d_time_plus"]),
        "d_time_minus": final_solver.Value(data["d_time_minus"]),
        "d_energy_plus": final_solver.Value(data["d_energy_plus"]),
        "d_energy_minus": final_solver.Value(data["d_energy_minus"]),
        "d_balance_plus": final_solver.Value(data["d_balance_plus"]),
        "d_balance_minus": final_solver.Value(data["d_balance_minus"]),
        "stage1_status": solver1.StatusName(status1),
        "stage2_status": solver2.StatusName(status2),
        "stage3_status": solver3.StatusName(status3) if solver3 is not None else "NOT_RUN",
        "final_stage": final_stage,
        "fallback_reason": fallback_reason,
        "stage3_energy_tolerance": STAGE3_ENERGY_TOLERANCE,
        "stage3_d_energy_tolerance": STAGE3_D_ENERGY_TOLERANCE,
    }

    # 最终返回的 result 是整个工程后续保存/画图/打印的统一数据源。
    result = ScheduleResult(
        status=final_solver.StatusName(final_status),
        final_stage=final_stage,
        fallback_reason=fallback_reason,
        model_name="考虑序列相关转移时间、任务组级中心区互斥与类型相关能耗的多机械臂目标规划模型",
        cmax=final_solver.Value(data["cmax"]),
        motion_energy=final_solver.Value(data["motion_energy"]),
        system_overhead_energy=int(data["system_overhead_energy"]),
        total_energy=final_solver.Value(data["total_energy"]),
        load_imbalance=final_solver.Value(data["load_imbalance"]),
        arm_loads={k: final_solver.Value(v) for k, v in data["arm_loads"].items()},
        goal_programming=goal_programming,
        schedule=selected,
        tasks=[asdict(t) for t in instance.tasks],
        arms={k: asdict(v) for k, v in instance.arms.items()},
    )
    return result
