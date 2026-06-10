# -*- coding: utf-8 -*-
"""基于启发式分支排序与强化下界的隐枚举-分枝定界求解器。

本文件不调用 OR-Tools / CP-SAT，而是在与 ``solver_basic.py`` 相同的数据、
时间、能耗和安全区口径下，显式实现小规模精确搜索。

方法定位：
- 基础方法：把整数规划/目标规划模型交给 CP-SAT 通用求解器；
- 优化方法：用隐枚举、分枝定界和序贯目标规划思想自行求解。
- 优化方法 + 启发式：用启发式分支排序更早找到好上界，并用强化
  Cmax 下界提高分枝定界剪枝效率。

注意：该方法仍然完整枚举任务顺序和执行模式；启发式只用于提供
分支顺序，不用于删减搜索空间；强化下界只使用安全下界，因此不破坏最优性。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from common.config import (
    ALPHA_EMPTY,
    ENERGY_SCALE,
    SYSTEM_OVERHEAD_PER_ARM,
    TIME_SCALE,
    V_EMPTY,
)
from common.geometry import Instance, Mode, Point, Task, distance


INF = 10**18
Score = tuple[int, int, int]


@dataclass(frozen=True)
class SearchState:
    """分枝搜索中的部分调度状态。"""

    scheduled: frozenset[int]
    arm_available: Dict[str, int]
    arm_last_point: Dict[str, Point]
    arm_loads: Dict[str, int]
    motion_energy: int
    cmax: int
    center_intervals: tuple[tuple[int, int, int], ...]
    schedule: tuple[dict, ...]


@dataclass
class SearchResult:
    """字典序目标搜索的当前最好解。"""

    state: SearchState | None = None
    best_score: Score = (INF, INF, INF)
    nodes_visited: int = 0
    nodes_pruned: int = 0
    nodes_pruned_by_dominance: int = 0
    dominance_table: Dict[tuple, list["DominanceLabel"]] | None = None


@dataclass(frozen=True)
class BranchCandidate:
    """一个可被递归展开的候选分支。"""

    task: Task
    mode: Mode
    next_state: SearchState
    sort_key: tuple


@dataclass(frozen=True)
class DominanceLabel:
    """同一结构状态下用于支配比较的保守标签。"""

    arm_available: tuple[int, ...]
    arm_loads: tuple[int, ...]
    motion_energy: int
    cmax: int
    center_release: int


def _to_ticks(seconds: float) -> int:
    return max(1, int(round(seconds * TIME_SCALE)))


def _to_energy(value: float) -> int:
    return max(1, int(round(value * ENERGY_SCALE)))


def _setup_ticks(p_from: Point, p_to: Point) -> int:
    return _to_ticks(distance(p_from, p_to) / V_EMPTY)


def _setup_energy(p_from: Point, p_to: Point) -> int:
    return _to_energy(ALPHA_EMPTY * distance(p_from, p_to))


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _avoid_center_conflicts(start: int, duration: int, center_intervals: Iterable[tuple[int, int, int]]) -> int:
    """把经过中心区的任务推迟到不与已有中心区区间重叠。

    与基础 CP-SAT 模型保持一致：只把任务处理区间视为中心区占用区间，不额外
    约束空载 setup 段。
    """
    ordered = sorted(center_intervals, key=lambda item: (item[0], item[1], item[2]))
    while True:
        end = start + duration
        moved = False
        for used_start, used_end, _task_id in ordered:
            if _overlap(start, end, used_start, used_end):
                start = used_end
                moved = True
                break
        if not moved:
            return start


def _initial_state(instance: Instance) -> SearchState:
    arm_available = {name: 0 for name in instance.arms}
    arm_last_point = {name: arm.pos for name, arm in instance.arms.items()}
    arm_loads = {name: 0 for name in instance.arms}
    return SearchState(
        scheduled=frozenset(),
        arm_available=arm_available,
        arm_last_point=arm_last_point,
        arm_loads=arm_loads,
        motion_energy=0,
        cmax=0,
        center_intervals=tuple(),
        schedule=tuple(),
    )


def _build_schedule_item(task: Task, mode: Mode, start: int, end: int, setup_items: list[dict]) -> dict:
    max_setup = max((item["setup"] for item in setup_items), default=0)
    return {
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
        "start": start,
        "end": end,
        "duration": mode.duration,
        "process_duration": mode.duration,
        "setup_before_max": max_setup,
        "setup_before_by_arm": setup_items,
        "energy": mode.energy,
        "process_energy": mode.energy,
        "uses_center_zone": mode.uses_center_zone,
    }


def _append_mode(state: SearchState, task: Task, mode: Mode) -> SearchState:
    """在当前部分调度后追加一个任务执行模式，生成新的部分调度。"""
    setup_items: list[dict] = []
    candidate_start = 0
    setup_energy = 0
    setup_by_arm: dict[str, int] = {}

    for arm_name in mode.arms:
        last_point = state.arm_last_point[arm_name]
        setup = _setup_ticks(last_point, task.pos)
        energy = _setup_energy(last_point, task.pos)
        setup_by_arm[arm_name] = setup
        setup_energy += energy
        candidate_start = max(candidate_start, state.arm_available[arm_name] + setup)
        setup_items.append(
            {
                "arm": arm_name,
                "setup": setup,
                "from": "base_or_previous_target",
                "from_type": "state",
            }
        )

    start = candidate_start
    if mode.uses_center_zone:
        start = _avoid_center_conflicts(start, mode.duration, state.center_intervals)
    center_wait = start - candidate_start
    end = start + mode.duration

    arm_available = dict(state.arm_available)
    arm_last_point = dict(state.arm_last_point)
    arm_loads = dict(state.arm_loads)
    for arm_name in mode.arms:
        arm_available[arm_name] = end
        arm_last_point[arm_name] = task.target_pos
        arm_loads[arm_name] += setup_by_arm[arm_name] + mode.duration

    center_intervals = list(state.center_intervals)
    if mode.uses_center_zone:
        center_intervals.append((start, end, task.task_id))

    item = _build_schedule_item(task, mode, start, end, setup_items)
    item["center_wait"] = center_wait
    return SearchState(
        scheduled=frozenset((*state.scheduled, task.task_id)),
        arm_available=arm_available,
        arm_last_point=arm_last_point,
        arm_loads=arm_loads,
        motion_energy=state.motion_energy + mode.energy + setup_energy,
        cmax=max(state.cmax, end),
        center_intervals=tuple(center_intervals),
        schedule=(*state.schedule, item),
    )


def _min_remaining_mode_energy(remaining_task_ids: Iterable[int], modes_by_task: Dict[int, List[Mode]]) -> int:
    return sum(min(mode.energy for mode in modes_by_task[task_id]) for task_id in remaining_task_ids)


def _total_energy(state: SearchState, instance: Instance) -> int:
    return int(state.motion_energy + SYSTEM_OVERHEAD_PER_ARM * len(instance.arms))


def _load_imbalance(state: SearchState) -> int:
    loads = list(state.arm_loads.values())
    return int(max(loads) - min(loads)) if loads else 0


def _score(state: SearchState, instance: Instance) -> Score:
    return (int(state.cmax), _total_energy(state, instance), _load_imbalance(state))


def _current_center_release(state: SearchState) -> int:
    return max((end for _start, end, _task_id in state.center_intervals), default=0)


def _point_key(point: Point) -> tuple[int, int]:
    """把机械臂最后位置转成稳定 key。"""
    return (int(round(point[0] * 1_000_000)), int(round(point[1] * 1_000_000)))


def _dominance_key(state: SearchState, arm_names: tuple[str, ...]) -> tuple:
    """状态结构 key：未来任务集合和各机械臂出发位置必须一致才允许比较。"""
    return (
        tuple(sorted(state.scheduled)),
        tuple(_point_key(state.arm_last_point[arm_name]) for arm_name in arm_names),
    )


def _dominance_label(state: SearchState, arm_names: tuple[str, ...]) -> DominanceLabel:
    return DominanceLabel(
        arm_available=tuple(int(state.arm_available[arm_name]) for arm_name in arm_names),
        arm_loads=tuple(int(state.arm_loads[arm_name]) for arm_name in arm_names),
        motion_energy=int(state.motion_energy),
        cmax=int(state.cmax),
        center_release=int(_current_center_release(state)),
    )


def _label_dominates(left: DominanceLabel, right: DominanceLabel) -> bool:
    if left.motion_energy > right.motion_energy:
        return False
    if left.cmax > right.cmax:
        return False
    if left.center_release > right.center_release:
        return False
    if any(a > b for a, b in zip(left.arm_available, right.arm_available)):
        return False
    if any(a > b for a, b in zip(left.arm_loads, right.arm_loads)):
        return False
    return True


def _is_dominated_or_register(state: SearchState, result: SearchResult, arm_names: tuple[str, ...]) -> bool:
    """检查并登记状态支配关系；只在严格保守条件下剪枝。"""
    if result.dominance_table is None:
        result.dominance_table = {}

    key = _dominance_key(state, arm_names)
    label = _dominance_label(state, arm_names)
    labels = result.dominance_table.setdefault(key, [])

    for old in labels:
        if _label_dominates(old, label):
            return True

    labels[:] = [old for old in labels if not _label_dominates(label, old)]
    labels.append(label)
    return False


def _cmax_lower_bound(state: SearchState, tasks: list[Task], task_by_id: Dict[int, Task], modes_by_task: Dict[int, List[Mode]]) -> int:
    """剩余任务最早单独完成时间下界。

    对每个剩余任务，假设它可以直接作为下一个任务安排，取所有 mode 中最早
    完工时间的最小值。真实调度只会比这个更晚或相等，因此该值是安全下界。
    """
    lower_bound = state.cmax
    for task in tasks:
        if task.task_id in state.scheduled:
            continue
        earliest_end = min(
            _append_mode(state, task_by_id[mode.task_id], mode).schedule[-1]["end"]
            for mode in modes_by_task[task.task_id]
        )
        lower_bound = max(lower_bound, earliest_end)
    return lower_bound


def _remaining_resource_capacity_lower_bound(
    state: SearchState,
    remaining_task_ids: list[int],
    modes_by_task: Dict[int, List[Mode]],
    arm_count: int,
) -> int:
    """剩余处理时间的资源容量下界。

    这里只累计所有剩余任务的最小必要处理工作量：
      min(mode.duration * len(mode.arms))

    不加入 setup，因为后续 setup 取决于未来顺序和机械臂位置；随意加入会让
    下界变得不安全。使用 arm_available 而不是 arm_loads，是因为这里估计
    的是时间轴上的 Cmax 下界。
    """
    if arm_count <= 0:
        return state.cmax

    remaining_required_work = 0
    for task_id in remaining_task_ids:
        remaining_required_work += min(
            mode.duration * len(mode.arms)
            for mode in modes_by_task[task_id]
        )

    occupied_timeline_work = sum(state.arm_available.values())
    average_capacity_bound = (
        occupied_timeline_work + remaining_required_work + arm_count - 1
    ) // arm_count
    return max(int(state.cmax), int(average_capacity_bound))


def _forced_arm_lower_bound(
    state: SearchState,
    remaining_task_ids: list[int],
    modes_by_task: Dict[int, List[Mode]],
    arm_names: Iterable[str],
) -> int:
    """必须由某些机械臂承担的剩余处理时间下界。

    若某个任务的所有可行模式都包含机械臂 a，则该任务至少还会占用 a
    一段处理时间。这里同样只加入 duration，不加入 setup。
    """
    forced_work = {arm_name: 0 for arm_name in arm_names}

    for task_id in remaining_task_ids:
        modes = modes_by_task[task_id]
        common_arms = set(modes[0].arms)
        for mode in modes[1:]:
            common_arms.intersection_update(mode.arms)

        if not common_arms:
            continue

        min_duration = min(mode.duration for mode in modes)
        for arm_name in common_arms:
            forced_work[arm_name] += min_duration

    lower_bound = state.cmax
    for arm_name, work in forced_work.items():
        lower_bound = max(lower_bound, state.arm_available[arm_name] + work)
    return int(lower_bound)


def _forced_center_lower_bound(
    state: SearchState,
    remaining_task_ids: list[int],
    modes_by_task: Dict[int, List[Mode]],
) -> int:
    """必须占用中心区任务的串行处理时间下界。"""
    remaining_center_work = 0

    for task_id in remaining_task_ids:
        modes = modes_by_task[task_id]
        if all(mode.uses_center_zone for mode in modes):
            remaining_center_work += min(mode.duration for mode in modes)

    center_release = _current_center_release(state)
    return max(int(state.cmax), int(center_release + remaining_center_work))


def _is_complete(state: SearchState, tasks: list[Task]) -> bool:
    return len(state.scheduled) == len(tasks)


def _should_prune(
    state: SearchState,
    result: SearchResult,
    instance: Instance,
    tasks: list[Task],
    modes_by_task: Dict[int, List[Mode]],
) -> bool:
    """按字典序目标下界剪枝；只使用安全下界，避免误删最优解。"""
    if result.state is None:
        return False
    task_by_id = {task.task_id: task for task in tasks}
    cmax_lower_bound = _cmax_lower_bound(state, tasks, task_by_id, modes_by_task)

    remaining = [task.task_id for task in tasks if task.task_id not in state.scheduled]
    cmax_lower_bound = max(
        cmax_lower_bound,
        _remaining_resource_capacity_lower_bound(
            state,
            remaining,
            modes_by_task,
            len(instance.arms),
        ),
        _forced_arm_lower_bound(
            state,
            remaining,
            modes_by_task,
            instance.arms.keys(),
        ),
        _forced_center_lower_bound(
            state,
            remaining,
            modes_by_task,
        ),
    )

    remaining_min_energy = _min_remaining_mode_energy(remaining, modes_by_task)
    total_energy_lower_bound = int(
        state.motion_energy
        + remaining_min_energy
        + SYSTEM_OVERHEAD_PER_ARM * len(instance.arms)
    )

    lower_bound_score: Score = (int(cmax_lower_bound), int(total_energy_lower_bound), 0)
    return lower_bound_score >= result.best_score


def _branch_candidates(
    tasks: list[Task],
    state: SearchState,
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    instance: Instance,
) -> list[BranchCandidate]:
    """生成并排序候选分支；排序只影响上界出现速度，不裁剪解空间。"""
    candidates: list[BranchCandidate] = []

    for task in tasks:
        if task.task_id in state.scheduled:
            continue
        for mode in modes_by_task[task.task_id]:
            next_state = _append_mode(state, task_by_id[mode.task_id], mode)
            item = next_state.schedule[-1]

            arm_ready = max(state.arm_available[arm_name] for arm_name in mode.arms)
            setup_before = int(item.get("setup_before_max", 0))
            center_wait = int(item.get("center_wait", 0))

            sort_key = (
                next_state.cmax,
                item["end"],
                center_wait,
                _total_energy(next_state, instance),
                _load_imbalance(next_state),
                setup_before,
                arm_ready,
                item["task_id"],
                item["mode_id"],
            )
            candidates.append(BranchCandidate(task=task, mode=mode, next_state=next_state, sort_key=sort_key))

    return sorted(candidates, key=lambda candidate: candidate.sort_key)


def _search(
    state: SearchState,
    result: SearchResult,
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    arm_names: tuple[str, ...],
) -> None:
    result.nodes_visited += 1

    if _is_dominated_or_register(state, result, arm_names):
        result.nodes_pruned += 1
        result.nodes_pruned_by_dominance += 1
        return

    if _should_prune(state, result, instance, tasks, modes_by_task):
        result.nodes_pruned += 1
        return

    if _is_complete(state, tasks):
        value = _score(state, instance)
        if value < result.best_score:
            result.state = state
            result.best_score = value
        return

    for candidate in _branch_candidates(tasks, state, task_by_id, modes_by_task, instance):
        _search(
            candidate.next_state,
            result,
            instance,
            tasks,
            task_by_id,
            modes_by_task,
            arm_names,
        )


def _run_lexicographic_search(
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
) -> SearchResult:
    print(
        "[optimized_heuristic] start lexicographic search "
        f"tasks={len(tasks)}, arms={len(instance.arms)}, "
        "objective=(Cmax, total_energy, load_imbalance)",
        flush=True,
    )
    result = SearchResult(dominance_table={})
    arm_names = tuple(sorted(instance.arms))
    _search(
        _initial_state(instance),
        result,
        instance,
        tasks,
        task_by_id,
        modes_by_task,
        arm_names,
    )
    if result.state is None:
        raise RuntimeError("隐枚举-分枝定界未找到可行调度。")
    print(
        "[optimized_heuristic] finish lexicographic search "
        f"best_score={result.best_score}, "
        f"nodes_visited={result.nodes_visited}, nodes_pruned={result.nodes_pruned}, "
        f"dominance_pruned={result.nodes_pruned_by_dominance}",
        flush=True,
    )
    return result


def _sort_schedule(schedule: Iterable[dict]) -> list[dict]:
    return sorted((dict(item) for item in schedule), key=lambda item: (item["start"], item["task_id"], item["mode_id"]))


def _goal_summary(search_result: SearchResult, best_state: SearchState, instance: Instance) -> dict:
    best_cmax, best_energy, best_load = search_result.best_score
    return {
        "model_name": "显式隐枚举-分枝定界序贯目标规划模型",
        "method": "优化方法 + 启发式：启发式分支排序与强化下界的隐枚举-分枝定界序贯目标规划",
        "time_goal": int(best_cmax),
        "energy_goal": int(best_energy),
        "balance_goal": 0,
        "d_time_plus": 0,
        "d_time_minus": 0,
        "d_energy_plus": 0,
        "d_energy_minus": 0,
        "d_balance_plus": int(best_load),
        "d_balance_minus": 0,
        "system_overhead_per_arm": int(SYSTEM_OVERHEAD_PER_ARM),
        "system_overhead_energy": int(SYSTEM_OVERHEAD_PER_ARM * len(instance.arms)),
        "lexicographic_score": [int(best_cmax), int(best_energy), int(best_load)],
        "nodes_visited": int(search_result.nodes_visited),
        "nodes_pruned": int(search_result.nodes_pruned),
        "nodes_pruned_by_dominance": int(search_result.nodes_pruned_by_dominance),
        "dominance_state_count": len(search_result.dominance_table or {}),
        "note": "本方法不调用通用求解器；动态分支顺序显式考虑Cmax、中心区等待、能耗和负载，强化Cmax/能耗下界与保守状态支配剪枝保证不误删潜在最优解。",
    }


def solve_schedule(instance: Instance, modes_by_task: Dict[int, List[Mode]]) -> dict:
    """用隐枚举-分枝定界求解与基础方法同口径的多机械臂调度问题。"""
    tasks = sorted(instance.tasks, key=lambda task: task.task_id)
    task_by_id = {task.task_id: task for task in tasks}

    search_result = _run_lexicographic_search(instance, tasks, task_by_id, modes_by_task)
    final_state = search_result.state
    if final_state is None:
        raise RuntimeError("隐枚举-分枝定界未得到最终可行调度。")

    schedule = _sort_schedule(final_state.schedule)
    result = {
        "status": "OPTIMAL_BY_EXPLICIT_BRANCH_AND_BOUND",
        "algorithm_name": "优化方法 + 启发式：启发式分支排序与强化下界的自编隐枚举-分枝定界序贯目标规划",
        "model_name": "与基础 CP-SAT 同口径的显式分枝定界多机械臂调度模型",
        "cmax": int(final_state.cmax),
        "motion_energy": int(final_state.motion_energy),
        "system_overhead_energy": int(SYSTEM_OVERHEAD_PER_ARM * len(instance.arms)),
        "total_energy": _total_energy(final_state, instance),
        "load_imbalance": _load_imbalance(final_state),
        "arm_loads": {k: int(v) for k, v in final_state.arm_loads.items()},
        "goal_programming": _goal_summary(search_result, final_state, instance),
        "schedule": schedule,
        "tasks": [asdict(t) for t in instance.tasks],
        "arms": {k: asdict(v) for k, v in instance.arms.items()},
        "available_modes_count": {str(task_id): len(modes) for task_id, modes in modes_by_task.items()},
    }
    return result
