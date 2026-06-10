# -*- coding: utf-8 -*-
"""基于大邻域搜索的启发式求解器。

本文件不调用 OR-Tools / CP-SAT，而是在与 ``solver_basic.py`` 相同的数据、
时间、能耗和安全区口径下，快速构造和改进可行调度。

方法定位：
- 基础方法：把整数规划/目标规划模型交给 CP-SAT 通用求解器；
- 优化方法：用隐枚举、分枝定界和序贯目标规划思想自行求解。
- 优化方法 + 启发式：用多策略贪心构造初始解，再用大邻域搜索
  反复破坏和修复部分任务，快速得到高质量可行解。

注意：该方法不证明全局最优性，主要用于与基础方法和自编精确方法
对比大规模算例下的计算时间与解质量。
"""

from __future__ import annotations

import random
import time
from itertools import combinations, permutations, product
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from common.config import (
    ALPHA_EMPTY,
    ENERGY_SCALE,
    OPTIMIZED_TIME_LIMIT,
    SYSTEM_OVERHEAD_PER_ARM,
    TIME_SCALE,
    V_EMPTY,
)
from common.geometry import Instance, Mode, Point, Task, distance


INF = 10**18
Score = tuple[int, int, int]
HEURISTIC_TIME_LIMIT = min(5.0, float(OPTIMIZED_TIME_LIMIT))
HEURISTIC_MAX_ITERATIONS = 5000
HEURISTIC_NO_IMPROVE_LIMIT = 800
HEURISTIC_RESTARTS = 5
EXACT_REPAIR_INTERVAL = 11
EXACT_REPAIR_MAX_REMOVED = 4
EXACT_REPAIR_TIME_SLICE = 0.15
EXACT_REPAIR_ENABLED = False
BEAM_WIDTH = 220
BEAM_CANDIDATE_LIMIT = 5000


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


def _mode_lookup(modes_by_task: Dict[int, List[Mode]]) -> dict[str, Mode]:
    return {mode.mode_id: mode for modes in modes_by_task.values() for mode in modes}


def _decode_plan(plan: list[tuple[int, str]], instance: Instance, task_by_id: Dict[int, Task], mode_by_id: dict[str, Mode]) -> SearchState:
    """按给定任务-模式序列解码为完整调度。"""
    state = _initial_state(instance)
    for task_id, mode_id in plan:
        state = _append_mode(state, task_by_id[task_id], mode_by_id[mode_id])
    return state


def _state_to_plan(state: SearchState) -> list[tuple[int, str]]:
    return [(int(item["task_id"]), str(item["mode_id"])) for item in state.schedule]


def _candidate_key(state: SearchState, instance: Instance, strategy: str) -> tuple:
    item = state.schedule[-1]
    center_wait = int(item.get("center_wait", 0))
    setup_before = int(item.get("setup_before_max", 0))
    score = _score(state, instance)

    if strategy == "energy":
        return (score[0], score[1], item["end"], center_wait, score[2], setup_before)
    if strategy == "center":
        return (score[0], center_wait, item["end"], score[1], score[2], setup_before)
    if strategy == "balance":
        return (score[0], score[2], item["end"], center_wait, score[1], setup_before)
    return (score[0], item["end"], center_wait, score[1], score[2], setup_before)


def _greedy_construct(
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    strategy: str,
) -> SearchState:
    """按指定排序策略构造一个完整可行解。"""
    state = _initial_state(instance)

    while not _is_complete(state, tasks):
        best_key = None
        best_state = None

        for task in tasks:
            if task.task_id in state.scheduled:
                continue
            for mode in modes_by_task[task.task_id]:
                next_state = _append_mode(state, task_by_id[task.task_id], mode)
                key = (*_candidate_key(next_state, instance, strategy), task.task_id, mode.mode_id)
                if best_key is None or key < best_key:
                    best_key = key
                    best_state = next_state

        if best_state is None:
            raise RuntimeError("启发式构造失败：没有可扩展候选。")
        state = best_state

    return state


def _best_initial_solution(
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
) -> SearchState:
    """构造多个贪心初始解，并取字典序目标最好的一个。"""
    strategies = ("time", "center", "energy", "balance")
    states = [
        _greedy_construct(instance, tasks, task_by_id, modes_by_task, strategy)
        for strategy in strategies
    ]
    return min(states, key=lambda state: _score(state, instance))


def _random_initial_solution(
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    rng: random.Random,
) -> SearchState:
    """构造带随机扰动的初始解，用于多启动。"""
    remaining = [task.task_id for task in tasks]
    rng.shuffle(remaining)
    state = _initial_state(instance)

    while remaining:
        sample_size = min(len(remaining), 3)
        sampled_tasks = rng.sample(remaining, sample_size)
        best_key = None
        best_state = None
        chosen_task = None

        for task_id in sampled_tasks:
            modes = list(modes_by_task[task_id])
            rng.shuffle(modes)
            for mode in modes:
                next_state = _append_mode(state, task_by_id[task_id], mode)
                key = (
                    *_score(next_state, instance),
                    rng.random(),
                    task_id,
                    mode.mode_id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_state = next_state
                    chosen_task = task_id

        if best_state is None or chosen_task is None:
            raise RuntimeError("随机初始解构造失败。")

        state = best_state
        remaining.remove(chosen_task)

    return state


def _repair_by_best_insertion(
    base_plan: list[tuple[int, str]],
    removed_task_ids: list[int],
    instance: Instance,
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    mode_by_id: dict[str, Mode],
) -> tuple[list[tuple[int, str]], SearchState]:
    """把被移除任务逐个插回，使当前完整计划的字典序目标最好。"""
    plan = [item for item in base_plan if item[0] not in set(removed_task_ids)]
    remaining = list(removed_task_ids)
    best_state = _decode_plan(plan, instance, task_by_id, mode_by_id) if plan else _initial_state(instance)

    while remaining:
        best_key = None
        best_plan = None
        chosen_task = None
        chosen_state = None

        for task_id in remaining:
            for mode in modes_by_task[task_id]:
                for pos in range(len(plan) + 1):
                    candidate_plan = plan[:pos] + [(task_id, mode.mode_id)] + plan[pos:]
                    candidate_state = _decode_plan(candidate_plan, instance, task_by_id, mode_by_id)
                    item = next(item for item in candidate_state.schedule if item["task_id"] == task_id)
                    key = (
                        *_score(candidate_state, instance),
                        int(item.get("center_wait", 0)),
                        int(item.get("setup_before_max", 0)),
                        task_id,
                        mode.mode_id,
                        pos,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_plan = candidate_plan
                        chosen_task = task_id
                        chosen_state = candidate_state

        if best_plan is None or chosen_task is None or chosen_state is None:
            raise RuntimeError("启发式修复失败：没有可插入候选。")

        plan = best_plan
        best_state = chosen_state
        remaining.remove(chosen_task)

    return plan, best_state


def _insert_items_by_positions(
    base_plan: list[tuple[int, str]],
    inserted_items: tuple[tuple[int, str], ...],
    positions: tuple[int, ...],
) -> list[tuple[int, str]]:
    """按组合位置把若干任务插入基准序列。"""
    result = list(base_plan)
    for item, pos in sorted(zip(inserted_items, positions), key=lambda pair: pair[1], reverse=True):
        result.insert(pos, item)
    return result


def _repair_by_exact_reinsert(
    base_plan: list[tuple[int, str]],
    removed_task_ids: list[int],
    instance: Instance,
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    mode_by_id: dict[str, Mode],
    deadline: float,
) -> tuple[list[tuple[int, str]], SearchState]:
    """对小破坏集合做组合枚举修复，覆盖贪心插入错过的联动变化。"""
    if len(removed_task_ids) > EXACT_REPAIR_MAX_REMOVED or time.perf_counter() >= deadline:
        return _repair_by_best_insertion(base_plan, removed_task_ids, instance, task_by_id, modes_by_task, mode_by_id)

    fixed_plan = [item for item in base_plan if item[0] not in set(removed_task_ids)]
    best_plan = None
    best_state = None
    best_key = None
    insert_count = len(removed_task_ids)
    position_count = len(fixed_plan) + insert_count

    for task_order in permutations(removed_task_ids):
        mode_options = [modes_by_task[task_id] for task_id in task_order]
        for mode_choice in product(*mode_options):
            inserted_items = tuple((task_id, mode.mode_id) for task_id, mode in zip(task_order, mode_choice))
            for positions in combinations(range(position_count), insert_count):
                if time.perf_counter() >= deadline:
                    if best_plan is not None and best_state is not None:
                        return best_plan, best_state
                    return _repair_by_best_insertion(base_plan, removed_task_ids, instance, task_by_id, modes_by_task, mode_by_id)

                candidate_plan = _insert_items_by_positions(fixed_plan, inserted_items, positions)
                candidate_state = _decode_plan(candidate_plan, instance, task_by_id, mode_by_id)
                key = (*_score(candidate_state, instance), positions, inserted_items)
                if best_key is None or key < best_key:
                    best_key = key
                    best_plan = candidate_plan
                    best_state = candidate_state

    if best_plan is None or best_state is None:
        return _repair_by_best_insertion(base_plan, removed_task_ids, instance, task_by_id, modes_by_task, mode_by_id)
    return best_plan, best_state


def _select_removed_tasks(plan: list[tuple[int, str]], state: SearchState, rng: random.Random, q: int, iteration: int) -> list[int]:
    """选择 LNS 破坏集合：混合随机、等待、末端和连续片段破坏。"""
    task_ids = [task_id for task_id, _mode_id in plan]
    if len(task_ids) <= q:
        return task_ids

    strategy = iteration % 5
    if strategy == 0:
        return rng.sample(task_ids, q)

    if strategy == 1:
        ranked_items = sorted(
            state.schedule,
            key=lambda item: (
                int(item.get("center_wait", 0)),
                int(item["end"]),
                int(item.get("setup_before_max", 0)),
            ),
            reverse=True,
        )
        return [int(item["task_id"]) for item in ranked_items[:q]]

    if strategy == 2:
        ranked_items = sorted(
            state.schedule,
            key=lambda item: (
                int(item["end"]),
                int(item.get("setup_before_max", 0)),
                int(item.get("center_wait", 0)),
            ),
            reverse=True,
        )
        return [int(item["task_id"]) for item in ranked_items[:q]]

    if strategy == 3:
        start = rng.randrange(0, len(plan) - q + 1)
        return [task_id for task_id, _mode_id in plan[start:start + q]]

    ranked_items = sorted(
        state.schedule,
        key=lambda item: (
            len(item.get("arms", [])),
            int(item["end"]),
            int(item.get("center_wait", 0)),
        ),
        reverse=True,
    )
    removed = [int(item["task_id"]) for item in ranked_items[: max(1, q - 1)]]
    candidates = [task_id for task_id in task_ids if task_id not in removed]
    if candidates and len(removed) < q:
        removed.append(rng.choice(candidates))
    return removed


def _refine_energy_same_cmax(
    state: SearchState,
    instance: Instance,
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    mode_by_id: dict[str, Mode],
    deadline: float,
) -> tuple[SearchState, int]:
    """在不增大 Cmax 的前提下，局部降低能耗。"""
    plan = _state_to_plan(state)
    best_state = state
    best_score = _score(state, instance)
    cmax_limit = best_score[0]
    improvements = 0
    changed = True

    while changed and time.perf_counter() < deadline:
        changed = False

        for idx, (task_id, mode_id) in enumerate(list(plan)):
            if time.perf_counter() >= deadline:
                break

            for mode in modes_by_task[task_id]:
                if mode.mode_id == mode_id:
                    continue
                candidate_plan = list(plan)
                candidate_plan[idx] = (task_id, mode.mode_id)
                candidate_state = _decode_plan(candidate_plan, instance, task_by_id, mode_by_id)
                candidate_score = _score(candidate_state, instance)
                if candidate_score[0] <= cmax_limit and candidate_score[1:] < best_score[1:]:
                    plan = candidate_plan
                    best_state = candidate_state
                    best_score = candidate_score
                    improvements += 1
                    changed = True
                    break
            if changed:
                break

        if changed:
            continue

        for i in range(len(plan)):
            if time.perf_counter() >= deadline:
                break
            task_item = plan[i]
            reduced = plan[:i] + plan[i + 1:]
            for pos in range(len(reduced) + 1):
                if pos == i:
                    continue
                candidate_plan = reduced[:pos] + [task_item] + reduced[pos:]
                candidate_state = _decode_plan(candidate_plan, instance, task_by_id, mode_by_id)
                candidate_score = _score(candidate_state, instance)
                if candidate_score[0] <= cmax_limit and candidate_score[1:] < best_score[1:]:
                    plan = candidate_plan
                    best_state = candidate_state
                    best_score = candidate_score
                    improvements += 1
                    changed = True
                    break
            if changed:
                break

        if changed:
            continue

        for i in range(len(plan)):
            if time.perf_counter() >= deadline:
                break
            for j in range(i + 1, len(plan)):
                candidate_plan = list(plan)
                candidate_plan[i], candidate_plan[j] = candidate_plan[j], candidate_plan[i]
                candidate_state = _decode_plan(candidate_plan, instance, task_by_id, mode_by_id)
                candidate_score = _score(candidate_state, instance)
                if candidate_score[0] <= cmax_limit and candidate_score[1:] < best_score[1:]:
                    plan = candidate_plan
                    best_state = candidate_state
                    best_score = candidate_score
                    improvements += 1
                    changed = True
                    break
            if changed:
                break

    return best_state, improvements


def _run_small_exhaustive_search(
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    mode_by_id: dict[str, Mode],
    deadline: float,
) -> tuple[SearchState, dict]:
    """小规模完整排列-模式扫描，避免 beam 在简单案例上近似失真。"""
    task_ids = [task.task_id for task in tasks]
    best_state = _best_initial_solution(instance, tasks, task_by_id, modes_by_task)
    best_score = _score(best_state, instance)
    checked = 0

    mode_options = [modes_by_task[task_id] for task_id in task_ids]
    for mode_choice in product(*mode_options):
        mode_by_task = {task_id: mode for task_id, mode in zip(task_ids, mode_choice)}
        for order in permutations(task_ids):
            if time.perf_counter() >= deadline:
                return best_state, {
                    "iterations": int(checked),
                    "improvements": 0,
                    "energy_refine_improvements": 0,
                    "exact_repairs": 0,
                    "destroy_size": 0,
                    "no_improve_limit": 0,
                    "restarts": 1,
                    "time_limit_s": float(HEURISTIC_TIME_LIMIT),
                    "method": "小规模限时排列-模式扫描",
                    "note": "小规模场景扫描任务排列和模式组合；若时间内未完成，则返回已发现的最好可行解。",
                }

            plan = [(task_id, mode_by_task[task_id].mode_id) for task_id in order]
            state = _decode_plan(plan, instance, task_by_id, mode_by_id)
            score = _score(state, instance)
            checked += 1
            if score < best_score:
                best_state = state
                best_score = score

    return best_state, {
        "iterations": int(checked),
        "improvements": 0,
        "energy_refine_improvements": 0,
        "exact_repairs": 0,
        "destroy_size": 0,
        "no_improve_limit": 0,
        "restarts": 1,
        "time_limit_s": float(HEURISTIC_TIME_LIMIT),
        "method": "小规模完整排列-模式扫描",
        "note": "小规模场景完整扫描任务排列和模式组合；完成扫描时可得到该启发式解码口径下的最优结果。",
    }


def _beam_state_key(state: SearchState, arm_names: tuple[str, ...]) -> tuple:
    return (
        tuple(sorted(state.scheduled)),
        tuple(_point_key(state.arm_last_point[arm_name]) for arm_name in arm_names),
        tuple(int(state.arm_available[arm_name]) for arm_name in arm_names),
    )


def _beam_sort_key(
    state: SearchState,
    instance: Instance,
    tasks: list[Task],
    modes_by_task: Dict[int, List[Mode]],
) -> tuple:
    remaining = [task.task_id for task in tasks if task.task_id not in state.scheduled]
    min_remaining_duration = sum(min(mode.duration for mode in modes_by_task[task_id]) for task_id in remaining)
    min_remaining_energy = _min_remaining_mode_energy(remaining, modes_by_task)
    avg_future = (sum(state.arm_available.values()) + min_remaining_duration) // max(1, len(instance.arms))
    optimistic_cmax = max(state.cmax, avg_future)
    return (
        int(optimistic_cmax),
        int(state.cmax),
        int(state.motion_energy + min_remaining_energy),
        _load_imbalance(state),
        len(remaining),
    )


def _run_beam_search_heuristic(
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
    deadline: float,
) -> tuple[SearchState, dict] | None:
    """限宽动态规划/beam search，系统保留多条部分调度路径。"""
    arm_names = tuple(sorted(instance.arms))
    beam = [_initial_state(instance)]
    best_state = None
    expanded = 0
    if len(tasks) <= 5:
        width = BEAM_WIDTH if len(instance.arms) == 2 else max(80, BEAM_WIDTH // 2)
    elif len(tasks) <= 7:
        width = 180 if len(instance.arms) == 2 else 100
    else:
        width = 160 if len(instance.arms) == 2 else 90

    for _depth in range(len(tasks)):
        candidates: list[SearchState] = []
        for state in beam:
            if time.perf_counter() >= deadline:
                return None
            for task in tasks:
                if task.task_id in state.scheduled:
                    continue
                for mode in modes_by_task[task.task_id]:
                    candidates.append(_append_mode(state, task_by_id[task.task_id], mode))
                    expanded += 1

        if not candidates:
            break

        candidates.sort(key=lambda state: _beam_sort_key(state, instance, tasks, modes_by_task))
        next_beam = []
        seen = set()
        for state in candidates:
            key = _beam_state_key(state, arm_names)
            if key in seen:
                continue
            seen.add(key)
            next_beam.append(state)
            if len(next_beam) >= width:
                break
            if len(seen) >= BEAM_CANDIDATE_LIMIT:
                break
        beam = next_beam

    complete_states = [state for state in beam if _is_complete(state, tasks)]
    if complete_states:
        best_state = min(complete_states, key=lambda state: _score(state, instance))

    if best_state is None:
        return None

    return best_state, {
        "iterations": int(expanded),
        "improvements": 0,
        "energy_refine_improvements": 0,
        "exact_repairs": 0,
        "destroy_size": 0,
        "no_improve_limit": 0,
        "restarts": 1,
        "time_limit_s": float(HEURISTIC_TIME_LIMIT),
        "method": f"限宽动态规划 Beam Search(width={width})",
        "note": "启发式限宽保留多条部分调度路径，不证明全局最优；用于提高大规模场景命中高质量解的概率。",
    }


def _run_lns_heuristic(
    instance: Instance,
    tasks: list[Task],
    task_by_id: Dict[int, Task],
    modes_by_task: Dict[int, List[Mode]],
) -> tuple[SearchState, dict]:
    """限时大邻域搜索：快速得到高质量可行解，不证明最优性。"""
    rng = random.Random(0)
    mode_by_id = _mode_lookup(modes_by_task)

    start_time = time.perf_counter()
    deadline = start_time + HEURISTIC_TIME_LIMIT

    if len(tasks) <= 5:
        beam_time = 0.12
        lns_time = 0.08
        refine_time = 0.03
    elif len(tasks) <= 7:
        beam_time = 0.35
        lns_time = 0.45
        refine_time = 0.08
    else:
        beam_time = 0.65
        lns_time = 0.85
        refine_time = 0.12

    beam_deadline = min(deadline, start_time + beam_time)
    beam_result = _run_beam_search_heuristic(instance, tasks, task_by_id, modes_by_task, beam_deadline)

    lns_deadline = min(deadline, time.perf_counter() + lns_time)
    global_best_state = _best_initial_solution(instance, tasks, task_by_id, modes_by_task)
    beam_info = None
    if beam_result is not None:
        beam_state, beam_info = beam_result
        if _score(beam_state, instance) < _score(global_best_state, instance):
            global_best_state = beam_state
    global_best_plan = _state_to_plan(global_best_state)
    global_best_score = _score(global_best_state, instance)
    total_iterations = 0
    total_improved = 0
    energy_refine_improvements = 0
    exact_repairs = 0
    base_q = min(max(2, len(tasks) // 3), max(1, len(tasks) - 1))

    for restart in range(HEURISTIC_RESTARTS):
        if time.perf_counter() >= lns_deadline or total_iterations >= HEURISTIC_MAX_ITERATIONS:
            break

        if restart == 0:
            current_state = global_best_state
        else:
            current_state = _random_initial_solution(instance, tasks, task_by_id, modes_by_task, rng)
            if _score(current_state, instance) < global_best_score:
                global_best_state = current_state
                global_best_plan = _state_to_plan(current_state)
                global_best_score = _score(current_state, instance)

        current_plan = _state_to_plan(current_state)
        local_best_state = current_state
        local_best_plan = list(current_plan)
        local_best_score = _score(local_best_state, instance)
        no_improve = 0

        while (
            total_iterations < HEURISTIC_MAX_ITERATIONS
            and no_improve < HEURISTIC_NO_IMPROVE_LIMIT
            and time.perf_counter() < lns_deadline
        ):
            total_iterations += 1
            elapsed_ratio = max(0.0, min(1.0, 1.0 - (lns_deadline - time.perf_counter()) / HEURISTIC_TIME_LIMIT))
            q = min(max(1, base_q + (total_iterations % 3) - 1), max(1, len(tasks) - 1))
            if EXACT_REPAIR_ENABLED:
                q = min(EXACT_REPAIR_MAX_REMOVED, max(1, len(tasks) - 1))
            removed = _select_removed_tasks(current_plan, current_state, rng, q, total_iterations)
            if (
                EXACT_REPAIR_ENABLED
                and len(removed) <= EXACT_REPAIR_MAX_REMOVED
                and total_iterations % EXACT_REPAIR_INTERVAL == 0
            ):
                exact_deadline = min(deadline, time.perf_counter() + EXACT_REPAIR_TIME_SLICE)
                candidate_plan, candidate_state = _repair_by_exact_reinsert(
                    current_plan,
                    removed,
                    instance,
                    task_by_id,
                    modes_by_task,
                    mode_by_id,
                    exact_deadline,
                )
                exact_repairs += 1
            else:
                candidate_plan, candidate_state = _repair_by_best_insertion(
                    current_plan,
                    removed,
                    instance,
                    task_by_id,
                    modes_by_task,
                    mode_by_id,
                )
            candidate_score = _score(candidate_state, instance)

            if candidate_score < local_best_score:
                local_best_state = candidate_state
                local_best_plan = list(candidate_plan)
                local_best_score = candidate_score
                current_state = candidate_state
                current_plan = list(candidate_plan)
                total_improved += 1
                no_improve = 0
                if candidate_score < global_best_score:
                    global_best_state = candidate_state
                    global_best_plan = list(candidate_plan)
                    global_best_score = candidate_score
            else:
                no_improve += 1
                current_score = _score(current_state, instance)
                if elapsed_ratio < 0.60:
                    cmax_slack = max(1, int(local_best_score[0] * 0.02))
                    accept_worse = (
                        candidate_score[0] <= local_best_score[0] + cmax_slack
                        and rng.random() < max(0.02, 0.20 * (1.0 - elapsed_ratio))
                    )
                    if candidate_score <= current_score or accept_worse or total_iterations % 13 == 0:
                        current_state = candidate_state
                        current_plan = list(candidate_plan)
                else:
                    if (
                        candidate_score[0] <= local_best_score[0]
                        and candidate_score[1:] <= current_score[1:]
                    ):
                        current_state = candidate_state
                        current_plan = list(candidate_plan)

            if total_iterations % 29 == 0:
                current_state = local_best_state
                current_plan = list(local_best_plan)

        if local_best_score < global_best_score:
            global_best_state = local_best_state
            global_best_plan = list(local_best_plan)
            global_best_score = local_best_score

    refined_state, energy_refine_improvements = _refine_energy_same_cmax(
        global_best_state,
        instance,
        task_by_id,
        modes_by_task,
        mode_by_id,
        min(deadline, time.perf_counter() + refine_time),
    )
    if _score(refined_state, instance) < global_best_score:
        global_best_state = refined_state
        global_best_plan = _state_to_plan(refined_state)
        global_best_score = _score(refined_state, instance)

    info = {
        "iterations": int(total_iterations),
        "improvements": int(total_improved),
        "energy_refine_improvements": int(energy_refine_improvements),
        "exact_repairs": int(exact_repairs),
        "destroy_size": int(base_q),
        "no_improve_limit": int(HEURISTIC_NO_IMPROVE_LIMIT),
        "restarts": int(HEURISTIC_RESTARTS),
        "time_limit_s": float(HEURISTIC_TIME_LIMIT),
        "method": "Beam Search warm start + 多启动LNS + 两阶段接受准则 + 同Cmax能耗精化",
        "note": "启发式只保证可行解，不证明全局最优；用于与精确算法比较大规模场景下的速度和解质量。",
    }
    if beam_info is not None:
        info["beam_iterations"] = beam_info["iterations"]
        info["beam_method"] = beam_info["method"]
    return global_best_state, info


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
    """用 LNS 启发式求解与基础方法同口径的多机械臂调度问题。"""
    tasks = sorted(instance.tasks, key=lambda task: task.task_id)
    task_by_id = {task.task_id: task for task in tasks}

    print(
        "[optimized_heuristic] start LNS heuristic "
        f"tasks={len(tasks)}, arms={len(instance.arms)}, "
        "objective=(Cmax, total_energy, load_imbalance)",
        flush=True,
    )
    final_state, heuristic_info = _run_lns_heuristic(instance, tasks, task_by_id, modes_by_task)
    print(
        "[optimized_heuristic] finish LNS heuristic "
        f"best_score={_score(final_state, instance)}, "
        f"iterations={heuristic_info['iterations']}, improvements={heuristic_info['improvements']}",
        flush=True,
    )

    schedule = _sort_schedule(final_state.schedule)
    best_cmax, best_energy, best_load = _score(final_state, instance)
    goal_programming = {
        "model_name": "启发式大邻域搜索调度模型",
        "method": heuristic_info["method"],
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
        "heuristic_iterations": heuristic_info["iterations"],
        "heuristic_improvements": heuristic_info["improvements"],
        "energy_refine_improvements": heuristic_info["energy_refine_improvements"],
        "exact_repairs": heuristic_info["exact_repairs"],
        "beam_iterations": heuristic_info.get("beam_iterations", 0),
        "beam_method": heuristic_info.get("beam_method", ""),
        "destroy_size": heuristic_info["destroy_size"],
        "restarts": heuristic_info["restarts"],
        "time_limit_s": heuristic_info["time_limit_s"],
        "note": heuristic_info["note"],
    }
    result = {
        "status": "FEASIBLE_BY_LNS_HEURISTIC",
        "algorithm_name": "优化方法 + 启发式：大邻域搜索快速可行调度算法",
        "model_name": "与基础 CP-SAT 同口径的启发式多机械臂调度模型",
        "cmax": int(final_state.cmax),
        "motion_energy": int(final_state.motion_energy),
        "system_overhead_energy": int(SYSTEM_OVERHEAD_PER_ARM * len(instance.arms)),
        "total_energy": _total_energy(final_state, instance),
        "load_imbalance": _load_imbalance(final_state),
        "arm_loads": {k: int(v) for k, v in final_state.arm_loads.items()},
        "goal_programming": goal_programming,
        "schedule": schedule,
        "tasks": [asdict(t) for t in instance.tasks],
        "arms": {k: asdict(v) for k, v in instance.arms.items()},
        "available_modes_count": {str(task_id): len(modes) for task_id, modes in modes_by_task.items()},
    }
    return result
