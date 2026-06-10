# -*- coding: utf-8 -*-
"""Exact optimized solver warmed by a fast heuristic incumbent.

The heuristic solution is used only as an initial upper bound for the explicit
branch-and-bound search. The exact search space and safe pruning rules remain
the ones implemented in ``solver_optimized.py``.
"""

from __future__ import annotations

import time
from typing import Dict

from common.geometry import Instance, Mode
from common import solver_optimized as exact_solver
from common import solver_optimized_heuristic as heuristic_solver


LIGHTWEIGHT_BEAM_TIME_LIMIT = 0.20
FULL_HEURISTIC_WARMSTART_MIN_TASKS = 8


def _heuristic_state_to_initial_state(
    heuristic_state: heuristic_solver.SearchState,
    instance: Instance,
) -> exact_solver.SearchState:
    """Convert a heuristic search state into an exact-solver incumbent."""
    schedule = tuple(dict(item) for item in heuristic_state.schedule)
    scheduled = frozenset(int(item["task_id"]) for item in schedule)

    center_intervals = tuple(
        (int(item["start"]), int(item["end"]), int(item["task_id"]))
        for item in schedule
        if item.get("uses_center_zone")
    )

    return exact_solver.SearchState(
        scheduled=scheduled,
        arm_available={k: int(v) for k, v in heuristic_state.arm_available.items()},
        arm_last_point=dict(heuristic_state.arm_last_point),
        arm_loads={k: int(v) for k, v in heuristic_state.arm_loads.items()},
        motion_energy=int(heuristic_state.motion_energy),
        cmax=int(heuristic_state.cmax),
        center_intervals=center_intervals,
        schedule=schedule,
    )


def _build_lightweight_heuristic_state(
    instance: Instance,
    modes_by_task: Dict[int, list[Mode]],
) -> heuristic_solver.SearchState:
    """Build a cheap incumbent for exact branch-and-bound warm start."""
    tasks = sorted(instance.tasks, key=lambda task: task.task_id)
    task_by_id = {task.task_id: task for task in tasks}

    best_state = heuristic_solver._best_initial_solution(
        instance,
        tasks,
        task_by_id,
        modes_by_task,
    )

    beam_result = heuristic_solver._run_beam_search_heuristic(
        instance,
        tasks,
        task_by_id,
        modes_by_task,
        time.perf_counter() + LIGHTWEIGHT_BEAM_TIME_LIMIT,
    )
    if beam_result is not None:
        beam_state, _beam_info = beam_result
        if heuristic_solver._score(beam_state, instance) < heuristic_solver._score(best_state, instance):
            best_state = beam_state

    return best_state


def _build_adaptive_heuristic_state(
    instance: Instance,
    modes_by_task: Dict[int, list[Mode]],
) -> tuple[heuristic_solver.SearchState, float]:
    """Use a stronger heuristic only when the exact proof is expected to be hard."""
    tasks = sorted(instance.tasks, key=lambda task: task.task_id)
    task_by_id = {task.task_id: task for task in tasks}

    if len(tasks) >= FULL_HEURISTIC_WARMSTART_MIN_TASKS and len(instance.arms) >= 3:
        heuristic_state, _info = heuristic_solver._run_lns_heuristic(
            instance,
            tasks,
            task_by_id,
            modes_by_task,
        )
        return heuristic_state, float(_info.get("time_limit_s", 0.0))

    return _build_lightweight_heuristic_state(instance, modes_by_task), float(LIGHTWEIGHT_BEAM_TIME_LIMIT)


def solve_schedule(instance: Instance, modes_by_task: Dict[int, list[Mode]]) -> dict:
    """Solve exactly, using a fast heuristic solution as the initial upper bound."""
    heuristic_state, heuristic_time_limit = _build_adaptive_heuristic_state(instance, modes_by_task)
    initial_state = _heuristic_state_to_initial_state(heuristic_state, instance)
    initial_cmax, initial_energy, initial_load = heuristic_solver._score(heuristic_state, instance)

    exact_result = exact_solver.solve_schedule(
        instance,
        modes_by_task,
        initial_state=initial_state,
    )

    goal = exact_result.setdefault("goal_programming", {})
    goal["heuristic_initial_cmax"] = int(initial_cmax)
    goal["heuristic_initial_energy"] = int(initial_energy)
    goal["heuristic_initial_load_imbalance"] = int(initial_load)
    goal["heuristic_initial_score"] = [
        int(initial_cmax),
        int(initial_energy),
        int(initial_load),
    ]
    goal["heuristic_warmstart_time_limit_s"] = float(heuristic_time_limit)
    goal["warmstart_note"] = (
        "Heuristic solution is used only as the initial incumbent; the final "
        "result is still proven by explicit branch-and-bound."
    )

    exact_result["status"] = "OPTIMAL_BY_BRANCH_AND_BOUND_WITH_HEURISTIC_INCUMBENT"
    exact_result["algorithm_name"] = "optimized exact branch-and-bound with heuristic initial incumbent"
    return exact_result
