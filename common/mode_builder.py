# -*- coding: utf-8 -*-
"""公共模式构造器。"""

from __future__ import annotations

from typing import Dict, List

from common.config import (
    BETA_LOADED_DUAL,
    BETA_LOADED_SINGLE,
    CENTER_ZONE_RADIUS,
    ENERGY_SCALE,
    GAMMA_DUAL,
    GAMMA_SINGLE,
    SAFETY_TIME,
    SERVICE_TIME_DUAL,
    SERVICE_TIME_SINGLE,
    TIME_SCALE,
    USE_CENTER_ZONE,
    V_LOADED_DUAL,
    V_LOADED_SINGLE,
)
from common.geometry import Arm, Instance, Mode, Task, arm_combinations, distance, reachable, segment_distance_to_origin


def to_ticks(seconds: float) -> int:
    return max(1, int(round(seconds * TIME_SCALE)))


def to_energy(value: float) -> int:
    return max(1, int(round(value * ENERGY_SCALE)))


def crosses_center_zone(a: tuple[float, float], b: tuple[float, float]) -> bool:
    if not USE_CENTER_ZONE:
        return False
    return segment_distance_to_origin(a, b) <= CENTER_ZONE_RADIUS


def single_duration_energy(task: Task, _arm: Arm) -> tuple[int, int]:
    d = distance(task.pos, task.target_pos)
    seconds = d / V_LOADED_SINGLE + SERVICE_TIME_SINGLE + SAFETY_TIME
    energy = BETA_LOADED_SINGLE * task.weight * d + GAMMA_SINGLE * task.weight
    return to_ticks(seconds), to_energy(energy)


def dual_duration_energy(task: Task) -> tuple[int, int]:
    d = distance(task.pos, task.target_pos)
    seconds = d / V_LOADED_DUAL + SERVICE_TIME_DUAL + SAFETY_TIME
    energy = BETA_LOADED_DUAL * task.weight * d + GAMMA_DUAL * task.weight
    return to_ticks(seconds), to_energy(energy)


def build_modes(instance: Instance) -> Dict[int, List[Mode]]:
    """为每个任务构造所有可行模式。

    2 臂场景：Type3/Type4 只能由 arm1+arm2 执行。
    3 臂场景：Type3/Type4 可以选择 arm1+arm2、arm1+arm3 或 arm2+arm3。
    """
    modes: Dict[int, List[Mode]] = {}

    for task in instance.tasks:
        task_modes: List[Mode] = []
        if task.required_arms == 1:
            for arm_name, arm in instance.arms.items():
                if reachable(arm, task.pos) and reachable(arm, task.target_pos):
                    duration, energy = single_duration_energy(task, arm)
                    task_modes.append(
                        Mode(
                            mode_id=f"T{task.task_id}_{arm_name}",
                            task_id=task.task_id,
                            arms=(arm_name,),
                            duration=duration,
                            energy=energy,
                            uses_center_zone=crosses_center_zone(task.pos, task.target_pos),
                            description=f"任务{task.task_id}由{arm_name}单臂执行",
                        )
                    )
        else:
            if len(instance.arms) < task.required_arms:
                raise RuntimeError(f"任务{task.task_id}需要{task.required_arms}只机械臂，但当前机械臂数量不足。")
            duration, energy = dual_duration_energy(task)
            for arm_tuple in arm_combinations(instance.arms, task.required_arms):
                if all(reachable(instance.arms[a], task.pos) and reachable(instance.arms[a], task.target_pos) for a in arm_tuple):
                    arm_tag = "+".join(arm_tuple)
                    task_modes.append(
                        Mode(
                            mode_id=f"T{task.task_id}_{arm_tag}",
                            task_id=task.task_id,
                            arms=tuple(arm_tuple),
                            duration=duration,
                            energy=energy,
                            uses_center_zone=crosses_center_zone(task.pos, task.target_pos),
                            description=f"任务{task.task_id}由{arm_tag}协同执行",
                        )
                    )

        if not task_modes:
            raise RuntimeError(f"任务 {task.task_id} 没有可行执行模式，请检查机械臂数量或可达半径。")
        modes[task.task_id] = task_modes
    return modes
