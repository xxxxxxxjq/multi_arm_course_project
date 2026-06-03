# -*- coding: utf-8 -*-
"""公共模式构造器。"""

from __future__ import annotations

from typing import Dict, List

from common.config import (
    BETA_LOADED_DUAL,
    BETA_LOADED_SINGLE,
    BLOCK_TYPE_ENERGY,
    CENTER_ZONE_RADIUS,
    ENERGY_SCALE,
    GAMMA_DUAL,
    GAMMA_SINGLE,
    SAFETY_TIME,
    SERVICE_TIME_DUAL,
    SERVICE_TIME_SINGLE,
    SYNC_COORD_FACTOR_PER_EXTRA_ARM,
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


def _type_energy_params(task: Task) -> dict:
    """读取物块类型相关能耗参数。"""
    return BLOCK_TYPE_ENERGY.get(
        task.block_type,
        {"payload_factor": 1.0, "hold_power": 0.0, "sync_energy": 0.0},
    )


def _coordination_factor(arm_count: int) -> float:
    """计算双臂协同任务在不同系统臂数下的协调系数。

    二臂系统：factor = 1。
    三臂系统：factor = 1 + SYNC_COORD_FACTOR_PER_EXTRA_ARM。

    这样做的目的不是改变任务可行性，而是让 Type3/Type4 这类双臂协同任务
    在三臂系统下体现额外同步、调度和安全协调代价。
    """
    return 1.0 + SYNC_COORD_FACTOR_PER_EXTRA_ARM * max(0, arm_count - 2)


def single_duration_energy(task: Task, _arm: Arm) -> tuple[int, int]:
    """计算单臂任务的处理时间和处理能耗。

    能耗模型：
        单臂处理能耗 = 负载搬运能耗 + 抓放服务能耗 + 保持能耗

    其中负载搬运能耗和抓放服务能耗会乘以 payload_factor。
    因此 Type2 这种“小但较重”的物块，会比 Type1 产生更高能耗。
    """
    params = _type_energy_params(task)
    payload_factor = float(params["payload_factor"])
    hold_power = float(params["hold_power"])

    d = distance(task.pos, task.target_pos)
    seconds = d / V_LOADED_SINGLE + SERVICE_TIME_SINGLE + SAFETY_TIME

    loaded_energy = BETA_LOADED_SINGLE * task.weight * d * payload_factor
    service_energy = GAMMA_SINGLE * task.weight * payload_factor
    hold_energy = hold_power * seconds

    energy = loaded_energy + service_energy + hold_energy
    return to_ticks(seconds), to_energy(energy)


def dual_duration_energy(task: Task, arm_count: int) -> tuple[int, int]:
    """计算双臂协同任务的处理时间和处理能耗。

    能耗模型：
        双臂处理能耗 = 负载搬运能耗 + 抓放服务能耗 + 保持能耗 + 协同同步能耗

    与旧模型相比，新增了两类差异化因素：
    1. payload_factor：体现物块重量/搬运难度；
    2. sync_energy × coordination_factor：体现大物块双臂协同和三臂系统协调代价。

    这能让 Type3/Type4 在二臂和三臂之间产生更明显的能耗差异，
    而不是仅仅让三臂比二臂多一个固定开销。
    """
    params = _type_energy_params(task)
    payload_factor = float(params["payload_factor"])
    hold_power = float(params["hold_power"])
    sync_energy = float(params["sync_energy"])

    d = distance(task.pos, task.target_pos)
    seconds = d / V_LOADED_DUAL + SERVICE_TIME_DUAL + SAFETY_TIME

    loaded_energy = BETA_LOADED_DUAL * task.weight * d * payload_factor
    service_energy = GAMMA_DUAL * task.weight * payload_factor

    # 双臂任务由两只臂共同夹持和搬运，保持能耗按参与机械臂数量计入。
    hold_energy = hold_power * seconds * task.required_arms

    # 三臂系统下，虽然每个双臂任务仍只用两只臂，但可选组合和安全协调更复杂，
    # 因此对 Type3/Type4 增加系统协调系数。
    sync = sync_energy * _coordination_factor(arm_count)

    energy = loaded_energy + service_energy + hold_energy + sync
    return to_ticks(seconds), to_energy(energy)


def build_modes(instance: Instance) -> Dict[int, List[Mode]]:
    """为每个任务构造所有可行模式。

    2 臂场景：Type3/Type4 只能由 arm1+arm2 执行。
    3 臂场景：Type3/Type4 可以选择 arm1+arm2、arm1+arm3 或 arm2+arm3。

    注意：本文件只改变“模式处理能耗”的计算方式，不改变：
    - 任务是否可达；
    - 单臂/双臂任务规则；
    - 中心区约束；
    - 目标规划 d1/d2/d3 的优先级。
    """
    modes: Dict[int, List[Mode]] = {}
    arm_count = len(instance.arms)

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

            duration, energy = dual_duration_energy(task, arm_count)
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
