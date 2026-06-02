# -*- coding: utf-8 -*-
"""公共任务实例生成器。"""

from __future__ import annotations

import random
from math import cos, pi, sin, sqrt
from typing import List, Sequence

from common.config import BLOCK_HALF_SIZE, BLOCK_TYPES, WORK_RADIUS
from common.geometry import Instance, Task, distance, make_arms, min_separation_for_blocks


def sample_position(rng: random.Random, half_size: float) -> tuple[float, float]:
    """在圆形作业区内均匀随机生成一个方块中心位置。"""
    block_radius = sqrt(2.0) * half_size
    max_r = WORK_RADIUS - block_radius - 0.006
    if max_r <= 0:
        raise ValueError("作业区域太小，方块放不进去。")
    r = max_r * sqrt(rng.random())
    theta = 2.0 * pi * rng.random()
    return r * cos(theta), r * sin(theta)


def generate_instance(seed: int, block_types: Sequence[int], arm_count: int, max_retry: int = 4000) -> Instance:
    """生成指定机械臂数量和指定 Type 序列的实例。"""
    rng = random.Random(seed)
    tasks: List[Task] = []

    for idx, block_type in enumerate(block_types, start=1):
        if block_type not in BLOCK_TYPES:
            raise ValueError(f"未知方块类型：{block_type}")
        cfg = BLOCK_TYPES[block_type]
        half = BLOCK_HALF_SIZE.get(block_type, 0.015)

        for _ in range(max_retry):
            x, y = sample_position(rng, half)
            ok = True
            for old in tasks:
                d_min = min_separation_for_blocks(half, old.half_size)
                if distance((x, y), old.pos) < d_min:
                    ok = False
                    break
            if ok:
                tasks.append(
                    Task(
                        task_id=idx,
                        block_type=block_type,
                        x=x,
                        y=y,
                        weight=cfg["weight"],
                        task_type=cfg["task_type"],
                        required_arms=cfg["required_arms"],
                        target_box=cfg["target_box"],
                        half_size=half,
                    )
                )
                break
        else:
            raise RuntimeError("随机生成方块失败：请换一个 seed 或减少任务数量。")

    return Instance(seed=seed, tasks=tasks, arms=make_arms(arm_count), arm_count=arm_count)
