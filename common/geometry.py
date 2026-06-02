# -*- coding: utf-8 -*-
"""公共几何函数和基础数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import cos, hypot, pi, sin, sqrt
from typing import Dict, Iterable, List, Tuple

from common.config import ARM_LAYOUTS, BOX_POSITIONS, REACH_RADIUS

Point = Tuple[float, float]


@dataclass(frozen=True)
class Arm:
    """机械臂资源。"""

    name: str
    x: float
    y: float

    @property
    def pos(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class Task:
    """一个方块搬运任务。"""

    task_id: int
    block_type: int
    x: float
    y: float
    weight: float
    task_type: str
    required_arms: int
    target_box: int
    half_size: float = 0.015

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    @property
    def mujoco_name(self) -> str:
        """静态场景中使用的方块名称。

        名称只由任务编号和类型组成，保证每个方块唯一。
        """
        return f"block_t{self.task_id}_type{self.block_type}"

    @property
    def target_pos(self) -> Point:
        return BOX_POSITIONS[self.target_box]


@dataclass(frozen=True)
class Mode:
    """任务执行模式。

    single 任务通常有多个单臂模式；dual 任务在三臂场景中有多个两臂协同模式。
    """

    mode_id: str
    task_id: int
    arms: Tuple[str, ...]
    duration: int
    energy: int
    uses_center_zone: bool
    description: str


@dataclass
class Instance:
    """一次完整实验实例。"""

    seed: int
    tasks: List[Task]
    arms: Dict[str, Arm]
    arm_count: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "arm_count": self.arm_count,
            "tasks": [asdict(t) for t in self.tasks],
            "arms": {k: asdict(v) for k, v in self.arms.items()},
        }


def distance(p1: Point, p2: Point) -> float:
    """二维欧氏距离。"""
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def reachable(arm: Arm, point: Point, reach: float = REACH_RADIUS) -> bool:
    """判断 point 是否在机械臂等效可达半径内。"""
    return distance(arm.pos, point) <= reach


def make_arms(arm_count: int) -> Dict[str, Arm]:
    """创建 2 臂或 3 臂资源。

    二臂场景采用左右对称布局；三臂场景采用圆周 120° 均匀布局。
    这样可以保证调度模型、Matplotlib 工作空间图和 MuJoCo 静态仿真读取到同一套机械臂坐标。
    """
    if arm_count not in ARM_LAYOUTS:
        raise ValueError("当前工程只整理了 2 臂和 3 臂两种场景。")
    layout = ARM_LAYOUTS[arm_count]
    return {name: Arm(name, *pos) for name, pos in layout.items()}


def arm_combinations(arms: Dict[str, Arm], k: int) -> Iterable[Tuple[str, ...]]:
    """返回 k 只机械臂的所有组合。"""
    return combinations(arms.keys(), k)


def circle_points(radius: float, n: int = 240) -> Tuple[List[float], List[float]]:
    """生成圆形边界点，用于画图。"""
    xs, ys = [], []
    for i in range(n + 1):
        theta = 2.0 * pi * i / n
        xs.append(radius * cos(theta))
        ys.append(radius * sin(theta))
    return xs, ys


def min_separation_for_blocks(half_a: float, half_b: float, margin: float = 0.012) -> float:
    """两个方块中心之间的最小安全间距。"""
    return sqrt(2.0) * (half_a + half_b) + margin


def segment_distance_to_origin(a: Point, b: Point) -> float:
    """原点到线段 ab 的最短距离。"""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy
    if length_sq == 0:
        return distance((0.0, 0.0), a)
    t = -(ax * vx + ay * vy) / length_sq
    t = max(0.0, min(1.0, t))
    px, py = ax + t * vx, ay + t * vy
    return distance((0.0, 0.0), (px, py))
