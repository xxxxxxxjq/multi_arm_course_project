# -*- coding: utf-8 -*-
"""公共可视化函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager

from common.config import ARM_CN_NAME, BOX_POSITIONS, CENTER_ZONE_RADIUS, FIG_DPI, FONT_FAMILY, WORK_RADIUS
from common.geometry import Instance, circle_points
from common.utils import arm_label, arms_label


def setup_figure_font() -> None:
    """统一图片字体：中文尽量宋体，英文和数字尽量 Times New Roman。"""
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["axes.unicode_minus"] = False


def save_close(fig, save_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    return save_path


def plot_instance(instance: Instance, result: Optional[dict], save_path: str | Path) -> Path:
    """绘制作业区域和任务分配图。"""
    setup_figure_font()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    xs, ys = circle_points(WORK_RADIUS)
    ax.plot(xs, ys, linewidth=1.5, label="作业区域")
    ax.fill(xs, ys, alpha=0.08)

    dx, dy = circle_points(CENTER_ZONE_RADIUS)
    ax.plot(dx, dy, linestyle=":", linewidth=1.2, label="中心参考区")

    for name, arm in instance.arms.items():
        ax.scatter([arm.x], [arm.y], s=140, marker="s")
        ax.text(arm.x, arm.y + 0.018, arm_label(name), ha="center", fontsize=10)

    for box_id, (x, y) in BOX_POSITIONS.items():
        ax.scatter([x], [y], s=150, marker="D")
        ax.text(x, y + 0.018, f"收集盒{box_id}", ha="center", fontsize=10)

    assign_by_task = {}
    if result:
        for item in result["schedule"]:
            assign_by_task[item["task_id"]] = item

    for task in instance.tasks:
        ax.scatter([task.x], [task.y], s=160, marker="o")
        ax.text(task.x, task.y + 0.012, f"任务{task.task_id}/类型{task.block_type}", ha="center", fontsize=9)
        tx, ty = task.target_pos
        ax.plot([task.x, tx], [task.y, ty], linestyle="--", linewidth=1.0, alpha=0.5)
        if task.task_id in assign_by_task:
            item = assign_by_task[task.task_id]
            center = "，经过中心区" if item.get("uses_center_zone", False) else ""
            ax.text(task.x, task.y - 0.018, arms_label(item["arms"]) + center, ha="center", fontsize=8)

    ax.set_title("多机械臂分类搬运任务分布与调度分配结果")
    ax.set_xlabel("横向坐标 / m")
    ax.set_ylabel("纵向坐标 / m")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    return save_close(fig, save_path)


def plot_gantt(result: dict, save_path: str | Path) -> Path:
    """绘制甘特图。"""
    setup_figure_font()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    arms = list(result["arm_loads"].keys())
    y_map = {arm: i for i, arm in enumerate(arms)}
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    for item in result["schedule"]:
        for arm in item["arms"]:
            y = y_map[arm]
            ax.barh(y, item["duration"], left=item["start"], height=0.45, edgecolor="black")
            mark = "*" if item.get("uses_center_zone", False) else ""
            ax.text(item["start"] + item["duration"] / 2, y, f"任务{item['task_id']}{mark}", ha="center", va="center", fontsize=9)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([arm_label(a) for a in arms])
    ax.set_xlabel("时间 / tick")
    ax.set_title(
        f"调度甘特图：最大完工时间={result['cmax']}，总能耗={result['total_energy']}，"
        "空白间隔表示序列相关转移或等待，*表示路径经过中心参考区"
    )
    ax.grid(True, axis="x", alpha=0.25)
    return save_close(fig, save_path)


def plot_metrics(result: dict, save_path: str | Path) -> Path:
    """绘制优化指标柱状图。"""
    setup_figure_font()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["最大完工时间", "总能耗", "负载差"]
    values = [result["cmax"], result["total_energy"], result["load_imbalance"]]

    gp = result.get("goal_programming") or {}
    if "d_time_plus" in gp:
        labels += ["时间正偏差", "能耗正偏差", "均衡正偏差"]
        values += [gp["d_time_plus"], gp["d_energy_plus"], gp["d_balance_plus"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(labels, values, edgecolor="black")
    ax.set_title("优化指标与目标规划正偏差")
    ax.set_ylabel("指标值")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(v), ha="center", va="bottom", fontsize=9)
    return save_close(fig, save_path)
