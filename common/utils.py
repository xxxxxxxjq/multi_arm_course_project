# -*- coding: utf-8 -*-
"""公共输入输出函数。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from common.config import ARM_CN_NAME


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def arm_label(arm_name: str) -> str:
    return ARM_CN_NAME.get(arm_name, arm_name)


def arms_label(arms: Iterable[str]) -> str:
    return "+".join(arm_label(a) for a in arms)


def schedule_to_rows(result: dict) -> list[dict]:
    rows = []
    for item in result["schedule"]:
        rows.append(
            {
                "任务编号": item["task_id"],
                "方块类型": f"类型{item['block_type']}",
                "任务类别": "单臂" if item["task_type"] == "single" else "双臂协同",
                "执行机械臂": arms_label(item["arms"]),
                "开始时间": item["start"],
                "结束时间": item["end"],
                "前置转移时间": item.get("setup_before_max", 0),
                "处理时间": item.get("process_duration", item["duration"]),
                "处理能耗": item.get("process_energy", item["energy"]),
                "是否经过中心区": "是" if item.get("uses_center_zone", False) else "否",
                "目标盒": f"收集盒{item['target_box']}",
            }
        )
    return rows


def save_csv(rows: Iterable[dict], path: str | Path) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_schedule(result: dict) -> None:
    print("\n========== 优化结果 ==========")
    print(f"模型名称：{result.get('model_name', '多机械臂调度模型')}")
    print(f"求解算法：{result.get('algorithm_name', '基础方法')}")
    print(f"求解状态：{result['status']}")
    print(f"最大完工时间：{result['cmax']} tick")
    print(f"总能耗：{result['total_energy']}")
    print("机械臂负载：" + "，".join(f"{arm_label(k)}={v}" for k, v in result["arm_loads"].items()))
    print(f"负载差：{result['load_imbalance']}")

    gp = result.get("goal_programming")
    if gp:
        print("\n目标规划 / 求解说明：")
        if "time_goal" in gp:
            print(f"时间目标={gp['time_goal']}，时间正偏差={gp.get('d_time_plus', '-')}")
        if "energy_goal" in gp:
            print(f"能耗目标={gp['energy_goal']}，能耗正偏差={gp.get('d_energy_plus', '-')}")
        print(f"方法说明：{gp.get('method', '')}")

    print("\n任务调度表：")
    print("任务  类型    执行机械臂              前置转移  开始   结束   处理   能耗  中心区")
    print("-" * 96)
    for item in result["schedule"]:
        print(
            f"{item['task_id']:>2}    "
            f"类型{item['block_type']:<2}  "
            f"{arms_label(item['arms']):<22} "
            f"{item.get('setup_before_max', 0):>8} "
            f"{item['start']:>5} "
            f"{item['end']:>6} "
            f"{item.get('process_duration', item['duration']):>6} "
            f"{item.get('process_energy', item['energy']):>6}  "
            f"{'是' if item.get('uses_center_zone', False) else '否'}"
        )
