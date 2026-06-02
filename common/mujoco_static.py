# -*- coding: utf-8 -*-
"""公共 MuJoCo 静态工作空间窗口。

本文件只负责“静态建模展示”，不负责真实抓取仿真。
运行各部分的 view_workspace.py 时，会自动读取对应部分最新的 result_*.json，
然后打开 MuJoCo Viewer 窗口，显示：
1. 本次主程序实际生成的方块数量和随机位置；
2. 四类收集盒和圆形工作区；
3. SO-ARM100 机械臂 STL 模型；
4. 机械臂保持安全收缩/待机姿态，避免视觉上出现互相碰撞。

注意：
- 这里使用 common/assets/ 中的 SO-ARM100 STL 文件；
- 不生成 static_scene_*.xml；
- 不截图；
- 不播放动画；
- 只弹出静态 MuJoCo 窗口。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

from common.config import (
    BLOCK_HALF_SIZE,
    BLOCK_RGBA,
    BOX_POSITIONS,
    CENTER_ZONE_RADIUS,
    MUJOCO_BLOCK_Z_OFFSET,
    OUTPUT_DIR,
    ROOT_DIR,
    WORK_RADIUS,
)

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 MuJoCo。请先运行：pip install mujoco") from exc


ASSET_DIR = ROOT_DIR / "common" / "assets"

# SO-ARM100 安全收缩姿态。
# 这个姿态只用于静态展示：两臂不伸向中心区域，从而避免画面上看起来互相碰撞。
SOARM_RETRACTED_QPOS = {
    "Rotation": 0.0,
    "Pitch": -3.20,
    "Elbow": 3.00,
    "Wrist_Pitch": 1.20,
    "Wrist_Roll": 0.0,
    "Jaw": 0.55,
}


def load_json(path: Path) -> dict:
    """读取 result_*.json。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_result(scenario_id: str) -> Path:
    """自动寻找某个实验部分最新生成的 result_*.json。"""
    result_dir = OUTPUT_DIR / scenario_id / "results"
    files = sorted(result_dir.glob("result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"没有找到结果文件，请先运行 parts/{scenario_id}/run.py")
    return files[0]


def fmt(x: float) -> str:
    """MuJoCo XML 中统一保留 6 位小数。"""
    return f"{x:.6f}"


def rgba_for_type(block_type: int) -> str:
    """根据方块类型返回颜色。"""
    return BLOCK_RGBA.get(block_type, "0.8 0.8 0.8 1")


def build_mesh_assets_xml() -> str:
    """加载 SO-ARM100 STL 网格模型。"""
    return """
    <mesh name="Base" file="Base.stl" />
    <mesh name="Base_Motor" file="Base_Motor.stl" />
    <mesh name="Rotation_Pitch" file="Rotation_Pitch.stl" />
    <mesh name="Rotation_Pitch_Motor" file="Rotation_Pitch_Motor.stl" />
    <mesh name="Upper_Arm" file="Upper_Arm.stl" />
    <mesh name="Upper_Arm_Motor" file="Upper_Arm_Motor.stl" />
    <mesh name="Lower_Arm" file="Lower_Arm.stl" />
    <mesh name="Lower_Arm_Motor" file="Lower_Arm_Motor.stl" />
    <mesh name="Wrist_Pitch_Roll" file="Wrist_Pitch_Roll.stl" />
    <mesh name="Wrist_Pitch_Roll_Motor" file="Wrist_Pitch_Roll_Motor.stl" />
    <mesh name="Fixed_Jaw" file="Fixed_Jaw.stl" />
    <mesh name="Fixed_Jaw_Motor" file="Fixed_Jaw_Motor.stl" />
    <mesh name="Fixed_Jaw_Collision_1" file="Fixed_Jaw_Collision_1.stl" />
    <mesh name="Fixed_Jaw_Collision_2" file="Fixed_Jaw_Collision_2.stl" />
    <mesh name="Moving_Jaw" file="Moving_Jaw.stl" />
    <mesh name="Moving_Jaw_Collision_1" file="Moving_Jaw_Collision_1.stl" />
    <mesh name="Moving_Jaw_Collision_2" file="Moving_Jaw_Collision_2.stl" />
    <mesh name="Moving_Jaw_Collision_3" file="Moving_Jaw_Collision_3.stl" />
"""


def build_default_xml() -> str:
    """定义 SO-ARM100 关节范围、显示几何和碰撞几何的默认属性。"""
    return """
  <default>
    <default class="so_arm100">
      <joint frictionloss="0.1" armature="0.1" />
      <default class="Rotation">
        <joint axis="0 1 0" range="-1.92 1.92" />
      </default>
      <default class="Pitch">
        <joint axis="1 0 0" range="-3.32 0.174" />
      </default>
      <default class="Elbow">
        <joint axis="1 0 0" range="-0.174 3.14" />
      </default>
      <default class="Wrist_Pitch">
        <joint axis="1 0 0" range="-1.66 1.66" />
      </default>
      <default class="Wrist_Roll">
        <joint axis="0 1 0" range="-2.79 2.79" />
      </default>
      <default class="Jaw">
        <joint axis="0 0 1" range="-0.174 1.75" />
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" density="0" group="2" material="white" />
        <default class="motor_visual">
          <geom material="black" />
        </default>
      </default>
      <default class="collision">
        <geom group="3" type="mesh" material="white" contype="0" conaffinity="0" />
        <default class="finger_collision">
          <geom type="box" contype="0" conaffinity="0" />
        </default>
      </default>
    </default>
  </default>
"""


def arm_base_euler_z(x: float, y: float) -> float:
    """让机械臂基座朝向圆心。

    旧版双臂模型中，基座角度大致等于“指向圆心的方向 + 90°”。
    这里把这个规则推广到三臂场景。
    """
    return math.atan2(-y, -x) + math.pi / 2.0


def build_soarm_body_xml(prefix: str, x: float, y: float) -> str:
    """生成一只 SO-ARM100 的 MuJoCo 结构。

    prefix 使用 arm1、arm2、arm3，保证四个实验部分命名统一。
    """
    euler_z = arm_base_euler_z(x, y)
    return f'''
    <body name="{prefix}_Base" childclass="so_arm100" pos="{fmt(x)} {fmt(y)} 0.000" euler="0 0 {fmt(euler_z)}">
      <geom type="mesh" mesh="Base" class="visual" />
      <geom type="mesh" mesh="Base_Motor" class="motor_visual" />
      <geom type="mesh" mesh="Base" class="collision" />
      <body name="{prefix}_Rotation_Pitch" pos="0 -0.0452 0.0165" quat="0.707105 0.707108 0 0">
        <inertial pos="-9.07886e-05 0.0590972 0.031089" quat="0.363978 0.441169 -0.623108 0.533504" mass="0.119226" diaginertia="5.94278e-05 5.89975e-05 3.13712e-05" />
        <joint name="{prefix}_Rotation" class="Rotation" />
        <geom type="mesh" mesh="Rotation_Pitch" class="visual" />
        <geom type="mesh" mesh="Rotation_Pitch_Motor" class="motor_visual" />
        <geom type="mesh" mesh="Rotation_Pitch" class="collision" />
        <body name="{prefix}_Upper_Arm" pos="0 0.1025 0.0306" euler="1.57079 0 0">
          <inertial pos="-1.72052e-05 0.0701802 0.00310545" quat="0.50104 0.498994 -0.493562 0.50632" mass="0.162409" diaginertia="0.000213312 0.000167164 7.01522e-05" />
          <joint name="{prefix}_Pitch" class="Pitch" />
          <geom type="mesh" mesh="Upper_Arm" class="visual" />
          <geom type="mesh" mesh="Upper_Arm_Motor" class="motor_visual" />
          <geom type="mesh" mesh="Upper_Arm" class="collision" />
          <body name="{prefix}_Lower_Arm" pos="0 0.11257 0.028" euler="-1.57079 0 0">
            <inertial pos="-0.00339604 0.00137796 0.0768007" quat="0.701995 0.0787996 0.0645626 0.704859" mass="0.147968" diaginertia="0.000138803 0.000107748 4.84242e-05" />
            <joint name="{prefix}_Elbow" class="Elbow" />
            <geom type="mesh" mesh="Lower_Arm" class="visual" />
            <geom type="mesh" mesh="Lower_Arm_Motor" class="motor_visual" />
            <geom type="mesh" mesh="Lower_Arm" class="collision" />
            <body name="{prefix}_Wrist_Pitch_Roll" pos="0 0.0052 0.1349" euler="-1.57079 0 0">
              <inertial pos="-0.00852653 -0.0352279 -2.34622e-05" quat="-0.0522806 0.705235 0.0549524 0.704905" mass="0.0661321" diaginertia="3.45403e-05 2.39041e-05 1.94704e-05" />
              <joint name="{prefix}_Wrist_Pitch" class="Wrist_Pitch" />
              <geom type="mesh" mesh="Wrist_Pitch_Roll" class="visual" />
              <geom type="mesh" mesh="Wrist_Pitch_Roll_Motor" class="motor_visual" />
              <geom type="mesh" mesh="Wrist_Pitch_Roll" class="collision" />
              <body name="{prefix}_Fixed_Jaw" pos="0 -0.0601 0" euler="0 1.57079 0">
                <inertial pos="0.00552377 -0.0280167 0.000483583" quat="0.41836 0.620891 -0.350644 0.562599" mass="0.0929859" diaginertia="5.03136e-05 4.64098e-05 2.72961e-05" />
                <joint name="{prefix}_Wrist_Roll" class="Wrist_Roll" />
                <geom type="mesh" mesh="Fixed_Jaw" class="visual" />
                <geom type="mesh" mesh="Fixed_Jaw_Motor" class="motor_visual" />
                <geom type="mesh" mesh="Fixed_Jaw_Collision_1" class="collision" />
                <geom type="mesh" mesh="Fixed_Jaw_Collision_2" class="collision" />
                <geom class="finger_collision" name="{prefix}_fixed_jaw_pad_1" size="0.001 0.005 0.004" pos="0.0089 -0.1014 0" />
                <geom class="finger_collision" name="{prefix}_fixed_jaw_pad_2" size="0.001 0.005 0.006" pos="0.0109 -0.0914 0" />
                <geom class="finger_collision" name="{prefix}_fixed_jaw_pad_3" size="0.001 0.01 0.007" pos="0.0126 -0.0768 0" />
                <geom class="finger_collision" name="{prefix}_fixed_jaw_pad_4" size="0.001 0.01 0.008" pos="0.0143 -0.0572 0" />
                <body name="{prefix}_Moving_Jaw" pos="-0.0202 -0.0244 0" quat="1.34924e-11 -3.67321e-06 1 -3.67321e-06">
                  <inertial pos="-0.00161745 -0.0303473 0.000449646" quat="0.696562 0.716737 -0.0239844 -0.0227026" mass="0.0202444" diaginertia="1.11265e-05 8.99651e-06 2.99548e-06" />
                  <joint name="{prefix}_Jaw" class="Jaw" />
                  <geom type="mesh" mesh="Moving_Jaw" class="visual" />
                  <geom type="mesh" mesh="Moving_Jaw_Collision_1" class="collision" />
                  <geom type="mesh" mesh="Moving_Jaw_Collision_2" class="collision" />
                  <geom type="mesh" mesh="Moving_Jaw_Collision_3" class="collision" />
                  <geom class="finger_collision" name="{prefix}_moving_jaw_pad_1" size="0.001 0.005 0.004" pos="-0.0113 -0.077 0" />
                  <geom class="finger_collision" name="{prefix}_moving_jaw_pad_2" size="0.001 0.005 0.006" pos="-0.0093 -0.067 0" />
                  <geom class="finger_collision" name="{prefix}_moving_jaw_pad_3" size="0.001 0.01 0.006" pos="-0.0073 -0.055 0" />
                  <geom class="finger_collision" name="{prefix}_moving_jaw_pad_4" size="0.001 0.01 0.008" pos="-0.0073 -0.035 0" />
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>'''


def build_blocks_xml(result: dict) -> str:
    """按 result 中真实任务数量和位置生成方块。"""
    tasks = result.get("tasks") or result.get("schedule") or []
    lines = []
    for task in tasks:
        task_id = int(task["task_id"])
        block_type = int(task["block_type"])
        if "x" in task and "y" in task:
            x, y = float(task["x"]), float(task["y"])
        else:
            x, y = float(task["pick"][0]), float(task["pick"][1])
        half = float(task.get("half_size", BLOCK_HALF_SIZE.get(block_type, 0.015)))
        z = half + MUJOCO_BLOCK_Z_OFFSET
        lines.append(
            f'''
    <body name="block_task{task_id}_type{block_type}" pos="{fmt(x)} {fmt(y)} {fmt(z)}">
      <geom name="block_task{task_id}_type{block_type}_geom" type="box" size="{fmt(half)} {fmt(half)} {fmt(half)}" rgba="{rgba_for_type(block_type)}" />
    </body>'''
        )
    return "\n".join(lines)


def build_boxes_xml() -> str:
    """生成四个分类收集盒。"""
    lines = []
    rgba = {
        1: "1 0.2 0.2 0.55",
        2: "0.2 0.4 1 0.55",
        3: "0.2 1 0.3 0.55",
        4: "1 0.8 0.1 0.55",
    }
    for box_id, (x, y) in BOX_POSITIONS.items():
        size = "0.035 0.035 0.018" if box_id in (1, 2) else "0.042 0.042 0.018"
        lines.append(
            f'<geom name="collect_box_{box_id}" type="box" pos="{fmt(x)} {fmt(y)} 0.018" size="{size}" rgba="{rgba[box_id]}" contype="0" conaffinity="0" />'
        )
    return "\n    ".join(lines)


def build_scene_xml(result: dict) -> str:
    """根据求解结果动态生成完整 MuJoCo XML 字符串。"""
    if not ASSET_DIR.exists():
        raise FileNotFoundError(f"找不到 SO-ARM100 网格文件夹：{ASSET_DIR}")

    arms = result.get("arms", {})
    arm_lines = []
    for name, arm in arms.items():
        arm_lines.append(build_soarm_body_xml(name, float(arm["x"]), float(arm["y"])))

    meshdir = ASSET_DIR.resolve().as_posix()
    return f'''<mujoco model="static_multi_arm_workspace">
  <compiler angle="radian" meshdir="{meshdir}" />
  <option timestep="0.01" cone="elliptic" impratio="10" gravity="0 0 -9.81" />
  <asset>
    <material name="white" rgba="1 1 1 1" />
    <material name="black" rgba="0.1 0.1 0.1 1" />
    <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.90 0.90 0.90" rgb2="0.75 0.75 0.75" />
    <material name="grid_mat" texture="grid_tex" texrepeat="4 4" />
    {build_mesh_assets_xml()}
  </asset>
  {build_default_xml()}
  <worldbody>
    <light pos="0 0 1.6" dir="0 0 -1" directional="true" />
    <geom name="floor" type="plane" size="0.7 0.7 0.02" material="grid_mat" />
    <geom name="work_area" type="cylinder" pos="0 0 0.003" size="{fmt(WORK_RADIUS)} 0.003" rgba="0.2 0.6 1 0.18" contype="0" conaffinity="0" />
    <geom name="center_safe_area" type="cylinder" pos="0 0 0.007" size="{fmt(CENTER_ZONE_RADIUS)} 0.003" rgba="1.0 0.25 0.15 0.16" contype="0" conaffinity="0" />
    {build_boxes_xml()}
    {''.join(arm_lines)}
    {build_blocks_xml(result)}
  </worldbody>
</mujoco>'''


def set_joint_qpos(model, data, joint_name: str, value: float) -> None:
    """把单个关节设置到指定位置。"""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        return
    qpos_addr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_addr] = value


def set_retracted_arm_pose(model, data, result: dict) -> None:
    """把所有 SO-ARM100 设置为安全收缩姿态。"""
    for arm_name in result.get("arms", {}).keys():
        for joint_suffix, value in SOARM_RETRACTED_QPOS.items():
            set_joint_qpos(model, data, f"{arm_name}_{joint_suffix}", value)
    mujoco.mj_forward(model, data)


def print_summary(scenario_id: str, result_path: Path, result: dict) -> None:
    """运行前打印摘要，方便确认不是误运行旧文件。"""
    print("\n========== MuJoCo 静态建模窗口：SO-ARM100 收缩姿态版 ==========")
    print(f"实验部分：{scenario_id}")
    print(f"读取结果：{result_path}")
    print(f"方块总数：{len(result.get('tasks', []))}")
    print(f"机械臂数量：{len(result.get('arms', {}))}")
    print(f"网格文件夹：{ASSET_DIR}")
    print("说明：窗口中机械臂为 SO-ARM100 STL 模型，并保持安全收缩姿态。")


def open_viewer(model, data) -> None:
    """打开 MuJoCo 窗口。"""
    print("正在打开 MuJoCo Viewer。关闭窗口即可结束。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 静态展示只需要 mj_forward，不做 mj_step，避免机械臂因动力学掉落或抖动。
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.03)


def run_viewer_for_scenario(scenario_id: str, result_path: Path | None = None) -> None:
    """读取结果并打开静态窗口。"""
    if result_path is None:
        result_path = find_latest_result(scenario_id)
    result = load_json(result_path)
    print_summary(scenario_id, result_path, result)
    xml = build_scene_xml(result)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    set_retracted_arm_pose(model, data, result)
    open_viewer(model, data)


def main_for_scenario(scenario_id: str, argv: Sequence[str] | None = None) -> None:
    """供各部分 view_workspace.py 调用的入口。"""
    parser = argparse.ArgumentParser(description="打开当前实验部分最新结果对应的 MuJoCo 静态窗口")
    parser.add_argument("--result", type=str, default=None, help="指定 result_*.json；不填则自动读取本实验部分最新结果")
    args = parser.parse_args(argv)
    result_path = Path(args.result).resolve() if args.result else None
    run_viewer_for_scenario(scenario_id, result_path)
