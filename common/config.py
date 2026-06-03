# -*- coding: utf-8 -*-
"""公共参数配置。

四个实验部分共用本文件中的几何参数、时间参数、能耗参数和画图字体参数。
这样做的目的：
1. 双臂/三臂只改机械臂数量，不重复写基础模型；
2. 基础方法/优化方法只改求解算法，不重复写任务生成、画图和静态场景；
3. 所有图片和输出文件命名保持统一。
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT_DIR / "parts"
OUTPUT_DIR = ROOT_DIR / "outputs"

# ------------------------------
# 作业区域与机械臂位置
# ------------------------------
WORK_RADIUS = 0.150
REACH_RADIUS = 0.60

# 统一使用 arm1/arm2/arm3，避免一会 left/right 一会 static 的混乱命名。
# 二臂与三臂采用不同布局：二臂保持左右对称，三臂保持 120° 均匀分布。
ARM_LAYOUTS = {
    2: {
        "arm1": (-0.260, 0.000),
        "arm2": (0.260, 0.000),
    },
    3: {
        "arm1": (0.000, 0.285),
        "arm2": (-0.246817, -0.142500),
        "arm3": (0.246817, -0.142500),
    },
}

# 保留 ARM_POSITIONS 名称，方便旧代码读取。默认表示三臂场景下的 120° 布局。
ARM_POSITIONS = ARM_LAYOUTS[3]

ARM_CN_NAME = {
    "arm1": "机械臂1",
    "arm2": "机械臂2",
    "arm3": "机械臂3",
}

# 四类收集盒位置。
# 收集盒与圆心距离保持为原方案的约 0.280179 m，只调整角度位置，避免三臂 120° 布局下与机械臂基座过近。
# 二臂和三臂共用同一套收集盒位置，保证两种场景对比公平。
BOX_POSITIONS = {
    1: (0.140089, 0.242642),
    2: (0.140089, -0.242642),
    3: (-0.140089, 0.242642),
    4: (-0.140089, -0.242642),
}

# ------------------------------
# 方块类型
# ------------------------------
# Type1/Type2：单臂任务；Type3/Type4：双臂协同任务。
# 在三臂场景中，双臂协同任务可以从三只臂中选择任意两只共同执行。
BLOCK_TYPES = {
    1: {"task_type": "single", "required_arms": 1, "target_box": 1, "weight": 1.0},
    2: {"task_type": "single", "required_arms": 1, "target_box": 2, "weight": 1.5},
    3: {"task_type": "dual", "required_arms": 2, "target_box": 3, "weight": 3.0},
    4: {"task_type": "dual", "required_arms": 2, "target_box": 4, "weight": 4.0},
}

BLOCK_HALF_SIZE = {1: 0.012, 2: 0.015, 3: 0.020, 4: 0.024}
BLOCK_RGBA = {
    1: "1.0 0.2 0.2 1.0",
    2: "0.2 0.4 1.0 1.0",
    3: "0.2 1.0 0.3 1.0",
    4: "1.0 0.8 0.1 1.0",
}

# ------------------------------
# 时间模型
# ------------------------------
V_EMPTY = 0.18
V_LOADED_SINGLE = 0.12
V_LOADED_DUAL = 0.09
SERVICE_TIME_SINGLE = 0.60
SERVICE_TIME_DUAL = 0.90
SAFETY_TIME = 0.15
TIME_SCALE = 100

# ------------------------------
# 能耗模型
# ------------------------------
ALPHA_EMPTY = 3.0
BETA_LOADED_SINGLE = 5.0
BETA_LOADED_DUAL = 7.0
GAMMA_SINGLE = 0.8
GAMMA_DUAL = 1.5
ENERGY_SCALE = 100

# 系统级能耗修正项。
# 原模型中的任务能耗主要是“运动能耗”：空载移动、搬运移动和抓放动作。
# 为了更符合实际系统，本文将每只启用机械臂的上电、启动、伺服保持、控制通信等
# 等效为固定系统开销。
#
# 注意：该开销按系统启用机械臂数量计入：二臂系统计 2 份，三臂系统计 3 份。
# 因此，在相同任务结构下，三臂系统比二臂系统天然多 1 份系统开销。
# 根据已有 basic_2B3B_time_energy.csv 数据，三臂原始运动能耗最多比二臂低 61；
# 这里每只机械臂固定开销取 300，因此三臂至少多 300，能够保证已有实验中
# 修正后的三臂总能耗大于二臂总能耗。
ARM_STARTUP_ENERGY = 200       # 单只机械臂启动/上电等效能耗
ARM_STATIC_HOLD_ENERGY = 100   # 单只机械臂静态保持/控制通信等效能耗
SYSTEM_OVERHEAD_PER_ARM = ARM_STARTUP_ENERGY + ARM_STATIC_HOLD_ENERGY

# ------------------------------
# 中心安全区参数
# ------------------------------
# USE_CENTER_ZONE：是否识别任务路径是否经过中心区域。
# CENTER_ZONE_RADIUS：中心安全区半径，默认以工作台圆心为中心。
# ENFORCE_CENTER_ZONE_EXCLUSION：是否启用中心区互斥约束。
#
# 重要说明：
# 这里的中心区不是“单个机械臂互斥资源”，而是“任务组级共享资源”。
# 也就是说：
# 1. 同一个双臂协同任务中的两只机械臂可以同时进入中心区；
# 2. 不同任务组不能同时占用中心区；
# 3. 如果某任务组需要进入中心区，而中心区已被其他任务组占用，则该任务组必须等待。
#
# 这样既能体现中心安全区约束，又不会因为 Type3/Type4 双臂任务位于中心区而直接不可行。
USE_CENTER_ZONE = True
CENTER_ZONE_RADIUS = 0.050
ENFORCE_CENTER_ZONE_EXCLUSION = True
CENTER_ZONE_CONSTRAINT_LEVEL = "task_group"

# ------------------------------
# 求解器参数
# ------------------------------
SOLVER_TIME_LIMIT = 30.0
NUM_WORKERS = 8
OPTIMIZED_TIME_LIMIT = 20.0

# ------------------------------
# 图片字体规范
# ------------------------------
# Matplotlib 会按顺序寻找字体：英文字母/数字优先 Times New Roman，中文回退到 SimSun。
FONT_FAMILY = ["Times New Roman", "SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
FIG_DPI = 240

# MuJoCo 静态场景参数。
MUJOCO_BLOCK_Z_OFFSET = 0.010
