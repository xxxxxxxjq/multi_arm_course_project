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

# ------------------------------
# 中心危险区显示参数
# ------------------------------
# USE_CENTER_ZONE 只用于标记任务路径是否经过中心区域，方便图片和表格说明。
# 注意：本工程的 MuJoCo 部分现在是静态建模展示，不播放真实抓取动作。
# 因此中心区不再作为硬互斥约束，否则随机实例会被过度限制，出现本来可以排队完成却被判为不可行的问题。
USE_CENTER_ZONE = True
CENTER_ZONE_RADIUS = 0.050
ENFORCE_CENTER_ZONE_EXCLUSION = False

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
