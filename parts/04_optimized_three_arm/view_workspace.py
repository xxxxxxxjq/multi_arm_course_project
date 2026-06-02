# -*- coding: utf-8 -*-
"""第四部分：优化方法 三机械臂调度：MuJoCo 静态窗口入口。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.mujoco_static import main_for_scenario  # noqa: E402


if __name__ == "__main__":
    main_for_scenario("04_optimized_three_arm")
