# -*- coding: utf-8 -*-
"""第三部分：运筹学方法 双机械臂调度：独立运行入口。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.experiment_runner import main_for_scenario  # noqa: E402


if __name__ == "__main__":
    main_for_scenario(
        scenario_id="03_optimized_two_arm",
        scenario_name="第三部分：优化方法 双机械臂调度",
        arm_count=2,
        algorithm="optimized",
    )
