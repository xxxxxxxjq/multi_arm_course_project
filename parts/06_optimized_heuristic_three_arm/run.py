# -*- coding: utf-8 -*-
"""第六部分：运筹学 + 启发式算法 三机械臂调度：独立运行入口。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.experiment_runner import main_for_scenario  # noqa: E402


if __name__ == "__main__":
    main_for_scenario(
        scenario_id="06_optimized_heuristic_three_arm",
        scenario_name="第六部分：运筹学 + 启发式算法 三机械臂调度",
        arm_count=3,
        algorithm="optimized_heuristic2",
    )
