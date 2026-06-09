# 多机械臂分类搬运调度优化工程说明

本工程用于运筹学大作业，研究多机械臂在分类搬运任务中的调度优化问题。工程主要完成二臂系统和三臂系统的任务生成、可行执行模式构造、调度求解、结果可视化和批量数据导出。

本文档主要说明：

1. 打开工程后应该先看什么；
2. 如何安装环境；
3. 如何单独运行一个场景并生成图像；
4. 如何批量生成二臂/三臂对比数据；
5. 如何运行非线性规划扩展分析；
6. 当前哪些部分已经完成，哪些部分还只是接口预留。

---

## 1. 工程当前完成情况

当前工程的主线分为三部分：

| 部分 | 作用 | 当前状态 |
|---|---|---|
| 基础方法 | 使用 CP-SAT 求解整数规划/目标规划模型 | 已完成，可运行 |
| 优化方法 | 自编隐枚举—分枝定界算法，不调用 CP-SAT | 已完成，可运行，但大规模案例耗时较长 |
| 启发式方法 | 用于后续加速求解 | 接口已预留，主算法尚未完成，不建议作为正式实验结果 |
| 非线性规划扩展 | 在已有调度结果基础上分析速度—能耗权衡、KKT、鲁棒性 | 已完成脚本，可在生成优化方法 CSV 后运行 |

本工程的主问题是：

```text
整数规划 + 目标规划
```

其中：

- 整数规划用于描述任务分配、执行模式选择、机械臂资源占用和任务顺序；
- 目标规划用于描述多个目标之间的优先级；
- 非线性规划扩展不改变主调度模型，只是在固定离散调度结果后进一步研究速度倍率、时间约束和能耗之间的关系。

当前目标优先级为：

```text
第一优先级：最小化 Cmax，即最大完工时间最短；
第二优先级：在 Cmax 最优的前提下，最小化总能耗；
第三优先级：在前两个目标尽量满足的前提下，使机械臂负载更加均衡。
```

二臂/三臂对比不是直接假设谁一定更好，而是在二臂系统和三臂系统分别得到自身最优调度结果后，再比较 Cmax、总能耗和计算时间等指标。

---

## 2. 环境安装

建议使用 Python 3.10 及以上版本。

打开命令行，进入工程根目录，也就是能看到 `common/`、`parts/`、`scripts/`、`outputs/`、`requirements.txt` 的位置。

安装依赖：

```bash
pip install -r requirements.txt
```

当前依赖主要包括：

```text
ortools
matplotlib
mujoco
```

说明：

- `ortools`：基础方法中 CP-SAT 求解器需要；
- `matplotlib`：生成工作空间图、甘特图、指标图需要；
- `mujoco`：打开静态机械臂场景窗口时需要；
- 如果只生成 CSV 数据，通常重点依赖 `ortools` 和 `matplotlib`。

---

## 3. 工程目录说明

工程目录结构如下：

```text
multi_arm_course_project/
│
├─ common/
├─ parts/
├─ scripts/
├─ outputs/
├─ README.md
└─ requirements.txt
```

### 3.1 `common/`

`common/` 是公共代码目录，存放建模、求解和画图相关的通用模块。

| 文件 | 作用 |
|---|---|
| `config.py` | 公共参数配置，包括机械臂位置、收集盒位置、速度、能耗参数、字体参数等 |
| `geometry.py` | 定义机械臂、任务、执行模式、实例等数据结构，并提供距离等几何计算函数 |
| `instance_generator.py` | 根据物块类型和随机种子生成任务实例 |
| `mode_builder.py` | 为每个任务生成可行执行模式，例如单臂模式、双臂协同模式 |
| `solver_basic.py` | 基础方法求解器，使用 CP-SAT 进行整数规划/目标规划求解 |
| `solver_optimized.py` | 自编优化方法求解器，使用隐枚举、分枝定界和支配剪枝求解 |
| `solver_optimized_heuristic.py` | 启发式算法接口，目前主算法尚未完成，暂不作为正式结果 |
| `experiment_runner.py` | 单个实验场景共用运行器 |
| `visualization.py` | 生成工作空间图、甘特图、指标图 |
| `mujoco_static.py` | MuJoCo 静态展示模块 |
| `utils.py` | JSON、CSV 保存和调度结果转换工具 |

### 3.2 `parts/`

`parts/` 存放单个实验场景的独立运行入口。

```text
parts/
├─ 01_basic_two_arm/
├─ 02_basic_three_arm/
├─ 03_optimized_two_arm/
├─ 04_optimized_three_arm/
├─ 05_optimized_heuristic_two_arm/
└─ 06_optimized_heuristic_three_arm/
```

各文件夹含义如下：

| 文件夹 | 含义 | 当前建议 |
|---|---|---|
| `01_basic_two_arm/` | 双臂基础方法 | 可运行 |
| `02_basic_three_arm/` | 三臂基础方法 | 可运行 |
| `03_optimized_two_arm/` | 双臂自编优化方法 | 可运行 |
| `04_optimized_three_arm/` | 三臂自编优化方法 | 可运行 |
| `05_optimized_heuristic_two_arm/` | 双臂启发式方法 | 主算法未完成，暂不建议作为正式结果 |
| `06_optimized_heuristic_three_arm/` | 三臂启发式方法 | 主算法未完成，暂不建议作为正式结果 |

每个场景中的 `run.py` 用于运行该场景并输出结果。

部分场景中还有 `view_workspace.py`，用于读取最近一次结果并打开 MuJoCo 静态展示窗口。

### 3.3 `scripts/`

`scripts/` 存放批量实验和扩展分析脚本。

| 脚本 | 作用 | 输出位置 |
|---|---|---|
| `run_basic_case.py` | 批量生成基础方法下二臂/三臂的时间和能耗 CSV | `outputs/four_case_framework/basic_2B3B_time_energy.csv` |
| `run_optimized_case.py` | 批量生成自编优化方法下二臂/三臂的时间和能耗 CSV | `outputs/four_case_framework/optimized_2B3B_time_energy.csv` |
| `run_optimized_case_heuristic.py` | 批量生成启发式方法 CSV | 主算法未完成前不建议使用 |
| `nonlinear_speed_energy_optimization.py` | 基于优化方法 CSV 做速度—能耗非线性规划、KKT 验证和模式比较 | `outputs/nonlinear_programming/speed_energy_optimization/` |
| `nonlinear_robust_mode_selection.py` | 基于优化方法 CSV 做 seed 扰动下的鲁棒速度—能耗分析 | `outputs/nonlinear_programming/robust_mode_selection/` |

### 3.4 `outputs/`

`outputs/` 存放程序运行生成的所有结果。

常见输出目录包括：

```text
outputs/01_basic_two_arm/
outputs/02_basic_three_arm/
outputs/03_optimized_two_arm/
outputs/04_optimized_three_arm/
outputs/four_case_framework/
outputs/nonlinear_programming/
```

单次运行会输出 JSON、CSV 和 PNG 图像。

批量运行主要输出汇总 CSV。

---

## 4. 方块类型和输入格式

本工程共有四类方块：

| 类型 | 任务性质 | 说明 |
|---|---|---|
| Type1 | 单臂任务 | 小而轻，单臂搬运 |
| Type2 | 单臂任务 | 小但更重，单臂搬运 |
| Type3 | 双臂协同任务 | 大块，需要两只机械臂协同 |
| Type4 | 双臂协同任务 | 更大更重，需要两只机械臂协同 |

运行程序时，常用 `--counts` 表示四类方块数量。

例如：

```bash
--counts 1,1,1,1
```

表示：

```text
Type1 数量 = 1
Type2 数量 = 1
Type3 数量 = 1
Type4 数量 = 1
```

批量脚本中常用 `counts_code` 表示四类方块数量。

例如：

```text
1112
```

表示：

```text
Type1 数量 = 1
Type2 数量 = 1
Type3 数量 = 1
Type4 数量 = 2
```

---

## 5. 单次运行：生成一个场景的详细结果

单次运行适合用于调试、展示图片和检查调度过程。

### 5.1 双臂基础方法

```bash
python parts/01_basic_two_arm/run.py --counts 1,1,1,1 --seed 0
```

输出位置：

```text
outputs/01_basic_two_arm/
```

### 5.2 三臂基础方法

```bash
python parts/02_basic_three_arm/run.py --counts 1,1,1,1 --seed 0
```

输出位置：

```text
outputs/02_basic_three_arm/
```

### 5.3 双臂自编优化方法

```bash
python parts/03_optimized_two_arm/run.py --counts 1,1,1,1 --seed 0
```

输出位置：

```text
outputs/03_optimized_two_arm/
```

### 5.4 三臂自编优化方法

```bash
python parts/04_optimized_three_arm/run.py --counts 1,1,1,1 --seed 0
```

输出位置：

```text
outputs/04_optimized_three_arm/
```

### 5.5 单次运行会生成什么文件

以 `outputs/01_basic_two_arm/` 为例，运行后通常会生成：

```text
instances/instance_*.json
results/result_*.json
results/schedule_*.csv
results/workspace_*.png
results/gantt_*.png
results/metrics_*.png
```

各文件含义如下：

| 文件 | 含义 |
|---|---|
| `instance_*.json` | 本次随机生成的任务实例，包括机械臂、物块位置、物块类型等 |
| `result_*.json` | 求解结果，包括目标值、任务开始时间、结束时间、机械臂选择等 |
| `schedule_*.csv` | 调度表，适合复制到 Excel 或报告中 |
| `workspace_*.png` | 工作空间图，显示机械臂、物块和收集盒位置 |
| `gantt_*.png` | 甘特图，显示各机械臂的任务执行时间 |
| `metrics_*.png` | 指标图，显示 Cmax、总能耗、负载等指标 |

---

## 6. 批量运行：生成二臂/三臂对比 CSV

批量运行适合用于论文或报告中的数据分析。

### 6.1 生成基础方法数据

运行：

```bash
python scripts/run_basic_case.py
```

默认会运行 16 种 `counts_code`：

```text
1111, 1112, 1121, 1211, 2111, 1122, 1212, 2112,
1221, 2121, 2211, 1222, 2122, 2212, 2221, 2222
```

每种 `counts_code` 默认运行 3 个随机种子：

```text
seed = 0, 1, 2
```

输出文件：

```text
outputs/four_case_framework/basic_2B3B_time_energy.csv
```

该 CSV 主要字段包括：

| 字段 | 含义 |
|---|---|
| `n1`、`n2`、`n3`、`n4` | 四类方块数量 |
| `total_tasks` | 总任务数 |
| `seed` | 随机种子 |
| `cmax_2B_basic` | 双臂基础方法最大完工时间 |
| `energy_2B_basic` | 双臂基础方法总能耗 |
| `cmax_3B_basic` | 三臂基础方法最大完工时间 |
| `energy_3B_basic` | 三臂基础方法总能耗 |
| `calc_time_basic_s` | 该行实验的计算耗时 |

### 6.2 生成自编优化方法数据

运行：

```bash
python scripts/run_optimized_case.py
```

输出文件：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

该 CSV 主要字段包括：

| 字段 | 含义 |
|---|---|
| `n1`、`n2`、`n3`、`n4` | 四类方块数量 |
| `total_tasks` | 总任务数 |
| `seed` | 随机种子 |
| `cmax_2B_optimized` | 双臂自编优化方法最大完工时间 |
| `energy_2B_optimized` | 双臂自编优化方法总能耗 |
| `cmax_3B_optimized` | 三臂自编优化方法最大完工时间 |
| `energy_3B_optimized` | 三臂自编优化方法总能耗 |
| `calc_time_optimized_s` | 该行实验的计算耗时 |

注意：自编优化方法是精确搜索方法，任务数较多时可能运行较慢。建议先用少量案例测试。

例如只运行一个案例：

```bash
python scripts/run_optimized_case.py --counts-list 1111 --seeds 0
```

例如运行两个案例、两个 seed：

```bash
python scripts/run_optimized_case.py --counts-list 1111/1112 --seeds 0,1
```

确认无误后，再运行完整默认实验。

### 6.3 批量脚本的参数说明

`run_basic_case.py` 和 `run_optimized_case.py` 都支持：

```bash
--counts-list
--seeds
```

#### `--counts-list`

可以写成：

```bash
--counts-list default
```

或：

```bash
--counts-list all
```

表示运行默认 16 组案例。

也可以只指定部分案例：

```bash
--counts-list 1111/1112/1222
```

或：

```bash
--counts-list 1111,1112,1222
```

如果只想运行一组具体数量，也可以写：

```bash
--counts-list 1,1,1,2
```

#### `--seeds`

例如：

```bash
--seeds 0,1,2
```

表示每种任务数量下分别使用 seed 0、1、2 生成随机位置。

---

## 7. 非线性规划扩展分析

非线性规划扩展依赖自编优化方法的批量结果。

因此应先运行：

```bash
python scripts/run_optimized_case.py
```

生成：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

然后再运行非线性规划扩展。

### 7.1 速度—能耗优化与 KKT 验证

运行：

```bash
python scripts/nonlinear_speed_energy_optimization.py
```

输出目录：

```text
outputs/nonlinear_programming/speed_energy_optimization/
```

主要输出：

```text
speed_energy_optimization.csv
kkt_verification.csv
```

其中：

| 文件 | 含义 |
|---|---|
| `speed_energy_optimization.csv` | 不同 deadline 下二臂/三臂速度倍率、能耗和推荐模式结果 |
| `kkt_verification.csv` | 对连续优化结果进行 KKT 条件验证 |

### 7.2 鲁棒速度—能耗分析

运行：

```bash
python scripts/nonlinear_robust_mode_selection.py
```

输出目录：

```text
outputs/nonlinear_programming/robust_mode_selection/
```

主要输出：

```text
robust_speed_mode_selection.csv
```

该文件用于分析 seed 扰动下，在同一 `counts_code` 中选择统一速度倍率时，二臂/三臂在最坏情况下的能耗和可行性。

---

## 8. 启发式算法说明

当前工程已经预留启发式算法入口：

```text
parts/05_optimized_heuristic_two_arm/
parts/06_optimized_heuristic_three_arm/
scripts/run_optimized_case_heuristic.py
common/solver_optimized_heuristic.py
```

但是当前版本中，启发式主算法尚未正式完成，因此不建议把启发式部分作为最终实验结果。

如果后续补充启发式算法，建议将其定位为：

```text
在精确隐枚举—分枝定界方法耗时较长时，用启发式构造较快得到可行解。
```

可选思路包括：

1. 最早完工时间优先；
2. 增量能耗最小优先；
3. 综合时间、能耗和负载均衡的贪心构造；
4. 在初始贪心解基础上进行任务交换或局部改进。

在启发式主代码完成前，报告主线建议只使用：

```text
基础方法 + 自编优化方法 + 非线性规划扩展
```

---

## 9. 推荐运行顺序

第一次打开工程时，建议按照下面顺序运行。

### 第一步：确认环境安装成功

```bash
pip install -r requirements.txt
```

### 第二步：先跑一个最小单次实验

```bash
python parts/01_basic_two_arm/run.py --counts 1,1,1,1 --seed 0
```

如果能在 `outputs/01_basic_two_arm/` 中看到 JSON、CSV 和 PNG，说明基础环境正常。

### 第三步：跑三臂基础方法对照

```bash
python parts/02_basic_three_arm/run.py --counts 1,1,1,1 --seed 0
```

### 第四步：跑自编优化方法小案例

```bash
python parts/03_optimized_two_arm/run.py --counts 1,1,1,1 --seed 0
python parts/04_optimized_three_arm/run.py --counts 1,1,1,1 --seed 0
```

### 第五步：批量生成基础方法 CSV

```bash
python scripts/run_basic_case.py
```

生成：

```text
outputs/four_case_framework/basic_2B3B_time_energy.csv
```

### 第六步：先小规模测试自编优化方法 CSV

```bash
python scripts/run_optimized_case.py --counts-list 1111 --seeds 0
```

确认没问题后，再根据时间选择是否运行完整默认实验：

```bash
python scripts/run_optimized_case.py
```

生成：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

### 第七步：运行非线性规划扩展

```bash
python scripts/nonlinear_speed_energy_optimization.py
python scripts/nonlinear_robust_mode_selection.py
```

生成：

```text
outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization.csv
outputs/nonlinear_programming/speed_energy_optimization/kkt_verification.csv
outputs/nonlinear_programming/robust_mode_selection/robust_speed_mode_selection.csv
```

---

## 10. 报告中建议使用哪些数据

建议报告主线使用以下数据：

| 数据 | 文件 |
|---|---|
| 基础方法二臂/三臂对比 | `outputs/four_case_framework/basic_2B3B_time_energy.csv` |
| 自编优化方法二臂/三臂对比 | `outputs/four_case_framework/optimized_2B3B_time_energy.csv` |
| 速度—能耗非线性规划结果 | `outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization.csv` |
| KKT 验证结果 | `outputs/nonlinear_programming/speed_energy_optimization/kkt_verification.csv` |
| 鲁棒速度—能耗结果 | `outputs/nonlinear_programming/robust_mode_selection/robust_speed_mode_selection.csv` |
| 单次调度过程图 | 各 `outputs/xx_xxx/results/` 中的 `workspace_*.png`、`gantt_*.png`、`metrics_*.png` |

不建议在启发式主算法完成前使用：

```text
outputs/four_case_framework/optimized_heuristic_2B3B_time_energy.csv
```

---

## 11. 常见问题

### 11.1 为什么基础方法和优化方法结果可能完全一样？

这是正常现象。

基础方法和自编优化方法使用的是同一个调度模型。区别在于：

- 基础方法将模型交给 CP-SAT 通用求解器；
- 自编优化方法使用隐枚举、分枝定界和支配剪枝自行求解。

因此在同一案例下，如果两者都求得最优解，那么 Cmax 和总能耗应该一致。

报告中不要写“自编优化方法比基础方法目标值更优”。

更合理的写法是：

```text
自编优化方法在不调用通用求解器的情况下，能够复现 CP-SAT 的最优调度结果，说明隐枚举—分枝定界算法实现具有正确性。
```

### 11.2 为什么自编优化方法运行时间比较长？

自编优化方法属于小规模精确搜索，会枚举任务顺序和执行模式，并通过下界和支配关系进行剪枝。

当任务数量增加时，搜索空间会快速增大，因此部分案例计算时间会明显变长。

这也正是后续引入启发式算法的原因。

### 11.3 为什么非线性规划要先运行 `run_optimized_case.py`？

因为非线性规划不是重新生成调度，而是在已有二臂/三臂调度结果上继续分析速度倍率和能耗。

它需要读取：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

如果这个文件不存在，非线性规划脚本就无法获得基础的 Cmax 和能耗数据。

### 11.4 如果只想快速检查代码，不想跑完整实验怎么办？

可以只运行一组任务和一个 seed：

```bash
python scripts/run_basic_case.py --counts-list 1111 --seeds 0
python scripts/run_optimized_case.py --counts-list 1111 --seeds 0
```

这样可以快速检查 CSV 是否能正常生成。

---

## 12. 提交前建议清理的文件

最终提交作业前，建议删除以下临时文件：

```text
.git/
__pycache__/
*.pyc
```

这些文件不是课程作业运行所必需的，删除后工程会更干净。

建议保留：

```text
common/
parts/
scripts/
outputs/
README.md
requirements.txt
```

其中 `outputs/` 可以根据老师要求选择是否提交。如果老师要求提交运行结果，则保留关键 CSV 和图片；如果只要求提交代码，则可以只保留空目录或删除旧输出后重新运行生成。
