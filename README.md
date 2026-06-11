# 多机械臂搬运调度与模式选择

本项目用于完成多机械臂搬运任务的建模、调度求解、机械臂数目对比、非线性速度-能耗优化和鲁棒分析。项目中四类物块分别记为 Type1、Type2、Type3、Type4，其中 Type1、Type2 主要对应单臂任务，Type3、Type4 主要对应双臂协同任务。一个任务结构用四位数字表示，例如 `1121` 表示四类物块数量分别为 1、1、2、1。需要特别说明的是，本项目关注的是多机械臂搬运任务的运筹学调度求解与二臂/三臂运行模式选择，不是机器人运动学、动力学建模或轨迹控制问题；机械臂在本文中主要作为可分配的作业资源参与调度优化。

## 1. 项目目录说明

```text
common/
```

存放公共代码，包括实例生成、模式构造、调度求解器、可视化函数等。

```text
parts/
```

存放 01-08 八个独立实验入口。每个入口运行一次具体案例，并输出实例文件、调度结果、甘特图、工作空间图和指标图。

```text
scripts/
```

存放批量实验、汇总分析、非线性规划和鲁棒分析脚本。

```text
outputs/
```

存放所有运行结果，包括 01-08 的单案例图像和 CSV，以及批量结果表、模式选择分析表、非线性规划结果表等。

## 2. 01-08 单案例运行与图像输出

`parts/` 下的 01-08 文件夹是八个独立案例入口。它们主要用于生成单个案例的可视化结果，包括：

```text
workspace_*.png   任务空间/机械臂/物块位置图
gantt_*.png       调度甘特图
metrics_*.png     时间、能耗等指标图
schedule_*.csv    任务调度明细表
result_*.json     该次调度的完整结果
instance_*.json   该次随机生成的任务实例
```

当前项目中，01-08 分别已经用下面 8 组任务结构测试过：

| 部分 | 方法 | 机械臂数量 | 测试 counts |
|---|---|---:|---|
| 01_basic_two_arm | 基础方法 | 2 | 1111 |
| 02_basic_three_arm | 基础方法 | 3 | 2111 |
| 03_optimized_two_arm | 运筹学优化方法 | 2 | 1121 |
| 04_optimized_three_arm | 运筹学优化方法 | 3 | 2211 |
| 05_optimized_heuristic_two_arm | 启发式算法 | 2 | 1221 |
| 06_optimized_heuristic_three_arm | 启发式算法 | 3 | 1122 |
| 07_optimized_warmstart_two_arm | 运筹学方法 + 启发式初始解 | 2 | 2221 |
| 08_optimized_warmstart_three_arm | 运筹学方法 + 启发式初始解 | 3 | 1222 |

重新运行某一个部分时，使用：

```bash
python parts/01_basic_two_arm/run.py --counts 1,1,1,1
python parts/02_basic_three_arm/run.py --counts 2,1,1,1
python parts/03_optimized_two_arm/run.py --counts 1,1,2,1
python parts/04_optimized_three_arm/run.py --counts 2,2,1,1
python parts/05_optimized_heuristic_two_arm/run.py --counts 1,2,2,1
python parts/06_optimized_heuristic_three_arm/run.py --counts 1,1,2,2
python parts/07_optimized_warmstart_two_arm/run.py --counts 2,2,2,1
python parts/08_optimized_warmstart_three_arm/run.py --counts 1,2,2,2
```

01-08 的输出位置分别是：

```text
outputs/01_basic_two_arm/
outputs/02_basic_three_arm/
outputs/03_optimized_two_arm/
outputs/04_optimized_three_arm/
outputs/05_optimized_heuristic_two_arm/
outputs/06_optimized_heuristic_three_arm/
outputs/07_optimized_warmstart_two_arm/
outputs/08_optimized_warmstart_three_arm/
```

每个文件夹下通常包含：

```text
instances/   保存输入实例 JSON
results/     保存甘特图、工作空间图、指标图、调度 CSV 和结果 JSON
```

## 4. 批量生成二臂与三臂模式下时间和能耗 CSV

批量脚本不会生成甘特图，主要用于生成后续分析所需的表格数据。默认会运行 16 种任务结构，每种结构运行 seed=0,1,2 三次，并分别计算二臂和三臂结果。

### 4.1 基础方法

```bash
python scripts/run_basic_case.py
```

输出：

```text
outputs/four_case_framework/basic_2B3B_time_energy.csv
```

### 4.2 运筹学优化方法

```bash
python scripts/run_optimized_case.py
```

输出：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

这个 CSV 是后续 `analyse_case.py`、确定性非线性规划和鲁棒非线性规划默认读取的数据源。

### 4.3 启发式算法

```bash
python scripts/run_optimized_case_heuristic.py
```

输出：

```text
outputs/four_case_framework/optimized_heuristic_2B3B_time_energy.csv
```

除了启发式结果外，还会记录与最优结果之间的差距，例如：

```text
cmax_gap_2B_pct
energy_gap_2B_pct
is_optimal_2B
cmax_gap_3B_pct
energy_gap_3B_pct
is_optimal_3B
```

### 4.4 运筹学方法 + 启发式初始解

```bash
python scripts/run_optimized_warmstart_case.py
```

输出：

```text
outputs/four_case_framework/optimized_warmstart_2B3B_time_energy.csv
```

该方法先用启发式算法得到初始解，再用优化搜索改进，并记录搜索节点信息，例如：

```text
nodes_visited_2B
nodes_pruned_2B
nodes_visited_3B
nodes_pruned_3B
```

## 5. 二臂/三臂模式选择分析

运行：

```bash
python scripts/analyse_case.py
```

默认输入：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

输出：

```text
outputs/mode_decision/mode_decision_summary_optimized.csv
```

这个文件按 `counts_code` 汇总 seed=0,1,2 的平均结果，比较二臂和三臂在时间和能耗上的差异。核心指标为：

```text
eta_T = (Cmax_2B - Cmax_3B) / Cmax_2B * 100
```

表示三臂相对二臂节省了多少时间，越大越支持三臂。

```text
eta_E = (Energy_3B - Energy_2B) / Energy_2B * 100
```

表示三臂相对二臂增加了多少能耗，越大越不支持三臂。

综合评分为：

```text
S_lambda = eta_T - lambda * eta_E
```

## 6. 确定性非线性速度-能耗优化

运行：

```bash
python scripts/nonlinear_speed_energy_optimization.py
```

默认输入：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

输出：

```text
outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization.csv
outputs/nonlinear_programming/speed_energy_optimization/kkt_verification.csv
```

该部分先对同一 `counts_code` 下的 seed 结果取平均，然后在平均时间和平均能耗基础上进行速度层非线性优化。决策变量为：

```text
s_single   单臂任务速度倍率
s_dual     双臂协同任务速度倍率
```

能耗函数采用 U 形经济速度模型：

```text
E_k(s)=E_k0*(lambda_k/s+(1-lambda_k)*((1-rho_k)+rho_k*s^2))
```

含义是：速度过慢会带来等待、保持、夹持能耗，速度过快会带来动态损耗、冲击和协同控制代价。

该文件采用的课内方法是：

```text
内点制约函数法 + 最速下降方向 + 0.618 黄金分割一维搜索 + KKT 条件验证
```

其中：

```text
speed_energy_optimization.csv
```

保存不同任务结构、不同截止时间比例、不同参数组下的 2B/3B 速度优化和推荐结果；

```text
kkt_verification.csv
```

单独保存 KKT 验证结果，包括原始可行性、互补松弛误差、驻点残差和 `kkt_pass`。

## 7. 鲁棒非线性速度-能耗优化

运行：

```bash
python scripts/nonlinear_robust_mode_selection.py
```

默认输入：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

输出：

```text
outputs/nonlinear_programming/robust_mode_selection/robust_speed_mode_selection.csv
outputs/nonlinear_programming/robust_mode_selection/robust_kkt_verification.csv
```

这一部分不对 seed 取平均，而是保留 seed=0,1,2 的随机扰动。对同一个 `counts_code`，要求所有 seed 使用同一组速度：

```text
s_single
s_dual
```

并引入最坏能耗上界：

```text
z
```

鲁棒模型可以理解为：

```text
min z
s.t. 每个 seed 的能耗 E_j(s_single, s_dual) <= z
     每个 seed 的完工时间 C_j(s_single, s_dual) <= D
     s_min <= s_single, s_dual <= s_max
```

它解决的问题是：在随机 seed 扰动下，怎样选择一组统一速度，使所有 seed 都按时完成，并且最坏情况下的能耗最低。

该文件采用的课内方法是：

```text
SLP 逐次线性规划法 + 小型线性规划子问题求解 + KKT 条件验证
```

其中：

```text
robust_speed_mode_selection.csv
```

保存鲁棒优化后的 2B/3B 最坏能耗、平均能耗、最紧时间 seed、最坏能耗 seed 和最终推荐模式；

```text
robust_kkt_verification.csv
```

保存鲁棒模型的 KKT 验证结果，包括 active time seed、active energy seed、KKT 乘子、驻点残差和 `kkt_pass`。

## 8. 推荐运行顺序

如果从零开始完整运行，建议顺序如下：

```bash
python scripts/run_optimized_case.py
python scripts/analyse_case.py
python scripts/nonlinear_speed_energy_optimization.py --no-sensitivity
python scripts/nonlinear_robust_mode_selection.py
```

如果需要四种批量方法都生成，可以运行：

```bash
python scripts/run_basic_case.py
python scripts/run_optimized_case.py
python scripts/run_optimized_case_heuristic.py
python scripts/run_optimized_warmstart_case.py
```

如果只想生成报告展示用图，可以运行 01-08：

```bash
python parts/01_basic_two_arm/run.py --counts 1,1,1,1
python parts/02_basic_three_arm/run.py --counts 2,1,1,1
python parts/03_optimized_two_arm/run.py --counts 1,1,2,1
python parts/04_optimized_three_arm/run.py --counts 2,2,1,1
python parts/05_optimized_heuristic_two_arm/run.py --counts 1,2,2,1
python parts/06_optimized_heuristic_three_arm/run.py --counts 1,1,2,2
python parts/07_optimized_warmstart_two_arm/run.py --counts 2,2,2,1
python parts/08_optimized_warmstart_three_arm/run.py --counts 1,2,2,2
```

## 9. 当前 outputs 中主要文件的来源

```text
outputs/01_basic_two_arm/ 到 outputs/08_optimized_warmstart_three_arm/
```

由 `parts/01` 到 `parts/08` 的单案例入口生成，主要用于展示实例、甘特图、调度明细和指标图。

```text
outputs/four_case_framework/basic_2B3B_time_energy.csv
```

由 `scripts/run_basic_case.py` 生成，记录基础方法下 2B/3B 的 Cmax、能耗和计算耗时。

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

由 `scripts/run_optimized_case.py` 生成，记录运筹学优化方法下 2B/3B 的 Cmax、能耗和计算耗时。

```text
outputs/four_case_framework/optimized_heuristic_2B3B_time_energy.csv
```

由 `scripts/run_optimized_case_heuristic.py` 生成，记录启发式算法结果及其与最优结果的差距。

```text
outputs/four_case_framework/optimized_warmstart_2B3B_time_energy.csv
```

由 `scripts/run_optimized_warmstart_case.py` 生成，记录运筹学方法结合启发式初始解后的结果和搜索节点数量。

```text
outputs/mode_decision/mode_decision_summary_optimized.csv
```

由 `scripts/analyse_case.py` 生成，用于汇总 seed 平均后的二臂/三臂时间-能耗偏好选择。

```text
outputs/nonlinear_programming/speed_energy_optimization/speed_energy_optimization.csv
```

由 `scripts/nonlinear_speed_energy_optimization.py` 生成，是确定性非线性速度-能耗优化主结果表。

```text
outputs/nonlinear_programming/speed_energy_optimization/kkt_verification.csv
```

由 `scripts/nonlinear_speed_energy_optimization.py` 生成，是确定性非线性优化的 KKT 验证表。

```text
outputs/nonlinear_programming/robust_mode_selection/robust_speed_mode_selection.csv
```

由 `scripts/nonlinear_robust_mode_selection.py` 生成，是随机 seed 扰动下鲁棒速度-能耗优化主结果表。

```text
outputs/nonlinear_programming/robust_mode_selection/robust_kkt_verification.csv
```

由 `scripts/nonlinear_robust_mode_selection.py` 生成，是鲁棒非线性规划的 KKT 验证表。


