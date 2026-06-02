# 多机械臂分类搬运调度优化工程 README

## 1. 项目简介

本工程面向多机械臂分类搬运任务，研究在不同机械臂数量和不同求解方法下，如何完成物块分配、任务排序和运行效果对比。

工程主体采用：

```text
整数规划 + 目标规划
```

其中：

- **整数规划**用于描述任务分配、执行模式选择、机械臂资源占用、任务先后顺序等离散决策；
- **目标规划**用于描述多个优化目标之间的优先级关系；
- **非线性规划**作为额外扩展内容，用于在固定调度序列后进一步研究连续运动速度、时间和能耗优化。

当前目标优先级为：

```text
第一优先级：最小化 Cmax，即最大完工时间最短；
第二优先级：在 Cmax 最优的前提下，最小化总能耗；
第三优先级：在前两个目标尽量满足的前提下，使机械臂负载更加均衡。
```

二臂/三臂对比不是直接比较谁一定更好，而是在二臂系统和三臂系统分别得到自身最优调度结果后，再对时间、能耗等指标进行对比。

---

## 2. 当前工程能实现什么

当前工程已经具备以下能力：

1. 生成四类物块的随机任务实例；
2. 支持双臂和三臂两种机械臂布局；
3. 双臂布局保持左右对称；
4. 三臂布局保持 120° 均匀分布；
5. 二臂和三臂共用同一套收集盒位置，保证对比公平；
6. 基础方法可以调用求解器完成整数规划 + 目标规划求解；
7. 优化方法已经预留统一接口，但真实优化算法主体尚未完成；
8. 可以输出 JSON、CSV、PNG 图像；
9. 可以打开 MuJoCo 静态窗口展示机械臂和物块布局；
10. 非线性规划部分已经有占位接口，后续可以继续扩展。

当前工程还不能完成以下内容：

1. 不能真实控制机械臂运动；
2. 不能生成真实动态抓取过程；
3. 优化方法当前仍是占位，不能作为最终实验结论；
4. 非线性规划当前仅有接口说明，还没有实际求解；
5. 二臂/三臂对比分析部分仍需要继续补充。

---

## 3. 环境安装

建议使用 Python 3.10 及以上版本。

进入工程根目录，也就是能看到以下文件夹的位置：

```text
common/
parts/
scripts/
extensions/
outputs/
requirements.txt
```

安装依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 中主要包括：

```text
ortools
matplotlib
mujoco
```

如果只运行调度求解和生成图像，重点依赖是 `ortools` 和 `matplotlib`。

如果要打开静态仿真窗口，还需要 `mujoco`。

---

## 4. 工程目录说明

```text
multi_arm_course_project/
│
├─ common/
├─ parts/
├─ scripts/
├─ extensions/
├─ outputs/
├─ README.md
└─ requirements.txt
```

### 4.1 `common/`

`common/` 存放建模和通用内容，是整个工程的核心公共模块。

主要文件包括：

| 文件 | 作用 |
|---|---|
| `config.py` | 公共参数配置，包括机械臂位置、收集盒位置、速度、能耗、求解器参数等 |
| `geometry.py` | 机械臂、任务、模式、实例等数据结构，以及几何计算函数 |
| `instance_generator.py` | 根据 seed 和物块类型生成随机任务实例 |
| `mode_builder.py` | 为每个任务生成可行执行模式，如单臂模式、双臂协同模式 |
| `solver_basic.py` | 基础方法求解器，主体为整数规划 + 序贯目标规划 |
| `solver_optimized.py` | 优化方法占位接口，后续应在这里实现 2B/3B 优化算法主体 |
| `experiment_runner.py` | 四个独立场景共用的运行器 |
| `visualization.py` | 生成工作空间图、甘特图、指标图 |
| `mujoco_static.py` | MuJoCo 静态仿真展示 |
| `utils.py` | JSON、CSV 保存、调度表转换、打印结果等工具函数 |

### 4.2 `parts/`

`parts/` 存放四种情形的独立运行入口。

```text
parts/
├─ 01_basic_two_arm/
├─ 02_basic_three_arm/
├─ 03_optimized_two_arm/
└─ 04_optimized_three_arm/
```

四种情形分别是：

| 文件夹 | 含义 | 当前状态 |
|---|---|---|
| `01_basic_two_arm/` | 双臂基础方法 | 可运行，调用基础求解器 |
| `02_basic_three_arm/` | 三臂基础方法 | 可运行，调用基础求解器 |
| `03_optimized_two_arm/` | 双臂优化方法 | 可运行占位接口，但优化算法未实现 |
| `04_optimized_three_arm/` | 三臂优化方法 | 可运行占位接口，但优化算法未实现 |

每个文件夹中通常有：

```text
run.py
view_workspace.py
```

其中：

- `run.py` 用于运行该情形并输出结果；
- `view_workspace.py` 用于读取最近一次结果并打开 MuJoCo 静态展示窗口。

### 4.3 `scripts/`

`scripts/` 存放批量实验、对比实验或 CSV 生成脚本。

当前主要文件为：

```text
scripts/run_four_case_framework.py
```

注意：当前这个文件已经被改成“批量生成优化方法下二臂/三臂时间和能量参数 CSV”的脚本。它目前不是完整的“四种情形详细输出脚本”。

运行后会生成：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

这个 CSV 只记录优化方法下二臂和三臂的 `Cmax` 与 `total_energy`，不做任何分析，也不下结论。

### 4.4 `extensions/`

`extensions/` 存放扩展内容。

当前主要扩展是：

```text
extensions/nonlinear_programming/continuous_motion_nlp_placeholder.py
```

它用于预留非线性规划扩展接口。目前只保留理论接口，不进行实际求解。

### 4.5 `outputs/`

`outputs/` 存放所有程序运行后生成的结果文件。

主要输出目录包括：

```text
outputs/01_basic_two_arm/
outputs/02_basic_three_arm/
outputs/03_optimized_two_arm/
outputs/04_optimized_three_arm/
outputs/four_case_framework/
outputs/nonlinear_programming/
```

其中：

- `01_basic_two_arm/` 到 `04_optimized_three_arm/` 用于四种情形单独运行的输出；
- `four_case_framework/` 用于批量 CSV 或四情形框架输出；
- `nonlinear_programming/` 用于后续非线性规划扩展输出，目前只有占位。

---

## 5. 使用方法

### 5.1 四个情形独立调试

这部分用于单独检查每一个场景是否能够运行。

#### 5.1.1 双臂基础方法

```bash
python parts/01_basic_two_arm/run.py --counts 1,1,1,0 --seed 1
```

输出位置：

```text
outputs/01_basic_two_arm/
```

主要输出包括：

```text
instances/instance_*.json
results/result_*.json
results/schedule_*.csv
results/workspace_*.png
results/gantt_*.png
results/metrics_*.png
```

#### 5.1.2 三臂基础方法

```bash
python parts/02_basic_three_arm/run.py --counts 1,1,1,0 --seed 1
```

输出位置：

```text
outputs/02_basic_three_arm/
```

#### 5.1.3 双臂优化方法

```bash
python parts/03_optimized_two_arm/run.py --counts 1,1,1,0 --seed 1
```

输出位置：

```text
outputs/03_optimized_two_arm/
```

注意：当前 `common/solver_optimized.py` 仍是占位接口，所以结果状态通常为：

```text
PLACEHOLDER_NOT_SOLVED
```

#### 5.1.4 三臂优化方法

```bash
python parts/04_optimized_three_arm/run.py --counts 1,1,1,0 --seed 1
```

输出位置：

```text
outputs/04_optimized_three_arm/
```

注意：当前仍是优化方法占位接口。

---

### 5.2 四个情形一起生成

从工程设计上，应该有一个脚本用于一次性运行四种情形：

```text
双臂基础方法
三臂基础方法
双臂优化方法
三臂优化方法
```

理想输出位置为：

```text
outputs/four_case_framework/
```

但是当前 `scripts/run_four_case_framework.py` 已被改成“优化方法二臂/三臂时间能量 CSV 生成脚本”。

因此，如果后续需要真正实现“四个情形一起生成详细结果”，建议新增或恢复一个脚本，例如：

```text
scripts/run_all_four_cases.py
```

具体怎么补充见《行动指南》。

---

### 5.3 生成二臂/三臂优化方法时间能量 CSV

当前可用脚本为：

```bash
python scripts/run_four_case_framework.py
```

默认会运行 16 种 Type 输入：

```text
1111/1112/1121/1211/2111/1122/1212/2112/
1221/2121/2211/1222/2122/2212/2221/2222
```

每种输入默认运行 3 个随机种子：

```text
seed = 0, 1, 2
```

输出文件：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

CSV 字段包括：

```text
counts_code
n1
n2
n3
n4
total_tasks
seed
status_2B_optimized
cmax_2B_optimized
energy_2B_optimized
status_3B_optimized
cmax_3B_optimized
energy_3B_optimized
```

这个 CSV 是后续对比函数的输入，只反映时间和能量参数，不负责分析结论。

如果只想运行部分输入，可以使用：

```bash
python scripts/run_four_case_framework.py --counts-list 1111/1112/1121
```

如果想改随机种子，可以使用：

```bash
python scripts/run_four_case_framework.py --seeds 0,1,2,3,4
```

---

### 5.4 进行对比分析

当前工程已经能生成对比所需的输入 CSV，但完整的对比分析脚本尚未完成。

建议后续新增：

```text
scripts/analyze_2B3B_comparison.py
```

该脚本读取：

```text
outputs/four_case_framework/optimized_2B3B_time_energy.csv
```

然后再生成分析用 CSV 或图表，例如：

```text
outputs/four_case_framework/optimized_2B3B_comparison_summary.csv
```

对比分析阶段可以计算：

```text
二臂平均 Cmax
三臂平均 Cmax
二臂平均能耗
三臂平均能耗
不同 Type 输入下二臂/三臂指标变化
```

注意：输入 CSV 不应该直接写结论，结论应放在后续分析脚本或报告中完成。

---

### 5.5 非线性规划扩展独立运行

当前非线性规划文件为：

```text
extensions/nonlinear_programming/continuous_motion_nlp_placeholder.py
```

可以直接运行查看占位接口：

```bash
python extensions/nonlinear_programming/continuous_motion_nlp_placeholder.py
```

当前输出为占位结果，表示尚未实现求解。

后续建议新增一个独立运行脚本，例如：

```text
scripts/run_nonlinear_extension.py
```

用于读取某次调度结果，从中提取固定任务序列，再进行连续运动速度或能耗优化。

推荐输出位置：

```text
outputs/nonlinear_programming/
```

---

### 5.6 MuJoCo 静态仿真展示

每个情形都有对应的 `view_workspace.py`。

例如先运行双臂基础方法：

```bash
python parts/01_basic_two_arm/run.py --counts 1,1,1,0 --seed 1
```

再打开静态展示：

```bash
python parts/01_basic_two_arm/view_workspace.py
```

三臂基础方法：

```bash
python parts/02_basic_three_arm/run.py --counts 1,1,1,0 --seed 1
python parts/02_basic_three_arm/view_workspace.py
```

静态展示会读取对应 `outputs/.../results/` 中最新的 `result_*.json`。

---

## 6. 参数说明

### 6.1 counts

`--counts` 表示四类物块数量，例如：

```bash
--counts 1,1,1,0
```

含义为：

```text
Type1 数量 = 1
Type2 数量 = 1
Type3 数量 = 1
Type4 数量 = 0
```

其中：

- Type1、Type2 为单臂任务；
- Type3、Type4 为双臂协同任务。

### 6.2 counts_code

批量 CSV 中使用 `counts_code` 表示四类物块数量，例如：

```text
1112
```

表示：

```text
Type1 = 1
Type2 = 1
Type3 = 1
Type4 = 2
```

### 6.3 seed

`seed` 是随机种子，用于控制物块随机位置。

同一个 seed 会生成同一批物块位置；不同 seed 会生成不同物块位置。

为了使对比更稳定，当前批量脚本默认每种 Type 输入运行 3 个 seed：

```text
0, 1, 2
```

---

## 7. 当前未完成内容

当前工程最主要还缺少三部分：

1. **2B/3B 优化算法主体**  
   文件位置：`common/solver_optimized.py`。  
   当前只是占位，后续需要实现真正的优化方法。

2. **对比分析部分**  
   输入 CSV 已经可以由 `scripts/run_four_case_framework.py` 生成。  
   后续需要新增分析脚本，读取 CSV 后生成分析结果。

3. **非线性规划扩展**  
   文件位置：`extensions/nonlinear_programming/continuous_motion_nlp_placeholder.py`。  
   当前只有主题接口，没有实际求解。

---

## 8. 建议使用顺序

建议按照下面顺序使用和开发：

```text
第一步：运行 01_basic_two_arm，确认双臂基础方法正常；
第二步：运行 02_basic_three_arm，确认三臂基础方法正常；
第三步：运行 03/04，确认优化方法占位接口输出格式正常；
第四步：运行 scripts/run_four_case_framework.py，生成优化方法二臂/三臂时间能量 CSV；
第五步：补充对比分析脚本，读取 CSV 进行统计分析；
第六步：补充 common/solver_optimized.py，实现真正优化算法；
第七步：补充 extensions/nonlinear_programming，实现非线性规划扩展；
第八步：根据最终结果整理报告、图片和表格。
```

---

## 9. 注意事项

1. 修改 `common/config.py` 后，需要重新运行脚本，旧的输出图和 CSV 不会自动更新；
2. `outputs/` 中的结果文件可以删除，重新运行脚本会重新生成；
3. 不要把当前优化方法占位结果当成最终实验结论；
4. 当前 `scripts/run_four_case_framework.py` 输出的是优化方法二臂/三臂时间能量 CSV，不是完整四情形详细输出；
5. 如果修改文件名，需要同步修改 README 和脚本引用；
6. 提交前建议删除 `__pycache__/` 和无用旧输出。
