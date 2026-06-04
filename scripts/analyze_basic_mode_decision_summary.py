# -*- coding: utf-8 -*-
"""二臂/三臂运行模式选择分析：按 counts_code 汇总 seed 平均，并进行多组时间-能耗偏好对比。

上游输入：
    outputs/four_case_framework/basic_2B3B_time_energy.csv

该文件通常由：
    python scripts/run_basic_case.py
生成，包含同一任务输入下二臂和三臂的 Cmax 与总能耗。

本脚本做的事情：
1. 不重新求解调度问题；
2. 读取 basic_2B3B_time_energy.csv；
3. 对每一种 counts_code，只在该 counts_code 内部对 seed=0,1,2 取平均；
4. 输出一个 16 行左右的汇总 CSV；
5. 设置 10 组时间-能耗偏好参数 lambda，比较不同偏好下推荐二臂还是三臂；
6. 在每一行最后增加 4 个敏感性分析条件字段：
   - decision_lambda_star
   - lambda_condition_recommend_3arm
   - lambda_condition_recommend_2arm
   - lambda_condition_similar

核心指标：
    eta_T = (Cmax_2B - Cmax_3B) / Cmax_2B * 100
        表示三臂相对二臂的时间节省率，越大越支持三臂。

    eta_E = (Energy_3B - Energy_2B) / Energy_2B * 100
        表示三臂相对二臂的能耗增加率，越大越不支持三臂。

    S_lambda = eta_T - lambda * eta_E
        若 S_lambda > 0，说明在该偏好下三臂的时间收益足以抵消能耗代价，推荐三臂；
        若 S_lambda < 0，说明能耗代价更重要，推荐二臂。

lambda 的含义：
    lambda 越小，越重视时间；
    lambda 越大，越重视能耗。
    例如 lambda=5 表示 1% 的能耗增加需要至少 5% 的时间节省来抵消。

敏感性分析含义：
    decision_lambda_star 表示二臂/三臂推荐结果发生切换的临界 lambda。
    当 lambda 小于或大于该临界值时，推荐结果会保持不变。

输出：
    outputs/mode_decision/mode_decision_summary_basic.csv

运行方式：
    python scripts/analyze_basic_mode_decision_summary.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_DIR / "outputs" / "four_case_framework" / "basic_2B3B_time_energy.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "mode_decision" / "mode_decision_summary_basic.csv"


# 不再使用严格的 VALID_STATUS。
# 原因：不同求解器/不同代码版本可能输出 optimal、success、OK、done 等不同状态。
# 如果只允许 {"OPTIMAL", "FEASIBLE", "SUCCESS", "success"}，很容易把所有行都跳过。
# 这里改成：只排除明确失败的状态。
INVALID_STATUS = {
    "INFEASIBLE",
    "NO_SOLUTION",
    "FAILED",
    "FAIL",
    "ERROR",
    "TIMEOUT",
}


# 10 组时间-能耗偏好参数。
# lambda 越小越偏时间，lambda 越大越偏能耗。
# 输出顺序为：只考虑时间 → 多组 lambda 过渡 → 只考虑能耗。
LAMBDA_VALUES = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]

# 判断浮点数正负时使用的小阈值，防止因为极小误差导致分类错误。
EPS = 1e-9


def lambda_label(lam: float) -> str:
    """把 lambda 转成适合 CSV 字段名的短标签。"""
    return str(lam).replace(".", "_")


def normalize_status(value: Any) -> str:
    """把状态字段统一转成大写字符串，便于判断。"""
    if value is None:
        return ""
    return str(value).strip().upper()


def safe_float(value: Any) -> float:
    """安全转换为浮点数。"""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    """安全转换为整数。"""
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def round2(value: Any) -> Any:
    """保留两位小数；无穷大写为 inf。"""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return round(float(value), 2)


def mean(values: list[float]) -> float:
    """计算平均值。"""
    return sum(values) / len(values) if values else 0.0


def read_csv(path: Path) -> list[dict]:
    """读取 CSV 文件。"""
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    """保存 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # 如果确实没有可用行，写入空文件。
        # 这种情况通常说明输入 CSV 本身为空，或者关键数值列缺失/为0。
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_scenario(n1: int, n2: int, n3: int, n4: int) -> str:
    """根据任务结构给出简单标签，方便报告解释。"""
    single_arm_tasks = n1 + n2
    dual_arm_tasks = n3 + n4

    if n1 == n2 == n3 == n4:
        return "balanced"

    if n4 > max(n1, n2, n3):
        return "type4_dominant"

    if single_arm_tasks > dual_arm_tasks:
        return "single_arm_dominant"

    if dual_arm_tasks > single_arm_tasks:
        return "dual_arm_dominant"

    return "mixed_balanced"


def calc_one_seed_metrics(row: dict) -> dict | None:
    """计算单个 seed 下的二/三臂对比指标。

    注意：
    这里不再用固定 VALID_STATUS 强行筛选。
    只要状态不是明确失败，并且 cmax / energy 有有效数值，就参与汇总。
    """

    status_2b_raw = row.get("status_2B_basic", "")
    status_3b_raw = row.get("status_3B_basic", "")

    status_2b = normalize_status(status_2b_raw)
    status_3b = normalize_status(status_3b_raw)

    # 只有明确失败的状态才跳过，避免因为状态名字不一致导致全表被过滤。
    if status_2b in INVALID_STATUS or status_3b in INVALID_STATUS:
        return None

    cmax_2 = safe_float(row.get("cmax_2B_basic"))
    cmax_3 = safe_float(row.get("cmax_3B_basic"))
    energy_2 = safe_float(row.get("energy_2B_basic"))
    energy_3 = safe_float(row.get("energy_3B_basic"))

    # 如果关键数值不存在或为 0，才跳过。
    if cmax_2 <= 0 or cmax_3 <= 0 or energy_2 <= 0 or energy_3 <= 0:
        return None

    eta_t = (cmax_2 - cmax_3) / cmax_2 * 100
    eta_e = (energy_3 - energy_2) / energy_2 * 100

    # 单个 seed 下的临界 lambda。
    # 这里保留原有逻辑，不改变原输出列 mean_lambda_star 的来源。
    if eta_e > 0:
        lambda_star = eta_t / eta_e
    elif eta_t > 0 and eta_e <= 0:
        lambda_star = math.inf
    else:
        lambda_star = 0.0

    return {
        "counts_code": row.get("counts_code", ""),
        "seed": safe_int(row.get("seed")),
        "n1": safe_int(row.get("n1")),
        "n2": safe_int(row.get("n2")),
        "n3": safe_int(row.get("n3")),
        "n4": safe_int(row.get("n4")),
        "total_tasks": safe_int(row.get("total_tasks")),
        "status_2B_basic": status_2b_raw,
        "status_3B_basic": status_3b_raw,
        "cmax_2B": cmax_2,
        "cmax_3B": cmax_3,
        "energy_2B": energy_2,
        "energy_3B": energy_3,
        "eta_T": eta_t,
        "eta_E": eta_e,
        "lambda_star": lambda_star,
    }


def recommend_time_only(mean_cmax_2b: float, mean_cmax_3b: float) -> str:
    """只考虑时间：谁 Cmax 更小选谁。"""
    if mean_cmax_3b < mean_cmax_2b:
        return "recommend_3arm"
    if mean_cmax_3b > mean_cmax_2b:
        return "recommend_2arm"
    return "similar_prefer_2arm"


def recommend_energy_only(mean_energy_2b: float, mean_energy_3b: float) -> str:
    """只考虑能耗：谁能耗更低选谁。"""
    if mean_energy_3b < mean_energy_2b:
        return "recommend_3arm"
    if mean_energy_3b > mean_energy_2b:
        return "recommend_2arm"
    return "similar_prefer_2arm"


def recommend_by_score(score: float) -> str:
    """根据综合收益判断推荐结果。"""
    if score > 0:
        return "recommend_3arm"
    if score < 0:
        return "recommend_2arm"
    return "similar_prefer_2arm"


def format_lambda_boundary(value: float) -> str:
    """把 lambda 临界值格式化成适合 CSV 阅读的字符串。"""
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def make_lambda_condition(mean_eta_t: float, mean_eta_e: float) -> dict:
    """生成二臂/三臂适用条件。

    这里使用同一个 counts_code 汇总后的 mean_eta_T 和 mean_eta_E 计算临界 lambda。

    记：
        S_lambda = mean_eta_T - lambda * mean_eta_E

    若 S_lambda > 0，推荐三臂；
    若 S_lambda < 0，推荐二臂；
    若 S_lambda = 0，二臂和三臂近似无差别。

    这部分对应敏感性分析中的：
        参数 lambda 变化时，推荐方案保持不变的范围。
    """

    # 情况1：三臂更快，而且能耗不增加。
    # 三臂在时间和能耗上都不吃亏，所以所有 lambda 下都推荐三臂。
    if mean_eta_t > EPS and mean_eta_e <= EPS:
        return {
            "decision_lambda_star": "inf",
            "lambda_condition_recommend_3arm": "all lambda >= 0",
            "lambda_condition_recommend_2arm": "none",
            "lambda_condition_similar": "none",
        }

    # 情况2：三臂不更快，而且能耗不降低。
    # 二臂在时间和能耗上都不吃亏，所以所有 lambda 下都推荐二臂。
    if mean_eta_t <= EPS and mean_eta_e >= -EPS:
        return {
            "decision_lambda_star": "0.00",
            "lambda_condition_recommend_3arm": "none",
            "lambda_condition_recommend_2arm": "all lambda >= 0",
            "lambda_condition_similar": "lambda = 0 only if eta_T = eta_E = 0",
        }

    # 情况3：三臂更快，但能耗更高。
    # 这是最典型的时间-能耗权衡：
    # lambda 小，重视时间，推荐三臂；
    # lambda 大，重视能耗，推荐二臂。
    if mean_eta_t > EPS and mean_eta_e > EPS:
        boundary = mean_eta_t / mean_eta_e
        boundary_text = format_lambda_boundary(boundary)

        return {
            "decision_lambda_star": boundary_text,
            "lambda_condition_recommend_3arm": f"0 <= lambda < {boundary_text}",
            "lambda_condition_recommend_2arm": f"lambda > {boundary_text}",
            "lambda_condition_similar": f"lambda = {boundary_text}",
        }

    # 情况4：三臂更慢，但能耗更低。
    # lambda 小，重视时间，推荐二臂；
    # lambda 大，重视能耗，推荐三臂。
    if mean_eta_t < -EPS and mean_eta_e < -EPS:
        boundary = mean_eta_t / mean_eta_e
        boundary_text = format_lambda_boundary(boundary)

        return {
            "decision_lambda_star": boundary_text,
            "lambda_condition_recommend_3arm": f"lambda > {boundary_text}",
            "lambda_condition_recommend_2arm": f"0 <= lambda < {boundary_text}",
            "lambda_condition_similar": f"lambda = {boundary_text}",
        }

    # 情况5：三臂时间基本相同，但能耗更低。
    # 当 lambda = 0 时只看时间，所以二者近似无差别；
    # 当 lambda > 0 时考虑能耗，推荐三臂。
    if abs(mean_eta_t) <= EPS and mean_eta_e < -EPS:
        return {
            "decision_lambda_star": "0.00",
            "lambda_condition_recommend_3arm": "lambda > 0",
            "lambda_condition_recommend_2arm": "none",
            "lambda_condition_similar": "lambda = 0",
        }

    # 理论兜底，正常情况下很少进入这里。
    return {
        "decision_lambda_star": "",
        "lambda_condition_recommend_3arm": "",
        "lambda_condition_recommend_2arm": "",
        "lambda_condition_similar": "",
    }


def make_summary_row(group_rows: list[dict]) -> dict:
    """对同一个 counts_code 的多个 seed 取平均，生成一行汇总结果。"""
    first = group_rows[0]

    cmax_2_values = [r["cmax_2B"] for r in group_rows]
    cmax_3_values = [r["cmax_3B"] for r in group_rows]
    energy_2_values = [r["energy_2B"] for r in group_rows]
    energy_3_values = [r["energy_3B"] for r in group_rows]
    eta_t_values = [r["eta_T"] for r in group_rows]
    eta_e_values = [r["eta_E"] for r in group_rows]

    finite_lambda_values = [
        r["lambda_star"]
        for r in group_rows
        if not math.isinf(r["lambda_star"])
    ]

    mean_cmax_2 = mean(cmax_2_values)
    mean_cmax_3 = mean(cmax_3_values)
    mean_energy_2 = mean(energy_2_values)
    mean_energy_3 = mean(energy_3_values)
    mean_eta_t = mean(eta_t_values)
    mean_eta_e = mean(eta_e_values)

    # 这里的 mean_lambda_star 保持原逻辑：
    # 每个 seed 先算 lambda_star，再在同一 counts_code 内部求平均。
    # 该列用于保留原表格内容。
    if finite_lambda_values:
        mean_lambda_star = mean(finite_lambda_values)
    else:
        mean_lambda_star = math.inf

    row = {
        "counts_code": first["counts_code"],
        "n1": first["n1"],
        "n2": first["n2"],
        "n3": first["n3"],
        "n4": first["n4"],
        "total_tasks": first["total_tasks"],
        "scenario_type": classify_scenario(first["n1"], first["n2"], first["n3"], first["n4"]),
        "seed_count": len(group_rows),
        "seeds_used": ",".join(str(r["seed"]) for r in sorted(group_rows, key=lambda x: x["seed"])),

        "mean_cmax_2B": round2(mean_cmax_2),
        "mean_cmax_3B": round2(mean_cmax_3),
        "mean_energy_2B": round2(mean_energy_2),
        "mean_energy_3B": round2(mean_energy_3),

        "mean_eta_T_percent": round2(mean_eta_t),
        "mean_eta_E_percent": round2(mean_eta_e),
        "mean_lambda_star": round2(mean_lambda_star),

        "recommend_time_only": recommend_time_only(mean_cmax_2, mean_cmax_3),
    }

    # 动态加入 10 组 lambda 的综合收益和推荐结果。
    for lam in LAMBDA_VALUES:
        label = lambda_label(lam)
        score = mean_eta_t - lam * mean_eta_e
        row[f"S_lambda_{label}"] = round2(score)
        row[f"recommend_lambda_{label}"] = recommend_by_score(score)

    # “只考虑能耗”放在最后，作为能耗极限情况。
    row["recommend_energy_only"] = recommend_energy_only(mean_energy_2, mean_energy_3)

    # 新增：敏感性分析条件区间。
    # 这里只保留 4 个字段：
    # decision_lambda_star
    # lambda_condition_recommend_3arm
    # lambda_condition_recommend_2arm
    # lambda_condition_similar
    condition_info = make_lambda_condition(mean_eta_t, mean_eta_e)
    row.update(condition_info)

    return row


def build_summary_rows(raw_rows: list[dict]) -> tuple[list[dict], int]:
    """按 counts_code 分组，只在同一种输入内部对 seed 取平均。"""
    groups: dict[tuple, list[dict]] = {}
    group_order: list[tuple] = []
    skipped = 0

    for row in raw_rows:
        metrics = calc_one_seed_metrics(row)
        if metrics is None:
            skipped += 1
            continue

        key = (
            metrics["counts_code"],
            metrics["n1"],
            metrics["n2"],
            metrics["n3"],
            metrics["n4"],
        )

        if key not in groups:
            groups[key] = []
            group_order.append(key)

        groups[key].append(metrics)

    summary_rows = []

    # 按输入 CSV 中第一次出现的 counts_code 顺序输出。
    # 这样可以保留 run_basic_case.py 生成结果时的顺序。
    for key in group_order:
        group_rows = sorted(groups[key], key=lambda r: r["seed"])
        summary_rows.append(make_summary_row(group_rows))

    return summary_rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize 2B/3B mode decision by counts_code with seed averaging and 10 lambda values."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="input CSV generated by scripts/run_basic_case.py",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="output summary CSV path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    raw_rows = read_csv(input_path)
    summary_rows, skipped = build_summary_rows(raw_rows)

    save_csv(summary_rows, output_path)

    print("Mode decision summary finished.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Raw rows read: {len(raw_rows)}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Skipped rows because of invalid status or invalid numeric values: {skipped}")
    print("Lambda values used:", ", ".join(str(x) for x in LAMBDA_VALUES))
    print("Note: each counts_code is summarized independently; no overall averaging is performed.")
    print("Sensitivity columns added at the end:")
    print("  decision_lambda_star")
    print("  lambda_condition_recommend_3arm")
    print("  lambda_condition_recommend_2arm")
    print("  lambda_condition_similar")

    if len(raw_rows) > 0 and len(summary_rows) == 0:
        print("\nWarning: input CSV was read, but no valid summary rows were generated.")
        print("Please check whether these columns exist and contain positive numbers:")
        print("  cmax_2B_basic")
        print("  energy_2B_basic")
        print("  cmax_3B_basic")
        print("  energy_3B_basic")


if __name__ == "__main__":
    main()