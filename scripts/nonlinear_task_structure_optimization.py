# -*- coding: utf-8 -*-
"""非线性规划拓展 1：任务结构比例下的第三臂收益最大化。

输入：
    outputs/mode_decision/mode_decision_summary_basic.csv

输出：
    outputs/nonlinear_programming/task_structure_optimization/
        task_structure_response_surface_coefficients.csv
        task_structure_fit_check.csv
        task_structure_existing_best.csv
        task_structure_bounded_optimum.csv

模型改进：
    1. 引入任务规模变量 r = n / 8。
    2. 优化时限制在实验设计可信区域：
           1/n <= p_i <= 2/n
       避免出现 p1=0, p3=0 这种样本外外推。
    3. 同时输出已有样本中的最优结构，方便和连续优化结果对照。

运行：
    python scripts/nonlinear_task_structure_optimization.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "mode_decision"
    / "mode_decision_summary_basic.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "nonlinear_programming"
    / "task_structure_optimization"
)

# 与 mode_decision_summary_basic.csv 保持一致
LAMBDA_VALUES = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]

# 当前实验中每类任务数量为 1 或 2
COUNT_MIN = 1
COUNT_MAX = 2

# 任务规模归一化基准
N_MAX = 8.0

# 网格步长。0.01 较细，0.02 较快。
P_GRID_STEP = 0.01

# 岭回归正则项。用于抑制小样本二次模型过拟合。
RIDGE_ALPHA = 1e-4


def lambda_label(lam: float) -> str:
    """字段名中使用的 lambda 标签。

    0.5 -> 0_5
    1.0 -> 1
    2.0 -> 2
    """
    if abs(lam - int(lam)) < 1e-9:
        return str(int(lam))
    return str(lam).replace(".", "_")


def lambda_display(lam: float) -> str:
    """输出表中展示用的 lambda。"""
    if abs(lam - int(lam)) < 1e-9:
        return str(int(lam))
    return str(lam)


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def round2(value: float) -> float:
    return round(float(value), 2)


def round4(value: float) -> float:
    return round(float(value), 4)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_n(row: dict) -> int:
    """读取任务总数。"""
    if "total_tasks" in row:
        return safe_int(row["total_tasks"])
    if "n" in row:
        return safe_int(row["n"])
    return (
        safe_int(row["n1"])
        + safe_int(row["n2"])
        + safe_int(row["n3"])
        + safe_int(row["n4"])
    )


def get_eta_t(row: dict) -> float:
    if "mean_eta_T_percent" in row:
        return safe_float(row["mean_eta_T_percent"])
    return safe_float(row.get("mean_eta_T", 0.0))


def get_eta_e(row: dict) -> float:
    if "mean_eta_E_percent" in row:
        return safe_float(row["mean_eta_E_percent"])
    return safe_float(row.get("mean_eta_E", 0.0))


def get_proportions(row: dict) -> tuple[float, float, float, float, float]:
    """返回 p1,p2,p3,p4,r。"""
    n1 = safe_float(row["n1"])
    n2 = safe_float(row["n2"])
    n3 = safe_float(row["n3"])
    n4 = safe_float(row["n4"])

    total = n1 + n2 + n3 + n4

    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    p1 = n1 / total
    p2 = n2 / total
    p3 = n3 / total
    p4 = n4 / total

    r = total / N_MAX

    return p1, p2, p3, p4, r


def build_features(p1: float, p2: float, p3: float, p4: float, r: float) -> list[float]:
    """改进后的非线性响应面特征。

    为了避免样本只有 16 个时过拟合，这里不再加入所有 p_i p_j 交互项，
    而是采用：
        常数项
        一次项：p1,p2,p3,p4,r
        平方项：p1^2,p2^2,p3^2,p4^2,r^2
        规模交互项：p1*r,p2*r,p3*r,p4*r

    模型：
        S = a0
          + sum ai*pi + ar*r
          + sum bi*pi^2 + br*r^2
          + sum ci*pi*r
    """
    return [
        1.0,
        p1,
        p2,
        p3,
        p4,
        r,
        p1 * p1,
        p2 * p2,
        p3 * p3,
        p4 * p4,
        r * r,
        p1 * r,
        p2 * r,
        p3 * r,
        p4 * r,
    ]


FEATURE_NAMES = [
    "constant",
    "p1",
    "p2",
    "p3",
    "p4",
    "r",
    "p1_square",
    "p2_square",
    "p3_square",
    "p4_square",
    "r_square",
    "p1_r",
    "p2_r",
    "p3_r",
    "p4_r",
]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元求解 Ax=b。"""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))

        if abs(M[pivot][col]) < 1e-12:
            raise RuntimeError("Linear system is singular or nearly singular.")

        M[col], M[pivot] = M[pivot], M[col]

        pivot_value = M[col][col]

        for j in range(col, n + 1):
            M[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue

            factor = M[row][col]

            if abs(factor) < 1e-15:
                continue

            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    return [M[i][n] for i in range(n)]


def fit_response_surface(rows: list[dict], lam: float) -> tuple[list[float], dict]:
    """用岭回归拟合 S_lambda 的响应面。"""
    X = []
    y = []

    for row in rows:
        p1, p2, p3, p4, r = get_proportions(row)
        eta_t = get_eta_t(row)
        eta_e = get_eta_e(row)

        score = eta_t - lam * eta_e

        X.append(build_features(p1, p2, p3, p4, r))
        y.append(score)

    n_features = len(FEATURE_NAMES)

    XtX = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    Xty = [0.0 for _ in range(n_features)]

    for xi, yi in zip(X, y):
        for i in range(n_features):
            Xty[i] += xi[i] * yi

            for j in range(n_features):
                XtX[i][j] += xi[i] * xi[j]

    # 岭回归正则项
    for i in range(n_features):
        XtX[i][i] += RIDGE_ALPHA

    coef = solve_linear_system(XtX, Xty)

    preds = [dot(coef, xi) for xi in X]

    y_mean = sum(y) / len(y)
    sse = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
    sst = sum((yi - y_mean) ** 2 for yi in y)

    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    rmse = math.sqrt(sse / len(y))
    max_abs_error = max(abs(yi - pi) for yi, pi in zip(y, preds))

    metrics = {
        "sample_count": len(y),
        "r2": r2,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
    }

    return coef, metrics


def predict_score(
    coef: list[float],
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    r: float,
) -> float:
    return dot(coef, build_features(p1, p2, p3, p4, r))


def get_recommendation(score: float) -> str:
    if score > 0:
        return "recommend_3arm"
    if score < 0:
        return "recommend_2arm"
    return "similar_prefer_2arm"


def get_unique_n_levels(rows: list[dict]) -> list[int]:
    levels = sorted({get_n(row) for row in rows if get_n(row) > 0})
    return levels


def bounded_simplex_grid(n: int, step: float):
    """生成受实验设计约束的 p 网格。

    由于当前实验中每类任务数量为 1 或 2，
    对于给定任务总数 n，有：
        1/n <= p_i <= 2/n

    这样可以避免无约束优化跑到 p_i=0 的样本外区域。
    """
    unit = int(round(1.0 / step))

    lower = COUNT_MIN / n
    upper = COUNT_MAX / n

    for i in range(unit + 1):
        p1 = i / unit
        if p1 < lower - 1e-12 or p1 > upper + 1e-12:
            continue

        for j in range(unit - i + 1):
            p2 = j / unit
            if p2 < lower - 1e-12 or p2 > upper + 1e-12:
                continue

            for k in range(unit - i - j + 1):
                p3 = k / unit
                p4 = (unit - i - j - k) / unit

                if p3 < lower - 1e-12 or p3 > upper + 1e-12:
                    continue

                if p4 < lower - 1e-12 or p4 > upper + 1e-12:
                    continue

                yield p1, p2, p3, p4


def find_nearest_existing_structure(
    rows: list[dict],
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    r: float,
) -> tuple[dict, float]:
    """寻找与连续解最接近的已有 counts_code。"""
    best_row = None
    best_dist = float("inf")

    for row in rows:
        q1, q2, q3, q4, qr = get_proportions(row)

        dist = math.sqrt(
            (p1 - q1) ** 2
            + (p2 - q2) ** 2
            + (p3 - q3) ** 2
            + (p4 - q4) ** 2
            + (r - qr) ** 2
        )

        if dist < best_dist:
            best_dist = dist
            best_row = row

    return best_row, best_dist


def optimize_bounded_structure(
    coef: list[float],
    rows: list[dict],
    step: float,
) -> dict:
    """在受信任区域内求最大 S。"""
    best_score = -float("inf")
    best_solution = None

    for n in get_unique_n_levels(rows):
        r = n / N_MAX

        for p1, p2, p3, p4 in bounded_simplex_grid(n, step):
            score = predict_score(coef, p1, p2, p3, p4, r)

            if score > best_score:
                best_score = score
                best_solution = (n, r, p1, p2, p3, p4)

    if best_solution is None:
        raise RuntimeError("No feasible solution found. Please check grid step or bounds.")

    n, r, p1, p2, p3, p4 = best_solution
    nearest_row, nearest_dist = find_nearest_existing_structure(rows, p1, p2, p3, p4, r)

    return {
        "optimal_n": n,
        "optimal_r": r,
        "optimal_p1": p1,
        "optimal_p2": p2,
        "optimal_p3": p3,
        "optimal_p4": p4,
        "equivalent_n1": p1 * n,
        "equivalent_n2": p2 * n,
        "equivalent_n3": p3 * n,
        "equivalent_n4": p4 * n,
        "predicted_max_S": best_score,
        "recommendation_at_optimum": get_recommendation(best_score),
        "nearest_counts_code": nearest_row["counts_code"],
        "nearest_scenario_type": nearest_row["scenario_type"],
        "nearest_distance": nearest_dist,
    }


def build_coefficient_rows(rows: list[dict]) -> tuple[list[dict], dict[str, list[float]]]:
    coefficient_rows = []
    models = {}

    for lam in LAMBDA_VALUES:
        label = lambda_label(lam)

        coef, metrics = fit_response_surface(rows, lam)
        models[label] = coef

        row = {
            "lambda": lambda_display(lam),
            "lambda_label": label,
            "sample_count": metrics["sample_count"],
            "r2": round4(metrics["r2"]),
            "rmse": round4(metrics["rmse"]),
            "max_abs_error": round4(metrics["max_abs_error"]),
            "ridge_alpha": RIDGE_ALPHA,
        }

        for name, value in zip(FEATURE_NAMES, coef):
            row[f"coef_{name}"] = round4(value)

        coefficient_rows.append(row)

    return coefficient_rows, models


def build_fit_check_rows(rows: list[dict], models: dict[str, list[float]]) -> list[dict]:
    result_rows = []

    for row in rows:
        p1, p2, p3, p4, r = get_proportions(row)
        n = get_n(row)

        result = {
            "counts_code": row["counts_code"],
            "scenario_type": row["scenario_type"],
            "total_tasks": n,
            "p1": round4(p1),
            "p2": round4(p2),
            "p3": round4(p3),
            "p4": round4(p4),
            "r": round4(r),
            "mean_eta_T_percent": round2(get_eta_t(row)),
            "mean_eta_E_percent": round2(get_eta_e(row)),
        }

        for lam in LAMBDA_VALUES:
            label = lambda_label(lam)

            actual_score = get_eta_t(row) - lam * get_eta_e(row)
            pred_score = predict_score(models[label], p1, p2, p3, p4, r)

            result[f"actual_S_lambda_{label}"] = round2(actual_score)
            result[f"predicted_S_lambda_{label}"] = round2(pred_score)
            result[f"fit_error_lambda_{label}"] = round2(pred_score - actual_score)

        result_rows.append(result)

    return result_rows


def build_existing_best_rows(rows: list[dict]) -> list[dict]:
    """找出每个 lambda 下已有实验样本中的最高收益结构。"""
    best_rows = []

    for lam in LAMBDA_VALUES:
        best_row = None
        best_score = -float("inf")

        for row in rows:
            score = get_eta_t(row) - lam * get_eta_e(row)

            if score > best_score:
                best_score = score
                best_row = row

        p1, p2, p3, p4, r = get_proportions(best_row)

        best_rows.append(
            {
                "lambda": lambda_display(lam),
                "lambda_label": lambda_label(lam),
                "best_counts_code": best_row["counts_code"],
                "best_scenario_type": best_row["scenario_type"],
                "total_tasks": get_n(best_row),
                "p1": round4(p1),
                "p2": round4(p2),
                "p3": round4(p3),
                "p4": round4(p4),
                "r": round4(r),
                "actual_best_S": round2(best_score),
                "recommendation": get_recommendation(best_score),
                "mean_eta_T_percent": round2(get_eta_t(best_row)),
                "mean_eta_E_percent": round2(get_eta_e(best_row)),
            }
        )

    return best_rows


def build_bounded_optimum_rows(
    rows: list[dict],
    models: dict[str, list[float]],
    step: float,
) -> list[dict]:
    optimum_rows = []

    for lam in LAMBDA_VALUES:
        label = lambda_label(lam)
        coef = models[label]

        opt = optimize_bounded_structure(coef, rows, step)

        optimum_rows.append(
            {
                "lambda": lambda_display(lam),
                "lambda_label": label,
                "p_grid_step": step,
                "optimal_n": opt["optimal_n"],
                "optimal_r": round4(opt["optimal_r"]),
                "optimal_p1": round4(opt["optimal_p1"]),
                "optimal_p2": round4(opt["optimal_p2"]),
                "optimal_p3": round4(opt["optimal_p3"]),
                "optimal_p4": round4(opt["optimal_p4"]),
                "equivalent_n1": round2(opt["equivalent_n1"]),
                "equivalent_n2": round2(opt["equivalent_n2"]),
                "equivalent_n3": round2(opt["equivalent_n3"]),
                "equivalent_n4": round2(opt["equivalent_n4"]),
                "predicted_max_S": round2(opt["predicted_max_S"]),
                "recommendation_at_optimum": opt["recommendation_at_optimum"],
                "nearest_counts_code": opt["nearest_counts_code"],
                "nearest_scenario_type": opt["nearest_scenario_type"],
                "nearest_distance": round4(opt["nearest_distance"]),
                "constraint_note": "bounded by 1/n <= p_i <= 2/n to avoid extrapolation",
            }
        )

    return optimum_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trust-region nonlinear task structure optimization."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="input mode_decision_summary_basic.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory",
    )
    parser.add_argument(
        "--p-grid-step",
        type=float,
        default=P_GRID_STEP,
        help="grid step for p1,p2,p3,p4",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        print("Please make sure mode_decision_summary_basic.csv exists.")
        return

    rows = read_csv(input_path)

    coefficient_rows, models = build_coefficient_rows(rows)
    fit_check_rows = build_fit_check_rows(rows, models)
    existing_best_rows = build_existing_best_rows(rows)
    bounded_optimum_rows = build_bounded_optimum_rows(rows, models, args.p_grid_step)

    save_csv(
        coefficient_rows,
        output_dir / "task_structure_response_surface_coefficients.csv",
    )

    save_csv(
        fit_check_rows,
        output_dir / "task_structure_fit_check.csv",
    )

    save_csv(
        existing_best_rows,
        output_dir / "task_structure_existing_best.csv",
    )

    save_csv(
        bounded_optimum_rows,
        output_dir / "task_structure_bounded_optimum.csv",
    )

    print("Trust-region nonlinear task structure optimization finished.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("Generated:")
    print("  task_structure_response_surface_coefficients.csv")
    print("  task_structure_fit_check.csv")
    print("  task_structure_existing_best.csv")
    print("  task_structure_bounded_optimum.csv")
    print()
    print("Model:")
    print("  S_lambda = nonlinear response surface of p1,p2,p3,p4,r")
    print("  r = n / 8")
    print("  constraints: p1+p2+p3+p4=1, 1/n <= p_i <= 2/n")
    print()
    print("This version avoids unconstrained extrapolation such as p_i = 0.")


if __name__ == "__main__":
    main()