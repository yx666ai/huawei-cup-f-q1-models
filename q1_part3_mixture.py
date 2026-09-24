#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9: regularized mixture-response modelling with support-aware optimization.

The script uses A4/A5 only for fitting and hyper-parameter selection.  A6/A7
is a frozen 1M test, A8--A11 are zero-refit scale-transfer tests, and A12--A15
are model-estimated extrapolation tables used only for descriptive stability.
It does not recompute any Step 2--8 quality preprocessing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


SEED = 20260923
PAIR_SPECS = [
    ("train_1m", "1M", "train_mixture_1m.csv", "train_pile_loss_1m.csv", "training"),
    ("blind_1m", "1M", "test_mixture_1m.csv", "test_pile_loss_1m.csv", "blind_test"),
    ("transfer_60m", "60M", "test_mixture_60m.csv", "test_pile_loss_60m.csv", "zero_refit_transfer"),
    ("transfer_1B", "1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv", "zero_refit_transfer"),
    ("estimated_10b", "10B", "est_mixture_10b.csv", "est_pile_loss_10b.csv", "estimated_extrapolation"),
    ("estimated_70b", "70B", "est_mixture_70b.csv", "est_pile_loss_70b.csv", "estimated_extrapolation"),
]
RIDGE_ALPHAS = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0])
ENET_ALPHAS = np.array([1e-5, 1e-4, 1e-3, 1e-2])
ENET_L1_RATIOS = np.array([0.1, 0.5, 0.9])
SUPPORT_K_VALUES = (20, 30, 50)
SUPPORT_MAIN_K = 30
STABILITY_REPEATS = 3
STABILITY_FOLDS = 5
ENET_MAX_ITER = 30000


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label_cn: str
    family: str
    order: int


MODEL_SPECS = [
    ModelSpec("ridge_linear", "一阶 Ridge", "ridge", 1),
    ModelSpec("ridge_quadratic", "二阶 Ridge", "ridge", 2),
    ModelSpec("elasticnet_quadratic", "二阶 ElasticNet", "elasticnet", 2),
]
SPEC_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}


def find_input_root(cwd: Path) -> Path:
    candidates = [
        cwd / "real_attachments" / "A_data_value",
        Path.home() / "Desktop" / "F题" / "real_attachments" / "A_data_value",
    ]
    for candidate in candidates:
        if (candidate / "regmix_tables" / "train_mixture_1m.csv").exists():
            return candidate
    raise FileNotFoundError("未找到 A4--A16 数据目录。")


def short_mix_name(col: str) -> str:
    return col.removeprefix("train_the_pile_")


def short_loss_name(col: str) -> str:
    return col.removeprefix("metric/the_pile_").removesuffix("_val_loss")


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3 or np.std(y_true[mask]) <= 1e-15 or np.std(y_pred[mask]) <= 1e-15:
        return np.nan
    return float(spearmanr(y_true[mask], y_pred[mask]).statistic)


def top_fraction_overlap(y_true: np.ndarray, y_pred: np.ndarray, fraction: float = 0.10) -> float:
    n = len(y_true)
    top_n = max(1, math.ceil(fraction * n))
    true_top = set(np.argsort(y_true)[:top_n].tolist())
    pred_top = set(np.argsort(y_pred)[:top_n].tolist())
    return float(len(true_top & pred_top) / top_n)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "spearman": safe_spearman(y_true, y_pred),
        "best_candidate_regret": float(y_true[int(np.argmin(y_pred))] - np.min(y_true)),
        "top10pct_overlap_rate": top_fraction_overlap(y_true, y_pred, 0.10),
        "mean_residual_actual_minus_pred": float(np.mean(y_true - y_pred)),
    }


def setup_plot_style() -> str:
    sns.set_theme(style="whitegrid", context="notebook")
    font_manager = mpl.font_manager
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
        font_name = next((name for name in candidates if name in available), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 301,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    return font_name


def finalize_png(path: Path) -> None:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.save(path, format="PNG", dpi=(301, 301), optimize=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    finalize_png(path)


def align_pair(
    regmix_dir: Path,
    pair_name: str,
    scale: str,
    mixture_file: str,
    loss_file: str,
    role: str,
    expected_mix_cols: list[str] | None,
    expected_loss_cols: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], dict, list[dict]]:
    mix = pd.read_csv(regmix_dir / mixture_file)
    loss = pd.read_csv(regmix_dir / loss_file)
    if "index" not in mix.columns or "index" not in loss.columns:
        raise ValueError(f"{pair_name}: 缺少 index 列")
    if not mix["index"].is_unique or not loss["index"].is_unique:
        raise ValueError(f"{pair_name}: index 存在重复")
    if set(mix["index"]) != set(loss["index"]):
        only_mix = sorted(set(mix["index"]) - set(loss["index"]))[:10]
        only_loss = sorted(set(loss["index"]) - set(mix["index"]))[:10]
        raise ValueError(f"{pair_name}: index 错位；仅配比={only_mix}，仅Loss={only_loss}")

    mix_cols = [c for c in mix.columns if c.startswith("train_the_pile_")]
    loss_cols = [c for c in loss.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]
    if len(mix_cols) != 17 or len(loss_cols) != 13:
        raise ValueError(f"{pair_name}: 预期17个配比列和13个Loss列，实际{len(mix_cols)}、{len(loss_cols)}")
    if expected_mix_cols is not None and mix_cols != expected_mix_cols:
        raise ValueError(f"{pair_name}: 配比列或顺序与训练集不一致")
    if expected_loss_cols is not None and loss_cols != expected_loss_cols:
        raise ValueError(f"{pair_name}: Loss列或顺序与训练集不一致")

    merged = mix.merge(loss, on="index", how="inner", validate="one_to_one", sort=True)
    if len(merged) != len(mix) or len(merged) != len(loss):
        raise ValueError(f"{pair_name}: 一一连接后行数改变")
    if merged[mix_cols + loss_cols].isna().any().any():
        raise ValueError(f"{pair_name}: 配比或Loss存在缺失")

    values = merged[mix_cols].to_numpy(dtype=float)
    row_sums = values.sum(axis=1)
    negative_count = int((values < 0).sum())
    zero_sum_count = int((row_sums <= 0).sum())
    if negative_count or zero_sum_count:
        raise ValueError(f"{pair_name}: 发现负配比({negative_count})或非正配比和({zero_sum_count})")
    severe_mask = np.abs(row_sums - 1.0) > 0.02
    anomalies: list[dict] = []
    for idx, total in zip(merged.loc[severe_mask, "index"], row_sums[severe_mask]):
        anomalies.append({
            "record_type": "anomaly",
            "source_pair": pair_name,
            "index": idx,
            "warning": f"配比和严重偏离1: {total:.8f}",
        })
    merged["mixture_sum_original"] = row_sums
    merged["mixture_renormalized"] = np.abs(row_sums - 1.0) > 1e-12
    merged.loc[:, mix_cols] = values / row_sums[:, None]
    merged["composite_loss_equal13"] = merged[loss_cols].mean(axis=1)
    rounded = merged[mix_cols].round(12).astype(str).agg("|".join, axis=1)
    merged["recipe_group_id"] = rounded.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:20])
    merged.insert(0, "evidence_role", role)
    merged.insert(0, "scale", scale)
    merged.insert(0, "source_pair", pair_name)

    summary = {
        "record_type": "pair_summary",
        "source_pair": pair_name,
        "scale": scale,
        "evidence_role": role,
        "mixture_file": mixture_file,
        "loss_file": loss_file,
        "rows_mixture": len(mix),
        "rows_loss": len(loss),
        "rows_aligned": len(merged),
        "index_unique_mixture": bool(mix["index"].is_unique),
        "index_unique_loss": bool(loss["index"].is_unique),
        "index_sets_equal": True,
        "n_mixture_columns": len(mix_cols),
        "n_loss_columns": len(loss_cols),
        "mixture_columns": " | ".join(mix_cols),
        "loss_columns": " | ".join(loss_cols),
        "mixture_sum_min_before": float(row_sums.min()),
        "mixture_sum_max_before": float(row_sums.max()),
        "mixture_sum_max_abs_deviation": float(np.max(np.abs(row_sums - 1.0))),
        "renormalized_rows": int((np.abs(row_sums - 1.0) > 1e-12).sum()),
        "negative_cells": negative_count,
        "severe_deviation_rows": int(severe_mask.sum()),
        "duplicate_recipe_rows": int(merged["recipe_group_id"].duplicated(keep=False).sum()),
        "unique_recipe_groups": int(merged["recipe_group_id"].nunique()),
        "missing_mixture_cells": int(mix[mix_cols].isna().sum().sum()),
        "missing_loss_cells": int(loss[loss_cols].isna().sum().sum()),
        "warning": "",
    }
    return merged, mix_cols, loss_cols, summary, anomalies


def build_design(P: np.ndarray, pairs: list[tuple[int, int]], order: int) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if order == 1:
        return P.copy()
    interactions = np.column_stack([P[:, i] * P[:, j] for i, j in pairs])
    return np.column_stack([P, interactions])


def make_estimator(family: str, params: dict):
    if family == "ridge":
        return Ridge(alpha=float(params["alpha"]), fit_intercept=False, solver="cholesky", tol=1e-8)
    return ElasticNet(
        alpha=float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
        fit_intercept=False,
        max_iter=ENET_MAX_ITER,
        tol=1e-6,
        selection="cyclic",
        precompute=True,
        random_state=SEED,
    )


def candidate_grid(family: str) -> list[dict]:
    if family == "ridge":
        return [{"alpha": float(a), "l1_ratio": np.nan} for a in RIDGE_ALPHAS]
    return [
        {"alpha": float(a), "l1_ratio": float(r)}
        for a, r in itertools.product(ENET_ALPHAS, ENET_L1_RATIOS)
    ]


def grouped_splits(X: np.ndarray, groups: np.ndarray, max_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = int(pd.Series(groups).nunique())
    n_splits = min(max_splits, n_groups)
    if n_splits < 2:
        raise ValueError("独立配方组不足，无法进行 GroupKFold")
    return list(GroupKFold(n_splits=n_splits).split(X, groups=groups))


def repeated_group_folds(groups: np.ndarray, n_splits: int, repeats: int, seed: int):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 1009 * repeat)
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, n_splits)
        for fold, val_groups in enumerate(chunks, start=1):
            va = np.flatnonzero(np.isin(groups, val_groups))
            tr = np.flatnonzero(~np.isin(groups, val_groups))
            yield repeat + 1, fold, tr, va


def select_params(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    family: str,
    max_splits: int = 4,
) -> tuple[dict, float]:
    splits = grouped_splits(X, groups, max_splits)
    prepared = []
    for tr, va in splits:
        scaler = StandardScaler(with_mean=False)
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        prepared.append((Xtr, y[tr], Xva, y[va]))
    best_params: dict | None = None
    best_rmse = np.inf
    for params in candidate_grid(family):
        fold_rmse = []
        for Xtr, ytr, Xva, yva in prepared:
            estimator = make_estimator(family, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(Xtr, ytr)
            pred = estimator.predict(Xva)
            fold_rmse.append(float(np.sqrt(mean_squared_error(yva, pred))))
        score = float(np.mean(fold_rmse))
        if score < best_rmse - 1e-15:
            best_rmse = score
            best_params = params.copy()
    if best_params is None:
        raise RuntimeError("内部交叉验证未能选择超参数")
    return best_params, best_rmse


def fit_final_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    family: str,
    inner_splits: int = 5,
) -> tuple[StandardScaler, object, dict, np.ndarray, np.ndarray]:
    params, inner_rmse = select_params(X, y, groups, family, max_splits=inner_splits)
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    estimator = make_estimator(family, params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(Xs, y)
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    coef_scaled = np.asarray(estimator.coef_, dtype=float)
    coef_original = coef_scaled / np.asarray(scaler.scale_, dtype=float)
    n_iter_raw = getattr(estimator, "n_iter_", 0)
    n_iter_value = 0 if n_iter_raw is None else int(np.max(np.atleast_1d(n_iter_raw)))
    params = {
        **params,
        "full_train_inner_cv_rmse": inner_rmse,
        "n_iter": n_iter_value,
        "converged": len(convergence_warnings) == 0 and (
            family != "elasticnet" or n_iter_value < ENET_MAX_ITER
        ),
        "convergence_warning": " | ".join(str(w.message) for w in convergence_warnings),
    }
    return scaler, estimator, params, coef_scaled, coef_original


def refit_with_frozen_params(
    X: np.ndarray,
    y: np.ndarray,
    family: str,
    params: dict,
) -> tuple[np.ndarray, bool, str]:
    """Fast resample refit using hyperparameters already selected within A4/A5."""
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    estimator = make_estimator(family, params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(Xs, y)
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    n_iter_raw = getattr(estimator, "n_iter_", 0)
    n_iter_value = 0 if n_iter_raw is None else int(np.max(np.atleast_1d(n_iter_raw)))
    converged = len(convergence_warnings) == 0 and (
        family != "elasticnet" or n_iter_value < ENET_MAX_ITER
    )
    coef_original = np.asarray(estimator.coef_, dtype=float) / np.asarray(scaler.scale_, dtype=float)
    warning_text = " | ".join(str(w.message) for w in convergence_warnings)
    return coef_original, bool(converged), warning_text


def fit_nested_target(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    spec: ModelSpec,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    target: str,
) -> tuple[np.ndarray, list[dict], tuple]:
    oof = np.full(len(y), np.nan)
    rows: list[dict] = []
    for fold, (tr, va) in enumerate(outer_splits, start=1):
        params, inner_rmse = select_params(X[tr], y[tr], groups[tr], spec.family, max_splits=4)
        scaler = StandardScaler(with_mean=False)
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        estimator = make_estimator(spec.family, params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(Xtr, y[tr])
        convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        n_iter_raw = getattr(estimator, "n_iter_", 0)
        n_iter_value = 0 if n_iter_raw is None else int(np.max(np.atleast_1d(n_iter_raw)))
        pred = estimator.predict(Xva)
        oof[va] = pred
        rows.append({
            "model_id": spec.model_id,
            "model_label": spec.label_cn,
            "model_family": spec.family,
            "mixture_order": spec.order,
            "target": target,
            "fold": fold,
            "n_train": len(tr),
            "n_validation": len(va),
            "alpha": params["alpha"],
            "l1_ratio": params.get("l1_ratio", np.nan),
            "inner_cv_rmse": inner_rmse,
            "n_iter": n_iter_value,
            "converged": len(convergence_warnings) == 0 and (
                spec.family != "elasticnet" or n_iter_value < ENET_MAX_ITER
            ),
            "convergence_warning": " | ".join(str(w.message) for w in convergence_warnings),
            **regression_metrics(y[va], pred),
            "validation_scheme": "nested_GroupKFold_outer5_inner4",
        })
    rows.append({
        "model_id": spec.model_id,
        "model_label": spec.label_cn,
        "model_family": spec.family,
        "mixture_order": spec.order,
        "target": target,
        "fold": "overall_oof",
        "n_train": len(y),
        "n_validation": len(y),
        "alpha": np.nan,
        "l1_ratio": np.nan,
        "inner_cv_rmse": np.nan,
        "n_iter": np.nan,
        "converged": bool(all(row.get("converged", True) for row in rows)),
        "convergence_warning": "",
        **regression_metrics(y, oof),
        "validation_scheme": "nested_GroupKFold_out_of_fold",
    })
    final = fit_final_model(X, y, groups, spec.family, inner_splits=5)
    return oof, rows, final


def predict_fitted(fitted: tuple, X: np.ndarray) -> np.ndarray:
    scaler, estimator, _, _, _ = fitted
    return np.asarray(estimator.predict(scaler.transform(X)), dtype=float)


def surface_value(p: np.ndarray, coef: np.ndarray, d: int, pairs: list[tuple[int, int]], order: int) -> float:
    value = float(np.dot(coef[:d], p))
    if order == 2:
        for c, (i, j) in zip(coef[d:], pairs):
            value += float(c * p[i] * p[j])
    return value


def surface_gradient(p: np.ndarray, coef: np.ndarray, d: int, pairs: list[tuple[int, int]], order: int) -> np.ndarray:
    grad = np.array(coef[:d], dtype=float, copy=True)
    if order == 2:
        for c, (i, j) in zip(coef[d:], pairs):
            grad[i] += c * p[j]
            grad[j] += c * p[i]
    return grad


def optimize_wide(
    coef: np.ndarray,
    order: int,
    starts: list[np.ndarray],
    d: int,
    pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, float, int]:
    constraints = [{"type": "eq", "fun": lambda p: float(np.sum(p) - 1.0), "jac": lambda p: np.ones_like(p)}]
    bounds = [(0.0, 1.0)] * d
    best_x = None
    best_fun = np.inf
    successes = 0
    for start in starts:
        res = minimize(
            fun=lambda p: surface_value(p, coef, d, pairs, order),
            x0=np.asarray(start, dtype=float),
            jac=lambda p: surface_gradient(p, coef, d, pairs, order),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1500, "ftol": 1e-12, "disp": False},
        )
        if res.success and np.isfinite(res.fun) and abs(np.sum(res.x) - 1.0) < 1e-7 and np.min(res.x) >= -1e-8:
            successes += 1
            x = np.clip(res.x, 0.0, 1.0)
            x /= x.sum()
            fun = surface_value(x, coef, d, pairs, order)
            if fun < best_fun:
                best_fun = fun
                best_x = x
    if best_x is None:
        raise RuntimeError("宽单纯形 SLSQP 多起点优化未获得可行解")
    return best_x, float(best_fun), successes


def optimize_convex_hull(
    P_local: np.ndarray,
    coef: np.ndarray,
    order: int,
    d: int,
    pairs: list[tuple[int, int]],
    center_local_position: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    n = len(P_local)
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)}]
    bounds = [(0.0, 1.0)] * n
    one_hot = np.zeros(n)
    one_hot[center_local_position] = 1.0
    rng = np.random.default_rng(seed)
    starts = [one_hot, np.ones(n) / n, rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))]
    best_p = None
    best_w = None
    best_fun = np.inf
    successes = 0

    def objective(w: np.ndarray) -> float:
        return surface_value(w @ P_local, coef, d, pairs, order)

    def jacobian(w: np.ndarray) -> np.ndarray:
        p = w @ P_local
        return P_local @ surface_gradient(p, coef, d, pairs, order)

    for start in starts:
        res = minimize(
            objective,
            start,
            jac=jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-11, "disp": False},
        )
        if res.success and np.isfinite(res.fun) and abs(np.sum(res.x) - 1.0) < 1e-7 and np.min(res.x) >= -1e-8:
            successes += 1
            w = np.clip(res.x, 0.0, 1.0)
            w /= w.sum()
            p = w @ P_local
            fun = surface_value(p, coef, d, pairs, order)
            if fun < best_fun:
                best_fun = fun
                best_p = p
                best_w = w
    if best_p is None:
        raise RuntimeError("局部凸包优化未获得可行解")
    assert best_w is not None
    return np.asarray(best_p), np.asarray(best_w), float(best_fun), successes


def local_support_search(
    P: np.ndarray,
    actual_composite: np.ndarray,
    coef: np.ndarray,
    order: int,
    d: int,
    pairs: list[tuple[int, int]],
    k_values: tuple[int, ...],
    seed: int,
) -> tuple[list[dict], dict[int, dict]]:
    n_centers = max(1, math.ceil(0.10 * len(P)))
    centers = np.argsort(actual_composite)[:n_centers]
    records: list[dict] = []
    best_by_k: dict[int, dict] = {}
    for k in k_values:
        k_eff = min(int(k), len(P))
        seen_neighborhoods: set[tuple[int, ...]] = set()
        for rank, center in enumerate(centers, start=1):
            distances = np.abs(P - P[center]).sum(axis=1)
            neighbors = np.argsort(distances)[:k_eff]
            key = tuple(sorted(neighbors.tolist()))
            if key in seen_neighborhoods:
                continue
            seen_neighborhoods.add(key)
            local_pos = int(np.flatnonzero(neighbors == center)[0])
            try:
                p, weights, pred_loss, successes = optimize_convex_hull(
                    P[neighbors], coef, order, d, pairs, local_pos, seed + 100000 * k + int(center)
                )
            except RuntimeError:
                continue
            nearest_l1 = float(np.min(np.abs(P - p).sum(axis=1)))
            record = {
                "K": k,
                "K_effective": k_eff,
                "center_row_position": int(center),
                "center_rank_by_actual_loss": rank,
                "center_actual_composite_loss": float(actual_composite[center]),
                "predicted_composite_loss": pred_loss,
                "nearest_training_L1": nearest_l1,
                "successful_multistarts": successes,
                "neighborhood_size": len(neighbors),
                "neighborhood_rows": "|".join(map(str, neighbors.tolist())),
                "neighbors": neighbors,
                "weights": weights,
                "p": p,
            }
            records.append(record)
            if k not in best_by_k or pred_loss < best_by_k[k]["predicted_composite_loss"]:
                best_by_k[k] = record
        if k not in best_by_k:
            raise RuntimeError(f"K={k} 的局部支持搜索没有可行结果")
    return records, best_by_k


def nearest_l1(P: np.ndarray, p: np.ndarray) -> float:
    return float(np.min(np.abs(P - p).sum(axis=1)))


def interval_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(a_hi, b_hi) - min(a_lo, b_lo)
    return 1.0 if union <= 1e-15 else inter / union


def markdown_table(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def finite_difference_complementarity(
    p: np.ndarray,
    coef: np.ndarray,
    order: int,
    d: int,
    pairs: list[tuple[int, int]],
    i: int,
    j: int,
    requested_delta: float,
) -> tuple[float, float, float]:
    donor_mask = np.ones(d, dtype=bool)
    donor_mask[[i, j]] = False
    donor_mass = float(p[donor_mask].sum())
    if donor_mass <= 1e-15:
        return np.nan, np.nan, 0.0
    delta = min(float(requested_delta), 0.45 * donor_mass)
    donor_weights = np.zeros(d)
    donor_weights[donor_mask] = p[donor_mask] / donor_mass
    delta_i = np.zeros(d)
    delta_j = np.zeros(d)
    delta_i[i] = delta
    delta_j[j] = delta
    delta_i -= delta * donor_weights
    delta_j -= delta * donor_weights
    p_i = p + delta_i
    p_j = p + delta_j
    p_ij = p + delta_i + delta_j
    if min(p_i.min(), p_j.min(), p_ij.min()) < -1e-9:
        raise RuntimeError("组合效应有限差分产生负配比")
    base = surface_value(p, coef, d, pairs, order)
    s = (
        surface_value(p_ij, coef, d, pairs, order)
        - surface_value(p_i, coef, d, pairs, order)
        - surface_value(p_j, coef, d, pairs, order)
        + base
    )
    return float(s), float(s / (delta ** 2)), float(delta)


def build_effect_tables(
    model_id: str,
    mix_names: list[str],
    target_names: list[str],
    coef_matrix: np.ndarray,
    composite_coef: np.ndarray,
    order: int,
    reference_points: dict[str, np.ndarray],
    pairs: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build simplex-respecting transfer and pair-effect diagnostics."""
    d = len(mix_names)
    target_coef_map = {target: coef_matrix[k] for k, target in enumerate(target_names)}
    target_coef_map["composite_equal13"] = composite_coef

    directional_rows: list[dict] = []
    for reference_name, p_ref in reference_points.items():
        for target, coef in target_coef_map.items():
            grad = surface_gradient(p_ref, coef, d, pairs, order)
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue
                    derivative = float(grad[j] - grad[i])
                    eps = min(1e-5, 0.25 * p_ref[i]) if p_ref[i] > 0 else 0.0
                    if eps > 0:
                        p_fd = p_ref.copy()
                        p_fd[i] -= eps
                        p_fd[j] += eps
                        fd = (
                            surface_value(p_fd, coef, d, pairs, order)
                            - surface_value(p_ref, coef, d, pairs, order)
                        ) / eps
                    else:
                        fd = np.nan
                    directional_rows.append({
                        "model_id": model_id,
                        "reference_point": reference_name,
                        "target_loss_domain": target,
                        "from_domain": mix_names[i],
                        "to_domain": mix_names[j],
                        "donor_share": float(p_ref[i]),
                        "receiver_share": float(p_ref[j]),
                        "directional_derivative_loss_per_unit_transfer": derivative,
                        "finite_difference_check": fd,
                        "analytic_minus_finite_difference": derivative - fd if np.isfinite(fd) else np.nan,
                        "predicted_loss_change_for_1pct_transfer": derivative * 0.01,
                        "one_percent_transfer_feasible": bool(p_ref[i] >= 0.01 - 1e-12),
                        "interpretation": (
                            "负值表示该保总量转移预计降低Loss"
                            if derivative < 0
                            else "正值表示该保总量转移预计提高Loss"
                        ),
                    })

    complementarity_rows: list[dict] = []
    for reference_name, p_ref in reference_points.items():
        for target, coef in target_coef_map.items():
            for i, j in pairs:
                for requested_delta in (0.005, 0.01, 0.02):
                    s_value, normalized, actual_delta = finite_difference_complementarity(
                        p_ref, coef, order, d, pairs, i, j, requested_delta
                    )
                    if actual_delta <= 0 or not np.isfinite(s_value):
                        interpretation = "基准点其余域无可供转移质量，局部组合效应不可计算"
                    elif s_value < 0:
                        interpretation = "局部互补：共同增加的Loss低于两个单独效应之和"
                    else:
                        interpretation = "局部竞争：共同增加的Loss高于两个单独效应之和"
                    complementarity_rows.append({
                        "model_id": model_id,
                        "reference_point": reference_name,
                        "target_loss_domain": target,
                        "domain_i": mix_names[i],
                        "domain_j": mix_names[j],
                        "requested_delta": requested_delta,
                        "actual_delta": actual_delta,
                        "second_finite_difference": s_value,
                        "normalized_by_delta_squared": normalized,
                        "interpretation": interpretation,
                        "donor_policy": "从除i、j外其余域按基准占比比例抽出总量",
                        "causal_claim": False,
                    })
    return pd.DataFrame(directional_rows), pd.DataFrame(complementarity_rows)


def safe_file_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def main() -> None:
    cwd = Path.cwd()
    input_root = find_input_root(cwd)
    regmix_dir = input_root / "regmix_tables"
    output_dir = cwd / "step9_mixture_response"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    font_name = setup_plot_style()

    run_config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": SEED,
        "input_root": str(input_root.resolve()),
        "output_root": str(output_dir.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": mpl.__version__,
            "seaborn": sns.__version__,
        },
        "font": font_name,
        "ridge_alphas": RIDGE_ALPHAS.tolist(),
        "elasticnet_alphas": ENET_ALPHAS.tolist(),
        "elasticnet_l1_ratios": ENET_L1_RATIOS.tolist(),
        "support_K_values": list(SUPPORT_K_VALUES),
        "support_main_K": SUPPORT_MAIN_K,
        "stability_repeats": STABILITY_REPEATS,
        "stability_folds": STABILITY_FOLDS,
        "stability_refit_policy": (
            "A4/A5内部CV选出的全样本超参数在重复折中冻结；每折仅重拟合系数、"
            "重建局部支持域并优化。稳定性不作为外部误差估计。"
        ),
        "composite_loss_policy": "13 validation-domain losses weighted equally; policy choice, not an official total loss",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 1. Alignment and audit
    # ------------------------------------------------------------------
    aligned: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []
    anomalies: list[dict] = []
    mix_cols: list[str] | None = None
    loss_cols: list[str] | None = None
    for spec in PAIR_SPECS:
        frame, current_mix, current_loss, summary, pair_anomalies = align_pair(
            regmix_dir, *spec, mix_cols, loss_cols
        )
        mix_cols = current_mix if mix_cols is None else mix_cols
        loss_cols = current_loss if loss_cols is None else loss_cols
        aligned[spec[0]] = frame
        summaries.append(summary)
        anomalies.extend(pair_anomalies)
        print(
            f"[对齐] {spec[0]}: {len(frame)} 行；列="
            f"{len(current_mix)}个配比+{len(current_loss)}个Loss；index一一对应"
        )
    assert mix_cols is not None and loss_cols is not None

    train_recipe_groups = set(aligned["train_1m"]["recipe_group_id"])
    for summary in summaries:
        pair_groups = set(aligned[summary["source_pair"]]["recipe_group_id"])
        summary["overlap_recipe_groups_with_train"] = len(train_recipe_groups & pair_groups)
        summary["overlap_rate_with_train"] = len(train_recipe_groups & pair_groups) / max(len(pair_groups), 1)

    aligned_all = pd.concat(aligned.values(), ignore_index=True)
    aligned_all.to_csv(output_dir / "aligned_data.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summaries + anomalies)
    summary_df.to_csv(output_dir / "data_summary.csv", index=False, encoding="utf-8-sig")

    mix_names = [short_mix_name(c) for c in mix_cols]
    target_names = [short_loss_name(c) for c in loss_cols]
    d = len(mix_cols)
    interaction_pairs = list(itertools.combinations(range(d), 2))
    train = aligned["train_1m"]
    P_train = train[mix_cols].to_numpy(dtype=float)
    y_matrix = train[loss_cols].to_numpy(dtype=float)
    y_composite = y_matrix.mean(axis=1)
    recipe_groups = train["recipe_group_id"].to_numpy()
    outer_splits = grouped_splits(P_train, recipe_groups, max_splits=5)
    observed_best_pos = int(np.argmin(y_composite))
    observed_best = P_train[observed_best_pos]
    mean_recipe = P_train.mean(axis=0)
    pairwise_l1_train = np.abs(P_train[:, None, :] - P_train[None, :, :]).sum(axis=2)
    np.fill_diagonal(pairwise_l1_train, np.inf)
    nearest_neighbor_l1 = pairwise_l1_train.min(axis=1)
    support_nearest_l1_threshold = float(np.quantile(nearest_neighbor_l1, 0.95))
    run_config["support_nearest_training_L1_threshold"] = support_nearest_l1_threshold
    run_config["support_threshold_definition"] = "A4/A5每个配方到最近其他训练配方L1距离的95%分位数"
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[审计] A4/A5={len(train)}行，唯一配方组={pd.Series(recipe_groups).nunique()}，"
        f"17个配比域，13个Loss域"
    )

    design_by_model = {
        spec.model_id: build_design(P_train, interaction_pairs, spec.order)
        for spec in MODEL_SPECS
    }
    feature_names_by_model = {
        spec.model_id: (
            mix_names
            if spec.order == 1
            else mix_names + [f"{mix_names[i]}×{mix_names[j]}" for i, j in interaction_pairs]
        )
        for spec in MODEL_SPECS
    }

    # ------------------------------------------------------------------
    # 2. Nested CV: first-order baseline and quadratic candidates
    # ------------------------------------------------------------------
    fitted: dict[str, dict[str, tuple]] = {spec.model_id: {} for spec in MODEL_SPECS}
    oof_predictions: dict[str, dict[str, np.ndarray]] = {spec.model_id: {} for spec in MODEL_SPECS}
    cv_rows: list[dict] = []
    coefficient_rows: list[dict] = []

    for spec in MODEL_SPECS:
        X_train = design_by_model[spec.model_id]
        for col, target in zip(loss_cols, target_names):
            y = train[col].to_numpy(dtype=float)
            oof, rows, final = fit_nested_target(X_train, y, recipe_groups, spec, outer_splits, target)
            fitted[spec.model_id][target] = final
            oof_predictions[spec.model_id][target] = oof
            cv_rows.extend(rows)
            _, estimator, params, coef_scaled, coef_original = final
            feature_names = feature_names_by_model[spec.model_id]
            for feature_index, feature in enumerate(feature_names):
                if feature_index < d:
                    term_type = "linear"
                    component_i = mix_names[feature_index]
                    component_j = ""
                else:
                    term_type = "interaction"
                    i, j = interaction_pairs[feature_index - d]
                    component_i, component_j = mix_names[i], mix_names[j]
                coefficient_rows.append({
                    "model_id": spec.model_id,
                    "model_label": spec.label_cn,
                    "model_family": spec.family,
                    "mixture_order": spec.order,
                    "target_loss_domain": target,
                    "term": feature,
                    "term_type": term_type,
                    "component_i": component_i,
                    "component_j": component_j,
                    "coefficient_scaled_feature_space": float(coef_scaled[feature_index]),
                    "coefficient_original_mixture_scale": float(coef_original[feature_index]),
                    "alpha": params["alpha"],
                    "l1_ratio": params.get("l1_ratio", np.nan),
                    "full_training_inner_cv_rmse": params["full_train_inner_cv_rmse"],
                    "n_iter": params["n_iter"],
                    "converged": params["converged"],
                    "convergence_warning": params["convergence_warning"],
                    "fit_intercept": False,
                    "scaler_with_mean": False,
                })
            print(
                f"[拟合] {spec.model_id:24s} / {target:20s}: "
                f"alpha={params['alpha']:.2g}, l1={params.get('l1_ratio', np.nan)}"
            )

        pred_composite = np.column_stack(
            [oof_predictions[spec.model_id][target] for target in target_names]
        ).mean(axis=1)
        for fold, (_, va) in enumerate(outer_splits, start=1):
            cv_rows.append({
                "model_id": spec.model_id,
                "model_label": spec.label_cn,
                "model_family": spec.family,
                "mixture_order": spec.order,
                "target": "composite_equal13",
                "fold": fold,
                "n_train": len(train) - len(va),
                "n_validation": len(va),
                "alpha": np.nan,
                "l1_ratio": np.nan,
                "inner_cv_rmse": np.nan,
                **regression_metrics(y_composite[va], pred_composite[va]),
                "validation_scheme": "nested_GroupKFold_outer5_aggregated_from_13_targets",
            })
        cv_rows.append({
            "model_id": spec.model_id,
            "model_label": spec.label_cn,
            "model_family": spec.family,
            "mixture_order": spec.order,
            "target": "composite_equal13",
            "fold": "overall_oof",
            "n_train": len(train),
            "n_validation": len(train),
            "alpha": np.nan,
            "l1_ratio": np.nan,
            "inner_cv_rmse": np.nan,
            **regression_metrics(y_composite, pred_composite),
            "validation_scheme": "nested_GroupKFold_out_of_fold_aggregated",
        })

    cv_df = pd.DataFrame(cv_rows)
    coefficient_df = pd.DataFrame(coefficient_rows)
    cv_df.to_csv(output_dir / "cv_results.csv", index=False, encoding="utf-8-sig")
    coefficient_df.to_csv(output_dir / "model_coefficients.csv", index=False, encoding="utf-8-sig")

    composite_folds = cv_df[
        (cv_df["target"] == "composite_equal13")
        & cv_df["fold"].astype(str).str.fullmatch(r"\d+")
    ].copy()
    comparison_rows: list[dict] = []
    for spec in MODEL_SPECS:
        sub = composite_folds[composite_folds["model_id"] == spec.model_id]
        comparison_rows.append({
            "model_id": spec.model_id,
            "model_label": spec.label_cn,
            "mean_outer_rmse": float(sub["rmse"].mean()),
            "sd_outer_rmse": float(sub["rmse"].std(ddof=1)),
            "se_outer_rmse": float(sub["rmse"].std(ddof=1) / math.sqrt(len(sub))),
            "mean_outer_mae": float(sub["mae"].mean()),
            "mean_outer_r2": float(sub["r2"].mean()),
            "mean_outer_spearman": float(sub["spearman"].mean()),
            "mean_outer_regret": float(sub["best_candidate_regret"].mean()),
            "n_outer_folds": len(sub),
        })
    model_comparison = pd.DataFrame(comparison_rows)
    quadratic_only = model_comparison[
        model_comparison["model_id"].isin(["ridge_quadratic", "elasticnet_quadratic"])
    ]
    best_quad = quadratic_only.loc[quadratic_only["mean_outer_rmse"].idxmin()]
    best_quad_id = str(best_quad["model_id"])
    one_se_threshold = float(best_quad["mean_outer_rmse"] + best_quad["se_outer_rmse"])
    ridge_quad_row = model_comparison.set_index("model_id").loc["ridge_quadratic"]
    linear_row = model_comparison.set_index("model_id").loc["ridge_linear"]
    ridge_within_one_se = bool(ridge_quad_row["mean_outer_rmse"] <= one_se_threshold)
    pivot_rmse = composite_folds.pivot(index="fold", columns="model_id", values="rmse")
    quadratic_win_folds = int((pivot_rmse[best_quad_id] < pivot_rmse["ridge_linear"]).sum())
    interaction_supported = bool(
        float(best_quad["mean_outer_rmse"]) < linear_row["mean_outer_rmse"]
        and quadratic_win_folds >= 3
    )
    if interaction_supported:
        selected_model_id = "ridge_quadratic" if ridge_within_one_se else best_quad_id
    else:
        selected_model_id = "ridge_linear"
    provisional_selected_model_id = selected_model_id
    model_comparison["within_one_se_of_best_quadratic"] = (
        model_comparison["model_id"].isin(["ridge_quadratic", "elasticnet_quadratic"])
        & (model_comparison["mean_outer_rmse"] <= one_se_threshold)
    )
    model_comparison["best_quadratic_wins_vs_linear_folds"] = quadratic_win_folds
    model_comparison["best_quadratic_model"] = best_quad_id
    model_comparison["interaction_terms_supported"] = interaction_supported
    model_comparison["provisional_selected_model"] = model_comparison["model_id"].eq(selected_model_id)
    model_comparison["selected_predictive_model"] = False
    model_comparison["selection_rule"] = (
        "二阶先与一阶基线比较；若二阶获支持，再在二阶Ridge/ElasticNet间使用一标准误规则"
    )
    print(
        f"[选模-暂定] 最佳二阶{best_quad_id}相对一阶胜出{quadratic_win_folds}/5折；"
        f"交互项支持={interaction_supported}；暂定模型={selected_model_id}，待稳定性门禁"
    )

    # ------------------------------------------------------------------
    # 3. Frozen external validation and optional separated calibration
    # ------------------------------------------------------------------
    validation_rows: list[dict] = []
    prediction_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for pair_name, frame in aligned.items():
        if pair_name == "train_1m":
            continue
        P = frame[mix_cols].to_numpy(dtype=float)
        actual_matrix = frame[loss_cols].to_numpy(dtype=float)
        for spec in MODEL_SPECS:
            X = build_design(P, interaction_pairs, spec.order)
            pred_matrix = np.column_stack([
                predict_fitted(fitted[spec.model_id][target], X) for target in target_names
            ])
            prediction_cache[(pair_name, spec.model_id)] = (actual_matrix, pred_matrix)
            for k, target in enumerate(target_names):
                validation_rows.append({
                    "dataset": pair_name,
                    "scale": frame["scale"].iloc[0],
                    "evidence_role": frame["evidence_role"].iloc[0],
                    "model_id": spec.model_id,
                    "model_label": spec.label_cn,
                    "target": target,
                    "n": len(frame),
                    "evaluation_mode": "frozen_A4_A5_zero_refit",
                    **regression_metrics(actual_matrix[:, k], pred_matrix[:, k]),
                })
            validation_rows.append({
                "dataset": pair_name,
                "scale": frame["scale"].iloc[0],
                "evidence_role": frame["evidence_role"].iloc[0],
                "model_id": spec.model_id,
                "model_label": spec.label_cn,
                "target": "composite_equal13",
                "n": len(frame),
                "evaluation_mode": "frozen_A4_A5_zero_refit",
                **regression_metrics(actual_matrix.mean(axis=1), pred_matrix.mean(axis=1)),
            })

        # Secondary affine scale calibration: fixed 25% calibration and disjoint 75% holdout.
        if pair_name in {"transfer_60m", "transfer_1B"}:
            rng_cal = np.random.default_rng(SEED + len(frame) + (60 if pair_name == "transfer_60m" else 1000))
            order_idx = rng_cal.permutation(len(frame))
            n_cal = max(10, int(round(0.25 * len(frame))))
            cal_idx, eval_idx = order_idx[:n_cal], order_idx[n_cal:]
            target_pairs = list(enumerate(target_names)) + [(None, "composite_equal13")]
            for calibration_spec in MODEL_SPECS:
                actual_matrix, pred_matrix = prediction_cache[(pair_name, calibration_spec.model_id)]
                for k, target in target_pairs:
                    if k is None:
                        y = actual_matrix.mean(axis=1)
                        yhat = pred_matrix.mean(axis=1)
                    else:
                        y = actual_matrix[:, k]
                        yhat = pred_matrix[:, k]
                    slope, intercept = np.polyfit(yhat[cal_idx], y[cal_idx], 1)
                    calibrated = intercept + slope * yhat[eval_idx]
                    validation_rows.append({
                        "dataset": pair_name,
                        "scale": frame["scale"].iloc[0],
                        "evidence_role": "calibrated_transfer_secondary",
                        "model_id": calibration_spec.model_id,
                        "model_label": calibration_spec.label_cn,
                        "target": target,
                        "n": len(eval_idx),
                        "n_calibration": len(cal_idx),
                        "evaluation_mode": "affine_calibration_25pct_then_disjoint_75pct_holdout",
                        "calibration_intercept": float(intercept),
                        "calibration_slope": float(slope),
                        **regression_metrics(y[eval_idx], calibrated),
                    })
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 4. Wide simplex and local-convex-hull supported optimization
    # ------------------------------------------------------------------
    coef_matrix_by_model = {
        spec.model_id: np.vstack([fitted[spec.model_id][target][4] for target in target_names])
        for spec in MODEL_SPECS
    }
    composite_coef_by_model = {
        model_id: matrix.mean(axis=0) for model_id, matrix in coef_matrix_by_model.items()
    }
    rng = np.random.default_rng(SEED)
    best_positions = np.argsort(y_composite)[: min(50, len(train))]
    wide_starts = [mean_recipe, observed_best, np.ones(d) / d]
    wide_starts.extend([P_train[pos] for pos in best_positions])
    wide_starts.extend(rng.dirichlet(np.ones(d), size=60))

    wide_rows: list[dict] = []
    support_rows: list[dict] = []
    wide_optima: dict[str, dict] = {}
    support_best: dict[str, dict[int, dict]] = {}
    for spec in MODEL_SPECS:
        coef = composite_coef_by_model[spec.model_id]
        p_wide, loss_wide, success_wide = optimize_wide(
            coef, spec.order, wide_starts, d, interaction_pairs
        )
        wide_record = {
            "model_id": spec.model_id,
            "model_label": spec.label_cn,
            "predicted_composite_loss": loss_wide,
            "nearest_training_L1": nearest_l1(P_train, p_wide),
            "successful_multistarts": success_wide,
            "mixture_sum": float(p_wide.sum()),
            "minimum_share": float(p_wide.min()),
            "diagnostic_only": True,
            "recommendation": "宽单纯形外推诊断，不能直接推荐",
            **{name: float(value) for name, value in zip(mix_names, p_wide)},
        }
        wide_rows.append(wide_record)
        wide_optima[spec.model_id] = {**wide_record, "p": p_wide}

        records, best_by_k = local_support_search(
            P_train,
            y_composite,
            coef,
            spec.order,
            d,
            interaction_pairs,
            SUPPORT_K_VALUES,
            SEED + 17 * (MODEL_SPECS.index(spec) + 1),
        )
        support_best[spec.model_id] = best_by_k
        for k, best in best_by_k.items():
            support_rows.append({
                "model_id": spec.model_id,
                "model_label": spec.label_cn,
                "K": k,
                "is_main_K": k == SUPPORT_MAIN_K,
                "support_definition": "top10pct observed centers; K-nearest L1 local convex hull",
                "center_row_position": best["center_row_position"],
                "center_index": train.iloc[best["center_row_position"]]["index"],
                "center_rank_by_actual_loss": best["center_rank_by_actual_loss"],
                "center_actual_composite_loss": best["center_actual_composite_loss"],
                "predicted_composite_loss": best["predicted_composite_loss"],
                "nearest_training_L1": best["nearest_training_L1"],
                "successful_multistarts": best["successful_multistarts"],
                "neighborhood_rows": best["neighborhood_rows"],
                "neighborhood_indices": "|".join(
                    train.iloc[best["neighbors"]]["index"].astype(str).tolist()
                ),
                "convex_weights": "|".join(f"{value:.12g}" for value in best["weights"]),
                "nonzero_convex_weights": int(np.sum(best["weights"] > 1e-8)),
                "mixture_sum": float(best["p"].sum()),
                "minimum_share": float(best["p"].min()),
                **{name: float(value) for name, value in zip(mix_names, best["p"])},
            })
        print(
            f"[优化] {spec.model_id}: 宽单纯形Loss={loss_wide:.6f}, "
            f"局部凸包K30 Loss={best_by_k[SUPPORT_MAIN_K]['predicted_composite_loss']:.6f}"
        )

    wide_df = pd.DataFrame(wide_rows)
    support_df = pd.DataFrame(support_rows)
    observed_row = {
        "source": "A4_A5_observed_best",
        "index": train.iloc[observed_best_pos]["index"],
        "actual_composite_loss": float(y_composite[observed_best_pos]),
        "mixture_sum": float(observed_best.sum()),
        **{name: float(value) for name, value in zip(mix_names, observed_best)},
    }
    for spec in MODEL_SPECS:
        observed_row[f"predicted_loss_{spec.model_id}"] = surface_value(
            observed_best,
            composite_coef_by_model[spec.model_id],
            d,
            interaction_pairs,
            spec.order,
        )
    observed_df = pd.DataFrame([observed_row])
    wide_df.to_csv(output_dir / "optimal_mixture_wide.csv", index=False, encoding="utf-8-sig")
    observed_df.to_csv(output_dir / "optimal_mixture_observed.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 5. Repeated grouped-fold optimum stability
    # ------------------------------------------------------------------
    stability_raw_rows: list[dict] = []
    stability_vector_rows: list[dict] = []
    stability_specs = [SPEC_BY_ID["ridge_quadratic"], SPEC_BY_ID["elasticnet_quadratic"]]
    if provisional_selected_model_id not in {spec.model_id for spec in stability_specs}:
        stability_specs.append(SPEC_BY_ID[provisional_selected_model_id])
    for repeat, fold, tr, va in repeated_group_folds(
        recipe_groups, STABILITY_FOLDS, STABILITY_REPEATS, SEED
    ):
        fold_optima: dict[str, np.ndarray] = {}
        fold_mean = P_train[tr].mean(axis=0)
        for spec in stability_specs:
            fold_coefs = []
            fold_converged = []
            X_spec = design_by_model[spec.model_id]
            for k, target in enumerate(target_names):
                frozen_params = fitted[spec.model_id][target][2]
                coef_original, converged, _ = refit_with_frozen_params(
                    X_spec[tr], y_matrix[tr, k], spec.family, frozen_params
                )
                fold_coefs.append(coef_original)
                fold_converged.append(converged)
            fold_composite_coef = np.vstack(fold_coefs).mean(axis=0)
            _, best = local_support_search(
                P_train[tr],
                y_composite[tr],
                fold_composite_coef,
                spec.order,
                d,
                interaction_pairs,
                (SUPPORT_MAIN_K,),
                SEED + repeat * 10000 + fold * 100 + MODEL_SPECS.index(spec) + 1,
            )
            p_opt = best[SUPPORT_MAIN_K]["p"]
            fold_optima[spec.model_id] = p_opt
            for domain, share, baseline in zip(mix_names, p_opt, fold_mean):
                stability_raw_rows.append({
                    "record_type": "fold_optimum",
                    "repeat": repeat,
                    "fold": fold,
                    "model_id": spec.model_id,
                    "mixture_domain": domain,
                    "optimal_share": float(share),
                    "fold_training_mean_share": float(baseline),
                    "direction_vs_fold_mean": int(np.sign(share - baseline)),
                    "n_training": len(tr),
                    "n_validation": len(va),
                    "predicted_composite_loss": best[SUPPORT_MAIN_K]["predicted_composite_loss"],
                    "nearest_fold_training_L1": best[SUPPORT_MAIN_K]["nearest_training_L1"],
                    "all_13_targets_converged": bool(all(fold_converged)),
                })
        ridge_p = fold_optima["ridge_quadratic"]
        enet_p = fold_optima["elasticnet_quadratic"]
        stability_vector_rows.append({
            "record_type": "model_vector_comparison",
            "repeat": repeat,
            "fold": fold,
            "ridge_elasticnet_spearman": safe_spearman(ridge_p, enet_p),
            "ridge_elasticnet_L1": float(np.abs(ridge_p - enet_p).sum()),
        })
        print(f"[稳定性] 重复{repeat}/{STABILITY_REPEATS}，折{fold}/{STABILITY_FOLDS}完成")

    stability_raw = pd.DataFrame(stability_raw_rows)
    stability_summary_rows: list[dict] = []
    for domain in mix_names:
        domain_all = stability_raw[stability_raw["mixture_domain"] == domain]
        pivot = domain_all.pivot_table(
            index=["repeat", "fold"], columns="model_id", values="direction_vs_fold_mean", aggfunc="first"
        )
        agreement = float(
            (pivot["ridge_quadratic"] == pivot["elasticnet_quadratic"]).mean()
        )
        for spec in stability_specs:
            values = domain_all.loc[domain_all["model_id"] == spec.model_id, "optimal_share"].to_numpy(float)
            stability_summary_rows.append({
                "record_type": "domain_summary",
                "model_id": spec.model_id,
                "mixture_domain": domain,
                "mean_share": float(np.mean(values)),
                "median_share": float(np.median(values)),
                "p10_share": float(np.quantile(values, 0.10)),
                "p90_share": float(np.quantile(values, 0.90)),
                "zero_frequency": float(np.mean(values <= 1e-8)),
                "n_resampled_optima": len(values),
                "ridge_elasticnet_direction_agreement": agreement,
                "interval_definition": "15次组级重复划分的10%--90%稳定性区间，不是置信区间",
            })
    stability_summary = pd.DataFrame(stability_summary_rows)
    stability_vector = pd.DataFrame(stability_vector_rows)

    stability_model_rows: list[dict] = []
    for spec in stability_specs:
        sub = stability_raw[stability_raw["model_id"] == spec.model_id]
        matrix = (
            sub.pivot_table(
                index=["repeat", "fold"], columns="mixture_domain", values="optimal_share", aggfunc="first"
            )
            .loc[:, mix_names]
            .to_numpy(dtype=float)
        )
        median_vector = np.median(matrix, axis=0)
        full_vector = support_best[spec.model_id][SUPPORT_MAIN_K]["p"]
        l1_to_median = np.abs(matrix - median_vector).sum(axis=1)
        l1_to_full = np.abs(matrix - full_vector).sum(axis=1)
        pair_rhos = [
            safe_spearman(matrix[i], matrix[j])
            for i, j in itertools.combinations(range(len(matrix)), 2)
        ]
        stability_model_rows.append({
            "record_type": "model_summary",
            "model_id": spec.model_id,
            "mixture_domain": "ALL_17",
            "mean_L1_to_fold_median": float(np.mean(l1_to_median)),
            "mean_L1_to_full_solution": float(np.mean(l1_to_full)),
            "se_L1_to_full_solution": float(
                np.std(l1_to_full, ddof=1) / math.sqrt(len(l1_to_full))
            ),
            "median_L1_to_full_solution": float(np.median(l1_to_full)),
            "p90_L1_to_full_solution": float(np.quantile(l1_to_full, 0.90)),
            "mean_pairwise_fold_spearman": float(np.nanmean(pair_rhos)),
            "all_fold_targets_converged": bool(sub["all_13_targets_converged"].all()),
            "n_resampled_optima": int(len(matrix)),
            "interval_definition": "最优配方折间稳定性；L1范围0--2，越低越稳定",
        })
    stability_model = pd.DataFrame(stability_model_rows)
    stability_output = pd.concat(
        [stability_raw, stability_summary, stability_vector, stability_model],
        ignore_index=True,
        sort=False,
    )
    stability_output.to_csv(
        output_dir / "optimal_mixture_stability.csv", index=False, encoding="utf-8-sig"
    )

    vector_rho_mean = float(stability_vector["ridge_elasticnet_spearman"].mean())
    vector_l1_mean = float(stability_vector["ridge_elasticnet_L1"].mean())

    # Final selection: interaction support first; inside the predictive one-SE set,
    # use optimum stability, with Ridge preferred when it is also within one SE.
    stability_lookup = stability_model.set_index("model_id")
    if interaction_supported:
        admissible = quadratic_only[
            quadratic_only["mean_outer_rmse"] <= one_se_threshold
        ]["model_id"].tolist()
        eligible_stability = stability_model[
            stability_model["model_id"].isin(admissible)
        ]
        best_stability = eligible_stability.loc[
            eligible_stability["mean_L1_to_full_solution"].idxmin()
        ]
        stability_one_se_threshold = float(
            best_stability["mean_L1_to_full_solution"]
            + best_stability["se_L1_to_full_solution"]
        )
        ridge_stability_admissible = bool(
            "ridge_quadratic" in admissible
            and float(stability_lookup.loc["ridge_quadratic", "mean_L1_to_full_solution"])
            <= stability_one_se_threshold
        )
        selected_model_id = (
            "ridge_quadratic" if ridge_stability_admissible else str(best_stability["model_id"])
        )
    else:
        ridge_stability_admissible = False
        stability_one_se_threshold = np.nan
        selected_model_id = "ridge_linear"

    effect_model_id = selected_model_id
    selected_stability = stability_lookup.loc[selected_model_id]
    selected_stability_l1 = float(selected_stability["median_L1_to_full_solution"])
    selected_stability_p90_l1 = float(selected_stability["p90_L1_to_full_solution"])
    model_comparison["median_L1_to_full_solution"] = model_comparison["model_id"].map(
        stability_lookup["median_L1_to_full_solution"]
    )
    model_comparison["p90_L1_to_full_solution"] = model_comparison["model_id"].map(
        stability_lookup["p90_L1_to_full_solution"]
    )
    model_comparison["mean_pairwise_fold_spearman"] = model_comparison["model_id"].map(
        stability_lookup["mean_pairwise_fold_spearman"]
    )
    model_comparison["selected_predictive_model"] = model_comparison["model_id"].eq(selected_model_id)
    model_comparison["effect_interpretation_model"] = model_comparison["model_id"].eq(effect_model_id)
    model_comparison["selection_rule"] = (
        "二阶须优于一阶且至少胜3/5折；二阶一个SE候选内优先Ridge，"
        "但其最优配方稳定性也须处于最稳候选的一个SE范围"
    )
    model_comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(
        f"[选模-最终] {selected_model_id}；效应解释模型={effect_model_id}；"
        f"所选模型折间中位L1={selected_stability_l1:.6f}"
    )

    selected_support_by_k = support_best[selected_model_id]
    support_k_l1 = max(
        float(np.abs(selected_support_by_k[SUPPORT_MAIN_K]["p"] - selected_support_by_k[k]["p"]).sum())
        for k in SUPPORT_K_VALUES if k != SUPPORT_MAIN_K
    )
    selected_cv_oof = cv_df[
        (cv_df["model_id"] == selected_model_id)
        & (cv_df["target"] == "composite_equal13")
        & (cv_df["fold"].astype(str) == "overall_oof")
    ].iloc[0]
    predicted_observed_baseline = surface_value(
        observed_best,
        composite_coef_by_model[selected_model_id],
        d,
        interaction_pairs,
        SPEC_BY_ID[selected_model_id].order,
    )
    predicted_improvement = float(
        predicted_observed_baseline
        - selected_support_by_k[SUPPORT_MAIN_K]["predicted_composite_loss"]
    )
    selected_support_nearest_l1 = float(
        selected_support_by_k[SUPPORT_MAIN_K]["nearest_training_L1"]
    )
    blind_test_spearman = float(
        validation_df[
            (validation_df["dataset"] == "blind_1m")
            & (validation_df["model_id"] == selected_model_id)
            & (validation_df["target"] == "composite_equal13")
            & (validation_df["evaluation_mode"] == "frozen_A4_A5_zero_refit")
        ]["spearman"].iloc[0]
    )
    selected_converged = bool(
        coefficient_df.loc[coefficient_df["model_id"] == selected_model_id, "converged"].all()
        and bool(selected_stability["all_fold_targets_converged"])
    )
    stability_p90_limit = 0.50
    recommendation_pass = bool(
        predicted_improvement > float(selected_cv_oof["rmse"])
        and selected_support_nearest_l1 <= support_nearest_l1_threshold
        and support_k_l1 <= 0.50
        and selected_stability_p90_l1 <= stability_p90_limit
        and blind_test_spearman >= 0.50
        and selected_converged
    )
    support_df["selected_model"] = support_df["model_id"].eq(selected_model_id)
    support_df["recommended_main_solution"] = (
        support_df["selected_model"] & support_df["K"].eq(SUPPORT_MAIN_K) & recommendation_pass
    )
    support_df["recommendation_status"] = np.where(
        support_df["selected_model"] & support_df["K"].eq(SUPPORT_MAIN_K),
        "可执行推荐候选，仍需邻域真实实验" if recommendation_pass else "模型候选，仅建议给出区间并补充邻域实验",
        "敏感性或对照结果",
    )
    support_df["recommendation_checks"] = (
        f"model_predicted_improvement={predicted_improvement:.6f}; "
        f"outer_oof_rmse={float(selected_cv_oof['rmse']):.6f}; "
        f"nearest_training_L1={selected_support_nearest_l1:.6f}; "
        f"data_driven_L1_threshold={support_nearest_l1_threshold:.6f}; "
        f"selected_fold_p90_L1={selected_stability_p90_l1:.6f}; "
        f"stability_p90_limit={stability_p90_limit:.6f}; "
        f"blind_1m_spearman={blind_test_spearman:.6f}; "
        f"ridge_enet_mean_rho={vector_rho_mean:.6f}; "
        f"K_sensitivity_max_L1={support_k_l1:.6f}; converged={selected_converged}"
    )
    support_df.to_csv(output_dir / "optimal_mixture_support.csv", index=False, encoding="utf-8-sig")

    primary_support = selected_support_by_k[SUPPORT_MAIN_K]["p"]
    effect_spec = SPEC_BY_ID[effect_model_id]
    directional_df, complementarity_df = build_effect_tables(
        effect_model_id,
        mix_names,
        target_names,
        coef_matrix_by_model[effect_model_id],
        composite_coef_by_model[effect_model_id],
        effect_spec.order,
        {"training_mean": mean_recipe, "observed_best": observed_best},
        interaction_pairs,
    )
    directional_df.to_csv(
        output_dir / "directional_derivatives.csv", index=False, encoding="utf-8-sig"
    )
    complementarity_df.to_csv(
        output_dir / "pairwise_complementarity.csv", index=False, encoding="utf-8-sig"
    )

    selected_stability_domains = (
        stability_summary[stability_summary["model_id"] == selected_model_id]
        .set_index("mixture_domain")
        .loc[mix_names]
    )
    optimal_mixture_df = pd.DataFrame({
        "mixture_domain": mix_names,
        "recommended_share": primary_support,
        "stability_p10_share": selected_stability_domains["p10_share"].to_numpy(dtype=float),
        "stability_p90_share": selected_stability_domains["p90_share"].to_numpy(dtype=float),
        "training_mean_share": mean_recipe,
        "observed_best_share": observed_best,
        "selected_model_id": selected_model_id,
        "recommendation_pass": recommendation_pass,
        "interval_definition": "15次组级重复划分下最优配方的10%--90%范围，不是置信区间",
    })
    optimal_mixture_df.to_csv(
        output_dir / "optimal_mixture.csv", index=False, encoding="utf-8-sig"
    )

    # ------------------------------------------------------------------
    # 7. Extrapolation tables: descriptive evidence only
    # ------------------------------------------------------------------
    extrapolation_rows: list[dict] = []
    for pair_name in ["estimated_10b", "estimated_70b"]:
        frame = aligned[pair_name]
        P = frame[mix_cols].to_numpy(dtype=float)
        top_n = max(1, math.ceil(0.10 * len(frame)))
        for spec in MODEL_SPECS:
            actual, pred = prediction_cache[(pair_name, spec.model_id)]
            y = actual.mean(axis=1)
            yhat = pred.mean(axis=1)
            pred_order = np.argsort(yhat)
            est_order = np.argsort(y)
            pred_top = set(pred_order[:top_n].tolist())
            est_top = set(est_order[:top_n].tolist())
            interval_scores = []
            for j in range(d):
                a = P[list(pred_top), j]
                b = P[list(est_top), j]
                interval_scores.append(
                    interval_overlap(float(a.min()), float(a.max()), float(b.min()), float(b.max()))
                )
            extrapolation_rows.append({
                "dataset": pair_name,
                "scale": frame["scale"].iloc[0],
                "model_id": spec.model_id,
                "model_label": spec.label_cn,
                "n_candidate_recipes": len(frame),
                **regression_metrics(y, yhat),
                "predicted_best_index": frame.iloc[int(pred_order[0])]["index"],
                "estimated_best_index": frame.iloc[int(est_order[0])]["index"],
                "best_recipe_mixture_L1_distance": float(
                    np.abs(P[pred_order[0]] - P[est_order[0]]).sum()
                ),
                "top10pct_count": top_n,
                "mean_component_interval_overlap_top10pct": float(np.mean(interval_scores)),
                "evidence_boundary": "Loss标签本身为模型估算，仅作排序与响应趋势描述，不能证明真实大模型有效",
            })
    extrapolation_df = pd.DataFrame(extrapolation_rows)
    extrapolation_df["scale_specific_conclusion"] = np.select(
        [
            extrapolation_df["spearman"] < 0,
            extrapolation_df["spearman"] < 0.50,
        ],
        [
            "排序反向：不支持从1M无条件迁移该尺度配比效应",
            "排序一致性偏弱：仅可作候选区间参考",
        ],
        default="排序一致性尚可，但估计Loss不能替代真实大模型验证",
    )
    extrapolation_df.to_csv(
        output_dir / "extrapolation_comparison.csv", index=False, encoding="utf-8-sig"
    )

    # ------------------------------------------------------------------
    # 8. A16 quality mapping: description and uncertainty, no extra coefficient
    # ------------------------------------------------------------------
    mapping = pd.read_csv(input_root / "domain_mapping_guide.csv", encoding="utf-8-sig")
    quality_path = cwd / "step6_final_Q" / "A1_domain_Q.csv"
    if not quality_path.exists():
        raise FileNotFoundError(f"缺少既有域级质量结果: {quality_path}")
    quality = pd.read_csv(quality_path)
    q_lookup = quality.set_index("_source_domain")
    total_n = float(quality["n"].sum())
    overall_mean = float(np.average(quality["mean_Q"], weights=quality["n"]))
    pooled_ss = float(
        (((quality["n"] - 1) * quality["std_Q"] ** 2)
         + quality["n"] * (quality["mean_Q"] - overall_mean) ** 2).sum()
    )
    overall_std = math.sqrt(pooled_ss / max(total_n - 1, 1))
    mapped_rows: list[dict] = []
    for row in mapping.itertuples(index=False):
        mixture_domain = str(row.mixture_domain)
        quality_domain = str(row.quality_domain)
        mapping_type = str(row.mapping_type)
        if mapping_type in {"direct", "near_direct"} and quality_domain in q_lookup.index:
            qrow = q_lookup.loc[quality_domain]
            point = float(qrow["mean_Q"])
            source_std = float(qrow["std_Q"])
            source_n = int(qrow["n"])
            floor = 0.02 if mapping_type == "direct" else 0.05
            margin = max(1.96 * source_std / math.sqrt(source_n), floor)
            source = quality_domain
            policy = f"{mapping_type}: max(1.96SE,{floor:.2f})"
            confidence = "高" if mapping_type == "direct" else "中"
        else:
            point = overall_mean
            source_std = overall_std
            source_n = int(total_n)
            margin = max(1.96 * overall_std / math.sqrt(total_n), 0.10)
            source = "A1_all_7_quality_domains_weighted"
            policy = "inferred: A1七域样本量加权均值±至少0.10"
            confidence = "低"
        mapped_rows.append({
            "mixture_domain": mixture_domain,
            "quality_domain_source": source,
            "mapping_type": mapping_type,
            "mapping_confidence": confidence,
            "mapped_quality_point": point,
            "mapped_quality_lower": max(0.0, point - margin),
            "mapped_quality_upper": min(1.0, point + margin),
            "mapping_uncertainty_margin": margin,
            "source_Q_std": source_std,
            "source_n": source_n,
            "interval_policy": policy,
            "quality_use": "描述、机制解释、区间及后续质量约束；不估计独立质量系数",
        })
    mapped_quality_df = pd.DataFrame(mapped_rows)
    share_lookup = dict(zip(mix_names, primary_support))
    mapped_quality_df["selected_support_mixture_share"] = mapped_quality_df["mixture_domain"].map(share_lookup)
    mapped_quality_df["selected_support_quality_contribution"] = (
        mapped_quality_df["mapped_quality_point"]
        * mapped_quality_df["selected_support_mixture_share"]
    )
    mapped_quality_df["training_mean_share"] = mapped_quality_df["mixture_domain"].map(
        dict(zip(mix_names, mean_recipe))
    )
    mapped_quality_df.to_csv(output_dir / "mapped_quality.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(8.8, 5.3), layout="constrained")
    model_order = [spec.model_id for spec in MODEL_SPECS]
    model_labels = [spec.label_cn for spec in MODEL_SPECS]
    comp_plot = model_comparison.set_index("model_id").loc[model_order]
    x = np.arange(len(model_order))
    ax.errorbar(
        x,
        comp_plot["mean_outer_rmse"],
        yerr=comp_plot["se_outer_rmse"],
        fmt="o",
        markersize=8,
        capsize=5,
        color=palette[0],
        ecolor="#4C4C4C",
    )
    ax.set_xticks(x, model_labels)
    ax.set_ylabel("外层留出配方 RMSE（均值±SE）")
    ax.set_title("一阶与二阶混料响应模型的嵌套交叉验证")
    ax.grid(axis="x", visible=False)
    save_figure(fig, figures_dir / "cv_model_comparison.png")

    heat_source = complementarity_df[
        (complementarity_df["reference_point"] == "training_mean")
        & np.isclose(complementarity_df["requested_delta"], 0.01)
    ]
    for target in target_names + ["composite_equal13"]:
        sub = heat_source[heat_source["target_loss_domain"] == target]
        matrix = np.zeros((d, d), dtype=float)
        for row in sub.itertuples(index=False):
            i = mix_names.index(row.domain_i)
            j = mix_names.index(row.domain_j)
            matrix[i, j] = row.normalized_by_delta_squared
            matrix[j, i] = row.normalized_by_delta_squared
        np.fill_diagonal(matrix, np.nan)
        finite = np.abs(matrix[np.isfinite(matrix)])
        vmax = float(np.quantile(finite, 0.98)) if len(finite) else 1.0
        if vmax <= 0:
            vmax = 1.0
        fig, ax = plt.subplots(figsize=(11.5, 9.2), layout="constrained")
        sns.heatmap(
            matrix,
            xticklabels=mix_names,
            yticklabels=mix_names,
            cmap="vlag",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            mask=~np.isfinite(matrix),
            cbar_kws={"label": r"局部组合效应 $S_{ij}/\delta^2$"},
            ax=ax,
        )
        ax.set_xlabel("训练域 j")
        ax.set_ylabel("训练域 i")
        ax.set_title(f"{target}：训练均值配比处的局部组合效应")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        save_figure(fig, figures_dir / f"pairwise_complementarity_{safe_file_stem(target)}.png")

    selected_spec = SPEC_BY_ID[selected_model_id]
    selected_wide = wide_optima[selected_model_id]["p"]
    selected_support = support_best[selected_model_id][SUPPORT_MAIN_K]["p"]
    fig, ax = plt.subplots(figsize=(15.5, 7.4), layout="constrained")
    x = np.arange(d)
    width = 0.19
    comparison_series = [
        ("训练集观测最优", observed_best),
        (f"{selected_spec.label_cn}宽单纯形数学最优", selected_wide),
        (f"{selected_spec.label_cn}局部支持最优", selected_support),
        ("训练集平均配比", mean_recipe),
    ]
    for pos, (label, values) in enumerate(comparison_series):
        ax.bar(x + (pos - 1.5) * width, values, width=width, label=label, color=palette[pos])
    ax.set_xticks(x)
    ax.set_xticklabels(mix_names, rotation=45, ha="right")
    ax.set_ylabel("训练域配比")
    ax.set_xlabel("训练域")
    ax.set_title("数学最优、局部支持最优与观测安全基线")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, figures_dir / "optimal_mixture_comparison.png")

    selected_stability_plot = (
        stability_summary[stability_summary["model_id"] == selected_model_id]
        .set_index("mixture_domain")
        .loc[mix_names]
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14.8, 7.0), layout="constrained")
    x = np.arange(d)
    y = selected_stability_plot["median_share"].to_numpy()
    lower = y - selected_stability_plot["p10_share"].to_numpy()
    upper = selected_stability_plot["p90_share"].to_numpy() - y
    ax.errorbar(
        x, y, yerr=np.vstack([lower, upper]), fmt="o", capsize=4,
        label=f"{selected_spec.label_cn}中位数及10%–90%区间",
    )
    for pos, comparison_id in enumerate(
        mid for mid in ["ridge_quadratic", "elasticnet_quadratic"] if mid != selected_model_id
    ):
        comparison_stability = (
            stability_summary[stability_summary["model_id"] == comparison_id]
            .set_index("mixture_domain")
            .loc[mix_names]
        )
        ax.scatter(
            x,
            comparison_stability["median_share"],
            marker="s" if pos == 0 else "^",
            label=f"{SPEC_BY_ID[comparison_id].label_cn}中位数",
            color=palette[pos + 1],
            alpha=0.8,
        )
    ax.scatter(x, mean_recipe, marker="x", label="训练平均配比", color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(mix_names, rotation=45, ha="right")
    ax.set_ylabel("局部支持最优配比")
    ax.set_xlabel("训练域")
    ax.set_title("重复分组划分下的最优配方稳定性")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "optimal_mixture_stability.png")

    main_validation = validation_df[
        (validation_df["model_id"] == selected_model_id)
        & (validation_df["target"] == "composite_equal13")
        & (validation_df["evaluation_mode"] == "frozen_A4_A5_zero_refit")
        & validation_df["dataset"].isin(["blind_1m", "transfer_60m", "transfer_1B"])
    ].copy()
    dataset_order = ["blind_1m", "transfer_60m", "transfer_1B"]
    labels = {"blind_1m": "1M盲测", "transfer_60m": "60M零重拟合", "transfer_1B": "1B零重拟合"}
    main_validation["dataset"] = pd.Categorical(main_validation["dataset"], dataset_order, ordered=True)
    main_validation = main_validation.sort_values("dataset")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), layout="constrained")
    axes[0].bar(range(3), main_validation["rmse"], color=palette[0])
    axes[0].set_xticks(range(3), [labels[x] for x in dataset_order])
    axes[0].set_ylabel("综合Loss RMSE")
    axes[0].set_title("绝对Loss水平迁移")
    axes[1].bar(range(3), main_validation["spearman"], color=palette[1])
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(range(3), [labels[x] for x in dataset_order])
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_title("配方排序迁移")
    fig.suptitle(f"冻结{selected_spec.label_cn}的跨尺度检验")
    save_figure(fig, figures_dir / "scale_transfer_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), layout="constrained")
    for ax, pair_name in zip(axes, ["estimated_10b", "estimated_70b"]):
        actual, pred = prediction_cache[(pair_name, selected_model_id)]
        y = actual.mean(axis=1)
        yhat = pred.mean(axis=1)
        ax.scatter(y, yhat, s=34, alpha=0.75, color=palette[0], edgecolor="white", linewidth=0.4)
        rho = safe_spearman(y, yhat)
        ax.set_xlabel("外推表估计综合Loss")
        ax.set_ylabel("冻结1M模型预测综合Loss")
        ax.set_title(f"{aligned[pair_name]['scale'].iloc[0]}：Spearman ρ={rho:.3f}")
    fig.suptitle("外推估计表与冻结1M响应面的排序对照")
    save_figure(fig, figures_dir / "extrapolation_ranking_comparison.png")

    # ------------------------------------------------------------------
    # 10. Computed report
    # ------------------------------------------------------------------
    pair_table = pd.DataFrame(summaries)[[
        "source_pair", "scale", "evidence_role", "rows_aligned", "index_sets_equal",
        "n_mixture_columns", "n_loss_columns", "mixture_sum_min_before",
        "mixture_sum_max_before", "severe_deviation_rows", "unique_recipe_groups",
        "overlap_recipe_groups_with_train", "overlap_rate_with_train",
    ]]
    cv_report = model_comparison[[
        "model_label", "mean_outer_rmse", "se_outer_rmse", "mean_outer_mae",
        "mean_outer_r2", "mean_outer_spearman", "mean_outer_regret",
        "median_L1_to_full_solution", "p90_L1_to_full_solution",
        "interaction_terms_supported", "selected_predictive_model",
    ]]
    validation_report = validation_df[
        (validation_df["model_id"] == selected_model_id)
        & (validation_df["target"] == "composite_equal13")
        & (validation_df["evaluation_mode"] == "frozen_A4_A5_zero_refit")
        & validation_df["dataset"].isin(["blind_1m", "transfer_60m", "transfer_1B"])
    ][["dataset", "scale", "n", "rmse", "mae", "r2", "spearman", "best_candidate_regret", "top10pct_overlap_rate"]]
    extrap_report = extrapolation_df[
        (extrapolation_df["model_id"] == selected_model_id)
    ][["dataset", "scale", "rmse", "mae", "spearman", "best_candidate_regret", "top10pct_overlap_rate", "mean_component_interval_overlap_top10pct", "scale_specific_conclusion"]]
    support_report = support_df[
        support_df["model_id"] == selected_model_id
    ][["K", "center_index", "predicted_composite_loss", "nearest_training_L1", "recommendation_status"]]
    optimal_report = pd.DataFrame([
        {
            "层次": "宽单纯形数学最优",
            "Loss": wide_optima[selected_model_id]["predicted_composite_loss"],
            "最近训练配方L1": wide_optima[selected_model_id]["nearest_training_L1"],
            "用途": "外推风险诊断，不能推荐",
        },
        {
            "层次": "局部凸包支持最优(K=30)",
            "Loss": selected_support_by_k[SUPPORT_MAIN_K]["predicted_composite_loss"],
            "最近训练配方L1": selected_support_by_k[SUPPORT_MAIN_K]["nearest_training_L1"],
            "用途": "推荐候选" if recommendation_pass else "候选区间，需补实验",
        },
        {
            "层次": "训练集观测最优",
            "Loss": y_composite[observed_best_pos],
            "最近训练配方L1": 0.0,
            "用途": "安全基线",
        },
    ])
    comp_direction = directional_df[
        (directional_df["reference_point"] == "training_mean")
        & (directional_df["target_loss_domain"] == "composite_equal13")
        & directional_df["one_percent_transfer_feasible"]
    ]
    best_moves = comp_direction.nsmallest(5, "directional_derivative_loss_per_unit_transfer")[[
        "from_domain", "to_domain", "predicted_loss_change_for_1pct_transfer"
    ]]
    comp_pair = complementarity_df[
        (complementarity_df["reference_point"] == "training_mean")
        & (complementarity_df["target_loss_domain"] == "composite_equal13")
        & np.isclose(complementarity_df["requested_delta"], 0.01)
    ]
    complementary_pairs = comp_pair.nsmallest(5, "normalized_by_delta_squared")[[
        "domain_i", "domain_j", "normalized_by_delta_squared"
    ]]
    competitive_pairs = comp_pair.nlargest(5, "normalized_by_delta_squared")[[
        "domain_i", "domain_j", "normalized_by_delta_squared"
    ]]
    recommended_report = optimal_mixture_df[[
        "mixture_domain", "recommended_share", "stability_p10_share",
        "stability_p90_share", "training_mean_share", "observed_best_share",
    ]].rename(columns={
        "mixture_domain": "训练域",
        "recommended_share": "支持解配比",
        "stability_p10_share": "稳定性P10",
        "stability_p90_share": "稳定性P90",
        "training_mean_share": "训练均值配比",
        "observed_best_share": "观测最优配比",
    })
    mapped_quality_score = float(mapped_quality_df["selected_support_quality_contribution"].sum())
    extrap_conclusion = "；".join(
        f"{row.scale}: {row.scale_specific_conclusion}"
        for row in extrap_report.itertuples(index=False)
    )
    effect_scope_note = (
        "二阶交互获支持，组合效应与最终预测模型保持同一口径。"
        if SPEC_BY_ID[selected_model_id].order == 2
        else "二阶交互未获支持，最终一阶模型的组合二阶效应为0；不以探索性二阶模型替代最终口径。"
    )
    nonconverged = coefficient_df.loc[
        ~coefficient_df["converged"].astype(bool), ["model_label", "target_loss_domain"]
    ].drop_duplicates()
    if nonconverged.empty:
        convergence_note = "三个候选模型的全部最终拟合均通过收敛检查。"
    else:
        failed_items = "、".join(
            f"{row.model_label}/{row.target_loss_domain}"
            for row in nonconverged.itertuples(index=False)
        )
        convergence_note = (
            f"收敛审计发现 {failed_items} 未在上限内收敛；该模型仅作稳健性对照。"
            f"最终所选 {SPEC_BY_ID[selected_model_id].label_cn} 的13个目标均已收敛。"
        )

    report = rf"""# Step 9 领域配比—交叉熵损失混料响应面报告

## 1. 数据对齐与评价口径

六组配比表和Loss表均按 `index` 一一连接。17个训练域及13个验证Loss域在各尺度字段一致；未发现重复索引、错位、负配比或严重配比和偏差。小数舍入误差在保留原始行和后统一归一化。

{markdown_table(pair_table)}

`nih_exporter`、`enron_emails`、`europarl`、`philpapers` 没有同名验证Loss，但继续作为17维配比变量进入模型，通过单纯形替代关系及交互项影响13个输出域。综合Loss为13域等权均值，**这是政策选择，不是官方总Loss**。

## 2. 模型建立与嵌套验证

一阶基线为

$$\widehat L_k^{{(1)}}(p)=\sum_{{j=1}}^{{17}}b_{{kj}}p_j,$$

二阶Scheffé型混料响应面为

$$\widehat L_k^{{(2)}}(p)=\sum_{{j=1}}^{{17}}b_{{kj}}p_j+\sum_{{j<\ell}}c_{{kj\ell}}p_jp_\ell.$$

二阶模型不设截距、不加入纯平方项，共17个线性项和136个交互项。外层5折估计留出配方误差，内层4折选择超参数。`StandardScaler(with_mean=False)` 只在当前训练折拟合；`model_coefficients.csv` 同时保存缩放空间和原始配比尺度系数。A6--A15从未参与1M模型选参。

{markdown_table(cv_report)}

最佳二阶候选 **{SPEC_BY_ID[best_quad_id].label_cn}** 相对一阶Ridge在 **{quadratic_win_folds}/5** 个外层折取得更低RMSE，交互项支持判定为 **{interaction_supported}**。二阶Ridge是否处于最佳二阶模型预测误差的一个标准误范围内：**{ridge_within_one_se}**。随后在预测误差一个标准误候选中比较15次重复折最优配方稳定性；Ridge若也位于稳定性一个标准误范围内则优先。最终预测模型为 **{SPEC_BY_ID[selected_model_id].label_cn}**，用于效应分析的模型与其一致。{effect_scope_note}

{convergence_note}

![模型对比](figures/cv_model_comparison.png)

## 3. 单纯形替代与组合效应

本文不把普通偏导解释成可执行动作。保持总量不变，从域$i$向域$j$转移占比时使用

$$D_{{i\to j}}L_k(p)=\nabla L_k(p)^\top(e_j-e_i).$$

训练均值配比处，综合Loss最有利的五个可行1%转移动作为：

{markdown_table(best_moves)}

组合效应使用可行路径二阶有限差分

$$S_{{ij}}(p;\delta)=L(p+\Delta_i+\Delta_j)-L(p+\Delta_i)-L(p+\Delta_j)+L(p),$$

新增占比从其余15域按基准占比比例抽出。负值表示指定基准和供给路径下的局部互补，不构成因果协同。训练均值处最强局部互补组合为：

{markdown_table(complementary_pairs)}

最强局部竞争组合为：

{markdown_table(competitive_pairs)}

13个验证域和综合Loss的热力图位于 `figures/pairwise_complementarity_*.png`。

## 4. 三层最优配方

{markdown_table(optimal_report)}

宽单纯形只施加非负与和为1约束，容易离开样本支持区域，因此无论数值多优都不直接推荐。支持解以训练集中实际综合Loss前10%的配方为中心，在L1最近邻构成的局部凸包内优化；K=30为主，K=20和50作敏感性分析。

{markdown_table(support_report)}

支持解相对观测最优的预测改善为 **{predicted_improvement:.6f}**，所选模型OOF RMSE为 **{float(selected_cv_oof['rmse']):.6f}**；K敏感性最大L1差异为 **{support_k_l1:.6f}**。最终状态：**{'可执行推荐候选，但仍需邻域真实实验' if recommendation_pass else '模型候选，当前只建议给出配比区间并补充邻域实验'}**。

这里的“预测改善”严格比较同一冻结模型对观测最优配方和支持解的两个预测值：预测基线为 **{predicted_observed_baseline:.6f}**，支持解预测Loss为 **{selected_support_by_k[SUPPORT_MAIN_K]['predicted_composite_loss']:.6f}**。推荐门槛还要求：支持解最近训练点L1不超过训练配方最近邻L1的P95阈值 **{support_nearest_l1_threshold:.6f}**，最终模型折间P90 L1不超过 **{stability_p90_limit:.2f}**，1M盲测排序相关不低于0.50，且拟合收敛。实际支持距离为 **{selected_support_nearest_l1:.6f}**，折间P90 L1为 **{selected_stability_p90_l1:.6f}**，1M盲测Spearman为 **{blind_test_spearman:.6f}**。

17域最终支持解及重采样稳定性范围如下；范围是配方稳定性区间，不是统计置信区间：

{markdown_table(recommended_report)}

![最优配方对比](figures/optimal_mixture_comparison.png)

## 5. 最优配方折间稳定性

在3次组级随机划分、每次5折的15个训练子集中，使用已由A4/A5内部交叉验证选出的全样本超参数，逐折重新拟合系数、重建局部支持域并优化。冻结超参数可避免把这一描述性稳定性检验误写成新的外部调参环节；它不用于估计泛化误差，A6--A15仍完全未参与调参。10%–90%范围为重采样稳定性区间，不是置信区间。最终模型折内最优向量相对全样本支持解的中位L1为 **{selected_stability_l1:.6f}**、P90 L1为 **{selected_stability_p90_l1:.6f}**。作为求解器对照，Ridge与ElasticNet最优向量的平均Spearman为 **{vector_rho_mean:.6f}**，平均L1距离为 **{vector_l1_mean:.6f}**；该对照不替代最终模型自身的重采样稳定性。

![稳定性区间](figures/optimal_mixture_stability.png)

## 6. 1M盲测与60M、1B零重拟合迁移

{markdown_table(validation_report)}

A6/A7是冻结1M盲测。A8/A9和A10/A11均直接使用A4/A5模型。RMSE、MAE和R²回答绝对Loss水平能否迁移；Spearman、Top-10%重合率和遗憾值回答配方排序能否迁移。二者不能互相替代。`validation_metrics.csv` 另给出25%校准、75%独立留出的仿射校准结果，但它只是次要的“校准后验证”，不替代零重拟合结论。

![跨尺度检验](figures/scale_transfer_comparison.png)

## 7. 10B与70B外推边界

{markdown_table(extrap_report)}

这些Loss本身由模型估算，只能比较排序、候选重合与响应趋势，不能证明真实10B/70B训练有效性。逐尺度结论为：**{extrap_conclusion}**。因此后续只能使用分尺度方案、配比范围或新增大尺度实验，不能无条件迁移1M系数。

![外推排序对照](figures/extrapolation_ranking_comparison.png)

## 8. 质量映射及其定位

A16将7个已有质量域映射到17个配比域。直接映射给较窄区间，近似映射和推断映射依次放宽。所选局部支持配方的描述性映射质量为 **{mapped_quality_score:.6f}**。该值只用于解释和后续质量约束。

拒绝单独加入$Q_{{mix}}=\sum_jp_jQ_j$作为质量回归项，因为它是17个配比变量的确定性线性组合，在模型已包含全部$p_j$时不提供独立可识别信息。

## 9. 对题目的直接回答

1. 17域配比与13域交叉熵Loss之间使用无截距混料响应面刻画；一阶模型提供基线，正则化二阶模型刻画局部组合效应。
2. 域影响必须解释为总量不变的配比转移；`directional_derivatives.csv` 给出全部可执行方向。
3. 两域组合通过可行路径二阶有限差分判断局部互补或竞争，而非直接读取交互系数符号。
4. 数学最优、局部支持最优和观测安全基线已分开。当前推荐状态由同口径预测改善、数据支持距离、最终模型自身折间稳定性、K敏感性、1M盲测排序和收敛状态共同决定。
5. 跨尺度结论分为Loss水平迁移和配方排序迁移。10B/70B估计表不构成真实大模型验证。

## 10. 创新落点

- 避免重复赋权和不可识别的独立质量系数；
- 用局部支持域控制响应面离开数据云后的不合理补偿；
- 用单纯形方向导数解释可执行替代动作；
- 用可行路径二阶有限差分解释局部组合效应；
- 量化最优配方与质量映射的不确定性；
- 严格区分数学最优、数据支持下候选和观测安全基线。

创新来自约束、验证和解释纪律，不是额外堆叠算法。

## 11. 复现与图形方法

运行 `python step9_mixture_response.py` 可从A4--A16与既有Step 6域级Q重建全部结果。随机种子为{SEED}。图表使用真实计算结果、色盲友好配色、300 dpi不透明PNG，并明确区分标准误与重采样稳定性区间。

图形方法参考：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    # ------------------------------------------------------------------
    # 11. Final machine checks
    # ------------------------------------------------------------------
    required_files = [
        "aligned_data.csv", "data_summary.csv", "run_config.json",
        "model_coefficients.csv", "cv_results.csv", "model_comparison.csv",
        "directional_derivatives.csv", "pairwise_complementarity.csv",
        "optimal_mixture_wide.csv", "optimal_mixture_support.csv",
        "optimal_mixture_observed.csv", "optimal_mixture_stability.csv",
        "optimal_mixture.csv",
        "mapped_quality.csv", "validation_metrics.csv",
        "extrapolation_comparison.csv", "report.md",
    ]
    missing = [name for name in required_files if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"缺少交付文件: {missing}")
    for name in required_files:
        if (output_dir / name).stat().st_size == 0:
            raise RuntimeError(f"交付文件为空: {name}")
    if not np.allclose(aligned_all[mix_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-10):
        raise RuntimeError("最终aligned_data中的配比和未通过单纯形检查")
    if not np.isclose(float(selected_support.sum()), 1.0, atol=1e-8) or selected_support.min() < -1e-10:
        raise RuntimeError("最终局部支持配方未通过单纯形检查")
    pngs = sorted(figures_dir.glob("*.png"))
    if len(pngs) < 19:
        raise RuntimeError(f"图表数量不足，实际{len(pngs)}张")
    for path in pngs:
        with Image.open(path) as image:
            if image.width < 900 or image.height < 500 or image.mode != "RGB":
                raise RuntimeError(f"图表元数据异常: {path.name}, {image.size}, {image.mode}")

    print("\n[完成] 输出目录:", output_dir.resolve())
    print("[关键] 最终预测模型:", selected_model_id)
    print("[关键] 二阶交互项支持:", interaction_supported)
    print("[关键] 推荐状态:", "推荐候选" if recommendation_pass else "仅候选区间")
    print("[关键] 观测最优实际综合Loss:", float(y_composite[observed_best_pos]))
    print("[关键] 支持最优预测综合Loss:", selected_support_by_k[SUPPORT_MAIN_K]["predicted_composite_loss"])
    print("[关键] 宽单纯形预测综合Loss:", wide_optima[selected_model_id]["predicted_composite_loss"])
    print("[关键] Ridge/ElasticNet稳定性向量平均Spearman:", vector_rho_mean)
    print("[关键] 图表数量:", len(pngs))


if __name__ == "__main__":
    main()
