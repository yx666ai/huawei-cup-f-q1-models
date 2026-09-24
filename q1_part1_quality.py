#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""华为杯 F 题第一问第 1 小问：22 指标质量评价完整单文件版。

本文件把已经完成的 Step 2--6 核心流程整理为一份可独立阅读和运行的脚本：

1. 读取 A1/A2/A3 原始 JSONL.XZ；
2. 压缩 8 个列表字段，形成 14+8=22 个质量指标；
3. 全局中位数填补、方向统一和 Min-Max 归一化；
4. 使用 A1 已拟合并冻结的适宜性函数处理 10 个数值指标；
5. 计算“等数据集影响熵权”和“合并样本熵权”；
6. 在适宜性矩阵上直接执行 TOPSIS，不做第二次归一化；
7. 输出样本级 Q、域级 Q、扩展集对照、诊断表和论文图。

默认从当前目录下的 ``real_attachments/A_data_value`` 或 ``A_data_value`` 查找数据。
也可以显式指定：
    python q1_part1_quality.py --data-root <A_data_value目录> --output-root <输出根目录>
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


EPS = 1e-12
DATASETS = ("A1", "A2", "A3")
META = ["id", "_source_domain"]

SCALAR_FIELDS = [
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]

TRANSFORMED_FIELDS = [
    "fineweb_edu",
    "fluency_en_prob",
    "ad_en_prob",
    "quarter_mean",
    "modernbert_cleanliness_exp",
    "modernbert_reasoning_exp",
    "modernbert_professionalism_exp",
    "modernbert_readability_exp",
]

POSITIVE = [
    "fineweb_edu",
    "fluency_en_prob",
    "ad_en_prob",
    "quarter_mean",
    "modernbert_cleanliness_exp",
    "modernbert_reasoning_exp",
    "modernbert_professionalism_exp",
    "modernbert_readability_exp",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
]

NEGATIVE = [
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]

METRICS = [
    "fineweb_edu",
    "fluency_en_prob",
    "ad_en_prob",
    "quarter_mean",
    "modernbert_cleanliness_exp",
    "modernbert_reasoning_exp",
    "modernbert_professionalism_exp",
    "modernbert_readability_exp",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_mean_word_length",
]

SUITABILITY_SPECS: dict[str, dict[str, Any]] = {
    "rps_doc_word_count": {
        "function": "gaussian_shouldered_interval",
        "parameters": {
            "low": 342.0,
            "high": 2669.0999999999913,
            "left_scale": 0.9067758413771552,
            "right_scale": 1.6596602506772176,
            "transform": "log1p",
        },
    },
    "rps_doc_num_sentences": {
        "function": "gaussian_shouldered_interval",
        "parameters": {
            "low": 27.0,
            "high": 219.0,
            "left_scale": 0.8027930436158813,
            "right_scale": 1.59343791208349,
            "transform": "log1p",
        },
    },
    "rps_doc_mean_word_length": {
        "function": "gaussian",
        "parameters": {"center": 5.322730202758136, "sigma": 2.381904906309602},
    },
    "rps_lines_ending_with_terminal_punctution_mark": {
        "function": "smooth_trapezoid",
        "parameters": {
            "left_zero": 12.5,
            "left_full": 25.0,
            "right_full": 71.42857142857143,
            "right_zero": 100.0,
        },
    },
    "rps_lines_numerical_chars_fraction": {
        "function": "decreasing_logistic",
        "parameters": {"midpoint": 2.5564231046297645, "scale": 1.1634782948445876},
    },
    "rps_lines_uppercase_letter_fraction": {
        "function": "decreasing_logistic",
        "parameters": {"midpoint": 41.013242533255436, "scale": 17.221028204652665},
    },
    "rps_doc_frac_chars_top_2gram": {
        "function": "smooth_trapezoid",
        "parameters": {
            "left_zero": 0.0,
            "left_full": 1.0,
            "right_full": 2.2703260536077874,
            "right_zero": 7.962529274004685,
        },
    },
    "rps_doc_frac_chars_top_3gram": {
        "function": "smooth_trapezoid",
        "parameters": {
            "left_zero": 0.0,
            "left_full": 0.8005408121560158,
            "right_full": 2.2325603711071635,
            "right_zero": 10.0,
        },
    },
    "rps_doc_unigram_entropy": {
        "function": "linear_retained",
        "parameters": {"global_min": 0.0, "global_max": 7.84673154, "direction": "positive"},
    },
    "rps_doc_frac_unique_words": {
        "function": "decreasing_logistic",
        "parameters": {"midpoint": 55.73770491803278, "scale": 8.953202988575965},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def find_data_root(explicit: Path | None) -> Path:
    candidates = [] if explicit is None else [explicit]
    candidates += [
        Path.cwd() / "real_attachments" / "A_data_value",
        Path.cwd() / "A_data_value",
    ]
    for candidate in candidates:
        if (candidate / "slimpajama_quality_signal_sample.jsonl.xz").is_file():
            return candidate.resolve()
    raise FileNotFoundError("未找到 A_data_value；请使用 --data-root 显式指定。")


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def finite_vector(value: Any, length: int) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if np.isfinite(array).all() else None


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def second_probability(value: Any) -> float:
    vector = finite_vector(value, 2)
    return float(softmax(vector)[1]) if vector is not None else math.nan


def expected_level(value: Any) -> float:
    vector = finite_vector(value, 6)
    if vector is None:
        return math.nan
    return float(np.dot(np.arange(6, dtype=float), softmax(vector)))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = lzma.open if path.suffix.lower() == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON解析失败：{path}:{line_number}: {exc}") from exc


def raw_scalar(record: dict[str, Any], field: str) -> float:
    aliases = {
        "rps_doc_unigram_entropy": ["rps_doc_unigram_entropy", "rps_doc_uniqram_entropy"],
        "rps_lines_ending_with_terminal_punctution_mark": [
            "rps_lines_ending_with_terminal_punctution_mark",
            "rps_lines_ending_with_terminal_punctuation_mark",
            "rps_lines_ending_with_terminal_punctuation_nmark",
        ],
    }
    for candidate in aliases.get(field, [field]):
        if candidate in record:
            return finite_float(record[candidate])
    return math.nan


def load_dataset(path: Path, dataset: str, fixed_domain: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        quarter = finite_vector(record.get("qurater"), 4)
        fineweb = finite_vector(record.get("fineweb_edu"), 1)
        row: dict[str, Any] = {
            "id": record.get("id", ""),
            "_source_domain": fixed_domain or str(record.get("_source_domain", "unknown")),
            "fineweb_edu": float(fineweb[0]) if fineweb is not None else math.nan,
            "fluency_en_prob": second_probability(record.get("fluency_en")),
            "ad_en_prob": second_probability(record.get("ad_en")),
            "modernbert_cleanliness_exp": expected_level(record.get("modernbert_cleanliness")),
            "modernbert_reasoning_exp": expected_level(record.get("modernbert_reasoning")),
            "modernbert_professionalism_exp": expected_level(record.get("modernbert_professionalism")),
            "modernbert_readability_exp": expected_level(record.get("modernbert_readability")),
        }
        for index in range(4):
            row[f"_quarter_{index}"] = float(quarter[index]) if quarter is not None else math.nan
        for field in SCALAR_FIELDS:
            row[field] = raw_scalar(record, field)
        rows.append(row)
    frame = pd.DataFrame(rows)
    print(f"读取 {dataset}: {len(frame):,} 行 <- {path.name}")
    return frame


def compress_quarter(frames: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    parameters: dict[str, tuple[float, float]] = {}
    for index in range(4):
        column = f"_quarter_{index}"
        combined = pd.concat([frames[name][column] for name in DATASETS], ignore_index=True)
        minimum, maximum = float(combined.min()), float(combined.max())
        parameters[column] = (minimum, maximum)
    for frame in frames.values():
        parts = []
        for column, (minimum, maximum) in parameters.items():
            if np.isclose(minimum, maximum):
                part = pd.Series(0.5, index=frame.index)
            else:
                part = (frame[column] - minimum) / (maximum - minimum)
            parts.append(part)
        frame["quarter_mean"] = pd.concat(parts, axis=1).mean(axis=1, skipna=False)
        frame.drop(columns=list(parameters), inplace=True)
    return parameters


def save_frames(frames: dict[str, pd.DataFrame], directory: Path, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for dataset, frame in frames.items():
        frame.to_csv(
            directory / f"{prefix}_{dataset}.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.12g",
        )


def direction_normalize(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.Series, pd.Series, pd.Series]:
    combined = pd.concat([frames[name][METRICS] for name in DATASETS], ignore_index=True)
    medians = combined.median(skipna=True)
    if medians.isna().any():
        raise ValueError("存在整列缺失指标，无法执行中位数填补。")
    filled: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        current = frame[META + METRICS].copy()
        current.loc[:, METRICS] = current[METRICS].fillna(medians)
        filled[name] = current
    combined_filled = pd.concat([filled[name][METRICS] for name in DATASETS], ignore_index=True)
    minima, maxima = combined_filled.min(), combined_filled.max()
    normalized: dict[str, pd.DataFrame] = {}
    for name, frame in filled.items():
        result = frame[META].copy()
        for metric in METRICS:
            if np.isclose(minima[metric], maxima[metric]):
                values = np.full(len(frame), 0.5)
            else:
                values = (frame[metric].to_numpy(float) - minima[metric]) / (maxima[metric] - minima[metric])
                if metric in NEGATIVE:
                    values = 1.0 - values
            result[metric] = np.clip(values, 0.0, 1.0)
        normalized[name] = result
    return filled, normalized, medians, minima, maxima


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def apply_spec(values: Iterable[float], spec: dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    function = spec["function"]
    params = spec["parameters"]
    if function == "smooth_trapezoid":
        left = smoothstep((x - params["left_zero"]) / (params["left_full"] - params["left_zero"]))
        right = smoothstep((params["right_zero"] - x) / (params["right_zero"] - params["right_full"]))
        score = np.minimum(left, right)
    elif function == "gaussian_shouldered_interval":
        transformed = np.log1p(np.clip(x, 0.0, None)) if params.get("transform") == "log1p" else x
        low = math.log1p(params["low"]) if params.get("transform") == "log1p" else params["low"]
        high = math.log1p(params["high"]) if params.get("transform") == "log1p" else params["high"]
        score = np.ones_like(transformed)
        left_mask, right_mask = transformed < low, transformed > high
        score[left_mask] = np.exp(-0.5 * ((transformed[left_mask] - low) / params["left_scale"]) ** 2)
        score[right_mask] = np.exp(-0.5 * ((transformed[right_mask] - high) / params["right_scale"]) ** 2)
    elif function == "gaussian":
        score = np.exp(-0.5 * ((x - params["center"]) / params["sigma"]) ** 2)
    elif function == "decreasing_logistic":
        exponent = np.clip((x - params["midpoint"]) / params["scale"], -60.0, 60.0)
        score = 1.0 / (1.0 + np.exp(exponent))
    elif function == "linear_retained":
        low, high = params["global_min"], params["global_max"]
        score = np.full_like(x, 0.5) if np.isclose(low, high) else (x - low) / (high - low)
        if params.get("direction") == "negative":
            score = 1.0 - score
    else:
        raise ValueError(f"未知适宜性函数：{function}")
    return np.clip(score, 0.0, 1.0)


def build_suitability(
    filled: dict[str, pd.DataFrame], normalized: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name in DATASETS:
        current = normalized[name].copy()
        for metric, spec in SUITABILITY_SPECS.items():
            current[metric] = apply_spec(filled[name][metric], spec)
        values = current[METRICS].to_numpy(float)
        if not np.isfinite(values).all() or values.min() < -EPS or values.max() > 1 + EPS:
            raise ValueError(f"{name} 适宜性矩阵范围检查失败。")
        result[name] = current
    return result


def entropy_weight(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    sums = x.sum(axis=0)
    valid = sums > EPS
    p = np.zeros_like(x)
    p[:, valid] = x[:, valid] / sums[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        p_log_p = np.where(p > 0, p * np.log(p), 0.0)
    entropy = np.ones(x.shape[1])
    entropy[valid] = -p_log_p[:, valid].sum(axis=0) / math.log(x.shape[0])
    divergence = np.clip(1.0 - entropy, 0.0, None)
    weights = divergence / divergence.sum() if divergence.sum() > EPS else np.full(x.shape[1], 1 / x.shape[1])
    return entropy, divergence, weights


def equal_dataset_weights(frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, pd.DataFrame]:
    internals = {name: entropy_weight(frames[name][METRICS].to_numpy(float)) for name in DATASETS}
    weights = np.mean([internals[name][2] for name in DATASETS], axis=0)
    weights /= weights.sum()
    table = pd.DataFrame({"indicator": METRICS})
    for name in DATASETS:
        table[f"{name}_entropy"] = internals[name][0]
        table[f"{name}_divergence"] = internals[name][1]
        table[f"{name}_weight"] = internals[name][2]
    table["equal_domain_weight"] = weights
    table["rank"] = table["equal_domain_weight"].rank(ascending=False, method="min").astype(int)
    return weights, table.sort_values("rank", ignore_index=True)


def topsis(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = np.asarray(values, dtype=float) * np.asarray(weights, dtype=float)
    best, worst = weighted.max(axis=0), weighted.min(axis=0)
    d_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))
    return np.divide(d_worst, d_best + d_worst, out=np.full(len(values), 0.5), where=(d_best + d_worst) > EPS)


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(ascending=False, method="min").astype(np.int64).to_numpy()


def save_scores(
    suitability: dict[str, pd.DataFrame],
    normalized: dict[str, pd.DataFrame],
    output_root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    step5 = output_root / "step5_output"
    step5.mkdir(parents=True, exist_ok=True)
    main_weights, main_table = equal_dataset_weights(suitability)
    pooled_values = np.vstack([suitability[name][METRICS].to_numpy(float) for name in DATASETS])
    pooled_entropy, pooled_divergence, pooled_weights = entropy_weight(pooled_values)
    pooled_table = pd.DataFrame(
        {
            "indicator": METRICS,
            "pooled_entropy": pooled_entropy,
            "pooled_divergence": pooled_divergence,
            "pooled_weight": pooled_weights,
        }
    )
    pooled_table["rank"] = pooled_table["pooled_weight"].rank(ascending=False, method="min").astype(int)
    main_table.to_csv(step5 / "entropy_weights_equal_domain.csv", index=False, encoding="utf-8-sig")
    pooled_table.sort_values("rank").to_csv(step5 / "entropy_weights_pooled.csv", index=False, encoding="utf-8-sig")

    linear_weights, _ = equal_dataset_weights(normalized)
    linear_values = np.vstack([normalized[name][METRICS].to_numpy(float) for name in DATASETS])
    scenarios = {
        "equal_domain_entropy": topsis(pooled_values, main_weights),
        "pooled_entropy": topsis(pooled_values, pooled_weights),
        "equal_weight": topsis(pooled_values, np.full(len(METRICS), 1 / len(METRICS))),
        "linear_fixed_main_weight": topsis(linear_values, main_weights),
        "linear_own_entropy": topsis(linear_values, linear_weights),
    }
    global_ranks = {name: ranks(score) for name, score in scenarios.items()}
    outputs: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    start = 0
    for dataset in DATASETS:
        stop = start + len(suitability[dataset])
        result = suitability[dataset][META].copy()
        result.insert(0, "_dataset", dataset)
        for scenario, all_scores in scenarios.items():
            current = all_scores[start:stop]
            result[f"score_{scenario}"] = current
            result[f"rank_global_{scenario}"] = global_ranks[scenario][start:stop]
            result[f"rank_dataset_{scenario}"] = ranks(current)
            summaries.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "n": len(current),
                    "mean": float(np.mean(current)),
                    "median": float(np.median(current)),
                    "std": float(np.std(current, ddof=1)),
                }
            )
        result.to_csv(step5 / f"topsis_scores_{dataset}.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        outputs[dataset] = result
        start = stop
    summary = pd.DataFrame(summaries)
    summary.to_csv(step5 / "topsis_dataset_summary.csv", index=False, encoding="utf-8-sig")
    return outputs, summary, main_table


def aggregate_q(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, dropna=False)["Q"]
        .agg(n="size", mean_Q="mean", std_Q="std", median_Q="median", min_Q="min", max_Q="max")
        .reset_index()
    )


def final_q_outputs(scores: dict[str, pd.DataFrame], output_root: Path) -> pd.DataFrame:
    out = output_root / "step6_final_Q"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for dataset, frame in scores.items():
        parts.append(
            pd.DataFrame(
                {
                    "id": frame["id"],
                    "dataset": dataset,
                    "_source_domain": frame["_source_domain"],
                    "sub_path": "",
                    "Q": frame["score_equal_domain_entropy"],
                    "rank": frame["rank_global_equal_domain_entropy"],
                }
            )
        )
    sample = pd.concat(parts, ignore_index=True)
    sample.to_csv(out / "sample_level_Q.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    a1 = aggregate_q(sample[sample["dataset"] == "A1"], ["_source_domain"]).sort_values("mean_Q", ascending=False)
    extension = aggregate_q(sample[sample["dataset"].isin(["A2", "A3"])], ["dataset", "_source_domain"])
    a1.to_csv(out / "A1_domain_Q.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    extension.to_csv(out / "A2_A3_domain_Q.csv", index=False, encoding="utf-8-sig", float_format="%.10f")

    rows = []
    for domain, dataset in (("arxiv", "A2"), ("github", "A3")):
        sample_q = float(a1.loc[a1["_source_domain"].astype(str).str.lower() == domain, "mean_Q"].iloc[0])
        extension_q = float(extension.loc[extension["dataset"] == dataset, "mean_Q"].iloc[0])
        diff = extension_q - sample_q
        rows.append(
            {
                "domain": domain,
                "A1_sample_Q": sample_q,
                "extension_Q": extension_q,
                "diff": diff,
                "rel_diff": diff / sample_q,
                "consistency": "高度一致" if abs(diff / sample_q) < 0.05 else "存在差异",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "domain_Q_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    return sample


def make_figures(
    suitability: dict[str, pd.DataFrame], sample_q: pd.DataFrame, output_root: Path
) -> None:
    step5 = output_root / "step5_output"
    combined = pd.concat([suitability[name].assign(dataset=name) for name in DATASETS], ignore_index=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(combined[METRICS].corr(method="spearman"), cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("22个质量指标 Spearman 相关性")
    fig.tight_layout()
    fig.savefig(step5 / "corr_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=sample_q, x="dataset", y="Q", order=list(DATASETS), ax=ax)
    ax.set_title("A1/A2/A3 综合质量分分布")
    ax.set_xlabel("数据集")
    ax.set_ylabel("Q")
    fig.tight_layout()
    fig.savefig(step5 / "topsis_boxplot.png")
    plt.close(fig)


def write_reports(
    output_root: Path,
    quarter_params: dict[str, tuple[float, float]],
    medians: pd.Series,
    minima: pd.Series,
    maxima: pd.Series,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    sample_q: pd.DataFrame,
) -> None:
    step2, step3, step4, step5, step6 = [
        output_root / name
        for name in ("step2_output", "step3_output", "step4_output", "step5_output", "step6_final_Q")
    ]
    quarter_lines = ["# Quarter 处理报告", "", "四维分别使用 A1+A2+A3 全局 Min-Max 后求算术平均。", ""]
    for index, (minimum, maximum) in enumerate(quarter_params.values(), start=1):
        quarter_lines.append(f"- 维度{index}: min={minimum:.10g}, max={maximum:.10g}")
    (step2 / "quarter_processing_report.md").write_text("\n".join(quarter_lines), encoding="utf-8")

    direction = pd.DataFrame(
        {
            "指标名": METRICS,
            "原始方向": ["负向" if metric in NEGATIVE else "正向" for metric in METRICS],
            "最终方向": "正向（越高越好）",
            "处理方式（是否翻转）": ["全局 Min-Max 后执行 1-x（是）" if metric in NEGATIVE else "全局 Min-Max（否）" for metric in METRICS],
            "填补中位数": [medians[metric] for metric in METRICS],
            "全局最小值": [minima[metric] for metric in METRICS],
            "全局最大值": [maxima[metric] for metric in METRICS],
        }
    )
    direction.to_csv(step3 / "direction_table.csv", index=False, encoding="utf-8-sig")
    (step4 / "suitability_functions.json").write_text(
        json.dumps({"fit_dataset": "A1", "reused_without_refit_on_A2_A3": True, "metrics": SUITABILITY_SPECS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    main_summary = summary[summary["scenario"] == "equal_domain_entropy"]
    report = [
        "# 第一问第1小问：数据质量评价结果",
        "",
        "最终 Q 使用22指标适宜性矩阵、等数据集影响熵权和TOPSIS计算；TOPSIS前不再执行第二次Min-Max。",
        "",
        "## 数据集级结果",
        "",
        main_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 权重前10",
        "",
        weights.head(10)[["indicator", "equal_domain_weight", "rank"]].to_markdown(index=False, floatfmt=".6f"),
        "",
        f"样本总数：{len(sample_q):,}；Q范围：[{sample_q['Q'].min():.6f}, {sample_q['Q'].max():.6f}]。",
    ]
    (step5 / "step5_diagnostics.md").write_text("\n".join(report), encoding="utf-8")
    (step6 / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_plot_style()
    data_root = find_data_root(args.data_root)
    output_root = args.output_root.resolve()
    extended = data_root / "slimpajama_quality_extended"
    a2_files = sorted(extended.glob("arxiv_*.jsonl.xz"))
    a3_files = sorted(extended.glob("github_*.jsonl.xz"))
    if len(a2_files) != 1 or len(a3_files) != 1:
        raise FileNotFoundError("A2/A3 扩展文件应各唯一匹配一个。")
    specs = {
        "A1": (data_root / "slimpajama_quality_signal_sample.jsonl.xz", None),
        "A2": (a2_files[0], "arxiv"),
        "A3": (a3_files[0], "github"),
    }
    frames = {name: load_dataset(path, name, domain) for name, (path, domain) in specs.items()}
    quarter_params = compress_quarter(frames)
    preprocessed = {name: frames[name][META + METRICS].copy() for name in DATASETS}
    save_frames(preprocessed, output_root / "step2_output", "X_preprocessed")

    filled, normalized, medians, minima, maxima = direction_normalize(preprocessed)
    save_frames(normalized, output_root / "step3_output", "X_normalized")
    suitability = build_suitability(filled, normalized)
    save_frames(suitability, output_root / "step4_output", "X_suitability")

    scores, summary, weights = save_scores(suitability, normalized, output_root)
    sample_q = final_q_outputs(scores, output_root)
    make_figures(suitability, sample_q, output_root)
    write_reports(output_root, quarter_params, medians, minima, maxima, summary, weights, sample_q)

    print("\n第一问第1小问完成。")
    print(f"数据根目录：{data_root}")
    print(f"输出根目录：{output_root}")
    print(summary[summary["scenario"] == "equal_domain_entropy"].to_string(index=False))


if __name__ == "__main__":
    main()
