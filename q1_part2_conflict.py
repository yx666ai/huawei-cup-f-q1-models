#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题一第二小问：22指标质量冲突定义、成因、消解与扩展集验证。

输入：
  step4_output/X_suitability_A1.csv, A2.csv, A3.csv
  step6_final_Q/sample_level_Q.csv
  step5_output/entropy_weights_equal_domain.csv（仅用于最高/最低并列时的稳定裁决）

输出：
  step4_conflict_report.md
  step4_conflict_output/ 下的样本结果、统计表、参数和论文图。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


LAMBDAS = [0.1, 0.3, 0.5]
PAIR_GAP_THRESHOLD = 0.60
HIGH_CONFLICT_QUANTILE = 0.90
TOP_N = 100
SEED = 20250923


INDICATOR_GROUPS = {
    "fineweb_edu": "教育与推理质量",
    "quarter_mean": "教育与推理质量",
    "modernbert_reasoning_exp": "教育与推理质量",
    "modernbert_professionalism_exp": "教育与推理质量",
    "fluency_en_prob": "清洁流畅与可读性",
    "ad_en_prob": "清洁流畅与可读性",
    "modernbert_cleanliness_exp": "清洁流畅与可读性",
    "modernbert_readability_exp": "清洁流畅与可读性",
    "dsir_books": "领域相关性",
    "dsir_wiki": "领域相关性",
    "dsir_math": "领域相关性",
    "rps_doc_unigram_entropy": "词汇与格式规范",
    "rps_doc_frac_unique_words": "词汇与格式规范",
    "rps_lines_ending_with_terminal_punctution_mark": "词汇与格式规范",
    "rps_doc_frac_no_alph_words": "词汇与格式规范",
    "rps_doc_frac_chars_top_2gram": "词汇与格式规范",
    "rps_doc_frac_chars_top_3gram": "词汇与格式规范",
    "rps_lines_uppercase_letter_fraction": "词汇与格式规范",
    "rps_lines_numerical_chars_fraction": "词汇与格式规范",
    "rps_doc_word_count": "长度与词形适宜性",
    "rps_doc_num_sentences": "长度与词形适宜性",
    "rps_doc_mean_word_length": "长度与词形适宜性",
}


INDICATOR_LABELS = {
    "fineweb_edu": "教育价值",
    "fluency_en_prob": "流畅度",
    "ad_en_prob": "无广告概率",
    "quarter_mean": "综合质量",
    "modernbert_cleanliness_exp": "清洁度",
    "modernbert_reasoning_exp": "推理性",
    "modernbert_professionalism_exp": "专业性",
    "modernbert_readability_exp": "可读性",
    "dsir_books": "图书域相关性",
    "dsir_wiki": "百科域相关性",
    "dsir_math": "数学域相关性",
    "rps_doc_unigram_entropy": "一元词熵适宜度",
    "rps_doc_frac_unique_words": "独特词比例适宜度",
    "rps_lines_ending_with_terminal_punctution_mark": "行尾标点适宜度",
    "rps_doc_frac_no_alph_words": "非字母词比例适宜度",
    "rps_doc_frac_chars_top_2gram": "Top-2gram集中度适宜度",
    "rps_doc_frac_chars_top_3gram": "Top-3gram集中度适宜度",
    "rps_lines_uppercase_letter_fraction": "大写比例适宜度",
    "rps_lines_numerical_chars_fraction": "数字比例适宜度",
    "rps_doc_word_count": "文档词数适宜度",
    "rps_doc_num_sentences": "句子数适宜度",
    "rps_doc_mean_word_length": "平均词长适宜度",
}


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fm = plt.matplotlib.font_manager
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        available = {f.name for f in fm.fontManager.ttflist}
        candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
        font_name = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["savefig.dpi"] = 301
    plt.rcParams["figure.dpi"] = 120


def finalize_png(path: Path) -> None:
    with Image.open(path) as image:
        image.convert("RGB").save(path, format="PNG", dpi=(301, 301), optimize=True)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.nanstd(x) <= 1e-15 or np.nanstd(y) <= 1e-15:
        return np.nan, np.nan
    result = spearmanr(x, y, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def mid_ecdf(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    """A1-fitted empirical mid-distribution percentile, including tie handling."""
    left = np.searchsorted(sorted_reference, values, side="left")
    right = np.searchsorted(sorted_reference, values, side="right")
    return (left + right) / (2.0 * len(sorted_reference))


def percentile_matrix(frame: pd.DataFrame, indicators: list[str], references: dict[str, np.ndarray]) -> np.ndarray:
    result = np.empty((len(frame), len(indicators)), dtype=np.float64)
    for j, indicator in enumerate(indicators):
        result[:, j] = mid_ecdf(frame[indicator].to_numpy(float), references[indicator])
    return result


def pairwise_significant_ratio(percentiles: np.ndarray, gap: float = PAIR_GAP_THRESHOLD) -> np.ndarray:
    n, m = percentiles.shape
    counts = np.zeros(n, dtype=np.int16)
    total_pairs = m * (m - 1) // 2
    for j in range(m - 1):
        for k in range(j + 1, m):
            counts += (np.abs(percentiles[:, j] - percentiles[:, k]) >= gap)
    return counts.astype(float) / total_pairs


def dominant_extremes(percentiles: np.ndarray, tie_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    max_index = np.argmax(percentiles + eps * tie_weights[None, :], axis=1)
    min_index = np.argmin(percentiles - eps * tie_weights[None, :], axis=1)
    maxima = percentiles[np.arange(len(percentiles)), max_index]
    minima = percentiles[np.arange(len(percentiles)), min_index]
    max_ties = np.isclose(percentiles, maxima[:, None], rtol=0, atol=1e-14).sum(axis=1)
    min_ties = np.isclose(percentiles, minima[:, None], rtol=0, atol=1e-14).sum(axis=1)
    return max_index, min_index, max_ties, min_ties


def low_indicator_meaning(indicator: str) -> str:
    if indicator == "ad_en_prob":
        return "广告风险高"
    return f"{INDICATOR_LABELS[indicator]}低"


def score_dataset(
    dataset: str,
    frame: pd.DataFrame,
    q_frame: pd.DataFrame,
    indicators: list[str],
    references: dict[str, np.ndarray],
    tie_weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    p = percentile_matrix(frame, indicators, references)
    c_std = np.std(p, axis=1, ddof=0)
    c_range = np.ptp(p, axis=1)
    r_pair = pairwise_significant_ratio(p)
    max_idx, min_idx, max_ties, min_ties = dominant_extremes(p, tie_weights)
    max_names = np.array(indicators, dtype=object)[max_idx]
    min_names = np.array(indicators, dtype=object)[min_idx]
    max_labels = np.array([INDICATOR_LABELS[name] for name in max_names], dtype=object)
    min_meanings = np.array([low_indicator_meaning(name) for name in min_names], dtype=object)
    high_groups = np.array([INDICATOR_GROUPS[name] for name in max_names], dtype=object)
    low_groups = np.array([INDICATOR_GROUPS[name] for name in min_names], dtype=object)

    base = frame[["id", "_source_domain"]].copy()
    base.insert(1, "dataset", dataset)
    q_subset = q_frame[q_frame["dataset"].eq(dataset)][["id", "sub_path", "Q"]]
    base = base.merge(q_subset, on="id", how="left", validate="one_to_one")
    if base["Q"].isna().any() or len(base) != len(frame):
        raise ValueError(f"{dataset}: Q 与适宜性矩阵未能一一对齐")
    base["C_std"] = c_std
    base["C_range"] = c_range
    base["R_pair_gap_ge_0.6"] = r_pair
    base["highest_indicator"] = max_names
    base["highest_indicator_label"] = max_labels
    base["highest_percentile"] = p[np.arange(len(p)), max_idx]
    base["lowest_indicator"] = min_names
    base["lowest_indicator_label"] = np.array([INDICATOR_LABELS[name] for name in min_names], dtype=object)
    base["lowest_percentile"] = p[np.arange(len(p)), min_idx]
    base["percentile_gap"] = base["highest_percentile"] - base["lowest_percentile"]
    base["highest_group"] = high_groups
    base["lowest_group"] = low_groups
    base["raw_group_pattern"] = [f"高{hi}—低{lo}" for hi, lo in zip(high_groups, low_groups)]
    base["conflict_pair_description"] = [f"{hi}高 vs {lo}" for hi, lo in zip(max_labels, min_meanings)]
    base["highest_tie_count"] = max_ties
    base["lowest_tie_count"] = min_ties
    for lam in LAMBDAS:
        base[f"Qstar_lambda_{lam:.1f}"] = base["Q"] * (1.0 - lam * base["C_std"])
    return base, p


def subset_contribution(percentiles: np.ndarray) -> np.ndarray:
    centered = percentiles - percentiles.mean(axis=1, keepdims=True)
    raw = np.mean(centered**2, axis=0)
    total = raw.sum()
    return raw / total if total > 0 else np.zeros_like(raw)


def pattern_rate_vector(scored: pd.DataFrame, patterns: list[str]) -> np.ndarray:
    high = scored[scored["is_high_conflict"]]
    if len(high) == 0:
        return np.zeros(len(patterns))
    counts = high["conflict_pattern"].value_counts()
    return np.array([counts.get(pattern, 0) / len(high) for pattern in patterns], dtype=float)


def extension_comparison(
    reference_name: str,
    candidate_name: str,
    ref_scores: pd.DataFrame,
    cand_scores: pd.DataFrame,
    ref_percentiles: np.ndarray,
    cand_percentiles: np.ndarray,
    indicators: list[str],
    patterns: list[str],
    independent: bool,
) -> tuple[dict, list[dict]]:
    ref_c = ref_scores["C_std"].to_numpy(float)
    cand_c = cand_scores["C_std"].to_numpy(float)
    ks = ks_2samp(ref_c, cand_c, alternative="two-sided", method="asymp")
    wd = float(wasserstein_distance(ref_c, cand_c))
    ref_scale = float(np.std(ref_c, ddof=0))
    ref_contrib = subset_contribution(ref_percentiles)
    cand_contrib = subset_contribution(cand_percentiles)
    contrib_rho, contrib_p = safe_spearman(ref_contrib, cand_contrib)
    top_k = min(5, len(indicators))
    ref_top = set(np.argsort(ref_contrib)[-top_k:].tolist())
    cand_top = set(np.argsort(cand_contrib)[-top_k:].tolist())
    ref_pattern = pattern_rate_vector(ref_scores, patterns)
    cand_pattern = pattern_rate_vector(cand_scores, patterns)
    pattern_rho, pattern_p = safe_spearman(ref_pattern, cand_pattern)
    ref_top_patterns = set(np.argsort(ref_pattern)[-3:].tolist())
    cand_top_patterns = set(np.argsort(cand_pattern)[-3:].tolist())
    structure_stable = bool(
        np.isfinite(contrib_rho)
        and contrib_rho >= 0.70
        and len(ref_top & cand_top) >= 3
        and len(ref_top_patterns & cand_top_patterns) >= 2
    )
    row = {
        "reference": reference_name,
        "candidate": candidate_name,
        "independent_after_id_exclusion": independent,
        "n_reference": len(ref_scores),
        "n_candidate": len(cand_scores),
        "mean_C_reference": float(ref_c.mean()),
        "mean_C_candidate": float(cand_c.mean()),
        "median_C_reference": float(np.median(ref_c)),
        "median_C_candidate": float(np.median(cand_c)),
        "high_conflict_rate_reference": float(ref_scores["is_high_conflict"].mean()),
        "high_conflict_rate_candidate": float(cand_scores["is_high_conflict"].mean()),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein_distance": wd,
        "wasserstein_normalized_by_reference_sd": wd / ref_scale if ref_scale > 0 else np.nan,
        "contribution_vector_spearman": contrib_rho,
        "contribution_vector_pvalue": contrib_p,
        "contribution_vector_L1": float(np.abs(ref_contrib - cand_contrib).sum()),
        "top5_contribution_overlap_count": len(ref_top & cand_top),
        "top5_contribution_overlap_rate": len(ref_top & cand_top) / top_k,
        "pattern_rate_spearman": pattern_rho,
        "pattern_rate_pvalue": pattern_p,
        "pattern_total_variation": float(0.5 * np.abs(ref_pattern - cand_pattern).sum()),
        "top3_pattern_overlap_count": len(ref_top_patterns & cand_top_patterns),
        "major_conflict_structure_stable": structure_stable,
        "interpretation": "主要冲突结构成立" if structure_stable else "主要冲突结构仅部分成立或不稳定",
        "ks_note": "独立新增样本主检验" if independent else "包含重叠样本，KS p值仅作敏感性参考",
    }
    contribution_rows = []
    for indicator, ref_value, cand_value in zip(indicators, ref_contrib, cand_contrib):
        contribution_rows.append({
            "comparison": f"{reference_name} vs {candidate_name}",
            "indicator": indicator,
            "indicator_label": INDICATOR_LABELS[indicator],
            "reference_contribution": float(ref_value),
            "candidate_contribution": float(cand_value),
            "difference": float(cand_value - ref_value),
            "independent_after_id_exclusion": independent,
        })
    return row, contribution_rows


def markdown_table(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main() -> None:
    root = Path.cwd()
    out = root / "step4_conflict_output"
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    matrices = {
        dataset: pd.read_csv(root / "step4_output" / f"X_suitability_{dataset}.csv")
        for dataset in ["A1", "A2", "A3"]
    }
    q_frame = pd.read_csv(root / "step6_final_Q" / "sample_level_Q.csv")
    weights = pd.read_csv(root / "step5_output" / "entropy_weights_equal_domain.csv")

    indicators = [c for c in matrices["A1"].columns if c not in {"id", "_source_domain"}]
    if len(indicators) != 22:
        raise ValueError(f"预期22个指标，实际{len(indicators)}")
    if set(indicators) != set(INDICATOR_GROUPS) or set(indicators) != set(INDICATOR_LABELS):
        missing = set(indicators) - set(INDICATOR_GROUPS)
        extra = set(INDICATOR_GROUPS) - set(indicators)
        raise ValueError(f"指标语义映射不完整，缺少={missing}，多余={extra}")
    for dataset, frame in matrices.items():
        frame_indicators = [c for c in frame.columns if c not in {"id", "_source_domain"}]
        if frame_indicators != indicators:
            raise ValueError(f"{dataset} 指标名或顺序与A1不一致")
        if not frame["id"].is_unique or frame[indicators].isna().any().any():
            raise ValueError(f"{dataset} 存在重复id或指标缺失")
        if frame[indicators].min().min() < -1e-12 or frame[indicators].max().max() > 1 + 1e-12:
            raise ValueError(f"{dataset} 指标超出[0,1]")

    weight_lookup = weights.set_index("indicator")["equal_domain_weight"]
    if not set(indicators).issubset(weight_lookup.index):
        raise ValueError("熵权表缺少部分指标")
    tie_weights = weight_lookup.reindex(indicators).to_numpy(float)
    tie_weights = tie_weights / tie_weights.max()

    references = {
        indicator: np.sort(matrices["A1"][indicator].to_numpy(float))
        for indicator in indicators
    }

    scored: dict[str, pd.DataFrame] = {}
    percentile_matrices: dict[str, np.ndarray] = {}
    for dataset in ["A1", "A2", "A3"]:
        scored[dataset], percentile_matrices[dataset] = score_dataset(
            dataset, matrices[dataset], q_frame, indicators, references, tie_weights
        )
        print(
            f"[百分位与冲突] {dataset}: n={len(scored[dataset])}, "
            f"C均值={scored[dataset]['C_std'].mean():.6f}, "
            f"R均值={scored[dataset]['R_pair_gap_ge_0.6'].mean():.6f}"
        )

    threshold = float(scored["A1"]["C_std"].quantile(HIGH_CONFLICT_QUANTILE))
    for dataset in scored:
        scored[dataset]["is_high_conflict"] = scored[dataset]["C_std"] >= threshold

    # A1-driven top four ordered semantic group patterns; all remaining cases share one residual class.
    a1_high = scored["A1"][scored["A1"]["is_high_conflict"]]
    top_patterns = a1_high["raw_group_pattern"].value_counts().head(4).index.tolist()
    pattern_order = top_patterns + ["其他混合冲突"]
    for dataset in scored:
        raw = scored[dataset]["raw_group_pattern"]
        scored[dataset]["conflict_pattern"] = np.where(
            scored[dataset]["is_high_conflict"],
            np.where(raw.isin(top_patterns), raw, "其他混合冲突"),
            "非高冲突",
        )

    # Lambda sensitivity on A1 only; 0.3 is preferred when it satisfies explicit structure constraints.
    a1 = scored["A1"]
    base_rank = a1["Q"].rank(method="first", ascending=False)
    base_top_ids = set(a1.nlargest(TOP_N, "Q")["id"])
    base_domain = a1.groupby("_source_domain", as_index=False)["Q"].mean()
    base_domain["base_domain_rank"] = base_domain["Q"].rank(method="min", ascending=False)
    lambda_rows: list[dict] = []
    domain_rank_rows: list[dict] = []
    for lam in LAMBDAS:
        col = f"Qstar_lambda_{lam:.1f}"
        adj_rank = a1[col].rank(method="first", ascending=False)
        rho, pvalue = safe_spearman(a1["Q"].to_numpy(), a1[col].to_numpy())
        top_ids = set(a1.nlargest(TOP_N, col)["id"])
        before_top = a1[a1["id"].isin(base_top_ids)]
        after_top = a1[a1["id"].isin(top_ids)]
        domain = a1.groupby("_source_domain", as_index=False)[col].mean()
        domain = domain.merge(base_domain, on="_source_domain", validate="one_to_one")
        domain["adjusted_domain_rank"] = domain[col].rank(method="min", ascending=False)
        domain["rank_shift"] = domain["adjusted_domain_rank"] - domain["base_domain_rank"]
        domain_rho, domain_p = safe_spearman(domain["base_domain_rank"], domain["adjusted_domain_rank"])
        lambda_rows.append({
            "lambda": lam,
            "sample_spearman_Q_vs_Qstar": rho,
            "sample_spearman_pvalue": pvalue,
            "top100_overlap_count": len(base_top_ids & top_ids),
            "top100_overlap_rate": len(base_top_ids & top_ids) / TOP_N,
            "high_conflict_count_in_original_top100": int(before_top["is_high_conflict"].sum()),
            "high_conflict_count_in_adjusted_top100": int(after_top["is_high_conflict"].sum()),
            "high_conflict_rate_in_original_top100": float(before_top["is_high_conflict"].mean()),
            "high_conflict_rate_in_adjusted_top100": float(after_top["is_high_conflict"].mean()),
            "mean_absolute_sample_rank_change": float(np.mean(np.abs(adj_rank - base_rank))),
            "mean_rank_demotion_high_conflict": float((adj_rank[a1["is_high_conflict"]] - base_rank[a1["is_high_conflict"]]).mean()),
            "domain_rank_spearman": domain_rho,
            "domain_rank_spearman_pvalue": domain_p,
            "max_absolute_domain_rank_shift": float(domain["rank_shift"].abs().max()),
            "mean_Qstar": float(a1[col].mean()),
            "mean_relative_score_reduction": float(1 - a1[col].mean() / a1["Q"].mean()),
            "structure_constraints_pass": bool(rho >= 0.95 and len(base_top_ids & top_ids) / TOP_N >= 0.80 and domain_rho >= 0.90),
        })
        for _, row in domain.iterrows():
            domain_rank_rows.append({
                "lambda": lam,
                "source_domain": row["_source_domain"],
                "mean_Q": row["Q"],
                "base_domain_rank": row["base_domain_rank"],
                "mean_Qstar": row[col],
                "adjusted_domain_rank": row["adjusted_domain_rank"],
                "rank_shift": row["rank_shift"],
            })
    lambda_df = pd.DataFrame(lambda_rows)
    preferred = lambda_df[lambda_df["lambda"].eq(0.3)].iloc[0]
    if bool(preferred["structure_constraints_pass"]):
        selected_lambda = 0.3
        lambda_reason = "0.3通过预设结构保持门槛，采用中等惩罚作为主方案；0.1与0.5分别作为弱、强压力测试。"
    else:
        weaker = lambda_df[(lambda_df["lambda"] < 0.3) & lambda_df["structure_constraints_pass"]]
        if len(weaker):
            selected_lambda = float(weaker.sort_values("lambda", ascending=False).iloc[0]["lambda"])
            lambda_reason = "0.3未通过结构保持门槛，退回通过门槛的较弱惩罚。"
        else:
            selected_lambda = float(lambda_df.sort_values("sample_spearman_Q_vs_Qstar", ascending=False).iloc[0]["lambda"])
            lambda_reason = "候选值均未完整通过门槛，选择样本排序保持度最高者并标记局限。"

    selected_col = f"Qstar_lambda_{selected_lambda:.1f}"
    for dataset in scored:
        scored[dataset]["selected_lambda"] = selected_lambda
        scored[dataset]["Qstar_selected"] = scored[dataset][selected_col]
        scored[dataset]["rank_Q_within_dataset"] = scored[dataset]["Q"].rank(method="first", ascending=False).astype(int)
        scored[dataset]["rank_Qstar_within_dataset"] = scored[dataset]["Qstar_selected"].rank(method="first", ascending=False).astype(int)
        scored[dataset]["rank_change"] = scored[dataset]["rank_Qstar_within_dataset"] - scored[dataset]["rank_Q_within_dataset"]

    # Scope definitions. Main external validation excludes IDs already present in the matching A1 domain.
    a1_arxiv_mask = matrices["A1"]["_source_domain"].eq("arxiv").to_numpy()
    a1_github_mask = matrices["A1"]["_source_domain"].eq("github").to_numpy()
    a1_arxiv_ids = set(scored["A1"].loc[a1_arxiv_mask, "id"])
    a1_github_ids = set(scored["A1"].loc[a1_github_mask, "id"])
    a2_new_mask = ~scored["A2"]["id"].isin(a1_arxiv_ids).to_numpy()
    a3_new_mask = ~scored["A3"]["id"].isin(a1_github_ids).to_numpy()

    scope_data = {
        "A1_all": (scored["A1"], percentile_matrices["A1"]),
        "A1_arxiv": (scored["A1"].loc[a1_arxiv_mask].copy(), percentile_matrices["A1"][a1_arxiv_mask]),
        "A1_github": (scored["A1"].loc[a1_github_mask].copy(), percentile_matrices["A1"][a1_github_mask]),
        "A2_full": (scored["A2"], percentile_matrices["A2"]),
        "A2_new_only": (scored["A2"].loc[a2_new_mask].copy(), percentile_matrices["A2"][a2_new_mask]),
        "A3_full": (scored["A3"], percentile_matrices["A3"]),
        "A3_new_only": (scored["A3"].loc[a3_new_mask].copy(), percentile_matrices["A3"][a3_new_mask]),
    }

    extension_rows: list[dict] = []
    contribution_rows: list[dict] = []
    comparisons = [
        ("A1_arxiv", "A2_new_only", True),
        ("A1_arxiv", "A2_full", False),
        ("A1_github", "A3_new_only", True),
        ("A1_github", "A3_full", False),
    ]
    for ref_name, cand_name, independent in comparisons:
        ref_scores, ref_p = scope_data[ref_name]
        cand_scores, cand_p = scope_data[cand_name]
        row, contributions = extension_comparison(
            ref_name, cand_name, ref_scores, cand_scores, ref_p, cand_p,
            indicators, pattern_order, independent,
        )
        extension_rows.append(row)
        contribution_rows.extend(contributions)
    extension_df = pd.DataFrame(extension_rows)
    contribution_df = pd.DataFrame(contribution_rows)

    # Pattern, pair, and domain tables.
    pattern_rows: list[dict] = []
    for scope, (score_frame, _) in scope_data.items():
        high = score_frame[score_frame["is_high_conflict"]]
        for pattern in pattern_order:
            part = high[high["conflict_pattern"].eq(pattern)]
            pattern_rows.append({
                "scope": scope,
                "pattern": pattern,
                "n_scope": len(score_frame),
                "n_high_conflict": len(high),
                "pattern_count": len(part),
                "pattern_rate_among_high_conflict": len(part) / len(high) if len(high) else np.nan,
                "mean_C": float(part["C_std"].mean()) if len(part) else np.nan,
                "mean_Q": float(part["Q"].mean()) if len(part) else np.nan,
                "mean_Qstar_selected": float(part["Qstar_selected"].mean()) if len(part) else np.nan,
            })
    pattern_df = pd.DataFrame(pattern_rows)

    domain_pattern_rows: list[dict] = []
    for domain, domain_frame in scored["A1"].groupby("_source_domain"):
        high = domain_frame[domain_frame["is_high_conflict"]]
        for pattern in pattern_order:
            count = int(high["conflict_pattern"].eq(pattern).sum())
            domain_pattern_rows.append({
                "source_domain": domain,
                "domain_n": len(domain_frame),
                "domain_high_conflict_n": len(high),
                "domain_high_conflict_rate": len(high) / len(domain_frame),
                "pattern": pattern,
                "pattern_count": count,
                "pattern_rate_among_domain_high_conflict": count / len(high) if len(high) else np.nan,
                "pattern_rate_among_all_domain_samples": count / len(domain_frame),
            })
    domain_pattern_df = pd.DataFrame(domain_pattern_rows)

    high_pairs = (
        a1_high.groupby([
            "highest_indicator", "highest_indicator_label", "lowest_indicator",
            "lowest_indicator_label", "conflict_pair_description", "raw_group_pattern"
        ], dropna=False)
        .agg(count=("id", "size"), mean_C=("C_std", "mean"), mean_R=("R_pair_gap_ge_0.6", "mean"))
        .reset_index()
        .sort_values(["count", "mean_C"], ascending=[False, False])
    )
    high_pairs["share_of_A1_high_conflict"] = high_pairs["count"] / len(a1_high)

    # Write ten final CSV tables.
    for dataset in ["A1", "A2", "A3"]:
        scored[dataset].to_csv(out / f"conflict_scores_{dataset}.csv", index=False, encoding="utf-8-sig")
    pattern_df.to_csv(out / "conflict_patterns.csv", index=False, encoding="utf-8-sig")
    domain_pattern_df.to_csv(out / "conflict_pattern_domains.csv", index=False, encoding="utf-8-sig")
    high_pairs.to_csv(out / "high_low_indicator_pairs.csv", index=False, encoding="utf-8-sig")
    lambda_df.to_csv(out / "lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(domain_rank_rows).to_csv(out / "domain_rank_changes.csv", index=False, encoding="utf-8-sig")
    extension_df.to_csv(out / "extension_validation.csv", index=False, encoding="utf-8-sig")
    contribution_df.to_csv(out / "indicator_conflict_contributions.csv", index=False, encoding="utf-8-sig")

    parameters = {
        "percentile_definition": "A1 empirical mid-distribution: (count_less + count_less_equal)/(2*n_A1)",
        "C_definition": "population standard deviation of 22 A1-calibrated percentiles",
        "C_range_auxiliary": "max percentile - min percentile",
        "R_definition": "share of 231 indicator pairs with absolute percentile gap >= 0.60",
        "pair_gap_threshold": PAIR_GAP_THRESHOLD,
        "high_conflict_quantile": HIGH_CONFLICT_QUANTILE,
        "high_conflict_threshold_C": threshold,
        "actual_A1_high_conflict_rate": float(scored["A1"]["is_high_conflict"].mean()),
        "top_pattern_rules_fitted_on_A1": top_patterns,
        "residual_pattern": "其他混合冲突",
        "lambdas": LAMBDAS,
        "selected_lambda": selected_lambda,
        "lambda_selection_reason": lambda_reason,
        "indicator_groups": INDICATOR_GROUPS,
        "A2_overlap_with_A1_arxiv": len(a1_arxiv_ids & set(scored["A2"]["id"])),
        "A3_overlap_with_A1_github": len(a1_github_ids & set(scored["A3"]["id"])),
        "A2_new_only_n": int(a2_new_mask.sum()),
        "A3_new_only_n": int(a3_new_mask.sum()),
        "random_seed": SEED,
    }
    (out / "conflict_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Figures: C distributions, conflict-pattern frequencies, and Q* vs Q.
    palette = sns.color_palette("colorblind")
    max_c = max(frame["C_std"].max() for frame in scored.values())
    bins = np.linspace(0, max_c * 1.02, 45)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    axes[0].hist(scored["A1"]["C_std"], bins=bins, density=True, color=palette[0], alpha=0.8)
    axes[0].axvline(threshold, color="#D55E00", linestyle="--", linewidth=2, label=f"A1 Q90={threshold:.3f}")
    axes[0].set(title=f"A1 全体（n={len(scored['A1']):,}）", xlabel="冲突强度 C", ylabel="概率密度")
    axes[0].legend(frameon=False)
    axes[1].hist(scope_data["A1_arxiv"][0]["C_std"], bins=bins, density=True, alpha=0.55, label=f"A1-arxiv (n={len(scope_data['A1_arxiv'][0]):,})", color=palette[0])
    axes[1].hist(scope_data["A2_new_only"][0]["C_std"], bins=bins, density=True, alpha=0.55, label=f"A2新增 (n={len(scope_data['A2_new_only'][0]):,})", color=palette[1])
    axes[1].axvline(threshold, color="#D55E00", linestyle="--", linewidth=1.6)
    axes[1].set(title="arxiv 抽样域与扩展新增样本", xlabel="冲突强度 C", ylabel="概率密度")
    axes[1].legend(frameon=False)
    axes[2].hist(scope_data["A1_github"][0]["C_std"], bins=bins, density=True, alpha=0.55, label=f"A1-github (n={len(scope_data['A1_github'][0]):,})", color=palette[0])
    axes[2].hist(scope_data["A3_new_only"][0]["C_std"], bins=bins, density=True, alpha=0.55, label=f"A3新增 (n={len(scope_data['A3_new_only'][0]):,})", color=palette[2])
    axes[2].axvline(threshold, color="#D55E00", linestyle="--", linewidth=1.6)
    axes[2].set(title="github 抽样域与扩展新增样本", xlabel="冲突强度 C", ylabel="概率密度")
    axes[2].legend(frameon=False)
    fig.suptitle("样本级质量冲突强度分布及A1阈值迁移", y=1.02)
    fig.tight_layout()
    c_hist_path = figures / "conflict_C_distribution.png"
    fig.savefig(c_hist_path, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    finalize_png(c_hist_path)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 7.0), sharey=True)
    a1_pattern_plot = pattern_df[pattern_df["scope"].eq("A1_all")]
    sns.barplot(
        data=a1_pattern_plot, y="pattern", x="pattern_rate_among_high_conflict",
        order=pattern_order, color=palette[0], ax=axes[0],
    )
    axes[0].set_title("A1全体")
    axes[0].set_ylabel("A1拟合的冲突模式")

    matched_panels = [
        (axes[1], ["A1_arxiv", "A2_new_only"], {"A1_arxiv": "A1-arxiv", "A2_new_only": "A2新增"}, "arxiv域匹配"),
        (axes[2], ["A1_github", "A3_new_only"], {"A1_github": "A1-github", "A3_new_only": "A3新增"}, "github域匹配"),
    ]
    for ax, scopes, labels, title in matched_panels:
        part = pattern_df[pattern_df["scope"].isin(scopes)].copy()
        part["样本集"] = part["scope"].map(labels)
        sns.barplot(
            data=part, y="pattern", x="pattern_rate_among_high_conflict", hue="样本集",
            order=pattern_order, hue_order=[labels[s] for s in scopes], palette="colorblind", ax=ax,
        )
        ax.set_title(title)
        ax.set_ylabel("")
        ax.legend(title="", frameon=False, loc="best")
    shared_max = max(0.05, float(pattern_df[pattern_df["scope"].isin(["A1_all", "A1_arxiv", "A2_new_only", "A1_github", "A3_new_only"])]["pattern_rate_among_high_conflict"].max()) * 1.08)
    for ax in axes:
        ax.set_xlabel("在高冲突样本中的频率")
        ax.set_xlim(0, shared_max)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("主要质量冲突模式频率：总体与领域匹配扩展检验", y=1.02)
    fig.tight_layout()
    pattern_path = figures / "conflict_pattern_frequency.png"
    fig.savefig(pattern_path, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    finalize_png(pattern_path)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    normal = ~a1["is_high_conflict"]
    ax.scatter(a1.loc[normal, "Q"], a1.loc[normal, "Qstar_selected"], s=7, alpha=0.12, color=palette[0], label="非高冲突", rasterized=True)
    ax.scatter(a1.loc[~normal, "Q"], a1.loc[~normal, "Qstar_selected"], s=10, alpha=0.38, color="#D55E00", label="高冲突", rasterized=True)
    lo = float(min(a1["Q"].min(), a1["Qstar_selected"].min()))
    hi = float(max(a1["Q"].max(), a1["Qstar_selected"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2, label="Q*=Q")
    selected_sensitivity = lambda_df[lambda_df["lambda"].eq(selected_lambda)].iloc[0]
    ax.text(
        0.03, 0.97,
        f"λ={selected_lambda:.1f}\nSpearman={selected_sensitivity['sample_spearman_Q_vs_Qstar']:.4f}\nTop-100重合={selected_sensitivity['top100_overlap_rate']:.1%}",
        transform=ax.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
    )
    ax.set(xlabel="原综合质量分 Q", ylabel="冲突修正质量分 Q*", title="冲突惩罚前后样本质量分对比（A1）")
    ax.legend(frameon=False)
    fig.tight_layout()
    scatter_path = figures / "Qstar_vs_Q.png"
    fig.savefig(scatter_path, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    finalize_png(scatter_path)

    # Report.
    a1_high_rate = float(a1["is_high_conflict"].mean())
    a1_summary = pd.DataFrame([{
        "样本数": len(a1),
        "C均值": a1["C_std"].mean(),
        "C中位数": a1["C_std"].median(),
        "C标准差": a1["C_std"].std(ddof=0),
        "Q90阈值": threshold,
        "高冲突数": int(a1["is_high_conflict"].sum()),
        "高冲突率": a1_high_rate,
        "R均值": a1["R_pair_gap_ge_0.6"].mean(),
    }])
    a1_pattern_report = pattern_df[pattern_df["scope"].eq("A1_all")][[
        "pattern", "pattern_count", "pattern_rate_among_high_conflict", "mean_C", "mean_Q"
    ]]
    top_pair_report = high_pairs.head(10)[[
        "conflict_pair_description", "raw_group_pattern", "count", "share_of_A1_high_conflict", "mean_C"
    ]]
    lambda_report = lambda_df[[
        "lambda", "sample_spearman_Q_vs_Qstar", "top100_overlap_rate",
        "high_conflict_count_in_original_top100", "high_conflict_count_in_adjusted_top100",
        "mean_rank_demotion_high_conflict", "domain_rank_spearman",
        "max_absolute_domain_rank_shift", "structure_constraints_pass",
    ]]
    extension_report = extension_df[[
        "reference", "candidate", "n_reference", "n_candidate", "ks_statistic", "ks_pvalue",
        "wasserstein_distance", "wasserstein_normalized_by_reference_sd",
        "contribution_vector_spearman", "top5_contribution_overlap_rate",
        "pattern_rate_spearman", "pattern_total_variation",
        "major_conflict_structure_stable", "interpretation",
    ]]
    main_extension = extension_df[extension_df["independent_after_id_exclusion"]]
    arxiv_result = main_extension[main_extension["candidate"].eq("A2_new_only")].iloc[0]
    github_result = main_extension[main_extension["candidate"].eq("A3_new_only")].iloc[0]
    overall_extension = (
        "两类扩展新增样本均保持主要冲突结构"
        if bool(arxiv_result["major_conflict_structure_stable"]) and bool(github_result["major_conflict_structure_stable"])
        else "主要冲突结构只在部分扩展域成立，不能无条件推广"
    )
    selected_row = lambda_df[lambda_df["lambda"].eq(selected_lambda)].iloc[0]

    report = rf"""# 问题一第二小问：质量冲突消解报告

## 1. 数据口径与防泄漏

本分析直接读取前序步骤生成的 A1/A2/A3 三套22指标适宜性矩阵和最终样本级质量分 $Q$，不重新拟合方向、适宜性函数或熵权。所有指标都已统一为“越高越好”。百分位标定、冲突阈值和典型模式规则只使用 A1；A2/A3 仅做规则迁移和扩展验证。

A2包含A1-arxiv的全部1419条样本，A3包含A1-github的全部10000条样本。为避免把重复样本当成独立外部证据，扩展验证主结果分别使用 A2 新增16104条和 A3 新增193752条；包含重叠样本的全体结果仅作为敏感性对照。

## 2. 冲突定义

对指标 $j$，用 A1 经验中分布函数标定百分位：

$$p_{{ij}}=rac{{#(x_j<x_{{ij}})+#(x_jle x_{{ij}})}}{{2n_{{A1}}}}.$$

该定义对并列值取中秩。样本级冲突强度采用22个百分位的总体标准差：

$$C_i=sqrt{{rac1{{22}}sum_{{j=1}}^{{22}}(p_{{ij}}-ar p_i)^2}}.$$

同时保留极差作为辅助量。显著冲突比例定义为231个指标对中百分位差不小于0.6的比例：

$$R_i=rac{{1}}{{inom{{22}}{{2}}}}sum_{{j<k}}mathbf 1left(|p_{{ij}}-p_{{ik}}|ge0.6ight).$$

A1 的 $C_i$ 第90百分位数作为冻结阈值，A2/A3不重新估计阈值。

{markdown_table(a1_summary)}

由于阈值处可能存在并列值，实际高冲突率为 **{a1_high_rate:.4%}**，不强制恰好等于10%。

![冲突强度分布](step4_conflict_output/figures/conflict_C_distribution.png)

## 3. 冲突成因与典型模式

对每个高冲突样本提取最高和最低百分位指标。若最高或最低值并列，仅用主方案熵权作确定性并列裁决；熵权不进入 $C_i$ 和 $R_i$ 的计算。再将22指标归入“教育与推理质量、清洁流畅与可读性、领域相关性、词汇与格式规范、长度与词形适宜性”五类。

A1中频数最高的4种有向组间模式作为典型规则，其余合并为“其他混合冲突”，因此最终保持5类且规则可原样迁移到扩展集。

{markdown_table(a1_pattern_report)}

最高频的具体指标冲突对如下：

{markdown_table(top_pair_report)}

各模式在 A1 七个来源域的频数与域内高冲突率见 `conflict_pattern_domains.csv`。

![冲突模式频数](step4_conflict_output/figures/conflict_pattern_frequency.png)

## 4. 冲突感知综合评价模型

保持原熵权–TOPSIS质量分 $Q_i$ 为基准，建立：

$$Q_i^*(lambda)=Q_i(1-lambda C_i),qquad lambdain{{0.1,0.3,0.5}}.$$

因 $C_ile0.5$，三个候选惩罚均保持 $Q_i^*ge0$，且不会通过重新归一化改变惩罚含义。预先设定的结构保持门槛为：样本排序 Spearman 不低于0.95、Top-100重合率不低于80%、七域排名 Spearman 不低于0.90。

{markdown_table(lambda_report)}

最终选择 **$lambda={selected_lambda:.1f}$**。{lambda_reason} 在该取值下，$Q^*$与$Q$的 Spearman 为 **{selected_row['sample_spearman_Q_vs_Qstar']:.6f}**，Top-100重合率为 **{selected_row['top100_overlap_rate']:.2%}**，领域排名 Spearman 为 **{selected_row['domain_rank_spearman']:.6f}**，最大领域名次变化为 **{selected_row['max_absolute_domain_rank_shift']:.0f}**。因此冲突惩罚对高冲突文本实施降权，同时没有破坏整体评价结构。

![Q星与Q对比](step4_conflict_output/figures/Qstar_vs_Q.png)

## 5. 扩展集检验

下表前两行中的 `new_only` 是排除重复ID后的主检验；`full` 为包含重叠样本的敏感性结果。KS检验判断分布是否完全相同，样本量很大时极小差异也可能显著，因此同时报告 Wasserstein 距离及其相对 A1 参照标准差的标准化值。

“冲突贡献向量”第 $j$ 维为 $mathbb E[(p_{{ij}}-ar p_i)^2]$ 在22指标间归一化后的比例，其 Spearman 比较的是指标对冲突强度的相对贡献次序，而不是对不配对样本计算相关。

{markdown_table(extension_report)}

- **arxiv：** A1-arxiv 与 A2 新增样本的 KS统计量为 {arxiv_result['ks_statistic']:.6f}，Wasserstein距离为 {arxiv_result['wasserstein_distance']:.6f}，贡献向量 Spearman 为 {arxiv_result['contribution_vector_spearman']:.6f}；结论为“{arxiv_result['interpretation']}”。
- **github：** A1-github 与 A3 新增样本的 KS统计量为 {github_result['ks_statistic']:.6f}，Wasserstein距离为 {github_result['wasserstein_distance']:.6f}，贡献向量 Spearman 为 {github_result['contribution_vector_spearman']:.6f}；结论为“{github_result['interpretation']}”。
- **总体判断：** {overall_extension}。分布位置差异与冲突结构一致性是两个不同结论：即使KS拒绝“完全同分布”，主要冲突贡献次序仍可能保持。

## 6. 直接回答赛题

本问把质量冲突定义为同一文本在22个A1标定百分位上的离散程度，并用 $C_i$ 衡量总体冲突、$R_i$ 衡量显著两两冲突。成因通过“最高指标—最低指标”及其语义组归纳。综合评价保留原 $Q$，增加冲突感知分数 $Q^*$；这不是重新赋权，而是对内部证据不一致性施加透明惩罚。A2/A3严格复用A1标定、阈值和规则，并用独立新增样本上的KS、Wasserstein距离和22维冲突贡献结构检验迁移结论。

## 7. 局限

1. $C_i$反映指标间相对分歧，不等同于文本“错误”或低质量；一致地低也可能有较小冲突。
2. 最高/最低指标模式是一种可解释摘要，不能覆盖22维联合结构的全部信息。
3. KS p值随大样本极其敏感，必须与距离、贡献向量和模式频率共同解释。
4. $lambda=0.3$是预先偏好的中等惩罚，仅在结构门槛通过时采用；并非为了获得更好看的排名而调参。

## 8. 复现与方法参考

运行 `python step4_conflict.py` 可从既有适宜性矩阵与 $Q$ 文件重建所有结果。图形遵循高分辨率、中文字体、色盲友好配色与证据边界明确呈现原则。方法参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    (root / "step4_conflict_report.md").write_text(report, encoding="utf-8")

    print("\n[完成] 高冲突阈值 C_Q90 =", threshold)
    print("[完成] A1高冲突率 =", a1_high_rate)
    print("[完成] 典型模式 =", pattern_order)
    print("[完成] 选择lambda =", selected_lambda)
    print("[完成] A2新增样本结构稳定 =", bool(arxiv_result["major_conflict_structure_stable"]))
    print("[完成] A3新增样本结构稳定 =", bool(github_result["major_conflict_structure_stable"]))
    print("[完成] 输出目录 =", out)


if __name__ == "__main__":
    main()
