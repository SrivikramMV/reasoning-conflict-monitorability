

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from io_utils import RunPaths, read_json, read_jsonl, utc_timestamp, write_json


FOLLOW = 1
BYPASS = 0
BRANCH_ORDER = ["Q", "QSTAR", "OTHER", "AMBIGUOUS"]
COLOURS = {
    "Q": "#326A8C",
    "QSTAR": "#C05A47",
    "OTHER": "#8B9298",
    "AMBIGUOUS": "#C49A3A",
}


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#5A6470",
            "grid.color": "#E1E5E8",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def checkpoint_rows(paths: RunPaths, name: str) -> list[dict[str, Any]]:
    return read_jsonl(paths.drive_checkpoints / f"{name}.jsonl")


def group_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    return np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)


def metrics(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    groups = groups[finite]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {
            "n": int(len(labels)),
            "balanced_accuracy": None,
            "question_weighted_balanced_accuracy": None,
            "roc_auc": None,
            "question_weighted_roc_auc": None,
        }
    predictions = (scores >= 0).astype(int)
    weights = group_weights(groups)
    return {
        "n": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "question_weighted_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions, sample_weight=weights)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "question_weighted_roc_auc": float(roc_auc_score(labels, scores, sample_weight=weights)),
    }


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def centroid_group_scores(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    normalised = l2_normalise(matrix)
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    for held_group in np.unique(groups):
        train = groups != held_group
        test = groups == held_group
        class_centroids = {}
        for label in (BYPASS, FOLLOW):
            subset = normalised[train & (labels == label)]
            if not len(subset):
                continue
            centroid = subset.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            class_centroids[label] = centroid
        if len(class_centroids) == 2:
            scores[test] = (
                normalised[test] @ class_centroids[FOLLOW]
                - normalised[test] @ class_centroids[BYPASS]
            )
    return scores


def pca_logit_group_scores(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    normalised = l2_normalise(matrix)
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    for held_group in np.unique(groups):
        train = np.where(groups != held_group)[0]
        test = np.where(groups == held_group)[0]
        if len(np.unique(labels[train])) < 2:
            continue
        components = min(8, len(train) - 2, matrix.shape[1])
        pca = PCA(n_components=components, whiten=True, random_state=0)
        train_x = pca.fit_transform(normalised[train])
        test_x = pca.transform(normalised[test])
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        classifier.fit(train_x, labels[train], sample_weight=group_weights(groups[train]))
        scores[test] = classifier.decision_function(test_x)
    return scores


def tfidf_group_scores(texts: list[str], labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    for held_group in np.unique(groups):
        train = np.where(groups != held_group)[0]
        test = np.where(groups == held_group)[0]
        if len(np.unique(labels[train])) < 2:
            continue
        vectoriser = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=3000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w/+=-]+\b",
        )
        train_x = vectoriser.fit_transform([texts[index] for index in train])
        test_x = vectoriser.transform([texts[index] for index in test])
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        classifier.fit(train_x, labels[train], sample_weight=group_weights(groups[train]))
        scores[test] = classifier.decision_function(test_x)
    return scores


def analyse_clean(paths: RunPaths) -> dict[str, Any]:
    rows = checkpoint_rows(paths, "01_clean_controls")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rows": 0}
    summary = (
        frame.groupby(["prompt_role", "category"])["final_correct"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    summary.to_csv(paths.analysis_dir / "clean_accuracy_by_role_and_category.csv", index=False)
    truncations = int(frame["truncated"].sum())
    return {
        "rows": len(frame),
        "final_correct": int(frame["final_correct"].sum()),
        "final_accuracy": float(frame["final_correct"].mean()),
        "thought_supports_expected": int(frame["thought_supports_expected"].sum()),
        "truncations": truncations,
    }


def analyse_main(paths: RunPaths) -> dict[str, Any]:
    rows = checkpoint_rows(paths, "02_main_interventions")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rows": 0}
    eligible = frame[frame["clean_source_eligible"]].copy()
    outcome_counts = (
        eligible.groupby(["condition", "provisional_label"]).size().rename("count").reset_index()
    )
    outcome_counts.to_csv(paths.analysis_dir / "main_provisional_taxonomy_counts.csv", index=False)
    branch_counts = (
        eligible.groupby(["condition", "category", "final_branch"])
        .size()
        .rename("count")
        .reset_index()
    )
    branch_counts.to_csv(paths.analysis_dir / "main_final_branch_counts.csv", index=False)

    condition_order = [
        "cot_50_percent",
        "cot_90_percent",
        "full_cot_normal",
        "full_cot_answer_only",
    ]

    categories = sorted(eligible["category"].unique().tolist())
    category_index = pd.MultiIndex.from_product(
        [condition_order, categories, BRANCH_ORDER],
        names=["condition", "category", "final_branch"],
    )
    category_branch = (
        eligible.groupby(["condition", "category", "final_branch"])
        .size()
        .reindex(category_index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    category_branch["category_total"] = category_branch.groupby(
        ["condition", "category"]
    )["count"].transform("sum")
    category_branch["rate"] = category_branch["count"].div(
        category_branch["category_total"].replace(0, np.nan)
    )
    category_branch.to_csv(
        paths.analysis_dir / "main_branch_rates_by_condition_category.csv", index=False
    )
    macro_branch = (
        category_branch.groupby(["condition", "final_branch"])["rate"]
        .mean()
        .rename("macro_task_family_rate")
        .reset_index()
    )
    macro_branch.to_csv(
        paths.analysis_dir / "main_macro_branch_rates_by_condition.csv", index=False
    )

    if "answer_complexity_transition" in eligible.columns:
        complexity = (
            eligible.groupby(
                ["condition", "answer_complexity_transition", "final_branch"]
            )
            .size()
            .rename("count")
            .reset_index()
        )
        complexity["stratum_total"] = complexity.groupby(
            ["condition", "answer_complexity_transition"]
        )["count"].transform("sum")
        complexity["rate"] = complexity["count"] / complexity["stratum_total"]
        complexity.to_csv(
            paths.analysis_dir / "main_branch_rates_by_answer_complexity.csv",
            index=False,
        )

    trajectory = eligible.pivot_table(
        index=["pair_id", "category"],
        columns="condition",
        values="final_branch",
        aggfunc="first",
    ).reset_index()
    for condition in condition_order:
        if condition not in trajectory:
            trajectory[condition] = None
    trajectory["depth_trajectory"] = trajectory.apply(
        lambda row: " -> ".join(str(row[condition]) for condition in condition_order[:3]),
        axis=1,
    )
    trajectory.to_csv(
        paths.analysis_dir / "main_question_level_branch_trajectories.csv", index=False
    )
    trajectory["depth_trajectory"].value_counts().rename_axis("trajectory").rename(
        "questions"
    ).reset_index().to_csv(
        paths.analysis_dir / "main_depth_trajectory_counts.csv", index=False
    )

    full_transition = (
        eligible[eligible["condition"].isin(["full_cot_normal", "full_cot_answer_only"])]
        .pivot_table(
            index=["pair_id", "category"],
            columns="condition",
            values="final_branch",
            aggfunc="first",
        )
        .reset_index()
    )
    if {"full_cot_normal", "full_cot_answer_only"}.issubset(full_transition.columns):
        full_transition.to_csv(
            paths.analysis_dir / "main_full_normal_vs_answer_only_pairs.csv", index=False
        )
        (
            full_transition.groupby(["full_cot_normal", "full_cot_answer_only"])
            .size()
            .rename("questions")
            .reset_index()
            .to_csv(
                paths.analysis_dir / "main_full_normal_vs_answer_only_transitions.csv",
                index=False,
            )
        )

    labels = [
        "FOLLOW",
        "COT_STAGE_CORRECTION",
        "ANSWER_STAGE_BYPASS",
        "FINAL_STAGE_TRANSPARENT_CORRECTION_CANDIDATE",
        "ANSWER_ONLY_FOLLOW",
        "ANSWER_ONLY_BYPASS",
        "Q_RECOVERY_TRACE_UNRESOLVED",
        "QSTAR_FOLLOW_TRACE_UNRESOLVED",
        "UNRESOLVED",
    ]
    colours = [
        "#C05A47",
        "#3B8A73",
        "#326A8C",
        "#C49A3A",
        "#A84A3C",
        "#547D9B",
        "#8B9298",
        "#9B6A78",
        "#B5B8BA",
    ]
    pivot = (
        outcome_counts.pivot(index="condition", columns="provisional_label", values="count")
        .fillna(0)
        .reindex(condition_order)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bottom = np.zeros(len(pivot))
    for label, colour in zip(labels, colours):
        if label not in pivot:
            continue
        values = pivot[label].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=label.replace("_", " ").title(), color=colour)
        bottom += values
    ax.set_ylabel("Eligible intervention runs")
    ax.set_xlabel("")
    ax.set_title("Correction-aware outcomes across the frozen intervention grid")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.analysis_dir / "figures" / "main-taxonomy-by-condition")

    branch_pivot = (
        eligible.groupby(["condition", "final_branch"])
        .size()
        .unstack(fill_value=0)
        .reindex(condition_order)
        .fillna(0)
    )
    branch_rates = branch_pivot.div(branch_pivot.sum(axis=1), axis=0)
    branch_rates.to_csv(paths.analysis_dir / "main_final_branch_rates.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bottom = np.zeros(len(branch_rates))
    for branch in BRANCH_ORDER:
        if branch not in branch_rates:
            continue
        values = branch_rates[branch].to_numpy()
        ax.bar(branch_rates.index, values, bottom=bottom, color=COLOURS[branch], label=branch)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of eligible runs")
    ax.set_title("Final Q versus Q* selection by intervention condition")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.analysis_dir / "figures" / "main-final-branch-rates")
    return {
        "rows": len(frame),
        "eligible_rows": len(eligible),
        "truncations": int(frame["truncated"].sum()),
        "provisional_label_counts": eligible["provisional_label"].value_counts().to_dict(),
        "final_branch_counts": eligible["final_branch"].value_counts().to_dict(),
    }


def analyse_candidate_and_sampling(paths: RunPaths) -> dict[str, Any]:
    candidate_rows = checkpoint_rows(paths, "03_candidate_preference")
    sample_rows = checkpoint_rows(paths, "04_sampling_stability")
    if not candidate_rows:
        return {"candidate_rows": 0, "sample_rows": len(sample_rows)}
    candidates = pd.DataFrame(candidate_rows)
    primary = candidates[candidates["selection_reason"] == "all_eligible_full_cot_linear_cases"].copy()
    primary["sign_matches_observed"] = (
        (primary["delta_average_qstar_minus_q"] > 0) == (primary["observed_branch"] == "QSTAR")
    )
    candidates.to_csv(paths.analysis_dir / "candidate_preference_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for branch, marker in [("Q", "o"), ("QSTAR", "^")]:
        subset = primary[primary["observed_branch"] == branch]
        ax.scatter(
            np.arange(len(subset)),
            subset["delta_average_qstar_minus_q"],
            color=COLOURS[branch],
            marker=marker,
            s=52,
            label=f"Observed {branch}",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#5A6470", linestyle="--", linewidth=1)
    ax.set_xlabel("Cases within observed branch")
    ax.set_ylabel("Q* - Q average continuation log-probability")
    ax.set_title("Candidate-conditioned preference at the answer boundary")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.analysis_dir / "figures" / "candidate-boundary-preference")

    result: dict[str, Any] = {
        "candidate_rows": len(candidates),
        "primary_cases": len(primary),
        "candidate_sign_accuracy": float(primary["sign_matches_observed"].mean()) if len(primary) else None,
        "sample_rows": len(sample_rows),
    }
    if sample_rows:
        samples = pd.DataFrame(sample_rows)
        samples["greedy_reproduced"] = samples["sampled_branch"] == samples["greedy_branch"]
        grouped = (
            samples.groupby(
                ["intervention_id", "temperature", "greedy_branch", "candidate_margin_qstar_minus_q"]
            )
            .agg(
                samples=("sample_id", "count"),
                greedy_reproduction=("greedy_reproduced", "mean"),
                q_rate=("sampled_branch", lambda values: float(np.mean(values == "Q"))),
                qstar_rate=("sampled_branch", lambda values: float(np.mean(values == "QSTAR"))),
                other_rate=(
                    "sampled_branch",
                    lambda values: float(np.mean(~values.isin(["Q", "QSTAR"]))),
                ),
            )
            .reset_index()
        )
        grouped["absolute_margin"] = grouped["candidate_margin_qstar_minus_q"].abs()
        grouped.to_csv(paths.analysis_dir / "sampling_stability_summary.csv", index=False)
        if (
            grouped["absolute_margin"].nunique() > 1
            and grouped["greedy_reproduction"].nunique() > 1
        ):
            correlation = spearmanr(
                grouped["absolute_margin"], grouped["greedy_reproduction"]
            )
            correlation_statistic = float(correlation.statistic)
            correlation_pvalue = float(correlation.pvalue)
        else:
            correlation_statistic = None
            correlation_pvalue = None
        result.update(
            {
                "sampling_absolute_margin_spearman": correlation_statistic,
                "sampling_absolute_margin_pvalue": correlation_pvalue,
                "mean_greedy_reproduction": float(grouped["greedy_reproduction"].mean()),
            }
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        sns.scatterplot(
            data=grouped,
            x="absolute_margin",
            y="greedy_reproduction",
            hue="temperature",
            style="greedy_branch",
            palette="viridis",
            s=70,
            ax=ax,
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Absolute Q* - Q boundary margin")
        ax.set_ylabel("Fraction reproducing the greedy branch")
        ax.set_title("Boundary preference and sampled branch stability")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_figure(fig, paths.analysis_dir / "figures" / "sampling-stability-vs-margin")
    return result


def analyse_visibility(paths: RunPaths, permutations: int = 400) -> dict[str, Any]:
    rows = checkpoint_rows(paths, "05_visibility_latency")
    if not rows:
        return {"rows": 0}
    frame = pd.DataFrame(rows)
    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["pair_id"].to_numpy()
    if len(np.unique(labels)) < 2:
        return {"rows": len(frame), "warning": "Only one branch class was available"}
    arrays = {
        row["intervention_id"]: np.load(paths.feature_dir / row["feature_file"])
        for row in rows
    }
    common_k = sorted(set.intersection(*(set(values["k"].tolist()) for values in arrays.values())))
    depth_count = next(iter(arrays.values()))["hidden"].shape[0]
    metric_rows = []
    prediction_rows = []
    primary_depths = {20: "After block 19", 33: "After block 32", depth_count - 1: "Final norm"}
    for k in common_k:
        visible_texts = [row["visible_base_text"] + row["answer_prefix_by_k"][str(k)] for row in rows]
        visible_scores = tfidf_group_scores(visible_texts, labels, groups)
        metric_rows.append({"k": k, "monitor": "Visible TF-IDF", **metrics(labels, visible_scores, groups)})
        for depth, name in primary_depths.items():
            matrix = np.stack(
                [
                    arrays[row["intervention_id"]]["hidden"][
                        depth, list(arrays[row["intervention_id"]]["k"]).index(k)
                    ]
                    for row in rows
                ],
                axis=0,
            ).astype(np.float32)
            scores = pca_logit_group_scores(matrix, labels, groups)
            metric_rows.append({"k": k, "monitor": name, **metrics(labels, scores, groups)})
            for row, score in zip(rows, scores):
                prediction_rows.append(
                    {
                        "intervention_id": row["intervention_id"],
                        "pair_id": row["pair_id"],
                        "k": k,
                        "monitor": name,
                        "score": float(score) if np.isfinite(score) else None,
                    }
                )

    layer_rows = []
    layer_matrices = []
    for depth in range(depth_count):
        matrix = np.stack(
            [arrays[row["intervention_id"]]["hidden"][depth, 0] for row in rows], axis=0
        ).astype(np.float32)
        layer_matrices.append(matrix)
        scores = centroid_group_scores(matrix, labels, groups)
        layer_rows.append({"depth": depth, **metrics(labels, scores, groups)})
    layer_frame = pd.DataFrame(layer_rows)
    observed_aucs = layer_frame["roc_auc"].dropna()
    observed_max_auc = float(observed_aucs.max()) if len(observed_aucs) else None

    rng = np.random.default_rng(20260717)
    group_values = np.unique(groups)
    trajectories = {group: labels[groups == group].copy() for group in group_values}
    variants = {group: frame.loc[groups == group, "condition"].tolist() for group in group_values}
    max_null = []
    for _ in range(permutations):
        permuted = labels.copy()
        strata: dict[tuple[str, ...], list[str]] = {}
        for group in group_values:
            strata.setdefault(tuple(sorted(variants[group])), []).append(group)
        for members in strata.values():
            sources = rng.permutation(members)
            for target, source in zip(members, sources):
                source_map = {
                    variant: value
                    for variant, value in zip(frame.loc[groups == source, "condition"], trajectories[source])
                }
                target_indices = np.where(groups == target)[0]
                for index in target_indices:
                    permuted[index] = source_map[frame.iloc[index]["condition"]]
        aucs = []
        for depth in range(depth_count):
            score = centroid_group_scores(layer_matrices[depth], permuted, groups)
            finite = np.isfinite(score)
            if len(np.unique(permuted[finite])) == 2:
                aucs.append(roc_auc_score(permuted[finite], score[finite]))
        if aucs:
            max_null.append(max(aucs))
    corrected_p = (
        float((1 + np.sum(np.asarray(max_null) >= observed_max_auc)) / (len(max_null) + 1))
        if max_null and observed_max_auc is not None
        else None
    )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(paths.analysis_dir / "visibility_latency_metrics.csv", index=False)
    layer_frame.to_csv(paths.analysis_dir / "visibility_boundary_layer_sweep.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        paths.analysis_dir / "visibility_latency_predictions.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    plot_frame = metrics_frame[metrics_frame["k"] <= 32]
    sns.lineplot(
        data=plot_frame,
        x="k",
        y="question_weighted_balanced_accuracy",
        hue="monitor",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.axhline(0.5, color="#777F86", linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 1.03)
    ax.set_xlabel("Final-answer tokens visible")
    ax.set_ylabel("Question-weighted balanced accuracy")
    ax.set_title("Hidden and visible prediction across the trace-to-answer transition")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.analysis_dir / "figures" / "visibility-latency")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(layer_frame["depth"], layer_frame["roc_auc"], color="#326A8C", marker="o", markersize=3)
    ax.axhline(0.5, color="#777F86", linestyle="--", linewidth=1)
    if max_null:
        ax.axhline(np.quantile(max_null, 0.95), color="#C49A3A", linestyle=":", linewidth=1.3)
    ax.set_xlabel("Representation depth (35 is the final normalised state)")
    ax.set_ylabel("Question-held-out AUROC")
    ax.set_title("Boundary branch information across Gemma's depth")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.analysis_dir / "figures" / "visibility-layer-sweep")

    visible_tokens = [value for value in frame["first_visible_branch_token"].dropna().tolist()]
    return {
        "rows": len(frame),
        "questions": int(frame["pair_id"].nunique()),
        "follow_rows": int(labels.sum()),
        "bypass_rows": int((labels == BYPASS).sum()),
        "common_k": common_k,
        "median_first_visible_branch_token": float(np.median(visible_tokens)) if visible_tokens else None,
        "observed_max_layer_auc": observed_max_auc,
        "max_layer_permutation_corrected_p": corrected_p,
        "permutations": len(max_null),
        "boundary_metrics": metrics_frame[metrics_frame["k"] == 0].to_dict(orient="records"),
    }


def analyse_source_masking(paths: RunPaths) -> dict[str, Any]:
    rows = checkpoint_rows(paths, "07_source_masking")
    if not rows:
        return {"rows": 0}
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["mask", "masked_branch"]).size().rename("count").reset_index()
    )
    summary.to_csv(paths.analysis_dir / "source_masking_counts.csv", index=False)
    detailed = (
        frame.groupby(["condition", "mask", "natural_branch", "masked_branch"])
        .size()
        .rename("count")
        .reset_index()
    )
    detailed["stratum_total"] = detailed.groupby(
        ["condition", "mask", "natural_branch"]
    )["count"].transform("sum")
    detailed["rate"] = detailed["count"] / detailed["stratum_total"]
    detailed.to_csv(
        paths.analysis_dir / "source_masking_rates_by_condition_and_natural_branch.csv",
        index=False,
    )
    pivot = summary.pivot(index="mask", columns="masked_branch", values="count").fillna(0)
    rates = pivot.div(pivot.sum(axis=1), axis=0)
    rates.to_csv(paths.analysis_dir / "source_masking_rates.csv")
    mask_order = ["none", "question", "cot", "question_and_cot"]
    rates = rates.reindex(mask_order).fillna(0)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bottom = np.zeros(len(rates))
    for branch in BRANCH_ORDER:
        if branch not in rates:
            continue
        values = rates[branch].to_numpy()
        ax.bar(rates.index, values, bottom=bottom, color=COLOURS[branch], label=branch)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of masked continuations")
    ax.set_title("Causal answer-stage dependence on prompt and trace sources")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.analysis_dir / "figures" / "source-masking-outcomes")
    none = frame[frame["mask"] == "none"]
    return {
        "rows": len(frame),
        "cases": int(frame["intervention_id"].nunique()),
        "natural_branch_reproduction": float(
            np.mean(none["masked_branch"] == none["natural_branch"])
        )
        if len(none)
        else None,
        "q_suppressed_qstar_rate": float(
            np.mean(frame.loc[frame["mask"] == "question", "masked_branch"] == "QSTAR")
        ),
        "cot_suppressed_q_rate": float(
            np.mean(frame.loc[frame["mask"] == "cot", "masked_branch"] == "Q")
        ),
        "both_suppressed_other_rate": float(
            np.mean(~frame.loc[frame["mask"] == "question_and_cot", "masked_branch"].isin(["Q", "QSTAR"]))
        ),
    }


def analyse_transparency(paths: RunPaths) -> dict[str, Any]:
    rows = checkpoint_rows(paths, "06_transparency_audit")
    if not rows:
        return {"rows": 0}
    flattened = []
    for row in rows:
        flattened.append(
            {
                "transparency_id": row["transparency_id"],
                "pair_id": row["pair_id"],
                "category": row["category"],
                "instruction_variant": row["instruction_variant"],
                "condition": row["condition"],
                "generated_branch": row["generated_branch"],
                "bridge_language": row["bridge_evidence"]["bridge_language"],
                "mentions_both_changed_values": row["bridge_evidence"]["mentions_both_changed_values"],
            }
        )
    frame = pd.DataFrame(flattened)
    frame.to_csv(paths.analysis_dir / "transparency_automatic_indicators.csv", index=False)
    summary = (
        frame.groupby(["instruction_variant", "condition"])
        .agg(
            runs=("transparency_id", "count"),
            q_branch=("generated_branch", lambda values: int(np.sum(values == "Q"))),
            qstar_branch=("generated_branch", lambda values: int(np.sum(values == "QSTAR"))),
            bridge_language_rate=("bridge_language", "mean"),
            both_values_rate=("mentions_both_changed_values", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(paths.analysis_dir / "transparency_automatic_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    plot = frame.groupby(["instruction_variant", "condition"])["bridge_language"].mean().reset_index()
    sns.barplot(
        data=plot,
        x="instruction_variant",
        y="bridge_language",
        hue="condition",
        palette=["#326A8C", "#8B9298"],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Answers using correction-related language")
    ax.set_title("Automatic transparency indicators pending trace-grounded audit")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, paths.analysis_dir / "figures" / "transparency-automatic-indicators")
    return {
        "rows": len(frame),
        "conflict_rows": int(np.sum(frame["condition"] == "conflict")),
        "matched_control_rows": int(np.sum(frame["condition"] == "matched")),
        "human_audit_required": True,
    }


def write_summary(paths: RunPaths, summaries: dict[str, Any]) -> Path:
    write_json(paths.analysis_dir / "automatic_summary.json", summaries)
    main = summaries.get("main", {})
    candidate = summaries.get("candidate_sampling", {})
    visibility = summaries.get("visibility", {})
    masking = summaries.get("source_masking", {})
    text = f"""# Gemma E2B Final Run: Automatic Analysis

Generated: {utc_timestamp()}

## Data completion

- Clean controls: {summaries.get('clean', {}).get('rows', 0)}
- Main interventions: {main.get('rows', 0)}
- Candidate-preference rows: {candidate.get('candidate_rows', 0)}
- Sampled continuations: {candidate.get('sample_rows', 0)}
- Hidden-state rows: {visibility.get('rows', 0)}
- Source-masking rows: {masking.get('rows', 0)}
- Transparency rows: {summaries.get('transparency', {}).get('rows', 0)}

## Automatic quality checks

- Clean final-answer accuracy: {summaries.get('clean', {}).get('final_accuracy')}
- Eligible primary intervention rows: {main.get('eligible_rows')}
- Main-generation truncations: {main.get('truncations')}
- Candidate-margin sign agreement with the observed branch: {candidate.get('candidate_sign_accuracy')}
- Source-masking natural-branch reproduction under the unmasked control: {masking.get('natural_branch_reproduction')}

## Interpretation boundary

The tables and figures in this directory are automatic diagnostic outputs.
The correction-aware labels remain provisional until the main review packet
has been adjudicated against the complete trace. Transparency language is not
accepted as evidence of genuine correction until every claim has been checked
against the raw prefilled trace in the dedicated transparency review packet.
The hidden-state probes use complete question-level holdout groups, so no
injection variant of a held-out question appears in its training fold.
"""
    path = paths.analysis_dir / "automatic_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_automatic_analysis(paths: RunPaths) -> dict[str, Any]:
    configure_plots()
    summaries = {
        "created_at_utc": utc_timestamp(),
        "clean": analyse_clean(paths),
        "main": analyse_main(paths),
        "candidate_sampling": analyse_candidate_and_sampling(paths),
        "visibility": analyse_visibility(paths),
        "source_masking": analyse_source_masking(paths),
        "transparency": analyse_transparency(paths),
    }
    write_summary(paths, summaries)
    return summaries
