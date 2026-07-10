

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VARIANT_ORDER = ["cot_50_percent", "cot_90_percent", "full_cot_normal"]
VARIANT_LABELS = {
    "cot_50_percent": "50%",
    "cot_90_percent": "90%",
    "full_cot_normal": "Full CoT",
}
FOLLOW = 1
BYPASS = 0


def load_features(feature_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows = []
    hidden = []
    for path in sorted(feature_dir.glob("nat_lin3_*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("extraction_version") != 2:
            raise ValueError(f"Outdated extraction cache: {path}")
        arrays = np.load(path.with_suffix(".npz"))
        stack = np.concatenate([arrays["layer_inputs"], arrays["final_norm"][None, :]], axis=0)
        hidden.append(stack.astype(np.float32))
        rows.append(meta)
    frame = pd.DataFrame(rows)
    matrix = np.stack(hidden, axis=0)
                                                                        
                                                         
    layer_names = ["embedding"] + [f"after_block_{i}" for i in range(34)] + ["final_norm"]
    if matrix.shape[1] != len(layer_names):
        raise ValueError(f"Unexpected hidden-depth count {matrix.shape[1]}.")
    return frame, matrix, layer_names


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def group_row_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    return np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)


def centroid_scores_from_similarity(
    similarity: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_selector: Callable[[str], np.ndarray] | None = None,
    test_selector: Callable[[str], np.ndarray] | None = None,
) -> np.ndarray:
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    base_weights = group_row_weights(groups)
    for held_group in np.unique(groups):
        train = groups != held_group
        test = groups == held_group
        if train_selector is not None:
            train &= train_selector(held_group)
        if test_selector is not None:
            test &= test_selector(held_group)
        if not np.any(test):
            continue
        class_scores = {}
        for cls in (BYPASS, FOLLOW):
            idx = np.where(train & (labels == cls))[0]
            if len(idx) == 0:
                raise ValueError(f"No class {cls} training rows while holding out {held_group}.")
            weights = base_weights[idx]
            weights = weights / weights.sum()
            centroid_norm_sq = float(weights @ similarity[np.ix_(idx, idx)] @ weights)
            denominator = math.sqrt(max(centroid_norm_sq, 1e-12))
            class_scores[cls] = similarity[np.ix_(np.where(test)[0], idx)] @ weights / denominator
        scores[test] = class_scores[FOLLOW] - class_scores[BYPASS]
    return scores


def centroid_cv_scores(x: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    normalised = l2_normalise(x)
    similarity = normalised @ normalised.T
    return centroid_scores_from_similarity(similarity, labels, groups)


def grouped_model_scores(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    for held_group in np.unique(groups):
        train = np.where(groups != held_group)[0]
        test = np.where(groups == held_group)[0]
        scores[test] = fit_predict(train, test)
    return scores


def metric_bundle(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(scores)
    y = labels[mask]
    s = scores[mask]
    pred = (s >= 0).astype(int)
    weights = group_row_weights(groups[mask])
    return {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "question_weighted_balanced_accuracy": float(
            balanced_accuracy_score(y, pred, sample_weight=weights)
        ),
        "roc_auc": float(roc_auc_score(y, s)),
        "question_weighted_roc_auc": float(roc_auc_score(y, s, sample_weight=weights)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1_follow": float(f1_score(y, pred, pos_label=FOLLOW)),
    }


def cluster_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: str = "balanced_accuracy",
    repetitions: int = 5000,
    seed: int = 20260710,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.where(groups == group)[0] for group in sampled])
        y = labels[indices]
        s = scores[indices]
        if len(np.unique(y)) < 2:
            continue
        pred = (s >= 0).astype(int)
        if metric == "balanced_accuracy":
            values.append(balanced_accuracy_score(y, pred))
        elif metric == "roc_auc":
            values.append(roc_auc_score(y, s))
        else:
            raise ValueError(metric)
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def metadata_scores(frame: pd.DataFrame, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    numerical = frame[
        [
            "injection_depth",
            "boundary_token_count",
            "cot_chars",
            "cot_lines",
            "cot_fraction_slashes",
            "cot_verify_mentions",
            "cot_question_mentions",
            "cot_wait_mentions",
        ]
    ].to_numpy(dtype=np.float64)
    numerical[:, 1:4] = np.log1p(numerical[:, 1:4])

    def fit_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
            ]
        )
        model.fit(
            numerical[train],
            labels[train],
            logit__sample_weight=group_row_weights(groups[train]),
        )
        return model.decision_function(numerical[test])

    return grouped_model_scores(frame, labels, groups, fit_predict)


def tfidf_scores(frame: pd.DataFrame, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    texts = frame["visible_text"].tolist()

    def fit_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        vectoriser = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=2500,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w/+=-]+\b",
        )
        train_x = vectoriser.fit_transform([texts[i] for i in train])
        test_x = vectoriser.transform([texts[i] for i in test])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        model.fit(train_x, labels[train], sample_weight=group_row_weights(groups[train]))
        return model.decision_function(test_x)

    return grouped_model_scores(frame, labels, groups, fit_predict)


def hidden_pca_logit_scores(x: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    folds = prepare_hidden_pca_folds(x, groups)
    return hidden_pca_scores_from_folds(folds, labels, groups, len(labels))


def prepare_hidden_pca_folds(x: np.ndarray, groups: np.ndarray) -> list[dict[str, Any]]:
    folds = []
    normalised = l2_normalise(x)
    for held_group in np.unique(groups):
        train = np.where(groups != held_group)[0]
        test = np.where(groups == held_group)[0]
        components = min(8, len(train) - 2, x.shape[1])
        pca = PCA(n_components=components, whiten=True, random_state=0)
        train_x = pca.fit_transform(normalised[train])
        test_x = pca.transform(normalised[test])
        folds.append({"train": train, "test": test, "train_x": train_x, "test_x": test_x})
    return folds


def hidden_pca_scores_from_folds(
    folds: list[dict[str, Any]], labels: np.ndarray, groups: np.ndarray, n_rows: int
) -> np.ndarray:
    scores = np.full(n_rows, np.nan, dtype=np.float64)
    for fold in folds:
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        model.fit(
            fold["train_x"],
            labels[fold["train"]],
            sample_weight=group_row_weights(groups[fold["train"]]),
        )
        scores[fold["test"]] = model.decision_function(fold["test_x"])
    return scores


def hidden_pca_cross_depth_scores(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    variants: np.ndarray,
    target_variant: str,
) -> np.ndarray:
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    normalised = l2_normalise(x)
    for held_group in np.unique(groups):
        train = np.where((groups != held_group) & (variants != target_variant))[0]
        test = np.where((groups == held_group) & (variants == target_variant))[0]
        if len(test) == 0:
            continue
        components = min(8, len(train) - 2, x.shape[1])
        pca = PCA(n_components=components, whiten=True, random_state=0)
        train_x = pca.fit_transform(normalised[train])
        test_x = pca.transform(normalised[test])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        model.fit(train_x, labels[train], sample_weight=group_row_weights(groups[train]))
        scores[test] = model.decision_function(test_x)
    return scores


def depth_prior_scores(frame: pd.DataFrame, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    variants = frame["variant"].to_numpy()
    scores = np.zeros(len(labels), dtype=np.float64)
    for held_group in np.unique(groups):
        train = groups != held_group
        test = groups == held_group
        for index in np.where(test)[0]:
            subset = labels[train & (variants == variants[index])]
            probability = (float(subset.sum()) + 1.0) / (len(subset) + 2.0)
            scores[index] = math.log(probability / (1.0 - probability))
    return scores


def question_trajectory_permutation(
    labels: np.ndarray,
    variants: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    permuted = labels.copy()
    coverage_strata: dict[tuple[str, ...], list[str]] = {}
    for group in np.unique(groups):
        coverage = tuple(sorted(variants[groups == group].tolist()))
        coverage_strata.setdefault(coverage, []).append(group)
    for coverage, target_groups in coverage_strata.items():
        if len(target_groups) < 2:
            continue
        source_groups = rng.permutation(target_groups)
        for target_group, source_group in zip(target_groups, source_groups):
            source = {
                variant: labels[(groups == source_group) & (variants == variant)][0]
                for variant in coverage
            }
            for variant in coverage:
                target = (groups == target_group) & (variants == variant)
                permuted[target] = source[variant]
    return permuted


def plot_save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    layer_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    cross_depth: pd.DataFrame,
    max_null_95: float,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#5a6470",
            "grid.color": "#dfe3e8",
            "grid.linewidth": 0.7,
        }
    )
    follow_colour = "#177e89"
    bypass_colour = "#c45a45"
    navy = "#315b7d"
    gold = "#c48a24"

    counts = (
        frame.assign(outcome=np.where(labels == FOLLOW, "FOLLOW", "BYPASS"))
        .groupby(["variant", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(VARIANT_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for outcome, colour in [("BYPASS", bypass_colour), ("FOLLOW", follow_colour)]:
        values = counts[outcome].to_numpy()
        bars = ax.bar(x, values, bottom=bottom, color=colour, width=0.62, label=outcome)
        for bar, value, base in zip(bars, values, bottom):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, base + value / 2, str(value), ha="center", va="center", color="white", fontweight="bold")
        bottom += values
    ax.set_xticks(x, [VARIANT_LABELS[v] for v in counts.index])
    ax.set_ylabel("Manually audited rows")
    ax.set_title("FOLLOW and BYPASS outcomes by injection depth")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plot_save(fig, figure_dir / "cohort-by-injection-depth")

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    plotted_layers = layer_metrics[layer_metrics["depth"] > 0]
    ax.plot(plotted_layers["depth"], plotted_layers["roc_auc"], color=navy, marker="o", markersize=3.2, linewidth=1.8)
    ax.axhline(0.5, color="#6f7780", linewidth=1.0, linestyle="--", label="Chance AUROC")
    ax.axhline(max_null_95, color=gold, linewidth=1.2, linestyle=":", label="95% max-layer permutation threshold")
    best = plotted_layers.iloc[plotted_layers["roc_auc"].argmax()]
    ax.scatter([best["depth"]], [best["roc_auc"]], s=52, color=follow_colour, zorder=4)
    ax.annotate(
        f"Best depth {int(best['depth'])}: {best['roc_auc']:.2f}",
        (best["depth"], best["roc_auc"]),
        xytext=(8, 10),
        textcoords="offset points",
    )
    ax.set_xlim(0, layer_metrics["depth"].max())
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Boundary representation depth (0 = embedding; 35 = final normalised state)")
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Question-held-out prediction across Gemma's representation depth")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plot_save(fig, figure_dir / "heldout-layer-curve")

    ordered = model_metrics.sort_values("balanced_accuracy")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.barh(ordered["display_name"], ordered["balanced_accuracy"], color=[navy if "Hidden" in name else "#8a949e" for name in ordered["display_name"]])
    for bar, value in zip(bars, ordered["balanced_accuracy"]):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    ax.axvline(0.5, color="#6f7780", linewidth=1.0, linestyle="--")
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("Question-held-out balanced accuracy")
    ax.set_title("Boundary-state probes against visible and structural baselines")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    plot_save(fig, figure_dir / "baseline-comparison")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    display = predictions.copy()
    display["outcome"] = np.where(display["label"] == FOLLOW, "FOLLOW", "BYPASS")
    display["depth_label"] = display["variant"].map(VARIANT_LABELS)
    rng = np.random.default_rng(7)
    positions = {label: i for i, label in enumerate(["50%", "90%", "Full CoT"])}
    for outcome, colour, marker in [("BYPASS", bypass_colour, "o"), ("FOLLOW", follow_colour, "^")]:
        subset = display[display["outcome"] == outcome]
        xs = np.asarray([positions[value] for value in subset["depth_label"]], dtype=float)
        xs += rng.uniform(-0.09, 0.09, len(xs))
        ax.scatter(xs, subset["hidden_pca_score"], color=colour, marker=marker, s=48, alpha=0.9, label=outcome, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#4d5660", linewidth=1.0)
    ax.set_xticks(range(3), ["50%", "90%", "Full CoT"])
    ax.set_ylabel("Held-out PCA-logit score (positive predicts FOLLOW)")
    ax.set_xlabel("Injection condition")
    ax.set_title("Final boundary-state scores on unseen questions")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plot_save(fig, figure_dir / "heldout-score-distribution")

    from scipy.stats import spearmanr

    correlation = spearmanr(
        predictions["boundary_delta_qstar_minus_q"], predictions["hidden_pca_score"]
    ).statistic
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for outcome, colour, marker in [("BYPASS", bypass_colour, "o"), ("FOLLOW", follow_colour, "^")]:
        subset = display[display["outcome"] == outcome]
        ax.scatter(
            subset["boundary_delta_qstar_minus_q"],
            subset["hidden_pca_score"],
            color=colour,
            marker=marker,
            s=54,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            label=outcome,
        )
    exceptional = display[display["item_id"] == "nat_lin3_013"].copy()
    exceptional["coordinate"] = list(
        zip(
            exceptional["boundary_delta_qstar_minus_q"].round(6),
            exceptional["hidden_pca_score"].round(6),
        )
    )
    for coordinate, subset in exceptional.groupby("coordinate"):
        depth_names = subset["depth_label"].tolist()
        depth_label = " / ".join(depth_names)
        x_value, y_value = coordinate
        offset = (-64, 16) if y_value > 0 else (18, 18)
        ax.annotate(
            f"nat_lin3_013\n{depth_label}",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#6f7780", "linewidth": 0.7},
        )
    ax.axvline(0, color="#6f7780", linewidth=1.0, linestyle="--")
    ax.axhline(0, color="#6f7780", linewidth=1.0, linestyle="--")
    ax.text(
        0.03,
        0.96,
        rf"Spearman $\rho$ = {correlation:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.set_xlabel(r"Candidate-continuation preference (Q* $-$ Q average log-probability)")
    ax.set_ylabel("Question-held-out hidden-state score")
    ax.set_title("Hidden state tracks later Q versus Q* preference")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plot_save(fig, figure_dir / "hidden-score-vs-candidate-preference")

    cm = confusion_matrix(labels, (predictions["hidden_pca_score"].to_numpy() >= 0).astype(int), labels=[BYPASS, FOLLOW])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax, linewidths=1, linecolor="white")
    ax.set_xticklabels(["BYPASS", "FOLLOW"])
    ax.set_yticklabels(["BYPASS", "FOLLOW"], rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Manual label")
    ax.set_title("Regularised final-state probe confusion matrix")
    plot_save(fig, figure_dir / "hidden-final-confusion-matrix")

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    bars = ax.bar([VARIANT_LABELS[v] for v in cross_depth["target_variant"]], cross_depth["balanced_accuracy"], color=["#657b8a", navy, follow_colour])
    for bar, value in zip(bars, cross_depth["balanced_accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    ax.axhline(0.5, color="#6f7780", linewidth=1.0, linestyle="--")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Cross-depth transfer with the test question excluded")
    ax.spines[["top", "right"]].set_visible(False)
    plot_save(fig, figure_dir / "cross-depth-generalisation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("Results/local_gemma_follow_bypass_probes/heldout_boundary_prediction/features"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Results/local_gemma_follow_bypass_probes/heldout_boundary_prediction/analysis"),
    )
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame, hidden, layer_names = load_features(args.feature_dir)
    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["item_id"].to_numpy()
    variants = frame["variant"].to_numpy()
    n, depths, _ = hidden.shape

    normalised = hidden / np.clip(np.linalg.norm(hidden, axis=2, keepdims=True), 1e-12, None)
    similarities = np.einsum("ndh,mdh->dnm", normalised, normalised)
    layer_scores = np.stack(
        [centroid_scores_from_similarity(similarities[d], labels, groups) for d in range(depths)],
        axis=1,
    )
    layer_rows = []
    for depth in range(depths):
        metrics = metric_bundle(labels, layer_scores[:, depth], groups)
        layer_rows.append({"depth": depth, "layer_name": layer_names[depth], **metrics})
    layer_metrics = pd.DataFrame(layer_rows)

    hidden_final_score = layer_scores[:, -1]
    pca_folds = prepare_hidden_pca_folds(hidden[:, -1, :], groups)
    hidden_pca_score = hidden_pca_scores_from_folds(pca_folds, labels, groups, n)
    metadata_score = metadata_scores(frame, labels, groups)
    text_score = tfidf_scores(frame, labels, groups)
    depth_score = depth_prior_scores(frame, labels, groups)
    delta_score = frame["boundary_delta_qstar_minus_q"].to_numpy(dtype=float)
    majority_score = np.full(n, 1.0)

    score_map = {
        "majority_follow": majority_score,
        "injection_depth_prior": depth_score,
        "visible_structural": metadata_score,
        "visible_tfidf": text_score,
        "boundary_likelihood": delta_score,
        "hidden_final_centroid": hidden_final_score,
        "hidden_final_pca_logit": hidden_pca_score,
    }
    display_names = {
        "majority_follow": "Majority FOLLOW",
        "injection_depth_prior": "Injection-depth prior",
        "visible_structural": "Visible structural features",
        "visible_tfidf": "Visible CoT TF-IDF",
        "boundary_likelihood": "Boundary likelihood Q*−Q",
        "hidden_final_centroid": "Hidden final state (centroid)",
        "hidden_final_pca_logit": "Hidden final state (PCA-logit)",
    }
    model_rows = []
    for name, score in score_map.items():
        metrics = metric_bundle(labels, score, groups)
        ci_low, ci_high = cluster_bootstrap_ci(labels, score, groups)
        model_rows.append(
            {
                "model": name,
                "display_name": display_names[name],
                "balanced_accuracy_ci_low": ci_low,
                "balanced_accuracy_ci_high": ci_high,
                **metrics,
            }
        )
    model_metrics = pd.DataFrame(model_rows)

    rng = np.random.default_rng(20260710)
    final_null_auc = np.zeros(args.permutations)
    max_layer_null_auc = np.zeros(args.permutations)
    pca_null_balanced = np.zeros(args.permutations)
    pca_null_auc = np.zeros(args.permutations)
    direct_null_balanced = np.zeros(args.permutations)
    direct_null_auc = np.zeros(args.permutations)
    direct_score = frame["boundary_delta_qstar_minus_q"].to_numpy(dtype=np.float64)
    for repetition in range(args.permutations):
        permuted = question_trajectory_permutation(labels, variants, groups, rng)
        permutation_auc = []
        for depth in range(1, depths):
            score = centroid_scores_from_similarity(similarities[depth], permuted, groups)
            permutation_auc.append(roc_auc_score(permuted, score))
        final_null_auc[repetition] = permutation_auc[-1]
        max_layer_null_auc[repetition] = max(permutation_auc)
        pca_score = hidden_pca_scores_from_folds(pca_folds, permuted, groups, n)
        pca_null_balanced[repetition] = balanced_accuracy_score(permuted, pca_score >= 0)
        pca_null_auc[repetition] = roc_auc_score(permuted, pca_score)
        direct_null_balanced[repetition] = balanced_accuracy_score(permuted, direct_score >= 0)
        direct_null_auc[repetition] = roc_auc_score(permuted, direct_score)

    observed_final_auc = float(layer_metrics.iloc[-1]["roc_auc"])
    non_embedding = layer_metrics[layer_metrics["depth"] > 0]
    observed_best_auc = float(non_embedding["roc_auc"].max())
    primary_auc_p = float((1 + np.sum(final_null_auc >= observed_final_auc)) / (args.permutations + 1))
    corrected_best_p = float((1 + np.sum(max_layer_null_auc >= observed_best_auc)) / (args.permutations + 1))
    max_null_95 = float(np.quantile(max_layer_null_auc, 0.95))
    observed_pca_balanced = float(balanced_accuracy_score(labels, hidden_pca_score >= 0))
    observed_pca_auc = float(roc_auc_score(labels, hidden_pca_score))
    pca_balanced_p = float((1 + np.sum(pca_null_balanced >= observed_pca_balanced)) / (args.permutations + 1))
    pca_auc_p = float((1 + np.sum(pca_null_auc >= observed_pca_auc)) / (args.permutations + 1))
    observed_direct_balanced = float(balanced_accuracy_score(labels, direct_score >= 0))
    observed_direct_auc = float(roc_auc_score(labels, direct_score))
    direct_balanced_p = float(
        (1 + np.sum(direct_null_balanced >= observed_direct_balanced)) / (args.permutations + 1)
    )
    direct_auc_p = float((1 + np.sum(direct_null_auc >= observed_direct_auc)) / (args.permutations + 1))
    permutation_summary = {
        "permutations": args.permutations,
        "scheme": (
            "complete question-level label trajectories shuffled between questions with matching "
            "injection-depth coverage; question-level folds retained"
        ),
        "observed_final_centroid_auc": observed_final_auc,
        "final_centroid_auc_p": primary_auc_p,
        "observed_pca_logit_balanced_accuracy": observed_pca_balanced,
        "pca_logit_balanced_accuracy_p": pca_balanced_p,
        "observed_pca_logit_auc": observed_pca_auc,
        "pca_logit_auc_p": pca_auc_p,
        "observed_candidate_preference_balanced_accuracy": observed_direct_balanced,
        "candidate_preference_balanced_accuracy_p": direct_balanced_p,
        "observed_candidate_preference_auc": observed_direct_auc,
        "candidate_preference_auc_p": direct_auc_p,
        "observed_best_layer_auc": observed_best_auc,
        "best_layer_depth": int(non_embedding.iloc[non_embedding["roc_auc"].argmax()]["depth"]),
        "max_layer_corrected_p": corrected_best_p,
        "max_layer_null_95": max_null_95,
        "embedding_excluded_reason": "the <channel|> input embedding is exactly identical across all 36 rows",
    }

    cross_rows = []
    for target_variant in VARIANT_ORDER:
        score = hidden_pca_cross_depth_scores(
            hidden[:, -1, :],
            labels,
            groups,
            variants,
            target_variant,
        )
        metrics = metric_bundle(labels, score, groups)
        cross_rows.append({"target_variant": target_variant, **metrics})
    cross_depth = pd.DataFrame(cross_rows)

    predictions = frame[
        ["intervention_id", "item_id", "variant", "manual_label", "label", "boundary_delta_qstar_minus_q"]
    ].copy()
    for name, score in score_map.items():
        predictions[f"{name}_score"] = score
        predictions[f"{name}_prediction"] = (score >= 0).astype(int)
    predictions["hidden_final_score"] = hidden_final_score
    predictions["hidden_final_prediction"] = (hidden_final_score >= 0).astype(int)
    predictions["hidden_final_correct"] = predictions["hidden_final_prediction"] == labels
    predictions["hidden_pca_score"] = hidden_pca_score
    predictions["hidden_pca_prediction"] = (hidden_pca_score >= 0).astype(int)
    predictions["hidden_pca_correct"] = predictions["hidden_pca_prediction"] == labels

    layer_metrics.to_csv(args.out_dir / "layer_metrics.csv", index=False)
    model_metrics.to_csv(args.out_dir / "model_metrics.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    cross_depth.to_csv(args.out_dir / "cross_depth_metrics.csv", index=False)
    (args.out_dir / "permutation_summary.json").write_text(
        json.dumps(permutation_summary, indent=2), encoding="utf-8"
    )
    summary = {
        "cohort": {
            "rows": int(n),
            "questions": int(len(np.unique(groups))),
            "follow": int(labels.sum()),
            "bypass": int((labels == BYPASS).sum()),
        },
        "primary_hidden_final_pca_logit": metric_bundle(labels, hidden_pca_score, groups),
        "secondary_hidden_final_centroid": metric_bundle(labels, hidden_final_score, groups),
        "permutation": permutation_summary,
        "best_layer_by_auc_excluding_embedding": non_embedding.iloc[non_embedding["roc_auc"].argmax()].to_dict(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figures(
        frame,
        labels,
        groups,
        layer_metrics,
        model_metrics,
        predictions,
        cross_depth,
        max_null_95,
        args.out_dir / "figures",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
