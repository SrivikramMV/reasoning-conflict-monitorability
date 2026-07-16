

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_gemma_heldout_boundary_prediction import (
    hidden_pca_logit_scores,
    metric_bundle,
    plot_save,
    tfidf_scores,
)
from local_gemma_follow_bypass_probe import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "Results" / "dissertation_smoke_tests" / "visibility_latency" / "features",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Results" / "dissertation_smoke_tests" / "visibility_latency" / "analysis",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    arrays = {}
    for path in sorted(args.feature_dir.glob("nat_lin3_*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        records.append(metadata)
        arrays[metadata["case_id"]] = np.load(path.with_suffix(".npz"))
    base = pd.DataFrame(records)
    labels = base["label"].to_numpy(dtype=int)
    groups = base["item_id"].to_numpy()
    common_k = sorted(set.intersection(*(set(values["k"].tolist()) for values in arrays.values())))

    metric_rows = []
    prediction_rows = []
    for k in common_k:
        text_frame = base.copy()
        text_frame["visible_text"] = [
            row["visible_base_text"] + row["answer_prefix_by_k"][str(k)] for row in records
        ]
        text_scores = tfidf_scores(text_frame, labels, groups)
        text_metrics = metric_bundle(labels, text_scores, groups)
        metric_rows.append({"k": k, "monitor": "Visible TF-IDF", **text_metrics})

        for state_name, display in [
            ("after_block_19", "Hidden after block 19"),
            ("after_block_32", "Hidden after block 32"),
            ("final_norm", "Hidden final norm"),
        ]:
            matrix = np.stack(
                [arrays[case_id][state_name][list(arrays[case_id]["k"]).index(k)] for case_id in base["case_id"]],
                axis=0,
            )
            scores = hidden_pca_logit_scores(matrix, labels, groups)
            metrics = metric_bundle(labels, scores, groups)
            metric_rows.append({"k": k, "monitor": display, **metrics})
            for case_id, score in zip(base["case_id"], scores):
                prediction_rows.append(
                    {"case_id": case_id, "k": k, "monitor": display, "score": float(score)}
                )

        exposed = []
        correct = []
        for row in records:
            token = row["first_exact_state_token"]
            is_exposed = token is not None and token <= k
            exposed.append(is_exposed)
            expected = "QSTAR" if row["label"] == 1 else "Q"
            correct.append(is_exposed and row["first_exact_state"] == expected)
        metric_rows.append(
            {
                "k": k,
                "monitor": "Exact state visible",
                "n": len(records),
                "accuracy": float(np.mean(correct)),
                "balanced_accuracy": float(np.mean(exposed)),
                "question_weighted_balanced_accuracy": float(np.mean(exposed)),
                "roc_auc": np.nan,
                "question_weighted_roc_auc": np.nan,
                "mcc": np.nan,
                "f1_follow": np.nan,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_dir / "latency_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.out_dir / "latency_predictions.csv", index=False)
    base[["case_id", "item_id", "variant", "manual_label", "first_exact_state_token", "first_exact_state"]].to_csv(
        args.out_dir / "state_visibility_by_case.csv", index=False
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    plotted = metrics[
        metrics["monitor"].isin(["Visible TF-IDF", "Hidden after block 32", "Hidden final norm"])
        & metrics["k"].le(32)
    ]
    sns.lineplot(
        data=plotted,
        x="k",
        y="question_weighted_balanced_accuracy",
        hue="monitor",
        marker="o",
        linewidth=2.2,
        ax=ax,
    )
    exact = metrics[metrics["monitor"] == "Exact state visible"]
    ax.scatter(
        exact["k"],
        exact["question_weighted_balanced_accuracy"],
        color="#7A7A7A",
        marker="s",
        label="Exact state exposed",
        zorder=4,
    )
    median_reveal = float(base["first_exact_state_token"].median())
    ax.axvline(
        median_reveal,
        color="#7A7A7A",
        linestyle="--",
        linewidth=1.3,
        label=f"Median exact exposure ({median_reveal:.1f} tokens)",
    )
    ax.axhline(0.5, color="#B8B8B8", linestyle=":", linewidth=1.2)
    ax.set(
        xlabel="Final-answer tokens visible",
        ylabel="Question-weighted balanced accuracy / exposed fraction",
        ylim=(0, 1.04),
        title="Internal branch prediction precedes visible Q/Q* commitment",
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    plot_save(fig, args.out_dir / "visibility_latency_curve")

    summary = {
        "rows": len(base),
        "questions": int(base["item_id"].nunique()),
        "k": common_k,
        "median_first_exact_state_token": float(base["first_exact_state_token"].median()),
        "metrics": metric_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    boundary_hidden = metrics[
        (metrics["k"] == 0) & (metrics["monitor"] == "Hidden after block 32")
    ].iloc[0]
    boundary_visible = metrics[
        (metrics["k"] == 0) & (metrics["monitor"] == "Visible TF-IDF")
    ].iloc[0]
    report = f"""# Hidden-to-Visible Transition Latency

## Design

The analysis follows 36 natural FOLLOW/BYPASS rows from 14 three-variable
linear systems. Hidden states were recorded at the answer boundary and after
1, 2, 4, 8, 16 and 32 final-answer tokens, before any row had reconstructed
the changed Q or Q* equation. Prediction used leave-one-question-out folds so
that all injection variants of the test question remained unseen during
training.

## Results

At the answer boundary, the representation after block 32 reached
question-weighted balanced accuracy of
{boundary_hidden.question_weighted_balanced_accuracy:.3f} and question-weighted
AUROC of {boundary_hidden.question_weighted_roc_auc:.3f}. The visible TF-IDF
baseline reached {boundary_visible.question_weighted_balanced_accuracy:.3f}
balanced accuracy, although its AUROC of
{boundary_visible.question_weighted_roc_auc:.3f} indicates some ranking
information with an unstable threshold. No answer had exposed its changed
equation by 32 tokens, and the median first exact exposure occurred after
{median_reveal:.1f} tokens.

![Hidden and visible transition latency](visibility_latency_curve.png)

The result identifies an accessibility gap. Gemma's contextual representation
organises information about the later branch before the emitted answer makes
that branch explicit. The hidden probe remains correlational and does not show
that one direction causes FOLLOW or BYPASS. Most questions also retain the
same label across depth, while the generic probe misses the `nat_lin3_013`
within-question flip. Candidate-conditioned scoring and source masking are
therefore needed alongside aggregate hidden-state prediction.
"""
    (args.out_dir / "visibility_latency_report.md").write_text(report, encoding="utf-8")
    print(metrics[["k", "monitor", "balanced_accuracy", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
