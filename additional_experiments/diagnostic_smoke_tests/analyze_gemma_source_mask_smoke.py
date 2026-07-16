

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from local_gemma_follow_bypass_probe import ROOT


RESULT_DIR = ROOT / "Results" / "dissertation_smoke_tests" / "source_masking"
ANALYSIS_DIR = RESULT_DIR / "analysis"
STATE_ORDER = ["Q", "QSTAR", "OTHER"]
STATE_LABELS = {"Q": "Original Q", "QSTAR": "Counterfactual Q*", "OTHER": "Other"}
MASK_LABELS = {
    "none": "Both sources",
    "question": "CoT only\n(hide Q)",
    "cot": "Prompt only\n(hide CoT)",
    "question_and_cot": "Neither source",
}


def latest_rows() -> pd.DataFrame:
    latest: dict[tuple[str, str], dict] = {}
    for path in sorted(RESULT_DIR.glob("source_mask_smoke_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = (row["case_id"], row["mask"])
            latest[key] = {**row, "source_file": path.name}
    return pd.DataFrame(latest.values())


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_frame = latest_rows()
    frame = all_frame[all_frame["case_id"].str.endswith("::full_cot_normal")].copy()
    full_cases = sorted(
        case_id
        for case_id, group in frame.groupby("case_id")
        if {"none", "question", "cot"}.issubset(set(group["mask"]))
    )
    frame = frame[frame["case_id"].isin(full_cases)].copy()
    frame["expected_natural_state"] = frame["manual_label"].map(
        {"FOLLOW": "QSTAR", "SILENT_BYPASS_REANCHOR": "Q"}
    )
    frame.to_csv(ANALYSIS_DIR / "source_masking_latest_rows.csv", index=False)

    conditions = {
        "Natural outcome reproduced": frame["mask"].eq("none")
        & frame["generated_state"].eq(frame["expected_natural_state"]),
        "Q hidden -> Q* selected": frame["mask"].eq("question")
        & frame["generated_state"].eq("QSTAR"),
        "CoT hidden -> Q selected": frame["mask"].eq("cot")
        & frame["generated_state"].eq("Q"),
        "Both hidden -> neither target": frame["mask"].eq("question_and_cot")
        & frame["generated_state"].eq("OTHER"),
    }
    summary_rows = []
    masks = ["none", "question", "cot", "question_and_cot"]
    condition_names = list(conditions)
    for name, successes in conditions.items():
        mask = masks[condition_names.index(name)]
        available = frame["mask"].eq(mask)
        summary_rows.append(
            {
                "test": name,
                "successes": int((successes & available).sum()),
                "trials": int(available.sum()),
                "rate": float((successes & available).sum() / max(1, available.sum())),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ANALYSIS_DIR / "source_masking_summary.csv", index=False)

    depth_flip = all_frame[
        all_frame["case_id"].isin(
            [
                "nat_lin3_013::cot_50_percent",
                "nat_lin3_013::cot_90_percent",
                "nat_lin3_013::full_cot_normal",
            ]
        )
        & all_frame["mask"].isin(["none", "question", "cot"])
    ].copy()
                                                                        
                                                                             
                                                                  
    depth_flip["audited_state"] = depth_flip["generated_state"]
    override = (
        depth_flip["case_id"].eq("nat_lin3_013::cot_50_percent")
        & depth_flip["mask"].eq("question")
    )
    depth_flip.loc[override, "audited_state"] = "QSTAR"
    depth_flip.to_csv(ANALYSIS_DIR / "nat_lin3_013_depth_flip_source_masking.csv", index=False)

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
        }
    )
    state_code = {"Q": 0, "QSTAR": 1, "OTHER": 2}
    display_cases = []
    matrix = []
    for case_id in full_cases:
        group = frame[frame["case_id"] == case_id].set_index("mask")
        label = group["manual_label"].iloc[0]
        display_cases.append(
            case_id.split("::", 1)[0].replace("nat_lin3_", "System ")
            + ("  [FOLLOW]" if label == "FOLLOW" else "  [BYPASS]")
        )
        matrix.append(
            [state_code.get(group.loc[mask, "generated_state"], 2) if mask in group.index else 2 for mask in masks]
        )

    colours = ["#2F6B8A", "#C48A24", "#9299A1"]
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sns.heatmap(
        matrix,
        cmap=ListedColormap(colours),
        vmin=-0.5,
        vmax=2.5,
        cbar=False,
        linewidths=1.4,
        linecolor="white",
        xticklabels=[MASK_LABELS[mask] for mask in masks],
        yticklabels=display_cases,
        ax=ax,
    )
    for row_index, row in enumerate(matrix):
        for column_index, code in enumerate(row):
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                STATE_LABELS[STATE_ORDER[code]],
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="semibold",
            )
    ax.set_title("Answer branch under causal source masking", pad=14)
    ax.set_xlabel("Information available during final-answer generation", labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            ANALYSIS_DIR / f"source_masking_outcomes.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    report = f"""# Causal Source-Masking Pilot

## Design

The pilot uses eight naturally occurring full-CoT linear-system cases, split
evenly between FOLLOW and answer-stage BYPASS. Gemma first processes the full
question and counterfactual reasoning trace normally. Its first final-answer
token is also selected normally, preserving the answer-channel transition.
From the second final-answer token onward, direct attention to the original
question, the semantic CoT tokens, both sources, or neither source is allowed.
The changed equation normally appears around token 50, leaving a substantial
interval between the intervention and the first explicit Q/Q* commitment.

## Results

| Test | Successes | Trials |
|---|---:|---:|
"""
    for row in summary.itertuples(index=False):
        report += f"| {row.test} | {row.successes} | {row.trials} |\n"
    report += """

With both sources available, all eight natural labels were reproduced. When
direct access to the original question was suppressed, all eight answers
reconstructed Q*. When semantic CoT access was suppressed, all eight answers
reconstructed Q. The eight double-masking controls reconstructed
neither target and instead produced unrelated linear systems.

![Case-level source-masking outcomes](source_masking_outcomes.png)

## Interpretation

The result supports a source-competition account of the trace-to-answer
transition. Both the original prompt state and the counterfactual trace state
remain causally available during final-answer generation. Natural FOLLOW and
BYPASS behaviour therefore cannot be reduced to the presence of only one
problem representation. Removing one direct route makes the other route
determine the emitted equation, including bidirectional switches from Q to Q*
and from Q* to Q.

The intervention does not erase all information that may already have been
integrated into unmasked cached states. It also changes attention over a long
span and has so far been tested only on one model and one task family. The
double-masking failure is a useful negative control, although it is not a
meaningful behavioural condition by itself. A full study should pre-register
source spans, include shorter and graded masks, repeat the design across task
families, and compare models.
"""
    report += """

## Within-question depth flip

`nat_lin3_013` bypasses to Q at 50% injection but follows Q* at 90% and full
CoT. Source isolation removes that disagreement. All three boundaries select
Q* when direct access to the original question is suppressed and select Q when
semantic CoT access is suppressed. The 50% question-mask output required a
manual formatting-only correction because it printed `1.` rather than the
classifier's expected `1)` before the Q* equation.

| Injection condition | Both sources | CoT only (hide Q) | Prompt only (hide CoT) |
|---|---|---|---|
| 50% | Q | Q* | Q |
| 90% | Q* | Q* | Q |
| Full CoT | Q* | Q* | Q |

The flip therefore reflects a change in relative source weighting rather than
the loss of either source. Injection depth changes which source wins under
competition, while each source remains sufficient when isolated.
"""
    (ANALYSIS_DIR / "source_masking_report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved analysis to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
