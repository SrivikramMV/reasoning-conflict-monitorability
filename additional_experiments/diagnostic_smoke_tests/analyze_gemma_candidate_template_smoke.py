

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from local_gemma_follow_bypass_probe import ROOT


RESULT_DIR = (
    ROOT
    / "Results"
    / "dissertation_smoke_tests"
    / "candidate_template_robustness"
)
ANALYSIS_DIR = RESULT_DIR / "analysis"
TEMPLATE_ORDER = [
    "standard_system",
    "plain_restatement",
    "active_system",
    "verification",
    "minimal",
]
TEMPLATE_LABELS = {
    "standard_system": "Standard\nsystem",
    "plain_restatement": "Plain\nrestatement",
    "active_system": "Active\nsystem",
    "verification": "Verification\nopening",
    "minimal": "No supplied\nopening",
}


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs: list[list[dict]] = []
    for path in sorted(RESULT_DIR.glob("candidate_template_smoke_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            row["source_file"] = path.name
        runs.append(rows)
    if not runs:
        raise FileNotFoundError(f"No candidate-template results in {RESULT_DIR}")

    scoring_rows = max(runs, key=len)
    generated_rows = [
        row
        for rows in runs
        for row in rows
        if row.get("generated_state") not in {None, "NOT_RUN"}
    ]
    return pd.DataFrame(scoring_rows), pd.DataFrame(generated_rows)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    frame, generated = load_runs()
    frame["short_id"] = frame["item_id"].str.replace("nat_lin3_", "", regex=False)
    frame["branch"] = frame["manual_label"].map(
        {"FOLLOW": "FOLLOW", "SILENT_BYPASS_REANCHOR": "BYPASS"}
    )

    standard = (
        frame[frame["template"].eq("standard_system")]
        .set_index("case_id")["delta_avg_qstar_minus_q"]
        .to_dict()
    )
    case_order = sorted(
        frame["case_id"].unique(),
        key=lambda case_id: (abs(standard[case_id]), standard[case_id]),
    )
    matrix = (
        frame.pivot(
            index="case_id", columns="template", values="delta_avg_qstar_minus_q"
        )
        .loc[case_order, TEMPLATE_ORDER]
    )
    metadata = frame.drop_duplicates("case_id").set_index("case_id")
    labels = [
        f"System {metadata.loc[case_id, 'short_id']}  "
        f"[{metadata.loc[case_id, 'branch']}]"
        for case_id in case_order
    ]

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
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    norm = TwoSlopeNorm(vmin=-0.65, vcenter=0.0, vmax=0.55)
    sns.heatmap(
        matrix,
        cmap=sns.diverging_palette(235, 38, s=70, l=52, as_cmap=True),
        norm=norm,
        cbar_kws={"label": "Average log-probability margin  Q* - Q", "shrink": 0.82},
        linewidths=1.3,
        linecolor="white",
        xticklabels=[TEMPLATE_LABELS[name] for name in TEMPLATE_ORDER],
        yticklabels=labels,
        ax=ax,
    )
    for row_index, values in enumerate(matrix.to_numpy()):
        for column_index, value in enumerate(values):
            branch = "Q*" if value > 0 else "Q"
            colour = "white" if abs(value) >= 0.22 else "#202428"
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{branch}\n{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="semibold",
                color=colour,
            )
    ax.set_title("Weak trace preferences change with the answer opening", pad=14)
    ax.set_xlabel("Answer opening supplied after the reasoning trace", labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            ANALYSIS_DIR / f"candidate_template_robustness.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    frame.to_csv(ANALYSIS_DIR / "candidate_template_scores.csv", index=False)
    if not generated.empty:
        generated.to_csv(ANALYSIS_DIR / "candidate_template_generations.csv", index=False)

    bypass = frame[frame["branch"].eq("BYPASS")]
    stable_bypass = int(
        bypass.groupby("case_id")["preference"].apply(lambda x: set(x) == {"Q"}).sum()
    )
    follow = frame[frame["branch"].eq("FOLLOW")]
    stable_follow = int(
        follow.groupby("case_id")["preference"].apply(lambda x: set(x) == {"QSTAR"}).sum()
    )
    weak_follow = int(
        follow.groupby("case_id")["preference"]
        .apply(lambda x: set(x) == {"Q", "QSTAR"})
        .sum()
    )

    report = f"""# Candidate Preference Across Answer Openings

## Design

Eight complete-CoT three-variable linear-system cases were selected from the
original candidate-margin distribution. Four had produced FOLLOW and four had
produced answer-stage bypass under greedy decoding. Matched Q and Q* equation
continuations were scored after five neutral answer openings while the prompt,
counterfactual trace and answer boundary remained fixed.

## Results

- All {stable_bypass}/4 bypass cases preferred Q under every opening.
- {stable_follow}/4 FOLLOW cases preferred Q* under every opening.
- The remaining {weak_follow}/4 FOLLOW cases preferred Q* only under the
  standard system-solving opening and preferred Q under the other four forced
  openings.
- In the six-case generation check, outputs under supplied preambles agreed
  with the candidate preference. With no supplied opening, both weak FOLLOW
  cases generated a familiar system-solving introduction of their own and then
  reconstructed Q*.

![Candidate preference by answer opening](candidate_template_robustness.png)

## Interpretation

The shared candidate readout predicts the branch expressed under that readout,
although weak preferences do not represent an immutable decision at the bare
answer boundary. The opening of the final answer can amplify or redirect a
near-balanced prompt-versus-trace state. Large margins remain stable across
the tested openings, which motivates treating margin as a measure of branch
susceptibility and testing that interpretation with repeated sampling.

The cases were deliberately selected rather than sampled at random, all came
from one symbolic task family, and each candidate score depends on a known Q/Q*
pair. The result is therefore a robustness diagnostic for the benchmark rather
than a general-purpose deployment monitor.
"""
    (ANALYSIS_DIR / "candidate_template_robustness_report.md").write_text(
        report, encoding="utf-8"
    )
    print(
        f"Stable bypass: {stable_bypass}/4; stable follow: {stable_follow}/4; "
        f"opening-sensitive follow: {weak_follow}/4"
    )
    print(f"Saved analysis to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
