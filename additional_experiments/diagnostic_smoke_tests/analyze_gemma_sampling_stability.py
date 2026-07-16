

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from local_gemma_follow_bypass_probe import ROOT


RESULT_DIR = ROOT / "Results" / "dissertation_smoke_tests" / "sampling_stability"
ANALYSIS_DIR = RESULT_DIR / "analysis"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(RESULT_DIR.glob("sampling_stability_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame(rows).drop_duplicates(
        ["case_id", "temperature", "top_p", "seed", "sample_index"], keep="last"
    )
    frame.to_csv(ANALYSIS_DIR / "sampled_outputs.csv", index=False)

    summary_rows = []
    for case_id, group in frame.groupby("case_id"):
        expected = "QSTAR" if group["manual_label"].iloc[0] == "FOLLOW" else "Q"
        counts = group["sampled_state"].value_counts()
        q_count = int(counts.get("Q", 0))
        qstar_count = int(counts.get("QSTAR", 0))
        other_count = int(counts.get("OTHER", 0))
        n = len(group)
        valid = q_count + qstar_count
        branch_prob = qstar_count / valid if valid else np.nan
        if valid and branch_prob not in {0.0, 1.0}:
            entropy = -(
                branch_prob * np.log2(branch_prob)
                + (1.0 - branch_prob) * np.log2(1.0 - branch_prob)
            )
        else:
            entropy = 0.0
        summary_rows.append(
            {
                "case_id": case_id,
                "item_id": case_id.split("::", 1)[0],
                "manual_label": group["manual_label"].iloc[0],
                "expected_greedy_branch": expected,
                "margin": float(group["boundary_margin_qstar_minus_q"].iloc[0]),
                "absolute_margin": abs(float(group["boundary_margin_qstar_minus_q"].iloc[0])),
                "samples": n,
                "Q": q_count,
                "QSTAR": qstar_count,
                "OTHER": other_count,
                "greedy_branch_reproduction_rate": float((group["sampled_state"] == expected).mean()),
                "conditional_branch_entropy_bits": float(entropy),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("absolute_margin")
    summary.to_csv(ANALYSIS_DIR / "sampling_stability_summary.csv", index=False)
    rho_stability, p_stability = spearmanr(
        summary["absolute_margin"], summary["greedy_branch_reproduction_rate"]
    )
    rho_entropy, p_entropy = spearmanr(
        summary["absolute_margin"], summary["conditional_branch_entropy_bits"]
    )

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "grid.color": "#E0E4E8",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.0, 1.35]})
    colour_map = {"FOLLOW": "#C48A24", "SILENT_BYPASS_REANCHOR": "#2F6B8A"}
    marker_map = {"FOLLOW": "^", "SILENT_BYPASS_REANCHOR": "o"}
    for label, group in summary.groupby("manual_label"):
        axes[0].scatter(
            group["absolute_margin"],
            group["greedy_branch_reproduction_rate"],
            s=76,
            color=colour_map[label],
            marker=marker_map[label],
            edgecolor="white",
            linewidth=0.7,
            label="FOLLOW" if label == "FOLLOW" else "BYPASS",
            zorder=3,
        )
    annotation_offsets = {
        "005": (-24, -20),
        "009": (-30, 8),
        "008": (5, -20),
        "011": (5, 8),
    }
    for row in summary.itertuples(index=False):
        short_id = row.item_id.replace("nat_lin3_", "")
        axes[0].annotate(
            short_id,
            (row.absolute_margin, row.greedy_branch_reproduction_rate),
            xytext=annotation_offsets.get(short_id, (4, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.3, 1.04)
    axes[0].set_xlabel("Absolute candidate margin |Q* - Q|")
    axes[0].set_ylabel("Sampled reproduction of greedy branch")
    axes[0].set_title("Margin predicts branch stability")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].text(
        0.03,
        0.06,
        rf"Spearman $\rho$ = {rho_stability:.2f}",
        transform=axes[0].transAxes,
        fontsize=9,
    )
    axes[0].spines[["top", "right"]].set_visible(False)

    ordered = summary.reset_index(drop=True)
    x = np.arange(len(ordered))
    bottom = np.zeros(len(ordered))
    for state, colour, label in [
        ("Q", "#2F6B8A", "Q"),
        ("QSTAR", "#C48A24", "Q*"),
        ("OTHER", "#9299A1", "Other"),
    ]:
        values = ordered[state].to_numpy()
        axes[1].bar(x, values, bottom=bottom, width=0.68, color=colour, label=label)
        bottom += values
    axes[1].set_xticks(x, [value.replace("nat_lin3_", "") for value in ordered["item_id"]])
    axes[1].set_xlabel("Case, ordered by absolute margin")
    axes[1].set_ylabel("Samples")
    axes[1].set_title("Low-margin cases occupy both answer branches")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2.4)
    for suffix in ("png", "svg"):
        fig.savefig(
            ANALYSIS_DIR / f"sampling_stability_by_margin.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    report = f"""# Candidate Margin and Sampled Branch Stability

## Design

Eight full-CoT linear-system cases were selected before sampling to span
near-zero, moderate and large Q* minus Q candidate margins, with FOLLOW and
BYPASS represented at each end of the range. Each fixed boundary received 12
independent continuations at temperature 0.7 and top-p 0.95. The first
reconstructed changed equation classified each continuation as Q, Q* or other.

## Results

| Case | Greedy label | Margin | Q | Q* | Other | Greedy branch reproduced |
|---|---|---:|---:|---:|---:|---:|
"""
    for row in summary.itertuples(index=False):
        report += (
            f"| `{row.item_id}` | {'FOLLOW' if row.manual_label == 'FOLLOW' else 'BYPASS'} "
            f"| {row.margin:+.5f} | {row.Q} | {row.QSTAR} | {row.OTHER} "
            f"| {row.greedy_branch_reproduction_rate:.3f} |\n"
        )
    report += f"""

Absolute margin correlates strongly with reproduction of the greedy branch
(Spearman rho = {rho_stability:.3f}, nominal p = {p_stability:.4f}) and
negatively with conditional Q/Q* branch entropy (rho = {rho_entropy:.3f},
nominal p = {p_entropy:.4f}). The two near-zero cases occupy both branches,
the moderate cases flip occasionally, and all four high-margin cases remain
12/12 stable.

![Sampling stability by candidate margin](sampling_stability_by_margin.png)

## Interpretation

The deterministic taxonomy records the path selected under the benchmark's
greedy decoding policy, while the candidate margin measures how securely that
path is preferred. A FOLLOW or BYPASS label near zero should therefore be
treated as a fragile outcome rather than a categorical item property. The
margin supplies a useful susceptibility dimension that complements the
correction-aware taxonomy without replacing it.

The sample contains only eight deliberately stratified cases, and the 12
continuations within a case do not create eight new independent questions.
The nominal correlation p-values are descriptive rather than confirmatory.
Full validation should pre-register margin bins, use more cases, repeat several
sampling temperatures and preserve the greedy condition as the primary
benchmark result.
"""
    (ANALYSIS_DIR / "sampling_stability_report.md").write_text(report, encoding="utf-8")
    metrics = {
        "cases": len(summary),
        "samples_per_case": sorted(summary["samples"].unique().tolist()),
        "spearman_abs_margin_vs_greedy_reproduction": float(rho_stability),
        "nominal_p_stability": float(p_stability),
        "spearman_abs_margin_vs_branch_entropy": float(rho_entropy),
        "nominal_p_entropy": float(p_entropy),
    }
    (ANALYSIS_DIR / "sampling_stability_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
