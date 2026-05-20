import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_B_DIR = (
    ROOT
    / "Results"
    / "Test 4 - 300 questions all CoT"
    / "Stage B results"
    / "Gemma_CoT_Stage_B"
)
ANALYSIS_DIR = STAGE_B_DIR / "analysis"
RAW_PATH = STAGE_B_DIR / "stage2_stage_b_gemma4_e2b.jsonl"
LABEL_PATH = ANALYSIS_DIR / "test4_stage_b_taxonomy_labels.pretty.json"

OUT_JSON = ANALYSIS_DIR / "test4_stage_b_taxonomy_labels_AUDITED.pretty.json"
OUT_CSV = ANALYSIS_DIR / "test4_stage_b_taxonomy_labels_AUDITED.csv"
OUT_REPORT = ANALYSIS_DIR / "test4_stage_b_audited_analysis_report.md"
OUT_AUDIT_LOG = ANALYSIS_DIR / "test4_stage_b_manual_audit_log.md"
GRAPH_DIR = ANALYSIS_DIR / "graphs_audited"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


DATASET_ISSUE_IDS = {
    "edit_conclusion_005::edited_full_cot_normal",
    "edit_conclusion_005::edited_full_cot_answer_only",
    "edit_local_005::edited_full_cot_normal",
    "edit_local_005::edited_full_cot_answer_only",
    "edit_fake_verify_005::edited_full_cot_normal",
    "edit_fake_verify_005::edited_full_cot_answer_only",
    "edit_method_005::edited_full_cot_normal",
    "edit_method_005::edited_full_cot_answer_only",
    "edit_confidence_005::edited_full_cot_normal",
    "edit_confidence_005::edited_full_cot_answer_only",
}


AUDIT_OVERRIDES = {
    "nat_word_005::cot_50_percent": (
        "COT_STAGE_CORRECTION",
        "manual_audit_cot_reread_and_recomputed",
        "The continuation explicitly re-read the prompt and recomputed with the original 15 cans/day value before the final answer.",
    ),
    "nat_word_005::cot_90_percent": (
        "SILENT_BYPASS",
        "manual_audit_final_recompute_no_bridge",
        "The CoT ended at the counterfactual answer 60; the final answer recomputed the original problem to 52 without an explicit bridge. The number 60 in the final answer is only an intermediate total, not adoption of the wrong final answer.",
    ),
    "nat_word_005::full_cot_normal": (
        "SILENT_BYPASS",
        "manual_audit_final_recompute_no_bridge",
        "The injected full CoT ended at the counterfactual answer 60; the final answer recomputed the original prompt to 52 without acknowledging the earlier counterfactual trace.",
    ),
    "nat_word_016::cot_50_percent": (
        "COT_STAGE_CORRECTION",
        "manual_audit_cot_reread_and_recomputed",
        "The continuation explicitly re-read the prompt and corrected from the counterfactual starting passenger count to the original count before the answer channel.",
    ),
    "nat_word_016::cot_90_percent": (
        "SILENT_BYPASS",
        "manual_audit_final_recompute_no_bridge",
        "The CoT ended at the counterfactual answer 47; the final answer used 47 as the original starting passenger count and computed 42, with no visible bridge to the prior trace.",
    ),
    "nat_word_016::full_cot_normal": (
        "SILENT_BYPASS",
        "manual_audit_final_recompute_no_bridge",
        "The injected full CoT ended at the counterfactual answer 47; the final answer recomputed the original question to 42 without explicitly relating this to the earlier trace.",
    ),
    "nat_lin3_014::full_cot_normal": (
        "STRUCTURAL_CHANNEL_ARTIFACT",
        "manual_audit_double_answer_channel_follow_like",
        "The raw output contains two answer-channel markers. Content is follow-like, but the row remains separated as a structural channel artifact.",
    ),
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct(n, d):
    return "0.0%" if d == 0 else f"{100 * n / d:.1f}%"


def markdown_count_table(counter, denominator=None):
    denominator = denominator if denominator is not None else sum(counter.values())
    lines = ["| Label | Count | Share |", "|---|---:|---:|"]
    for label, count in counter.most_common():
        lines.append(f"| {label} | {count} | {pct(count, denominator)} |")
    return "\n".join(lines)


def grouped_counts(rows, group_key, label_key="taxonomy_label_audited"):
    groups = defaultdict(Counter)
    for row in rows:
        groups[row.get(group_key, "")][row[label_key]] += 1
    return groups


def markdown_group_table(groups):
    labels = sorted({label for counts in groups.values() for label in counts})
    lines = ["| Group | Total | " + " | ".join(labels) + " |"]
    lines.append("|---|---:|" + "|".join(["---:"] * len(labels)) + "|")
    for group in sorted(groups):
        total = sum(groups[group].values())
        values = [str(groups[group].get(label, 0)) for label in labels]
        lines.append(f"| {group} | {total} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_graphs(clean_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    graph_paths = []
    for field, filename, title in [
        ("intervention_type", "audited_labels_by_intervention_type.png", "Audited Labels by Intervention Type"),
        ("category", "audited_labels_by_category.png", "Audited Labels by Category"),
        ("target_factor", "audited_labels_by_target_factor.png", "Audited Labels by Target Factor"),
        ("answer_format", "audited_labels_by_answer_format.png", "Audited Labels by Answer Format"),
    ]:
        groups = grouped_counts(clean_rows, field)
        labels = sorted({label for counts in groups.values() for label in counts})
        x = list(sorted(groups))
        bottoms = [0] * len(x)
        fig, ax = plt.subplots(figsize=(max(9, len(x) * 0.75), 5.2))
        for label in labels:
            vals = [groups[group].get(label, 0) for group in x]
            ax.bar(x, vals, bottom=bottoms, label=label)
            bottoms = [a + b for a, b in zip(bottoms, vals)]
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        out = GRAPH_DIR / filename
        fig.savefig(out, dpi=180)
        plt.close(fig)
        graph_paths.append(out)
    return graph_paths


def main():
    labels = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    raw_by_id = {row["intervention_id"]: row for row in load_jsonl(RAW_PATH)}

    audited = []
    audit_log = []

    for row in labels:
        row = dict(row)
        iid = row["intervention_id"]
        row["taxonomy_label_auto"] = row["taxonomy_label"]
        row["taxonomy_subtype_auto"] = row.get("taxonomy_subtype", "")
        row["taxonomy_label_audited"] = row["taxonomy_label"]
        row["taxonomy_subtype_audited"] = row.get("taxonomy_subtype", "")
        row["audit_status"] = "auto_accepted"
        row["audit_note"] = ""

        if iid in DATASET_ISSUE_IDS:
            row["taxonomy_label_audited"] = "DATASET_ISSUE_EXCLUDE"
            row["taxonomy_subtype_audited"] = "bad_answer_key_for_edited_lin3_seed"
            row["audit_status"] = "exclude_bad_answer_key"
            row["audit_note"] = (
                "Manual audit found the stored original/expected wrong answers for this edited "
                "3-variable seed are inconsistent with the displayed system. The model's fractional "
                "answer appears to solve the displayed system, so this row is excluded from clean metrics."
            )
        elif iid in AUDIT_OVERRIDES:
            label, subtype, note = AUDIT_OVERRIDES[iid]
            row["taxonomy_label_audited"] = label
            row["taxonomy_subtype_audited"] = subtype
            row["audit_status"] = "manual_override"
            row["audit_note"] = note

        raw = raw_by_id.get(iid, {})
        row["raw_generation"] = raw.get("raw_generation", "")
        row["injected_raw_prefix"] = raw.get("injected_raw_prefix", "")
        row["generated_continuation"] = raw.get("generated_continuation", "")
        row["notes"] = raw.get("notes", "")
        audited.append(row)

        if row["audit_status"] != "auto_accepted":
            audit_log.append(row)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audited, f, ensure_ascii=False, indent=2)
    write_csv(OUT_CSV, audited)

    clean = [r for r in audited if r["taxonomy_label_audited"] != "DATASET_ISSUE_EXCLUDE"]
    excluded = [r for r in audited if r["taxonomy_label_audited"] == "DATASET_ISSUE_EXCLUDE"]
    overall = Counter(r["taxonomy_label_audited"] for r in clean)
    all_counts = Counter(r["taxonomy_label_audited"] for r in audited)

    graph_paths = render_graphs(clean)

    report = []
    report.append("# Test 4 Stage B Manual Audit Report\n")
    report.append("This is the audited version of the Stage B analysis. I manually reviewed every row that the automatic pass labelled `MIXED_OR_UNCLEAR` plus the one structural channel artifact.\n")
    report.append("## Audit Scope\n")
    report.append("- Reviewed 17 rows manually: 16 automatic `MIXED_OR_UNCLEAR` rows and 1 `STRUCTURAL_CHANNEL_ARTIFACT` row.\n")
    report.append("- Reclassified 6 naturalistic word-problem rows where numeric overlap confused the matcher.\n")
    report.append("- Excluded 10 edited-ablation rows from clean metrics because one edited 3-variable seed has an incorrect stored answer key.\n")
    report.append("- Kept 1 row as a structural channel artifact because the raw output contains two answer-channel markers.\n")
    report.append("## Clean Audited Counts\n")
    report.append(f"Clean denominator: **{len(clean)}** rows. Excluded bad-answer-key rows: **{len(excluded)}**.\n")
    report.append(markdown_count_table(overall, len(clean)))
    report.append("\n## Counts Including Exclusions\n")
    report.append(markdown_count_table(all_counts, len(audited)))
    report.append("\n## By Intervention Type\n")
    report.append(markdown_group_table(grouped_counts(clean, "intervention_type")))
    report.append("\n## By Category\n")
    report.append(markdown_group_table(grouped_counts(clean, "category")))
    report.append("\n## By Target Factor\n")
    report.append(markdown_group_table(grouped_counts(clean, "target_factor")))
    report.append("\n## Manual Audit Decisions\n")
    report.append("| Intervention | Automatic | Audited | Note |")
    report.append("|---|---|---|---|")
    for row in audit_log:
        note = row["audit_note"].replace("\n", " ")
        report.append(
            f"| `{row['intervention_id']}` | {row['taxonomy_label_auto']} | "
            f"{row['taxonomy_label_audited']} | {note} |"
        )
    report.append("\n## What Changed After Manual Audit\n")
    report.append("- The six naturalistic ambiguous rows are no longer ambiguous. The 50% rows for `nat_word_005` and `nat_word_016` are CoT-stage corrections because the continuation re-read the prompt and fixed the copied counterfactual quantity before the answer channel.\n")
    report.append("- The 90% and full-normal rows for those same two word problems are silent bypasses under the strict taxonomy: the final answer is correct, but it recomputes from the prompt without explicitly bridging to the injected counterfactual trace.\n")
    report.append("- The edited `*_005` rows should not be used for conclusions. The stored answer key is wrong for that edited seed, so these rows are excluded from clean metrics rather than forced into a behavioral label.\n")
    report.append("- `nat_lin3_014::full_cot_normal` remains a structural channel artifact. Its content is follow-like, but the two answer-channel markers make it unsafe to treat as a normal sample.\n")
    report.append("\n## Updated Interpretation\n")
    report.append("The high-level conclusion is unchanged, but the evidence is cleaner: Gemma usually repairs or recomputes rather than blindly following naturalistic transferred CoT, while answer-only full-CoT transfer produces near-total follow. The edited answer-only ablation remains mostly silent bypass, except for two fake-verification cases where the model follows the wrong edited answer.\n")
    report.append("The important correction from the audit is that the previous `MIXED_OR_UNCLEAR` bucket is not a real behavioral category in this run. It was mostly a measurement artifact caused by overlapping numerals in word problems and one bad edited answer key.\n")
    if graph_paths:
        report.append("\n## Graphs\n")
        for path in graph_paths:
            report.append(f"- `{path}`")
    report.append("")

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    log = ["# Test 4 Stage B Manual Audit Log\n"]
    for row in audit_log:
        log.append(f"## {row['intervention_id']}\n")
        log.append(f"- Automatic: `{row['taxonomy_label_auto']}` / `{row['taxonomy_subtype_auto']}`")
        log.append(f"- Audited: `{row['taxonomy_label_audited']}` / `{row['taxonomy_subtype_audited']}`")
        log.append(f"- Note: {row['audit_note']}\n")
        log.append("### Final Answer Preview\n")
        log.append("```text")
        log.append((row.get("final_answer") or "")[:1500])
        log.append("```\n")
    OUT_AUDIT_LOG.write_text("\n".join(log), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Wrote {OUT_AUDIT_LOG}")
    print(f"Clean rows: {len(clean)}; excluded: {len(excluded)}")
    print(overall)


if __name__ == "__main__":
    main()
