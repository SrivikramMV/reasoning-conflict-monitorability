import csv
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = (
    ROOT
    / "Results"
    / "Test 4 - 300 questions all CoT"
    / "Stage B results"
    / "Gemma_CoT_Stage_B"
    / "analysis"
    / "manual_review"
)
LABELS = MANUAL_DIR / "test4_stage_b_MANUAL_JUDGED.pretty.json"
PACKETS = MANUAL_DIR / "test4_stage_b_review_packets.jsonl"
OUT_DIR = MANUAL_DIR / "follow_vs_bypass_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "follow_vs_bypass_signal_table.csv"
OUT_REPORT = OUT_DIR / "follow_vs_bypass_signal_probe.md"


def load_packets():
    packets = {}
    with PACKETS.open("r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            packets[p["intervention_id"]] = p
    return packets


def num_parts(answer):
    return re.findall(r"-?\d+\s*/\s*-?\d+|-?\d+(?:\.\d+)?", str(answer or ""))


def fraction_complexity(answer):
    vals = []
    for part in num_parts(answer):
        try:
            vals.append(Fraction(part.replace(" ", "")))
        except Exception:
            pass
    if not vals:
        return {"answer_num_count": 0, "answer_nonint_count": 0, "answer_max_den": 1}
    return {
        "answer_num_count": len(vals),
        "answer_nonint_count": sum(v.denominator != 1 for v in vals),
        "answer_max_den": max(v.denominator for v in vals),
    }


def first_changed_values(packet):
    changed = packet.get("changed_line") or {}
    orig = changed.get("original") or ""
    cf = changed.get("counterfactual") or ""
                                                   
    orig_nums = re.findall(r"-?\d+", orig)
    cf_nums = re.findall(r"-?\d+", cf)
    vals = []
    for a, b in zip(orig_nums, cf_nums):
        if a != b:
            vals.append((a, b))
    return vals


def count_value_mentions(text, values):
    compact = (text or "").replace(" ", "")
    counts = {}
    for orig, cf in values:
        counts[f"orig_{orig}"] = len(re.findall(rf"(?<![\d/])-?{re.escape(orig)}(?![\d/])", compact))
        counts[f"cf_{cf}"] = len(re.findall(rf"(?<![\d/])-?{re.escape(cf)}(?![\d/])", compact))
    return counts


def text_features(text):
    text = text or ""
    lower = text.lower()
    terms = {
        "verification": ["verify", "verification", "check equation", "checks out", "correct."],
        "uncertainty": ["unlikely", "typo", "messy", "fraction", "fractions", "double-check", "re-check", "recheck"],
        "authority": ["correct based on the input", "solution is correct", "checks out", "final presentation"],
        "prompt_reanchor": ["original problem", "prompt", "re-read", "reread", "misread"],
    }
    out = {
        "chars": len(text),
        "lines": text.count("\n") + 1 if text else 0,
        "equation_refs": len(re.findall(r"equation|eq\.|\(\d+\)", lower)),
        "math_lines": len(re.findall(r"[$=]|\\frac|\\times|\\implies", text)),
        "fraction_mentions": len(re.findall(r"\d+\s*/\s*\d+|\\frac", text)),
        "step_mentions": len(re.findall(r"\bstep\b", lower)),
    }
    for key, needles in terms.items():
        out[f"{key}_terms"] = sum(lower.count(needle) for needle in needles)
    return out


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


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, group_key, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    lines = []
    for group in sorted(groups):
        lines.append(f"### {group} (n={len(groups[group])})")
        lines.append("| Feature | Mean |")
        lines.append("|---|---:|")
        for field in fields:
            lines.append(f"| `{field}` | {mean([r.get(field) for r in groups[group]]):.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    packets = load_packets()
    rows = []
    for label in labels:
        if label["intervention_family"] != "naturalistic_counterfactual_transfer":
            continue
        if label["answer_format"] != "normal":
            continue
        if label["manual_label"] not in {"FOLLOW", "SILENT_BYPASS_REANCHOR"}:
            continue
        if label["category"] not in {"linear_system_2var", "linear_system_3var"}:
            continue
        packet = packets[label["intervention_id"]]
        cot_text = (packet.get("injected_cot_full") or "") + "\n" + (packet.get("generated_cot_continuation_full") or "")
        final = packet.get("final_answer_full") or ""
        changed_vals = first_changed_values(packet)
        value_counts_cot = count_value_mentions(cot_text, changed_vals)
        value_counts_final = count_value_mentions(final, changed_vals)
        row = {
            "intervention_id": label["intervention_id"],
            "manual_label": label["manual_label"],
            "category": label["category"],
            "intervention_type": label["intervention_type"],
            "original_answer": label["original_answer"],
            "expected_wrong_answer": label["expected_wrong_answer"],
            "changed_line_original": (packet.get("changed_line") or {}).get("original"),
            "changed_line_counterfactual": (packet.get("changed_line") or {}).get("counterfactual"),
            "source_generated_token_count": packet.get("generated_token_count"),
        }
        for prefix, text in [("cot", cot_text), ("final", final), ("generated_cot", packet.get("generated_cot_continuation_full") or "")]:
            for k, v in text_features(text).items():
                row[f"{prefix}_{k}"] = v
        for k, v in fraction_complexity(label["original_answer"]).items():
            row[f"orig_{k}"] = v
        for k, v in fraction_complexity(label["expected_wrong_answer"]).items():
            row[f"wrong_{k}"] = v
        for k, v in value_counts_cot.items():
            row[f"cot_{k}_mentions"] = v
        for k, v in value_counts_final.items():
            row[f"final_{k}_mentions"] = v
                                                                       
        row["heuristic_predict_follow"] = (
            row["cot_authority_terms"] >= 2
            and row["cot_verification_terms"] >= 2
            and row["cot_fraction_mentions"] >= 8
            and row["cot_prompt_reanchor_terms"] == 0
        )
        rows.append(row)

    write_csv(OUT_CSV, rows)

    fields = [
        "cot_chars", "generated_cot_chars", "cot_fraction_mentions",
        "cot_verification_terms", "cot_uncertainty_terms", "cot_authority_terms",
        "cot_prompt_reanchor_terms", "orig_answer_nonint_count", "wrong_answer_nonint_count",
        "wrong_answer_max_den",
    ]
    report = []
    report.append("# Follow vs Bypass Signal Probe\n")
    report.append("Scope: naturalistic counterfactual-transfer rows, normal final-answer mode, simultaneous-equation categories only, labels restricted to `FOLLOW` and `SILENT_BYPASS_REANCHOR`.\n")
    report.append(f"Rows analysed: **{len(rows)}**\n")
    report.append("## Counts\n")
    report.append("| Group | Count |")
    report.append("|---|---:|")
    for key, count in Counter((r["category"], r["intervention_type"], r["manual_label"]) for r in rows).most_common():
        report.append(f"| `{key}` | {count} |")
    report.append("\n## Mean Feature Comparison By Label\n")
    report.append(summarize(rows, "manual_label", fields))
    report.append("\n## CoT-Only Heuristic Probe\n")
    pred_rows = [r for r in rows if r["manual_label"] in {"FOLLOW", "SILENT_BYPASS_REANCHOR"}]
    correct = sum((r["heuristic_predict_follow"] and r["manual_label"] == "FOLLOW") or ((not r["heuristic_predict_follow"]) and r["manual_label"] == "SILENT_BYPASS_REANCHOR") for r in pred_rows)
    report.append(f"Simple heuristic accuracy on this diagnostic set: **{correct}/{len(pred_rows)} = {100*correct/len(pred_rows):.1f}%**.\n")
    report.append("Heuristic: predict FOLLOW when CoT has high authority/verification/fraction signals and no prompt-reanchor terms. This is not a classifier for final results; it is only a rough diagnostic for whether the CoT itself carries predictive signal.\n")
    report.append("## Per-Row Table\n")
    report.append(f"Full table: `{OUT_CSV}`\n")
    for r in rows:
        report.append(
            f"- `{r['intervention_id']}`: {r['manual_label']}; "
            f"CoT chars={r['cot_chars']}, generated CoT chars={r['generated_cot_chars']}, "
            f"fractions={r['cot_fraction_mentions']}, verification={r['cot_verification_terms']}, "
            f"authority={r['cot_authority_terms']}, prompt-reanchor={r['cot_prompt_reanchor_terms']}; "
            f"orig={r['original_answer']}, wrong={r['expected_wrong_answer']}"
        )
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(OUT_REPORT)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
