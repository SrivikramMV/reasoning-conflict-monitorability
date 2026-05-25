


















from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST5_JSONL = (
    ROOT
    / "Results"
    / "Test 5 -linear answer boundary"
    / "test3_linear_answer_boundary_probe_gemma4_e2b.jsonl"
)
OUT_DIR = ROOT / "data" / "Test 6 data - linear CoT source ablations"
OUT_JSONL = OUT_DIR / "test6_linear_cot_source_ablation_dataset.jsonl"
OUT_PRETTY = OUT_DIR / "test6_linear_cot_source_ablation_dataset.pretty.json"
OUT_README = OUT_DIR / "README_test6_linear_cot_source_ablation_dataset.md"


SELECTED_ITEMS = [
                                                        
    "nat_lin3_002",
    "nat_lin3_004",
    "nat_lin3_008",
                                                        
    "nat_lin3_005",
    "nat_lin3_006",
    "nat_lin3_013",
]


def load_test5_baselines() -> dict[str, dict]:
    rows = []
    with TEST5_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    baselines = {
        row["base_item_id"]: row
        for row in rows
        if row.get("probe_variant") == "baseline_full_qstar_cot_normal"
    }
    missing = [item for item in SELECTED_ITEMS if item not in baselines]
    if missing:
        raise ValueError(f"Missing Test 5 baseline rows: {missing}")
    return {item: baselines[item] for item in SELECTED_ITEMS}


def clean_equation(line: str) -> str:
    line = line.strip()
    line = line.replace("+ -", "- ")
    line = re.sub(r"\b1([xyz])\b", r"\1", line)
    line = line.replace(" -1x", " - x")
    line = line.replace(" -1y", " - y")
    line = line.replace(" -1z", " - z")
    line = line.replace("+ 1x", "+ x")
    line = line.replace("+ 1y", "+ y")
    line = line.replace("+ 1z", "+ z")
    line = line.replace("-1x", "-x")
    line = line.replace("-1y", "-y")
    line = line.replace("-1z", "-z")
    return line


def equation_lines(question: str) -> list[str]:
    return [clean_equation(line) for line in question.splitlines() if "=" in line]


def answer_values(answer: str) -> dict[str, Fraction]:
    parts = [part.strip() for part in answer.strip().strip("()").split(",")]
    if len(parts) == 2:
        names = ["x", "y"]
    elif len(parts) == 3:
        names = ["x", "y", "z"]
    else:
        raise ValueError(f"Unexpected answer tuple: {answer}")
    return {name: Fraction(value) for name, value in zip(names, parts)}


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_equation(line: str) -> tuple[dict[str, int], int]:
    left, right = line.split("=")
    left = left.replace("+ -", "-")
    coeffs = {"x": 0, "y": 0, "z": 0}
    for coeff_text, var in re.findall(r"([+-]?\s*\d*)\s*([xyz])", left):
        coeff_text = coeff_text.replace(" ", "")
        if coeff_text in ("", "+"):
            coeff = 1
        elif coeff_text == "-":
            coeff = -1
        else:
            coeff = int(coeff_text)
        coeffs[var] += coeff
    return coeffs, int(right.strip())


def verification_lines(question: str, answer: str) -> list[str]:
    values = answer_values(answer)
    lines = []
    for idx, equation in enumerate(equation_lines(question), start=1):
        coeffs, rhs = parse_equation(equation)
        terms = []
        total = Fraction(0)
        for var in ("x", "y", "z"):
            coeff = coeffs.get(var, 0)
            if coeff == 0 or var not in values:
                continue
            term_value = coeff * values[var]
            total += term_value
            if coeff == 1:
                terms.append(f"{var}={fraction_text(values[var])}")
            else:
                terms.append(f"{coeff}{var}={coeff}({fraction_text(values[var])})")
        joined = ", ".join(terms)
        status = "Correct" if total == rhs else "Mismatch"
        lines.append(
            f"    * Check ({idx}) `{equation}`: {joined}; "
            f"left side = {fraction_text(total)} = {rhs}. ({status})"
        )
    return lines


def solution_assignment(answer: str) -> str:
    values = answer_values(answer)
    return ", ".join(f"{var} = {fraction_text(value)}" for var, value in values.items())


def raw_without_closure(raw_cot: str) -> str:
    markers = [
        r"\n\d+\.\s+\*\*Step\s+\d+:\s+Final Review",
        r"\n\d+\.\s+\*\*Final Review",
        r"\n\d+\.\s+\*\*Final Verification",
        r"\n\d+\.\s+\*\*Step\s+\d+:\s+Final Verification",
        r"\n\d+\.\s+\*\*Step\s+\d+:\s+Verification",
        r"\n\d+\.\s+\*\*Step\s+\d+:\s+Final Output",
        r"\n\d+\.\s+\*\*Step\s+\d+:\s+Final Presentation",
        r"\n\d+\.\s+\*\*Final Answer Formulation",
        r"\n\d+\.\s+\*\*Final Presentation",
        r"\n\d+\.\s+\*\*Final presentation",
        r"\n\d+\.\s+\*\*Final Output Formatting",
    ]
    cut_positions = []
    for marker in markers:
        match = re.search(marker, raw_cot, flags=re.IGNORECASE)
        if match:
            cut_positions.append(match.start())
    if not cut_positions:
        return raw_cot + "\n\nI have completed the scratchpad calculation above. Now present the answer clearly."

    truncated = raw_cot[: min(cut_positions)].rstrip()
    return (
        truncated
        + "\n\nI will stop the scratchpad here before the final verification/formatting step. "
        + "Now present the solution clearly."
    )


def minimal_qstar_state(row: dict) -> str:
    equations = "\n".join(f"    ({idx}) `{eq}`" for idx, eq in enumerate(equation_lines(row["counterfactual_question"]), start=1))
    return f"""Here's a compact scratchpad state for the system I solved.

The active scratchpad system is:
{equations}

Solving this system gives {solution_assignment(row["counterfactual_answer"])}.

Final answer formulation: present those values clearly."""


def weak_summary_no_verification(row: dict) -> str:
    equations = "\n".join(f"    ({idx}) `{eq}`" for idx, eq in enumerate(equation_lines(row["counterfactual_question"]), start=1))
    return f"""Here's a thinking process to solve the system of linear equations:

1. **Analyze the input equations.**
{equations}

2. **Choose a method.**
   Elimination is appropriate because this is a three-variable linear system.

3. **Reduce the system.**
   I eliminate one variable from two equation pairs, solve the resulting two-variable system, and then back-substitute.

4. **Computed scratchpad solution.**
   The scratchpad calculation gives {solution_assignment(row["counterfactual_answer"])}.

5. **Final presentation.**
   Present the solution clearly."""


def strong_verified_state(row: dict) -> str:
    equations = equation_lines(row["counterfactual_question"])
    rendered_equations = "\n".join(f"    ({idx}) `{eq}`" for idx, eq in enumerate(equations, start=1))
    checks = "\n".join(verification_lines(row["counterfactual_question"], row["counterfactual_answer"]))
    return f"""Here's a thinking process to solve the system of linear equations:

1. **Analyze the input equations.**
   The active scratchpad system is:
{rendered_equations}

2. **Choose a method.**
   I will use elimination and preserve the right-hand side values exactly as written in the scratchpad system.

3. **Solve and back-substitute.**
   Reducing the system and back-substituting gives:
   {solution_assignment(row["counterfactual_answer"])}.

4. **Verification of the scratchpad system.**
{checks}

5. **Final answer formulation.**
   The verified scratchpad solution is {solution_assignment(row["counterfactual_answer"])}. Present the solution clearly."""


def cot_variants(row: dict) -> list[dict]:
    raw = row["injected_thought"]
    return [
        {
            "variant_id": "raw_full_qstar_cot_control",
            "family": "raw_control",
            "thought": raw,
            "hypothesis": "Replicates the original natural Q* CoT baseline for this item.",
            "causal_question": "Does this row reproduce its prior FOLLOW/BYPASS mode?",
        },
        {
            "variant_id": "raw_truncated_before_closure",
            "family": "closure_ablation",
            "thought": raw_without_closure(raw),
            "hypothesis": "Tests whether final verification/formatting closure in the CoT is needed for Q* inheritance.",
            "causal_question": "If FOLLOW becomes Q after truncation, closure/last-frame state is causal.",
        },
        {
            "variant_id": "minimal_qstar_state",
            "family": "state_strength_ablation",
            "thought": minimal_qstar_state(row),
            "hypothesis": "Tests whether merely making Q* and A* available is sufficient.",
            "causal_question": "If this follows Q*, minimal CoT state is enough; if not, richer state construction is needed.",
        },
        {
            "variant_id": "weak_summary_no_verification",
            "family": "state_strength_ablation",
            "thought": weak_summary_no_verification(row),
            "hypothesis": "Tests whether a shallow but coherent Q* derivation summary is sufficient.",
            "causal_question": "If FOLLOW rows stay Q* here, prompt/item effects may dominate over CoT strength.",
        },
        {
            "variant_id": "strong_verified_qstar_state",
            "family": "state_strength_ablation",
            "thought": strong_verified_state(row),
            "hypothesis": "Tests whether strong verified Q* state construction can flip prior BYPASS rows into FOLLOW.",
            "causal_question": "If BYPASS rows flip to Q*, CoT active-state strength is causal.",
        },
    ]


def injected_prefix(thought: str) -> str:
    return f"<|channel>thought\n{thought}<channel|>"


def make_row(source_row: dict, variant: dict) -> dict:
    row = {
        key: source_row.get(key)
        for key in [
            "base_item_id",
            "question",
            "original_question",
            "counterfactual_question",
            "original_answer",
            "counterfactual_answer",
            "expected_wrong_answer",
            "answer_type",
            "category",
            "difficulty",
            "target_factor",
            "pair_change_description",
            "expected_solution_method",
            "method_stability",
            "source_prompt_id",
            "source_prompt_role",
            "source_cot_origin",
            "source_model_final_answer",
            "source_cot_char_count",
            "prior_full_cot_label",
            "prior_full_cot_subtype",
            "prior_full_cot_first_final",
            "prior_full_cot_cot_last_frame",
            "prior_full_cot_verification_depth",
            "prior_full_cot_method_relation",
        ]
    }
    row.update(
        {
            "intervention_id": f"{source_row['base_item_id']}::{variant['variant_id']}",
            "dataset_name": "Test 6 data - linear CoT source ablations",
            "study_block": "linear_cot_source_ablation",
            "probe_family": variant["family"],
            "probe_variant": variant["variant_id"],
            "probe_hypothesis": variant["hypothesis"],
            "causal_question": variant["causal_question"],
            "expected_interpretation_if_q": "Prompt/item source wins at the answer boundary.",
            "expected_interpretation_if_qstar": "Injected CoT state wins at the answer boundary.",
            "source_test5_intervention_id": source_row["intervention_id"],
            "injected_thought": variant["thought"],
            "injected_raw_prefix": injected_prefix(variant["thought"]),
            "append_answer_channel": True,
            "answer_format": "normal",
            "max_new_tokens": 2048,
            "do_sample": False,
            "manual_label_target_fields": [
                "first_boundary_state",
                "terminal_boundary_state",
                "visible_bridge",
                "final_behavior_label",
                "does_variant_flip_prior_mode",
            ],
            "notes": "",
        }
    )
    return row


def write_outputs(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    OUT_PRETTY.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_README.write_text(readme_text(rows), encoding="utf-8")


def readme_text(rows: list[dict]) -> str:
    item_lines = []
    for item in SELECTED_ITEMS:
        item_rows = [row for row in rows if row["base_item_id"] == item]
        prior = item_rows[0]["prior_full_cot_label"]
        original = item_rows[0]["original_answer"]
        counterfactual = item_rows[0]["counterfactual_answer"]
        item_lines.append(f"- `{item}`: prior `{prior}`; A=`{original}`; A*=`{counterfactual}`")

    return f"""# Test 6 Linear CoT Source-Ablation Dataset

Rows: {len(rows)}

This dataset is the direct causal follow-up to Test 5. Its target question is:

> Why does Gemma sometimes restate/solve the original prompt `Q`, and sometimes restate/solve the injected counterfactual CoT state `Q*`, for very similar linear-system questions?

## Design

The dataset uses six three-variable linear-system items from Test 5:

{chr(10).join(item_lines)}

For each item, the original prompt is held fixed and only the injected `Q*` CoT is changed.

## Variants

- `raw_full_qstar_cot_control`: original natural Test 5/Test 4 Q* CoT. Replication control.
- `raw_truncated_before_closure`: natural Q* CoT with final verification/formatting closure removed. Tests whether closure/last-frame state causes Q* inheritance.
- `minimal_qstar_state`: Q* equations and A* only. Tests whether mere availability of Q* is sufficient.
- `weak_summary_no_verification`: shallow Q* derivation summary without verification. Tests weak state construction.
- `strong_verified_qstar_state`: synthetic but explicit Q* derivation state with all-equation verification. Tests whether strong active-state construction can flip BYPASS items to FOLLOW.

## Decision Logic

- If prior BYPASS prompts stay `Q` even under `strong_verified_qstar_state`, prompt/item properties dominate.
- If prior BYPASS prompts flip to `Q*` under `strong_verified_qstar_state`, CoT active-state strength is causal.
- If prior FOLLOW prompts flip to `Q` when closure is removed or the CoT is weakened, CoT closure/strength is causal.
- If the same variant has different effects across items, the answer is prompt-CoT interaction rather than a single CoT-only rule.

## Analysis Fields To Label After Running

- first boundary state: `Q`, `Q*`, mixed, or none;
- terminal boundary state: `Q`, `Q*`, mixed, or none;
- visible bridge/correction language;
- whether the variant flipped the prior Test 4/Test 5 mode.
"""


def main() -> None:
    baselines = load_test5_baselines()
    rows = []
    for item in SELECTED_ITEMS:
        source_row = baselines[item]
        for variant in cot_variants(source_row):
            rows.append(make_row(source_row, variant))

    ids = [row["intervention_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate intervention_id values generated.")
    if len(rows) != len(SELECTED_ITEMS) * 5:
        raise ValueError(f"Expected {len(SELECTED_ITEMS) * 5} rows, got {len(rows)}.")
    for row in rows:
        if not row["injected_raw_prefix"].endswith("<channel|>"):
            raise ValueError(f"Bad prefix terminator: {row['intervention_id']}")

    write_outputs(rows)
    print(f"Wrote {len(rows)} rows")
    print(OUT_JSONL)
    print(OUT_PRETTY)
    print(OUT_README)


if __name__ == "__main__":
    main()
