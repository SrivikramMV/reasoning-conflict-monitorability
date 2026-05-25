

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "gemma4_e2b_linear_answer_boundary_probe_R8.ipynb"
OUT_NOTEBOOK = ROOT / "gemma4_e2b_linear_cot_source_ablation_R9.ipynb"


REPLACEMENTS = {
    "Gemma 4 E2B Linear Answer-Boundary Probe Runner R8": "Gemma 4 E2B Linear CoT Source-Ablation Runner R9",
    "focused Test 3 linear answer-boundary probe": "Test 6 linear CoT source-ablation",
    "test3_focused_linear_answer_boundary_probe_dataset.jsonl": "test6_linear_cot_source_ablation_dataset.jsonl",
    "linear answer-boundary probe": "linear CoT source-ablation",
    "Linear Boundary Probe": "Linear CoT Source-Ablation",
    "linear boundary probe": "linear CoT source-ablation",
    "linear answer-boundary probe rows": "linear CoT source-ablation rows",
    "/content/test3_focused_linear_answer_boundary_probe_dataset.jsonl": "/content/test6_linear_cot_source_ablation_dataset.jsonl",
    "/content/gemma_linear_boundary_probe_outputs": "/content/gemma_linear_cot_source_ablation_outputs",
    "/content/drive/MyDrive/Gemma_CoT_Test3_Linear_Boundary_Probe": "/content/drive/MyDrive/Gemma_CoT_Test6_Linear_CoT_Source_Ablation",
    "test3_linear_answer_boundary_probe_gemma4_e2b": "test6_linear_cot_source_ablation_gemma4_e2b",
    "Answer-Boundary Probe": "CoT Source-Ablation",
    "answer-boundary probe": "CoT source-ablation",
    "Probe rows selected": "Ablation rows selected",
    "Probe family": "Ablation family",
    "Probe variant": "Ablation variant",
    "Found probe dataset": "Found ablation dataset",
    "Probe file contains duplicate intervention_id values.": "Ablation file contains duplicate intervention_id values.",
}


def source(text: str) -> list[str]:
    return [line + "\n" for line in text.rstrip("\n").split("\n")]


def replace_in_cell(cell: dict) -> None:
    cell_source = cell.get("source")
    if not isinstance(cell_source, list):
        return
    text = "".join(cell_source)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    cell["source"] = source(text)


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        replace_in_cell(cell)

                                                                           
    notebook["cells"][0]["source"] = source(
        """# Gemma 4 E2B Linear CoT Source-Ablation Runner R9

This notebook runs the **Test 6 linear CoT source-ablation** dataset:

```text
test6_linear_cot_source_ablation_dataset.jsonl
```

Goal: determine whether Gemma's baseline `Q` vs `Q*` final-answer choice is driven by the original prompt/item, by the visible strength of the injected `Q*` CoT state, or by their interaction.

For each row it:

1. Builds the original linear-system prompt `Q`.
2. Prefills Gemma with one ablated `Q*` CoT variant.
3. Lets Gemma generate the final answer.
4. Saves raw generation, parsed CoT/final answer, token IDs, and all ablation metadata.
5. Checkpoints every row and syncs results to Google Drive.

Use a Colab GPU runtime and run top-to-bottom."""
    )

                                                                          
    notebook["cells"][22]["source"] = source(
        '''from collections import Counter

records = load_existing_records(RUN_JSONL)
print(f"Records: {len(records)}")
print("Status:", Counter(r.get("status") for r in records))
print("Ablation family:", Counter(r.get("probe_family") for r in records))
print("Ablation variant:", Counter(r.get("probe_variant") for r in records))
print("Category:", Counter(r.get("category") for r in records))
print("Prior Test 4/5 label:", Counter(r.get("prior_full_cot_label") for r in records))
print("Base items:", Counter(r.get("base_item_id") for r in records))
print("Answer format:", Counter(r.get("answer_format") for r in records))'''
    )

    notebook["cells"][16]["source"] = [
        line.replace('Example: {"last_frame_probe"}', 'Example: {"state_strength_ablation"}')
        .replace('Example: {"frame_qstar_self_check"}', 'Example: {"strong_verified_qstar_state"}')
        for line in notebook["cells"][16]["source"]
    ]

    OUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT_NOTEBOOK)


if __name__ == "__main__":
    main()
