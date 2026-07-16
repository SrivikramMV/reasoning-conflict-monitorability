








from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_candidate_template_smoke import candidate_pair, classify_generated
from local_gemma_follow_bypass_probe import (
    MODEL_ID,
    ROOT,
    build_gemma_prompt,
    gemma_stop_token_ids,
    load_case,
)
from local_gemma_heldout_boundary_extract import completed_cot


DEFAULT_CASES = [
    "nat_lin3_004::full_cot_normal",
    "nat_lin3_005::full_cot_normal",
    "nat_lin3_008::full_cot_normal",
    "nat_lin3_013::full_cot_normal",
]
MASKS = ["none", "question", "cot", "question_and_cot"]


def find_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    raise ValueError("Could not locate question tokens inside the formatted prompt.")


def make_source_mask(
    length: int,
    question_span: tuple[int, int],
    cot_span: tuple[int, int],
    condition: str,
    prefix_token_ids: list[int],
    special_token_ids: set[int],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones((1, length), dtype=torch.long, device=device)
    if condition in {"question", "question_and_cot"}:
        mask[:, question_span[0] : question_span[1]] = 0
    if condition in {"cot", "question_and_cot"}:
                                                                              
                                                                             
                                                                  
        for position in range(cot_span[0], min(cot_span[1], len(prefix_token_ids))):
            if prefix_token_ids[position] not in special_token_ids:
                mask[:, position] = 0
    return mask


def masked_generate(
    model: Any,
    processor: Any,
    boundary_text: str,
    question: str,
    condition: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    full_inputs = processor(text=boundary_text, return_tensors="pt")
    full_ids = full_inputs["input_ids"][0].tolist()
    prefix_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    prefix_len = prefix_ids.shape[-1]

    prompt = build_gemma_prompt(processor, question)
    prompt_ids = processor(text=prompt, return_tensors="pt")["input_ids"][0].tolist()
    question_ids = processor.tokenizer.encode(question, add_special_tokens=False)
    question_span = find_subsequence(prompt_ids, question_ids)
    cot_span = (len(prompt_ids), len(full_ids) - 1)
    if cot_span[1] <= cot_span[0]:
        raise ValueError("Empty CoT span.")

    with torch.inference_mode():
        prefill = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
        )
    cache = prefill.past_key_values
    stop_ids = set(gemma_stop_token_ids(processor))
    first_token = int(torch.argmax(prefill.logits[0, -1]).item())
    del prefill
    generated: list[int] = [] if first_token in stop_ids else [first_token]
    current = torch.tensor([[first_token]], dtype=torch.long, device=model.device)
    special_token_ids = set(processor.tokenizer.all_special_ids)
                                                                        
                                                                         
                                                                
    for step in range(max(0, max_new_tokens - len(generated))):
        if not generated:
            break
        total_length = prefix_len + len(generated)
        attention_mask = make_source_mask(
            total_length,
            question_span,
            cot_span,
            condition,
            full_ids,
            special_token_ids,
            model.device,
        )
        cache_position = torch.tensor([total_length - 1], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            output = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
        cache = output.past_key_values
        next_token = int(torch.argmax(output.logits[0, -1]).item())
        del output
        if next_token in stop_ids:
            break
        generated.append(next_token)
        current = torch.tensor([[next_token]], dtype=torch.long, device=model.device)

    text = processor.tokenizer.decode(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    del cache, prefix_ids, current
    torch.cuda.empty_cache()
    return {
        "text": text,
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "question_span": list(question_span),
        "cot_span": list(cot_span),
        "prefix_token_count_without_boundary": int(prefix_len),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--masks", nargs="*", default=MASKS)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Results" / "dissertation_smoke_tests" / "source_masking",
    )
    args = parser.parse_args()
    unknown = set(args.masks) - set(MASKS)
    if unknown:
        raise ValueError(f"Unknown masks: {sorted(unknown)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id in args.cases:
        packet, label = load_case(case_id)
        boundary = build_gemma_prompt(processor, packet["original_question"]) + completed_cot(packet)
        _, q_target, qstar_target = candidate_pair(packet, "standard_system")
        for condition in args.masks:
            print(f"Running {case_id} / mask={condition}", flush=True)
            result = masked_generate(
                model,
                processor,
                boundary,
                packet["original_question"],
                condition,
                args.max_new_tokens,
            )
            state = classify_generated(result["text"], q_target, qstar_target)
            row = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_id": MODEL_ID,
                "case_id": case_id,
                "manual_label": (label or {}).get("manual_label"),
                "mask": condition,
                "generated_state": state,
                "q_target": q_target,
                "qstar_target": qstar_target,
                **result,
            }
            rows.append(row)
            print(f"  state={state}: {result['text'][:180].replace(chr(10), ' | ')}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out_dir / f"source_mask_smoke_{stamp}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
