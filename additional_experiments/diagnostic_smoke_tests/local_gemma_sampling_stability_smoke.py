

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
    "nat_lin3_013::full_cot_normal",                     
    "nat_lin3_008::full_cot_normal",                      
    "nat_lin3_005::full_cot_normal",                      
]


def boundary_margin(case_id: str) -> float:
    stem = case_id.replace("::", "__") + ".json"
    path = (
        ROOT
        / "Results"
        / "local_gemma_follow_bypass_probes"
        / "heldout_boundary_prediction"
        / "features"
        / stem
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    return float(row["boundary_delta_qstar_minus_q"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--samples-per-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--base-seed", type=int, default=20260716)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Results" / "dissertation_smoke_tests" / "sampling_stability",
    )
    args = parser.parse_args()
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
    for case_index, case_id in enumerate(args.cases):
        packet, label = load_case(case_id)
        boundary = build_gemma_prompt(processor, packet["original_question"]) + completed_cot(packet)
        inputs = processor(text=boundary, return_tensors="pt").to(model.device)
        input_len = int(inputs["input_ids"].shape[-1])
        _, q_target, qstar_target = candidate_pair(packet, "standard_system")
        margin = boundary_margin(case_id)
        for batch_index in range(args.batches):
            seed = args.base_seed + case_index * 100 + batch_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            print(
                f"Running {case_id} batch {batch_index + 1}/{args.batches} seed={seed}",
                flush=True,
            )
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.samples_per_batch,
                    eos_token_id=gemma_stop_token_ids(processor),
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            for sample_index, sequence in enumerate(output):
                generated = processor.tokenizer.decode(
                    sequence[input_len:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                state = classify_generated(generated, q_target, qstar_target)
                row = {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model_id": MODEL_ID,
                    "case_id": case_id,
                    "manual_label": (label or {}).get("manual_label"),
                    "boundary_margin_qstar_minus_q": margin,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": seed,
                    "batch_index": batch_index,
                    "sample_index": sample_index,
                    "sampled_state": state,
                    "q_target": q_target,
                    "qstar_target": qstar_target,
                    "generated": generated,
                }
                rows.append(row)
                print(f"  sample {sample_index + 1}: {state}", flush=True)
            del output
            torch.cuda.empty_cache()
        del inputs
        torch.cuda.empty_cache()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out_dir / f"sampling_stability_{stamp}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
