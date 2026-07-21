from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from adapters import make_adapter
from classification import classify_answer_branch, classify_visible_branch
from common import JsonlCheckpoint, atomic_write_json, chunks, read_json, read_jsonl, stable_seed, utc_now
from runtime_utils import completed_boundary_ids, decode, ensure_cuda_a100


def parse_sample(adapter: Any, pair: dict[str, Any], generated: list[int], max_new_tokens: int) -> dict[str, Any]:
    answer_ids = []
    stop_reason = "max_new_tokens" if len(generated) >= max_new_tokens else "generation_stopped_without_registered_token"
    for value in generated:
        if value in set(adapter.markers.terminal_ids):
            stop_reason = "terminal"
            break
        answer_ids.append(int(value))
    text = decode(adapter, answer_ids)
    return {
        "generated_token_ids": [int(v) for v in generated],
        "generated_token_count": len(generated),
        "final_answer_token_ids": answer_ids,
        "final_answer": text.strip(),
        "truncated": stop_reason == "max_new_tokens",
        "stop_reason": stop_reason,
        "automatic_visible_branch": classify_visible_branch(text, pair),
        "automatic_final_branch": classify_answer_branch(text, pair["original_answer_spec"], pair["counterfactual_answer_spec"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    config = read_json(bundle_root / "config" / "standardized_diagnostics_config.json")
    profile = config["models"][args.model_key]
    ensure_cuda_a100(config)
    if profile["generation_backend"] != "vllm":
        raise ValueError(f"{args.model_key} is not configured for vLLM sampling")

    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != config["runtime"]["vllm_version"]:
        raise RuntimeError(f"Expected vLLM {config['runtime']['vllm_version']}, found {vllm.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(profile["model_id"], revision=profile["revision"], token=os.getenv("HF_TOKEN"))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = make_adapter(profile["adapter"], tokenizer, profile)
    pairs = {row["pair_id"]: row for row in read_jsonl(bundle_root / config["cohort_file"])}
    parent_rows = [row for row in read_jsonl(bundle_root / "data" / "model_inputs" / f"{args.model_key}_diagnostic_inputs.jsonl") if row["condition"] == "full_cot_normal"]
    checkpoint = JsonlCheckpoint(output_root / "models" / args.model_key / "checkpoints" / "04_boundary_sampling_standardized.jsonl", "sample_record_id")

    kwargs = {
        "model": profile["model_id"],
        "revision": profile["revision"],
        "tokenizer_revision": profile["revision"],
        "dtype": profile["dtype"],
        "max_model_len": int(profile["max_model_len"]),
        "gpu_memory_utilization": float(profile["gpu_memory_utilization"]),
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_num_seqs": int(profile["max_num_seqs"]),
        "trust_remote_code": False,
        "seed": int(config["seeds"]["sampling"]),
    }
    if profile.get("language_model_only"):
        kwargs["language_model_only"] = True
    llm = LLM(**kwargs)
    manifest_dir = output_root / "models" / args.model_key / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_dir / "vllm_sampling_runtime.json", {"created_at_utc": utc_now(), "model_key": args.model_key, "vllm_arguments": kwargs})

    settings = config["boundary_sampling"]
    requests = []
    for row in parent_rows:
        boundary = completed_boundary_ids(row, adapter)
        for temp in settings["temperatures"]:
            for sample_index in range(settings["samples_per_temperature"]):
                record_id = f"{args.model_key}::{row['pair_id']}::full_cot_normal::temp_{temp}::sample_{sample_index:02d}"
                if not checkpoint.missing(record_id):
                    continue
                seed = stable_seed(profile["revision"], record_id, int(config["seeds"]["sampling"]))
                params = SamplingParams(
                    n=1,
                    temperature=float(temp),
                    top_p=float(settings["top_p"]),
                    top_k=int(profile.get("top_k", -1)),
                    repetition_penalty=float(profile.get("repetition_penalty", 1.0)),
                    presence_penalty=float(profile.get("presence_penalty", 0.0)),
                    max_tokens=int(settings["max_new_tokens"]),
                    stop_token_ids=adapter.markers.terminal_ids,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                    seed=seed,
                )
                requests.append({"record_id": record_id, "row": row, "boundary": boundary, "temp": temp, "sample_index": sample_index, "seed": seed, "params": params})

    batch_size = int(profile["submission_batch_size"])
    for batch in chunks(requests, batch_size):
        prompts = [{"prompt_token_ids": item["boundary"]} for item in batch]
        params = [item["params"] for item in batch]
        started = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - started
        rows = []
        for item, output in zip(batch, outputs):
            completion = output.outputs[0]
            generated = [int(v) for v in completion.token_ids]
            parent = item["row"]
            pair = pairs[parent["pair_id"]]
            parsed = parse_sample(adapter, pair, generated, int(settings["max_new_tokens"]))
            parsed.update({
                "sample_record_id": item["record_id"],
                "created_at_utc": utc_now(),
                "model_key": args.model_key,
                "pair_id": parent["pair_id"],
                "category": parent["category"],
                "condition": parent["condition"],
                "temperature": item["temp"],
                "sample_index": item["sample_index"],
                "seed": item["seed"],
                "diagnostic_parent_id": parent["diagnostic_parent_id"],
                "batch_elapsed_seconds": elapsed,
            })
            rows.append(parsed)
        checkpoint.append(rows)


if __name__ == "__main__":
    main()
