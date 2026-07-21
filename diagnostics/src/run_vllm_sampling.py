from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from adapters import make_adapter
from classification import classify_answer_branch, classify_visible_branch
from common import (
    JsonlCheckpoint,
    atomic_write_json,
    chunks,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_ints,
    stable_seed,
    utc_now,
)
from runtime_utils import completed_boundary_ids, ensure_cuda_a100, parse_continuation


def stage_manifest(
    bundle_root: Path,
    output_root: Path,
    model_key: str,
    checkpoint: JsonlCheckpoint,
    expected: int,
    profile: dict[str, Any],
    runtime_arguments: dict[str, Any],
) -> None:
    path = output_root / "models" / model_key / "manifests" / "sampling_stage_manifest.json"
    payload = {
        "created_at_utc": utc_now(),
        "model_key": model_key,
        "stage": "sampling",
        "expected_records": expected,
        "observed_records": len(checkpoint.rows),
        "complete": len(checkpoint.rows) == expected,
        "checkpoint_file": str(checkpoint.path.relative_to(output_root)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint.path),
        "config_sha256": sha256_file(
            bundle_root / "config" / "standardized_diagnostics_config.json"
        ),
        "model_id": profile["model_id"],
        "model_revision": profile["revision"],
        "backend": "vllm",
        "runtime_arguments": runtime_arguments,
    }
    atomic_write_json(path, payload)
    if not payload["complete"]:
        raise RuntimeError(
            f"{model_key} sampling: {len(checkpoint.rows)} records, expected {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    config = read_json(
        bundle_root / "config" / "standardized_diagnostics_config.json"
    )
    profile = config["models"][args.model_key]
    ensure_cuda_a100(config)
    if profile["generation_backend"] != "vllm":
        raise ValueError(f"{args.model_key} is not configured for vLLM sampling")

    import vllm
    from vllm import LLM, SamplingParams

    required_version = config["runtime"]["vllm_version"]
    if vllm.__version__ != required_version:
        raise RuntimeError(
            f"Expected vLLM {required_version}, found {vllm.__version__}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        profile["model_id"],
        revision=profile["revision"],
        token=os.getenv("HF_TOKEN"),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = make_adapter(profile["adapter"], tokenizer, profile)
    adapter_validation = adapter.validate_template()
    pairs = {
        row["pair_id"]: row
        for row in read_jsonl(bundle_root / config["cohort_file"])
    }
    parent_rows = [
        row
        for row in read_jsonl(
            bundle_root / "data" / "model_inputs" / f"{args.model_key}_diagnostic_inputs.jsonl"
        )
        if row["condition"] == "full_cot_normal"
    ]
    checkpoint = JsonlCheckpoint(
        output_root
        / "models"
        / args.model_key
        / "checkpoints"
        / "04_boundary_sampling_standardized.jsonl",
        "sample_record_id",
    )
    settings = config["boundary_sampling"]

    llm_arguments = {
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
        llm_arguments["language_model_only"] = True
    llm = LLM(**llm_arguments)

    manifest_dir = output_root / "models" / args.model_key / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        manifest_dir / "vllm_sampling_runtime.json",
        {
            "created_at_utc": utc_now(),
            "model_key": args.model_key,
            "vllm_version": vllm.__version__,
            "vllm_arguments": llm_arguments,
            "adapter_validation": adapter_validation,
            "per_request_seeded": True,
            "exact_prompt_token_ids": True,
            "prefix_caching": True,
            "chunked_prefill": True,
        },
    )

    requests = []
    for parent in parent_rows:
        boundary = completed_boundary_ids(parent, adapter)
        if len(boundary) + int(settings["max_new_tokens"]) > int(
            profile["max_model_len"]
        ):
            raise RuntimeError(
                f"{parent['diagnostic_parent_id']} exceeds max_model_len "
                f"({len(boundary)} + {settings['max_new_tokens']} > "
                f"{profile['max_model_len']})"
            )
        for temperature in settings["temperatures"]:
            for sample_index in range(int(settings["samples_per_temperature"])):
                record_id = (
                    f"{args.model_key}::{parent['pair_id']}::full_cot_normal"
                    f"::temp_{temperature}::sample_{sample_index:02d}"
                )
                if not checkpoint.missing(record_id):
                    continue
                seed = stable_seed(
                    profile["revision"],
                    record_id,
                    int(config["seeds"]["sampling"]),
                )
                params = SamplingParams(
                    n=1,
                    temperature=float(temperature),
                    top_p=float(settings["top_p"]),
                    top_k=int(profile.get("top_k", -1)),
                    repetition_penalty=float(
                        profile.get("repetition_penalty", 1.0)
                    ),
                    presence_penalty=float(profile.get("presence_penalty", 0.0)),
                    max_tokens=int(settings["max_new_tokens"]),
                    stop_token_ids=[int(value) for value in adapter.markers.terminal_ids],
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                    seed=seed,
                )
                requests.append(
                    {
                        "record_id": record_id,
                        "parent": parent,
                        "boundary": boundary,
                        "temperature": float(temperature),
                        "sample_index": sample_index,
                        "seed": seed,
                        "params": params,
                    }
                )

    def execute(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [
            {"prompt_token_ids": item["boundary"]}
            for item in batch
        ]
        parameters = [item["params"] for item in batch]
        started = time.time()
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        elapsed = time.time() - started
        records = []
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(batch)} requests"
            )
        for item, output in zip(batch, outputs):
            completion = output.outputs[0]
            generated = [int(value) for value in completion.token_ids]
            parent = item["parent"]
            pair = pairs[parent["pair_id"]]
            parsed = parse_continuation(
                adapter,
                parent,
                generated,
                int(settings["max_new_tokens"]),
                int(settings["first_endpoint_tokens"]),
                backend_finish_reason=getattr(completion, "finish_reason", None),
                backend_stop_reason=getattr(completion, "stop_reason", None),
            )
            parsed.update(
                {
                    "sample_record_id": item["record_id"],
                    "created_at_utc": utc_now(),
                    "model_key": args.model_key,
                    "pair_id": parent["pair_id"],
                    "category": parent["category"],
                    "condition": parent["condition"],
                    "temperature": item["temperature"],
                    "top_p": float(settings["top_p"]),
                    "sample_index": item["sample_index"],
                    "seed": item["seed"],
                    "diagnostic_parent_id": parent["diagnostic_parent_id"],
                    "boundary_token_count": len(item["boundary"]),
                    "boundary_token_ids_sha256": sha256_ints(item["boundary"]),
                    "batch_elapsed_seconds": elapsed,
                    "automatic_visible_branch": classify_visible_branch(
                        parsed["final_answer"], pair
                    ),
                    "automatic_final_branch": classify_answer_branch(
                        parsed["final_answer"],
                        pair["original_answer_spec"],
                        pair["counterfactual_answer_spec"],
                    ),
                }
            )
            if parsed["generated_token_count"] == 0:
                raise RuntimeError(f"Empty sampling output for {item['record_id']}")
            records.append(parsed)
        return records

    preflight_path = manifest_dir / "sampling_preflight.json"
    if checkpoint.rows and not preflight_path.exists():
        first_rows = checkpoint.rows[:2]
        if not all(row.get("generated_token_ids") for row in first_rows):
            raise RuntimeError("Stored sampling checkpoint fails resumed preflight")
        atomic_write_json(
            preflight_path,
            {
                "created_at_utc": utc_now(),
                "passed": True,
                "record_ids": [row["sample_record_id"] for row in first_rows],
                "nonempty_outputs": True,
                "per_request_seeds": [row["seed"] for row in first_rows],
                "first_endpoint_token_limit": int(settings["first_endpoint_tokens"]),
                "resumed_from_checkpoint": True,
            },
        )
    if requests and not preflight_path.exists():
        preflight_batch = requests[:2]
        records = execute(preflight_batch)
        checkpoint.append(records)
        atomic_write_json(
            preflight_path,
            {
                "created_at_utc": utc_now(),
                "passed": True,
                "record_ids": [row["sample_record_id"] for row in records],
                "nonempty_outputs": True,
                "per_request_seeds": [row["seed"] for row in records],
                "first_endpoint_token_limit": int(
                    settings["first_endpoint_tokens"]
                ),
            },
        )
        requests = requests[2:]

    for batch in chunks(requests, int(profile["submission_batch_size"])):
        checkpoint.append(execute(batch))

    expected = (
        len(parent_rows)
        * len(settings["temperatures"])
        * int(settings["samples_per_temperature"])
    )
    stage_manifest(
        bundle_root,
        output_root,
        args.model_key,
        checkpoint,
        expected,
        profile,
        llm_arguments,
    )


if __name__ == "__main__":
    main()
