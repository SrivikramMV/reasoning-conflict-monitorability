

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import model_info, snapshot_download
from transformers import AutoConfig, AutoTokenizer

from adapters import ReasoningAdapter, make_adapter
from classification import bridge_evidence, classify_answer_branch, classify_visible_branch
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
from run_logging import RunLogger


def count_completed(output_root: Path) -> int:
    total = 0
    for path in output_root.glob("models/*/checkpoints/*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def pair_metadata(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "pair_index": pair["pair_index"],
        "category": pair["category"],
        "difficulty": pair["difficulty"],
        "original_question": pair["original_question"],
        "counterfactual_question": pair["counterfactual_question"],
        "original_answer": pair["original_answer"],
        "counterfactual_answer": pair["counterfactual_answer"],
        "original_answer_spec": pair["original_answer_spec"],
        "counterfactual_answer_spec": pair["counterfactual_answer_spec"],
        "pair_change_description": pair["pair_change_description"],
    }


class VllmReplication:
    def __init__(self, bundle_root: Path, output_root: Path, model_key: str) -> None:
        self.bundle_root = bundle_root
        self.output_root = output_root
        self.config = read_json(bundle_root / "config" / "cross_model_study.json")
        self.profile = self.config["models"][model_key]
        if self.profile["backend"] != "vllm":
            raise ValueError(f"{model_key} is not registered as a vLLM replication model")
        self.model_key = model_key
        self.model_root = output_root / "models" / model_key
        self.checkpoint_root = self.model_root / "checkpoints"
        self.manifest_root = self.model_root / "manifests"
        self.review_root = self.model_root / "review_packets"
        for path in (self.checkpoint_root, self.manifest_root, self.review_root):
            path.mkdir(parents=True, exist_ok=True)
        self.pairs = read_jsonl(bundle_root / self.config["benchmark_file"])
        self.pair_by_id = {row["pair_id"]: row for row in self.pairs}
        self.stability_ids = set(
            read_json(bundle_root / self.config["stability_subset_file"])["pair_ids"]
        )
        self.logger = RunLogger(
            output_root,
            int(self.config["heartbeat_seconds"]),
            int(self.config["expected_rows"]["new_rows_total"]),
        )
        if self.profile.get("tokenizer_backend") == "mistral_common":
            from transformers import MistralCommonBackend

            tokenizer_path = snapshot_download(
                repo_id=self.profile["model_id"],
                revision=self.profile["revision"],
                token=os.getenv("HF_TOKEN"),
                allow_patterns=[
                    "tekken.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                    "processor_config.json",
                ],
            )
            self.tokenizer = MistralCommonBackend.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.profile["model_id"], revision=self.profile["revision"], token=os.getenv("HF_TOKEN")
            )
        self.adapter: ReasoningAdapter = make_adapter(
            self.profile["adapter"], self.tokenizer, self.profile
        )
        self.llm: Any = None

    def preflight(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory
        runtime = self.config["runtime"]
        if transformers.__version__ != runtime["transformers_version"]:
            raise RuntimeError(
                f"Expected transformers {runtime['transformers_version']}, "
                f"found {transformers.__version__}"
            )
        if runtime["required_gpu_name_fragment"].upper() not in gpu_name.upper():
            raise RuntimeError(f"Expected an A100 runtime, found {gpu_name}")
        if memory < int(runtime["minimum_gpu_memory_bytes"]):
            raise RuntimeError(f"GPU memory is below the frozen minimum: {memory} bytes")
        remote = model_info(self.profile["model_id"], token=os.getenv("HF_TOKEN"))
        if remote.sha != self.profile["revision"]:
            self.logger.log(
                f"Hub main now points to {remote.sha}; retaining frozen revision {self.profile['revision']}",
                "WARNING",
            )
        hf_config = AutoConfig.from_pretrained(
            self.profile["model_id"], revision=self.profile["revision"], token=os.getenv("HF_TOKEN")
        )
        if self.profile.get("quantization"):
            quant = getattr(hf_config, "quantization_config", None) or {}
            method = quant.get("quant_method") if isinstance(quant, dict) else None
            if method != self.profile["quantization"]:
                raise RuntimeError(
                    f"Frozen quantization {self.profile['quantization']} does not match model config {method}"
                )
        template = self.adapter.validate_template()
        manifest = {
            "created_at_utc": utc_now(),
            "model_key": self.model_key,
            "model_profile": self.profile,
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": memory,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "template_validation": template,
            "benchmark_sha256": sha256_file(self.bundle_root / self.config["benchmark_file"]),
            "config_sha256": sha256_file(self.bundle_root / "config" / "cross_model_study.json"),
        }
        atomic_write_json(self.manifest_root / "preflight.json", manifest)
        self.logger.log(f"Preflight passed for {self.profile['model_id']} at {self.profile['revision']}")

    def load_model(self) -> None:
        import vllm
        from vllm import LLM

        expected_vllm = self.config["runtime"]["vllm_version"]
        if vllm.__version__ != expected_vllm:
            raise RuntimeError(f"Expected vLLM {expected_vllm}, found {vllm.__version__}")

        self.logger.log(f"Loading {self.profile['model_id']} with vLLM")
        started = time.time()
        kwargs: dict[str, Any] = {
            "model": self.profile["model_id"],
            "revision": self.profile["revision"],
            "tokenizer_revision": self.profile["revision"],
            "dtype": self.profile["dtype"],
            "max_model_len": int(self.profile["max_model_len"]),
            "gpu_memory_utilization": float(self.profile["gpu_memory_utilization"]),
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_seqs": int(self.profile["max_num_seqs"]),
            "trust_remote_code": False,
            "seed": int(self.config["seeds"]["primary"]),
        }
        if self.profile.get("language_model_only"):
            kwargs["language_model_only"] = True
        for optional_key in ("tokenizer_mode", "config_format", "load_format"):
            if self.profile.get(optional_key):
                kwargs[optional_key] = self.profile[optional_key]
        self.llm = LLM(**kwargs)
        atomic_write_json(
            self.manifest_root / "runtime.json",
            {
                "created_at_utc": utc_now(),
                "load_elapsed_seconds": time.time() - started,
                "model_profile": self.profile,
                "vllm_arguments": kwargs,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        )
        self.logger.log(f"Model loaded in {(time.time() - started) / 60:.1f} minutes")

    def runtime_smoke(self) -> None:

        pair = self.pairs[0]
        prompt = self.adapter.build_prompt_ids(pair["original_question"])
        clean_request = self._request(
            "__runtime_smoke__::clean",
            "primary",
            prompt,
            self.adapter.clean_response_prefix(),
            int(self.config["runtime_smoke_max_tokens"]),
            {
                **pair_metadata(pair),
                "stage": "runtime_smoke",
                "condition": "clean",
                "question_used": pair["original_question"],
            },
        )
        clean = self.generate_chunk([clean_request])[0]
        if clean["parse_status"] != "ok" or len(clean["thought_token_ids"]) < 12:
            raise RuntimeError(
                "Runtime smoke failed to recover a complete model-native reasoning span: "
                f"{clean['parse_status']}"
            )
        injected_input, response_prefix, cut = self.adapter.build_injected_input(
            prompt,
            [int(value) for value in clean["thought_token_ids"]],
            1.0,
            True,
        )
        injected_request = self._request(
            "__runtime_smoke__::complete_prefill",
            "primary",
            injected_input,
            response_prefix,
            256,
            {
                **pair_metadata(pair),
                "stage": "runtime_smoke",
                "condition": "complete_prefill",
                "question_used": pair["original_question"],
                "injection": cut,
            },
        )
        injected = self.generate_chunk([injected_request])[0]
        if injected["parse_status"] != "ok":
            raise RuntimeError(
                "Runtime smoke failed to reconstruct a complete prefilled response: "
                f"{injected['parse_status']}"
            )
        atomic_write_json(
            self.manifest_root / "runtime_smoke.json",
            {
                "created_at_utc": utc_now(),
                "model_key": self.model_key,
                "passed": True,
                "clean": clean,
                "complete_prefill": injected,
            },
        )
        self.logger.log("Runtime smoke passed for clean and complete-prefill boundaries")

    def _sampling_params(self, request: dict[str, Any]) -> Any:
        from vllm import SamplingParams

        temperature = float(request["temperature"])
        return SamplingParams(
            temperature=temperature,
            top_p=float(request["top_p"]) if temperature > 0 else 1.0,
            top_k=int(request["top_k"]) if temperature > 0 else -1,
            repetition_penalty=float(request["repetition_penalty"]),
            seed=int(request["seed"]),
            max_tokens=int(request["max_new_tokens"]),
            stop_token_ids=self.adapter.markers.terminal_ids,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def generate_chunk(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [{"prompt_token_ids": request["input_ids"]} for request in requests]
        params = [self._sampling_params(request) for request in requests]
        started = time.time()
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - started
        if len(outputs) != len(requests):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(requests)} requests")
        rows = []
        for request, output in zip(requests, outputs, strict=True):
            if not output.outputs:
                raise RuntimeError(f"No completion returned for {request['row_id']}")
            completion = output.outputs[0]
            generated = [int(value) for value in completion.token_ids]
            parsed = self.adapter.parse_response(
                request["response_prefix_ids"],
                generated,
                int(request["max_new_tokens"]),
                getattr(completion, "finish_reason", None),
                getattr(completion, "stop_reason", None),
            )
            row = {
                **request["metadata"],
                **parsed,
                "row_id": request["row_id"],
                "model_key": self.model_key,
                "model_id": self.profile["model_id"],
                "model_revision": self.profile["revision"],
                "seed_label": request["seed_label"],
                "seed": request["seed"],
                "input_token_ids": request["input_ids"],
                "input_token_count": len(request["input_ids"]),
                "input_token_sha256": sha256_ints(request["input_ids"]),
                "max_new_tokens": request["max_new_tokens"],
                "temperature": request["temperature"],
                "top_p": request["top_p"],
                "top_k": request["top_k"],
                "repetition_penalty": request["repetition_penalty"],
                "batch_size_realised": len(requests),
                "batch_elapsed_seconds": elapsed,
                "created_at_utc": utc_now(),
                "execution_status": "generated",
            }
            pair = self.pair_by_id[row["pair_id"]]
            final_branch = classify_answer_branch(
                row["final_answer"], pair["original_answer_spec"], pair["counterfactual_answer_spec"]
            )
            visible = classify_visible_branch(row["final_answer"], pair)
            row["automatic_final_branch"] = final_branch
            row["automatic_visible_branch"] = visible
            row["automatic_bridge_evidence"] = bridge_evidence(row["final_answer"], pair)
            rows.append(row)
        return rows

    def run_stage(
        self,
        stage: str,
        expected_rows: int,
        requests: list[dict[str, Any]],
        immediate_rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        checkpoint = JsonlCheckpoint(self.checkpoint_root / f"{stage}.jsonl", "row_id")
        expected_ids = {request["row_id"] for request in requests}
        expected_ids.update(row["row_id"] for row in immediate_rows or [])
        if len(expected_ids) != expected_rows:
            raise RuntimeError(
                f"{stage} request plan contains {len(expected_ids)} unique rows; "
                f"expected {expected_rows}"
            )
        unknown = checkpoint.ids - expected_ids
        if unknown:
            raise RuntimeError(f"{stage} checkpoint contains unknown row ids: {sorted(unknown)[:5]}")
        if immediate_rows:
            checkpoint.append(immediate_rows)
        pending = [request for request in requests if checkpoint.missing(request["row_id"])]
                                                                            
                                                                            
        pending.sort(
            key=lambda request: (
                int(request["max_new_tokens"]),
                len(request["input_ids"]),
                request["row_id"],
            )
        )
        self.logger.begin_stage(
            self.model_key, stage, len(checkpoint.rows), expected_rows, count_completed(self.output_root)
        )
        submission_batch_size = int(self.profile["submission_batch_size"])
        for batch in chunks(pending, submission_batch_size):
            try:
                rows = self.generate_chunk(batch)
                checkpoint.append(rows)
            except Exception as exc:
                self.logger.record_error(
                    exc, {"model": self.model_key, "stage": stage, "row_ids": [r["row_id"] for r in batch]}
                )
                raise
            generated_tokens = sum(int(row.get("generated_token_count") or 0) for row in rows)
            batch_elapsed = max(float(rows[0].get("batch_elapsed_seconds") or 0), 1e-9)
            self.logger.progress(
                len(checkpoint.rows),
                count_completed(self.output_root),
                batch[-1]["row_id"],
                batch_rows=len(rows),
                batch_generated_tokens=generated_tokens,
                batch_elapsed_seconds=batch_elapsed,
            )
            self.logger.log(
                f"{self.model_key} / {stage}: {len(checkpoint.rows)}/{expected_rows} rows"
            )
        if len(checkpoint.rows) != expected_rows:
            raise RuntimeError(
                f"{stage} row count mismatch: expected {expected_rows}, found {len(checkpoint.rows)}"
            )
        observed_ids = {str(row["row_id"]) for row in checkpoint.rows}
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            raise RuntimeError(
                f"{stage} identifier mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )
        atomic_write_json(
            self.manifest_root / f"{stage}.complete.json",
            {
                "stage": stage,
                "completed_at_utc": utc_now(),
                "row_count": len(checkpoint.rows),
                "expected_rows": expected_rows,
                "checkpoint_sha256": sha256_file(checkpoint.path),
            },
        )
        self.logger.finish_stage()
        return checkpoint.rows

    def _request(
        self,
        row_id: str,
        seed_label: str,
        input_ids: list[int],
        response_prefix_ids: list[int],
        max_new_tokens: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if len(input_ids) + max_new_tokens > int(self.profile["max_model_len"]):
            raise ValueError(
                f"{row_id} exceeds max_model_len: {len(input_ids)} + {max_new_tokens}"
            )
        seed_value = int(self.config["seeds"][seed_label])
        temperature = (
            float(self.profile["stability_temperature"])
            if seed_label.startswith("stability_")
            else float(self.profile["primary_temperature"])
        )
        return {
            "row_id": row_id,
            "seed_label": seed_label,
            "seed": stable_seed(self.profile["revision"], row_id, seed_value),
            "input_ids": input_ids,
            "response_prefix_ids": response_prefix_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": float(self.profile["top_p"]),
            "top_k": int(self.profile["top_k"]),
            "repetition_penalty": float(self.profile.get("repetition_penalty", 1.0)),
            "metadata": metadata,
        }

    def clean_requests(self) -> list[dict[str, Any]]:
        requests = []
        for pair in self.pairs:
            for role in ("original", "counterfactual"):
                question = pair[f"{role}_question"]
                row_id = f"{pair['pair_id']}::{role}::primary"
                prompt = self.adapter.build_prompt_ids(question)
                requests.append(
                    self._request(
                        row_id,
                        "primary",
                        prompt,
                        self.adapter.clean_response_prefix(),
                        int(self.config["generation_limits"]["clean"]),
                        {
                            **pair_metadata(pair),
                            "stage": "clean_controls",
                            "prompt_role": role,
                            "question_used": question,
                            "prompt_token_ids": prompt,
                            "prompt_token_sha256": sha256_ints(prompt),
                        },
                    )
                )
        return requests

    def source_map(self, clean_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        sources = {}
        semantic_flags = {}
        for row in clean_rows:
            if row["prompt_role"] != "counterfactual" or row["execution_status"] != "generated":
                continue
            structural = row["parse_status"] == "ok" and len(row["thought_token_ids"]) >= 12
            if not structural:
                continue
            pair = self.pair_by_id[row["pair_id"]]
            thought_branch = classify_visible_branch(row["thought"], pair)
            final_branch = classify_answer_branch(
                row["final_answer"],
                pair["original_answer_spec"],
                pair["counterfactual_answer_spec"],
            )
            row["source_thought_branch"] = thought_branch
            row["source_final_branch"] = final_branch
            row["automatic_source_semantically_eligible"] = (
                thought_branch["branch"] == "QSTAR" and final_branch["branch"] == "QSTAR"
            )
            semantic_flags[row["pair_id"]] = row["automatic_source_semantically_eligible"]
            sources[row["pair_id"]] = row
        rate = len(sources) / len(self.pairs)
        atomic_write_json(
            self.manifest_root / "source_structure_audit.json",
            {
                "created_at_utc": utc_now(),
                "structurally_available_sources": len(sources),
                "total_pairs": len(self.pairs),
                "structural_source_rate": rate,
                "minimum_required_rate": self.config["minimum_structural_source_rate"],
                "missing_pair_ids": sorted(set(self.pair_by_id) - set(sources)),
                "automatic_semantically_eligible_sources": sum(semantic_flags.values()),
                "automatic_semantically_ineligible_pair_ids": sorted(
                    pair_id for pair_id, eligible in semantic_flags.items() if not eligible
                ),
                "semantic_eligibility_requires_adjudication": True,
            },
        )
        if rate < float(self.config["minimum_structural_source_rate"]):
            raise RuntimeError(f"Structural source rate {rate:.3f} is below the frozen threshold")
        return sources

    def unavailable_row(self, row_id: str, pair: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "row_id": row_id,
            **pair_metadata(pair),
            **metadata,
            "model_key": self.model_key,
            "model_id": self.profile["model_id"],
            "model_revision": self.profile["revision"],
            "execution_status": "not_run_source_unavailable",
            "parse_status": "source_unavailable",
            "thought": "",
            "final_answer": "",
            "generated_token_ids": [],
            "generated_token_count": 0,
            "truncated": False,
            "created_at_utc": utc_now(),
        }

    def prefill_requests(
        self,
        sources: dict[str, dict[str, Any]],
        pair_ids: set[str],
        seed_labels: list[str],
        stage: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        requests, unavailable = [], []
        limits = self.config["generation_limits"]
        for pair in self.pairs:
            if pair["pair_id"] not in pair_ids:
                continue
            source = sources.get(pair["pair_id"])
            for seed_label in seed_labels:
                for condition in self.config["prefill_conditions"]:
                    row_id = f"{pair['pair_id']}::{condition['name']}::{seed_label}"
                    metadata = {
                        **pair_metadata(pair),
                        "stage": stage,
                        "condition": condition["name"],
                        "seed_label": seed_label,
                        "source_row_id": f"{pair['pair_id']}::counterfactual::primary",
                    }
                    if source is None:
                        unavailable.append(self.unavailable_row(row_id, pair, metadata))
                        continue
                    question = pair["original_question"]
                    if condition["answer_format"] == "answer_only":
                        question = question.rstrip() + "\n\n" + self.config["answer_only_instruction"]
                    prompt = self.adapter.build_prompt_ids(question)
                    input_ids, response_prefix, cut = self.adapter.build_injected_input(
                        prompt,
                        [int(value) for value in source["thought_token_ids"]],
                        float(condition["ratio"]),
                        bool(condition["close_reasoning"]),
                    )
                    metadata.update(
                        {
                            "question_used": question,
                            "answer_format": condition["answer_format"],
                            "source_thought": source["thought"],
                            "source_thought_token_ids": source["thought_token_ids"],
                            "automatic_source_semantically_eligible": source.get(
                                "automatic_source_semantically_eligible"
                            ),
                            "source_thought_branch": source.get("source_thought_branch"),
                            "source_final_branch": source.get("source_final_branch"),
                            "injected_reasoning": cut["decoded_prefix"],
                            "injection": cut,
                            "prompt_token_ids": prompt,
                            "prompt_token_sha256": sha256_ints(prompt),
                        }
                    )
                    requests.append(
                        self._request(
                            row_id,
                            seed_label,
                            input_ids,
                            response_prefix,
                            int(limits[condition["name"]]),
                            metadata,
                        )
                    )
        return requests, unavailable

    def grounding_requests(
        self,
        sources: dict[str, dict[str, Any]],
        pair_ids: set[str],
        seed_labels: list[str],
        placements: list[str],
        instruction_variants: list[str],
        stage: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        requests, unavailable = [], []
        for pair in self.pairs:
            if pair["pair_id"] not in pair_ids:
                continue
            source = sources.get(pair["pair_id"])
            for seed_label in seed_labels:
                for placement in placements:
                    for instruction_name in instruction_variants:
                        instruction = self.config["grounding_instructions"][instruction_name]
                        for relation in ("conflict", "matched"):
                            row_id = (
                                f"{pair['pair_id']}::{placement}::{instruction_name}::"
                                f"{relation}::{seed_label}"
                            )
                            metadata = {
                                **pair_metadata(pair),
                                "stage": stage,
                                "placement": placement,
                                "instruction_variant": instruction_name,
                                "relation": relation,
                                "seed_label": seed_label,
                                "source_row_id": f"{pair['pair_id']}::counterfactual::primary",
                            }
                            if source is None:
                                unavailable.append(self.unavailable_row(row_id, pair, metadata))
                                continue
                            active = (
                                pair["original_question"]
                                if relation == "conflict"
                                else pair["counterfactual_question"]
                            )
                            source_ids = [int(value) for value in source["thought_token_ids"]]
                            if placement == "hidden_prefill":
                                question = active.rstrip() + "\n\n" + instruction
                                prompt = self.adapter.build_prompt_ids(question)
                                input_ids, response_prefix, cut = self.adapter.build_injected_input(
                                    prompt, source_ids, 1.0, True
                                )
                            elif placement == "explicit_user_trace":
                                trace = source["thought"].strip()
                                explicit = self.config["explicit_trace"]
                                question = (
                                    active.rstrip()
                                    + f"\n\n{explicit['open_tag']}\n"
                                    + trace
                                    + f"\n{explicit['close_tag']}\n\n"
                                    + instruction
                                )
                                prompt = self.adapter.build_prompt_ids(question)
                                input_ids = prompt
                                response_prefix = self.adapter.clean_response_prefix()
                                cut = None
                            else:
                                raise ValueError(f"Unknown placement {placement}")
                            metadata.update(
                                {
                                    "question_used": question,
                                    "active_question": active,
                                    "source_thought": source["thought"],
                                    "source_thought_token_ids": source["thought_token_ids"],
                                    "automatic_source_semantically_eligible": source.get(
                                        "automatic_source_semantically_eligible"
                                    ),
                                    "source_thought_branch": source.get("source_thought_branch"),
                                    "source_final_branch": source.get("source_final_branch"),
                                    "injection": cut,
                                    "prompt_token_ids": prompt,
                                    "prompt_token_sha256": sha256_ints(prompt),
                                    "requires_trace_grounded_human_audit": True,
                                }
                            )
                            requests.append(
                                self._request(
                                    row_id,
                                    seed_label,
                                    input_ids,
                                    response_prefix,
                                    int(self.config["generation_limits"]["grounding"]),
                                    metadata,
                                )
                            )
        return requests, unavailable

    def write_review_packets(self) -> None:
        prefill_paths = [
            self.checkpoint_root / "02_primary_prefill.jsonl",
            self.checkpoint_root / "05_stability_prefill.jsonl",
        ]
        grounding_paths = [
            self.checkpoint_root / "03_hidden_grounding.jsonl",
            self.checkpoint_root / "04_explicit_grounding.jsonl",
            self.checkpoint_root / "06_stability_quote_grounding.jsonl",
        ]
        prefill_packets = []
        for path in prefill_paths:
            for row in read_jsonl(path):
                prefill_packets.append(
                    {
                        "row_id": row["row_id"],
                        "pair_id": row["pair_id"],
                        "category": row["category"],
                        "condition": row.get("condition"),
                        "seed_label": row.get("seed_label"),
                        "original_question": row["original_question"],
                        "counterfactual_question": row["counterfactual_question"],
                        "original_answer": row["original_answer"],
                        "counterfactual_answer": row["counterfactual_answer"],
                        "prefilled_reasoning": row.get("injected_reasoning", ""),
                        "generated_reasoning": row.get("thought", ""),
                        "final_answer": row.get("final_answer", ""),
                        "truncated": row.get("truncated"),
                        "execution_status": row.get("execution_status"),
                        "human_primary_label": "",
                        "human_subtype": "",
                        "human_boundary_state": "",
                        "human_final_state": "",
                        "human_confidence": "",
                        "human_rationale": "",
                    }
                )
        grounding_packets = []
        for path in grounding_paths:
            for row in read_jsonl(path):
                grounding_packets.append(
                    {
                        "row_id": row["row_id"],
                        "pair_id": row["pair_id"],
                        "category": row["category"],
                        "placement": row.get("placement"),
                        "relation": row.get("relation"),
                        "instruction_variant": row.get("instruction_variant"),
                        "seed_label": row.get("seed_label"),
                        "original_question": row["original_question"],
                        "counterfactual_question": row["counterfactual_question"],
                        "source_thought": row.get("source_thought", ""),
                        "generated_reasoning": row.get("thought", ""),
                        "final_answer": row.get("final_answer", ""),
                        "truncated": row.get("truncated"),
                        "human_transparency_label": "",
                        "quoted_trace_evidence_is_exact": "",
                        "human_confidence": "",
                        "human_notes": "",
                    }
                )
        from common import append_jsonl

        for path, packets in (
            (self.review_root / "trajectory_review.jsonl", prefill_packets),
            (self.review_root / "trace_grounding_review.jsonl", grounding_packets),
        ):
            if path.exists():
                path.unlink()
            append_jsonl(path, packets)

    def quality_gate(self) -> dict[str, Any]:
        expected = {
            "01_clean_controls": 240,
            "02_primary_prefill": 480,
            "03_hidden_grounding": 480,
            "04_explicit_grounding": 240,
            "05_stability_prefill": 240,
            "06_stability_quote_grounding": 240,
        }
        stages = {}
        hard_failures, warnings = [], []
        for stage, count in expected.items():
            path = self.checkpoint_root / f"{stage}.jsonl"
            rows = read_jsonl(path)
            generated = [row for row in rows if row.get("execution_status") == "generated"]
            stages[stage] = {
                "expected": count,
                "observed": len(rows),
                "generated": len(generated),
                "truncated": sum(bool(row.get("truncated")) for row in generated),
                "parse_status_counts": {
                    status: sum(row.get("parse_status") == status for row in generated)
                    for status in sorted({row.get("parse_status") for row in generated})
                },
            }
            if len(rows) != count:
                hard_failures.append(f"{stage}: expected {count}, observed {len(rows)}")
            unavailable = len(rows) - len(generated)
            if unavailable:
                warnings.append(f"{stage}: {unavailable} rows not generated because source was unavailable")
            truncated = stages[stage]["truncated"]
            if truncated:
                warnings.append(f"{stage}: {truncated} capped generations require review")
        gate = {
            "created_at_utc": utc_now(),
            "model_key": self.model_key,
            "passed": not hard_failures,
            "hard_failures": hard_failures,
            "warnings": warnings,
            "stages": stages,
        }
        atomic_write_json(self.model_root / "QUALITY_GATE.json", gate)
        if hard_failures:
            raise RuntimeError("; ".join(hard_failures))
        return gate

    def run(self) -> None:
        try:
            self.preflight()
            self.load_model()
            self.runtime_smoke()
            clean = self.run_stage("01_clean_controls", 240, self.clean_requests())
            sources = self.source_map(clean)
            all_ids = set(self.pair_by_id)
            requests, unavailable = self.prefill_requests(
                sources, all_ids, ["primary"], "primary_prefill"
            )
            self.run_stage("02_primary_prefill", 480, requests, unavailable)
            requests, unavailable = self.grounding_requests(
                sources,
                all_ids,
                ["primary"],
                ["hidden_prefill"],
                ["generic_compare", "quote_evidence"],
                "hidden_grounding",
            )
            self.run_stage("03_hidden_grounding", 480, requests, unavailable)
            requests, unavailable = self.grounding_requests(
                sources,
                all_ids,
                ["primary"],
                ["explicit_user_trace"],
                ["quote_evidence"],
                "explicit_grounding",
            )
            self.run_stage("04_explicit_grounding", 240, requests, unavailable)
            requests, unavailable = self.prefill_requests(
                sources,
                self.stability_ids,
                ["stability_a", "stability_b"],
                "stability_prefill",
            )
            self.run_stage("05_stability_prefill", 240, requests, unavailable)
            requests, unavailable = self.grounding_requests(
                sources,
                self.stability_ids,
                ["stability_a", "stability_b"],
                ["hidden_prefill", "explicit_user_trace"],
                ["quote_evidence"],
                "stability_quote_grounding",
            )
            self.run_stage("06_stability_quote_grounding", 240, requests, unavailable)
            self.write_review_packets()
            gate = self.quality_gate()
            atomic_write_json(
                self.model_root / "MODEL_RUN_COMPLETE.json",
                {
                    "completed_at_utc": utc_now(),
                    "model_key": self.model_key,
                    "model_id": self.profile["model_id"],
                    "model_revision": self.profile["revision"],
                    "quality_gate_passed": gate["passed"],
                },
            )
            self.logger.log(f"Completed all registered stages for {self.model_key}")
            self.logger.close("model_complete")
        except Exception as exc:
            self.logger.record_error(exc, {"model": self.model_key, "phase": "run"})
            self.logger.close("failed")
            raise
        finally:
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", choices=["qwen35_9b", "gpt_oss_20b", "ministral3_8b_reasoning_2512"], required=True)
    args = parser.parse_args()
    VllmReplication(args.bundle_root, args.output_root, args.model_key).run()


if __name__ == "__main__":
    main()
