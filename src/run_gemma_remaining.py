

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoProcessor

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


class GemmaExplicitTraceRun:
    def __init__(self, bundle_root: Path, output_root: Path) -> None:
        self.bundle_root = bundle_root
        self.output_root = output_root
        self.config = read_json(bundle_root / "config" / "cross_model_study.json")
        self.model_key = "gemma_e2b"
        self.profile = self.config["models"][self.model_key]
        self.model_root = output_root / "models" / self.model_key
        self.checkpoint_root = self.model_root / "checkpoints"
        self.manifest_root = self.model_root / "manifests"
        self.review_root = self.model_root / "review_packets"
        for path in (self.checkpoint_root, self.manifest_root, self.review_root):
            path.mkdir(parents=True, exist_ok=True)
        self.pairs = read_jsonl(bundle_root / self.config["benchmark_file"])
        self.pair_by_id = {pair["pair_id"]: pair for pair in self.pairs}
        source_rows = read_jsonl(bundle_root / self.config["gemma_source_file"])
        self.sources = {row["pair_id"]: row for row in source_rows}
        self.logger = RunLogger(
            output_root,
            int(self.config["heartbeat_seconds"]),
            int(self.config["expected_rows"]["new_rows_total"]),
        )
        self.processor: Any = None
        self.adapter: ReasoningAdapter | None = None
        self.model: Any = None

    def preflight(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        runtime = self.config["runtime"]
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory
        if runtime["required_gpu_name_fragment"].upper() not in gpu_name.upper():
            raise RuntimeError(f"Expected an A100 runtime, found {gpu_name}")
        if memory < int(runtime["minimum_gpu_memory_bytes"]):
            raise RuntimeError(f"GPU memory is below the frozen minimum: {memory} bytes")
        if transformers.__version__ != runtime["transformers_version"]:
            raise RuntimeError(
                f"Expected transformers {runtime['transformers_version']}, "
                f"found {transformers.__version__}"
            )
        if set(self.sources) != set(self.pair_by_id):
            missing = sorted(set(self.pair_by_id) - set(self.sources))
            extra = sorted(set(self.sources) - set(self.pair_by_id))
            raise RuntimeError(f"Gemma source mismatch; missing={missing[:5]}, extra={extra[:5]}")
        remote = model_info(self.profile["model_id"], token=os.getenv("HF_TOKEN"))
        if remote.sha != self.profile["revision"]:
            self.logger.log(
                f"Hub main now points to {remote.sha}; retaining frozen revision "
                f"{self.profile['revision']}",
                "WARNING",
            )
        self.processor = AutoProcessor.from_pretrained(
            self.profile["model_id"],
            revision=self.profile["revision"],
            token=os.getenv("HF_TOKEN"),
        )
        self.adapter = make_adapter("gemma4", self.processor, self.profile)
        template = self.adapter.validate_template()
        atomic_write_json(
            self.manifest_root / "preflight.json",
            {
                "created_at_utc": utc_now(),
                "model_key": self.model_key,
                "model_profile": self.profile,
                "gpu_name": gpu_name,
                "gpu_total_memory_bytes": memory,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "template_validation": template,
                "benchmark_sha256": sha256_file(
                    self.bundle_root / self.config["benchmark_file"]
                ),
                "gemma_source_sha256": sha256_file(
                    self.bundle_root / self.config["gemma_source_file"]
                ),
            },
        )
        self.logger.log("Gemma explicit-trace preflight passed")

    def load_model(self) -> None:
        assert self.processor is not None
        self.logger.log(f"Loading {self.profile['model_id']} with Transformers")
        started = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.profile["model_id"],
            revision=self.profile["revision"],
            token=os.getenv("HF_TOKEN"),
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        atomic_write_json(
            self.manifest_root / "runtime.json",
            {
                "created_at_utc": utc_now(),
                "load_elapsed_seconds": time.time() - started,
                "model_profile": self.profile,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        )
        self.logger.log(f"Gemma loaded in {(time.time() - started) / 60:.1f} minutes")

    def requests(self) -> list[dict[str, Any]]:
        assert self.adapter is not None
        requests = []
        explicit = self.config["explicit_trace"]
        instruction_name = explicit["instruction_variant"]
        instruction = self.config["grounding_instructions"][instruction_name]
        for pair in self.pairs:
            source = self.sources[pair["pair_id"]]
            trace = source["thought"].strip()
            for relation in ("conflict", "matched"):
                active = (
                    pair["original_question"]
                    if relation == "conflict"
                    else pair["counterfactual_question"]
                )
                question = (
                    active.rstrip()
                    + f"\n\n{explicit['open_tag']}\n"
                    + trace
                    + f"\n{explicit['close_tag']}\n\n"
                    + instruction
                )
                prompt = self.adapter.build_prompt_ids(question)
                row_id = (
                    f"{pair['pair_id']}::explicit_user_trace::{instruction_name}::"
                    f"{relation}::primary"
                )
                requests.append(
                    {
                        "row_id": row_id,
                        "input_ids": prompt,
                        "response_prefix_ids": self.adapter.clean_response_prefix(),
                        "metadata": {
                            **pair_metadata(pair),
                            "stage": "explicit_trace_grounding",
                            "placement": "explicit_user_trace",
                            "relation": relation,
                            "instruction_variant": instruction_name,
                            "seed_label": "greedy",
                            "question_used": question,
                            "active_question": active,
                            "source_row_id": source["source_row_id"],
                            "source_thought": source["thought"],
                            "source_thought_token_ids": source["thought_token_ids"],
                            "source_final_answer": source["final_answer"],
                            "source_clean_final_correct": source["final_correct"],
                            "source_clean_thought_supports_qstar": source[
                                "thought_supports_expected"
                            ],
                            "requires_trace_grounded_human_audit": True,
                            "prompt_token_ids": prompt,
                            "prompt_token_sha256": sha256_ints(prompt),
                        },
                    }
                )
        return requests

    @staticmethod
    def trim_generated(ids: list[int], terminal_ids: set[int]) -> list[int]:
        for index, value in enumerate(ids):
            if value in terminal_ids:
                return ids[: index + 1]
        return ids

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert self.processor is not None and self.adapter is not None and self.model is not None
        pad = self.adapter.markers.pad_token_id
        device = next(self.model.parameters()).device
        width = max(len(request["input_ids"]) for request in requests)
        input_tensor = torch.full(
            (len(requests), width), pad, dtype=torch.long, device=device
        )
        attention = torch.zeros_like(input_tensor)
        for index, request in enumerate(requests):
            values = torch.tensor(request["input_ids"], dtype=torch.long, device=device)
            input_tensor[index, -len(values) :] = values
            attention[index, -len(values) :] = 1
        max_new_tokens = int(self.config["generation_limits"]["grounding"])
        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.adapter.markers.terminal_ids,
                pad_token_id=pad,
                use_cache=True,
            )
        elapsed = time.time() - started
        rows = []
        terminals = set(self.adapter.markers.terminal_ids)
        for index, request in enumerate(requests):
            generated = [int(value) for value in output[index, width:].tolist()]
            generated = self.trim_generated(generated, terminals)
            parsed = self.adapter.parse_response(
                request["response_prefix_ids"],
                generated,
                max_new_tokens,
                "length" if len(generated) >= max_new_tokens else "stop",
                None,
            )
            pair = self.pair_by_id[request["metadata"]["pair_id"]]
            row = {
                **request["metadata"],
                **parsed,
                "row_id": request["row_id"],
                "model_key": self.model_key,
                "model_id": self.profile["model_id"],
                "model_revision": self.profile["revision"],
                "input_token_ids": request["input_ids"],
                "input_token_count": len(request["input_ids"]),
                "input_token_sha256": sha256_ints(request["input_ids"]),
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "batch_size_realised": len(requests),
                "batch_elapsed_seconds": elapsed,
                "created_at_utc": utc_now(),
                "execution_status": "generated",
            }
            row["automatic_final_branch"] = classify_answer_branch(
                row["final_answer"],
                pair["original_answer_spec"],
                pair["counterfactual_answer_spec"],
            )
            row["automatic_visible_branch"] = classify_visible_branch(row["final_answer"], pair)
            row["automatic_bridge_evidence"] = bridge_evidence(row["final_answer"], pair)
            rows.append(row)
        del output, input_tensor, attention
        return rows

    def write_review_packet(self, rows: list[dict[str, Any]]) -> None:
        from common import append_jsonl

        path = self.review_root / "trace_grounding_review.jsonl"
        if path.exists():
            path.unlink()
        packets = []
        for row in rows:
            packets.append(
                {
                    "row_id": row["row_id"],
                    "pair_id": row["pair_id"],
                    "category": row["category"],
                    "placement": row["placement"],
                    "relation": row["relation"],
                    "instruction_variant": row["instruction_variant"],
                    "seed_label": row["seed_label"],
                    "original_question": row["original_question"],
                    "counterfactual_question": row["counterfactual_question"],
                    "source_thought": row["source_thought"],
                    "generated_reasoning": row["thought"],
                    "final_answer": row["final_answer"],
                    "truncated": row["truncated"],
                    "human_transparency_label": "",
                    "quoted_trace_evidence_is_exact": "",
                    "human_confidence": "",
                    "human_notes": "",
                }
            )
        append_jsonl(path, packets)

    def run(self) -> None:
        stage = "01_explicit_trace_grounding"
        expected = int(self.config["expected_rows"]["gemma_explicit_trace_grounding"])
        try:
            self.preflight()
            self.load_model()
            requests = self.requests()
            expected_ids = {request["row_id"] for request in requests}
            if len(expected_ids) != expected:
                raise RuntimeError(f"Gemma request plan has {len(expected_ids)} rows, expected {expected}")
            checkpoint = JsonlCheckpoint(self.checkpoint_root / f"{stage}.jsonl", "row_id")
            unknown = checkpoint.ids - expected_ids
            if unknown:
                raise RuntimeError(f"Gemma checkpoint contains unknown ids: {sorted(unknown)[:5]}")
            pending = [request for request in requests if checkpoint.missing(request["row_id"])]
            self.logger.begin_stage(
                self.model_key,
                stage,
                len(checkpoint.rows),
                expected,
                count_completed(self.output_root),
            )
            for batch in chunks(pending, int(self.profile["batch_size"])):
                rows = self.generate_batch(batch)
                checkpoint.append(rows)
                generated_tokens = sum(
                    int(row.get("generated_token_count") or 0) for row in rows
                )
                batch_elapsed = max(
                    float(rows[0].get("batch_elapsed_seconds") or 0), 1e-9
                )
                self.logger.progress(
                    len(checkpoint.rows),
                    count_completed(self.output_root),
                    batch[-1]["row_id"],
                    batch_rows=len(rows),
                    batch_generated_tokens=generated_tokens,
                    batch_elapsed_seconds=batch_elapsed,
                )
                self.logger.log(
                    f"{self.model_key} / {stage}: {len(checkpoint.rows)}/{expected} rows"
                )
            observed_ids = {str(row["row_id"]) for row in checkpoint.rows}
            if observed_ids != expected_ids:
                raise RuntimeError("Gemma explicit-trace checkpoint does not match the frozen row plan")
            atomic_write_json(
                self.manifest_root / f"{stage}.complete.json",
                {
                    "stage": stage,
                    "completed_at_utc": utc_now(),
                    "row_count": len(checkpoint.rows),
                    "expected_rows": expected,
                    "checkpoint_sha256": sha256_file(checkpoint.path),
                },
            )
            self.logger.finish_stage()
            self.write_review_packet(checkpoint.rows)
            generated = [row for row in checkpoint.rows if row["execution_status"] == "generated"]
            gate = {
                "created_at_utc": utc_now(),
                "model_key": self.model_key,
                "passed": len(checkpoint.rows) == expected,
                "expected": expected,
                "observed": len(checkpoint.rows),
                "truncated": sum(bool(row.get("truncated")) for row in generated),
                "parse_status_counts": {
                    status: sum(row.get("parse_status") == status for row in generated)
                    for status in sorted({row.get("parse_status") for row in generated})
                },
            }
            atomic_write_json(self.model_root / "QUALITY_GATE.json", gate)
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
            self.logger.close("model_complete")
        except Exception as exc:
            self.logger.record_error(exc, {"model": self.model_key, "phase": "run"})
            self.logger.close("failed")
            raise
        finally:
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    GemmaExplicitTraceRun(args.bundle_root, args.output_root).run()


if __name__ == "__main__":
    main()
