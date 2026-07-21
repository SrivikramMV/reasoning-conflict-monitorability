

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import atomic_write_json, read_jsonl, utc_now
from run_replication_model import VllmReplication


class FinalQwenReplication(VllmReplication):
    def _request(
        self,
        row_id: str,
        seed_label: str,
        input_ids: list[int],
        response_prefix_ids: list[int],
        max_new_tokens: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if max_new_tokens == int(self.config["generation_limits"]["clean"]):
            max_new_tokens = int(self.profile["study_max_new_tokens"])
        return super()._request(
            row_id,
            seed_label,
            input_ids,
            response_prefix_ids,
            max_new_tokens,
            metadata,
        )

    def _sampling_params(self, request: dict[str, Any]) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            temperature=float(request["temperature"]),
            top_p=float(request["top_p"]),
            top_k=int(request["top_k"]),
            presence_penalty=float(self.profile.get("presence_penalty", 0.0)),
            repetition_penalty=float(request["repetition_penalty"]),
            seed=int(request["seed"]),
            max_tokens=int(request["max_new_tokens"]),
            stop_token_ids=self.adapter.markers.terminal_ids,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def generate_chunk(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().generate_chunk(requests)
        for row in rows:
            row["presence_penalty"] = float(self.profile.get("presence_penalty", 0.0))
        return rows

    def preflight(self) -> None:
        super().preflight()
        pair = self.pairs[0]
        example_questions = {
            "clean": pair["original_question"],
            "explicit_trace_placeholder": (
                pair["original_question"].rstrip()
                + "\n\n<scratchpad>\n[MODEL-GENERATED COUNTERFACTUAL TRACE]\n"
                + "</scratchpad>\n\n"
                + self.config["grounding_instructions"]["quote_evidence"]
            ),
        }
        decoded = {}
        for name, question in example_questions.items():
            ids = self.adapter.build_prompt_ids(question)
            decoded[name] = {
                "question": question,
                "prompt_token_ids": ids,
                "decoded_prompt": self.tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            }
        atomic_write_json(
            self.manifest_root / "rendered_prompt_audit.json",
            {
                "created_at_utc": utc_now(),
                "model_key": self.model_key,
                "examples": decoded,
            },
        )

    def quality_gate(self) -> dict[str, Any]:
        gate = super().quality_gate()
        thresholds = {
            "01_clean_controls": 0.95,
            "02_primary_prefill": 0.90,
            "03_hidden_grounding": 0.90,
            "05_stability_prefill": 0.90,
        }
        strict_rates = {}
        for stage, minimum in thresholds.items():
            rows = read_jsonl(self.checkpoint_root / f"{stage}.jsonl")
            generated = [row for row in rows if row.get("execution_status") == "generated"]
            strict = [
                row
                for row in generated
                if row.get("parse_status") == "ok" and not bool(row.get("truncated"))
            ]
            strict_rates[stage] = {
                "strict_usable": len(strict),
                "generated": len(generated),
                "expected": len(rows),
                "strict_rate_over_expected": len(strict) / len(rows) if rows else 0.0,
                "minimum_required": minimum,
            }
        gate["confirmatory_thresholds"] = strict_rates
        gate["analysis_ready"] = all(
            values["strict_rate_over_expected"] >= values["minimum_required"]
            for values in strict_rates.values()
        )
        gate["explicit_conditions_are_reported_separately"] = True
        atomic_write_json(self.model_root / "QUALITY_GATE.json", gate)
        return gate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    FinalQwenReplication(args.bundle_root, args.output_root, "qwen35_9b").run()


if __name__ == "__main__":
    main()
