

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from common import atomic_write_json, read_json, read_jsonl, utc_now
from run_replication_model import VllmReplication


class FinalGptRecovery(VllmReplication):
    def __init__(self, bundle_root: Path, output_root: Path) -> None:
        super().__init__(bundle_root, output_root, "gpt_oss_20b")
        recovery = read_json(bundle_root / self.config["recovery_manifest_file"])
        self.recovery = {
            stage: {str(row_id) for row_id in row_ids}
            for stage, row_ids in recovery["models"][self.model_key].items()
        }

    def run_selected(
        self,
        stage: str,
        requests: list[dict[str, Any]],
        unavailable: list[dict[str, Any]],
    ) -> None:
        wanted = self.recovery[stage]
        selected_requests = [row for row in requests if row["row_id"] in wanted]
        selected_unavailable = [row for row in unavailable if row["row_id"] in wanted]
        observed = {row["row_id"] for row in selected_requests}
        observed.update(row["row_id"] for row in selected_unavailable)
        if observed != wanted:
            missing = sorted(wanted - observed)
            raise RuntimeError(f"{stage} recovery plan is missing row ids: {missing[:5]}")
        self.run_stage(stage, len(wanted), selected_requests, selected_unavailable)

    def recovery_quality_gate(self) -> dict[str, Any]:
        stages = {}
        hard_failures = []
        for stage, expected_ids in self.recovery.items():
            rows = read_jsonl(self.checkpoint_root / f"{stage}.jsonl")
            observed = {str(row["row_id"]) for row in rows}
            generated = [row for row in rows if row.get("execution_status") == "generated"]
            strict = [
                row
                for row in generated
                if row.get("parse_status") == "ok" and not bool(row.get("truncated"))
            ]
            stages[stage] = {
                "expected": len(expected_ids),
                "observed": len(rows),
                "strict_usable": len(strict),
                "truncated": sum(bool(row.get("truncated")) for row in generated),
                "parse_status_counts": {
                    status: sum(row.get("parse_status") == status for row in generated)
                    for status in sorted({row.get("parse_status") for row in generated})
                },
            }
            if observed != expected_ids:
                hard_failures.append(f"{stage}: row identity mismatch")
        gate = {
            "created_at_utc": utc_now(),
            "model_key": self.model_key,
            "passed": not hard_failures,
            "analysis_ready": not hard_failures
            and all(stage["strict_usable"] == stage["expected"] for stage in stages.values()),
            "hard_failures": hard_failures,
            "stages": stages,
        }
        atomic_write_json(self.model_root / "QUALITY_GATE.json", gate)
        return gate

    def run(self) -> None:
        try:
            self.preflight()
            self.load_model()
            self.runtime_smoke()
            clean = read_jsonl(self.bundle_root / self.config["gpt_source_file"])
            sources = self.source_map(clean)
            all_ids = set(self.pair_by_id)

            requests, unavailable = self.prefill_requests(
                sources, all_ids, ["primary"], "primary_prefill"
            )
            self.run_selected("02_primary_prefill", requests, unavailable)

            requests, unavailable = self.grounding_requests(
                sources,
                all_ids,
                ["primary"],
                ["hidden_prefill"],
                ["generic_compare", "quote_evidence"],
                "hidden_grounding",
            )
            self.run_selected("03_hidden_grounding", requests, unavailable)

            requests, unavailable = self.grounding_requests(
                sources,
                all_ids,
                ["primary"],
                ["explicit_user_trace"],
                ["quote_evidence"],
                "explicit_grounding",
            )
            self.run_selected("04_explicit_grounding", requests, unavailable)

            requests, unavailable = self.prefill_requests(
                sources,
                self.stability_ids,
                ["stability_a", "stability_b"],
                "stability_prefill",
            )
            self.run_selected("05_stability_prefill", requests, unavailable)

            requests, unavailable = self.grounding_requests(
                sources,
                self.stability_ids,
                ["stability_a", "stability_b"],
                ["hidden_prefill", "explicit_user_trace"],
                ["quote_evidence"],
                "stability_quote_grounding",
            )
            self.run_selected("06_stability_quote_grounding", requests, unavailable)

            self.write_review_packets()
            gate = self.recovery_quality_gate()
            atomic_write_json(
                self.model_root / "MODEL_RUN_COMPLETE.json",
                {
                    "completed_at_utc": utc_now(),
                    "model_key": self.model_key,
                    "model_id": self.profile["model_id"],
                    "model_revision": self.profile["revision"],
                    "quality_gate_passed": gate["passed"],
                    "analysis_ready": gate["analysis_ready"],
                },
            )
            self.logger.close("model_complete")
        except Exception as exc:
            self.logger.record_error(exc, {"model": self.model_key, "phase": "recovery"})
            self.logger.close("failed")
            raise
        finally:
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    FinalGptRecovery(args.bundle_root, args.output_root).run()


if __name__ == "__main__":
    main()
