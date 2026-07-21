

from __future__ import annotations

from pathlib import Path

from common import read_json
from run_gemma_remaining import GemmaExplicitTraceRun


class FinalGemmaRecovery(GemmaExplicitTraceRun):
    def __init__(self, bundle_root: Path, output_root: Path) -> None:
        super().__init__(bundle_root, output_root)
        recovery = read_json(bundle_root / self.config["recovery_manifest_file"])
        stage = recovery["models"][self.model_key]["01_explicit_trace_grounding"]
        self.recovery_ids = {str(row_id) for row_id in stage}

    def requests(self) -> list[dict]:
        planned = super().requests()
        selected = [row for row in planned if row["row_id"] in self.recovery_ids]
        observed = {row["row_id"] for row in selected}
        if observed != self.recovery_ids:
            missing = sorted(self.recovery_ids - observed)
            raise RuntimeError(f"Gemma recovery plan is missing row ids: {missing[:5]}")
        return selected


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    FinalGemmaRecovery(args.bundle_root, args.output_root).run()


if __name__ == "__main__":
    main()
