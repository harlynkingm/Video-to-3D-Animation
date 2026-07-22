"""Shared harness so each pipeline stage file only has to contain its own logic.

A stage file should look like:

    from .pipeline_stage_base import cli_entrypoint
    from .progress_tracker import StageName

    def run(progress: ProgressRecord) -> dict[str, str]:
        ...  # the actual stage logic
        return {"some_output": "path/to/it"}

    if __name__ == "__main__":
        cli_entrypoint(run, stage_name=StageName.STAGE_0_INGEST)
"""

import argparse
import sys
import traceback
from typing import Callable

from .progress_tracker import ProgressRecord, StageName, StageStatus


def cli_entrypoint(run: Callable[[ProgressRecord], dict[str, str]], stage_name: StageName) -> None:
    parser = argparse.ArgumentParser(description=f"Run the {stage_name} stage")
    parser.add_argument("--progress-dir", required=True, help="Path to the run directory containing progress.json")
    args = parser.parse_args()

    progress = ProgressRecord.load(args.progress_dir)

    if not progress.dependencies_met(stage_name):
        print(f"{stage_name}: dependencies not met, aborting", file=sys.stderr)
        sys.exit(1)

    progress.mark_progress(stage_name, StageStatus.RUNNING)
    try:
        outputs = run(progress)
    except Exception:
        progress.mark_progress(stage_name, StageStatus.FAILED, error=traceback.format_exc())
        raise

    progress.mark_progress(stage_name, StageStatus.COMPLETE, outputs=outputs)
