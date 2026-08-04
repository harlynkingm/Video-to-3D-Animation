"""Terminal progress reporting for the pipeline's per-frame stages.

Stage 1 (SAM 3.1 tracking), stage 2 (GVHMR), stage 4 (HaMeR), and stage 7
(contact detection) each loop once per video frame -- `frame_progress` wraps
that loop with a single self-overwriting terminal line showing percent
complete, frames done/total, and a remaining-time estimate (tqdm smooths the
per-frame rate itself, so early GPU-warmup frames don't skew the estimate).

Stage 3 (depth) and stage 6 (scale) run once on a single anchor frame -- no
per-frame progress is possible, so `report_single_shot` prints a start/done
line with elapsed time instead.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")

# tqdm's {remaining} token already renders as MM:SS (or H:MM:SS past an hour).
# `{{...}}` braces are tqdm's own placeholders, left literal through this
# module's one `.format(unit=...)` call; only `{unit}` is substituted here.
_BAR_FORMAT = "[{{desc}}] {{percentage:3.0f}}% | {{n_fmt}}/{{total_fmt}} {unit}s | {{elapsed}} elapsed | {{remaining}} remaining"


def frame_progress(iterable: Iterable[T], total: int, label: str, unit: str = "frame") -> Iterable[T]:
    """Wrap a per-item loop's iterable with a terminal progress line. `label`
    identifies the stage (and, for stage 1, which of the two tracking
    passes); `unit` customizes the counted item for a caller that isn't
    counting video frames."""
    return tqdm(iterable, total=total, desc=label, bar_format=_BAR_FORMAT.format(unit=unit), unit=unit)


@contextmanager
def report_single_shot(label: str) -> Iterator[None]:
    """For stages with no per-frame loop: prints a start line, then a done line
    with elapsed time in MM:SS once the wrapped block finishes."""
    print(f"[{label}] running...")
    start = time.perf_counter()
    try:
        yield
    finally:
        minutes, seconds = divmod(int(time.perf_counter() - start), 60)
        if minutes == 0 and seconds == 0:
            print(f"[{label}] done")
        else:
            print(f"[{label}] done in {minutes:02d}:{seconds:02d}")
