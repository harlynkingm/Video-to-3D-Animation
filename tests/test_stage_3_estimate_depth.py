"""Stage 3 regression test: runs real Depth-Anything-3 (DA3METRIC-LARGE) on
the anchor frame `mask_and_track` resolved, and checks the resulting metric
depth map looks correct -- no NaN/Inf, and depth values in a physically
plausible range for a person standing a few meters from a handheld/tripod
camera. Also checks the optional PLY point-cloud preview (on for these tests,
see conftest.py's `dump_depth_preview=True`) is structurally valid. Needs a
CUDA GPU -- skipped automatically otherwise (see conftest.py). The checkpoint
itself auto-downloads on first use, so no manual setup is needed for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MIN_PLAUSIBLE_DEPTH_M = 0.1
MAX_PLAUSIBLE_DEPTH_M = 50.0


def test_depth_has_no_nan_or_inf(stage_3_result):
    depth = np.load(stage_3_result["anchor_depth"])
    assert depth.ndim == 2
    assert not np.isnan(depth).any()
    assert not np.isinf(depth).any()


def test_depth_values_are_physically_plausible(stage_3_result):
    depth = np.load(stage_3_result["anchor_depth"])
    assert depth.min() > 0.0
    assert MIN_PLAUSIBLE_DEPTH_M < np.median(depth) < MAX_PLAUSIBLE_DEPTH_M


def test_pointcloud_preview_is_a_valid_colored_ply(stage_3_result):
    ply_path = Path(stage_3_result["anchor_pointcloud_preview"])
    assert ply_path.exists()

    lines = ply_path.read_text().splitlines()
    assert lines[0] == "ply"
    vertex_count = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])
    assert vertex_count > 0

    header_end = lines.index("end_header")
    first_row = lines[header_end + 1].split()
    assert len(first_row) == 6  # x y z r g b
    r, g, b = (int(v) for v in first_row[3:])
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
