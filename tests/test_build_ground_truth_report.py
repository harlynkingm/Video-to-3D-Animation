"""Tests for the aggregate metric in the ground-truth HTML report."""

from __future__ import annotations

import pytest

from scripts.diagnostic.build_ground_truth_report import ARKIT_BLENDSHAPE_NAMES, overall_blendshape_tracking


def _stats(correlations: list[float | None]) -> dict[str, dict]:
    return {name: {"corr": correlation} for name, correlation in zip(ARKIT_BLENDSHAPE_NAMES, correlations)}


def test_overall_blendshape_tracking_is_an_equal_weight_mean_of_valid_correlations() -> None:
    correlations = [0.25, 0.75] + [None] * (len(ARKIT_BLENDSHAPE_NAMES) - 2)

    assert overall_blendshape_tracking(_stats(correlations)) == pytest.approx(50.0)


def test_overall_blendshape_tracking_excludes_flat_channels_and_handles_no_valid_correlations() -> None:
    correlations = [0.8, None] + [None] * (len(ARKIT_BLENDSHAPE_NAMES) - 2)

    assert overall_blendshape_tracking(_stats(correlations)) == pytest.approx(80.0)
    assert overall_blendshape_tracking(_stats([None] * len(ARKIT_BLENDSHAPE_NAMES))) is None
