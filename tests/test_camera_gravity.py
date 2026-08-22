"""Unit tests for `camera_gravity`: the camera-space gravity estimate that
replaces the pipeline's former unconditional "the camera is perfectly level"
assumption.

Synthetic scenes rather than fixture footage: a scene built from 3D segments
that are vertical *by construction* has an exactly known answer, which real
footage never does (the real-clip numbers this module is calibrated against
are recorded in its own module docstring instead).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline.algorithms.camera_gravity import (
    LEVEL_CAMERA_UP,
    angle_between_degrees,
    estimate_camera_up,
)

_K = np.array([[900.0, 0.0, 512.0], [0.0, 900.0, 384.0], [0.0, 0.0, 1.0]])
_IMAGE_SHAPE = (768, 1024)


def _render_vertical_scene(up: np.ndarray, n_poles: int = 14, seed: int = 0) -> np.ndarray:
    """An image of `n_poles` 3D segments all pointing along `up`, drawn as
    seen by a camera with `_K`. Every segment is a true scene vertical, so a
    correct estimator must return `up` itself."""
    rng = np.random.default_rng(seed)
    up = up / np.linalg.norm(up)
    image = np.zeros(_IMAGE_SHAPE, dtype=np.uint8)
    for _ in range(n_poles):
        base = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.5), rng.uniform(4.0, 9.0)])
        top = base + up * rng.uniform(1.5, 3.0)
        projected = (np.stack([base, top]) @ _K.T)
        projected = projected[:, :2] / projected[:, 2:]
        cv2.line(image, tuple(projected[0].astype(int)), tuple(projected[1].astype(int)), 255, 3)
    return image


def _write_frames(tmp_path, images: list[np.ndarray]) -> list:
    paths = []
    for index, image in enumerate(images):
        path = tmp_path / f"{index:06d}.jpg"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


@pytest.mark.parametrize("pitch_degrees", [0.0, 8.0, 17.0, -12.0])
def test_estimate_camera_up_recovers_a_known_tilt(tmp_path, pitch_degrees):
    """A camera pitched up or down still reports the scene's own vertical."""
    radians = np.radians(pitch_degrees)
    # Pitching the camera about its own X axis tips the scene vertical out of
    # camera -Y by the same angle, into Z.
    up = np.array([0.0, -np.cos(radians), np.sin(radians)])
    frames = _write_frames(tmp_path, [_render_vertical_scene(up, seed=s) for s in range(30)])

    estimated = estimate_camera_up(frames, _K)

    assert estimated is not None
    assert angle_between_degrees(estimated, up) < 1.0


def test_estimate_camera_up_returns_none_without_vertical_structure(tmp_path):
    """A scene with nothing vertical in it must not invent a tilt: `None`
    leaves callers on the level-camera assumption they already had."""
    rng = np.random.default_rng(1)
    blank = [np.zeros(_IMAGE_SHAPE, dtype=np.uint8) for _ in range(6)]
    for image in blank:
        for _ in range(12):  # horizontal clutter only
            y = int(rng.uniform(50, _IMAGE_SHAPE[0] - 50))
            cv2.line(image, (40, y), (_IMAGE_SHAPE[1] - 40, y + int(rng.uniform(-4, 4))), 255, 3)

    assert estimate_camera_up(_write_frames(tmp_path, blank), _K) is None


def test_estimate_camera_up_returns_none_for_too_few_frames(tmp_path):
    frames = _write_frames(tmp_path, [_render_vertical_scene(LEVEL_CAMERA_UP)])

    assert estimate_camera_up(frames, _K) is None


def test_estimate_camera_up_rejects_an_inconsistent_clip(tmp_path):
    """Windows that disagree wildly about which way is up are not measuring
    one scene vertical, and a median across them would be meaningless."""
    tilts = np.radians([-70.0, -25.0, 0.0, 30.0, 65.0, 85.0])
    frames = _write_frames(tmp_path, [
        _render_vertical_scene(np.array([np.sin(t), -np.cos(t), 0.0]), seed=i)
        for i, t in enumerate(tilts) for _ in range(5)
    ])

    assert estimate_camera_up(frames, _K) is None


def test_estimate_camera_up_survives_a_minority_of_wrong_directions(tmp_path):
    """The subject's own limbs produce long, convincingly vertical segments
    that are not scene verticals. The per-window median is what keeps those
    windows from moving the answer."""
    up = np.array([0.0, -np.cos(np.radians(15.0)), np.sin(np.radians(15.0))])
    misleading = np.array([np.sin(np.radians(40.0)), -np.cos(np.radians(40.0)), 0.0])
    # Contiguous blocks, so the last two of the six sample windows see nothing
    # but the misleading direction rather than a mixture of both.
    images = [_render_vertical_scene(up, seed=s) for s in range(20)]
    images += [_render_vertical_scene(misleading, seed=100 + s) for s in range(10)]

    estimated = estimate_camera_up(_write_frames(tmp_path, images), _K)

    assert estimated is not None
    assert angle_between_degrees(estimated, up) < 1.0
