"""Offline, one-time: builds `body_models/arkit/face_bases.npz`, the fixed
basis the runtime ARKit weight solve
(`pipeline.algorithms.face.face_blendshapes.solve_arkit_weights`) needs to
convert the tracked FLAME landmark position into `NoseSneerLeft/Right`/
`CheekSquintLeft/Right`/`EyeSquintLeft/Right` weights, the only 6 ARKit
channels this project's own hand-derived Group-S solve still covers (see
that module's own docstring for why). Never runs in the pipeline itself,
like `scripts/convert_*_checkpoint.py`, this is a setup-time tool. Its
output is gitignored (`body_models/*`, a derived work of SMPL-X/FLAME),
generate it locally, once, after `body_models/ict_facekit/` is populated
(see README Setup).

Landmark-space, not mesh-space: our own tracked signal
(`face_landmark_fit.fit_clip`) only ever measures FLAME's 51 static
landmarks, so any detail a fitted mesh shows beyond those points is PCA
extrapolation, not independently-observed geometry, a mesh-space solve
would add no real information over landmark-space while carrying real
mesh-registration risk. Matches the architecture BlendCap's own
`face_solver.py` uses: a small basis of ICT-FaceKit's blendshape deltas at
landmark points only, no mesh registration at all.

Pipeline:
1. Align ICT-FaceKit's raw mesh space into FLAME/SMPL-X's via a similarity
   transform (Umeyama) fit on the 51 shared landmark points (ICT-FaceKit's
   own `idx_to_landmark_verts` vs. FLAME's static dlib-68 landmark
   embedding).
2. For each of the 6 ICT-FaceKit expression shapes `GROUP_S_NAMES` maps to
   (`cheekSquint_L/R`, `noseSneer_L/R`, `eyeSquint_L/R`), extract its own 51
   landmark points, subtract the neutral's landmark points, and transform
   that raw-space delta into FLAME-space units via the same alignment.
3. Basis columns `D` (153, 6), everything else in `output_face.csv` comes
   from directly-tracked state, geometric measurement, or MediaPipe's own
   native blendshape output; see `face_blendshapes.py`'s own docstring for
   the full channel breakdown.
4. Also saves `jaw_landmark_jacobian` (153, 3): FLAME's own 51 landmarks,
   finite-differenced about `jaw_pose=0`, so the runtime solver can subtract
   jaw's own predicted landmark contribution before solving Group S (see
   `face_blendshapes.solve_arkit_weights`'s own docstring for why this
   ordering matters).

Usage: `pixi run -e main python scripts/build_face_bases.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `from pipeline...` resolves when run as a plain script, not `-m`
BODY_MODELS_DIR = REPO_ROOT / "body_models"
ICT_DIR = BODY_MODELS_DIR / "ict_facekit" / "FaceXModel"
FLAME_MODEL_DIR = BODY_MODELS_DIR  # smplx.create appends model_type/model_name itself
OUTPUT_PATH = BODY_MODELS_DIR / "arkit" / "face_bases.npz"

JAW_FD_EPS = 0.05  # rad, small enough to stay in the linear regime real jaw rotation lives in
NUM_FLAME_LANDMARKS = 51

# FLAME's static landmark embedding (`use_face_contour=False`) returns 51
# points that are, by construction, dlib-68 points 17-67 (the inner face,
# no jaw contour), see mp2dlib.py's own DLIB_INNER_51_START comment for
# the same fact used elsewhere in this project. ICT-FaceKit's own
# `idx_to_landmark_verts` (68 raw vertex indices) gives the same 51 points
# on ICT's mesh, confirmed dlib-68-ordered by checking that real
# inter-landmark distance ratios (eye-corner span vs. mouth-corner span)
# match FLAME's own to ~3%, not by assumption.
DLIB_INNER_51_START = 17

# Group S: the 6 ARKit channels this module's basis actually solves for,
# everything else in output_face.csv comes from either directly-tracked
# state (Group D: jaw, horizontal gaze), geometric measurement (Group E:
# eyelid), MediaPipe's own native blendshape output (Group S's own former
# 28 other channels, plus vertical gaze), or has no source at all (Group Z:
# TongueOut, Group W: JawForward/CheekPuff). See this module's own docstring
# for why these 6 specifically stayed on this hand-derived solve.
#
# A systematic real-data audit (all 34 original Group-S channels, all 3 real
# paired-capture clips) also found `MouthDimpleLeft` a consistent win here,
# deliberately NOT added: unlike the 6 below (each a symmetric pair, both
# sides agreeing), `MouthDimpleRight` didn't show the same pattern (MediaPipe
# wins it in one clip), and `Mouth*` as a whole is otherwise overwhelmingly
# MediaPipe-favored, a single asymmetric exception from only 3 clips is
# more likely session noise than a real structural advantage, and mixing
# mouth-region sources per-channel adds exactly the kind of bespoke
# complexity this rescope is meant to avoid. Left to MediaPipe with the rest
# of `Mouth*`.
GROUP_S_NAMES = [
    "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft", "NoseSneerRight",
    "EyeSquintLeft", "EyeSquintRight",
]
assert len(GROUP_S_NAMES) == 6

# ARKit shape name -> ICT-FaceKit source shape name(s).
_ARKIT_TO_ICT = {
    "CheekSquintLeft": ["cheekSquint_L"], "CheekSquintRight": ["cheekSquint_R"],
    "NoseSneerLeft": ["noseSneer_L"], "NoseSneerRight": ["noseSneer_R"],
    "EyeSquintLeft": ["eyeSquint_L"], "EyeSquintRight": ["eyeSquint_R"],
}


def _load_obj_vertices(path: Path) -> np.ndarray:
    """Minimal OBJ parser: returns (V, 3) vertex positions. Faces/normals/UVs
    are irrelevant here, only specific vertex *positions* (indexed via
    `idx_to_landmark_verts`) are ever used."""
    vertices = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
    return np.array(vertices, dtype=np.float64)


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama (1991) similarity transform: finds `(scale, R, t)` minimizing
    `sum(|dst - (scale * R @ src.T).T - t|^2)`. `src`/`dst`: (N, 3)
    corresponding points."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def _flame_model():
    import smplx

    return smplx.create(
        str(FLAME_MODEL_DIR), model_type="flame", gender="neutral", ext="npz",
        num_betas=300, num_expression_coeffs=50, use_face_contour=False, batch_size=1,
    )


def _flame_landmarks_51(jaw_pose: np.ndarray | None = None) -> np.ndarray:
    """FLAME's own 51 static landmarks (dlib points 17-67), in SMPL-X/meters
    space. Zero pose/expression by default (the canonical neutral this whole
    module's alignment and delta computations are relative to); pass
    `jaw_pose` (3,) to get the landmark response used for the jaw Jacobian."""
    import torch

    model = _flame_model()
    kwargs = {}
    if jaw_pose is not None:
        kwargs["jaw_pose"] = torch.tensor(jaw_pose, dtype=torch.float32).reshape(1, 3)
    out = model(**kwargs)
    return out.joints[0, -NUM_FLAME_LANDMARKS:].detach().numpy()


def _ict_to_flame_transform(ict_verts: np.ndarray, ict_lmk_idx: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fits the similarity transform mapping ICT-FaceKit's raw mesh space
    onto FLAME/SMPL-X's, via the shared 51-point landmark correspondence."""
    ict_lmks_51 = ict_verts[ict_lmk_idx][DLIB_INNER_51_START:]
    flame_lmks_51 = _flame_landmarks_51()
    scale, R, t = _umeyama_similarity(ict_lmks_51, flame_lmks_51)

    residual = np.linalg.norm((scale * (R @ ict_lmks_51.T).T + t) - flame_lmks_51, axis=1)
    print(
        f"      ICT->FLAME landmark alignment: scale={scale:.4f}, "
        f"residual mean={residual.mean() * 1000:.2f}mm, max={residual.max() * 1000:.2f}mm"
    )
    return scale, R, t


def _jaw_landmark_jacobian() -> np.ndarray:
    """FLAME's own 51 landmarks, finite-differenced about `jaw_pose=0` on
    all 3 axes, (51*3, 3), flattened the same [vertex-major, then xyz] way
    `solve_arkit_weights`'s own `D` basis is."""
    neutral = _flame_landmarks_51()
    axes = []
    for axis in range(3):
        jaw_pose = np.zeros(3, dtype=np.float32)
        jaw_pose[axis] = JAW_FD_EPS
        perturbed = _flame_landmarks_51(jaw_pose=jaw_pose)
        axes.append((perturbed - neutral) / JAW_FD_EPS)
    return np.stack(axes, axis=-1).reshape(-1, 3)


def build() -> None:
    vertex_indices = json.loads((ICT_DIR / "vertex_indices.json").read_text())
    ict_lmk_idx = np.array(vertex_indices["idx_to_landmark_verts"])

    print("[1/4] aligning ICT-FaceKit's neutral mesh into FLAME/SMPL-X space...")
    ict_neutral_verts = _load_obj_vertices(ICT_DIR / "generic_neutral_mesh.obj")
    scale, R, _t = _ict_to_flame_transform(ict_neutral_verts, ict_lmk_idx)
    ict_neutral_lmks_51 = ict_neutral_verts[ict_lmk_idx][DLIB_INNER_51_START:]

    def ict_delta_51(name: str) -> np.ndarray:
        expr_verts = _load_obj_vertices(ICT_DIR / f"{name}.obj")
        expr_lmks_51 = expr_verts[ict_lmk_idx][DLIB_INNER_51_START:]
        delta_raw = expr_lmks_51 - ict_neutral_lmks_51
        return scale * (R @ delta_raw.T).T  # displacement: no translation

    print(f"[2/4] building Group-S basis D (153, {len(GROUP_S_NAMES)})...")
    D = np.zeros((NUM_FLAME_LANDMARKS * 3, len(GROUP_S_NAMES)), dtype=np.float64)
    for k, name in enumerate(GROUP_S_NAMES):
        d_k = sum(ict_delta_51(n) for n in _ARKIT_TO_ICT[name])
        D[:, k] = d_k.reshape(-1)

    print("[3/4] finite-differencing FLAME's jaw-landmark Jacobian...")
    jaw_landmark_jacobian = _jaw_landmark_jacobian()

    print("[4/4] saving face_bases.npz...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        D=D.astype(np.float32),
        jaw_landmark_jacobian=jaw_landmark_jacobian.astype(np.float32),
        group_s_names=np.array(GROUP_S_NAMES),
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(f"D column norms (mm): {np.round(np.linalg.norm(D, axis=0) * 1000, 2)}")


if __name__ == "__main__":
    build()
