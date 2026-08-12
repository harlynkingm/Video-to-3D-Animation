"""Offline, one-time: builds `body_models/arkit/face_preview_shapes.npz`, a
full-mesh ARKit-52 blendshape basis for the `ARKit_face_preview.blend`
*visualization only*, never used by the runtime CSV solve
(`face_blendshapes.solve_arkit_weights`, which stays landmark-space; see
`build_face_bases.py`'s own docstring for why). Its purpose: let a solved
`output_face.csv` be watched as a real 3D face performance, driven by the
exact same 52 weights that get written to the CSV, so the CSV's accuracy can
be judged visually, not just numerically.

Nearest-VERTEX ICT-FaceKit transfer onto FLAME's own native 5023-vertex mesh
(reusing `build_face_bases.py`'s already-validated Umeyama alignment),
deliberately NOT the nearest-triangle-point projection abandoned for the
runtime solve. A debug visualization's accuracy bar is far lower than a
numeric solve's: nearest-vertex snapping is at most a few mm off, given
ICT's much higher vertex density than FLAME's own mesh, imperceptible for
watching a face performance.

Includes all 52 ARKit channels this time (not just Group S), `TongueOut`
still has no ICT-FaceKit source (stays a zero column), but jaw/gaze/eyelid
all get real ICT deltas here even though the runtime CSV computes those
directly rather than through this basis, so the preview can show the full
combined performance the CSV actually describes.

Usage: `pixi run -e main python scripts/build_arkit_preview_shapes.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.helpers.livelink_csv import ARKIT_BLENDSHAPE_NAMES  # noqa: E402
from scripts.build_face_bases import (  # noqa: E402
    BODY_MODELS_DIR, ICT_DIR, _ARKIT_TO_ICT as _ARKIT_TO_ICT_GROUP_S, _flame_model, _ict_to_flame_transform,
    _load_obj_vertices,
)

OUTPUT_PATH = BODY_MODELS_DIR / "arkit" / "face_preview_shapes.npz"

# Group S's own mapping (6 names, see build_face_bases.py) plus everything
# else with a real ICT source, jaw, gaze, eyelid, CheekPuff. TongueOut is
# absent (no ICT source at all, FLAME/SMPL-X have no tongue geometry), left as a zero
# column below.
_ARKIT_TO_ICT_FULL = {
    **_ARKIT_TO_ICT_GROUP_S,
    "JawForward": ["jawForward"], "JawLeft": ["jawLeft"], "JawRight": ["jawRight"], "JawOpen": ["jawOpen"],
    "CheekPuff": ["cheekPuff_L", "cheekPuff_R"],
    "EyeBlinkLeft": ["eyeBlink_L"], "EyeBlinkRight": ["eyeBlink_R"],
    "EyeWideLeft": ["eyeWide_L"], "EyeWideRight": ["eyeWide_R"],
    "EyeLookInLeft": ["eyeLookIn_L"], "EyeLookOutLeft": ["eyeLookOut_L"],
    "EyeLookUpLeft": ["eyeLookUp_L"], "EyeLookDownLeft": ["eyeLookDown_L"],
    "EyeLookInRight": ["eyeLookIn_R"], "EyeLookOutRight": ["eyeLookOut_R"],
    "EyeLookUpRight": ["eyeLookUp_R"], "EyeLookDownRight": ["eyeLookDown_R"],
}


def _flame_neutral_mesh() -> tuple[np.ndarray, np.ndarray]:
    """FLAME's own native mesh at zero pose/expression/betas: (5023, 3)
    vertices, (F, 3) faces."""
    import torch

    model = _flame_model()
    out = model(
        betas=torch.zeros(1, 300), expression=torch.zeros(1, 50), global_orient=torch.zeros(1, 3),
        neck_pose=torch.zeros(1, 3), jaw_pose=torch.zeros(1, 3), leye_pose=torch.zeros(1, 3),
        reye_pose=torch.zeros(1, 3), transl=torch.zeros(1, 3),
    )
    return out.vertices[0].detach().numpy(), model.faces


def build() -> None:
    vertex_indices = json.loads((ICT_DIR / "vertex_indices.json").read_text())
    ict_lmk_idx = np.array(vertex_indices["idx_to_landmark_verts"])
    fitting_verts = np.array(vertex_indices["idx_to_fitting_verts"])

    print("[1/4] aligning ICT-FaceKit's neutral mesh into FLAME space...")
    ict_neutral_raw = _load_obj_vertices(ICT_DIR / "generic_neutral_mesh.obj")
    scale, R, t = _ict_to_flame_transform(ict_neutral_raw, ict_lmk_idx)
    ict_neutral_aligned = scale * (R @ ict_neutral_raw.T).T + t

    print("[2/4] nearest-vertex registration onto FLAME's own 5023-vertex mesh...")
    flame_neutral, faces = _flame_neutral_mesh()
    fitting_only = ict_neutral_aligned[fitting_verts]
    tree = cKDTree(fitting_only)
    dist, nearest_local = tree.query(flame_neutral)
    nearest_ict_idx = fitting_verts[nearest_local]
    print(f"      nearest-vertex distance (mm): mean={dist.mean() * 1000:.2f}, max={dist.max() * 1000:.2f}")

    print("[3/4] building full 52-channel mesh delta basis...")
    D_mesh = np.zeros((flame_neutral.shape[0], 3, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    for k, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
        ict_names = _ARKIT_TO_ICT_FULL.get(name, [])
        if not ict_names:
            continue  # TongueOut: no source, zero column
        delta_sum = np.zeros_like(ict_neutral_raw)
        for n in ict_names:
            expr_verts_raw = _load_obj_vertices(ICT_DIR / f"{n}.obj")
            delta_sum += expr_verts_raw - ict_neutral_raw
        delta_aligned = scale * (R @ delta_sum.T).T
        D_mesh[:, :, k] = delta_aligned[nearest_ict_idx]

    print("[4/4] saving face_preview_shapes.npz...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        D_mesh=D_mesh, faces=faces, neutral_verts=flame_neutral.astype(np.float32),
        arkit_names=np.array(ARKIT_BLENDSHAPE_NAMES),
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(f"D_mesh per-channel max displacement (mm): "
          f"{np.round(np.linalg.norm(D_mesh, axis=1).max(axis=0) * 1000, 1)}")


if __name__ == "__main__":
    build()
