"""A cheap, betas-to-rest-pose-skeleton function for SMPL-X: given a body shape
(betas), returns where each joint sits before any pose is applied. Needed by
`gvhmr_endecoder.py`'s forward-kinematics step (turning predicted joint
rotations into actual 3D joint positions), and nothing else -- this project's
final SMPL-X mesh/vertex output is a separate, later pipeline stage, not this one.

Ported from `comfyui-motioncapture/nodes/body_model/smplx_lite.py`'s
`SmplxLite`, restricted to just `get_skeleton`/`parents`: the joint-position
regressor is linear in betas (`J_template + betas @ J_shapedirs`, both
precomputed once from the real model file), so this never needs the source
class's full vertex/mesh machinery (`posedirs`, `lbs_weights`, hand-pose
defaults, linear blend skinning) at all.

Needs the real SMPL-X model file (`SMPLX_NEUTRAL.npz`) -- the same one this
project's README already asks you to register for and download, placed at
`body_models/smplx/SMPLX_NEUTRAL.npz` in the repo root (gitignored, same as
`checkpoints/`; kept as a separate folder since these files carry their own
non-commercial registration terms, distinct from the neural-net checkpoints).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

NUM_BODY_JOINTS = 22  # root + 21 body joints; SMPL-X's remaining 33 are hands/jaw/eyes, unused here
NUM_BETAS = 10  # matches GVHMR's own predicted beta count

# Repo root is 3 levels up from this file (gvhmr/ -> adapters/ -> pipeline/ -> root).
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
)


class SmplxSkeleton:
    """Not an `nn.Module` -- no learned weights, just fixed data loaded once
    from the real SMPL-X model file, matching `EnDecoder`'s own treatment of
    the equivalent object as a plain attribute, not a checkpoint-loaded submodule."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, device: torch.device | None = None):
        data = np.load(model_path, allow_pickle=True)

        j_regressor = torch.from_numpy(np.asarray(data["J_regressor"])).float()  # (55, V)
        v_template = torch.from_numpy(np.asarray(data["v_template"])).float()  # (V, 3)
        shapedirs = torch.from_numpy(np.asarray(data["shapedirs"][:, :, :NUM_BETAS])).float()  # (V, 3, 10)

        # Precompute once: rest-pose joint positions are linear in betas, so at
        # runtime `get_skeleton` is just `J_template + betas @ J_shapedirs`,
        # never re-touching the full (10475-vertex) mesh.
        self.j_template = (j_regressor @ v_template).to(device)  # (55, 3)
        self.j_shapedirs = torch.einsum("jv,vcd->jcd", j_regressor, shapedirs).to(device)  # (55, 3, 10)

        parents = torch.from_numpy(np.asarray(data["kintree_table"][0]).astype(np.int64))
        parents[0] = -1  # the root's parent is stored as an unsigned -1 (wraps to a huge value) in the raw file
        self.parents = parents[:NUM_BODY_JOINTS].tolist()

    def get_skeleton(self, betas: torch.Tensor) -> torch.Tensor:
        """betas: (..., 10) -> (..., 55, 3) rest-pose joint positions. Caller
        slices to `[..., :NUM_BODY_JOINTS, :]` for the body-only skeleton."""
        return self.j_template + torch.einsum("...k,jck->...jc", betas, self.j_shapedirs)
