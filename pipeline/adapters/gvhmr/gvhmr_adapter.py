"""Thin `load()`/`infer()`/`unload()` wrapper tying every other `gvhmr_*` port
file together into the one call `stage_2_estimate_human_motion.py` needs: given
the human's per-frame tracked mask (from stage 1) and this clip's camera
intrinsics, produce a full-clip SMPL-X body pose in both camera-space
("incam") and world-grounded ("global") coordinates. Follows the same
`load()`/`infer()`/`unload()` convention as `sam31_adapter.py`.

Per-frame: mask -> bbox (`extract_bbox_from_numpy_mask`) -> crop -> ViTPose
keypoints + HMR2 image feature. Whole-clip: those sequences feed
`GVHMRTemporalTransformer` once, decoded by `EnDecoder`, then
`pp_static_joint_cam`/`process_ik` clean up the "global" result -- mirrors
`comfyui-motioncapture/nodes/gvhmr/model.py`'s `Pipeline.forward` /
`DemoPL.predict` call sequence.

**Static-camera-only scope simplification.** This project only ever records a
tripod-mounted, non-moving camera (see this repo's locked scope decisions), so
this adapter never runs GVHMR's alternative moving-camera path (DPVO visual
odometry). Two consequences, both confirmed against the real reference:
  - `cam_angvel` (the raw per-frame relative camera rotation) is the fixed
    constant `[1, 0, 0, 0, 1, 0]` for every frame -- confirmed by tracing
    `inference_node.py`'s static-camera branch (`R_w2c = torch.eye(3)` always)
    through `geo_transform.compute_cam_angvel`, whose formula collapses to
    exactly this 6D-identity-rotation constant when every `R_w2c` is the same.
  - The reference's `get_smpl_params_w_Rt_v2` (in `gvhmr/model.py`, not ported
    to its own file since it only exists as this call site's composition,
    never as a reusable function GVHMR itself calls elsewhere) computes a
    per-frame drift-correction rotation `R_t_to_0` from consecutive
    `cam_angvel` values. With `cam_angvel` constant across frames, that
    rotation is provably the identity for every frame (verified numerically
    against the real function while writing this file, on random synthetic
    pose data) -- so building `pred_smpl_params_global` reduces to a bare
    `global_orient_gv` roundtrip plus `rollout_local_transl_vel` +
    `get_tgtcoord_rootparam`, both of which already have their own ported,
    verified functions in `gvhmr_translation_math.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open

from pipeline.progress_tracker import StageName

from .gvhmr_camera_math import compute_bbox_info_bedlam, compute_transl_full_cam, normalize_kp2d
from .gvhmr_endecoder import EnDecoder
from .gvhmr_hmr2 import GVHMRHMR2
from .gvhmr_postprocess import pp_static_joint_cam, process_ik
from .gvhmr_preprocess import bbox_xywh_to_xys, crop_and_normalize
from .gvhmr_rotation_math import axis_angle_to_matrix, matrix_to_axis_angle
from .gvhmr_transformer import GVHMRTemporalTransformer
from .gvhmr_translation_math import get_tgtcoord_rootparam, rollout_local_transl_vel
from .gvhmr_vitpose import GVHMRViTPoseModel, estimate_keypoints

from ...helpers.progress_reporter import frame_progress

# Repo root is 3 levels up from this file (gvhmr/ -> adapters/ -> pipeline/ -> root).
CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
VITPOSE_CHECKPOINT = CHECKPOINT_DIR / "vitpose.safetensors"
HMR2_CHECKPOINT = CHECKPOINT_DIR / "hmr2.safetensors"
GVHMR_CHECKPOINT = CHECKPOINT_DIR / "gvhmr.safetensors"

# See this module's docstring: the only `cam_angvel` value a static camera ever produces.
CAM_ANGVEL_RAW = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
# GVHMR's own checkpoint-independent normalization stats for `cam_angvel`
# (`stats.py`'s `cam_angvel["manual"]` -- confirmed as the table actually used
# at runtime via `Pipeline.__init__`'s `normalize_cam_angvel=True` path, not
# its sibling `"emdb_none_test"` entry, which nothing calls).
CAM_ANGVEL_MEAN = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
CAM_ANGVEL_STD = torch.tensor([0.001, 0.1, 0.1, 0.1, 0.001, 0.1])

# HMR2's raw checkpoint bundles unrelated training-only keys (discriminator, an SMPL FK
# loss layer) alongside the backbone/head this port actually implements -- see _load_hmr2_state.
_HMR2_KEEP_PREFIXES = ("backbone", "smpl_head")
# The GVHMR checkpoint nests every weight under this prefix -- see _load_gvhmr_transformer_state.
_GVHMR_TRANSFORMER_KEY_PREFIX = "pipeline.denoiser3d."

# GVHMRTemporalTransformer.forward's own output dict keys (gvhmr_transformer.py).
KEY_PRED_X = "pred_x"
KEY_PRED_CAM = "pred_cam"
KEY_STATIC_CONF_LOGITS = "static_conf_logits"

# SMPL(-X) pose-parameter field names, shared by EnDecoder.decode()'s output and this
# adapter's own pred_smpl_params_incam/pred_smpl_params_global dicts.
KEY_BODY_POSE = "body_pose"
KEY_BETAS = "betas"
KEY_GLOBAL_ORIENT = "global_orient"
KEY_GLOBAL_ORIENT_GV = "global_orient_gv"
KEY_LOCAL_TRANSL_VEL = "local_transl_vel"
KEY_TRANSL = "transl"

# infer()'s top-level output dict keys.
KEY_PRED_SMPL_PARAMS_INCAM = "pred_smpl_params_incam"
KEY_PRED_SMPL_PARAMS_GLOBAL = "pred_smpl_params_global"


def _load_direct_state(path: Path) -> dict[str, torch.Tensor]:
    """ViTPose's checkpoint already uses this project's exact module names
    (`backbone.*`/`keypoint_head.*`) with no unrelated keys mixed in -- no
    filtering or prefix-stripping needed."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def _load_hmr2_state(path: Path) -> dict[str, torch.Tensor]:
    """HMR2's raw checkpoint bundles unrelated training-only `discriminator.*`
    and `smpl.*` keys alongside the `backbone.*`/`smpl_head.*` this port
    actually implements -- filtered out here, matching the reference's own
    `load_hmr2()` behavior."""
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.split(".")[0] in _HMR2_KEEP_PREFIXES:
                state[key] = f.get_tensor(key)
    return state


def _load_gvhmr_transformer_state(path: Path) -> dict[str, torch.Tensor]:
    """The GVHMR checkpoint nests every weight under `pipeline.denoiser3d.` --
    stripped here since `GVHMRTemporalTransformer` matches only that inner module."""
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key[len(_GVHMR_TRANSFORMER_KEY_PREFIX):]] = f.get_tensor(key)
    return state


def extract_bbox_from_numpy_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Boolean/uint8 (H, W) mask -> its bounding rect [x, y, w, h], in the
    mask's own pixel coordinates, or None if the mask is empty this frame
    (e.g. the tracked person was briefly occluded). Uses the union of all
    contours rather than the single largest one, so a person fragmented into
    a couple of disjoint blobs by partial occlusion still gets a bbox that
    covers all of them.

    **The returned coordinates are in the mask's own resolution, not
    necessarily the source video's.** SAM 3.1 always produces masks at its
    own fixed working resolution (1008x1008, non-aspect-preserving stretch)
    regardless of the input video's actual resolution -- callers cropping
    from the *original* frame or feeding this into camera-intrinsics math
    (which is built from the original resolution) must rescale this bbox
    first. See `_rescale_bbox_xywh` below.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(np.concatenate(contours, axis=0))


def _rescale_bbox_xywh(
    bbox_xywh: tuple[int, int, int, int], from_hw: tuple[int, int], to_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Rescale a [x, y, w, h] bbox from one pixel-coordinate space to another
    (independent x/y scale factors, matching SAM 3.1's own non-aspect-preserving
    stretch to its fixed working resolution -- see `extract_bbox_from_numpy_mask`).
    """
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    sx, sy = to_w / from_w, to_h / from_h
    x, y, w, h = bbox_xywh
    return (round(x * sx), round(y * sy), round(w * sx), round(h * sy))


class GVHMRAdapter:
    """`load()` once per stage run, `infer()` for the whole clip, `unload()`
    before the process exits -- same VRAM-budget reasoning as `Sam31Adapter`.
    """

    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._loaded = False

    def load(
        self,
        vitpose_checkpoint: Path = VITPOSE_CHECKPOINT,
        hmr2_checkpoint: Path = HMR2_CHECKPOINT,
        gvhmr_checkpoint: Path = GVHMR_CHECKPOINT,
    ) -> None:
        self.vitpose = GVHMRViTPoseModel().to(dtype=self.dtype)
        self.vitpose.load_state_dict(_load_direct_state(vitpose_checkpoint), strict=True)

        self.hmr2 = GVHMRHMR2().to(dtype=self.dtype)
        self.hmr2.load_state_dict(_load_hmr2_state(hmr2_checkpoint), strict=True)

        self.transformer = GVHMRTemporalTransformer().to(dtype=self.dtype)
        self.transformer.load_state_dict(_load_gvhmr_transformer_state(gvhmr_checkpoint), strict=True)

        for module in (self.vitpose, self.hmr2, self.transformer):
            module.to(self.device).eval()

        # Not checkpoint-loaded (fixed stats + the real SMPL-X model file) --
        # kept on CPU in float32, matching the reference's own "cast to
        # float32 for post-processing" step (matrix inversion in the IK solver
        # needs real precision headroom, unlike the vision models above; the
        # whole-clip pose vectors here are small enough that CPU is plenty fast).
        self.endecoder = EnDecoder()

        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        del self.vitpose, self.hmr2, self.transformer, self.endecoder
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._loaded = False

    def infer(self, frame_paths: list[Path], masks: torch.Tensor, K_fullimg: torch.Tensor) -> dict:
        """
        Args:
            frame_paths: N per-frame image paths, as extracted by stage 0.
            masks: (N, H, W) bool -- the human's per-frame tracked mask, already
                unpacked from stage 1's bit-packed format.
            K_fullimg: (3, 3) float camera intrinsics, constant for this clip
                (see this module's docstring for why a single K works here).

        Returns {"pred_smpl_params_incam": {...}, "pred_smpl_params_global": {...}},
        each a dict of (N, ...) tensors: body_pose (63), betas (10),
        global_orient (3), transl (3).
        """
        N = len(frame_paths)
        assert masks.shape[0] == N
        mask_hw = masks.shape[-2:]

        bbx_xys_list, kp2d_list, f_imgseq_list = [], [], []
        last_bbox_xywh = None
        with torch.inference_mode():
            for i, frame_path in frame_progress(enumerate(frame_paths), total=N, label=StageName.STAGE_2_ESTIMATE_HUMAN_MOTION.title):
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    raise RuntimeError(f"Could not read frame: {frame_path}")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                bbox_xywh = extract_bbox_from_numpy_mask(masks[i].numpy())
                if bbox_xywh is None:
                    if last_bbox_xywh is None:
                        raise RuntimeError(f"Human mask is empty on frame {i} (no prior frame to fall back on)")
                    bbox_xywh = last_bbox_xywh  # briefly-occluded frame: reuse the last known bbox
                else:
                    # Mask coordinates are in SAM 3.1's fixed working resolution, not this
                    # frame's actual resolution -- rescale before using them against the
                    # real frame or camera intrinsics (both in native-resolution pixels).
                    bbox_xywh = _rescale_bbox_xywh(bbox_xywh, mask_hw, frame_rgb.shape[:2])
                last_bbox_xywh = bbox_xywh

                bbx_xys = bbox_xywh_to_xys(bbox_xywh)
                crop = crop_and_normalize(frame_rgb, bbx_xys).unsqueeze(0).to(device=self.device, dtype=self.dtype)

                kp2d = estimate_keypoints(self.vitpose, frame_rgb, bbox_xywh, self.device, self.dtype)
                f_imgseq = self.hmr2(crop)[0].float().cpu()

                bbx_xys_list.append(torch.from_numpy(bbx_xys))
                kp2d_list.append(torch.from_numpy(kp2d))
                f_imgseq_list.append(f_imgseq)

        bbx_xys_seq = torch.stack(bbx_xys_list).unsqueeze(0)  # (1, N, 3)
        kp2d_seq = torch.stack(kp2d_list).unsqueeze(0).float()  # (1, N, 17, 3)
        f_imgseq_seq = torch.stack(f_imgseq_list).unsqueeze(0)  # (1, N, 1024)
        K_fullimg_seq = K_fullimg.view(1, 1, 3, 3).expand(1, N, 3, 3).float()

        obs = normalize_kp2d(kp2d_seq, bbx_xys_seq)
        f_cliffcam = compute_bbox_info_bedlam(bbx_xys_seq, K_fullimg_seq)
        f_cam_angvel = ((CAM_ANGVEL_RAW - CAM_ANGVEL_MEAN) / CAM_ANGVEL_STD).view(1, 1, 6).expand(1, N, 6)
        length = torch.tensor([N])

        with torch.inference_mode():
            transformer_out = self.transformer(
                length=length.to(self.device),
                obs=obs.to(device=self.device, dtype=self.dtype),
                f_cliffcam=f_cliffcam.to(device=self.device, dtype=self.dtype),
                f_cam_angvel=f_cam_angvel.to(device=self.device, dtype=self.dtype),
                f_imgseq=f_imgseq_seq.to(device=self.device, dtype=self.dtype),
            )

        pred_x = transformer_out[KEY_PRED_X].float().cpu()
        pred_cam = transformer_out[KEY_PRED_CAM].float().cpu()
        static_conf_logits = transformer_out[KEY_STATIC_CONF_LOGITS].float().cpu()

        decoded = self.endecoder.decode(pred_x)

        pred_smpl_params_incam = {
            KEY_BODY_POSE: decoded[KEY_BODY_POSE],
            KEY_BETAS: decoded[KEY_BETAS],
            KEY_GLOBAL_ORIENT: decoded[KEY_GLOBAL_ORIENT],
            KEY_TRANSL: compute_transl_full_cam(pred_cam, bbx_xys_seq, K_fullimg_seq),
        }

        # Static-camera simplification of `get_smpl_params_w_Rt_v2` -- see this
        # module's docstring.
        global_orient_gv = matrix_to_axis_angle(axis_angle_to_matrix(decoded[KEY_GLOBAL_ORIENT_GV]))
        transl_global = rollout_local_transl_vel(decoded[KEY_LOCAL_TRANSL_VEL], global_orient_gv)
        global_orient_global, transl_global, _ = get_tgtcoord_rootparam(global_orient_gv, transl_global, tsf="any->ay")

        outputs = {
            KEY_PRED_SMPL_PARAMS_INCAM: pred_smpl_params_incam,
            KEY_PRED_SMPL_PARAMS_GLOBAL: {
                KEY_BODY_POSE: decoded[KEY_BODY_POSE],
                KEY_BETAS: decoded[KEY_BETAS],
                KEY_GLOBAL_ORIENT: global_orient_global,
                KEY_TRANSL: transl_global,
            },
            KEY_STATIC_CONF_LOGITS: static_conf_logits,
        }

        outputs[KEY_PRED_SMPL_PARAMS_GLOBAL][KEY_TRANSL] = pp_static_joint_cam(outputs, self.endecoder)
        body_pose_ik = process_ik(outputs, self.endecoder)
        outputs[KEY_PRED_SMPL_PARAMS_GLOBAL][KEY_BODY_POSE] = body_pose_ik
        outputs[KEY_PRED_SMPL_PARAMS_INCAM][KEY_BODY_POSE] = body_pose_ik

        return {
            KEY_PRED_SMPL_PARAMS_INCAM: {k: v[0] for k, v in outputs[KEY_PRED_SMPL_PARAMS_INCAM].items()},
            KEY_PRED_SMPL_PARAMS_GLOBAL: {k: v[0] for k, v in outputs[KEY_PRED_SMPL_PARAMS_GLOBAL].items()},
        }
