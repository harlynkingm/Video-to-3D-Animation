"""Contact detection: per-frame, checks whether the body's candidate joints
(fingertips + wrist for each hand, plus a couple of representative joints for
every other body region) land near the tracked object's 2D mask in image
space -- pure geometry, no learned model, no GPU. Deliberately never compares
against absolute real-world depth: GVHMR's own Z estimate is measurably
unreliable for a reaching/foreshortened arm, and for a convincing *animation*
what matters is that the body and object agree with each other in the SAME
(GVHMR/HaMeR's own) space, not that either matches real-world ground truth.

A rolling-window hysteresis (same lock/release pattern as
`hand_retarget.reject_biomechanically_implausible_wrist`, just inverted --
seeding on HIGH confidence instead of rejecting on a high implausibility
score) turns raw per-frame proximity into contact *events*: a real contact
persists over a run of frames, so an isolated one-frame mask-jitter spike
shouldn't count as its own event, and a genuine contact's own naturally-lower-
confidence edge frames shouldn't fragment it into several tiny ones.

Two secondary passes address the two failure modes 2D-only detection can't
avoid on its own: `consolidate_overlapping_events` merges events from
different regions that share an underlying joint (the wrist is a candidate
for both a hand and its own arm region, so one real grip can produce two
region-events for the same physical contact); `depth_gap_for_joint` measures
whether the object and body were actually physically close where their masks
overlapped, rather than one merely passing in front of/behind the other in
the image -- using an already-computed depth map (e.g. from
Depth-Anything-3, which this project's own depth investigation found
reliable), never GVHMR's own retargeted joint position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter1d

# Pixel distance (at the object mask's own resolution) beyond which a joint is
# considered definitely not touching the object. Confidence decays linearly
# from 1.0 (inside the mask) to 0.0 at this distance. Loose enough to bridge
# a real, temporary retargeted-hand-position drift during fast object motion
# (a continuous grip measured a drift of up to ~32px mid-lift, which a
# tighter threshold split into two events with a gap in between) -- safe to
# keep loose because `depth_gap_for_joint` is the actual arbiter of real
# contact vs. incidental occlusion; this threshold only decides what's worth
# proposing as a candidate.
CONTACT_PIXEL_THRESHOLD = 35.0

# Hysteresis, same lock/release idea as the wrist-plausibility gate: a run of
# elevated-confidence frames only becomes a contact event if its own peak
# confidence crosses the stricter SEED threshold; the event's boundaries are
# wherever a WINDOW-frame rolling max of confidence stays above the looser
# RELEASE threshold.
CONTACT_SEED_CONFIDENCE = 0.7
CONTACT_RELEASE_CONFIDENCE = 0.3
CONTACT_WINDOW = 5

# SMPL-X joint indices (55-joint body+hands layout; see smplx_bvh_preview.py's
# own LEFT_FINGERS/RIGHT_FINGERS for the source of this convention, and the
# standard SMPL/SMPL-X body-joint order for joints 0-21: pelvis, l/r hip,
# spine1, l/r knee, spine2, l/r ankle, spine3, l/r foot, neck, l/r collar,
# head, l/r shoulder, l/r elbow, l/r wrist). Each hand's 15 finger joints run
# [Index1-3, Middle1-3, Pinky1-3, Ring1-3, Thumb1-3]; the "tip" of each finger
# is its 3rd (last) phalanx joint -- not the literal fingertip surface point,
# but a good-enough proxy for contact detection, consistent with this
# module's other geometric approximations.
LEFT_WRIST_JOINT = 20
RIGHT_WRIST_JOINT = 21
_FINGERTIP_OFFSETS = [2, 5, 8, 11, 14]  # Index3, Middle3, Pinky3, Ring3, Thumb3
LEFT_FINGERTIP_JOINTS = [25 + o for o in _FINGERTIP_OFFSETS]
RIGHT_FINGERTIP_JOINTS = [40 + o for o in _FINGERTIP_OFFSETS]

HEAD_JOINT = 15
SPINE2_JOINT = 6
SPINE3_JOINT = 9
LEFT_SHOULDER_JOINT = 16
RIGHT_SHOULDER_JOINT = 17
LEFT_ELBOW_JOINT = 18
RIGHT_ELBOW_JOINT = 19
LEFT_HIP_JOINT = 1
RIGHT_HIP_JOINT = 2
LEFT_KNEE_JOINT = 4
RIGHT_KNEE_JOINT = 5
LEFT_ANKLE_JOINT = 7
RIGHT_ANKLE_JOINT = 8

_HAND_JOINT_NAMES = ["index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip", "wrist"]

# Body region -> candidate SMPL-X joint indices to test for proximity to the
# object mask. Deliberately coarse -- 8 regions total, including the two
# hands -- rather than a separate region per limb segment (upper arm vs.
# forearm, thigh vs. shin vs. foot): a finer split would just fragment one
# real limb-object contact into several near-duplicate events without adding
# information this project needs. Each non-hand region uses its own joint
# chain's endpoints plus midpoint (e.g. shoulder/elbow/wrist for an arm) so a
# contact anywhere along that limb's length is caught, not just at one end.
REGION_JOINTS: dict[str, list[int]] = {
    "left_hand": LEFT_FINGERTIP_JOINTS + [LEFT_WRIST_JOINT],
    "right_hand": RIGHT_FINGERTIP_JOINTS + [RIGHT_WRIST_JOINT],
    "head": [HEAD_JOINT],
    "chest": [SPINE2_JOINT, SPINE3_JOINT],
    "left_arm": [LEFT_SHOULDER_JOINT, LEFT_ELBOW_JOINT, LEFT_WRIST_JOINT],
    "right_arm": [RIGHT_SHOULDER_JOINT, RIGHT_ELBOW_JOINT, RIGHT_WRIST_JOINT],
    "left_leg": [LEFT_HIP_JOINT, LEFT_KNEE_JOINT, LEFT_ANKLE_JOINT],
    "right_leg": [RIGHT_HIP_JOINT, RIGHT_KNEE_JOINT, RIGHT_ANKLE_JOINT],
}

# Parallels REGION_JOINTS, one name per candidate joint -- purely for a
# human-readable event label, never used for indexing.
REGION_JOINT_NAMES: dict[str, list[str]] = {
    "left_hand": _HAND_JOINT_NAMES,
    "right_hand": _HAND_JOINT_NAMES,
    "head": ["head"],
    "chest": ["spine2", "spine3"],
    "left_arm": ["shoulder", "elbow", "wrist"],
    "right_arm": ["shoulder", "elbow", "wrist"],
    "left_leg": ["hip", "knee", "ankle"],
    "right_leg": ["hip", "knee", "ankle"],
}

REGION_NAMES: list[str] = list(REGION_JOINTS.keys())


@dataclass
class ContactEvent:
    regions: list[str]  # usually one region; >1 after consolidate_overlapping_events merges a shared-joint pair
    joint: str  # one of REGION_JOINT_NAMES[regions[0]] -- the same joint in every region in `regions`
    start_frame: int
    end_frame: int  # inclusive
    peak_frame: int  # frame with this event's own highest confidence
    mean_confidence: float
    depth_gap_m: float | None = None  # set by a caller that runs depth_gap_for_joint; None until then


def candidate_joint_indices(region: str) -> list[int]:
    return REGION_JOINTS[region]


def project_to_pixels(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 3) camera-space points -> (N, 2) pixel coordinates. Public (not
    just an internal helper): stage_7's contacts-preview renderer reuses this
    same projection to draw a circle at a joint's actual pixel location."""
    projected = points @ K.T
    return projected[:, :2] / projected[:, 2:3]


def _confidence_from_distance_field(pixels: np.ndarray, dist_outside: np.ndarray, height: int, width: int) -> np.ndarray:
    u = np.round(pixels[:, 0]).astype(int)
    v = np.round(pixels[:, 1]).astype(int)
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    distance = np.full(len(pixels), np.inf)
    distance[in_frame] = dist_outside[v[in_frame], u[in_frame]]
    return np.clip(1.0 - distance / CONTACT_PIXEL_THRESHOLD, 0.0, 1.0)


def _confidence_from_mask_distance(pixels: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """(N,) confidence: 1.0 if a pixel lands inside the mask, decaying to 0 at
    CONTACT_PIXEL_THRESHOLD -- distance to the mask's nearest True pixel."""
    height, width = object_mask.shape
    dist_outside = distance_transform_edt(~object_mask)
    return _confidence_from_distance_field(pixels, dist_outside, height, width)


def frame_confidence_for_region(
    joints_xyz: np.ndarray, region: str, K: np.ndarray, object_masks: list[np.ndarray | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the MAX confidence across this region's candidate joints
    (which joint is closest can vary frame to frame -- a grip might shift from
    fingertip-only contact to the whole hand, or an object might slide from a
    forearm towards the wrist). Returns `(confidence, joint_idx)`, both (F,);
    `joint_idx` indexes into `candidate_joint_indices(region)`, or -1 on a
    frame with no tracked object mask.

    Args:
        joints_xyz: (F, J, 3) all SMPL-X joints, camera-space (GVHMR/HaMeR's
            own retargeted space, NOT real-world/depth-aligned).
        object_masks: one (H, W) bool mask per frame, or None where the object
            wasn't tracked that frame (no contact possible without a mask).
    """
    joint_ids = candidate_joint_indices(region)
    n_frames = joints_xyz.shape[0]
    confidence = np.zeros((n_frames, len(joint_ids)))

    for f in range(n_frames):
        mask = object_masks[f]
        if mask is None:
            continue
        pixels = project_to_pixels(joints_xyz[f, joint_ids], K)
        confidence[f] = _confidence_from_mask_distance(pixels, mask)

    best_idx = confidence.argmax(axis=1)
    best_confidence = confidence[np.arange(n_frames), best_idx]
    has_mask = np.array([m is not None for m in object_masks])
    best_confidence = np.where(has_mask, best_confidence, 0.0)
    best_idx = np.where(has_mask, best_idx, -1)
    return best_confidence, best_idx


def per_frame_region_confidence(
    frame_joints: np.ndarray, K: np.ndarray, object_mask: np.ndarray | None,
) -> dict[str, tuple[float, int]]:
    """Confidence + winning candidate-joint index for every region, for a
    SINGLE frame. Computes the mask's own distance field once and shares it
    across all 8 regions -- calling `frame_confidence_for_region` once per
    region instead would recompute that same distance field 8 times over,
    real wasted work once a caller (stage 7) needs every region for the same
    frame anyway. Returns `{region: (confidence, joint_idx)}`; `joint_idx`
    indexes into `candidate_joint_indices(region)`, or `(0.0, -1)` when
    `object_mask` is None (no tracked object this frame).
    """
    if object_mask is None:
        return {region: (0.0, -1) for region in REGION_NAMES}

    height, width = object_mask.shape
    dist_outside = distance_transform_edt(~object_mask)

    result: dict[str, tuple[float, int]] = {}
    for region in REGION_NAMES:
        joint_ids = REGION_JOINTS[region]
        pixels = project_to_pixels(frame_joints[joint_ids], K)
        confidence = _confidence_from_distance_field(pixels, dist_outside, height, width)
        best = int(confidence.argmax())
        result[region] = (float(confidence[best]), best)
    return result


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) inclusive index pairs for each contiguous True run."""
    runs = []
    n, i = len(mask), 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def detect_contact_events(
    confidence: np.ndarray, joint_idx: np.ndarray, region: str, joint_names: list[str]
) -> list[ContactEvent]:
    """Turns raw per-frame confidence into contact events via the same
    lock/release hysteresis `hand_retarget.reject_biomechanically_implausible_wrist`
    uses, seeded on HIGH confidence (a real, unambiguous contact frame) instead
    of a high implausibility score."""
    rolling_max = maximum_filter1d(confidence, size=CONTACT_WINDOW, mode="nearest")
    elevated = rolling_max > CONTACT_RELEASE_CONFIDENCE
    seed = confidence > CONTACT_SEED_CONFIDENCE

    events = []
    for start, end in _contiguous_true_runs(elevated):
        if not seed[start:end + 1].any():
            continue
        peak_offset = int(np.argmax(confidence[start:end + 1]))
        peak_frame = start + peak_offset
        joint = joint_names[joint_idx[peak_frame]]
        events.append(ContactEvent(
            regions=[region], joint=joint, start_frame=start, end_frame=end, peak_frame=peak_frame,
            mean_confidence=float(confidence[start:end + 1].mean()),
        ))
    return events


def _resolve_joint_index(region: str, joint_name: str) -> int:
    return REGION_JOINTS[region][REGION_JOINT_NAMES[region].index(joint_name)]


def _merge_two_events(a: ContactEvent, b: ContactEvent) -> ContactEvent:
    regions = a.regions + [r for r in b.regions if r not in a.regions]
    peak_source = a if a.mean_confidence >= b.mean_confidence else b
    return ContactEvent(
        regions=regions, joint=a.joint,
        start_frame=min(a.start_frame, b.start_frame), end_frame=max(a.end_frame, b.end_frame),
        peak_frame=peak_source.peak_frame, mean_confidence=max(a.mean_confidence, b.mean_confidence),
    )


def consolidate_overlapping_events(events: list[ContactEvent]) -> list[ContactEvent]:
    """Merges events from different regions that are really the same
    physical contact reported twice -- e.g. `left_hand` and `left_arm` both
    include the wrist as a candidate joint (see `REGION_JOINTS`), so a single
    wrist-driven grip produces one event per region unless merged here. Two
    events merge when they resolve to the same actual SMPL-X joint index
    (never just the same joint *name* -- "wrist" means a different index
    under `left_arm` vs. `right_arm`) and their frame ranges overlap. The
    merged event's `mean_confidence`/`peak_frame` come from whichever
    original event was itself more confident -- the two aren't independent
    evidence, just the same signal seen through two regions, so this doesn't
    average or add them.
    """
    by_joint: dict[int, list[ContactEvent]] = {}
    for event in events:
        joint_idx = _resolve_joint_index(event.regions[0], event.joint)
        by_joint.setdefault(joint_idx, []).append(event)

    consolidated: list[ContactEvent] = []
    for group in by_joint.values():
        group.sort(key=lambda e: e.start_frame)
        current = group[0]
        for candidate in group[1:]:
            if candidate.start_frame <= current.end_frame:
                current = _merge_two_events(current, candidate)
            else:
                consolidated.append(current)
                current = candidate
        consolidated.append(current)

    return consolidated


def _nearest_depth(depth: np.ndarray, mask: np.ndarray, pixel: np.ndarray) -> float | None:
    """DA3's own depth value at `mask`'s own pixel nearest to `pixel` -- not a
    literal same-pixel lookup, since `pixel` (a joint's 2D projection) can
    land on a pixel the OTHER mask owns (e.g. the object occluding the joint
    it's testing against), in which case `mask` itself has no pixel there at
    all. Returns None if `mask` is empty this frame."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    nearest = np.argmin((xs - pixel[0]) ** 2 + (ys - pixel[1]) ** 2)
    return float(depth[ys[nearest], xs[nearest]])


def depth_gap_for_joint(
    depth: np.ndarray, object_mask: np.ndarray, human_mask: np.ndarray, joint_pixel: np.ndarray,
) -> float | None:
    """Metric depth gap, at one frame, between the tracked object and the
    body surface right where a candidate joint projects -- using an
    already-computed depth map (e.g. Depth-Anything-3's own estimate of the
    real photographed frame), never GVHMR's own retargeted joint Z (this
    project's depth investigation found GVHMR's own estimate unreliable
    there, but never found DA3's own monocular depth unreliable -- this
    reuses the latter, not the former). A small gap means the object and
    body are physically close where their 2D masks overlapped in image space
    (plausible contact); a large gap means that overlap was incidental
    occlusion -- one merely passing in front of or behind the other from the
    camera's viewpoint, not actually touching. Returns None if either mask is
    empty this frame (nothing to compare).
    """
    object_depth = _nearest_depth(depth, object_mask, joint_pixel)
    body_depth = _nearest_depth(depth, human_mask, joint_pixel)
    if object_depth is None or body_depth is None:
        return None
    return abs(object_depth - body_depth)
