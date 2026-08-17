"""Standalone preview `.blend` builders: the raw FLAME mesh, raw-vs-smoothed
MediaPipe landmarks, and the ARKit-52 CSV visualization, each played back
via Blender's native Mesh Cache modifier reading a PC2 animation. Entirely
separate scenes from `output.blend` itself.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from .blender_constants import _FIRST_MOTION_BLENDER_FRAME
from .blender_scene import _clear_scene

# Matches the identity axis mapping face_preview.py's own module docstring
# establishes ("POS_Z"/"POS_Y" maps file (x, y, z) straight to Blender (x, y,
# z)), defined locally, not imported, since that module imports torch at
# module level (needed for its own FLAME forward pass, which never runs
# here) and the `export` env has no torch installed.
_FACE_PREVIEW_PC2_UP_AXIS = "POS_Z"
_FACE_PREVIEW_PC2_FORWARD_AXIS = "POS_Y"

# Matches face_preview.KEY_TEMPLATE_VERTICES/KEY_TEMPLATE_FACES.
_KEY_TEMPLATE_VERTICES = "vertices"
_KEY_TEMPLATE_FACES = "faces"

# How large the reference video plane reads next to the FLAME mesh, and
# where it sits in Blender units
VIDEO_PLANE_HEIGHT_M = 1.2
VIDEO_PLANE_POSITION_M = (-1.0, -1.0, 0.0)


def _add_video_reference_plane(bpy: types.ModuleType, frames_dir: Path, n_frames: int) -> None:
    """Adds a plane beside the face mesh, textured with the source clip's
    own frame sequence (`stage_0_ingest_video`'s own `NNNNNN.jpg` output,
    zero-indexed and frame-for-frame aligned with the tracked motion) so the
    real footage can be checked without a second window open. Unlit
    (Emission shader): this is a reference image, not a lit scene object,
    and should read the same regardless of the preview's own lighting setup.

    Sign/orientation isn't derived from any other axis-convention constant
    in this file, if the plane reads mirrored or rotated relative to the
    face mesh, this is the function to fix; the mesh/PC2 pipeline needs no
    change either way.
    """
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        return

    image = bpy.data.images.load(str(frame_paths[0]))
    image.source = "SEQUENCE"
    width_px, height_px = image.size

    mesh = bpy.data.meshes.new("video_reference_mesh")
    half_h = VIDEO_PLANE_HEIGHT_M / 2.0
    half_w = half_h * (width_px / height_px)
    x, y, z = VIDEO_PLANE_POSITION_M
    mesh.from_pydata(
        [(x, y - half_w, z - half_h), (x, y + half_w, z - half_h), (x, y + half_w, z + half_h), (x, y - half_w, z + half_h)],
        [], [(0, 1, 2, 3)],
    )
    uv_layer = mesh.uv_layers.new()
    for loop, uv in zip(mesh.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv_layer.data[loop.index].uv = uv
    mesh.update()
    obj = bpy.data.objects.new("video_reference", mesh)
    bpy.context.scene.collection.objects.link(obj)

    material = bpy.data.materials.new("video_reference_material")
    nodes = material.node_tree.nodes  # populated by default, no use_nodes toggle needed (deprecated in 6.0)
    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    emission_node = nodes.new("ShaderNodeEmission")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = image
    image_node.image_user.frame_start = _FIRST_MOTION_BLENDER_FRAME
    image_node.image_user.frame_duration = n_frames
    image_node.image_user.use_auto_refresh = True
    material.node_tree.links.new(image_node.outputs["Color"], emission_node.inputs["Color"])
    material.node_tree.links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    obj.data.materials.append(material)


def _build_face_preview_blend(
    bpy: types.ModuleType, template_npz_path: Path, pc2_path: Path, n_frames: int, fps: float, out_path: Path,
    frames_dir: Path | None = None,
) -> None:
    """A standalone .blend, entirely separate from `output.blend`: builds
    stage 9's raw FLAME template mesh and drives it with a Mesh Cache
    modifier reading the PC2 animation directly. Deliberately isolated
    from the main scene (no SMPL-X expression mapping, no SMPL-X body), see
    `face_preview.py`'s own module docstring for why.

    The template mesh is built by hand from a plain .npz rather than
    imported from an .obj: Blender's OBJ importer applies its own Y-up ->
    Z-up conversion as an object rotation, which would compound with the
    conversion `face_preview.py` already baked into the vertex data.
    Building it directly keeps `matrix_world` at identity.

    `frames_dir`: optional, adds the source video as a reference plane
    beside the mesh (see `_add_video_reference_plane`). Omitted entirely
    when not given, e.g. for a run with no cached frames available.
    """
    _clear_scene(bpy)
    with np.load(template_npz_path) as template:
        vertices = template[_KEY_TEMPLATE_VERTICES]
        faces = template[_KEY_TEMPLATE_FACES]

    mesh = bpy.data.meshes.new("flame_preview_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("flame_preview", mesh)
    bpy.context.scene.collection.objects.link(obj)

    modifier = obj.modifiers.new("face_preview_cache", "MESH_CACHE")
    modifier.cache_format = "PC2"
    modifier.filepath = str(pc2_path)
    modifier.up_axis = _FACE_PREVIEW_PC2_UP_AXIS
    modifier.forward_axis = _FACE_PREVIEW_PC2_FORWARD_AXIS
    modifier.frame_start = float(_FIRST_MOTION_BLENDER_FRAME)

    if frames_dir is not None:
        _add_video_reference_plane(bpy, frames_dir, n_frames)

    # Never set previously, Blender's own scene default (24fps) silently
    # won, so a source clip's real ~60fps footage played back at ~2.5x too
    # slow. `round()`/`fps_base=1.0` is an approximation (drops a fractional
    # rate like 59.94 to 60), fine for a debug preview, `output.blend`'s
    # own fps (via `stage_10_export._write_body_amass`/the SMPL-X addon)
    # already carries the real value exactly and is unaffected by this.
    bpy.context.scene.render.fps = round(fps)
    bpy.context.scene.render.fps_base = 1.0

    bpy.context.scene.frame_end = n_frames - 1 + _FIRST_MOTION_BLENDER_FRAME
    bpy.context.scene.frame_set(_FIRST_MOTION_BLENDER_FRAME)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))


def _build_landmark_preview_blend(
    bpy: types.ModuleType, raw_template_path: Path, raw_pc2_path: Path, smoothed_template_path: Path,
    smoothed_pc2_path: Path, n_frames: int, fps: float, out_path: Path,
) -> None:
    """A standalone .blend with two wireframes side by side, MediaPipe's
    raw 478-point face detection and the same points smoothed at the
    production `FineTuningOptions.face_smoothing_window` setting, entirely upstream of
    DECA/MICA/FLAME, so a clip's own landmark noise is visible before any of
    that machinery runs. See `face_preview.write_landmark_preview`'s own
    docstring for why this exists and what the wireframe connectivity is (not
    real face topology, cosmetic only).

    Two objects rather than `_build_face_preview_blend`'s one: unlike that
    function's single triangle-faced mesh, `face_preview.py` stores EDGE
    pairs here (under the same on-disk key, `_KEY_TEMPLATE_FACES`, for
    template-file-format reuse) since MediaPipe's own face topology isn't
    available in this project's install, `mesh.from_pydata`'s edge
    argument, not its face argument, is what actually needs them.
    """
    _clear_scene(bpy)

    def build(name: str, template_path: Path, pc2_path: Path):
        with np.load(template_path) as template:
            vertices = template[_KEY_TEMPLATE_VERTICES]
            edges = template[_KEY_TEMPLATE_FACES]
        mesh = bpy.data.meshes.new(f"{name}_mesh")
        mesh.from_pydata(vertices.tolist(), edges.tolist(), [])
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        modifier = obj.modifiers.new(f"{name}_cache", "MESH_CACHE")
        modifier.cache_format = "PC2"
        modifier.filepath = str(pc2_path)
        modifier.up_axis = _FACE_PREVIEW_PC2_UP_AXIS
        modifier.forward_axis = _FACE_PREVIEW_PC2_FORWARD_AXIS
        modifier.frame_start = float(_FIRST_MOTION_BLENDER_FRAME)

    build("landmark_preview_raw", raw_template_path, raw_pc2_path)
    build("landmark_preview_smoothed", smoothed_template_path, smoothed_pc2_path)

    bpy.context.scene.render.fps = round(fps)  # see _build_face_preview_blend's own comment for why
    bpy.context.scene.render.fps_base = 1.0

    bpy.context.scene.frame_end = n_frames - 1 + _FIRST_MOTION_BLENDER_FRAME
    bpy.context.scene.frame_set(_FIRST_MOTION_BLENDER_FRAME)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
