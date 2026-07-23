# Single-Point Video to 3D Animation
Uses SAM3, GVHMR, DepthAnything, and 4DHOI to convert a single monocular video to a 3D FBX file, generating SMPL-X human and object animation.

## Prerequisites

- **Windows** with an NVIDIA GPU (CUDA 12.8-compatible driver). This project targets native Windows, not WSL2.
- **[Blender](https://www.blender.org/download/) 4.2+**, for the SMPL-X addon below.
- **[pixi](https://pixi.sh)** manages this project's Python environments and dependencies. Install it, then from the repo root:

  ```bash
  pixi install
  ```

  This sets up two environments, each pinned to **Python 3.13**:
  - `main` handles most pipeline stages (SAM 3.1, GVHMR, etc.), including a CUDA 12.8 build of PyTorch.
  - `fbx-export` is kept separate because it depends on `bpy` (Blender's Python API), which needs its own exact Python version independent of the rest of the stack.

  Run any script inside one of these with `pixi run -e <environment> python ...` (e.g. `pixi run -e main python -m pipeline.stages.stage_0_ingest_video ...`).

## Setup

### 1. SMPL-X body model

1. Register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) (free, academic license) and download the SMPL-X model files. Registration is required, since these files can't be redistributed with this repo.
2. Place `SMPLX_NEUTRAL.npz` at `body_models/smplx/SMPLX_NEUTRAL.npz` in the repo root (gitignored, same as `checkpoints/`).
3. Install the Blender addon from [`jtesch/smplx_blender_addon`](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon) on GitLab (Blender 4.2+, via the Extensions system). Use this one specifically, not the older archived `Meshcapade/SMPL_blender_addon`.

### 2. Model checkpoints

Download these files and place them in a `checkpoints/` folder at the repo root (this folder is gitignored, so nothing here gets committed):

| File | Source | Size |
|---|---|---|
| `sam3.1_multiplex_fp16.safetensors` | [huggingface.co/Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | ~1.6GB |
| `vitpose.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~2.5GB |
| `hmr2.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~2.7GB |
| `gvhmr.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~163MB |

Depth-Anything-3's checkpoint (`DA3METRIC-LARGE`, ~1.3GB) isn't in this table because it needs no manual step: it auto-downloads into `checkpoints/depth_anything_3/` the first time that stage runs.

None of these require registration.

## Processing a Video

The pipeline shares state through one `progress.json` file per run (a "run" being one video processed start to finish). Each stage checks its dependencies already completed before running, and records its own outputs when done. If a stage crashes or you stop partway through, rerunning the same command picks up where it left off instead of starting over.

**Today, running a video end to end means two steps: create a run, then run each implemented stage in sequence.** This will change in the future.

A single one-shot command is planned (see the [Pipeline](#pipeline) section), but until then, this is the full process:

**Create a run:**

```bash
pixi run -e main python -m pipeline.create_run \
  --progress-dir runs/my_clip \
  --video-path /path/to/video.mp4 \
  --human-prompt "a person" \
  --object-prompt "a basketball" \
  --focal-length-mm 26 \
  --sensor-width-mm 36
```

All available options:

| Option | Required | Default | Description |
|---|---|---|---|
| `--progress-dir` | Yes | | Directory to create for this run's state and outputs. |
| `--video-path` | Yes | | Path to the source video file. |
| `--human-prompt` | Yes | | Text description of the person to track, e.g. `"a tennis player"`. |
| `--focal-length-mm` | Yes | | Camera focal length in mm, used to build the intrinsics matrix stage 0 needs. |
| `--sensor-width-mm` | Yes | | Camera sensor width in mm, used alongside focal length to build the intrinsics matrix. |
| `--run-id` | No | `--progress-dir`'s own folder name | A human-readable label for the run. Doesn't affect anything on disk. |
| `--object-prompt` | No | none | Text description of the object to track, e.g. `"a basketball"`. Omit if there's no object to track. |
| `--object-shape-hint` | No | `auto` | Forces the tracked object's proxy shape to `box` or `sphere` instead of letting a later stage auto-fit it (relevant once `align_scene_scale` is implemented). |
| `--anchor-frame-override` | No | auto-selected | Forces a specific frame index as the "anchor" frame instead of letting stage 1 pick the frame with the clearest view of the object. |
| `--dump-mask-previews` | No | off | Stage 1 also writes black/white JPEG mask previews for visual spot-checking. See [stage 1](#1-mask-and-track) below. |
| `--dump-motion-preview` | No | off | Stage 2 also writes an AMASS `.npz` importable into Blender for visual spot-checking. See [stage 2](#2-estimate-human-motion) below. |
| `--dump-depth-preview` | No | off | Stage 3 also writes a colored `.ply` point cloud importable into Blender for visual spot-checking. See [stage 3](#3-estimate-depth) below. |

**Run each implemented stage, in order**, pointing every one at the same `--progress-dir`:

```bash
pixi run -e main python -m pipeline.stages.stage_0_ingest_video --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_1_mask_and_track --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth --progress-dir runs/my_clip
```

That's everything implemented so far. See [Pipeline](#pipeline) below for what each stage does.

## Pipeline

The pipeline is a sequence of stages, each a separate script. This section documents each one individually: what it does, how to run just that stage on its own, and any optional outputs it can produce.

| Stage | Script | Input | Output |
|---|---|---|---|
| 0. Ingest video | `stage_0_ingest_video` | source video file | `frames/*.jpg`, camera intrinsics `K` (stored in `progress.json`) |
| 1. Mask and track | `stage_1_mask_and_track` | `frames/*.jpg` | `masks/human.pt`, `masks/object.pt` (if object given), anchor frame index, `masks/preview_human/*.jpg` (optional), `masks/preview_object/*.jpg` (optional) |
| 2. Estimate human motion | `stage_2_estimate_human_motion` | `frames/*.jpg`, `masks/human.pt` | `motion/human_motion.pt` (camera-space + world-grounded SMPL-X body pose), `motion/blender_preview.npz` (optional) |
| 3. Estimate depth | `stage_3_estimate_depth` | `frames/*.jpg` (anchor frame), anchor frame index | `depth/anchor_depth.npy` (metric depth, meters), `depth/anchor_pointcloud.ply` (optional) |
| 4. Estimate hands *(not yet implemented)* | `stage_4_estimate_hands` | `frames/*.jpg` | per-frame hand pose params |
| 5. Retarget hands *(not yet implemented)* | `stage_5_retarget_hands` | `motion/human_motion.pt`, hand pose params | unified per-frame SMPL-X body+hands sequence |
| 6. Align scene scale *(not yet implemented)* | `stage_6_align_scene_scale` | depth map, SMPL-X sequence, `masks/object.pt` | metric scale factor, object proxy shape (box/sphere) |
| 7. Annotate contacts *(not yet implemented)* | `stage_7_annotate_contacts` | SMPL-X sequence, object proxy shape | per-frame hand↔object contact points |
| 8. Optimize HOI *(not yet implemented)* | `stage_8_optimize_hoi` | contact points, object proxy shape | refined SMPL-X sequence, per-frame object 6DoF pose |
| 9. Export FBX *(not yet implemented)* | `stage_9_export_fbx` | refined SMPL-X sequence, object pose | final `.fbx` |

**Every stage skips itself if `progress.json` already shows it as complete.** Re-running the same command after a successful run just prints `already complete, skipping` rather than redoing the work. Pass `--force` to re-run a stage anyway.

### 0. Ingest video

```bash
pixi run -e main python -m pipeline.stages.stage_0_ingest_video --progress-dir runs/my_clip
```

Extracts every frame to disk as JPEG (to `runs/my_clip/frames/`), and computes the camera intrinsics matrix from `--focal-length-mm`/`--sensor-width-mm` and the video's actual resolution.

### 1. Mask and track

```bash
pixi run -e main python -m pipeline.stages.stage_1_mask_and_track --progress-dir runs/my_clip
```

SAM 3.1 tracks the human (and object, if `--object-prompt` was given) across every frame, text-prompted. Also resolves which frame later stages use as their object "anchor" (the frame with the clearest view of the object). Uses `--anchor-frame-override` if you specify one when creating the run.

**Optional: JPEG Mask Output.** Use `--dump-mask-previews` when creating the run to also have this stage write `runs/my_clip/masks/preview_human/000000.jpg`, `000001.jpg`, ... (and `preview_object/` if an object was tracked). These are plain black-and-white images at the video's native resolution, indicating white where SAM 3.1 thinks the entity is. You can scroll through these images on disk to confirm it tracked the right thing. This roughly doubles this stage's disk writes, so it's off by default.

### 2. Estimate human motion

```bash
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion --progress-dir runs/my_clip
```

GVHMR turns the tracked human mask into a full-clip SMPL-X body pose, producing both a camera-space and a world-grounded version of the motion. Works at any source video resolution. Both SAM 3.1 (stage 1) and GVHMR internally resize to their own small fixed working resolutions regardless of the input, so 1080p and 4K source video cost the same in GPU memory as a lower-resolution clip, though larger frames do mean more disk space and slightly slower per-frame I/O.

**Optional: 3D Motion Preview Output.** Use `--dump-motion-preview` when creating the run to also have this stage write `runs/my_clip/motion/blender_preview.npz`

The NPZ is an AMASS-format file (hands and face are left flat/neutral, since GVHMR doesn't estimate those). This NPZ is importable via the SMPL-X addon's own **Add Animation** operator (`Object > SMPL-X > Add Animation`) in Blender, once the addon is installed per [Setup](#setup). **Important (for accurate preview):** When the import dialog appears, **set "Format" to `SMPL-X`, not `AMASS`** to view the 3D animation at the correct orientation.

### 3. Estimate depth

```bash
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth --progress-dir runs/my_clip
```

Depth-Anything-3 (`DA3METRIC-LARGE`) runs once on the anchor frame resolved in stage 1, not the whole clip. This produces a metric depth map in real-world meters. The Depth-Anything-3 checkpoint (~1.3GB) auto-downloads into `checkpoints/depth_anything_3/` the first time this stage runs, with no manual action needed.

**Optional: PLY Point Cloud Output.** Use `--dump-depth-preview` when creating the run to also have this stage write `runs/my_clip/depth/anchor_pointcloud.ply`, a colored point cloud (meters) built by unprojecting the depth map using the anchor frame's own pixel colors. Sky pixels (if any are detected) are excluded. Blender can import this `.ply` file natively via **File > Import > Stanford (.ply)** with no addon needed.

**Note:** Blender's default Solid shading mode doesn't display vertex colors. Here is how to get the vertex colors to appear in Blender, from the PLY file:
1. Import the PLY
2. Go to the Geometry Node editor
3. Press 'New'
4. Add > Mesh > Operations > Mesh to Points
5. Add > Geometry > Material > Set Material 
6. Connect these Nodes
7. Open the Shader Editor
8. Press 'New'
9. Add > Input > Attributes > Col
10. Connect 'Color' to 'Base Color' on 'Principled BSDF'
11. Go back to Geometry Node Editor
12. Set Material of 'Set Material' Node to the material you just created. Color will appear! (Must be in Material Preview or Rendered mode)

### 4–9. Hands, scene scale, contacts, optimization, FBX export

Not yet implemented. These will estimate per-frame hand pose, align everything to real-world metric scale, detect and score hand-object contact, refine the full motion against those contacts, and finally export an animated FBX. Each will get its own subsection here once it exists, following the same pattern as the stages above.

## Testing

`tests/` holds whole-stage regression tests, one file per implemented stage plus a full end-to-end test, run against a small (20-frame) committed test clip (`tests/assets/tiny_tennis_clip.mp4`). Each test actually runs the real stage and checks its outputs look correct (right shapes, no NaN, plausible mask areas and joint rotations), not just that it doesn't crash.

```bash
pixi run -e main python -m pytest tests/
```

Stage tests need the real SAM 3.1/GVHMR checkpoints and a CUDA GPU (see [Setup](#setup)). If either are missing, tests are skipped, not failed.

## Licensing

This repo's own code is Apache 2.0, but the checkpoints above carry their own separate license terms (attribution requirements, and a research/personal-use-only restriction on the GVHMR checkpoint specifically). See [NOTICE](NOTICE) before using this project commercially.
