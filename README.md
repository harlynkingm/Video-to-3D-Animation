# Video to 3D Motion Capture Animation
Uses SAM3, GVHMR, DepthAnything, and 4DHOI to convert any video with a human and object to a 3D FBX file, generating human and object animation.

## Prerequisites

- **Windows** with an NVIDIA GPU (CUDA 12.8-compatible driver). This project targets native Windows, not WSL2.
- **[Blender](https://www.blender.org/download/) 4.2+**, for the SMPL-X addon below.
- **[pixi](https://pixi.sh)** manages this project's Python environments and dependencies. Install it, then from the repo root:

  ```bash
  pixi install
  ```

Run any script from this project with `pixi run -e <environment> python ...`

## Setup

### 1. Download 3D body models

**These three steps must be done by hand.** SMPL-X and MANO are projects that sit behind free registration and license acceptance on their respective sites, and cannot be auto-downloaded. If you skip this section, stages that require a body or hand model will fail.

1. **SMPL-X**: register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) (free, academic license), download the model files, and place `SMPLX_NEUTRAL.npz` in `body_models/smplx/SMPLX_NEUTRAL.npz`
2. **MANO** (hand model, required by stage 4): register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (free), download the models zip file, and place `MANO_RIGHT.pkl` in `body_models/mano/MANO_RIGHT.pkl`
3. **Blender addon**: install [`jtesch/smplx_blender_addon`](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon) from GitLab (Blender 4.2+)

### 2. Download model checkpoints

After `pixi install` and downloading the body models, the quickest way to get every checkpoint is:

```bash
bash scripts/download_checkpoints.sh
```

This downloads SAM 3.1, ViTPose, HMR2, and GVHMR from HuggingFace and converts the HaMeR checkpoint to a safetensors file, placing everything in `checkpoints/`. It skips files you already have and reminds you about the registration-gated body models that it can't fetch, if you don't have them downloaded.

**Manual alternative:** If you want, you can download each file into `checkpoints/` yourself.

| File | Source | Size |
|---|---|---|
| `sam3.1_multiplex_fp16.safetensors` | [huggingface.co/Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | ~1.6GB |
| `vitpose.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~2.5GB |
| `hmr2.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~2.7GB |
| `gvhmr.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~163MB |
| `hamer.safetensors` (+ `mano_mean_params.npz`) | converted from the HaMeR tarball, see below | ~2.6GB |

HaMeR ships a PyTorch-Lightning `.ckpt` inside a ~6GB tarball. Download [hamer_demo_data.tar.gz](https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz) then convert it using the following script:

```bash
pixi run -e main python scripts/convert_hamer_checkpoint.py path/to/hamer_demo_data.tar.gz
```

## Processing a Video

The pipeline shares state through a single `progress.json` file per run (a "run" being one video processed start to finish). Each stage checks its dependencies already completed before running, and records its own outputs when done. If a stage crashes or you stop partway through, rerunning the same command picks up where it left off instead of starting over.

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
| `--progress-dir` | **Yes** | | Directory to create for this run's state and outputs. |
| `--video-path` | **Yes** | | Path to the source video file. |
| `--human-prompt` | **Yes** | | Text description of the person to track, e.g. `"a tennis player"`. |
| `--focal-length-mm` | **Yes** | | Camera focal length in mm, used to build the intrinsics matrix stage 0 requires. |
| `--sensor-width-mm` | **Yes** | | Camera sensor width in mm, used alongside focal length to build the intrinsics matrix. |
| `--run-id` | No | `--progress-dir`'s own folder name | A human-readable label for the run. Doesn't affect anything on disk. |
| `--object-prompt` | No | none | Text description of the object to track, e.g. `"a basketball"`. Omit if there's no object to track. |
| `--object-shape-hint` | No | `auto` | Forces the tracked object's proxy shape to `box` or `sphere` instead of letting a later stage auto-fit it (relevant once `align_scene_scale` is implemented). |
| `--anchor-frame-override` | No | auto-selected | Forces a specific frame index as the "anchor" frame instead of letting stage 1 pick the frame with the clearest view of the object. |
| `--dump-mask-previews` | No | off | Stage 1 also writes black/white JPEG mask previews for visual spot-checking. See [stage 1](#1-mask-and-track) below. |
| `--dump-motion-preview` | No | off | Stage 2 also writes an AMASS `.npz` importable into Blender for visual spot-checking. See [stage 2](#2-estimate-human-motion) below. |
| `--dump-depth-preview` | No | off | Stage 3 also writes a colored `.ply` point cloud importable into Blender for visual spot-checking. See [stage 3](#3-estimate-depth) below. |
| `--dump-hands-preview` | No | off | Stage 4 also writes a `.bvh` hand-skeleton animation (both hands, bones only) importable into Blender for visual spot-checking. See [stage 4](#4-estimate-hands) below. |
| `--dump-scene-preview` | No | off | Stage 6 also writes a `.ply` combining the human, object, and depth scene in one aligned space for confirming the scale fit in Blender. See [stage 6](#6-align-scene-scale) below. |

**Run each implemented stage, in order**, pointing every one at the same `--progress-dir`:

```bash
pixi run -e main python -m pipeline.stages.stage_0_ingest_video --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_1_mask_and_track --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_4_estimate_hands --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_6_align_scene_scale --progress-dir runs/my_clip
```

Stage 5 (hand retargeting) isn't implemented yet, and stage 6 doesn't depend on it, so the current runnable sequence skips from stage 4 to stage 6. See [Pipeline](#pipeline) below for what each stage does.

## Pipeline

The pipeline is a sequence of stages, each a separate script. This section documents each one individually: what it does, how to run just that stage on its own, and any optional outputs it can produce.

| Stage | Script | Input | Output |
|---|---|---|---|
| 0. Ingest video | `stage_0_ingest_video` | source video file | `frames/*.jpg` <br> camera intrinsics in `progress.json` |
| 1. Mask and track | `stage_1_mask_and_track` | `frames/*.jpg` | `masks/human.pt` <br> `masks/object.pt` <br> anchor frame index in `progress.json` <br> `masks/preview_human/*.jpg` (optional) <br> `masks/preview_object/*.jpg` (optional) |
| 2. Estimate human motion | `stage_2_estimate_human_motion` | `frames/*.jpg` <br> `masks/human.pt` | `motion/human_motion.pt` <br> `motion/blender_preview.npz` (optional) |
| 3. Estimate depth | `stage_3_estimate_depth` | `frames/*.jpg` <br> anchor frame index in `progress.json` | `depth/anchor_depth.npy` <br> `depth/anchor_pointcloud.ply` (optional) |
| 4. Estimate hands | `stage_4_estimate_hands` | `frames/*.jpg` <br> `masks/human.pt` | `hands/hand_pose.npz` <br> `hands/hands_preview.bvh` (optional) |
| 5. Retarget hands *(not yet implemented)* | `stage_5_retarget_hands` | `motion/human_motion.pt` <br> hand pose params | unified per-frame SMPL-X body+hands sequence |
| 6. Align scene scale | `stage_6_align_scene_scale` | `depth/anchor_depth.npy` <br> `motion/human_motion.pt` <br> `masks/human.pt` | `scale/scene_scale.json` <br> `scale/scale_preview.ply` (optional). Object proxy shape (box/sphere) is planned here too but not yet implemented. |
| 7. Annotate contacts *(not yet implemented)* | `stage_7_annotate_contacts` | SMPL-X sequence <br> object proxy shape | per-frame hand↔object contact points |
| 8. Optimize HOI *(not yet implemented)* | `stage_8_optimize_hoi` | contact points <br> object proxy shape | refined SMPL-X sequence <br> per-frame object 6DoF pose |
| 9. Export FBX *(not yet implemented)* | `stage_9_export_fbx` | refined SMPL-X sequence <br> object pose | final `.fbx` |

**Every stage skips itself if `progress.json` already shows it as complete.** Re-running the same command after a successful run just prints `already complete, skipping` rather than redoing the work. Pass `--force` to re-run a stage anyway.

### Initial Stage: Ingest video

```bash
pixi run -e main python -m pipeline.stages.stage_0_ingest_video --progress-dir runs/my_clip
```

Extracts every frame to disk as JPEG (to `runs/my_clip/frames/`), and computes the camera intrinsics matrix from `--focal-length-mm`/`--sensor-width-mm` and the video's actual resolution.

### Stage 1. Mask and track

```bash
pixi run -e main python -m pipeline.stages.stage_1_mask_and_track --progress-dir runs/my_clip
```

SAM 3.1 tracks the human (and object, if `--object-prompt` was given) across every frame, text-prompted. Also resolves which frame later stages use as their object "anchor" (the frame with the clearest view of the object). Uses `--anchor-frame-override` if you specify one when creating the run.

**Optional: JPEG Mask Output.** Use `--dump-mask-previews` when creating the run to also have this stage write `runs/my_clip/masks/preview_human/000000.jpg`, `000001.jpg`, ... (and `preview_object/` if an object was tracked). These are plain black-and-white images at the video's native resolution, indicating white where SAM 3.1 thinks the entity is. You can scroll through these images on disk to confirm it tracked the right thing. This roughly doubles this stage's disk writes, so it's off by default.

### Stage 2. Estimate human motion

```bash
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion --progress-dir runs/my_clip
```

GVHMR turns the tracked human mask into a full-clip SMPL-X body pose, producing both a camera-space and a world-grounded version of the motion. Works at any source video resolution. Both SAM 3.1 (stage 1) and GVHMR internally resize to their own small fixed working resolutions regardless of the input, so 1080p and 4K source video cost the same in GPU memory as a lower-resolution clip, though larger frames do mean more disk space and slightly slower per-frame I/O.

**Optional: 3D Motion Preview Output.** Use `--dump-motion-preview` when creating the run to also have this stage write `runs/my_clip/motion/blender_preview.npz`

This NPZ is importable via the SMPL-X addon's own **Add Animation** operator (`Object > SMPL-X > Add Animation`) in Blender, once the addon is installed per [Setup](#setup). **Important for accurate preview:** When the import dialog appears, **set "Format" to `SMPL-X`, not `AMASS`** to view the 3D animation at the correct orientation.

### Stage 3. Estimate depth

```bash
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth --progress-dir runs/my_clip
```

Depth-Anything-3 (`DA3METRIC-LARGE`) runs once on the anchor frame resolved in stage 1, not the whole clip. This produces a metric depth map in real-world meters. The Depth-Anything-3 checkpoint (~1.3GB) auto-downloads into `checkpoints/depth_anything_3/` the first time this stage runs.

**Optional: PLY Point Cloud Output.** Use `--dump-depth-preview` when creating the run to also have this stage write `runs/my_clip/depth/anchor_pointcloud.ply`, a colored point cloud built by unprojecting the depth map using the anchor frame's own pixel colors. Sky pixels (if any) are excluded. Blender can import this `.ply` file natively via **File > Import > Stanford (.ply)**

**Note:** Blender's default Solid shading mode doesn't display vertex colors in a PLY file. Here is how to get the vertex colors to appear in Blender:
1. `File > Import > Stanford PLY (.ply)`
2. Go to the Geometry Node editor
3. Press `'New'`
4. `Add > Mesh > Operations > Mesh to Points`
5. `Add > Geometry > Material > Set Material`
6. Connect these Nodes
7. Open the Shader Editor
8. Press `'New'`
9. `Add > Input > Attributes > Col`
10. Connect 'Color' to 'Base Color' on 'Principled BSDF'
11. Go back to Geometry Node Editor
12. Set Material of 'Set Material' Node to the material you just created. Color will appear! (Must be in Material Preview or Rendered viewport mode)

### Stage 4. Estimate hands

```bash
pixi run -e main python -m pipeline.stages.stage_4_estimate_hands --progress-dir runs/my_clip
```

HaMeR estimates per-frame MANO hand pose for both hands. It finds the person from the stage 1 mask, runs ViTPose to locate each hand from the wrist/elbow, crops each hand, and predicts finger articulation plus wrist orientation. The output `hands/hand_pose.npz` holds per-frame left/right hand pose, wrist orientation, and a validity flag per hand (a hand may be off-screen or too occluded on a given frame). Attaching it to the GVHMR body happens in stage 5.

This stage requires the MANO body model (see [Setup](#setup)).

**Optional: Hand Skeleton Preview.** Use `--dump-hands-preview` when creating the run to also have this stage write `runs/my_clip/hands/hands_preview.bvh`, a bone-only animation of both hands importable via **File > Import > Motion Capture (.bvh)** in Blender. Each hand is shown in isolation, side by side, animating over the clip, so you can confirm the finger articulation looks right before it's grafted onto a body in stage 5. This preview requires `SMPLX_NEUTRAL.npz` (already required by [Setup](#setup)) but no MANO mesh.

### Stage 6. Align scene scale

```bash
pixi run -e main python -m pipeline.stages.stage_6_align_scene_scale --progress-dir runs/my_clip
```

The depth map ([stage 3](#3-estimate-depth)) and the SMPL-X human body ([stage 2](#2-estimate-human-motion)) are both nominally in real-world meters, but on real footage they disagree by a systematic factor. This stage fits the single scale + translation that reconciles them at the anchor frame by matching the SMPL-X body's visible surface against the depth values under the human mask. The result is written to `runs/my_clip/scale/scene_scale.json` and lets any depth-derived geometry be placed correctly in the human's metric space.

**Optional: Aligned Scene Preview.** Use `--dump-scene-preview` when creating the run to also have this stage write `runs/my_clip/scale/scene_preview.ply`, a single point cloud that puts all three elements in the human's metric space, color-coded so you can confirm the fit in Blender: the **green** SMPL-X body mesh, the **red** tracked object, and the **RGB** depth scene. Import and enable vertex colors the same way as the [stage 3 preview](#3-estimate-depth) above.

### Stage 5, 7, 8, 9. Hand retargeting, contacts, optimization, FBX export

Not yet implemented. These will retarget the hands onto the body, detect and score hand-object contact, refine the full motion against those contacts, and finally export an animated FBX. Each will get its own subsection here once it exists, following the same pattern as the stages above.

## Testing

`tests/` contains whole-stage regression tests, one file per implemented stage plus a full end-to-end test, run against a small (20-frame) committed test clip (`tests/assets/tiny_tennis_clip.mp4`). Each test actually runs the real stage and checks that its outputs look correct.

```bash
pixi run -e main python -m pytest tests/
```

Stage tests require the real SAM 3.1/GVHMR checkpoints and a CUDA GPU (see [Setup](#setup)). If either are missing, tests are skipped, not failed.

## Pixi Details

Installing pixi sets up two environments for this project, each pinned to **Python 3.13**:
- `main` handles most pipeline stages (SAM 3.1, GVHMR, etc.), including a CUDA 12.8 build of PyTorch.
- `fbx-export` is kept separate because it depends on `bpy` (Blender's Python API), which requires its own exact Python version independent of the rest of the stack.

## Licensing

This repo's own code is Apache 2.0, but the checkpoints above carry their own separate license terms (attribution requirements, and a research/personal-use-only restriction on the GVHMR checkpoint specifically). See [NOTICE](NOTICE) before using this project commercially.
