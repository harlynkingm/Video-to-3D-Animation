# Video to 3D Motion Capture Animation
Uses SAM3, GVHMR, DepthAnything, and 4DHOI to convert any video with a human and object to a 3D FBX file, generating human and object motion capture animation.

## Prerequisites

- **Windows** with an NVIDIA GPU (CUDA 12.8-compatible driver). This project targets native Windows, not WSL2.
- **[Blender](https://www.blender.org/download/) 4.2+**, for the SMPL-X addon below.
- **[pixi](https://pixi.sh)** manages this project's Python environments and dependencies. Install it, then from the repo root:

  ```bash
  pixi install
  ```

Run any script from this project with `pixi run -e <environment> python ...`

<details>
<summary>Pixi Environment Details</summary>
Installing pixi sets up two environments for this project, each pinned to **Python 3.13**:

- `main` handles most pipeline stages (SAM 3.1, GVHMR, etc.), including a CUDA 12.8 build of PyTorch.
- `fbx-export` is kept separate because it depends on `bpy` (Blender's Python API), which requires its own exact Python version independent of the rest of the stack.
</details>

## Setup

### 1. Download 3D body models

**These three steps must be done by hand.** SMPL-X and MANO are projects that sit behind free registration and license acceptance on their respective sites, and cannot be auto-downloaded. If you skip this section, stages that require a body or hand model will fail.

1. **SMPL-X**: register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) (free), download the model files, and place `SMPLX_NEUTRAL.npz` in `body_models/smplx/SMPLX_NEUTRAL.npz`
2. **MANO** (hand model, required by stage 4): register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (free), download the models zip file, and place `MANO_RIGHT.pkl` in `body_models/mano/MANO_RIGHT.pkl`
3. **Blender addon**: install [`jtesch/smplx_blender_addon`](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon) from GitLab (Blender 4.2+)

### 2. Download model checkpoints

After `pixi install` and downloading the body models, the quickest way to get every checkpoint is:

```bash
bash scripts/download_checkpoints.sh
```

This downloads SAM 3.1, ViTPose, HMR2, and GVHMR from HuggingFace and converts the HaMeR checkpoint to a safetensors file, placing everything in `checkpoints/`. It skips files you already have and reminds you about the registration-gated body models that it can't fetch, if you don't have them downloaded.

<details>
<summary>Manual installation instructions</summary>
If you want, you can download each file into `checkpoints/` yourself.

| File | Source | Size |
|---|---|---|
| `sam3.1_multiplex_fp16.safetensors` | [huggingface.co/Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | ~1.6GB |
| `vitpose.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~2.5GB |
| `hmr2.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~2.7GB |
| `gvhmr.safetensors` | [huggingface.co/apozz/motion-capture-safetensors](https://huggingface.co/apozz/motion-capture-safetensors) | ~163MB |
| `hamer_demo_data.tar.gz` | [cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz](https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz) | ~6GB |

The ~6GB `hamer_demo_data.tar.gz` download is temporary and can be deleted after checkpoint conversion. Convert it to the smaller ~2.6GB checkpoint using the following script:

```bash
pixi run -e main python scripts/convert_hamer_checkpoint.py path/to/hamer_demo_data.tar.gz
```

There is an additional ~1.3GB Depth-Anything-3 checkpoint which is automatically downloaded when [stage 3]((#stage-3-estimate-depth)) runs for the first time.
</details>

## Processing a Video

The pipeline shares state through a single `progress.json` file which tracks the progress of a single run. If a stage crashes or you stop partway through, rerunning the same command picks up where it left off.

**Today, running a video end to end means two steps: create a run, then run each implemented stage in sequence.** This will change in the future.

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

<details>
<summary>All available run options</summary>

| Option | Required | Default | Description |
|---|---|---|---|
| `--progress-dir` | **Yes** | | Directory to create for this run's state and outputs. |
| `--video-path` | **Yes** | | Path to the source video file (MP4, MOV, MPEG, FLV, or WMV). |
| `--human-prompt` | **Yes** | | Text description of the person to track, e.g. `"a tennis player"`. |
| `--focal-length-mm` | **Yes** | | Camera focal length in mm, used to build the intrinsics matrix stage 0 requires. |
| `--sensor-width-mm` | **Yes** | | Camera sensor width in mm, used alongside focal length to build the intrinsics matrix. |
| `--run-id` | No | `--progress-dir`'s own folder name | A human-readable label for the run. Doesn't affect anything on disk. |
| `--object-prompt` | No | none | Text description of the object to track, e.g. `"a basketball"`. Omit if there's no object to track. |
| `--object-shape-hint` | No | `auto` | Forces the tracked object's proxy shape to `box` or `sphere` instead of letting a later stage auto-fit it (relevant once `align_scene_scale` is implemented). |
| `--anchor-frame-override` | No | auto-selected | Forces a specific frame index as the "anchor" frame instead of letting stage 1 pick the frame with the clearest view of the object. |
| `--dump-mask-previews` | No | off | Stage 1 also writes black/white JPEG mask previews for visual spot-checking. See [stage 1](#stage-1-mask-and-track) below. |
| `--dump-motion-preview` | No | off | Stage 2 also writes an AMASS `.npz` importable into Blender for visual spot-checking. See [stage 2](#stage-2-estimate-human-motion) below. |
| `--dump-depth-preview` | No | off | Stage 3 also writes a colored `.ply` point cloud importable into Blender for visual spot-checking. See [stage 3](#stage-3-estimate-depth) below. |
| `--dump-hands-preview` | No | off | Stage 4 also writes a `.bvh` hand-skeleton animation (both hands, bones only) importable into Blender for visual spot-checking. See [stage 4](#stage-4-estimate-hands) below. |
| `--dump-scene-preview` | No | off | Stage 6 also writes a `.ply` combining the human, object, and depth scene in one aligned space for confirming the scale fit in Blender. See [stage 6](#stage-6-align-scene-scale) below. |
</details>

**Run each stage, in order**, pointing every one at the same `--progress-dir`:

```bash
pixi run -e main python -m pipeline.stages.stage_0_ingest_video --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_1_mask_and_track --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_4_estimate_hands --progress-dir runs/my_clip
pixi run -e main python -m pipeline.stages.stage_6_align_scene_scale --progress-dir runs/my_clip
```

## Pipeline

The pipeline is a sequence of stages, each a separate script. This section documents each one individually: what the stage does, how to run just that stage on its own, and any optional outputs it can produce.

<details>
<summary>All stage input/output details</summary>

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

</details>

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

SAM 3.1 tracks the human (and object, if `--object-prompt` was given). Also resolves which frame to use as an object "anchor" frame, which has the clearest object view. Uses `--anchor-frame-override` if you specify one when creating the run.

<details>
<summary><strong>Optional: JPEG Mask Output</strong></summary>

Use `--dump-mask-previews` when creating the run to also have this stage write `runs/my_clip/masks/preview_human/000000.jpg`, `000001.jpg`, ... (and `preview_object/` if an object was tracked). These are plain black-and-white mask images at the video's native resolution. You can scroll through these images on disk to confirm SAM 3.1 tracked the right thing.
</details>

### Stage 2. Estimate human motion

```bash
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion --progress-dir runs/my_clip
```

GVHMR turns the human mask into a 3D SMPL-X body pose animation. Works at any source video resolution, however larger frames mean more disk space and slightly slower per-frame I/O.

<details>
<summary><strong>Optional: 3D Motion Preview Output</strong></summary>

Use `--dump-motion-preview` when creating the run to also have this stage write `runs/my_clip/motion/blender_preview.npz` This NPZ is importable in Blender via the SMPL-X addon's own **Add Animation** operator (`Object > SMPL-X > Add Animation`) if the addon is installed (see [Setup](#setup)). **For accurate preview,** when the import dialog appears, **set "Format" to `SMPL-X`, not `AMASS`** to view the 3D animation at the correct orientation.
</details>

### Stage 3. Estimate depth

```bash
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth --progress-dir runs/my_clip
```

Depth-Anything-3 (`DA3METRIC-LARGE`) runs once on a single anchor frame, not the whole clip. This produces a depth map in real-world meters.

<details>
<summary><strong>Optional: PLY Point Cloud Output</strong></summary>

Use `--dump-depth-preview` when creating the run to also have this stage write `runs/my_clip/depth/anchor_pointcloud.ply`, a colored point cloud estimating the depth in the image. Blender can import this `.ply` file natively via **File > Import > Stanford (.ply)**
</details>

**Note:** The imported .ply may appear all-black in Blender by default.

<details>
<summary>Details on how to get .ply colors to appear</summary>

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
12. Set Material of 'Set Material' Node to the material you just created. Color will appear!

**Note:** Must be in Material Preview or Rendered viewport mode.
</details>

### Stage 4. Estimate hands

```bash
pixi run -e main python -m pipeline.stages.stage_4_estimate_hands --progress-dir runs/my_clip
```

HaMeR estimates per-frame MANO hand pose for both hands. It finds the person from the stage 1 mask, runs ViTPose to locate each hand, crops in, and predicts finger articulation plus wrist orientation. The output `hands/hand_pose.npz` holds per-frame left/right hand pose, wrist orientation, and validity per hand (a hand may be off-screen or too occluded in a given frame). Attaching it to the body happens in stage 5.

This stage requires the MANO body model (see [Setup](#setup)).

<details>
<summary><strong>Optional: Hand Skeleton Preview</strong></summary>

Use `--dump-hands-preview` when creating the run to also have this stage write `runs/my_clip/hands/hands_preview.bvh`, a bone-only animation of both hands. This .bvh is importable in Blender via **File > Import > Motion Capture (.bvh)**. Each hand is shown in isolation, side by side, so you can confirm the finger articulation looks right before it's attached to a body. This preview requires `SMPLX_NEUTRAL.npz` (see [Setup](#setup)).
</details>

### Stage 6. Align scene scale

```bash
pixi run -e main python -m pipeline.stages.stage_6_align_scene_scale --progress-dir runs/my_clip
```

The depth map ([stage 3](#stage-3-estimate-depth)) and SMPL-X human body ([stage 2](#stage-2-estimate-human-motion)) are both represented in real-world meters, but disagree on scale. This stage reconciles them at the anchor frame by matching the SMPL-X body pose against depth values within the SAM-3 human mask. The result is written to `runs/my_clip/scale/scene_scale.json`

<details>
<summary><strong>Optional: Aligned Scene Preview</strong></summary>

Use `--dump-scene-preview` when creating the run to also have this stage write `runs/my_clip/scale/scene_preview.ply`, a single point cloud that puts all three elements in the human's metric space, color-coded so you can confirm the fit. Import in Blender and enable vertex colors the same way as [stage 3](#stage-3-estimate-depth).
</details>

### Stage 5, 7, 8, 9. Hand retargeting, contacts, optimization, FBX export

Not yet implemented.

## Testing

`tests/` contains whole-stage regression tests, one file per implemented stage, plus a full end-to-end test. Tests are run against a small (20-frame) committed test clip (`tests/assets/tiny_tennis_clip.mp4`). Each test runs the real stage and checks that its outputs look correct.

```bash
pixi run -e main python -m pytest tests/
```

Stage tests require the real SAM 3.1/GVHMR checkpoints and a CUDA GPU (see [Setup](#setup)). If either are missing, tests are skipped, not failed.

## Licensing

This repo's own code is Apache 2.0, but the checkpoints above carry their own separate license terms (attribution requirements, and a research/personal-use-only restriction on the GVHMR checkpoint specifically). See [NOTICE](NOTICE) before using this project commercially.
