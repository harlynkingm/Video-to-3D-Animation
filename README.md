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

Once setup is complete, the pipeline runs fully offline.

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

## Quick Start

Create a run and process a video:

```bash
pixi run -e main python -m pipeline.run \
  --input-video /path/to/video.mp4 \
  --output-dir runs/my_clip \
  --human-prompt "a person" \
  --object-prompt "a teddy bear" \
  --focal-length-mm 26 \
  --sensor-width-mm 36
```

This creates a progress file at `runs/my_clip/progress.json` then runs every stage in order.

<details>
<summary>All available run options</summary>

| Option | Required | Default | Description |
|---|---|---|---|
| `--input-video` | **Yes** | | Path to the source video file (MP4, MOV, MPEG, FLV, or WMV). |
| `--output-dir` (`-o`) | **Yes** | | Directory to create for this run's state and outputs. |
| `--human-prompt` | **Yes** | | Text description of the person to track, e.g. `"a tennis player"`. |
| `--focal-length-mm` | **Yes** | | Camera focal length in mm, used to build the intrinsics matrix stage 0 requires. |
| `--sensor-width-mm` | **Yes** | | Camera sensor width in mm, used alongside focal length to build the intrinsics matrix. |
| `--run-id` | No | `--output-dir`'s own folder name | A human-readable label for the run. Doesn't affect anything on disk. |
| `--object-prompt` | No | none | Text description of the object to track, e.g. `"a teddy bear"`. Omit if there's no object to track. |
| `--object-shape-hint` | No | `auto` | Forces the tracked object's proxy shape to `box`, `ellipsoid`, or `cylinder` instead of letting stage 6 auto-fit whichever shape better matches the object. |
| `--anchor-frame-override` | No | auto-selected | Forces a specific frame index as the "anchor" frame instead of letting stage 1 pick the frame with the clearest view of the object. |
| `--stop-after-stage` | No | runs every implemented stage | Stops after the given stage number, inclusive -- e.g. `5` runs stages 0-5 and stops before stage 6. |
| `--render-previews` | No | off | Enables every `--render-*-preview` flag below at once. |
| `--render-mask-previews` | No | off | Stage 1 also writes black/white JPEG mask previews for visual spot-checking. See [stage 1](#stage-1-mask-and-track) below. |
| `--render-motion-preview` | No | off | Stage 2 also writes an AMASS `.npz` importable into Blender for visual spot-checking. See [stage 2](#stage-2-estimate-human-motion) below. |
| `--render-depth-preview` | No | off | Stage 3 also writes a colored `.ply` point cloud importable into Blender for visual spot-checking. See [stage 3](#stage-3-estimate-depth) below. |
| `--render-hands-preview` | No | off | Stage 4 also writes a `.bvh` hand-skeleton animation (both hands, bones only) importable into Blender for visual spot-checking. See [stage 4](#stage-4-estimate-hands) below. |
| `--render-retarget-preview` | No | off | Stage 5 also writes a `.bvh` full-body-plus-hands skeleton animation importable into Blender for confirming the hands sit correctly on the body. See [stage 5](#stage-5-retarget-hands) below. |
| `--render-scene-preview` | No | off | Stage 6 also writes a `.ply` combining the human, object, and depth scene in one aligned space for confirming the scale fit in Blender. See [stage 6](#stage-6-align-scene-scale) below. |
| `--render-contacts-preview` | No | off | Stage 7 also writes one annotated JPEG per contact event, circling the contact point on the source frame, for visual spot-checking. See [stage 7](#stage-7-annotate-contacts) below. |
</details>

**Running stages individually**: `pipeline.run` is just `pipeline.create_run` followed by every stage's own script, run in sequence. See [Pipeline](#pipeline) below for each stage's individual command and options.

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
| 5. Retarget hands | `stage_5_retarget_hands` | `motion/human_motion.pt` <br> `hands/hand_pose.npz` | `retarget/retargeted_motion.pt` <br> `retarget/retarget_preview.bvh` (optional) |
| 6. Align scene scale | `stage_6_align_scene_scale` | `depth/anchor_depth.npy` <br> `motion/human_motion.pt` <br> `masks/human.pt` <br> `masks/object.pt` (optional) | `scale/scene_scale.json` <br> `scale/object_shape.json` (if an object was tracked) <br> `scale/scene_preview.ply` (optional) |
| 7. Annotate contacts | `stage_7_annotate_contacts` | `retarget/retargeted_motion.pt` <br> `masks/object.pt` | `contacts/contact_events.json` <br> `contacts/contacts_preview/*.jpg` (optional) |
| 8. Optimize HOI *(not yet implemented)* | `stage_8_optimize_hoi` | contact points <br> object proxy shape | refined SMPL-X sequence <br> per-frame object 6DoF pose |
| 9. Export FBX *(not yet implemented)* | `stage_9_export_fbx` | refined SMPL-X sequence <br> object pose | final `.fbx` |

</details>

**Every stage skips itself if `progress.json` already shows it as complete.** Pass `--force` to re-run a stage anyway.

### Create a Run File

The pipeline shares state through a single `progress.json` file which tracks the progress of a single run. If a stage crashes or you stop partway through, rerunning the same command picks up where it left off.

To create a `progress.json` file:

```bash
pixi run -e main python -m pipeline.create_run \
  --input-video /path/to/video.mp4 \
  --output-dir runs/my_clip \
  --human-prompt "a person" \
  --object-prompt "a teddy bear" \
  --focal-length-mm 26 \
  --sensor-width-mm 36
```

### Initial Stage: Process video

```bash
pixi run -e main python -m pipeline.stages.stage_0_ingest_video -o runs/my_clip
```

Extracts every frame to disk as JPEG (to `runs/my_clip/frames/`), and computes the camera intrinsics matrix from `--focal-length-mm`/`--sensor-width-mm` and the video's actual resolution.

### Stage 1. Mask and track

```bash
pixi run -e main python -m pipeline.stages.stage_1_mask_and_track -o runs/my_clip
```

SAM 3.1 tracks the human (and object, if `--object-prompt` was given). Also resolves which frame to use as an object "anchor" frame, which has the clearest object view. Uses `--anchor-frame-override` if you specify one when creating the run.

<details>
<summary><strong>Optional: JPEG Mask Output</strong></summary>

Use `--render-mask-previews` when creating the run to also have this stage write `runs/my_clip/masks/preview_human/000000.jpg`, `000001.jpg`, ... (and `preview_object/` if an object was tracked). These are plain black-and-white mask images at the video's native resolution. You can scroll through these images on disk to confirm SAM 3.1 tracked the right thing.
</details>

### Stage 2. Estimate human motion

```bash
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion -o runs/my_clip
```

GVHMR turns the human mask into a 3D SMPL-X body pose animation. Works at any source video resolution, however larger frames mean more disk space and slightly slower per-frame I/O.

The body motion is temporally smoothed before saving to remove residual per-frame jitter.

If you want to tune the smoothing method yourself, edit `body_smoothing_window` (affecting rotation) or `body_translation_cutoff` (affecting root position) in the run's `progress.json` before running this stage. See [Motion smoothing](#motion-smoothing) below.

<details>
<summary><strong>Optional: 3D Motion Preview Output</strong></summary>

Use `--render-motion-preview` when creating the run to also have this stage write `runs/my_clip/motion/blender_preview.npz` This NPZ is importable in Blender via the SMPL-X addon's own **Add Animation** operator (`Object > SMPL-X > Add Animation`) if the addon is installed (see [Setup](#setup)). **For accurate preview,** when the import dialog appears, **set "Format" to `SMPL-X`, not `AMASS`** to view the 3D animation at the correct orientation.
</details>

### Stage 3. Estimate depth

```bash
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth -o runs/my_clip
```

Depth-Anything-3 (`DA3METRIC-LARGE`) runs once on a single anchor frame, not the whole clip. This produces a depth map in real-world meters.

<details>
<summary><strong>Optional: PLY Point Cloud Output</strong></summary>

Use `--render-depth-preview` when creating the run to also have this stage write `runs/my_clip/depth/anchor_pointcloud.ply`, a colored point cloud estimating the depth in the image. Blender can import this `.ply` file natively via **File > Import > Stanford (.ply)**
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
pixi run -e main python -m pipeline.stages.stage_4_estimate_hands -o runs/my_clip
```

HaMeR estimates per-frame MANO hand pose for both hands. It finds the person from the stage 1 mask, runs ViTPose to locate each hand, crops in, and predicts finger articulation plus wrist orientation. The output `hands/hand_pose.npz` holds per-frame left/right hand pose, wrist orientation, and validity per hand. Attaching it to the body happens in stage 5.

If a hand is off-screen, too occluded, or if the ViTPose based confidence score is too low, or if the estimated wrist pose is anatomically impossible, the wrist is held in place instead of assuming incorrect motion.

This stage also temporally smooths both hand movements after making corrections based on the above. The smoothing happens in 3 passes: first a zero-phase smoothing to reduce jitter, then an adaptive filter, then a keyframe-based reduction. Smoothness tuning can be adjusted in the `hand_*` fields of the run's `progress.json`. See [Motion smoothing](#motion-smoothing) below.

This stage requires the MANO body model and the SMPL-X model file (see [Setup](#setup)).

<details>
<summary><strong>Optional: Hand Skeleton Preview</strong></summary>

Use `--render-hands-preview` when creating the run to also have this stage write `runs/my_clip/hands/hands_preview.bvh`, a bone-only animation of both hands. This .bvh is importable in Blender via **File > Import > Motion Capture (.bvh)**. Each hand is shown in isolation, side by side, so you can confirm the finger articulation looks right before it's attached to a body. This preview requires `SMPLX_NEUTRAL.npz` (see [Setup](#setup)).
</details>

### Stage 5. Retarget hands

```bash
pixi run -e main python -m pipeline.stages.stage_5_retarget_hands -o runs/my_clip
```

Attaches the stage 4 hands onto the stage 2 body, producing one merged full-body-plus-hands SMPL-X sequence in `runs/my_clip/retarget/retargeted_motion.pt`. A hand never detected anywhere in the clip keeps GVHMR's own wrist and flat fingers throughout. Any hand detected at least once gets every frame filled with that detected pose.

This stage requires `SMPLX_NEUTRAL.npz` (see [Setup](#setup)).

<details>
<summary><strong>Optional: Full Body and Hands Preview</strong></summary>

Use `--render-retarget-preview` when creating the run to also have this stage write `runs/my_clip/retarget/retarget_preview.bvh`, a bone-only animation of the whole body with the stage 4 hands attached, including the body's real motion (walking, sitting down, etc.). This .bvh is importable in Blender via **File > Import > Motion Capture (.bvh)**

This preview requires `SMPLX_NEUTRAL.npz` (see [Setup](#setup)).
</details>

### Stage 6. Align scene scale

```bash
pixi run -e main python -m pipeline.stages.stage_6_align_scene_scale -o runs/my_clip
```

The depth map ([stage 3](#stage-3-estimate-depth)) and SMPL-X human body ([stage 2](#stage-2-estimate-human-motion)) are both represented in real-world meters, but disagree on scale. This stage reconciles them at the anchor frame by matching the SMPL-X body pose against depth values within the SAM-3 human mask. The result is written to `runs/my_clip/scale/scene_scale.json`

If an object was tracked, this stage also fits a shape to it (a box, ellipsoid, or cylinder chosen via `--object-shape-hint`, or 'auto' which automatically chooses the shape of best fit), written to `runs/my_clip/scale/object_shape.json`.

<details>
<summary><strong>Optional: Aligned Scene Preview</strong></summary>

Use `--render-scene-preview` when creating the run to also have this stage write `runs/my_clip/scale/scene_preview.ply`, a single point cloud that puts every element in the human's metric space, color-coded so you can confirm the fit: green for the SMPL-X body, red for the tracked object's depth points, and (if an object was tracked) a yellow wireframe of its fitted primitive shape. Import in Blender and enable vertex colors the same way as [stage 3](#stage-3-estimate-depth).
</details>

### Stage 7. Detect human-object contact points

```bash
pixi run -e main python -m pipeline.stages.stage_7_annotate_contacts -o runs/my_clip
```

Detects contact points between the body and object across 8 body regions. The mask + depth inferred by Depth-Anything-3 are used to determine where the body and object interact. Interaction events written to `runs/my_clip/contacts/contact_events.json`

<details>
<summary><strong>Optional: Contact Points Preview</strong></summary>

Use `--render-contacts-preview` when creating the run to also have this stage write `runs/my_clip/contacts/contacts_preview/` This saves one JPEG per contact event named `{peak_confidence_frame:06d}_{regions-joined-by-plus}.jpg`. Each image is the source frame at that event's most-confident moment, with a circle drawn around the joint that triggered it.
</details>

### Stage 8, 9. Optimization, FBX export

Not yet implemented.

### Pausing and Resuming a Run

If you interrupt a run and want to continue it, just re-run with `--output-dir` pointing at the same folder.

```bash
pixi run -e main python -m pipeline.run --output-dir runs/my_clip
```

Passed stages are skipped. You can also add new flags to the run this way.

### Update a Run File

If you want to update some of the run options used in a `progress.json` file without opening the file in a text editor, you can use `pipeline.update_run` with any of the run options listed in the [Quick Start](#quick-start).

For example, the following command updates a run file to render preview files for every stage:

```bash
pixi run -e main python -m pipeline.update_run \
  --output-dir runs/my_clip \
  --render-previews
```

After updating, a backup of your previous version is saved in the same folder, in case you want to revert your changes.

## Motion smoothing

Stage 2 (body) and stage 4 (hands) temporally smooth their output before saving. The hands need much stronger smoothing than the body because HaMeR runs independently per frame, while GVHMR already runs a temporal model over the entire video.

<details>
<summary>Smoothness tuning</summary>

If you want to tune the amount of smoothing, edit these fields in the run's `progress.json` before running stage 2 or 4:

| Field | Default | Effect |
|---|---|---|
| `body_smoothing_window` | `9` | Savitzky-Golay window (odd, in frames) for body rotation. Larger is smoother but can smear fast motion. |
| `body_translation_cutoff` | `0.15` | Butterworth low-pass cutoff (fraction of Nyquist) for the body root position. Lower is smoother but adds lag. |
| `hand_smoothing_window` | `15` | Savitzky-Golay pre-pass window (odd, in frames), applied to both fingers and wrist. Larger strips more raw jitter without adding lag; too large smears fast motion. |
| `hand_beta` | `0.3` | How quickly the adaptive filter loosens as motion speeds up (both fingers and wrist). Higher tracks fast motion more closely, at the cost of passing more jitter through while moving. |
| `hand_finger_min_cutoff_hz` | `0.15` | Adaptive-filter smoothing strength at rest for the **fingers**. Lower is steadier when still but slower to respond. |
| `hand_wrist_min_cutoff_hz` | `0.10` | Same, for the **wrist** — lower than the fingers, since the wrist starts from noisier data and needs more smoothing. |
| `hand_finger_decimate_deg` | `1.5` | Keyframe-reduction tolerance for the **fingers**, in degrees: the most the refitted curve may deviate from the filtered motion. Larger means fewer keyframes and a smoother, flatter curve. |
| `hand_wrist_decimate_deg` | `3.0` | Same, for the **wrist** (looser than the fingers). |
| `hand_max_wrist_deviation_deg` | `110.0` | Max plausible wrist rotation degrees relative to the forearm, checked against stage 2's body motion before smoothing. A raw estimate past this is treated as 'undetected'. Lower this value to be stricter, raise if a fast real wrist motion is ever incorrectly flagged. |
</details>

## Testing

`tests/` contains whole-stage regression tests, one file per implemented stage, plus a full end-to-end test. Tests are run against a small (20-frame) committed test clip (`tests/assets/tiny_tennis_clip.mp4`). Each test runs the real stage and checks that its outputs look correct.

```bash
pixi run -e main python -m pytest tests/
```

Stage tests require the real SAM 3.1/GVHMR checkpoints and a CUDA GPU (see [Setup](#setup)). If either are missing, tests are skipped, not failed.

## Licensing

This repo's own code is Apache 2.0, but the checkpoints above carry their own separate license terms (attribution requirements, and a research/personal-use-only restriction on the GVHMR checkpoint specifically). See [NOTICE](NOTICE) before using this project commercially.
