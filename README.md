# Video to 3D Motion Capture Animation
Converts a still video of a person and optional object into a Blender motion capture animation and facial performance capture CSV.

This project combines SAM 3.1 tracking, ViTPose/HMR2/GVHMR body capture, Depth-Anything-3 metric depth, HaMeR/MANO hand capture, Open4DHOI-derived interaction processing, and DECA/MICA/FLAME/MediaPipe FaceLandmarker face capture.

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

Installing pixi sets up three environments for this project, each pinned to its own Python version:

- `main` (Python 3.13) handles most pipeline stages (SAM 3.1, GVHMR, face capture, etc.), including a CUDA 12.8 build of PyTorch.
- `export` (Python 3.13) is kept separate because it depends on `bpy` (Blender's Python API), which requires its own exact Python version independent of the rest of the stack.
- `flame-convert` (Python 3.10) is a one-time-use environment for `scripts/convert_flame_model.py` and is never imported by pipeline code. It exists because the officially-released FLAME model is a `chumpy`-pickled `.pkl`, and `chumpy` needs numpy <1.24 and Python <=3.10
</details>

## Setup

### 1. Download 3D body models

**These steps must be done by hand.** SMPL-X, MANO, and FLAME are projects that sit behind free registration and license acceptance on their respective sites, and cannot be auto-downloaded. If you skip this section, stages that require a body, hand, or face model will fail.

1. **SMPL-X**: register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) (free), download the model files, and place `SMPLX_NEUTRAL.npz` in `body_models/smplx/SMPLX_NEUTRAL.npz`
2. **MANO** (hand model, required by stage 4): register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (free), download the models zip file, and place `MANO_RIGHT.pkl` in `body_models/mano/MANO_RIGHT.pkl`
3. **FLAME 2020** (face model, required by the face-capture stage): register at [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de) (get the **FLAME 2020** download specifically, not the newer FLAME 2023 "Open Model"), and place `generic_model.pkl` in `body_models/flame/generic_model.pkl`
4. **SMPL-X↔FLAME vertex correspondence** (also required by the face-capture stage): from the same [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) Downloads page, download the "Correspondences" zip and place `SMPL-X__FLAME_vertex_ids.npy` at `body_models/correspondences/SMPL-X__FLAME_vertex_ids.npy`
5. **Blender addon**: install [`jtesch/smplx_blender_addon`](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon) from GitLab (Blender 4.2+)

### 2. Download model checkpoints

After `pixi install` and downloading the body models, the quickest way to get every checkpoint is:

```bash
bash scripts/download_checkpoints.sh
```

This downloads SAM 3.1, ViTPose, HMR2, GVHMR, and MediaPipe's FaceLandmarker from HuggingFace/Google, converts the HaMeR checkpoint to a safetensors file, fetches DECA and MICA from Google Drive, converts DECA and MICA to safetensors, fetches FLAME's static landmark embedding, fetches ICT-FaceKit's neutral mesh and expression shapes, and converts the FLAME model. Once both FLAME and ICT-FaceKit are in place, it also builds `body_models/arkit/face_bases.npz` (required by the face-capture stage) and `body_models/arkit/face_preview_shapes.npz`. Everything is downloaded into `checkpoints/` and `body_models/`. It skips files you already have and reminds you about the registration-gated files it can't fetch, if you don't have them downloaded.

The pipeline runs fully offline after its first complete setup and first Stage 3 run. Depth-Anything-3 downloads its ~1.3GB checkpoint the first time Stage 3 is run.

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
| `checkpoints/deca/deca_model.tar` | [DECA's official repo](https://github.com/yfeng95/DECA) (Google Drive) | ~414MB |
| `checkpoints/mica/mica.tar` | [MICA's official repo](https://github.com/Zielon/MICA) (Google Drive) | ~480MB |
| `checkpoints/face_landmarker.task` | [storage.googleapis.com/mediapipe-models](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | ~4MB |
| `body_models/flame/flame_static_embedding.pkl` | [github.com/soubhiksanyal/RingNet](https://github.com/soubhiksanyal/RingNet/raw/master/flame_model/flame_static_embedding.pkl) | ~4KB |
| `body_models/ict_facekit/FaceXModel/*` | [github.com/USC-ICT/ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit) | ~135MB |

The ~6GB `hamer_demo_data.tar.gz` download is temporary and can be deleted after checkpoint conversion. Convert it to the smaller ~2.6GB checkpoint using the following script:

```bash
pixi run -e main python scripts/convert_hamer_checkpoint.py path/to/hamer_demo_data.tar.gz
```

Similarly, once `body_models/flame/generic_model.pkl` is in place (step 3 above), convert it to the `.npz` file the pipeline actually reads using this script:

```bash
pixi run -e flame-convert python scripts/convert_flame_model.py
```

This writes `body_models/flame/FLAME_NEUTRAL.npz`, using a separate `flame-convert` pixi environment (Python 3.10) rather than `main` for reasons listed in "Pixi Environment Details" above.

DECA and MICA, if manually downloaded, need to be converted to `.safetensors` files located in the `/checkpoints` directory.

```bash
pixi run -e main python scripts/convert_deca_checkpoint.py checkpoints/deca/deca_model.tar
pixi run -e main python scripts/convert_mica_checkpoint.py checkpoints/mica/mica.tar
```

Once `body_models/ict_facekit/FaceXModel/` and the converted FLAME model (`body_models/flame/FLAME_NEUTRAL.npz`) are both in place, build the face-capture stage's ARKit basis:

```bash
pixi run -e main python scripts/build_face_bases.py
```

This writes `body_models/arkit/face_bases.npz`, which the face-capture stage's ARKit CSV export requires. Optionally, also build the full-mesh preview basis (only needed for `--render-face-preview`):

```bash
pixi run -e main python scripts/build_arkit_preview_shapes.py
```

This writes `body_models/arkit/face_preview_shapes.npz`.

There is an additional ~1.3GB Depth-Anything-3 checkpoint which is automatically downloaded when [stage 3]((#stage-3-estimate-depth)) runs for the first time.

The tasks above are handled automatically by running `bash scripts/download_checkpoints.sh` except for the registration-gated body models and the Depth-Anything-3 checkpoint, which is downloaded when Stage 3 runs for the first time. 
</details>

## Quick Start

### UI Application

Run `pixi run ui` to launch the desktop app (Windows-only). Select a source video or image sequence, an output folder, a human prompt, an optional object prompt, and the camera's focal length/sensor width, then click **Run**.

The advanced controls can select a stage range, choose an object proxy shape, force a rerun, and render previews for each stage. You can also add multiple runs to a queue for sequential processing.

If you select an output folder with an existing `progress.json` file, you can continue the run, or force a rerun, without filling out any other fields.

> [!TIP]
> Select a still video, captured on a tripod, for the best results.

### Terminal / CLI

To create a run and process a video from the command line:

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
| `--input-video` | **Yes** | | Path to the source video file (MP4, MOV, MPEG, FLV, or WMV), or a directory of already-extracted JPEG/PNG frames sorted by filename |
| `--source-fps` | No, unless `--input-video` is a directory | | Frame rate for an image-sequence `--input-video` directory |
| `--output-dir` (`-o`) | **Yes** | | Directory to create for this run's state and outputs. |
| `--human-prompt` | **Yes** | | Text description of the person to track, e.g. `"a tennis player"`. |
| `--focal-length-mm` | **Yes** (unless using `--intrinsics-k`) | | Camera focal length in mm, used to build the intrinsics matrix stage 0 requires. |
| `--sensor-width-mm` | **Yes** (unless using `--intrinsics-k`) | | Camera sensor width in mm, used alongside focal length to build the intrinsics matrix. |
| `--intrinsics-k` | Alternative to the 2 above | | Raw 3x3 camera intrinsics matrix as JSON, e.g. `'[[fx,0,cx],[0,fy,cy],[0,0,1]]'` for real calibration data. More accurate than the lens-spec path even when both are available, since that path has to assume a perfectly centered principal point and this doesn't. If you provide this, focal length and sensor width cannot also be used. |
| `--run-id` | No | `--output-dir`'s own folder name | A human-readable label for the run. Doesn't affect anything on disk. |
| `--object-prompt` | No | none | Text description of the object to track, e.g. `"a teddy bear"`. Omit if there's no object to track. |
| `--object-shape-hint` | No | `auto` | Forces the tracked object's proxy shape to `box`, `ellipsoid`, or `cylinder` instead of letting stage 6 auto-fit whichever shape better matches the object. |
| `--finger-motion` | No | `smooth` | Finger-motion profile: `smooth` prioritizes stable hands; `detailed` retains more rapid, subtle finger articulation. |
| `--anchor-frame-override` | No | auto-selected | Forces a specific frame index as the "anchor" frame instead of letting stage 1 pick the frame with the clearest view of the object. |
| `--start-on-stage` | No | runs every implemented stage | Starts the run on the given stage number, inclusive, e.g. `4` starts the run on stage 4. |
| `--stop-after-stage` | No | runs every implemented stage | Stops after the given stage number, inclusive, e.g. `5` runs stages 0-5 and stops before stage 6. Can be combined with `--start-on-stage` to only run a range, and can be combined with `--force-all` to force-rerun a range. |
| `--skip-face-capture` | No | off (face capture runs by default) | Disables stage 9 (face capture) entirely, no `face_params.npz`/`face_motion.npz`. See [stage 9](#stage-9-capture-face) below. |
| `--render-previews` | No | off | Enables every `--render-*-preview` flag below at once. |
| `--render-mask-previews` | No | off | Stage 1 also writes black/white JPEG mask previews for visual spot-checking. See [stage 1](#stage-1-mask-and-track) below. |
| `--render-motion-preview` | No | off | Stage 2 also writes an AMASS `.npz` importable into Blender for visual spot-checking. See [stage 2](#stage-2-estimate-human-motion) below. |
| `--render-depth-preview` | No | off | Stage 3 also writes a colored `.ply` point cloud importable into Blender for visual spot-checking. See [stage 3](#stage-3-estimate-depth) below. |
| `--render-hands-preview` | No | off | Stage 4 also writes a `.bvh` hand-skeleton animation (both hands, bones only) importable into Blender for visual spot-checking. See [stage 4](#stage-4-estimate-hands) below. |
| `--render-retarget-preview` | No | off | Stage 5 also writes a `.bvh` full-body-plus-hands skeleton animation importable into Blender for confirming the hands sit correctly on the body. See [stage 5](#stage-5-retarget-hands) below. |
| `--render-scene-preview` | No | off | Stage 6 also writes a `.ply` combining the human, object, and depth scene in one aligned space for confirming the scale fit in Blender. See [stage 6](#stage-6-align-scene-scale) below. |
| `--render-contacts-preview` | No | off | Stage 7 also writes one annotated JPEG per contact event, circling the contact point on the source frame, for visual spot-checking. See [stage 7](#stage-7-annotate-contacts) below. |
| `--render-face-preview` | No | off | Stage 9 writes preview template/PC2 data, and stage 10 assembles `FLAME_face_preview.blend`, `landmark_preview.blend`, and `ARKit_face_preview.blend` for visual spot-checking. See [stage 9](#stage-9-capture-face) below. |
| `--force-all` (`-f`) | No | off | Forces all stages to re-run, even if they have already run. The equivalent of passing `--force` to each stage individually. |
</details>

**Running stages individually**: `pipeline.run` is just `pipeline.create_run` followed by every stage's own script, run in sequence. See [Pipeline](#pipeline) below for each stage's individual command and options.

## Pipeline

The pipeline is a sequence of stages, each a separate script. This section documents each one individually: what the stage does, how to run just that stage on its own, and any optional outputs it can produce.

<details>
<summary>All stage input/output details</summary>

| Stage | Script | Input | Output |
|---|---|---|---|
| 0. Ingest video | `stage_0_ingest_video` | source video file, or a directory of JPEG/PNG frames | `input_frames/*.jpg` <br> camera intrinsics in `progress.json` |
| 1. Mask and track | `stage_1_mask_and_track` | `input_frames/*.jpg` | `stage1_masks/human.pt` <br> `stage1_masks/object.pt` <br> anchor frame index in `progress.json` <br> `stage1_masks/preview_human/*.jpg` (optional) <br> `stage1_masks/preview_object/*.jpg` (optional) |
| 2. Estimate human motion | `stage_2_estimate_human_motion` | `input_frames/*.jpg` <br> `stage1_masks/human.pt` | `stage2_motion/human_motion.pt` <br> `stage2_motion/blender_preview.npz` (optional) |
| 3. Estimate depth | `stage_3_estimate_depth` | `input_frames/*.jpg` <br> anchor frame index in `progress.json` | `stage3_depth/anchor_depth.npy` <br> `stage3_depth/anchor_pointcloud.ply` (optional) |
| 4. Estimate hands | `stage_4_estimate_hands` | `input_frames/*.jpg` <br> `stage1_masks/human.pt` | `stage4_hands/hand_pose.npz` <br> `stage4_hands/hands_preview.bvh` (optional) |
| 5. Retarget hands | `stage_5_retarget_hands` | `stage2_motion/human_motion.pt` <br> `stage4_hands/hand_pose.npz` | `stage5_retarget/retargeted_motion.pt` <br> `stage5_retarget/retargeted_motion.npz` <br> `stage5_retarget/retarget_preview.bvh` (optional) |
| 6. Align scene scale | `stage_6_align_scene_scale` | `stage3_depth/anchor_depth.npy` <br> `stage2_motion/human_motion.pt` <br> `stage1_masks/human.pt` <br> `stage1_masks/object.pt` (optional) | `stage6_scale/scene_scale.json` <br> `stage6_scale/object_shape.json` (if an object was tracked) <br> `stage6_scale/scene_preview.ply` (optional) |
| 7. Annotate contacts | `stage_7_annotate_contacts` | `stage5_retarget/retargeted_motion.pt` <br> `stage1_masks/object.pt` | `stage7_contacts/contact_events.json` <br> `stage7_contacts/contacts_preview/*.jpg` (optional) |
| 8. Optimize human-object interaction | `stage_8_optimize_hoi` | `stage7_contacts/contact_events.json` <br> `stage6_scale/object_shape.json` <br> `stage6_scale/scene_scale.json` <br> `stage5_retarget/retargeted_motion.pt` | `stage8_interaction/object_pose.pt` <br> `stage8_interaction/object_pose.npz` <br> `stage8_interaction/attachment_events.json` |
| 9. Face capture | `stage_9_capture_face` | `input_frames/*.jpg` <br> `stage1_masks/human.pt` | `stage9_face/face_params.npz` (raw DECA/MICA/MediaPipe output) <br> `stage9_face/face_motion.npz` (fitted FLAME parameters) <br> `output_face.csv` (Live Link Face CSV: 52 ARKit weights plus 9 head/eye Euler columns) <br> preview template/PC2 data (optional) |
| 10. Export animation | `stage_10_export` | `stage5_retarget/retargeted_motion.npz` <br> `stage9_face/face_motion.npz` (unless face capture was skipped) <br> `stage6_scale/object_shape.json` (if an object was tracked) <br> `stage8_interaction/object_pose.npz` (if an object was tracked) | `output.blend` <br> face preview `.blend` files (optional) |

</details>

**Every stage skips itself if `progress.json` already shows it as complete.** Pass `--force` or `-f` to re-run a stage anyway.

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

Extracts every frame to disk as JPEG (to `runs/my_clip/input_frames/`), and resolves the camera intrinsics matrix computed from `--focal-length-mm`/`--sensor-width-mm` and the video's resolution (or used directly from `--intrinsics-k` if provided).

If `--input-video` is a directory of images instead of a video file, each is re-encoded as JPEG in filename order (if needed) and `--source-fps` provides the frame rate.

### Stage 1. Mask and track

```bash
pixi run -e main python -m pipeline.stages.stage_1_mask_and_track -o runs/my_clip
```

SAM 3.1 tracks the human (and object, if `--object-prompt` was given). Also resolves which frame to use as an object "anchor" frame, which has the clearest object view. Uses `--anchor-frame-override` if you specify one when creating the run.

<details>
<summary><strong>Optional: JPEG Mask Output</strong></summary>

Use `--render-mask-previews` when creating the run to also have this stage write `runs/my_clip/stage1_masks/preview_human/000000.jpg`, `000001.jpg`, ... (and `preview_object/` if an object was tracked). These are plain black-and-white mask images at the video's native resolution. You can scroll through these images on disk to confirm SAM 3.1 tracked the right thing.
</details>

### Stage 2. Estimate human motion

```bash
pixi run -e main python -m pipeline.stages.stage_2_estimate_human_motion -o runs/my_clip
```

GVHMR turns the human mask into a 3D SMPL-X body pose animation. Works at any source video resolution, however larger frames mean more disk space and slightly slower per-frame I/O.

Stage 2 also performs a foot/wrist drift-lock pass on wrists, ankles, and feet to prevent sliding. Then, the body motion is temporally smoothed to remove residual per-frame jitter.

The default pipeline smoothing profile is used this stage is run. See [Motion smoothing](#motion-smoothing) for optional smoothing overrides.

<details>
<summary><strong>Optional: 3D Motion Preview Output</strong></summary>

Use `--render-motion-preview` when creating the run to also have this stage write `runs/my_clip/stage2_motion/blender_preview.npz` This NPZ is importable in Blender via the SMPL-X addon's own **Add Animation** operator (`Object > SMPL-X > Add Animation`) if the addon is installed (see [Setup](#setup)). **For accurate preview,** when the import dialog appears, **set "Format" to `SMPL-X`, not `AMASS`** to view the 3D animation at the correct orientation.
</details>

### Stage 3. Estimate depth

```bash
pixi run -e main python -m pipeline.stages.stage_3_estimate_depth -o runs/my_clip
```

Depth-Anything-3 (`DA3METRIC-LARGE`) runs once on a single anchor frame, not the whole clip. This produces a depth map in real-world meters.

<details>
<summary><strong>Optional: PLY Point Cloud Output</strong></summary>

Use `--render-depth-preview` when creating the run to also have this stage write `runs/my_clip/stage3_depth/anchor_pointcloud.ply`, a colored point cloud estimating the depth in the image. Blender can import this `.ply` file natively via **File > Import > Stanford (.ply)**
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

HaMeR estimates per-frame MANO hand pose for both hands. It finds the person from the stage 1 mask, runs ViTPose to locate each hand, crops in, and predicts finger articulation plus wrist orientation. The output `stage4_hands/hand_pose.npz` holds per-frame left/right hand pose, wrist orientation, and validity per hand. Attaching it to the body happens in stage 5.

If a hand is off-screen, too occluded, or if the ViTPose based confidence score is too low, or if the estimated wrist pose is anatomically impossible, the wrist is held in place instead of assuming incorrect motion.

This stage also temporally smooths both hand movements after making corrections based on the above. The smoothing happens in 3 passes: first a zero-phase smoothing to reduce jitter, then an adaptive filter, then a keyframe-based reduction. Reruns use the current pipeline profile; see [Motion smoothing](#motion-smoothing) to intentionally pin a custom value.

This stage requires the MANO body model and the SMPL-X model file (see [Setup](#setup)).

<details>
<summary><strong>Optional: Hand Skeleton Preview</strong></summary>

Use `--render-hands-preview` when creating the run to also have this stage write `runs/my_clip/stage4_hands/hands_preview.bvh`, a bone-only animation of both hands. This .bvh is importable in Blender via **File > Import > Motion Capture (.bvh)**. Each hand is shown in isolation, side by side, so you can confirm the finger articulation looks right before it's attached to a body. This preview requires `SMPLX_NEUTRAL.npz` (see [Setup](#setup)).
</details>

### Stage 5. Retarget hands

```bash
pixi run -e main python -m pipeline.stages.stage_5_retarget_hands -o runs/my_clip
```

Attaches the stage 4 hands onto the stage 2 body, producing one merged full-body-plus-hands SMPL-X sequence in `runs/my_clip/stage5_retarget/retargeted_motion.pt` and a NumPy interchange copy at `retargeted_motion.npz`. A hand never detected anywhere in the clip keeps GVHMR's own wrist and flat fingers throughout. Any hand detected at least once gets every frame filled with that detected pose.

This stage requires `SMPLX_NEUTRAL.npz` (see [Setup](#setup)).

<details>
<summary><strong>Optional: Full Body and Hands Preview</strong></summary>

Use `--render-retarget-preview` when creating the run to also have this stage write `runs/my_clip/stage5_retarget/retarget_preview.bvh`, a bone-only animation of the whole body with the stage 4 hands attached, including the body's real motion (walking, sitting down, etc.). This .bvh is importable in Blender via **File > Import > Motion Capture (.bvh)**

This preview requires `SMPLX_NEUTRAL.npz` (see [Setup](#setup)).
</details>

### Stage 6. Align scene scale

```bash
pixi run -e main python -m pipeline.stages.stage_6_align_scene_scale -o runs/my_clip
```

The depth map ([stage 3](#stage-3-estimate-depth)) and SMPL-X human body ([stage 2](#stage-2-estimate-human-motion)) are both represented in real-world meters, but disagree on scale. This stage reconciles them at the anchor frame by matching the SMPL-X body pose against depth values within the SAM-3 human mask. The result is written to `runs/my_clip/stage6_scale/scene_scale.json`

If an object was tracked, this stage also fits a shape to it (a box, ellipsoid, or cylinder chosen via `--object-shape-hint`, or 'auto' which automatically chooses the shape of best fit), written to `runs/my_clip/stage6_scale/object_shape.json`.

<details>
<summary><strong>Optional: Aligned Scene Preview</strong></summary>

Use `--render-scene-preview` when creating the run to also have this stage write `runs/my_clip/stage6_scale/scene_preview.ply`, a single point cloud that puts every element in the human's metric space, color-coded so you can confirm the fit: green for the SMPL-X body, red for the tracked object's depth points, and (if an object was tracked) a yellow wireframe of its fitted primitive shape. Import in Blender and enable vertex colors the same way as [stage 3](#stage-3-estimate-depth).
</details>

### Stage 7. Detect human-object interaction points

```bash
pixi run -e main python -m pipeline.stages.stage_7_annotate_contacts -o runs/my_clip
```

Detects contact points between the body and object across 9 body regions. The mask + depth inferred by Depth-Anything-3 are used to determine where the body and object interact. Interaction events written to `runs/my_clip/stage7_contacts/contact_events.json`

<details>
<summary><strong>Optional: Contact Points Preview</strong></summary>

Use `--render-contacts-preview` when creating the run to also have this stage write `runs/my_clip/stage7_contacts/contacts_preview/` This saves one JPEG per contact event named `{peak_confidence_frame:06d}_{regions-joined-by-plus}.jpg`. Each image is the source frame at that event's most-confident moment, with a circle drawn around the joint that triggered it.
</details>

### Stage 8. Optimize human-object interaction

```bash
pixi run -e main python -m pipeline.stages.stage_8_optimize_hoi -o runs/my_clip
```

If an object was tracked, attach it to a body region if a hold was detected for a duration of time. Writes the object positional animation to `runs/my_clip/stage8_interaction/object_pose.npz` and the qualifying hold intervals to `runs/my_clip/stage8_interaction/attachment_events.json`.

### Stage 9. Face capture

```bash
pixi run -e main python -m pipeline.stages.stage_9_capture_face -o runs/my_clip
```

Runs DECA, MICA, and MediaPipe's FaceLandmarker on every frame, then fits FLAME against MediaPipe's detected 2D landmarks (using DECA/MICA as the initial guess). Writes the raw per-model output to `runs/my_clip/stage9_face/face_params.npz`, the fitted FLAME parameters (identity, expression, jaw, head rotation/translation) to `runs/my_clip/stage9_face/face_motion.npz`, and `runs/my_clip/output_face.csv` in Live Link Face format: 52 ARKit blendshape weights plus 9 head/eye Euler columns per frame.

Capturing a face is optional but runs by default. Use `--skip-face-capture` when creating the run to disable face capture.

This stage requires `body_models/arkit/face_bases.npz` (see [Setup](#setup))

<details>
<summary><strong>Optional: Face-Only Preview</strong></summary>

Pass `--render-face-preview` when creating the run. Stage 9 writes the preview template and animation data; [stage 10](#stage-10-export) assembles three files from it: `stage9_face/FLAME_face_preview.blend` (the fitted FLAME mesh), `stage9_face/landmark_preview.blend` (raw and smoothed MediaPipe landmarks), and `stage9_face/ARKit_face_preview.blend` (the same ARKit channels written to `output_face.csv`).

</details>

<details>
<summary><strong>Optional: Compare against iOS Live Link Face capture</strong></summary>

If a run folder also contains paired Live Link Face ground-truth files (`*_raw.csv`, `*_neutral.csv`, and `frame_log.csv`), generate an HTML comparison report with:

```bash
pixi run compare-ground-truth runs/my_clip
```

The report defaults to `runs/my_clip/stage9_face/ground_truth_report.html`. When `--render-face-preview` is enabled, stage 9 also generates it automatically when it finds a `*_raw.csv` file in the run folder.

</details>

### Stage 10. Export

```bash
pixi run -e export python -m pipeline.stages.stage_10_export -o runs/my_clip
```

Combines the body+hands motion, optional tracked-object animation, and the face-capture result into a single `runs/my_clip/output.blend` file. The face in this Blender preview is an approximation. Use `output_face.csv` to drive a downstream character face rig.

**Note:** This runs in a separate, `export` pixi environment (the main environment still executes it as a subprocess).

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

Finger articulation uses the run-level `--finger-motion` choice to have `smooth` or `detailed` finger motion.

<details>
<summary>Smoothness tuning</summary>

To intentionally override a smoothing default for a run, add it as a key/value pair in the top-level `fine_tuning_overrides` object in `progress.json`. For example:

```json
{
  "fine_tuning_overrides": {
    "hand_finger_min_cutoff_hz": 0.18
  }
}
```

Here are the possible smoothing overrides:

| Field | Default | Effect |
|---|---|---|
| `body_smoothing_window` | `9` | Savitzky-Golay window (odd, in frames) for body rotation. Larger is smoother but can smear fast motion. |
| `body_translation_cutoff` | `0.15` | Butterworth low-pass cutoff (fraction of Nyquist) for the body root position. Lower is smoother but adds lag. |
| `hand_smoothing_window` | `15` | Savitzky-Golay pre-pass window for the wrist. It keeps broad wrist motion stable without affecting finger detail. |
| `hand_beta` | `0.3` | Wrist adaptive-filter responsiveness. |
| `hand_finger_smoothing_window` | `15` / `5` (smooth / detailed) | Savitzky-Golay pre-pass for fingers. The smooth profile uses broad cleanup; detailed uses a short pass to retain small articulations. |
| `hand_finger_beta` | `0.3` / `1.85` (smooth / detailed) | Finger adaptive-filter responsiveness. Higher follows quick, small bends more closely, at the cost of allowing more movement-time jitter. |
| `hand_finger_derivative_cutoff_hz` | `1.0` / `2.75` (smooth / detailed) | How quickly the finger filter recognizes a change in speed. Higher retains short keypresses; lower is steadier but slower to react. |
| `hand_finger_min_cutoff_hz` | `0.15` / `0.225` (smooth / detailed) | Adaptive-filter smoothing strength at rest for the fingers. Lower is steadier when still but slower to respond. |
| `hand_wrist_min_cutoff_hz` | `0.10` | Same, for the wrist — lower than the fingers, since the wrist starts from noisier data and needs more smoothing. |
| `hand_finger_decimate_deg` | `1.5` / `0.375` (smooth / detailed) | Keyframe-reduction tolerance for the fingers, in degrees: the most the refitted curve may deviate from the filtered motion. Keep this low for detail; larger means fewer keyframes and a flatter curve. |
| `hand_wrist_decimate_deg` | `3.0` | Same, for the wrist (looser than the fingers). |
| `hand_max_wrist_deviation_deg` | `110.0` | Max plausible wrist rotation degrees relative to the forearm, checked against stage 2's body motion before smoothing. A raw estimate past this is treated as 'undetected'. Lower this value to be stricter, raise if a fast real wrist motion is ever incorrectly flagged. |
</details>

## Testing

`tests/` contains whole-stage regression tests, one file per implemented stage, plus a full end-to-end test. Tests are run against a small (20-frame) committed test clip (`tests/assets/tiny_tennis_clip.mp4`). Each test runs the real stage and checks that its outputs look correct.

```bash
pixi run test
```

Stage 10's tests (`tests/test_stage_10_export.py`) need the `bpy` module and only run under the `export` environment. To run either environment's tests:

```bash
pixi run -e main python -m pytest tests/
pixi run -e export python scripts/run_export_tests.py
pixi run -e main python -m pytest ui/tests
```

Stage tests require the real SAM 3.1/GVHMR checkpoints and a CUDA GPU (see [Setup](#setup)). If either are missing, tests are skipped, not failed.

## Licensing

This repo's own code is Apache 2.0, but the checkpoints and body models carry their own separate license terms. In particular, GVHMR and the face-capture stack (DECA, MICA, and FLAME 2020) are restricted to research, education, artistic, or personal/non-commercial use unless you obtain separate rights. See [NOTICE](NOTICE) before using this project commercially.
