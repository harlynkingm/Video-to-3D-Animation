#!/usr/bin/env bash
# Download every non-registration-gated model checkpoint into checkpoints/, then
# convert the one checkpoint (HaMeR) that upstream ships only as a .ckpt.
#
# What this handles automatically:
#   - SAM 3.1, ViTPose, HMR2, GVHMR  (direct safetensors from HuggingFace)
#   - HaMeR                          (downloads the upstream tarball + converts it)
#   - MediaPipe FaceLandmarker       (checkpoints/face_landmarker.task, Apache-2.0,
#     Google-hosted, no registration)
#   - DECA, MICA                     (Google Drive, see download_gdrive() below;
#     not registration-gated, just not a stable direct-URL host, so this is
#     more fragile than everything else here: if Google changes their large-
#     file download flow, this fails loudly with manual-download instructions
#     rather than silently saving a corrupt file, see download_gdrive())
#   - FLAME's static landmark embedding (body_models/flame/flame_static_embedding.pkl,
#     MIT-licensed metadata from soubhiksanyal/RingNet, not the FLAME model itself)
#   - FLAME model conversion (body_models/flame/generic_model.pkl -> FLAME_NEUTRAL.npz),
#     once you've placed the registration-gated generic_model.pkl yourself, see below
#   - ICT-FaceKit                    (body_models/ict_facekit/FaceXModel/, MIT, no
#     registration, scripts/build_face_bases.py's own offline input)
#   - ARKit basis builds              (body_models/arkit/face_bases.npz, required by
#     stage 9's runtime CSV solve, and body_models/arkit/face_preview_shapes.npz,
#     optional, only for --render-face-preview), built automatically below once
#     ICT-FaceKit and the converted FLAME model are both in place; their outputs
#     are gitignored
#
# What it can NOT handle (registration-gated, must be done by hand, see README):
#   - SMPL-X body model              (body_models/smplx/SMPLX_NEUTRAL.npz)
#   - MANO right hand                (body_models/mano/MANO_RIGHT.pkl)
#   - FLAME 2020 body model          (body_models/flame/generic_model.pkl)
#   - SMPL-X<->FLAME vertex correspondence (body_models/correspondences/SMPL-X__FLAME_vertex_ids.npy)
#
# Depth-Anything-3 is not here either: it auto-downloads on first use.
#
# Usage (from anywhere; requires `pixi install` to have been run first):
#   bash scripts/download_checkpoints.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints"
mkdir -p "$CKPT_DIR"

download() {  # download <url> <dest-filename>
  local url="$1" dest="$CKPT_DIR/$2"
  if [ -f "$dest" ]; then
    echo "[skip] $2 already present"
  else
    echo "[get ] $2"
    curl -L -f --create-dirs -o "$dest" "$url"
  fi
}

# Google Drive large-file download (>~100MB triggers a "can't scan for viruses"
# interstitial page instead of streaming the file directly, this fetches that
# page, pulls the one-time `uuid` confirmation token out of it, and re-requests
# with that token, exactly what a human clicking "Download anyway" would do).
#
# This is real scraping of an undocumented Google UI, not a stable API, it has
# changed before and can change again. If it ever breaks, the failure mode must
# not be "silently save the HTML error page as if it were the checkpoint" (that
# fails confusingly, much later, when something tries to untar/load it), so
# this checks the response actually looks binary before accepting it, and
# deletes + warns instead of leaving a corrupt file in place.
download_gdrive() {  # download_gdrive <gdrive-file-id> <absolute-dest-path> <label>
  local file_id="$1" dest="$2" label="$3"
  if [ -f "$dest" ]; then
    echo "[skip] $label already present"
    return 0
  fi
  echo "[get ] $label (Google Drive)"

  local cookie_jar interstitial uuid
  cookie_jar="$(mktemp)"
  interstitial="$(curl -sL -c "$cookie_jar" "https://drive.google.com/uc?export=download&id=${file_id}")"
  uuid="$(printf '%s' "$interstitial" | grep -oE 'name="uuid" value="[^"]*"' | grep -oE '[0-9a-f-]{36}' || true)"

  if [ -n "$uuid" ]; then
    curl -sL -f -b "$cookie_jar" --create-dirs -o "$dest" \
      "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t&uuid=${uuid}"
  else
    # Small files (or if Google's flow changed and dropped the interstitial)
    # stream directly from the first URL, try that as a fallback.
    curl -sL -f -b "$cookie_jar" --create-dirs -o "$dest" \
      "https://drive.google.com/uc?export=download&id=${file_id}"
  fi
  rm -f "$cookie_jar"

  if [ ! -s "$dest" ] || head -c 20 "$dest" | grep -qi "<!doctype\|<html"; then
    rm -f "$dest"
    echo "[FAIL] $label: Google's download flow may have changed, got a web page, not the file."
    echo "       Download it by hand instead (see README Setup > Manual installation instructions)."
    return 1
  fi
}

HF="https://huggingface.co"
download "$HF/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors" "sam3.1_multiplex_fp16.safetensors"
download "$HF/apozz/motion-capture-safetensors/resolve/main/vitpose.safetensors" "vitpose.safetensors"
download "$HF/apozz/motion-capture-safetensors/resolve/main/hmr2.safetensors" "hmr2.safetensors"
download "$HF/apozz/motion-capture-safetensors/resolve/main/gvhmr.safetensors" "gvhmr.safetensors"

# MediaPipe's own FaceLandmarker model (Apache-2.0, Google-hosted, no registration).
download "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" "face_landmarker.task"

# HaMeR: upstream ships a Lightning .ckpt inside a ~6GB tarball, not safetensors.
if [ -f "$CKPT_DIR/hamer.safetensors" ]; then
  echo "[skip] hamer.safetensors already present"
else
  HAMER_TAR="$CKPT_DIR/hamer_demo_data.tar.gz"
  echo "[get ] hamer_demo_data.tar.gz (~6GB, one-time)"
  curl -L -f -o "$HAMER_TAR" "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
  echo "[conv] hamer.ckpt -> hamer.safetensors"
  ( cd "$REPO_ROOT" && pixi run -e main python scripts/convert_hamer_checkpoint.py "$HAMER_TAR" )
  rm -f "$HAMER_TAR"
fi

# DECA and MICA checkpoints: source repos host these on Google Drive (not
# registration-gated, just not a stable direct-URL host, see download_gdrive()
# above). `|| true` so a failure here (e.g. Google's flow changing) is reported,
# not fatal to the rest of this script
download_gdrive "1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje" "$CKPT_DIR/deca/deca_model.tar" "deca_model.tar" || true
download_gdrive "1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt" "$CKPT_DIR/mica/mica.tar" "mica.tar" || true

# Convert DECA safetensors if needed
DECA_SAFETENSORS_DEST="$CKPT_DIR/deca.safetensors"
if [ -f "$DECA_SAFETENSORS_DEST" ]; then
  echo "[skip] deca.safetensors already present"
else
  echo "[conv] deca_model.tar -> deca.safetensors"
  ( cd "$REPO_ROOT" && pixi run -e main python scripts/convert_deca_checkpoint.py "$CKPT_DIR/deca/deca_model.tar" )
fi

# Convert MICA safetensors if needed
MICA_SAFETENSORS_DEST="$CKPT_DIR/mica.safetensors"
if [ -f "$MICA_SAFETENSORS_DEST" ]; then
  echo "[skip] mica.safetensors already present"
else
  echo "[conv] mica.tar -> mica.safetensors"
  ( cd "$REPO_ROOT" && pixi run -e main python scripts/convert_mica_checkpoint.py "$CKPT_DIR/mica/mica.tar" )
fi

# ICT-FaceKit (MIT, USC-ICT/ICT-FaceKit): the neutral mesh + 53 expression
# blendshapes the offline ARKit basis builder (scripts/build_face_bases.py) needs.
# Only the files that basis actually reads, not the 86 identity*.obj PCA
# basis files, not PupilDilate_L/R or cheekRaiser_L/R (neither maps to an
# ARKit-52 channel), keeping this a ~135MB fetch instead of the full
# ~355MB FaceXModel/ directory.
ICT_FACEKIT_DIR="$REPO_ROOT/body_models/ict_facekit/FaceXModel"
ICT_FACEKIT_BASE="https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master/FaceXModel"
ICT_FACEKIT_FILES="generic_neutral_mesh.obj ICTFaceModelMaterial.mtl vertex_indices.json \
browDown_L.obj browDown_R.obj browInnerUp_L.obj browInnerUp_R.obj browOuterUp_L.obj browOuterUp_R.obj \
cheekPuff_L.obj cheekPuff_R.obj cheekSquint_L.obj cheekSquint_R.obj \
eyeBlink_L.obj eyeBlink_R.obj eyeLookDown_L.obj eyeLookDown_R.obj eyeLookIn_L.obj eyeLookIn_R.obj \
eyeLookOut_L.obj eyeLookOut_R.obj eyeLookUp_L.obj eyeLookUp_R.obj eyeSquint_L.obj eyeSquint_R.obj \
eyeWide_L.obj eyeWide_R.obj jawForward.obj jawLeft.obj jawOpen.obj jawRight.obj mouthClose.obj \
mouthDimple_L.obj mouthDimple_R.obj mouthFrown_L.obj mouthFrown_R.obj mouthFunnel.obj mouthLeft.obj \
mouthLowerDown_L.obj mouthLowerDown_R.obj mouthPress_L.obj mouthPress_R.obj mouthPucker.obj mouthRight.obj \
mouthRollLower.obj mouthRollUpper.obj mouthShrugLower.obj mouthShrugUpper.obj mouthSmile_L.obj \
mouthSmile_R.obj mouthStretch_L.obj mouthStretch_R.obj mouthUpperUp_L.obj mouthUpperUp_R.obj \
noseSneer_L.obj noseSneer_R.obj"
if [ -f "$ICT_FACEKIT_DIR/vertex_indices.json" ]; then
  echo "[skip] ICT-FaceKit assets already present"
else
  echo "[get ] ICT-FaceKit assets (~135MB, one-time)"
  mkdir -p "$ICT_FACEKIT_DIR"
  for f in $ICT_FACEKIT_FILES; do
    curl -sL -f -o "$ICT_FACEKIT_DIR/$f" "$ICT_FACEKIT_BASE/$f"
  done
fi

# flame_static_embedding.pkl: fixed landmark-index/barycentric metadata for FLAME's
# own topology (not model weights), MIT-licensed, freely hosted, unlike the model
# itself this needs no registration. Lives under body_models/, not checkpoints/,
# hence the direct curl call instead of the checkpoints-scoped `download` helper above.
FLAME_LMK_DEST="$REPO_ROOT/body_models/flame/flame_static_embedding.pkl"
if [ -f "$FLAME_LMK_DEST" ]; then
  echo "[skip] flame_static_embedding.pkl already present"
else
  echo "[get ] flame_static_embedding.pkl"
  curl -L -f --create-dirs -o "$FLAME_LMK_DEST" "https://github.com/soubhiksanyal/RingNet/raw/master/flame_model/flame_static_embedding.pkl"
fi

# FLAME model conversion: generic_model.pkl is registration-gated (can't be fetched
# here), but once you've placed it by hand, converting it to FLAME_NEUTRAL.npz needs
# no further manual steps, runs under the separate `flame-convert` pixi environment
# (see pixi.toml/README for why: the official release is a chumpy-pickled file,
# incompatible with this project's normal Python/numpy versions).
if [ -f "$REPO_ROOT/body_models/flame/FLAME_NEUTRAL.npz" ]; then
  echo "[skip] FLAME_NEUTRAL.npz already present"
elif [ -f "$REPO_ROOT/body_models/flame/generic_model.pkl" ]; then
  echo "[conv] generic_model.pkl -> FLAME_NEUTRAL.npz"
  ( cd "$REPO_ROOT" && pixi run -e flame-convert python scripts/convert_flame_model.py )
else
  echo "[skip] FLAME_NEUTRAL.npz (generic_model.pkl not yet downloaded, see README Setup)"
fi

# ARKit basis builds: both need the converted FLAME model (FLAME_NEUTRAL.npz,
# just-conditionally-built above) and the ICT-FaceKit assets (downloaded above).
# face_bases.npz is REQUIRED, stage 9's runtime CSV solve
# (face_blendshapes.py's solve_arkit_weights) raises if it's missing, not an
# optional preview asset like face_preview_shapes.npz below.
if [ -f "$REPO_ROOT/body_models/flame/FLAME_NEUTRAL.npz" ] && [ -f "$ICT_FACEKIT_DIR/vertex_indices.json" ]; then
  if [ -f "$REPO_ROOT/body_models/arkit/face_bases.npz" ]; then
    echo "[skip] face_bases.npz already present"
  else
    echo "[conv] building body_models/arkit/face_bases.npz"
    ( cd "$REPO_ROOT" && pixi run -e main python scripts/build_face_bases.py )
  fi

  if [ -f "$REPO_ROOT/body_models/arkit/face_preview_shapes.npz" ]; then
    echo "[skip] face_preview_shapes.npz already present"
  else
    echo "[conv] building body_models/arkit/face_preview_shapes.npz (optional, --render-face-preview only)"
    ( cd "$REPO_ROOT" && pixi run -e main python scripts/build_arkit_preview_shapes.py )
  fi
else
  echo "[skip] face_bases.npz / face_preview_shapes.npz (need FLAME_NEUTRAL.npz and ICT-FaceKit, see above)"
fi

echo
echo "Google Drive auto-download status (fragile, see comment above download_gdrive()):"
[ -f "$REPO_ROOT/checkpoints/deca/deca_model.tar" ] || [ -f "$REPO_ROOT/checkpoints/deca.safetensors" ] && echo "  [ok]  DECA checkpoint" || echo "  [FAILED, get by hand] DECA checkpoint  (checkpoints/deca/deca_model.tar, see README Setup)"
[ -f "$REPO_ROOT/checkpoints/mica/mica.tar" ] || [ -f "$REPO_ROOT/checkpoints/mica.safetensors" ] && echo "  [ok]  MICA checkpoint" || echo "  [FAILED, get by hand] MICA checkpoint  (checkpoints/mica/mica.tar, see README Setup)"

echo
echo "Done with auto-downloadable checkpoints."
echo "Still required by hand (registration-gated, see README Setup):"
[ -f "$REPO_ROOT/body_models/smplx/SMPLX_NEUTRAL.npz" ] && echo "  [ok]  SMPL-X body model" || echo "  [needs download] SMPL-X body model  (body_models/smplx/SMPLX_NEUTRAL.npz)"
[ -f "$REPO_ROOT/body_models/mano/MANO_RIGHT.pkl" ] && echo "  [ok]  MANO right hand" || echo "  [needs download] MANO right hand     (body_models/mano/MANO_RIGHT.pkl)"
[ -f "$REPO_ROOT/body_models/flame/FLAME_NEUTRAL.npz" ] && echo "  [ok]  FLAME 2020 body model (converted)" || echo "  [needs download] FLAME 2020 body model  (body_models/flame/generic_model.pkl, auto-converted once present)"
[ -f "$REPO_ROOT/body_models/correspondences/SMPL-X__FLAME_vertex_ids.npy" ] && echo "  [ok]  SMPL-X<->FLAME vertex correspondence" || echo "  [needs download] SMPL-X<->FLAME vertex correspondence  (body_models/correspondences/SMPL-X__FLAME_vertex_ids.npy)"
