#!/usr/bin/env bash
# Download every non-registration-gated model checkpoint into checkpoints/, then
# convert the one checkpoint (HaMeR) that upstream ships only as a .ckpt.
#
# What this handles automatically:
#   - SAM 3.1, ViTPose, HMR2, GVHMR  (direct safetensors from HuggingFace)
#   - HaMeR                          (downloads the upstream tarball + converts it)
#
# What it can NOT handle (registration-gated, must be done by hand -- see README):
#   - SMPL-X body model  (body_models/smplx/SMPLX_NEUTRAL.npz)
#   - MANO right hand     (body_models/mano/MANO_RIGHT.pkl)
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

HF="https://huggingface.co"
download "$HF/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors" "sam3.1_multiplex_fp16.safetensors"
download "$HF/apozz/motion-capture-safetensors/resolve/main/vitpose.safetensors" "vitpose.safetensors"
download "$HF/apozz/motion-capture-safetensors/resolve/main/hmr2.safetensors" "hmr2.safetensors"
download "$HF/apozz/motion-capture-safetensors/resolve/main/gvhmr.safetensors" "gvhmr.safetensors"

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

echo
echo "Done with auto-downloadable checkpoints."
echo "Still required by hand (registration-gated, see README Setup):"
[ -f "$REPO_ROOT/body_models/smplx/SMPLX_NEUTRAL.npz" ] && echo "  [ok]  SMPL-X body model" || echo "  [needs download] SMPL-X body model  (body_models/smplx/SMPLX_NEUTRAL.npz)"
[ -f "$REPO_ROOT/body_models/mano/MANO_RIGHT.pkl" ] && echo "  [ok]  MANO right hand" || echo "  [needs download] MANO right hand     (body_models/mano/MANO_RIGHT.pkl)"
