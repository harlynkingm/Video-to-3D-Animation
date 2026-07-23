"""Convert the upstream HaMeR checkpoint into this project's checkpoints/ folder.

HaMeR is the one model here whose weights aren't already published as
safetensors: upstream ships a PyTorch-Lightning `.ckpt` inside a ~6GB tarball
(see the README's Setup section for the download link). This script extracts
just the two files this pipeline needs and writes them into checkpoints/:

  - hamer.safetensors      (backbone + MANO head + MANO layer buffers; the
                            training-only discriminator is dropped)
  - mano_mean_params.npz   (the MANO head's initial pose/shape/camera means)

Run inside the `main` pixi environment:

    pixi run -e main python scripts/convert_hamer_checkpoint.py path/to/hamer_demo_data.tar.gz

The download-everything setup script (scripts/download_checkpoints.sh) calls
this for you; run it by hand only if you're doing manual setup.
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# Members inside hamer_demo_data.tar.gz that we need.
CKPT_MEMBER = "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
MANO_MEAN_MEMBER = "_DATA/data/mano_mean_params.npz"

# Inference modules to keep from the Lightning checkpoint's state_dict. The
# adversarial `discriminator.*` and the actnorm `initialized` flag are
# training-only and dropped.
KEEP_PREFIXES = ("backbone.", "mano_head.", "mano.")

OUTPUT_SAFETENSORS = "hamer.safetensors"
OUTPUT_MANO_MEAN = "mano_mean_params.npz"


def convert(tar_path: Path) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print(f"Extracting {CKPT_MEMBER} and {MANO_MEAN_MEMBER} from {tar_path.name} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extract(CKPT_MEMBER, tmp_dir)
            tar.extract(MANO_MEAN_MEMBER, tmp_dir)

        ckpt = torch.load(tmp_dir / CKPT_MEMBER, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        kept = {
            k: v.contiguous().clone()
            for k, v in state_dict.items()
            if k.startswith(KEEP_PREFIXES) and isinstance(v, torch.Tensor)
        }
        print(f"Keeping {len(kept)} of {len(state_dict)} tensors (dropped training-only weights).")

        out_safetensors = CHECKPOINTS_DIR / OUTPUT_SAFETENSORS
        save_file(kept, str(out_safetensors))
        print(f"Wrote {out_safetensors} ({out_safetensors.stat().st_size / 1e9:.2f} GB)")

        mano_mean = np.load(tmp_dir / MANO_MEAN_MEMBER)
        out_mano_mean = CHECKPOINTS_DIR / OUTPUT_MANO_MEAN
        np.savez(out_mano_mean, **{k: mano_mean[k] for k in mano_mean.files})
        print(f"Wrote {out_mano_mean}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the HaMeR checkpoint tarball to safetensors")
    parser.add_argument("tar_path", type=Path, help="Path to the downloaded hamer_demo_data.tar.gz")
    args = parser.parse_args()
    if not args.tar_path.exists():
        raise SystemExit(f"Tarball not found: {args.tar_path}")
    convert(args.tar_path)


if __name__ == "__main__":
    main()
