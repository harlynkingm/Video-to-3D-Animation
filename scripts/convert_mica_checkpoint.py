"""Convert the upstream MICA checkpoint into this project's checkpoints/ folder.

Like DECA's, `mica.tar` is a raw `torch.save()` pickle despite the `.tar`
extension, `torch.load` reads it directly. Its top-level dict holds
`arcface` (the ArcFace identity backbone) and `flameModel` (a `regressor`
mapping ArcFace embeddings to FLAME shape, plus a `generator` sub-dict that's
just MICA's own bundled copy of FLAME's generic-model data, this project
already has that from `scripts/convert_flame_model.py`, so it's dropped
here), plus training-only optimizer/scheduler bookkeeping (also dropped).

Run inside the `main` pixi environment:

    pixi run -e main python scripts/convert_mica_checkpoint.py checkpoints/mica/mica.tar

The download-everything setup script (scripts/download_checkpoints.sh) runs this automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

OUTPUT_SAFETENSORS = "mica.safetensors"


def convert(tar_path: Path) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {tar_path.name} ...")
    ckpt = torch.load(tar_path, map_location="cpu", weights_only=False)

    kept = {f"arcface.{k}": v.contiguous().clone() for k, v in ckpt["arcface"].items() if isinstance(v, torch.Tensor)}
    regressor = {
        k: v for k, v in ckpt["flameModel"].items() if k.startswith("regressor.") and isinstance(v, torch.Tensor)
    }
    kept.update({k: v.contiguous().clone() for k, v in regressor.items()})
    print(f"Keeping {len(kept)} tensors (arcface backbone + shape regressor; "
          f"dropped {len(ckpt['flameModel']) - len(regressor)} bundled FLAME generator tensors).")

    out_path = CHECKPOINTS_DIR / OUTPUT_SAFETENSORS
    save_file(kept, str(out_path))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the MICA checkpoint to safetensors")
    parser.add_argument("tar_path", type=Path, nargs="?", default=CHECKPOINTS_DIR / "mica" / "mica.tar")
    args = parser.parse_args()
    if not args.tar_path.exists():
        raise SystemExit(f"Checkpoint not found: {args.tar_path}")
    convert(args.tar_path)


if __name__ == "__main__":
    main()
