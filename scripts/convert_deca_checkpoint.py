"""Convert the upstream DECA checkpoint into this project's checkpoints/ folder.

Despite the `.tar` extension (a naming convention inherited from old PyTorch
training scripts), `deca_model.tar` is a raw `torch.save()` pickle, not an
actual tar archive -- confirmed by its magic bytes (`\\x80\\x02`, a pickle
protocol header) and by `torch.load` reading it directly with no extraction
step. Its top-level dict holds `E_flame` (the coarse encoder this project
uses), `E_detail`/`D_detail` (DECA's detail-displacement branch, unused here),
and training-only optimizer/epoch bookkeeping (dropped).

Run inside the `main` pixi environment:

    pixi run -e main python scripts/convert_deca_checkpoint.py checkpoints/deca/deca_model.tar

The download-everything setup script (scripts/download_checkpoints.sh) runs this automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# Only the coarse encoder's feature extractor + regression head; E_detail/
# D_detail (DECA's photometric detail branch) are never used by this project.
KEEP_PREFIXES = ("encoder.", "layers.")

OUTPUT_SAFETENSORS = "deca.safetensors"


def convert(tar_path: Path) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {tar_path.name} ...")
    ckpt = torch.load(tar_path, map_location="cpu", weights_only=False)
    print(f"Ignoring E_detail/D_detail (DECA's photometric detail branches, unused here); keeping E_flame only.")
    state_dict = ckpt["E_flame"]
    kept = {
        k: v.contiguous().clone()
        for k, v in state_dict.items()
        if k.startswith(KEEP_PREFIXES) and isinstance(v, torch.Tensor)
    }
    print(f"Keeping {len(kept)} of {len(state_dict)} tensors from E_flame.")

    out_path = CHECKPOINTS_DIR / OUTPUT_SAFETENSORS
    save_file(kept, str(out_path))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the DECA checkpoint to safetensors")
    parser.add_argument("tar_path", type=Path, nargs="?", default=CHECKPOINTS_DIR / "deca" / "deca_model.tar")
    args = parser.parse_args()
    if not args.tar_path.exists():
        raise SystemExit(f"Checkpoint not found: {args.tar_path}")
    convert(args.tar_path)


if __name__ == "__main__":
    main()
