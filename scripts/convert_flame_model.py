"""Converts the officially-released FLAME 2020 model (a chumpy-pickled .pkl)
into a plain .npz that `smplx`'s own native FLAME loader reads directly
(`smplx.body_models.FLAME`, `ext='npz'` branch, a bare `np.load`, no chumpy).

Run once, under the `flame-convert` pixi environment (see pixi.toml for why
that environment exists and why chumpy can't live anywhere near `main`):

    pixi run -e flame-convert python scripts/convert_flame_model.py \
        body_models/flame/generic_model.pkl body_models/flame/FLAME_NEUTRAL.npz

The output filename matters: `smplx.create(model_path, model_type='flame',
gender='neutral', ext='npz')` looks for exactly `FLAME_NEUTRAL.npz` in the
given directory.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse


def _to_plain(value):
    """Chumpy `Ch` objects expose their underlying array via `.r`; `J_regressor`
    specifically is a scipy sparse matrix (not chumpy) in the original pickle.
    Everything else (numpy arrays, python scalars/strings) is already plain
    and passed through unchanged.
    """
    if scipy.sparse.issparse(value):
        return np.asarray(value.todense())
    return np.asarray(value.r) if hasattr(value, "r") else value


def convert_flame_model(pkl_path: Path, npz_path: Path) -> None:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    converted = {key: _to_plain(value) for key, value in raw.items()}

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **converted)

    print(f"Wrote {npz_path}")
    for key, value in converted.items():
        shape = getattr(value, "shape", None)
        print(f"  {key}: shape={shape}" if shape else f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl_path", type=Path, nargs="?", default=Path("body_models/flame/generic_model.pkl"))
    parser.add_argument("npz_path", type=Path, nargs="?", default=Path("body_models/flame/FLAME_NEUTRAL.npz"))
    args = parser.parse_args()
    convert_flame_model(args.pkl_path, args.npz_path)
