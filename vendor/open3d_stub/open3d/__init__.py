"""Stub package satisfying `depth-anything-3`'s declared (but unused) `open3d`
dependency, which has no PyPI wheel at all for Python 3.13 and would otherwise
block installing that package on this project's pinned Python version.

Confirmed by inspecting the real Depth-Anything-3 source
(github.com/ByteDance-Seed/Depth-Anything-3): nothing in it actually imports
`open3d` -- it's declared in `pyproject.toml` but dead weight in every code
path this project's `depth_anything3_adapter.py` exercises (single-image
monocular inference, no glb/colmap/gaussian-splatting export). If this stub
is ever really imported, that means some new code path genuinely needs
`open3d` -- install the real package instead of relying on this one.
"""

raise ImportError(
    "This is a stub 'open3d' package (see vendor/open3d_stub/open3d/__init__.py). "
    "If you're seeing this, some code path now actually needs open3d -- install "
    "the real package instead."
)
