"""Low-level Blender scene/action utilities shared by the other
`pipeline/bpy/` modules.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


def _iter_action_fcurves(action: bpy.types.Action) -> Iterator[bpy.types.FCurve]:
    """Blender 4.4+ (this addon's target, 5.x here) moved F-curves off of
    `Action.fcurves` directly and into a layered layers/strips/channelbags
    structure, `Action.fcurves` doesn't exist at all anymore on a plain
    `bpy.types.Action`. Falls back to the old direct attribute in case a
    future/older bpy build still has it, so this doesn't silently stop
    working across a bpy version bump."""
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                yield from channelbag.fcurves


def _clear_scene(bpy: types.ModuleType) -> None:
    """Clears whatever the current scene contains (a stray cube/camera/
    light, or a previous pass's own armature+mesh) without touching
    preferences, `bpy.ops.wm.read_factory_settings` also resets which
    addons are enabled, which would disable the SMPL-X addon itself.

    Also purges orphaned data-blocks (`do_recursive=True` catches the
    cascade: removing an object orphans its mesh/armature/action data,
    which can itself then orphan more), a real bug found while building
    the floor-grounding fix: this stage runs `smplx_add_animation` twice
    (the floor-offset two-pass build, see `run()`), and leftover orphaned
    action data-blocks from the *first* (uncorrected) pass were somehow
    resurfacing in place of the correct, second-pass one. Purging after
    every clear removes the ambiguity entirely rather than chasing the
    exact reuse mechanism.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
