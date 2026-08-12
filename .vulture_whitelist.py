# Confirmed false positives from `pixi run deadcode` (vulture), referenced
# only through dynamic dispatch vulture's static analysis can't trace (a
# framework callback, a Blender API attribute, a pytest convention). Each
# bare-attribute reference below whitelists that name everywhere it's used
# in the scanned tree (vulture's own documented idiom), not just one class,
# confirmed appropriate here since every occurrence found by hand was the
# same kind of framework callback, not a mix of real and false positives.

# torch.nn.Module.forward(), called via Module.__call__, never directly;
# every `unused method 'forward'` hit in pipeline/adapters/**/*.py is this.
_.forward

# Blender (bpy) API attributes, set on a bpy object/struct and read by
# Blender's own C-level code, never from Python; every `unused attribute`
# hit in pipeline/stages/stage_10_export.py and tests/test_stage_10_export.py
# is this (rotation_euler, interpolation, subtarget, inverse_matrix,
# influence, active, use_connect).
_.rotation_euler
_.interpolation
_.subtarget
_.inverse_matrix
_.influence
_.active
_.use_connect

# pytest conventions: a module-level `pytestmark` is read by pytest's own
# collection machinery, not literal code; an `autouse=True` fixture (e.g.
# test_stage_10_export.py's `_isolated_bpy_scene`) runs for every test in the
# file without ever being referenced by name.
pytestmark
_isolated_bpy_scene

# sam31_detector.py's own `semantic_seg_head`: already documented in that
# file's own comment (checkpoint has it, never invoked, matches the
# reference implementation's own behavior), not investigated further here.
semantic_seg_head
