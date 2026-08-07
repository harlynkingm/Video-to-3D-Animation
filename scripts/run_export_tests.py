"""Runs `tests/test_stage_10_export.py` (the only test file needing the
`export` pixi environment's own bpy) as a child process, and reports the
*real* pass/fail result from its `--junit-xml` report rather than the
child's own exit code.

bpy (Blender-as-a-Python-module, not the full application) has a
reproducible access-violation crash during its own interpreter teardown on
Windows -- well after every test has already finished and pytest has already
written its results to disk (confirmed: the junit report is written before
the crash happens). See `pipeline/run.py`'s own module docstring for the
same crash class hit elsewhere in this project. Trusting the child's exit
code directly would report this benign crash as a test failure; running it
as a child (so the crash lands there, not here) and reading its junit
report instead reports what actually happened.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

TEST_FILE = "tests/test_stage_10_export.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        junit_path = Path(tmp_dir) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", TEST_FILE, f"--junit-xml={junit_path}"])

        if not junit_path.exists():
            # No report at all means pytest crashed before finishing (or
            # before writing one), not after -- a real failure, not the
            # benign post-success crash this script exists to tolerate.
            print("export-env pytest produced no junit report -- treating as a real failure.")
            return 1

        root = ET.parse(junit_path).getroot()
        suite = root.find("testsuite") if root.tag == "testsuites" else root
        tests = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        print(f"export-env pytest: {tests} tests, {failures} failures, {errors} errors (see output above)")
        return 1 if (failures or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
