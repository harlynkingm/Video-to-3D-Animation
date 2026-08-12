"""Regression test for a Qt-only crash: `_QueueListWidget.resizeEvent`
re-entering `QueueWindow._refresh_list` while a rebuild was already in
progress, triggered once enough queued items overflowed the visible list and
the vertical scrollbar appeared. Kept in its own file, unlike every other
module in ui/tests/, since this specific crash can only be exercised through
Qt's own resize/scrollbar machinery, unlike the rest of ui/tests/'s pure
functions. `QT_QPA_PLATFORM` defaults to offscreen so this still runs without
a real display (e.g. in CI); it's left alone if already set, so a real
display still exercises the real backend during local/manual runs.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from ui.pipeline_runner import RunFormState
from ui.queue_window import QueueWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakePipelineRunner:
    @staticmethod
    def is_running() -> bool:
        return False


class _FakeMainWindow:
    pipeline_runner = _FakePipelineRunner()


def test_adding_enough_items_to_need_a_scrollbar_does_not_crash(qapp):
    # The real crash was a native "already deleted" fault, not a catchable
    # Python exception, this test only proves anything by running to
    # completion rather than aborting the whole process partway through.
    window = QueueWindow(_FakeMainWindow())
    window.resize(400, 300)
    window.show()

    for i in range(12):
        state = RunFormState(destination_folder=f"C:/runs/item_{i}", human_prompt=f"person {i}" * 3)
        window.add_item(state)
        qapp.processEvents()

    assert window.list_widget.count() == 12
    # hide(), not close(): closeEvent() reads self.main_window.isVisible(),
    # which _FakeMainWindow doesn't implement, irrelevant to what this test
    # is actually verifying.
    window.hide()
