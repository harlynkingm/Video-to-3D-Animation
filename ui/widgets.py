"""Small reusable Qt widgets shared between MainWindow and QueueWindow."""

from __future__ import annotations

import re

from PySide6.QtGui import QFontDatabase, QTextCursor
from PySide6.QtWidgets import QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QWidget

from ui.pipeline_runner import StageProgress

# Splits process output on line breaks while keeping the delimiters, so a
# lone "\r" (tqdm's per-frame progress updates) can be told apart from a real
# "\n"/"\r\n" line break in ConsoleLog.feed().
_LINE_BREAK_RE = re.compile(r"(\r\n|\r|\n)")

# Wide enough for the longest button label in either window ("Update in
# Queue", "Remove Item") without clipping.
DEFAULT_BUTTON_WIDTH = 100

ACTIVE_BUTTON_STYLESHEET = """
QPushButton { background-color: #0078D4; color: white; padding: 4px 16px; }
QPushButton:hover:!disabled { background-color: #106EBE; }
QPushButton:disabled { background-color: #999999; color: #dddddd; }
"""


class DroppableLineEdit(QLineEdit):
    """A read-only path field: typing stays blocked (setReadOnly), but it
    still accepts a single file/folder dragged in from Explorer and fills
    itself with that path, setText() works fine on a read-only QLineEdit,
    same as a paired Browse button already relies on.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.setText(urls[0].toLocalFile())
            event.acceptProposedAction()


class ConsoleLog(QPlainTextEdit):
    """Read-only, fixed-width process-output viewer that understands a bare
    "\\r" (tqdm's per-frame progress bar) as "overwrite the current line"
    instead of appending a new one, and only follows new output to the
    bottom when the viewport was already there, a user who's scrolled up
    to read something shouldn't get yanked back down on every update.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.setMinimumHeight(150)

    def feed(self, text: str) -> None:
        scrollbar = self.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        # A plain QTextCursor (as returned by textCursor()) still edits the
        # widget's real document regardless of whether it's ever handed back
        # via setTextCursor(), so the insert/select operations below always
        # take effect. Only setTextCursor() itself (which moves the widget's
        # own visible caret) triggers Qt's auto-scroll-into-view; calling it
        # unconditionally would override the at_bottom check below and force
        # the scrollbar to the end on every single update.
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for piece in _LINE_BREAK_RE.split(text):
            if not piece:
                continue
            if piece in ("\r\n", "\n"):
                cursor.insertBlock()
            elif piece == "\r":
                # tqdm-style progress update: overwrite the current line
                # instead of appending a new one, like a real terminal would.
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
                cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            else:
                cursor.insertText(piece)
        if at_bottom:
            self.setTextCursor(cursor)
            scrollbar.setValue(scrollbar.maximum())


def build_status_label() -> QLabel:
    label = QLabel()
    label.setVisible(False)
    return label


def build_stage_progress_bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 1)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setVisible(False)
    bar.setStyleSheet(
        "QProgressBar { border: 1px solid #555; border-radius: 3px; }"
        "QProgressBar::chunk { background-color: #107C10; }"
    )
    return bar


SECTION_SPACING = 12


def build_vertical_spacer(height: int = SECTION_SPACING) -> QWidget:
    """A fixed-height, initially-hidden gap, unlike a bare layout spacing
    value, this can be shown/hidden in step with a sibling widget (see
    RunStatusIndicator's own spacer, tied to the progress bar's visibility)
    so the gap doesn't linger as dead space once that sibling disappears.
    """
    spacer = QWidget()
    spacer.setFixedHeight(height)
    spacer.setVisible(False)
    return spacer


class RunStatusIndicator:
    """Owns a status label + progress bar pair (built via build_status_label()/
    build_stage_progress_bar() and placed by whichever window needs them,
    since MainWindow and QueueWindow lay them out differently) and
    centralizes the begin/apply/end show-hide-and-text logic both windows
    need around a running process. `spacer`, if given (see
    build_vertical_spacer()), is shown/hidden in lockstep with the progress
    bar, e.g. a gap between the progress bar and the console below it that
    should only take up space while the progress bar itself is visible.
    """

    def __init__(self, status_label_left: QLabel, status_label_right: QLabel | None, progress_bar: QProgressBar, spacer: QWidget | None = None) -> None:
        self.status_label_left = status_label_left
        self.status_label_right = status_label_right
        self.progress_bar = progress_bar
        self.spacer = spacer

    def begin(self, left_text: str = "Starting...", right_text: str | None = None) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label_left.setText(left_text)
        self.status_label_left.setVisible(True)
        if self.status_label_right is not None:
            if right_text is not None:
                self.status_label_right.setText(right_text)
                self.status_label_right.setVisible(True)
            else:
                self.status_label_right.setVisible(False)
        if self.spacer is not None:
            self.spacer.setVisible(True)

    def apply(self, progress: StageProgress) -> None:
        self.progress_bar.setRange(0, max(progress.total, 1))
        self.progress_bar.setValue(progress.completed)
        self.status_label_left.setText(progress.status_text)
        if self.status_label_right is not None:
            if progress.overall_status_text is not None:
                self.status_label_right.setText(progress.overall_status_text)
                self.status_label_right.setVisible(True)
            else:
                self.status_label_right.setVisible(False)

    def end(self) -> None:
        self.progress_bar.setVisible(False)
        self.status_label_left.setVisible(False)
        if self.status_label_right is not None:
            self.status_label_right.setVisible(False)
        if self.spacer is not None:
            self.spacer.setVisible(False)
