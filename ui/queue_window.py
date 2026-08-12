"""The run queue window: a list of queued runs plus Edit/Remove/Stop/Run
controls, a shared status/progress indicator, and a console log, driven by
a QueueRunner. Hides instead of closing while MainWindow is still open, so
reopening it via MainWindow's "Add to Queue" button always shows the same
in-memory queue for the life of the app (session-only persistence); once
MainWindow itself is gone, closing this window is allowed to actually close
it too, so the app can still quit on the last window closing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pipeline.progress_tracker import RunRecord
from ui.pipeline_runner import RunFormState
from ui.queue import QueueItem, QueueItemStatus, QueueRunner, compute_queue_progress, summarize_run_form_state
from ui.widgets import (
    ConsoleLog,
    RunStatusIndicator,
    DEFAULT_BUTTON_WIDTH,
    ACTIVE_BUTTON_STYLESHEET,
    SECTION_SPACING,
    build_stage_progress_bar,
    build_status_label,
    build_vertical_spacer,
)

if TYPE_CHECKING:
    from ui.main_window import MainWindow

PROGRESS_POLL_INTERVAL_MS = 750

UI_QUEUE_WINDOW_TITLE = "Video to 3D Animation - Render Queue"
UI_EDIT_BUTTON = "Edit Item"
UI_REMOVE_BUTTON = "Remove Item"
UI_STOP_BUTTON = "Stop Queue"
UI_RUN_BUTTON = "Run Queue"
UI_CONSOLE_LOG = "Console"

_STATUS_PREFIX = {
    QueueItemStatus.COMPLETE: ("✓", QColor("#238636")),
    QueueItemStatus.FAILED: ("✗", QColor("#D9534F")),
    QueueItemStatus.RUNNING: ("▶", QColor("#999999")),
    QueueItemStatus.PENDING: ("", None),
}
_COMPLETE_COLOR = QColor("#999999")
_FAILED_COLOR = QColor("#D9534F")
_PREFIX_COLUMN_WIDTH = 20


class _QueueListWidget(QListWidget):
    """A plain QListWidget doesn't reliably clear its selection when you
    click empty space (below the last item, or between items), this makes
    that explicit: clicking anywhere with no item under the cursor deselects
    everything. Horizontal scrolling is disabled outright (rather than just
    relying on the description label wrapping) so a long description can
    never push the status column off the visible edge.
    """

    def __init__(self, on_resize, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._on_resize = on_resize

    def mousePressEvent(self, event) -> None:
        if self.itemAt(event.position().toPoint()) is None:
            self.setCurrentRow(-1)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # The viewport just got narrower/wider, every row's word-wrapped
        # description needs to re-wrap (and its sizeHint's height needs to be
        # recomputed) at the new width, not just left as it was.
        self._on_resize()


_ROW_TEXT_SPACING = 1  # vertical gap between the title and description labels


class _QueueListItemWidget(QWidget):
    """One queue-list row: a title line over a smaller
    description line (with a small left inset so the text doesn't sit flush
    against the selection bar), followed by a status column on the right
    (fixed width, vertically centered, so pending items with no symbol still
    reserve the same space everyone else gets rather than shifting the
    row's right edge). A light bottom border separates it from the next row.

    Overrides heightForWidth()/sizeHint() itself rather than trusting Qt's
    own layout-computed sizeHint(): a QHBoxLayout containing a QVBoxLayout
    doesn't correctly propagate the word-wrapped description label's
    heightForWidth up through nested layouts, so left alone this both inflates
    every row's height and leaves visible empty space between the two labels
    (each vertically centered within its own oversized allotted rect).
    """

    def __init__(self, status: tuple[str, QColor | None], title: str, description: str, color: QColor | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("border-bottom: 1px solid #3c3c3c;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(_ROW_TEXT_SPACING)

        self._title_label = QLabel(title)
        title_font = self._title_label.font()
        title_font.setPointSize(title_font.pointSize() + 1)
        self._title_label.setFont(title_font)
        text_layout.addWidget(self._title_label)

        self._description_label = QLabel(description)
        self._description_label.setWordWrap(True)
        text_layout.addWidget(self._description_label)

        layout.addLayout(text_layout, 1)

        status_label = QLabel(status[0])
        if status[1] is not None:
            status_label.setStyleSheet(f"color: {status[1].name()};")
        status_label.setFixedWidth(_PREFIX_COLUMN_WIDTH)
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)

        if color is not None:
            self._title_label.setStyleSheet(f"color: {color.name()};")
            self._description_label.setStyleSheet(f"color: {color.name()};")

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        margins = self.layout().contentsMargins()
        text_width = width - margins.left() - margins.right() - self.layout().spacing() - _PREFIX_COLUMN_WIDTH
        description_height = self._description_label.heightForWidth(max(text_width, 1))
        return (
            margins.top() + self._title_label.sizeHint().height() + _ROW_TEXT_SPACING
            + description_height + margins.bottom()
        )

    def sizeHint(self) -> QSize:
        width = self.width() if self.width() > 0 else self._title_label.sizeHint().width()
        return QSize(width, self.heightForWidth(width))


class QueueWindow(QWidget):
    def __init__(self, main_window: MainWindow) -> None:
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle(UI_QUEUE_WINDOW_TITLE)

        self.queue_runner = QueueRunner(self)
        self.queue_runner.output_received.connect(self._on_output_received)
        self.queue_runner.item_started.connect(self._on_item_changed)
        self.queue_runner.item_finished.connect(self._on_item_changed)
        self.queue_runner.running_changed.connect(self._on_queue_running_changed)

        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(PROGRESS_POLL_INTERVAL_MS)
        self.progress_timer.timeout.connect(self._on_progress_tick)

        # See _refresh_list's own comment. Must exist before list_widget
        # does, since constructing/showing it can fire a resize immediately.
        self._refreshing = False
        self.list_widget = _QueueListWidget(on_resize=self._refresh_list)
        self.list_widget.currentRowChanged.connect(self.update_interactive_state)

        self.console_log = ConsoleLog()

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget, 1)
        layout.addLayout(self._build_button_row())
        layout.addSpacing(SECTION_SPACING)
        layout.addLayout(self._build_status_row())
        self.progress_bar = build_stage_progress_bar()
        layout.addWidget(self.progress_bar)
        progress_console_spacer = build_vertical_spacer()
        layout.addWidget(progress_console_spacer)
        layout.addWidget(QLabel(UI_CONSOLE_LOG))
        layout.addWidget(self.console_log, 1)

        self.status_indicator = RunStatusIndicator(self.status_label_left, self.status_label_right, self.progress_bar, progress_console_spacer)

        self.resize(560, 640)
        self.update_interactive_state()

    # -- construction -----------------------------------------------------

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.edit_button = QPushButton(UI_EDIT_BUTTON)
        self.edit_button.setFixedWidth(DEFAULT_BUTTON_WIDTH)
        self.edit_button.clicked.connect(self._on_edit_clicked)
        row.addWidget(self.edit_button)

        self.remove_button = QPushButton(UI_REMOVE_BUTTON)
        self.remove_button.setFixedWidth(DEFAULT_BUTTON_WIDTH)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        row.addWidget(self.remove_button)

        row.addStretch(1)

        self.stop_button = QPushButton(UI_STOP_BUTTON)
        self.stop_button.setFixedWidth(DEFAULT_BUTTON_WIDTH)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        row.addWidget(self.stop_button)

        self.run_button = QPushButton(UI_RUN_BUTTON)
        self.run_button.setFixedWidth(DEFAULT_BUTTON_WIDTH)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.run_button.setStyleSheet(ACTIVE_BUTTON_STYLESHEET)
        row.addWidget(self.run_button)

        return row

    def _build_status_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.status_label_left = build_status_label()
        row.addWidget(self.status_label_left)
        row.addStretch(1)
        self.status_label_right = build_status_label()
        row.addWidget(self.status_label_right)
        return row

    # -- queue mutation (called by MainWindow, and by our own Remove) -----

    def add_item(self, state: RunFormState) -> None:
        item = QueueItem(state=state)
        self.queue_runner.queue.append(item)
        self._refresh_list(select=item)

    def update_item(self, item: QueueItem, state: RunFormState) -> None:
        item.state = state
        item.status = QueueItemStatus.PENDING
        self._refresh_list(select=item)

    def remove_item(self, item: QueueItem) -> None:
        queue = self.queue_runner.queue
        index = next((i for i, existing in enumerate(queue) if existing is item), None)
        if index is None:
            return
        del queue[index]
        new_selection = None
        if queue:
            new_selection = queue[index - 1] if index > 0 else queue[0]
        self._refresh_list(select=new_selection)

    def show_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    # -- list display -------------------------------------------------------

    def _selected_queue_item(self) -> QueueItem | None:
        current = self.list_widget.currentItem()
        if current is None:
            return None
        return current.data(Qt.ItemDataRole.UserRole)

    def _refresh_list(self, select: QueueItem | None = None) -> None:
        # Re-entrancy guard: clear()+addItem() below can itself trigger the
        # viewport to gain/lose its scrollbar mid-rebuild, which fires
        # _QueueListWidget.resizeEvent -> this same method again. Without this
        # guard that nested call's clear() deletes the QListWidgetItems the
        # outer call is still holding, and the outer call's next
        # setItemWidget()/setCurrentItem() on one of them crashes with
        # "Internal C++ object already deleted". The nested call is simply
        # dropped, the outer call is already about to rebuild every row
        # against the current (post-resize) viewport width anyway.
        if self._refreshing:
            return
        self._refreshing = True
        try:
            # By default, re-select whatever was already selected (a refresh
            # can be triggered by all sorts of things, a resize, an item
            # finishing, that shouldn't disturb the user's current
            # selection). Callers that want a specific item selected instead
            # (add_item(), for the newly-added one) pass it explicitly.
            selected = select if select is not None else self._selected_queue_item()
            self.list_widget.clear()
            # The description label word-wraps, so its (and therefore the
            # whole row's) required height depends on exactly how wide it'll
            # actually be rendered, constrain each row to the viewport's
            # width *before* asking for its sizeHint(), or Qt computes the
            # height for some other (usually much wider, unwrapped) width and
            # rows end up too short.
            row_width = self.list_widget.viewport().width()
            for item in self.queue_runner.queue:
                title = Path(item.state.destination_folder).name or item.state.destination_folder
                description = summarize_run_form_state(item.state)[0]
                color = None
                if item.status == QueueItemStatus.COMPLETE:
                    color = _COMPLETE_COLOR
                elif item.status == QueueItemStatus.FAILED:
                    color = _FAILED_COLOR
                row_widget = _QueueListItemWidget(_STATUS_PREFIX[item.status], title, description, color)
                if row_width > 0:
                    row_widget.setFixedWidth(row_width)

                list_item = QListWidgetItem()
                list_item.setData(Qt.ItemDataRole.UserRole, item)
                list_item.setSizeHint(row_widget.sizeHint())
                self.list_widget.addItem(list_item)
                self.list_widget.setItemWidget(list_item, row_widget)

                if item is selected:
                    self.list_widget.setCurrentItem(list_item)
        finally:
            self._refreshing = False
        self.update_interactive_state()

    # -- button handlers ------------------------------------------------

    def _on_edit_clicked(self) -> None:
        item = self._selected_queue_item()
        if item is not None:
            self.main_window.load_form_state(item)

    def _on_remove_clicked(self) -> None:
        item = self._selected_queue_item()
        if item is not None and item.status != QueueItemStatus.RUNNING:
            self.remove_item(item)

    def _on_run_clicked(self) -> None:
        if self.main_window.pipeline_runner.is_running():
            QMessageBox.warning(self, "Cannot start queue", "The main window is currently running its own job.")
            return
        self.queue_runner.start()

    def _on_stop_clicked(self) -> None:
        self.queue_runner.stop()

    # -- queue runner signal handlers ------------------------------------

    def _on_output_received(self, text: str) -> None:
        self.console_log.feed(text)

    def _on_item_changed(self, *_args) -> None:
        self._refresh_list()

    def _on_queue_running_changed(self, running: bool) -> None:
        if running:
            self.status_indicator.begin("Starting queue...")
            self.progress_timer.start()
        else:
            self.progress_timer.stop()
            self.status_indicator.end()
        self.update_interactive_state()
        self.main_window.update_interactive_state()

    def _on_progress_tick(self) -> None:
        index = self.queue_runner.current_index
        run_record = None
        if index is not None:
            destination = Path(self.queue_runner.queue[index].state.destination_folder)
            try:
                run_record = RunRecord.load(destination)
            except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
                run_record = None  # progress.json doesn't exist yet, or is mid-write, try again next tick
        progress = compute_queue_progress(self.queue_runner.queue, index, run_record)
        self.status_indicator.apply(progress)

    # -- interactive lockdown ---------------------------------------------

    def update_interactive_state(self) -> None:
        queue_running = self.queue_runner.is_running()
        own_run_active = self.main_window.pipeline_runner.is_running()
        can_interact = not queue_running and not own_run_active
        has_selection = self._selected_queue_item() is not None
        self.list_widget.setEnabled(can_interact)
        self.edit_button.setEnabled(can_interact and has_selection)
        self.remove_button.setEnabled(can_interact and has_selection)
        self.run_button.setEnabled(can_interact)
        self.stop_button.setEnabled(queue_running)

    def closeEvent(self, event) -> None:
        if self.main_window.isVisible():
            # Main window still open, keep the queue alive in the
            # background if active, and reopening via "Add to Queue" later
            # sees the same in-memory state (this is the whole of
            # "session-only persistence": the window/queue simply outlive
            # being hidden).
            event.ignore()
            self.hide()
            return
        # This is the last open window, let it actually close (rather than
        # just hide) so Qt's quitOnLastWindowClosed can end the app, instead
        # of leaving an invisible process running forever.
        if self.queue_runner.is_running():
            self.queue_runner.stop()
        super().closeEvent(event)
