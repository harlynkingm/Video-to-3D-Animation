"""The pipeline UI's single window: a form that composes a `pipeline.run`
invocation and a console log that streams its output, in place of running
that command by hand on the CLI (see README.md's Quick Start section).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QFontDatabase, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pipeline.progress_tracker import PROGRESS_JSON_NAME, ObjectShapeHint, StageName

from ui.pipeline_runner import PipelineRunner, RunFormState, build_run_argv

MAX_STAGE_NUMBER = max(stage.stage_number for stage in StageName)
VIDEO_FILE_FILTER = "Video files (*.mp4 *.mov *.mpeg *.mpg *.flv *.wmv);;All files (*)"

WINDOW_TITLE = "Video to 3D Animation"

UI_SOURCE_VIDEO = "Source video:"
UI_DESTINATION_FOLDER = "Destination folder:"
UI_HUMAN_PROMPT = "Human prompt:"
UI_OBJECT_PROMPT = "Object prompt:"
UI_CAMERA = "Camera:"
UI_BROWSE = "Browse..."
UI_IMAGE_SEQUENCE = "Use image sequence for source video"
UI_FPS = "FPS:"
UI_FOCAL_LENGTH = "Focal length (mm):"
UI_SENSOR_WIDTH = "Sensor width (mm):"
UI_OBJECT_SHAPE = "Object shape:"
UI_ADVANCED_OPTIONS = "Advanced options"
UI_STAGE_RANGE = "Stage range:"
UI_CONSOLE_LOG = "Console log:"
UI_RUN_BUTTON = "Run"
UI_STOP_BUTTON = "Stop"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

        self.pipeline_runner = PipelineRunner(self)
        self.pipeline_runner.output_received.connect(self._on_output_received)
        self.pipeline_runner.finished.connect(self._on_process_finished)

        self.form_container = self._build_form_container()

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log_edit.setMinimumHeight(150)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.form_container)
        layout.addWidget(QLabel(UI_CONSOLE_LOG))
        layout.addWidget(self.log_edit, 1)
        self.setCentralWidget(central)
        # Sized to the layout's own minimum (narrowest usable width, shortest
        # usable height) rather than an arbitrary fixed size -- both shrink
        # further still since the advanced section and fps field start hidden.
        self.adjustSize()

    # -- construction -----------------------------------------------------

    def _build_form_container(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        top_form = QFormLayout()
        top_form.addRow(UI_SOURCE_VIDEO, self._build_source_row())
        top_form.addRow(UI_DESTINATION_FOLDER, self._build_destination_row())

        self.human_prompt_edit = QLineEdit()
        top_form.addRow(UI_HUMAN_PROMPT, self.human_prompt_edit)

        self.object_prompt_edit = QLineEdit()
        top_form.addRow(UI_OBJECT_PROMPT, self.object_prompt_edit)

        top_form.addRow(UI_CAMERA, self._build_camera_row())
        layout.addLayout(top_form)

        layout.addWidget(self._build_advanced_toggle())
        self.advanced_frame = self._build_advanced_frame()
        layout.addWidget(self.advanced_frame)

        layout.addLayout(self._build_run_stop_row())
        return container

    def _build_source_row(self) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        row_layout.addWidget(self.video_path_edit, 1)

        browse_button = QPushButton(UI_BROWSE)
        browse_button.clicked.connect(self._on_browse_source_clicked)
        row_layout.addWidget(browse_button)

        return row

    def _build_image_sequence_row(self) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        self.image_sequence_checkbox = QCheckBox(UI_IMAGE_SEQUENCE)
        self.image_sequence_checkbox.toggled.connect(self._on_image_sequence_toggled)
        row_layout.addWidget(self.image_sequence_checkbox)

        self.fps_label = QLabel(UI_FPS)
        self.fps_label.setVisible(False)
        row_layout.addWidget(self.fps_label)

        self.fps_edit = QLineEdit()
        self.fps_edit.setValidator(QDoubleValidator(0.0001, 1000.0, 4, self.fps_edit))
        self.fps_edit.setVisible(False)
        self.fps_edit.setMaximumWidth(80)
        row_layout.addWidget(self.fps_edit)

        row_layout.addStretch(1)
        return row

    def _build_destination_row(self) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        self.destination_edit = QLineEdit()
        self.destination_edit.setReadOnly(True)
        row_layout.addWidget(self.destination_edit, 1)

        browse_button = QPushButton(UI_BROWSE)
        browse_button.clicked.connect(self._on_browse_destination_clicked)
        row_layout.addWidget(browse_button)

        return row

    def _build_camera_row(self) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        row_layout.addWidget(QLabel(UI_FOCAL_LENGTH))
        self.focal_length_edit = QLineEdit()
        self.focal_length_edit.setValidator(QDoubleValidator(0.0001, 100000.0, 4, self.focal_length_edit))
        row_layout.addWidget(self.focal_length_edit)

        row_layout.addWidget(QLabel(UI_SENSOR_WIDTH))
        self.sensor_width_edit = QLineEdit()
        self.sensor_width_edit.setValidator(QDoubleValidator(0.0001, 100000.0, 4, self.sensor_width_edit))
        row_layout.addWidget(self.sensor_width_edit)

        return row

    def _build_advanced_toggle(self) -> QToolButton:
        toggle = QToolButton()
        toggle.setText(UI_ADVANCED_OPTIONS)
        toggle.setCheckable(True)
        toggle.setChecked(False)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setStyleSheet("QToolButton { border: none; }")
        toggle.toggled.connect(self._on_advanced_toggled)
        self.advanced_toggle = toggle
        return toggle

    def _build_advanced_frame(self) -> QFrame:
        frame = QFrame()
        frame.setVisible(False)
        form = QFormLayout(frame)
        # Unlike top_form (whose text fields should fill the available width),
        # this form's fields -- the object-shape combo, the stage-range row --
        # should only take the width their own content needs.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        self.object_shape_combo = QComboBox()
        self.object_shape_combo.addItems([hint.value for hint in ObjectShapeHint])
        self.object_shape_combo.setCurrentText(ObjectShapeHint.AUTO.value)
        form.addRow(UI_OBJECT_SHAPE, self.object_shape_combo)

        stage_row = QWidget()
        stage_row_layout = QHBoxLayout(stage_row)
        stage_row_layout.setContentsMargins(0, 0, 0, 0)
        stage_numbers = [str(n) for n in range(MAX_STAGE_NUMBER + 1)]
        self.stage_from_combo = QComboBox()
        self.stage_from_combo.addItems(stage_numbers)
        self.stage_from_combo.setCurrentText("0")
        self.stage_to_combo = QComboBox()
        self.stage_to_combo.addItems(stage_numbers)
        self.stage_to_combo.setCurrentText(str(MAX_STAGE_NUMBER))
        stage_row_layout.addWidget(self.stage_from_combo)
        stage_row_layout.addWidget(QLabel("to"))
        stage_row_layout.addWidget(self.stage_to_combo)
        form.addRow(UI_STAGE_RANGE, stage_row)

        self.force_rerun_checkbox = QCheckBox("Force rerun all stages")
        form.addRow("", self.force_rerun_checkbox)

        self.render_previews_checkbox = QCheckBox("Render preview outputs for each stage")
        form.addRow("", self.render_previews_checkbox)

        form.addRow("", self._build_image_sequence_row())

        return frame

    def _build_run_stop_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)

        button_width = 90

        self.stop_button = QPushButton(UI_STOP_BUTTON)
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedWidth(button_width)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        row.addWidget(self.stop_button)

        self.run_button = QPushButton(UI_RUN_BUTTON)
        self.run_button.setFixedWidth(button_width)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; padding: 4px 16px; }"
            "QPushButton:hover:!disabled { background-color: #106EBE; }"
            "QPushButton:disabled { background-color: #999999; color: #dddddd; }"
        )
        row.addWidget(self.run_button)

        return row

    # -- browse / toggle handlers ------------------------------------------

    def _on_browse_source_clicked(self) -> None:
        if self.image_sequence_checkbox.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select image sequence folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select source video", filter=VIDEO_FILE_FILTER)
        if path:
            self.video_path_edit.setText(path)

    def _on_browse_destination_clicked(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if path:
            self.destination_edit.setText(path)

    def _on_image_sequence_toggled(self, checked: bool) -> None:
        self.fps_label.setVisible(checked)
        self.fps_edit.setVisible(checked)

    def _on_advanced_toggled(self, checked: bool) -> None:
        self.advanced_frame.setVisible(checked)
        self.advanced_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    # -- run / stop ---------------------------------------------------------

    def _parse_positive_float(
        self, text: str, label: str, errors: list[str], required: bool = True,
    ) -> float | None:
        text = text.strip()
        if not text:
            if required:
                errors.append(f"{label} is required.")
            return None
        try:
            value = float(text)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return None
        if value <= 0:
            errors.append(f"{label} must be greater than 0.")
            return None
        return value

    def _on_run_clicked(self) -> None:
        if self.pipeline_runner.is_running():
            return

        errors: list[str] = []

        destination = self.destination_edit.text().strip()
        if not destination:
            errors.append("Destination folder is required.")

        # An existing run at destination already has its own stored input --
        # pipeline.run resumes it, applying only whichever of these fields
        # are actually given as overrides, so none of them are required here.
        resuming = bool(destination) and (Path(destination) / PROGRESS_JSON_NAME).exists()

        video_path = self.video_path_edit.text().strip()
        if not video_path:
            if not resuming:
                errors.append("Source video/image sequence path is required.")
        elif not Path(video_path).exists():
            errors.append(f"Source path does not exist: {video_path}")

        is_image_sequence = self.image_sequence_checkbox.isChecked()
        source_fps: float | None = None
        if is_image_sequence:
            source_fps = self._parse_positive_float(self.fps_edit.text(), "FPS", errors)

        human_prompt = self.human_prompt_edit.text().strip()
        if not human_prompt and not resuming:
            errors.append("Human prompt is required.")

        focal_length = self._parse_positive_float(
            self.focal_length_edit.text(), "Focal length", errors, required=not resuming,
        )
        sensor_width = self._parse_positive_float(
            self.sensor_width_edit.text(), "Sensor width", errors, required=not resuming,
        )

        if errors:
            QMessageBox.warning(self, "Cannot start run", "\n".join(errors))
            return

        state = RunFormState(
            destination_folder=destination,
            video_path=video_path or None,
            human_prompt=human_prompt or None,
            object_prompt=self.object_prompt_edit.text().strip() or None,
            focal_length_mm=focal_length,
            sensor_width_mm=sensor_width,
            is_image_sequence=is_image_sequence,
            source_fps=source_fps,
            start_stage=int(self.stage_from_combo.currentText()),
            stop_stage=int(self.stage_to_combo.currentText()),
            object_shape=self.object_shape_combo.currentText(),
            force_all=self.force_rerun_checkbox.isChecked(),
            render_previews=self.render_previews_checkbox.isChecked(),
        )
        argv = build_run_argv(state)
        self.log_edit.appendPlainText("$ " + " ".join(argv))
        self._set_running(True)
        self.pipeline_runner.start(argv)

    def _on_stop_clicked(self) -> None:
        self.pipeline_runner.stop()

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.form_container.setEnabled(not running)

    # -- process output -------------------------------------------------

    def _on_output_received(self, text: str) -> None:
        scrollbar = self.log_edit.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_edit.setTextCursor(cursor)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _on_process_finished(self, exit_code: int) -> None:
        self.log_edit.appendPlainText(f"\n[process exited with code {exit_code}]")
        self._set_running(False)

    def closeEvent(self, event) -> None:
        if self.pipeline_runner.is_running():
            self.pipeline_runner.stop()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
