"""Dialog for launching image registration from the GUI."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class RegistrationDialog(QDialog):
    def __init__(
        self,
        image_names: list[str],
        moving_image_name: str | None = None,
        reference_image_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Register Images")

        if not image_names:
            raise ValueError("No images are available for registration.")

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QLabel(
            "Choose a moving image, a reference image, and the registration quality. "
            "The resulting transform will be stored in the session."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        self._moving_combo = QComboBox()
        self._reference_combo = QComboBox()
        self._quality_combo = QComboBox()
        self._name_edit = QLineEdit()

        for image_name in image_names:
            self._moving_combo.addItem(image_name)
            self._reference_combo.addItem(image_name)

        self._quality_combo.addItem("Fast", "fast")
        self._quality_combo.addItem("Standard", "standard")
        self._quality_combo.addItem("Accurate", "accurate")

        if moving_image_name and moving_image_name in image_names:
            self._moving_combo.setCurrentText(moving_image_name)
        if reference_image_name and reference_image_name in image_names:
            self._reference_combo.setCurrentText(reference_image_name)
        elif len(image_names) > 1 and self._reference_combo.currentText() == self._moving_combo.currentText():
            self._reference_combo.setCurrentIndex(1)

        self._name_edit.setPlaceholderText("Optional explicit transform name")
        self._name_edit.textEdited.connect(self._disable_auto_name)

        self._auto_name_enabled = True
        self._moving_combo.currentTextChanged.connect(self._update_auto_name)
        self._reference_combo.currentTextChanged.connect(self._update_auto_name)
        self._update_auto_name()

        form.addRow("Moving image", self._moving_combo)
        form.addRow("Reference image", self._reference_combo)
        form.addRow("Quality preset", self._quality_combo)
        form.addRow("Output transform name", self._name_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _disable_auto_name(self, _text: str) -> None:
        self._auto_name_enabled = False

    def _update_auto_name(self, *_args) -> None:
        if not self._auto_name_enabled:
            return

        moving = self.moving_image_name
        reference = self.reference_image_name
        quality = self.quality
        self._name_edit.setText(f"register_{quality}_{moving}_to_{reference}")

    @property
    def moving_image_name(self) -> str:
        return self._moving_combo.currentText().strip()

    @property
    def reference_image_name(self) -> str:
        return self._reference_combo.currentText().strip()

    @property
    def quality(self) -> str:
        return str(self._quality_combo.currentData())

    @property
    def transform_name(self) -> str:
        return self._name_edit.text().strip()