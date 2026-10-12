"""Readable, copyable PMEFF results; no widgets are created by the engine."""

from __future__ import annotations

from typing import Any, Sequence

# PyQt's runtime C-extension symbols are not visible to pylint's static importer.
# pylint: disable=no-name-in-module
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_open_dialogs: list[QDialog] = []


def _show_data(
    title: str,
    summary: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: str = "",
    parent: Any = None,
) -> None:
    """Open the complete, copyable data in a separate details window."""
    dialog = QDialog(parent if isinstance(parent, QWidget) else None)
    dialog.setWindowTitle(title + " — Details")
    dialog.resize(680, 520)
    layout = QVBoxLayout(dialog)
    label = QLabel(summary)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(label)
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.setItem(i, j, QTableWidgetItem(str(value)))
    layout.addWidget(table)
    if note:
        footer = QLabel(note)
        footer.setTextFormat(Qt.TextFormat.PlainText)
        footer.setWordWrap(True)
        layout.addWidget(footer)
    copy_button = QPushButton("Copy results")
    text = (
        summary
        + "\n"
        + "\t".join(headers)
        + "\n"
        + "\n".join("\t".join(str(v) for v in row) for row in rows)
        + "\n"
        + note
    )
    copy_button.clicked.connect(lambda: QApplication.clipboard().setText(text))
    layout.addWidget(copy_button)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    _show_modeless(dialog)


def _show_modeless(dialog: QDialog) -> None:
    """Keep each window alive and release it even if its parent closes."""
    _open_dialogs.append(dialog)

    def forget() -> None:
        if dialog in _open_dialogs:
            _open_dialogs.remove(dialog)

    dialog.finished.connect(forget)
    dialog.destroyed.connect(forget)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.show()


def show_result(
    title: str,
    summary: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: str = "",
    parent: Any = None,
    review: bool = False,
) -> bool:
    """Show a compact outcome with an optional full-data details window."""
    if QApplication.instance() is None:
        return True
    dialog = QDialog(parent if isinstance(parent, QWidget) else None)
    dialog.setWindowTitle(title)
    dialog.resize(470, 200)
    layout = QVBoxLayout(dialog)
    label = QLabel(summary)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(label)
    scope_text = "Initial DFT structure preparation only; not quantitatively accurate."
    if headers and headers[0] == "Mode":
        scope_text += (
            " Mode values are not physical wavenumbers (cm⁻¹) or accurate spectra."
        )
    scope = QLabel(scope_text)
    scope.setWordWrap(True)
    layout.addWidget(scope)
    buttons = QDialogButtonBox()
    if headers or rows or note:
        details = buttons.addButton("Details…", QDialogButtonBox.ButtonRole.ActionRole)
        details.setAutoDefault(False)
        details.clicked.connect(
            lambda: _show_data(title, summary, headers, rows, note, dialog)
        )
    if review:
        apply_button = buttons.addButton(
            "Apply unconverged geometry", QDialogButtonBox.ButtonRole.AcceptRole
        )
        apply_button.setAutoDefault(False)
        keep = buttons.addButton(
            "Keep original geometry", QDialogButtonBox.ButtonRole.RejectRole
        )
        keep.setDefault(True)
    else:
        buttons.setStandardButtons(QDialogButtonBox.StandardButton.Close)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if review:
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        dialog.deleteLater()
        return accepted
    _show_modeless(dialog)
    return True
