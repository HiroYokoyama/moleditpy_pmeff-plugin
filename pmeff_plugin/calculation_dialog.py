"""PMEFF worker and progress UI; imported lazily by the plugin entry point."""

from __future__ import annotations

import logging
from threading import Event
from time import monotonic
from typing import Any, Callable

# PyQt's runtime C-extension symbols are not visible to pylint's static importer.
# pylint: disable=no-name-in-module
from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .forcefield import OptimizationCancelled, OptimizationProgress

logger = logging.getLogger(__name__)
Job = Callable[[Callable[..., None], Callable[[], bool]], Any]


class CalculationThread(QThread):
    """Run pure molecular work; GUI updates travel through queued signals."""

    progress = Signal(object)

    def __init__(self, job: Job, parent: QWidget) -> None:
        super().__init__(parent)
        self.job = job
        self.cancel_event = Event()
        self.result: Any = None
        self.error: Exception | None = None
        self._last_update = 0.0

    def report(self, *update: Any) -> None:
        """Limit queued progress updates to ten per second."""
        now = monotonic()
        value: Any = update[0] if len(update) == 1 else update
        final = isinstance(value, OptimizationProgress) and value.phase == "Done"
        if now - self._last_update >= 0.1 or final:
            self._last_update = now
            self.progress.emit(value)

    def run(self) -> None:
        """Calculate without accessing widgets or the host's live molecule."""
        try:
            self.result = self.job(self.report, self.cancel_event.is_set)
        except OptimizationCancelled as exc:
            self.error = exc
        except Exception as exc:  # worker boundary: deliver errors to the GUI
            logger.exception("PMEFF calculation failed")
            self.error = exc


class CalculationDialog(QDialog):
    """Modal progress UI preserves the host's synchronous bool callback API.

    The dialog's Qt event loop keeps repainting and cancellation responsive.
    Closing it requests cancellation and waits for the worker's finished signal;
    a running QThread is never destroyed or forcibly terminated.
    """

    def __init__(self, title: str, job: Job, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        self.label = QLabel("Calculating… Preparing molecular data.")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.worker = CalculationThread(job, self)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.accept)

    @Slot(object)
    def update_progress(self, update: Any) -> None:
        """Apply progress only on the GUI thread."""
        if self.worker.cancel_event.is_set():
            return
        if isinstance(update, OptimizationProgress):
            self.progress_bar.setRange(0, max(update.max_iter, 1))
            self.progress_bar.setValue(update.step)
            self.label.setText(
                f"Calculating… {update.phase}: step {update.step}/{update.max_iter}\n"
                f"Energy: {update.energy:.4f}   Maximum force: {update.max_force:.5f}"
            )
        else:
            step, total = update
            self.progress_bar.setRange(0, max(total, 1))
            self.progress_bar.setValue(step)
            self.label.setText(
                f"Calculating Hessian… coordinate {step}/{total}"
                if step < total
                else "Calculating… Diagonalizing Hessian."
            )

    def reject(self) -> None:
        """Escape, Cancel and window close all request cooperative cancellation."""
        self.worker.cancel_event.set()
        self.cancel_button.setEnabled(False)
        self.label.setText("Cancelling… Waiting for the current calculation to stop.")

    def closeEvent(self, event: Any) -> None:
        """Keep the dialog alive until its worker has stopped."""
        self.reject()
        event.ignore()


def run_job(title: str, job: Job, parent: Any = None) -> Any:
    """Wait with a responsive Qt dialog, or calculate directly without a GUI."""
    app = QApplication.instance()
    if app is None:
        return job(lambda *_args: None, lambda: False)
    if QThread.currentThread() != app.thread():
        raise RuntimeError("PMEFF dialogs must be opened on the GUI thread")
    dialog = CalculationDialog(
        title, job, parent if isinstance(parent, QWidget) else None
    )
    QTimer.singleShot(0, dialog.worker.start)
    dialog.exec()
    dialog.worker.wait()
    result, error = dialog.worker.result, dialog.worker.error
    cancelled = dialog.worker.cancel_event.is_set()
    dialog.deleteLater()
    if cancelled:
        raise OptimizationCancelled("PMEFF calculation cancelled")
    if error is not None:
        raise error
    return result
