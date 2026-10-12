"""Exercise real Qt thread affinity, cancellation and persistent result UI."""

from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QApplication, QTableWidget
from rdkit import Chem
from rdkit.Chem import AllChem

import pmeff_plugin as plugin
from pmeff_plugin.calculation_dialog import CalculationDialog, run_job
from pmeff_plugin.forcefield import OptimizationCancelled
from pmeff_plugin.result_dialog import _open_dialogs, show_result


def test_job_runs_off_gui_thread_and_gui_keeps_ticking(qapp):
    gui_id = int(QThread.currentThreadId())
    ids, ticks = [], []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()

    def job(report, _cancel):
        ids.append(int(QThread.currentThreadId()))
        for i in range(5):
            Event().wait(0.02)
            report(i + 1, 5)
        return 42

    try:
        assert run_job("PMEFF Test", job) == 42
    finally:
        timer.stop()
    assert ids and ids[0] != gui_id
    assert len(ticks) >= 3


def test_closing_progress_cancels_without_destroying_running_thread(qapp):
    stopped = Event()

    def job(_report, cancel):
        for _ in range(500):
            if cancel():
                stopped.set()
                raise OptimizationCancelled("cancelled")
            Event().wait(0.005)
        pytest.fail("close did not request cancellation")

    def close_dialog():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, CalculationDialog) and widget.isVisible():
                widget.close()

    QTimer.singleShot(30, close_dialog)
    with pytest.raises(OptimizationCancelled):
        run_job("PMEFF Cancel Test", job)
    assert stopped.is_set()


def test_worker_errors_reach_caller(qapp):
    def job(_report, _cancel):
        raise ValueError("bad geometry")

    with pytest.raises(ValueError, match="bad geometry"):
        run_job("PMEFF Error Test", job)


def test_cancelled_plugin_optimization_preserves_live_molecule(qapp, monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    before = mol.GetConformer().GetPositions().copy()
    ctx = SimpleNamespace(
        get_main_window=lambda: None, show_status_message=lambda *_args: None
    )

    def slow_engine(copy, **kwargs):
        copy.GetConformer().SetAtomPosition(0, (99.0, 99.0, 99.0))
        for _ in range(500):
            if kwargs["cancel_callback"]():
                raise OptimizationCancelled("cancelled")
            Event().wait(0.005)
        return True, None

    monkeypatch.setattr("pmeff_plugin.forcefield.optimize_rdkit_mol", slow_engine)

    def cancel():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, CalculationDialog) and widget.isVisible():
                widget.reject()

    QTimer.singleShot(30, cancel)
    assert plugin._optimize(mol, ctx) is False
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), before)


def test_result_table_can_be_copied(qapp):
    show_result(
        "PMEFF Results",
        "Imaginary modes: 1",
        ["Mode", "Value"],
        [["1", "-2.0"]],
        "Internal units",
    )
    dialog = _open_dialogs[-1]
    try:
        assert dialog.isVisible()
        assert dialog.findChild(QTableWidget) is None
        details = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Details…"
        )
        details.click()
        data_dialog = _open_dialogs[-1]
        assert data_dialog is not dialog and data_dialog.isVisible()
        table = data_dialog.findChild(QTableWidget)
        assert table.item(0, 1).text() == "-2.0"
        button = next(
            b
            for b in data_dialog.findChildren(QPushButton)
            if b.text() == "Copy results"
        )
        button.click()
        assert "1\t-2.0" in QApplication.clipboard().text()
        assert "Internal units" in QApplication.clipboard().text()
    finally:
        if len(_open_dialogs) > 1:
            _open_dialogs[-1].close()
        dialog.close()


def test_unconverged_geometry_is_applied_silently(qapp, monkeypatch):
    """Unconverged optimization applies the geometry with no dialog.

    No review popup is shown; the result is reported in the status bar only
    and the molecule's coordinates are updated in place.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    before = mol.GetConformer().GetPositions().copy()
    status_messages = []
    ctx = SimpleNamespace(
        get_main_window=lambda: None,
        show_status_message=lambda msg, *_a: status_messages.append(msg),
    )
    # Force the optimizer to stop after a single iteration so convergence
    # cannot be reached; the geometry still gets written back.
    monkeypatch.setattr(plugin, "_iteration_budget", lambda: 1)

    assert plugin._optimize(mol, ctx) is True
    # No review dialog was opened — _open_dialogs should not have grown.
    # (The progress dialog has already closed by this point.)
    stopped_msg = next(
        (m for m in status_messages if "stopped" in m.lower()), None
    )
    assert stopped_msg is not None, "status bar must report 'stopped' for unconverged"
