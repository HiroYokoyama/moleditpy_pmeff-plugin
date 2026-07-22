"""Tests for pmeff_plugin.settings_dialog under a stubbed PyQt6.

Exercises the real dialog-construction/accept/reject code paths with the
hand-built Qt stand-ins in ``tests/qt_stubs.py`` (not just import coverage):
Accept returns the checkbox states, Cancel/Reject returns None, "Restore
Defaults" resets checkboxes, and the "Metal Geometry Override..." button
closes the dialog (Accept) and then invokes the on_open_geometry callback.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tests import qt_stubs

MODULE_NAME = "pmeff_plugin.settings_dialog"


@pytest.fixture()
def settings_dialog_mod():
    qt_stubs.install_qt_stubs()
    sys.modules.pop(MODULE_NAME, None)
    mod = importlib.import_module(MODULE_NAME)
    try:
        yield mod
    finally:
        sys.modules.pop(MODULE_NAME, None)
        qt_stubs.remove_qt_stubs()


def test_headless_returns_none_without_pyqt6():
    # pytest-qt may already have imported a *real* PyQt6 by the time this
    # runs, so force the guarded import in the module to fail rather than
    # relying on PyQt6 simply being absent (sys.modules[name] = None is the
    # documented way to make `import name` raise ImportError).
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "PyQt6" or k.startswith("PyQt6.")
    }
    sys.modules["PyQt6"] = None  # type: ignore[assignment]
    sys.modules.pop(MODULE_NAME, None)
    try:
        mod = importlib.import_module(MODULE_NAME)
        result = mod.open_settings_dialog(None, {"electronic_effects": True})
        assert result is None
    finally:
        sys.modules.pop(MODULE_NAME, None)
        for k in list(sys.modules):
            if k == "PyQt6" or k.startswith("PyQt6."):
                del sys.modules[k]
        sys.modules.update(saved)


def test_accept_returns_updated_checkbox_states(settings_dialog_mod):
    def hook(dlg):
        dlg.accept()

    qt_stubs.set_exec_hook(hook)
    current = {
        "electronic_effects": True,
        "morse_bonds": False,
        "hbond": True,
        "dispersion": False,
        "polar_contraction": True,
    }
    result = settings_dialog_mod.open_settings_dialog(None, current)
    assert result == current


def test_reject_returns_none(settings_dialog_mod):
    def hook(dlg):
        dlg.reject()

    qt_stubs.set_exec_hook(hook)
    result = settings_dialog_mod.open_settings_dialog(None, {})
    assert result is None


def test_cancel_button_signal_returns_none(settings_dialog_mod):
    def hook(dlg):
        buttons = qt_stubs.find_widget(dlg, qt_stubs.QDialogButtonBox)
        buttons.rejected.emit()

    qt_stubs.set_exec_hook(hook)
    result = settings_dialog_mod.open_settings_dialog(None, {})
    assert result is None


def test_ok_button_signal_accepts(settings_dialog_mod):
    def hook(dlg):
        buttons = qt_stubs.find_widget(dlg, qt_stubs.QDialogButtonBox)
        buttons.accepted.emit()

    qt_stubs.set_exec_hook(hook)
    result = settings_dialog_mod.open_settings_dialog(None, {"morse_bonds": False})
    assert result["morse_bonds"] is False
    assert result["electronic_effects"] is True  # option default


def test_missing_current_keys_fall_back_to_option_defaults(settings_dialog_mod):
    def hook(dlg):
        dlg.accept()

    qt_stubs.set_exec_hook(hook)
    result = settings_dialog_mod.open_settings_dialog(None, {})
    assert result == {
        "electronic_effects": True,
        "morse_bonds": True,
        "hbond": True,
        "dispersion": False,
        "polar_contraction": True,
    }


def test_restore_defaults_button_resets_checkboxes(settings_dialog_mod):
    def hook(dlg):
        buttons = qt_stubs.find_widget(dlg, qt_stubs.QDialogButtonBox)
        restore_btn = buttons.button(
            qt_stubs.QDialogButtonBox.StandardButton.RestoreDefaults
        )
        restore_btn.clicked.emit()
        dlg.accept()

    qt_stubs.set_exec_hook(hook)
    # Start from all-inverted current values; Restore Defaults should reset
    # every checkbox back to the option defaults (all True except dispersion).
    current = {
        "electronic_effects": False,
        "morse_bonds": False,
        "hbond": False,
        "dispersion": True,
        "polar_contraction": False,
    }
    result = settings_dialog_mod.open_settings_dialog(None, current)
    assert result == {
        "electronic_effects": True,
        "morse_bonds": True,
        "hbond": True,
        "dispersion": False,
        "polar_contraction": True,
    }


def test_restore_defaults_uses_caller_supplied_defaults(settings_dialog_mod):
    def hook(dlg):
        buttons = qt_stubs.find_widget(dlg, qt_stubs.QDialogButtonBox)
        restore_btn = buttons.button(
            qt_stubs.QDialogButtonBox.StandardButton.RestoreDefaults
        )
        restore_btn.clicked.emit()
        dlg.accept()

    qt_stubs.set_exec_hook(hook)
    current = {"dispersion": False}
    caller_defaults = {"dispersion": True, "hbond": False}
    result = settings_dialog_mod.open_settings_dialog(
        None, current, defaults=caller_defaults
    )
    assert result["dispersion"] is True
    assert result["hbond"] is False
    # Options not present in caller_defaults still fall back to their own.
    assert result["electronic_effects"] is True


def test_no_geometry_button_when_callback_omitted(settings_dialog_mod):
    def hook(dlg):
        btn = qt_stubs.find_widget(dlg, qt_stubs.QPushButton, "Metal Geometry Override…")
        assert btn is None
        dlg.accept()

    qt_stubs.set_exec_hook(hook)
    settings_dialog_mod.open_settings_dialog(None, {})


def test_geometry_button_closes_dialog_and_invokes_callback(settings_dialog_mod):
    called = []

    def hook(dlg):
        btn = qt_stubs.find_widget(
            dlg, qt_stubs.QPushButton, "Metal Geometry Override…"
        )
        assert btn is not None
        btn.clicked.emit()
        # Clicking the button calls dlg.accept() internally (see source), so
        # exec() should already report Accepted at this point.
        assert dlg._result == qt_stubs.QDialog.DialogCode.Accepted

    qt_stubs.set_exec_hook(hook)
    result = settings_dialog_mod.open_settings_dialog(
        None, {}, on_open_geometry=lambda: called.append(True)
    )
    assert result is not None  # accepted (via the geometry button)
    assert called == [True]


def test_geometry_callback_not_invoked_on_plain_cancel(settings_dialog_mod):
    called = []

    def hook(dlg):
        dlg.reject()

    qt_stubs.set_exec_hook(hook)
    result = settings_dialog_mod.open_settings_dialog(
        None, {}, on_open_geometry=lambda: called.append(True)
    )
    assert result is None
    assert called == []
