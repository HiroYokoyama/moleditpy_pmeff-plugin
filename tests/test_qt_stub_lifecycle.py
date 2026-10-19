"""Temporary GUI stubs must preserve native Qt's module and enum identities."""

import importlib
import sys

from tests import qt_stubs


def test_native_qt_modules_and_message_types_survive_stub_swap(qtlog):
    originals = {
        name: importlib.import_module(name) for name in qt_stubs._MODULE_NAMES
    }
    try:
        qt_stubs.install_qt_stubs()
        qt_stubs.install_qt_stubs()  # Repeated installation must keep the snapshot.
        assert sys.modules["PySide6.QtCore"] is not originals["PySide6.QtCore"]
    finally:
        qt_stubs.remove_qt_stubs()
    qt_stubs.remove_qt_stubs()  # Repeated removal must leave native modules intact.
    for name, module in originals.items():
        assert sys.modules[name] is module

    originals["PySide6.QtCore"].qWarning("native Qt warning after stub teardown")
    assert any(
        record.message == "native Qt warning after stub teardown"
        for record in qtlog.records
    )


def test_viz_stub_removal_restores_existing_modules(monkeypatch):
    originals = {name: object() for name in ("vtk", "pyvista")}
    for name, module in originals.items():
        monkeypatch.setitem(sys.modules, name, module)
    try:
        qt_stubs.install_viz_stubs()
        qt_stubs.install_viz_stubs()
        assert sys.modules["vtk"] is not originals["vtk"]
    finally:
        qt_stubs.remove_viz_stubs()
    for name, module in originals.items():
        assert sys.modules[name] is module
