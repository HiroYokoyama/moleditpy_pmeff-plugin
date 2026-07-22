"""Rich hand-built PyQt6 stand-ins for driving the PMEFF dialog modules.

These are lightweight stub classes (not MagicMock-based) so that real branch
logic in ``settings_dialog.py`` / ``geometry_override_dialog.py`` executes:
signals actually call their connected callbacks, ``QDialog.exec()`` can be
made to simulate an Accept/Cancel/Restore-Defaults/Geometry-button click
sequence, and ``QComboBox``/``QTableWidget`` behave enough like the real
widgets that the dialog code paths run unmodified.

Install with :func:`install_qt_stubs` (idempotent) and tear down with
:func:`remove_qt_stubs` so these never leak into the forcefield/rdkit tests
that need the *real* PyQt6-absent (headless) behavior.
"""

from __future__ import annotations

import sys
import types

_MODULE_NAMES = ("PyQt6", "PyQt6.QtWidgets", "PyQt6.QtCore", "PyQt6.QtGui")

# Stack of "current signal emitter", used to implement QObject.sender().
_SENDER_STACK: list = []


class _Signal:
    def __init__(self, owner=None):
        self._owner = owner
        self._fns = []

    def connect(self, fn):
        self._fns.append(fn)

    def emit(self, *args, **kwargs):
        # Real Qt lets a slot declare fewer parameters than a signal emits
        # (e.g. ``toggled(bool)`` connected to a plain ``load_atoms(self)``);
        # mirror that by falling back to a no-arg call on a TypeError.
        _SENDER_STACK.append(self._owner)
        try:
            for fn in list(self._fns):
                try:
                    fn(*args, **kwargs)
                except TypeError:
                    fn()
        finally:
            _SENDER_STACK.pop()


class _QObjectBase:
    def __init__(self, *args, **kwargs):
        pass

    def sender(self):
        return _SENDER_STACK[-1] if _SENDER_STACK else None


# --------------------------------------------------------------------------
# QtCore
# --------------------------------------------------------------------------
class Qt:
    class WindowType:
        Window = 1

    class ItemFlag:
        ItemIsEditable = 1
        NoItemFlags = 0

        def __ror__(self, other):
            return other

    class MouseButton:
        LeftButton = 1

    class KeyboardModifier:
        ControlModifier = 1
        NoModifier = 0


class QObject(_QObjectBase):
    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent


class QEvent:
    class Type:
        MouseButtonPress = 2
        MouseButtonRelease = 3


class QTimer(_QObjectBase):
    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent
        self.timeout = _Signal(owner=self)
        self._active = False
        self._interval = None

    def start(self, ms=None):
        self._interval = ms
        self._active = True

    def stop(self):
        self._active = False

    def isActive(self):
        return self._active


# --------------------------------------------------------------------------
# QtGui
# --------------------------------------------------------------------------
class QColor:
    def __init__(self, *args):
        self.args = args


# --------------------------------------------------------------------------
# QtWidgets
# --------------------------------------------------------------------------
class _LayoutBase(_QObjectBase):
    def __init__(self, parent=None):
        super().__init__()
        self._widgets = []
        if parent is not None:
            parent._layout = self

    def addWidget(self, w):
        self._widgets.append(w)

    def addLayout(self, lay):
        self._widgets.append(lay)

    def addSpacing(self, n):
        pass

    def addStretch(self, n=0):
        pass

    def setSpacing(self, n):
        pass


class QVBoxLayout(_LayoutBase):
    pass


class QHBoxLayout(_LayoutBase):
    pass


class QLabel(_QObjectBase):
    def __init__(self, text="", parent=None):
        super().__init__()
        self.text_ = text

    def setWordWrap(self, *a):
        pass

    def setContentsMargins(self, *a):
        pass

    def setStyleSheet(self, *a):
        pass

    def text(self):
        return self.text_


class QCheckBox(_QObjectBase):
    def __init__(self, label="", parent=None):
        super().__init__()
        self.label = label
        self._checked = False
        self.toggled = _Signal(owner=self)

    def setChecked(self, v):
        self._checked = bool(v)
        self.toggled.emit(self._checked)

    def isChecked(self):
        return self._checked


class QPushButton(_QObjectBase):
    def __init__(self, text="", parent=None):
        super().__init__()
        self.text_ = text
        self.clicked = _Signal(owner=self)

    def setToolTip(self, *a):
        pass

    def text(self):
        return self.text_


class QDialogButtonBox(_QObjectBase):
    class StandardButton:
        Ok = 1
        Cancel = 2
        RestoreDefaults = 4

    def __init__(self, buttons=0, parent=None):
        super().__init__()
        self._mask = buttons
        self.accepted = _Signal(owner=self)
        self.rejected = _Signal(owner=self)
        self._buttons = {}

    def button(self, std):
        if std not in self._buttons:
            label = {
                self.StandardButton.RestoreDefaults: "Restore Defaults",
                self.StandardButton.Ok: "OK",
                self.StandardButton.Cancel: "Cancel",
            }.get(std, "")
            self._buttons[std] = QPushButton(label)
        return self._buttons[std]


class QDialog(_QObjectBase):
    class DialogCode:
        Accepted = 1
        Rejected = 0

    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent
        self._result = self.DialogCode.Rejected
        self._layout = None

    def setWindowTitle(self, *a):
        pass

    def setMinimumWidth(self, *a):
        pass

    def setWindowFlags(self, *a):
        pass

    def resize(self, *a):
        pass

    def accept(self):
        self._result = self.DialogCode.Accepted

    def reject(self):
        self._result = self.DialogCode.Rejected

    def exec(self):
        hook = _EXEC_HOOKS.get(self, None) or _EXEC_HOOKS.get("*")
        if hook is not None:
            hook(self)
        return self._result


class QWidget(_QObjectBase):
    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent
        self._layout = None
        self._flags = None
        self._shown = False
        self._closed = False

    def setWindowFlags(self, flags):
        self._flags = flags

    def setWindowTitle(self, *a):
        pass

    def resize(self, *a):
        pass

    def show(self):
        self._shown = True

    def raise_(self):
        pass

    def activateWindow(self):
        pass

    def close(self):
        self._closed = True
        event = types.SimpleNamespace(accept=lambda: None, ignore=lambda: None)
        self.closeEvent(event)

    def closeEvent(self, event):  # default no-op, subclasses override
        pass


class _FakeTableWidgetItem:
    def __init__(self, text=""):
        self._text = text
        self._flags = 0
        self._bg = None

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t

    def flags(self):
        return self._flags

    def setFlags(self, f):
        self._flags = f

    def setBackground(self, brush):
        self._bg = brush


class _FakeIndex:
    def __init__(self, row, col=0):
        self._row = row
        self._col = col

    def row(self):
        return self._row

    def column(self):
        return self._col


class _FakeModel:
    def __init__(self, table):
        self._table = table

    def index(self, row, col):
        return _FakeIndex(row, col)

    def item(self, i):
        return self._table._combo_items[i] if i < len(self._table._combo_items) else None


class _HeaderView:
    def setSectionResizeMode(self, *a):
        pass


class QHeaderView:
    class ResizeMode:
        Stretch = 1


class QTableWidget(_QObjectBase):
    def __init__(self, parent=None):
        super().__init__()
        self._rows = 0
        self._cols = 0
        self._items = {}
        self._cell_widgets = {}
        self._selected_rows = []
        self.itemSelectionChanged = _Signal(owner=self)
        self._header = _HeaderView()
        self._header_labels = []
        self._signals_blocked = False

    def setColumnCount(self, n):
        self._cols = n

    def columnCount(self):
        return self._cols

    def setHorizontalHeaderLabels(self, labels):
        self._header_labels = list(labels)

    def horizontalHeader(self):
        return self._header

    def blockSignals(self, b):
        prev = self._signals_blocked
        self._signals_blocked = bool(b)
        return prev

    def setRowCount(self, n):
        self._rows = n
        self._items = {k: v for k, v in self._items.items() if k[0] < n}
        self._cell_widgets = {k: v for k, v in self._cell_widgets.items() if k[0] < n}

    def rowCount(self):
        return self._rows

    def insertRow(self, row):
        self._rows += 1

    def setItem(self, row, col, item):
        self._items[(row, col)] = item

    def item(self, row, col):
        return self._items.get((row, col))

    def setCellWidget(self, row, col, widget):
        self._cell_widgets[(row, col)] = widget

    def cellWidget(self, row, col):
        return self._cell_widgets.get((row, col))

    def selectedIndexes(self):
        return [_FakeIndex(r) for r in self._selected_rows]

    def clearSelection(self):
        self._selected_rows = []

    def selectRow(self, row):
        if row not in self._selected_rows:
            self._selected_rows.append(row)
        if not self._signals_blocked:
            self.itemSelectionChanged.emit()

    def model(self):
        return _FakeModel(self)

    def scrollTo(self, index):
        pass

    # test helper
    def _set_selected_rows(self, rows):
        self._selected_rows = list(rows)
        if not self._signals_blocked:
            self.itemSelectionChanged.emit()


class _ComboItem:
    def __init__(self):
        self._enabled = True

    def setEnabled(self, v):
        self._enabled = bool(v)

    def isEnabled(self):
        return self._enabled


class _ComboModel:
    def __init__(self, items):
        self._items = items

    def item(self, i):
        if 0 <= i < len(self._items):
            return self._items[i]
        return None


class QComboBox(_QObjectBase):
    def __init__(self, parent=None):
        super().__init__()
        self._entries = []  # (label, data)
        self._model_items = []
        self._current = -1
        self._props = {}
        self.currentIndexChanged = _Signal(owner=self)

    def addItem(self, label, data=None):
        self._entries.append((label, data))
        self._model_items.append(_ComboItem())
        if self._current == -1:
            self._current = 0

    def model(self):
        return _ComboModel(self._model_items)

    def setCurrentIndex(self, i):
        self._current = i
        self.currentIndexChanged.emit(i)

    def currentIndex(self):
        return self._current

    def currentData(self):
        if 0 <= self._current < len(self._entries):
            return self._entries[self._current][1]
        return None

    def setProperty(self, name, value):
        self._props[name] = value

    def property(self, name):
        return self._props.get(name)


class QApplication:
    pass


# --------------------------------------------------------------------------
# QDialog.exec() hook registry — tests register a callable that runs while
# ``exec()`` is "blocking", so it can click buttons / toggle checkboxes before
# returning the configured Accepted/Rejected code.
# --------------------------------------------------------------------------
_EXEC_HOOKS: dict = {}


def set_exec_hook(hook, dialog=None):
    """Register *hook(dialog)* to run inside the next QDialog.exec() call.

    If *dialog* is None the hook applies to any QDialog instance (keyed
    "*"); pass the specific instance once you have a reference to target it.
    """
    _EXEC_HOOKS[dialog if dialog is not None else "*"] = hook


def clear_exec_hooks():
    _EXEC_HOOKS.clear()


def find_widget(container, cls, text=None):
    """Depth-first search of a layout/dialog's ``_widgets`` for an instance."""
    widgets = getattr(container, "_widgets", None)
    if widgets is None:
        lay = getattr(container, "_layout", None)
        widgets = getattr(lay, "_widgets", [])
    for w in widgets:
        if isinstance(w, cls) and (text is None or getattr(w, "text_", None) == text):
            return w
        found = find_widget(w, cls, text)
        if found is not None:
            return found
    return None


def find_all_widgets(container, cls):
    out = []
    widgets = getattr(container, "_widgets", None)
    if widgets is None:
        lay = getattr(container, "_layout", None)
        widgets = getattr(lay, "_widgets", [])
    for w in widgets:
        if isinstance(w, cls):
            out.append(w)
        out.extend(find_all_widgets(w, cls))
    return out


def install_qt_stubs():
    """Install the stub PyQt6 package tree into sys.modules (idempotent)."""
    existing = sys.modules.get("PyQt6")
    if existing is not None and _PYQT6_MODULE_SENTINEL.get("mod") is existing:
        return

    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.Qt = Qt
    qt_core.QObject = QObject
    qt_core.QEvent = QEvent
    qt_core.QTimer = QTimer

    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_gui.QColor = QColor

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = QWidget
    qt_widgets.QDialog = QDialog
    qt_widgets.QDialogButtonBox = QDialogButtonBox
    qt_widgets.QVBoxLayout = QVBoxLayout
    qt_widgets.QHBoxLayout = QHBoxLayout
    qt_widgets.QLabel = QLabel
    qt_widgets.QCheckBox = QCheckBox
    qt_widgets.QPushButton = QPushButton
    qt_widgets.QTableWidget = QTableWidget
    qt_widgets.QTableWidgetItem = _FakeTableWidgetItem
    qt_widgets.QComboBox = QComboBox
    qt_widgets.QHeaderView = QHeaderView
    qt_widgets.QApplication = QApplication

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    pyqt6.QtWidgets = qt_widgets
    pyqt6.QtGui = qt_gui

    _PYQT6_MODULE_SENTINEL["mod"] = pyqt6
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules["PyQt6.QtWidgets"] = qt_widgets
    sys.modules["PyQt6.QtGui"] = qt_gui


_PYQT6_MODULE_SENTINEL: dict = {}


def remove_qt_stubs():
    """Remove the stub PyQt6 modules installed by :func:`install_qt_stubs`."""
    for name in _MODULE_NAMES:
        sys.modules.pop(name, None)
    _PYQT6_MODULE_SENTINEL.pop("mod", None)
    clear_exec_hooks()
    _SENDER_STACK.clear()


# --------------------------------------------------------------------------
# vtk / pyvista fakes for geometry_override_dialog's optional 3D-picking and
# selection-highlight code paths. Installed deterministically regardless of
# whether the real packages happen to be present on the dev machine, so
# behavior matches the CI environment (where neither is installed).
# --------------------------------------------------------------------------
_PICK_POSITION = [0.0, 0.0, 0.0]


class _FakeCellPicker:
    def SetTolerance(self, tol):
        pass

    def Pick(self, px, py, pz, renderer):
        pass

    def GetPickPosition(self):
        return tuple(_PICK_POSITION)


def set_pick_position(pos):
    _PICK_POSITION[:] = list(pos)


class _FakePolyData:
    def __init__(self, positions):
        self.positions = positions
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def glyph(self, geom=None, scale=None, orient=False):
        return ("glyph_mesh", geom, scale, orient)


def install_viz_stubs():
    vtk_mod = types.ModuleType("vtk")
    vtk_mod.vtkCellPicker = _FakeCellPicker
    sys.modules["vtk"] = vtk_mod

    pv_mod = types.ModuleType("pyvista")
    pv_mod.PolyData = _FakePolyData
    pv_mod.Sphere = lambda radius=1.0: ("sphere", radius)
    sys.modules["pyvista"] = pv_mod


def remove_viz_stubs():
    sys.modules.pop("vtk", None)
    sys.modules.pop("pyvista", None)
    _PICK_POSITION[:] = [0.0, 0.0, 0.0]


def block_module(name):
    """Force ``import name`` to raise ImportError (documented sys.modules trick)."""
    sys.modules[name] = None  # type: ignore[assignment]


def unblock_module(name):
    if sys.modules.get(name) is None:
        del sys.modules[name]
