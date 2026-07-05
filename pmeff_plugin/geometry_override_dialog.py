"""PMEFF Metal Geometry Override — modeless per-atom geometry table (PyQt6).

Opens a non-modal window with a table of the current molecule's atoms and a
per-atom coordination-geometry drop-down (Auto / Linear / Trigonal Planar /
Square Planar / Tetrahedral / Octahedral). The chosen overrides are handed back
to the plugin, which feeds them to the PMEFF force field on the next optimize.

The window mirrors the official *XYZ Editor* plugin's interaction model:

* a *Show metals only* filter (on by default), since overrides matter chiefly
  for metal centers whose geometry connectivity alone cannot determine, and
* interactive selection — clicking an atom in the 3D view selects its table
  row, and selecting rows highlights the atoms in 3D with a yellow halo.

Rows with an active override are tinted light blue. Geometry options whose
coordination number does not match the atom's neighbor count are disabled (e.g.
*Linear* is unavailable on a 3-coordinate center). Overrides are committed to
the plugin on Apply, on Apply & Optimize, and when the window is closed, so they
survive closing/reopening the window and are saved with the project.

All PyQt6 / PyVista / VTK imports are guarded so this module imports cleanly in
headless (test) environments; :func:`open_override_window` returns ``None`` and
the pure helpers below (:data:`GEOMETRY_CHOICES`, :func:`is_metal`) stay usable.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Display label -> canonical geometry key understood by the force field
# (forcefield._VALID_GEOMETRIES). "Auto" (None) leaves the atom on the default
# hybridization- / auto-metal-derived geometry.
# Ordered by coordination number so the drop-down reads low → high.
GEOMETRY_CHOICES: List[tuple] = [
    ("Auto (detect)", None),
    ("Linear", "linear"),
    ("Trigonal Planar", "trigonal_planar"),
    ("Square Planar", "square_planar"),
    ("Tetrahedral", "tetrahedral"),
    ("Trigonal Bipyramidal", "trigonal_bipyramidal"),
    ("Square Pyramidal", "square_pyramidal"),
    ("Octahedral", "octahedral"),
]

# Coordination number each geometry is meaningful for. Options whose value does
# not match an atom's neighbor count are disabled for that atom.
_GEOM_COORDINATION: Dict[str, int] = {
    "linear": 2,
    "trigonal_planar": 3,
    "square_planar": 4,
    "tetrahedral": 4,
    "trigonal_bipyramidal": 5,
    "square_pyramidal": 5,
    "octahedral": 6,
}

# Non-metallic elements (incl. noble gases and the metalloids B, Si, Ge, As,
# Sb, Te, At). Everything else with Z >= 3 is treated as a metal for the filter.
_NONMETAL_Z: frozenset = frozenset(
    {
        1,
        2,
        5,
        6,
        7,
        8,
        9,
        10,
        14,
        15,
        16,
        17,
        18,
        32,
        33,
        34,
        35,
        36,
        51,
        52,
        53,
        54,
        85,
        86,
    }
)


def is_metal(atomic_number: int) -> bool:
    """Whether *atomic_number* is a metal (for the 'metals only' filter)."""
    try:
        z = int(atomic_number)
    except (TypeError, ValueError):
        return False
    return z >= 3 and z not in _NONMETAL_Z


def geometry_allowed(key: Optional[str], degree: int) -> bool:
    """Whether geometry *key* is meaningful for an atom with *degree* neighbors.

    ``Auto`` (None) is always allowed; every named geometry requires its own
    coordination number (linear→2, trigonal_planar→3, square/tetra→4, octa→6).
    """
    if key is None:
        return True
    return _GEOM_COORDINATION.get(key) == degree


def open_override_window(
    context: object,
    overrides: Optional[Dict[int, str]] = None,
    on_apply: Optional[Callable[[Dict[int, str]], None]] = None,
    on_apply_and_optimize: Optional[Callable[[Dict[int, str]], None]] = None,
) -> Optional[object]:
    """Open (or re-show) the modeless geometry-override window.

    Returns the window, or ``None`` in a headless environment (no PyQt6). The
    window is registered with the host under the key ``"pmeff_geometry_override"``
    so repeated calls re-show the same instance rather than stacking windows.
    """
    try:
        from PyQt6.QtWidgets import QApplication  # type: ignore[import]  # noqa: F401
    except ImportError:
        return None

    get_window = getattr(context, "get_window", None)
    win = get_window("pmeff_geometry_override") if callable(get_window) else None
    if win is None:
        win = _OverrideWindow(context, overrides or {}, on_apply, on_apply_and_optimize)
        register = getattr(context, "register_window", None)
        if callable(register):
            register("pmeff_geometry_override", win)
    else:
        # Re-seed the working copy from the caller's current overrides.
        win.set_overrides(overrides or {})
    win.show()
    win.raise_()
    win.activateWindow()
    win.load_atoms()
    return win


try:  # pragma: no cover - exercised only with a real GUI stack
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QTableWidget,
        QTableWidgetItem,
        QComboBox,
        QCheckBox,
        QPushButton,
        QLabel,
        QHeaderView,
    )
    from PyQt6.QtGui import QColor
    from PyQt6.QtCore import Qt, QObject, QEvent, QTimer

    _HAVE_QT = True
except ImportError:  # headless / test environment
    _HAVE_QT = False


if _HAVE_QT:
    # Row tints: light blue = unsaved (not yet applied) change; light green =
    # an applied (committed) override; transparent = no override.
    _BLUE = QColor(205, 227, 251)
    _GREEN = QColor(206, 240, 206)
    _NO_BRUSH = QColor(0, 0, 0, 0)

    class _ClickFilter(QObject):
        """Qt event filter: detect non-drag left clicks on the 3D plotter."""

        def __init__(self, callback, parent=None):
            super().__init__(parent)
            self._callback = callback
            self._press_pos = None

        def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
            t = event.type()
            if t == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._press_pos = event.position().toPoint()
            elif t == QEvent.Type.MouseButtonRelease:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self._press_pos is not None
                ):
                    rel = event.position().toPoint()
                    dx = rel.x() - self._press_pos.x()
                    dy = rel.y() - self._press_pos.y()
                    if dx * dx + dy * dy <= 25:  # <=5 px -> click, not drag
                        self._callback(rel.x(), rel.y(), obj, event.modifiers())
                    self._press_pos = None
            return False  # never consume — camera interaction still works

    class _OverrideWindow(QWidget):
        """Modeless table for forcing per-atom coordination geometries."""

        _COL_ID, _COL_ELEM, _COL_NBR, _COL_GEOM = range(4)

        def __init__(self, context, overrides, on_apply, on_apply_and_optimize=None):
            super().__init__(parent=_main_window(context))
            self.setWindowFlags(Qt.WindowType.Window)
            self.context = context
            self._on_apply = on_apply
            self._on_apply_and_optimize = on_apply_and_optimize
            # Working copy: atom_index -> canonical geometry key.
            self._overrides: Dict[int, str] = dict(overrides or {})
            # The applied (committed) overrides — the state pending edits revert
            # to on close, and the green-tinted set.
            self._committed: Dict[int, str] = dict(overrides or {})
            # Atom indices changed since the last Apply — tinted until committed.
            self._dirty: set = set()
            self._row_atom: List[int] = []
            self._click_filter = None
            self._last_natoms = None
            self.setWindowTitle("PMEFF — Metal Geometry Override")
            self.resize(560, 480)
            self._init_ui()
            self.load_atoms()
            self._enable_plotter_picking()
            # Reload when the molecule is swapped or atoms added/removed.
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._maybe_reload)
            self._timer.start(600)

        # -- construction ------------------------------------------------
        def _init_ui(self):
            layout = QVBoxLayout(self)

            intro = QLabel(
                "Force the coordination geometry PMEFF uses for individual "
                "atoms — mainly metal centers, whose geometry connectivity "
                "alone cannot determine. <b>Auto</b> keeps the default "
                "behavior. Options that don't fit an atom's neighbor count are "
                "disabled. Unsaved changes are tinted blue; applied overrides "
                "turn green. Click an atom in the 3D view to locate its row."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.metals_only = QCheckBox("Show metals only")
            self.metals_only.setChecked(True)
            self.metals_only.toggled.connect(self.load_atoms)
            layout.addWidget(self.metals_only)

            self.table = QTableWidget()
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(
                ["Atom ID", "Element", "Neighbors", "Geometry"]
            )
            self.table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
            self.table.itemSelectionChanged.connect(self._highlight_selected)
            layout.addWidget(self.table)

            btns = QHBoxLayout()
            self.apply_btn = QPushButton("Apply")
            self.apply_btn.setToolTip(
                "Store these overrides (saved with the project). Does not "
                "re-optimize — run Optimize 3D (PMEFF) to apply them."
            )
            self.apply_btn.clicked.connect(self.apply)
            btns.addWidget(self.apply_btn)

            self.apply_opt_btn = QPushButton("Apply and Optimize")
            self.apply_opt_btn.setToolTip(
                "Store these overrides and immediately re-optimize the current "
                "molecule with PMEFF."
            )
            self.apply_opt_btn.clicked.connect(self.apply_and_optimize)
            btns.addWidget(self.apply_opt_btn)

            self.clear_btn = QPushButton("Clear All")
            self.clear_btn.clicked.connect(self.clear_all)
            btns.addWidget(self.clear_btn)

            btns.addStretch(1)
            self.close_btn = QPushButton("Close")
            self.close_btn.clicked.connect(self.close)
            btns.addWidget(self.close_btn)
            layout.addLayout(btns)

        # -- data --------------------------------------------------------
        def set_overrides(self, overrides):
            # Re-seed the working copy from the applied set; any pending
            # (unsaved) edits are dropped.
            self._overrides = dict(overrides or {})
            self._committed = dict(overrides or {})
            self._dirty = set()

        def _mol(self):
            return getattr(self.context, "current_molecule", None) or getattr(
                self.context, "current_mol", None
            )

        def load_atoms(self):
            """(Re)populate the table from the current molecule."""
            mol = self._mol()
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            self._row_atom = []
            if mol is not None:
                self._last_natoms = mol.GetNumAtoms()
                metals_only = self.metals_only.isChecked()
                for atom in mol.GetAtoms():
                    z = atom.GetAtomicNum()
                    if metals_only and not is_metal(z):
                        continue
                    self._add_row(atom.GetIdx(), atom.GetSymbol(), atom.GetDegree())
            self.table.blockSignals(False)

        def _add_row(self, idx, symbol, degree):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._row_atom.append(idx)

            def _ro(text):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                return item

            self.table.setItem(row, self._COL_ID, _ro(str(idx)))
            self.table.setItem(row, self._COL_ELEM, _ro(symbol))
            self.table.setItem(row, self._COL_NBR, _ro(str(degree)))

            current = self._overrides.get(idx)
            combo = QComboBox()
            for label, key in GEOMETRY_CHOICES:
                combo.addItem(label, key)
            # Disable options that don't fit this atom's coordination number
            # (keep the currently selected one enabled so a loaded override is
            # never silently dropped).
            model = combo.model()
            sel = 0
            for i, (_label, key) in enumerate(GEOMETRY_CHOICES):
                if key == current:
                    sel = i
                allowed = geometry_allowed(key, degree) or key == current
                item = model.item(i)
                if item is not None:
                    item.setEnabled(allowed)
            combo.setCurrentIndex(sel)
            combo.setProperty("atom_idx", idx)
            combo.currentIndexChanged.connect(self._on_geometry_changed)
            self.table.setCellWidget(row, self._COL_GEOM, combo)

            self._paint_row(row)

        def _paint_row(self, row):
            """Tint a row: blue = unsaved change, green = applied override."""
            if row >= len(self._row_atom):
                return
            idx = self._row_atom[row]
            if idx in self._dirty:
                brush = _BLUE
            elif idx in self._overrides:
                brush = _GREEN
            else:
                brush = _NO_BRUSH
            for col in (self._COL_ID, self._COL_ELEM, self._COL_NBR):
                item = self.table.item(row, col)
                if item is not None:
                    item.setBackground(brush)

        def _on_geometry_changed(self, _index):
            combo = self.sender()
            if combo is None:
                return
            idx = int(combo.property("atom_idx"))
            key = combo.currentData()
            if key is None:
                self._overrides.pop(idx, None)
            else:
                self._overrides[idx] = str(key)
            # Any edit is an unsaved change until Apply — mark the row blue.
            self._dirty.add(idx)
            if idx in self._row_atom:
                self._paint_row(self._row_atom.index(idx))

        # -- actions -----------------------------------------------------
        def _commit(self):
            if callable(self._on_apply):
                self._on_apply(dict(self._overrides))

        def _mark_committed(self):
            """Snapshot the applied set; drop blue tint (applied rows go green)."""
            self._committed = dict(self._overrides)
            self._dirty = set()
            for row in range(self.table.rowCount()):
                self._paint_row(row)

        def apply(self):
            self._commit()
            self._mark_committed()

        def apply_and_optimize(self):
            if callable(self._on_apply_and_optimize):
                self._on_apply_and_optimize(dict(self._overrides))
            else:  # pragma: no cover - defensive
                self._commit()
            self._mark_committed()

        def clear_all(self):
            self._overrides = {}
            self._dirty = set()
            self.load_atoms()
            self._commit()

        def _maybe_reload(self):
            mol = self._mol()
            n = mol.GetNumAtoms() if mol is not None else None
            if n != self._last_natoms:
                self.load_atoms()

        # -- 3D interaction ---------------------------------------------
        def _enable_plotter_picking(self):
            try:
                plotter = getattr(self.context, "plotter", None)
                interactor = getattr(plotter, "interactor", None) if plotter else None
                if interactor is None:
                    return
                self._click_filter = _ClickFilter(self._on_plotter_click, parent=self)
                interactor.installEventFilter(self._click_filter)
            except Exception as exc:  # pragma: no cover - defensive GUI guard
                logger.debug("PMEFF override picking unavailable: %s", exc)

        def _disable_plotter_picking(self):
            try:
                plotter = getattr(self.context, "plotter", None)
                interactor = getattr(plotter, "interactor", None) if plotter else None
                if interactor and self._click_filter:
                    interactor.removeEventFilter(self._click_filter)
            except Exception as exc:  # pragma: no cover - defensive GUI guard
                logger.debug("PMEFF override unhook failed: %s", exc)
            self._click_filter = None

        def _on_plotter_click(self, x, y, widget, modifiers):
            try:
                import vtk

                plotter = getattr(self.context, "plotter", None)
                mol = self._mol()
                if plotter is None or mol is None or not mol.GetNumConformers():
                    return
                vtk_y = widget.height() - y
                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.005)
                picker.Pick(x, vtk_y, 0, plotter.renderer)
                pick_pos = picker.GetPickPosition()

                conf = mol.GetConformer()
                best_idx, best_dist = -1, float("inf")
                for atom in mol.GetAtoms():
                    p = conf.GetAtomPosition(atom.GetIdx())
                    d = (
                        (p.x - pick_pos[0]) ** 2
                        + (p.y - pick_pos[1]) ** 2
                        + (p.z - pick_pos[2]) ** 2
                    )
                    if d < best_dist:
                        best_dist, best_idx = d, atom.GetIdx()
                if best_idx < 0 or best_idx not in self._row_atom:
                    return
                row = self._row_atom.index(best_idx)
                ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                if not ctrl:
                    self.table.clearSelection()
                self.table.selectRow(row)
                self.table.scrollTo(self.table.model().index(row, 0))
            except Exception as exc:  # pragma: no cover - defensive GUI guard
                logger.debug("PMEFF override pick failed: %s", exc)

        def _highlight_selected(self):
            plotter = getattr(self.context, "plotter", None)
            if plotter is None:
                return
            try:
                import numpy as np
                import pyvista as pv
                from rdkit import Chem
            except Exception:  # pragma: no cover - optional viz deps
                return

            rows = {ix.row() for ix in self.table.selectedIndexes()}
            mol = self._mol()
            points = []
            if mol is not None and mol.GetNumConformers():
                conf = mol.GetConformer()
                pt = Chem.GetPeriodicTable()
                for row in rows:
                    if row >= len(self._row_atom):
                        continue
                    idx = self._row_atom[row]
                    p = conf.GetAtomPosition(idx)
                    z = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    try:
                        r = pt.GetRvdw(z) * 1.2 * 0.3
                    except RuntimeError:
                        r = 0.45
                    points.append(([p.x, p.y, p.z], r))

            try:
                cam = plotter.camera_position
            except (AttributeError, RuntimeError, TypeError):
                cam = None
            if not points:
                plotter.remove_actor("pmeff_geom_selection")
            else:
                poly = pv.PolyData(np.array([pt_ for pt_, _r in points]))
                poly["radii"] = [r for _pt, r in points]
                spheres = poly.glyph(
                    geom=pv.Sphere(radius=1.0), scale="radii", orient=False
                )
                plotter.add_mesh(
                    spheres,
                    name="pmeff_geom_selection",
                    color="yellow",
                    opacity=0.5,
                    pickable=False,
                )
            if cam is not None:
                try:
                    plotter.camera_position = cam
                except (AttributeError, RuntimeError, TypeError):
                    pass
            plotter.render()

        def closeEvent(self, event):  # noqa: N802 (Qt override)
            # Closing does NOT save: unsaved (blue) changes are discarded. Only
            # applied (green) overrides — already committed via Apply — persist
            # and are restored when the window is reopened. Revert the working
            # copy to the applied snapshot so a reused window reopens clean.
            self._overrides = dict(self._committed)
            self._dirty = set()
            self._disable_plotter_picking()
            plotter = getattr(self.context, "plotter", None)
            if plotter is not None:
                try:
                    plotter.remove_actor("pmeff_geom_selection")
                    plotter.render()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("PMEFF override close cleanup: %s", exc)
            super().closeEvent(event)


def _main_window(context) -> object:
    """Return the host main window from *context*, tolerating either API."""
    getter = getattr(context, "get_main_window", None)
    if callable(getter):
        return getter()
    return getattr(context, "main_window", None)
