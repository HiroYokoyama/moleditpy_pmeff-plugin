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
GEOMETRY_CHOICES: List[tuple] = [
    ("Auto (detect)", None),
    ("Linear", "linear"),
    ("Trigonal Planar", "trigonal_planar"),
    ("Square Planar", "square_planar"),
    ("Tetrahedral", "tetrahedral"),
    ("Octahedral", "octahedral"),
]

# Non-metallic elements (incl. noble gases and the metalloids B, Si, Ge, As,
# Sb, Te, At). Everything else with Z >= 3 is treated as a metal for the filter.
_NONMETAL_Z: frozenset = frozenset(
    {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18,
     32, 33, 34, 35, 36, 51, 52, 53, 54, 85, 86}
)


def is_metal(atomic_number: int) -> bool:
    """Whether *atomic_number* is a metal (for the 'metals only' filter)."""
    try:
        z = int(atomic_number)
    except (TypeError, ValueError):
        return False
    return z >= 3 and z not in _NONMETAL_Z


def open_override_window(
    context: object,
    overrides: Optional[Dict[int, str]] = None,
    on_apply: Optional[Callable[[Dict[int, str]], None]] = None,
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
        win = _OverrideWindow(context, overrides or {}, on_apply)
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
    from PyQt6.QtCore import Qt, QObject, QEvent, QTimer

    _HAVE_QT = True
except ImportError:  # headless / test environment
    _HAVE_QT = False


if _HAVE_QT:

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

        _COL_IDX, _COL_ELEM, _COL_NBR, _COL_GEOM = range(4)

        def __init__(self, context, overrides, on_apply):
            super().__init__(parent=_main_window(context))
            self.setWindowFlags(Qt.WindowType.Window)
            self.context = context
            self._on_apply = on_apply
            # Working copy: atom_index -> canonical geometry key.
            self._overrides: Dict[int, str] = dict(overrides or {})
            self._row_atom: List[int] = []
            self._click_filter = None
            self._last_natoms = None
            self.setWindowTitle("PMEFF — Metal Geometry Override")
            self.resize(560, 460)
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
                "behavior. Click an atom in the 3D view to locate its row."
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
                ["Index", "Element", "Neighbors", "Geometry"]
            )
            self.table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
            self.table.itemSelectionChanged.connect(self._highlight_selected)
            layout.addWidget(self.table)

            btns = QHBoxLayout()
            self.apply_btn = QPushButton("Apply")
            self.apply_btn.setToolTip(
                "Store these overrides; run Optimize 3D (PMEFF) to apply them."
            )
            self.apply_btn.clicked.connect(self.apply)
            btns.addWidget(self.apply_btn)

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
            self._overrides = dict(overrides or {})

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
                    self._add_row(atom.GetIdx(), atom.GetSymbol(), z, atom.GetDegree())
            self.table.blockSignals(False)

        def _add_row(self, idx, symbol, z, degree):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._row_atom.append(idx)

            item_idx = QTableWidgetItem(str(idx))
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, self._COL_IDX, item_idx)

            item_el = QTableWidgetItem(symbol)
            item_el.setFlags(item_el.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, self._COL_ELEM, item_el)

            item_nb = QTableWidgetItem(str(degree))
            item_nb.setFlags(item_nb.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, self._COL_NBR, item_nb)

            combo = QComboBox()
            for label, key in GEOMETRY_CHOICES:
                combo.addItem(label, key)
            current = self._overrides.get(idx)
            sel = 0
            for i, (_label, key) in enumerate(GEOMETRY_CHOICES):
                if key == current:
                    sel = i
                    break
            combo.setCurrentIndex(sel)
            combo.setProperty("atom_idx", idx)
            combo.currentIndexChanged.connect(self._on_geometry_changed)
            self.table.setCellWidget(row, self._COL_GEOM, combo)

        def _on_geometry_changed(self, _index):
            combo = self.sender()
            if combo is None:
                return
            idx = combo.property("atom_idx")
            key = combo.currentData()
            if key is None:
                self._overrides.pop(int(idx), None)
            else:
                self._overrides[int(idx)] = str(key)

        # -- actions -----------------------------------------------------
        def apply(self):
            if callable(self._on_apply):
                self._on_apply(dict(self._overrides))

        def clear_all(self):
            self._overrides = {}
            self.load_atoms()
            if callable(self._on_apply):
                self._on_apply({})

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
                    d = (p.x - pick_pos[0]) ** 2 + (p.y - pick_pos[1]) ** 2 + (
                        p.z - pick_pos[2]
                    ) ** 2
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
