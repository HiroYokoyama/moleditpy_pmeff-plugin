"""Tests for pmeff_plugin.geometry_override_dialog under a stubbed PyQt6.

Drives the real ``_OverrideWindow`` class (constructed directly, not through
Qt's event loop) with the hand-built stand-ins in ``tests/qt_stubs.py``, so
row population, the per-atom geometry combo boxes, apply/clear/commit
bookkeeping, the reload timer, and the optional 3D-picking / selection
highlight paths all execute as real code rather than being mocked away.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from tests import qt_stubs

MODULE_NAME = "pmeff_plugin.geometry_override_dialog"


def make_mol(nums, bonds, coords):
    rw = Chem.RWMol()
    for z in nums:
        rw.AddAtom(Chem.Atom(z))
    for i, j in bonds:
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


# Fe(0) tetrahedrally coordinated by 4 F, plus an isolated non-metal C(5).
NUMS = [26, 9, 9, 9, 9, 6]
BONDS = [(0, 1), (0, 2), (0, 3), (0, 4)]
COORDS = [
    [0, 0, 0],
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [10, 10, 10],
]


class FakePoint:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


class FakePos:
    def __init__(self, x, y):
        self._pt = FakePoint(x, y)

    def toPoint(self):
        return self._pt


class FakeEvent:
    def __init__(self, etype, button=None, xy=None, modifiers=0):
        self._type = etype
        self._button = button
        self._xy = xy
        self._modifiers = modifiers

    def type(self):
        return self._type

    def button(self):
        return self._button

    def position(self):
        return FakePos(*self._xy)

    def modifiers(self):
        return self._modifiers


class FakeInteractor:
    def __init__(self, raise_on_install=False):
        self.filters = []
        self._raise_on_install = raise_on_install
        self._dpr = 1.0
        self._height = 100

    def installEventFilter(self, f):
        if self._raise_on_install:
            raise RuntimeError("boom")
        self.filters.append(f)

    def removeEventFilter(self, f):
        if f in self.filters:
            self.filters.remove(f)

    def devicePixelRatioF(self):
        return self._dpr

    def height(self):
        return self._height


class FakePlotter:
    def __init__(self, interactor=None):
        self.interactor = interactor
        self.renderer = object()
        self._camera_position = "cam0"
        self.added = []
        self.removed = []
        self.render_calls = 0

    @property
    def camera_position(self):
        return self._camera_position

    @camera_position.setter
    def camera_position(self, value):
        self._camera_position = value

    def add_mesh(self, mesh, name=None, color=None, opacity=None, pickable=None):
        self.added.append(name)

    def remove_actor(self, name):
        self.removed.append(name)

    def render(self):
        self.render_calls += 1


class FakeContext:
    def __init__(self, mol=None, plotter=None):
        self._windows = {}
        self.current_molecule = mol
        self.plotter = plotter

    def get_window(self, key):
        return self._windows.get(key)

    def register_window(self, key, win):
        self._windows[key] = win

    def get_main_window(self):
        return None


@pytest.fixture()
def god_mod():
    qt_stubs.install_qt_stubs()
    qt_stubs.install_viz_stubs()
    sys.modules.pop(MODULE_NAME, None)
    mod = importlib.import_module(MODULE_NAME)
    try:
        yield mod
    finally:
        sys.modules.pop(MODULE_NAME, None)
        qt_stubs.remove_viz_stubs()
        qt_stubs.remove_qt_stubs()


# --------------------------------------------------------------------------
# open_override_window
# --------------------------------------------------------------------------
def test_open_override_window_headless_returns_none():
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "PyQt6" or k.startswith("PyQt6.")
    }
    sys.modules["PyQt6"] = None  # type: ignore[assignment]
    sys.modules.pop(MODULE_NAME, None)
    try:
        mod = importlib.import_module(MODULE_NAME)
        assert mod.open_override_window(FakeContext()) is None
    finally:
        sys.modules.pop(MODULE_NAME, None)
        for k in list(sys.modules):
            if k == "PyQt6" or k.startswith("PyQt6."):
                del sys.modules[k]
        sys.modules.update(saved)


def test_open_override_window_creates_and_registers(god_mod):
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol)
    win = god_mod.open_override_window(ctx, {0: "tetrahedral"})
    assert win is not None
    assert ctx.get_window("pmeff_geometry_override") is win
    assert win._shown is True
    assert win._overrides == {0: "tetrahedral"}


def test_open_override_window_reshows_existing(god_mod):
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol)
    win1 = god_mod.open_override_window(ctx, {0: "tetrahedral"})
    win2 = god_mod.open_override_window(ctx, {0: "linear"})
    assert win1 is win2
    assert win2._overrides == {0: "linear"}


# --------------------------------------------------------------------------
# Construction / row population
# --------------------------------------------------------------------------
def test_init_ui_builds_expected_widgets(god_mod):
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    assert win.metals_only.isChecked() is True
    assert win.table.columnCount() == 4
    assert win.table._header_labels == ["Atom ID", "Element", "Neighbors", "Geometry"]
    assert win.apply_btn.text() == "Apply"
    assert win.apply_opt_btn.text() == "Apply and Optimize"
    assert win.clear_btn.text() == "Clear All"
    assert win.close_btn.text() == "Close"


def test_load_atoms_metals_only_filters_nonmetals(god_mod):
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    assert win.table.rowCount() == 1  # only Fe
    assert win._row_atom == [0]


def test_load_atoms_all_atoms_when_filter_off(god_mod):
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.metals_only.setChecked(False)
    assert win.table.rowCount() == len(NUMS)
    assert win._row_atom == list(range(len(NUMS)))


def test_load_atoms_no_molecule_leaves_table_empty(god_mod):
    ctx = FakeContext(mol=None)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    assert win.table.rowCount() == 0
    assert win._row_atom == []


def test_add_row_preselects_existing_override_and_disables_bad_options(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {0: "tetrahedral"}, on_apply=None)
    combo = win.table.cellWidget(0, win._COL_GEOM)
    assert combo.currentData() == "tetrahedral"
    # Linear (coordination 2) does not fit Fe's degree-4 center -> disabled.
    model = combo.model()
    linear_item = model.item(1)  # index 1 == "Linear" in GEOMETRY_CHOICES
    assert linear_item.isEnabled() is False
    # Tetrahedral (index matching current) stays enabled even though "wrong"
    # coordination would also disable it for a non-matching degree; here it's
    # correct so this just double-checks it's on.
    tet_idx = [i for i, (_l, k) in enumerate(god_mod.GEOMETRY_CHOICES) if k == "tetrahedral"][0]
    assert model.item(tet_idx).isEnabled() is True


def test_add_row_keeps_mismatched_current_override_enabled(god_mod):
    # Fe has degree 4, but seed an override ("linear", coordination 2) that
    # doesn't fit — the code must keep it enabled anyway so a loaded/legacy
    # override is never silently hidden.
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {0: "linear"}, on_apply=None)
    combo = win.table.cellWidget(0, win._COL_GEOM)
    linear_idx = [i for i, (_l, k) in enumerate(god_mod.GEOMETRY_CHOICES) if k == "linear"][0]
    assert combo.model().item(linear_idx).isEnabled() is True
    assert combo.currentData() == "linear"


# --------------------------------------------------------------------------
# Row tinting
# --------------------------------------------------------------------------
def test_paint_row_tints_dirty_blue(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win._dirty.add(0)
    win._paint_row(0)
    item = win.table.item(0, win._COL_ID)
    assert item._bg is god_mod._BLUE


def test_paint_row_tints_applied_green(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {0: "tetrahedral"}, on_apply=None)
    win._paint_row(0)
    item = win.table.item(0, win._COL_ID)
    assert item._bg is god_mod._GREEN


def test_paint_row_no_tint_when_clean(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win._paint_row(0)
    item = win.table.item(0, win._COL_ID)
    assert item._bg == god_mod._NO_BRUSH


def test_paint_row_out_of_range_is_noop(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win._paint_row(99)  # no row 99 -> should not raise


# --------------------------------------------------------------------------
# Geometry combo edits
# --------------------------------------------------------------------------
def test_on_geometry_changed_sets_override_and_marks_dirty(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    combo = win.table.cellWidget(0, win._COL_GEOM)
    tet_idx = [i for i, (_l, k) in enumerate(god_mod.GEOMETRY_CHOICES) if k == "tetrahedral"][0]
    combo.setCurrentIndex(tet_idx)
    assert win._overrides[0] == "tetrahedral"
    assert 0 in win._dirty


def test_on_geometry_changed_auto_clears_override(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {0: "tetrahedral"}, on_apply=None)
    combo = win.table.cellWidget(0, win._COL_GEOM)
    combo.setCurrentIndex(0)  # "Auto (detect)"
    assert 0 not in win._overrides
    assert 0 in win._dirty


# --------------------------------------------------------------------------
# apply / apply_and_optimize / clear_all
# --------------------------------------------------------------------------
def test_apply_commits_and_clears_dirty(god_mod):
    applied = []
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=applied.append)
    win._overrides[0] = "tetrahedral"
    win._dirty.add(0)
    win.apply()
    assert applied == [{0: "tetrahedral"}]
    assert win._committed == {0: "tetrahedral"}
    assert win._dirty == set()


def test_apply_and_optimize_uses_dedicated_callback(god_mod):
    applied, optimized = [], []
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(
        ctx, {}, on_apply=applied.append, on_apply_and_optimize=optimized.append
    )
    win._overrides[0] = "linear"
    win.apply_and_optimize()
    assert optimized == [{0: "linear"}]
    assert applied == []
    assert win._committed == {0: "linear"}


def test_clear_all_empties_overrides_and_commits(god_mod):
    applied = []
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {0: "tetrahedral"}, on_apply=applied.append)
    win.clear_all()
    assert win._overrides == {}
    assert applied == [{}]
    assert win.table.rowCount() == 1  # reloaded


def test_set_overrides_reseeds_working_copy(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {0: "tetrahedral"}, on_apply=None)
    win._dirty.add(0)
    win.set_overrides({0: "linear", 3: "square_planar"})
    assert win._overrides == {0: "linear", 3: "square_planar"}
    assert win._committed == {0: "linear", 3: "square_planar"}
    assert win._dirty == set()


# --------------------------------------------------------------------------
# reload polling
# --------------------------------------------------------------------------
def test_maybe_reload_reloads_on_atom_count_change(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    assert win._last_natoms == len(NUMS)
    bigger = make_mol(NUMS + [6], BONDS, COORDS + [[20, 20, 20]])
    ctx.current_molecule = bigger
    win._maybe_reload()
    assert win._last_natoms == len(NUMS) + 1


def test_maybe_reload_noop_when_unchanged(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.table.setRowCount(0)  # tamper to prove reload doesn't fire
    win._maybe_reload()
    assert win.table.rowCount() == 0


# --------------------------------------------------------------------------
# 3D picking hookup
# --------------------------------------------------------------------------
def test_enable_plotter_picking_installs_filter(god_mod):
    interactor = FakeInteractor()
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=FakePlotter(interactor))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    assert win._click_filter is not None
    assert win._click_filter in interactor.filters


def test_enable_plotter_picking_noop_without_plotter(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=None)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    assert win._click_filter is None


def test_enable_plotter_picking_swallows_exception(god_mod):
    interactor = FakeInteractor(raise_on_install=True)
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=FakePlotter(interactor))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)  # must not raise
    assert win._click_filter is not None  # filter object created, just not installed


def test_disable_plotter_picking_removes_filter(god_mod):
    interactor = FakeInteractor()
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=FakePlotter(interactor))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win._disable_plotter_picking()
    assert win._click_filter is None
    assert interactor.filters == []


def test_on_plotter_click_selects_matching_row(god_mod):
    interactor = FakeInteractor()
    mol = make_mol(NUMS, BONDS, COORDS)
    plotter = FakePlotter(interactor)
    ctx = FakeContext(mol=mol, plotter=plotter)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.metals_only.setChecked(False)  # so every atom has a row
    qt_stubs.set_pick_position(COORDS[2])  # exactly atom idx 2
    win._on_plotter_click(10, 10, interactor, modifiers=0)
    assert win.table._selected_rows == [win._row_atom.index(2)]


def test_on_plotter_click_ctrl_does_not_clear_selection(god_mod):
    interactor = FakeInteractor()
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol, plotter=FakePlotter(interactor))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.metals_only.setChecked(False)
    win.table._set_selected_rows([win._row_atom.index(1)])
    qt_stubs.set_pick_position(COORDS[2])
    ctrl = god_mod.Qt.KeyboardModifier.ControlModifier
    win._on_plotter_click(10, 10, interactor, modifiers=ctrl)
    assert win._row_atom.index(1) in win.table._selected_rows
    assert win._row_atom.index(2) in win.table._selected_rows


def test_on_plotter_click_no_plotter_or_mol_is_noop(god_mod):
    ctx = FakeContext(mol=None, plotter=None)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win._on_plotter_click(0, 0, FakeInteractor(), modifiers=0)  # must not raise


def test_on_plotter_click_best_atom_not_in_rows_is_noop(god_mod):
    interactor = FakeInteractor()
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol, plotter=FakePlotter(interactor))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)  # metals only -> row_atom=[0]
    qt_stubs.set_pick_position(COORDS[5])  # nearest atom is idx 5 (C), filtered out
    win._on_plotter_click(0, 0, interactor, modifiers=0)
    assert win.table._selected_rows == []


def test_on_plotter_click_missing_vtk_is_swallowed(god_mod):
    qt_stubs.block_module("vtk")
    try:
        interactor = FakeInteractor()
        mol = make_mol(NUMS, BONDS, COORDS)
        ctx = FakeContext(mol=mol, plotter=FakePlotter(interactor))
        win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
        win._on_plotter_click(0, 0, interactor, modifiers=0)  # must not raise
    finally:
        qt_stubs.unblock_module("vtk")


# --------------------------------------------------------------------------
# selection highlight
# --------------------------------------------------------------------------
def test_highlight_selected_no_plotter_is_noop(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=None)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win._highlight_selected()  # must not raise


def test_highlight_selected_adds_mesh_for_selection(god_mod):
    plotter = FakePlotter(FakeInteractor())
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol, plotter=plotter)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.metals_only.setChecked(False)
    win.table._set_selected_rows([0])
    assert plotter.added == ["pmeff_geom_selection"]
    assert plotter.render_calls >= 1


def test_highlight_selected_removes_actor_when_empty(god_mod):
    plotter = FakePlotter(FakeInteractor())
    mol = make_mol(NUMS, BONDS, COORDS)
    ctx = FakeContext(mol=mol, plotter=plotter)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.metals_only.setChecked(False)
    win.table._set_selected_rows([0])
    win.table._set_selected_rows([])
    assert "pmeff_geom_selection" in plotter.removed


def test_highlight_selected_missing_pyvista_is_swallowed(god_mod):
    qt_stubs.remove_viz_stubs()  # pyvista now unimportable (blocked by neither install)
    qt_stubs.block_module("pyvista")
    try:
        plotter = FakePlotter(FakeInteractor())
        mol = make_mol(NUMS, BONDS, COORDS)
        ctx = FakeContext(mol=mol, plotter=plotter)
        win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
        win.metals_only.setChecked(False)
        win.table._set_selected_rows([0])  # must not raise
        assert plotter.added == []
    finally:
        qt_stubs.unblock_module("pyvista")
        qt_stubs.install_viz_stubs()


# --------------------------------------------------------------------------
# closeEvent
# --------------------------------------------------------------------------
def test_close_event_reverts_uncommitted_edits(god_mod):
    plotter = FakePlotter(FakeInteractor())
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=plotter)
    win = god_mod._OverrideWindow(ctx, {0: "tetrahedral"}, on_apply=None)
    win._overrides[0] = "linear"  # unsaved edit
    win._dirty.add(0)
    win.close()
    assert win._overrides == {0: "tetrahedral"}
    assert win._dirty == set()
    assert plotter.removed == ["pmeff_geom_selection"]
    assert plotter.render_calls >= 1


def test_close_event_without_plotter_is_safe(god_mod):
    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=None)
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.close()  # must not raise


def test_close_event_swallows_plotter_exception(god_mod):
    class BadPlotter(FakePlotter):
        def remove_actor(self, name):
            raise RuntimeError("boom")

    ctx = FakeContext(mol=make_mol(NUMS, BONDS, COORDS), plotter=BadPlotter(FakeInteractor()))
    win = god_mod._OverrideWindow(ctx, {}, on_apply=None)
    win.close()  # must not raise


# --------------------------------------------------------------------------
# _main_window helper
# --------------------------------------------------------------------------
def test_main_window_prefers_get_main_window(god_mod):
    class Ctx:
        def get_main_window(self):
            return "mw"

    assert god_mod._main_window(Ctx()) == "mw"


def test_main_window_falls_back_to_attribute(god_mod):
    class Ctx:
        main_window = "mw2"

    assert god_mod._main_window(Ctx()) == "mw2"


# --------------------------------------------------------------------------
# _ClickFilter
# --------------------------------------------------------------------------
def test_click_filter_triggers_callback_on_click(god_mod):
    calls = []
    cf = god_mod._ClickFilter(lambda x, y, obj, mods: calls.append((x, y, mods)))
    left = god_mod.Qt.MouseButton.LeftButton
    press = FakeEvent(god_mod.QEvent.Type.MouseButtonPress, button=left, xy=(10, 10))
    release = FakeEvent(god_mod.QEvent.Type.MouseButtonRelease, button=left, xy=(11, 11))
    assert cf.eventFilter(None, press) is False
    assert cf.eventFilter(None, release) is False
    assert calls == [(11, 11, 0)]


def test_click_filter_ignores_drag(god_mod):
    calls = []
    cf = god_mod._ClickFilter(lambda x, y, obj, mods: calls.append((x, y, mods)))
    left = god_mod.Qt.MouseButton.LeftButton
    press = FakeEvent(god_mod.QEvent.Type.MouseButtonPress, button=left, xy=(0, 0))
    release = FakeEvent(god_mod.QEvent.Type.MouseButtonRelease, button=left, xy=(50, 50))
    cf.eventFilter(None, press)
    cf.eventFilter(None, release)
    assert calls == []


def test_click_filter_ignores_non_left_button(god_mod):
    calls = []
    cf = god_mod._ClickFilter(lambda x, y, obj, mods: calls.append((x, y, mods)))
    press = FakeEvent(god_mod.QEvent.Type.MouseButtonPress, button=99, xy=(0, 0))
    cf.eventFilter(None, press)
    assert cf._press_pos is None


def test_click_filter_release_without_prior_press_is_noop(god_mod):
    calls = []
    cf = god_mod._ClickFilter(lambda x, y, obj, mods: calls.append((x, y, mods)))
    left = god_mod.Qt.MouseButton.LeftButton
    release = FakeEvent(god_mod.QEvent.Type.MouseButtonRelease, button=left, xy=(1, 1))
    cf.eventFilter(None, release)
    assert calls == []
