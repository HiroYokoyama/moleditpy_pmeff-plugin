"""Tests for per-atom coordination-geometry overrides (v1.1.0 feature).

Covers the engine (``build_topology`` angle/out-of-plane assignment and name
normalization), the RDKit and pip-package pass-through layers, and the pure
helpers of the override dialog. The GUI window itself is only smoke-checked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pmeff_plugin import forcefield as ff
from pmeff_plugin.forcefield import build_topology, _normalize_geometry


def _angles_deg(topo):
    return sorted(round(math.degrees(a[3]), 2) for a in topo.angles)


# A central atom (Fe) with four ligands arranged squarely in a plane, so the
# coordinate-based cis/trans assignment has a well-defined answer.
_NUMS4 = [26, 9, 9, 9, 9]
_BONDS4 = [(0, 1), (0, 2), (0, 3), (0, 4)]
_COORDS4 = np.array(
    [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0]], dtype=float
)


def _topo4(override):
    return build_topology(
        _NUMS4, _BONDS4, coords=_COORDS4, geometry_overrides=override
    )


# --------------------------------------------------------------------------
# Fixed-angle geometries
# --------------------------------------------------------------------------
def test_linear_override_sets_180():
    assert set(_angles_deg(_topo4({0: "linear"}))) == {180.0}


def test_tetrahedral_override_sets_10947():
    assert set(_angles_deg(_topo4({0: "tetrahedral"}))) == {109.47}


def test_trigonal_planar_override_sets_120_and_planarity():
    nums = [26, 9, 9, 9]
    bonds = [(0, 1), (0, 2), (0, 3)]
    coords = np.array(
        [[0, 0, 0], [2, 0, 0], [-1, 1.7, 0], [-1, -1.7, 0]], dtype=float
    )
    topo = build_topology(
        nums, bonds, coords=coords, geometry_overrides={0: "trigonal_planar"}
    )
    assert set(_angles_deg(topo)) == {120.0}
    # A trigonal-planar override adds an out-of-plane (planarity) term.
    assert len(topo.oops) == 1


# --------------------------------------------------------------------------
# Cis/trans geometries
# --------------------------------------------------------------------------
def test_square_planar_override_cis_trans():
    # 4 in-plane ligands: 4 cis (90) + 2 trans (180) = C(4,2) = 6 angles.
    assert _angles_deg(_topo4({0: "square_planar"})) == [90.0, 90.0, 90.0, 90.0, 180.0, 180.0]


def test_octahedral_override_cis_trans():
    nums = [26] + [9] * 6
    bonds = [(0, i) for i in range(1, 7)]
    coords = np.array(
        [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2]],
        dtype=float,
    )
    topo = build_topology(
        nums, bonds, coords=coords, geometry_overrides={0: "octahedral"}
    )
    from collections import Counter

    hist = Counter(round(math.degrees(a[3])) for a in topo.angles)
    assert hist == {90: 12, 180: 3}


# --------------------------------------------------------------------------
# Name normalization & robustness
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("Square Planar", "square_planar"),
        ("square-planar", "square_planar"),
        ("  TETRAHEDRAL ", "tetrahedral"),
        ("Trigonal-Planar", "trigonal_planar"),
        ("banana", None),
        (None, None),
        (123, None),
    ],
)
def test_normalize_geometry(name, expected):
    assert _normalize_geometry(name) == expected


def test_unknown_name_is_ignored():
    # Unknown geometry falls back to the default (auto) angles, unchanged.
    assert _angles_deg(_topo4({0: "banana"})) == _angles_deg(_topo4(None))


def test_out_of_range_index_is_ignored():
    assert _angles_deg(_topo4({99: "linear"})) == _angles_deg(_topo4(None))


# --------------------------------------------------------------------------
# Default behavior is untouched
# --------------------------------------------------------------------------
def test_no_overrides_matches_none():
    assert _angles_deg(_topo4({})) == _angles_deg(_topo4(None))


def test_default_ch4_geometry_unchanged():
    nums = [6, 1, 1, 1, 1]
    topo = build_topology(nums, _BONDS4, ["SP3", "", "", "", ""], coords=_COORDS4)
    assert set(_angles_deg(topo)) == {109.47}


def test_override_removes_sp2_planarity():
    # An sp2 center that would normally get an out-of-plane term loses it when
    # forced to a non-planar geometry.
    nums = [6, 8, 8, 8]  # carbonate-like, 3-coordinate sp2 carbon
    bonds = [(0, 1), (0, 2), (0, 3)]
    coords = np.array(
        [[0, 0, 0], [1.3, 0, 0], [-0.65, 1.1, 0], [-0.65, -1.1, 0]], dtype=float
    )
    planar = build_topology(nums, bonds, ["SP2", "", "", ""], coords=coords)
    assert len(planar.oops) == 1
    forced = build_topology(
        nums, bonds, ["SP2", "", "", ""], coords=coords,
        geometry_overrides={0: "tetrahedral"},
    )
    assert len(forced.oops) == 0
    assert set(_angles_deg(forced)) == {109.47}


# --------------------------------------------------------------------------
# RDKit + pip-package pass-through
# --------------------------------------------------------------------------
def test_optimize_coords_accepts_overrides():
    from pmeff import optimize_coords

    coords, result = optimize_coords(
        _NUMS4, _BONDS4, _COORDS4, geometry_overrides={0: "square_planar"}
    )
    assert coords.shape == (5, 3)
    assert np.all(np.isfinite(coords))


def test_overrides_apply_with_electronic_effects_disabled():
    # Overrides are independent of the electronic-effects flag: forcing a
    # geometry works even for a plain carbon with electronic effects OFF.
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pmeff_plugin.forcefield import topology_from_rdkit

    mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # methane
    AllChem.EmbedMolecule(mol, randomSeed=3)
    topo = topology_from_rdkit(
        mol, electronic_effects=False, geometry_overrides={0: "square_planar"}
    )
    # Square-planar carbon: 4 cis (90) + 2 trans (180), even with e-effects off.
    assert _angles_deg(topo) == [90.0, 90.0, 90.0, 90.0, 180.0, 180.0]


def test_optimize_rdkit_mol_threads_overrides(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)

    captured = {}
    real_build = ff.build_topology

    def spy(*args, **kwargs):
        captured["geometry_overrides"] = kwargs.get("geometry_overrides")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(ff, "build_topology", spy)
    ok, _ = ff.optimize_rdkit_mol(mol, max_iter=20, geometry_overrides={1: "tetrahedral"})
    assert ok
    assert captured["geometry_overrides"] == {1: "tetrahedral"}


# --------------------------------------------------------------------------
# Override-dialog pure helpers
# --------------------------------------------------------------------------
def test_dialog_geometry_choices_match_engine():
    from pmeff_plugin.geometry_override_dialog import GEOMETRY_CHOICES

    keys = {key for _label, key in GEOMETRY_CHOICES if key is not None}
    assert keys == set(ff._VALID_GEOMETRIES)
    # "Auto" must be present and map to None.
    assert GEOMETRY_CHOICES[0][1] is None


@pytest.mark.parametrize(
    "z,metal",
    [(26, True), (78, True), (3, True), (13, True), (6, False), (8, False),
     (1, False), (14, False), (17, False), (2, False)],
)
def test_is_metal(z, metal):
    from pmeff_plugin.geometry_override_dialog import is_metal

    assert is_metal(z) is metal
