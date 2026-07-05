"""Tests for the standalone ``pmeff`` PyPI distribution.

The ``pmeff`` package is derived from the plugin: ``pmeff/forcefield.py`` is a
verbatim copy of ``pmeff_plugin/forcefield.py`` (produced by
``scripts/sync_forcefield.py``) and ``pmeff/_version.py`` is stamped from
``PLUGIN_VERSION``. These tests check the public API surface (``optimize_mol``,
``optimize_coords``, version, exports) and — importantly — that the copied
engine and version have not drifted from their single source of truth.
"""

from __future__ import annotations

import math
import pathlib
import re

import numpy as np
import pytest

import pmeff

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SYNC_HEADER_LINES = 5  # 4-line banner + 1 blank, see scripts/sync_forcefield.py


# --- Packaging integrity ----------------------------------------------------


def test_copied_engine_matches_source():
    # The committed pmeff/forcefield.py must be an exact copy of the canonical
    # pmeff_plugin/forcefield.py (modulo the generated header). If this fails,
    # someone edited the source without running scripts/sync_forcefield.py.
    source = (ROOT / "pmeff_plugin" / "forcefield.py").read_text(encoding="utf-8")
    copied = (ROOT / "pmeff" / "forcefield.py").read_text(encoding="utf-8")
    body = "".join(copied.splitlines(keepends=True)[_SYNC_HEADER_LINES:])
    assert body == source, (
        "pmeff/forcefield.py is stale — run scripts/sync_forcefield.py"
    )


def test_version_matches_plugin_version():
    plugin_init = (ROOT / "pmeff_plugin" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'PLUGIN_VERSION\s*=\s*["\']([^"\']+)["\']', plugin_init)
    assert match is not None
    assert pmeff.__version__ == match.group(1)


def test_public_api_is_exported():
    for name in (
        "optimize_mol",
        "optimize_coords",
        "optimize",
        "build_topology",
        "energy_and_gradient",
        "energy_components",
        "vibrational_analysis",
        "qeq_charges",
        "bond_rest_length",
        "Topology",
        "OptimizeResult",
    ):
        assert hasattr(pmeff, name), f"pmeff.{name} missing from public API"
        assert name in pmeff.__all__


# --- Pure-NumPy path (no RDKit) ---------------------------------------------


def test_optimize_coords_relaxes_water_angle():
    coords = np.array([[0.0, 0, 0], [1.3, 0, 0], [-0.4, 1.2, 0.0]])
    out, result = pmeff.optimize_coords(
        [8, 1, 1], [(0, 1), (0, 2)], coords, hybridizations=["SP3", None, None]
    )
    assert result.converged
    v1, v2 = out[1] - out[0], out[2] - out[0]
    ang = math.degrees(
        math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    )
    assert ang == pytest.approx(104.5, abs=1.5)  # lone-pair-compressed sp3 O


def test_optimize_coords_does_not_mutate_input():
    coords = np.array([[0.0, 0, 0], [1.3, 0, 0]])
    before = coords.copy()
    pmeff.optimize_coords([6, 6], [(0, 1)], coords)
    assert np.array_equal(coords, before)


def test_optimize_coords_with_qeq_charges_runs():
    coords = np.array([[0.0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]])
    charges = pmeff.qeq_charges([8, 6, 7], coords, total_charge=0.0)
    out, result = pmeff.optimize_coords(
        [8, 6, 7], [(0, 1), (1, 2)], coords, charges=charges
    )
    assert np.all(np.isfinite(out))
    assert math.isfinite(result.energy)


# --- RDKit convenience path -------------------------------------------------


def test_optimize_mol_returns_same_object_relaxed():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("O[SiH3]"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    returned, result = pmeff.optimize_mol(mol)
    assert returned is mol  # same object, conformer updated
    assert result.converged
    conf = returned.GetConformer()
    si_o = np.linalg.norm(
        np.array(conf.GetAtomPosition(0)) - np.array(conf.GetAtomPosition(1))
    )
    assert si_o == pytest.approx(1.63, abs=0.03)  # polar Si-O contraction active


def test_optimize_mol_without_conformer_raises():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")  # 2D only, no conformer
    with pytest.raises(ValueError):
        pmeff.optimize_mol(mol)
