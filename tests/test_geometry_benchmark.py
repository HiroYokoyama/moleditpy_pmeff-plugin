"""Geometry-accuracy regression benchmark for the PMEFF engine.

Unlike the unit tests, which pin individual hand-picked parameters, this
module embeds real molecules with RDKit, relaxes them with the full PMEFF
force field (all physics terms on, as the plugin ships), and checks the
*optimized* geometry against reference bond lengths and angles. It is the
guardrail that catches a parameter change silently regressing real geometries
even when every targeted unit test stays green.

Thresholds are deliberately loose (≈0.07 Å on bonds, ≈4.5° on angles): PMEFF
is a pre-DFT cleanup field, not a high-accuracy method, and carries a known
systematic ~0.04 Å underestimate on C–C from the Pyykko radii. The point is to
catch *gross* regressions (e.g. the 0.16 Å Si–O error the polar-bond
contraction fixed), not to assert QM accuracy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pmeff_plugin import forcefield as ff

# Full physics stack, matching the plugin's shipped defaults.
_KW = dict(
    electronic_effects=True,
    use_morse=True,
    use_hbond=True,
    use_polar_contraction=True,
)

# Per-bond tolerance (Å) and aggregate RMS ceiling (Å). Chosen with margin
# over the measured errors so the suite is not flaky, yet tight enough that a
# ~0.1 Å regression trips it.
_BOND_TOL_A = 0.07
_BOND_RMS_CEILING_A = 0.05
_ANGLE_TOL_DEG = 4.5

# (name, SMILES, [(atom_i, atom_j, reference_length_A)]). Atom indices follow
# RDKit's ordering after AddHs (heavy atoms in SMILES order, then hydrogens).
_BOND_CASES = [
    ("ethane",         "CC",        [(0, 1, 1.54)]),
    ("ethylene",       "C=C",       [(0, 1, 1.34)]),
    ("acetylene",      "C#C",       [(0, 1, 1.20)]),
    ("methanol",       "CO",        [(0, 1, 1.43)]),
    ("methylamine",    "CN",        [(0, 1, 1.47)]),
    ("carbon dioxide", "O=C=O",     [(0, 1, 1.16)]),
    ("hydrogen cyanide", "C#N",     [(0, 1, 1.16)]),
    ("fluoromethane",  "CF",        [(0, 1, 1.38)]),
    ("chloromethane",  "CCl",       [(0, 1, 1.78)]),
    ("bromomethane",   "CBr",       [(0, 1, 1.94)]),
    ("methanethiol",   "CS",        [(0, 1, 1.82)]),
    ("benzene",        "c1ccccc1",  [(0, 1, 1.39)]),
    ("silanol",        "O[SiH3]",   [(0, 1, 1.63)]),  # polar Si-O
]

# (name, SMILES, (atom_i, vertex_j, atom_k), reference_angle_deg).
_ANGLE_CASES = [
    ("water H-O-H",         "O",     (1, 0, 2), 104.5),
    ("ammonia H-N-H",       "N",     (1, 0, 2), 106.7),
    ("methane H-C-H",       "C",     (1, 0, 2), 109.5),
    ("carbon dioxide O-C-O", "O=C=O", (0, 1, 2), 180.0),
]


def _optimized_positions(smiles: str) -> tuple:
    """Embed *smiles*, relax it with full-physics PMEFF, return (mol, coords)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0, smiles
    success, result = ff.optimize_rdkit_mol(mol, max_iter=1000, **_KW)
    assert success is True
    conf = mol.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )
    return mol, coords, result


def _bond_length(coords: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(coords[i] - coords[j]))


def _angle_deg(coords: np.ndarray, i: int, j: int, k: int) -> float:
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


@pytest.mark.parametrize("name,smiles,refs", _BOND_CASES, ids=[c[0] for c in _BOND_CASES])
def test_bond_length_accuracy(name, smiles, refs):
    _mol, coords, _res = _optimized_positions(smiles)
    for i, j, ref in refs:
        got = _bond_length(coords, i, j)
        assert abs(got - ref) < _BOND_TOL_A, (
            f"{name}: bond {i}-{j} = {got:.3f} A, expected {ref:.3f} "
            f"(|error| {abs(got - ref):.3f} >= {_BOND_TOL_A})"
        )


def test_bond_length_aggregate_rms():
    # A single RMS over every reference bond: guards the *fleet* accuracy so a
    # broad small drift (each bond just under the per-bond tol) still trips.
    errors = []
    for _name, smiles, refs in _BOND_CASES:
        _mol, coords, _res = _optimized_positions(smiles)
        errors.extend(_bond_length(coords, i, j) - ref for i, j, ref in refs)
    rms = float(np.sqrt(np.mean(np.square(errors))))
    assert rms < _BOND_RMS_CEILING_A, f"bond-length RMS {rms:.4f} A too large"


@pytest.mark.parametrize(
    "name,smiles,ijk,ref", _ANGLE_CASES, ids=[c[0] for c in _ANGLE_CASES]
)
def test_angle_accuracy(name, smiles, ijk, ref):
    _mol, coords, _res = _optimized_positions(smiles)
    got = _angle_deg(coords, *ijk)
    assert abs(got - ref) < _ANGLE_TOL_DEG, (
        f"{name}: angle = {got:.1f} deg, expected {ref:.1f} "
        f"(|error| {abs(got - ref):.1f} >= {_ANGLE_TOL_DEG})"
    )


@pytest.mark.parametrize(
    "smiles", [c[1] for c in _BOND_CASES], ids=[c[0] for c in _BOND_CASES]
)
def test_benchmark_molecules_converge(smiles):
    # Every benchmark molecule must reach the force tolerance within budget.
    _mol, _coords, result = _optimized_positions(smiles)
    assert result is None or result.converged, f"{smiles} did not converge"


@pytest.mark.parametrize(
    "smiles",
    ["CC", "CO", "CN", "c1ccccc1", "O[SiH3]", "O", "CCO"],
)
def test_optimizer_reaches_topology_rest_lengths(smiles):
    # Self-consistency: an unstrained small molecule should relax so every
    # bond sits close to the rest length its own topology prescribes. This
    # separates optimizer failures from parameter choices — it holds no matter
    # what the r0 values are.
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    topo = ff.topology_from_rdkit(mol, **_KW)
    coords = np.array(
        [
            list(mol.GetConformer().GetAtomPosition(i))
            for i in range(mol.GetNumAtoms())
        ]
    )
    opt_coords, result = ff.optimize(coords, topo, max_iter=1000)
    assert result.converged
    for i, j, r0, _k in topo.bonds:
        got = _bond_length(opt_coords, i, j)
        # Competing angle/vdW terms perturb r0 slightly; 0.05 A is generous
        # for these unstrained cases while still catching a broken bond term.
        assert abs(got - r0) < 0.05, (
            f"{smiles}: bond {i}-{j} = {got:.3f}, rest length {r0:.3f}"
        )
