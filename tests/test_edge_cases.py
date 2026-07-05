"""Robustness and edge-case tests for the PMEFF engine.

Covers degenerate inputs (empty, single atom, coincident atoms, disconnected
fragments), physically special cases (charged species, isolated ions, large
floppy chains, rings), and invariants that must hold for *any* topology
(energy translational/rotational invariance, gradient consistency, finite
output). These guard the engine against crashes and silent NaNs on the messy
molecules a structure-cleanup tool actually receives.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pmeff_plugin import forcefield as ff


# --- Degenerate / trivial inputs -------------------------------------------


def test_empty_topology_zero_energy():
    topo = ff.build_topology([], [])
    e, g = ff.energy_and_gradient(np.zeros((0, 3)), topo)
    assert e == 0.0
    assert g.shape == (0, 3)


def test_single_atom_zero_energy_and_gradient():
    topo = ff.build_topology([6], [])
    e, g = ff.energy_and_gradient(np.array([[1.0, 2.0, 3.0]]), topo)
    assert e == 0.0
    assert np.allclose(g, 0.0)


def test_single_atom_qeq_charge_equals_total():
    # One atom carries the whole molecular charge.
    q = ff.qeq_charges([8], np.array([[0.0, 0.0, 0.0]]), total_charge=-1.0)
    assert q == pytest.approx([-1.0])


def test_qeq_conserves_total_charge():
    coords = np.array([[0.0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    q = ff.qeq_charges([8, 6, 7], coords, total_charge=1.0)
    assert float(np.sum(q)) == pytest.approx(1.0, abs=1e-9)


def test_coincident_atoms_do_not_crash():
    # Two atoms on top of each other: the bond term guards r->0, the result
    # must stay finite (no divide-by-zero NaN escaping).
    topo = ff.build_topology([6, 6], [(0, 1)])
    e, g = ff.energy_and_gradient(np.zeros((2, 3)), topo)
    assert math.isfinite(e)
    assert np.all(np.isfinite(g))


def test_disconnected_fragments_energy_is_additive_at_distance():
    # Two far-apart H2 molecules: total energy ~ sum of the isolated pair
    # energies (non-bonded interaction negligible at large separation).
    pair = ff.build_topology([1, 1], [(0, 1)])
    r0 = pair.bonds[0][2]
    c_pair = np.array([[0.0, 0, 0], [r0 + 0.3, 0, 0]])
    e_pair, _ = ff.energy_and_gradient(c_pair, pair)

    two = ff.build_topology([1, 1, 1, 1], [(0, 1), (2, 3)])
    c_two = np.array(
        [[0.0, 0, 0], [r0 + 0.3, 0, 0], [0.0, 0, 100.0], [r0 + 0.3, 0, 100.0]]
    )
    e_two, _ = ff.energy_and_gradient(c_two, two)
    assert e_two == pytest.approx(2.0 * e_pair, abs=1e-6)


# --- Invariants that must hold for any topology ----------------------------


def _random_molecule_topo(seed: int = 0):
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1O"))  # a heteroatom mix
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    coords = np.array(
        [list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )
    return ff.topology_from_rdkit(mol, electronic_effects=True), coords


def test_energy_is_translation_invariant():
    topo, coords = _random_molecule_topo()
    e0, _ = ff.energy_and_gradient(coords, topo)
    e1, _ = ff.energy_and_gradient(coords + np.array([3.1, -2.2, 0.7]), topo)
    assert e1 == pytest.approx(e0, rel=1e-9, abs=1e-6)


def test_energy_is_rotation_invariant():
    topo, coords = _random_molecule_topo()
    e0, _ = ff.energy_and_gradient(coords, topo)
    theta = 0.9
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    e1, _ = ff.energy_and_gradient(coords @ rot.T, topo)
    assert e1 == pytest.approx(e0, rel=1e-9, abs=1e-6)


def test_gradient_sums_to_zero():
    # Internal forces are self-balancing: the net force on the molecule is 0.
    topo, coords = _random_molecule_topo()
    _e, g = ff.energy_and_gradient(coords, topo)
    assert np.allclose(np.sum(g, axis=0), 0.0, atol=1e-7)


def test_full_gradient_matches_numeric_on_real_molecule():
    # A finite-difference check of the *entire* assembled force field on a
    # realistic molecule (all terms active at once), not just per-term.
    topo, coords = _random_molecule_topo(seed=3)
    _e, g = ff.energy_and_gradient(coords, topo)
    step = 1e-5
    flat = coords.ravel()
    num = np.zeros_like(flat)
    for a in range(flat.size):
        up = flat.copy()
        up[a] += step
        dn = flat.copy()
        dn[a] -= step
        e_up, _ = ff.energy_and_gradient(up.reshape(coords.shape), topo)
        e_dn, _ = ff.energy_and_gradient(dn.reshape(coords.shape), topo)
        num[a] = (e_up - e_dn) / (2.0 * step)
    assert np.allclose(g.ravel(), num, atol=1e-3)


# --- Charged and unusual species -------------------------------------------


def test_ammonium_stays_tetrahedral():
    # NH4+ has no lone pair (CN 4) -> angles stay ~109.5, not compressed.
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ff.optimize_rdkit_mol(mol, max_iter=800, electronic_effects=True)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    hs = [a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors()]
    v1 = coords[hs[0]] - coords[0]
    v2 = coords[hs[1]] - coords[0]
    ang = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1
                )
            )
        )
    )
    assert ang == pytest.approx(109.5, abs=3.0)


def test_isolated_monatomic_ion_optimizes_trivially():
    mol = Chem.MolFromSmiles("[Na+]")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    success, result = ff.optimize_rdkit_mol(mol, electronic_effects=True)
    assert success is True
    assert result is None  # < 2 atoms: nothing to do


def test_optimize_does_not_produce_nan_on_clashed_start():
    # Overlapping initial coordinates must relax to a finite geometry, never
    # NaN/inf (the FIRE clamp + Morse bound are what keep this stable).
    topo = ff.build_topology([6, 6, 6], [(0, 1), (1, 2)], ["SP3", "SP3", "SP3"])
    coords = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]])
    out, result = ff.optimize(coords, topo, max_iter=1000)
    assert np.all(np.isfinite(out))
    assert math.isfinite(result.energy)


# --- Structural correctness on optimized geometries ------------------------


def test_benzene_ring_stays_planar():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ff.optimize_rdkit_mol(mol, max_iter=1000, use_morse=True)
    conf = mol.GetConformer()
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    pts = np.array([list(conf.GetAtomPosition(i)) for i in ring])
    centroid = pts.mean(axis=0)
    # Best-fit plane normal via SVD; max out-of-plane deviation must be tiny.
    _u, _s, vt = np.linalg.svd(pts - centroid)
    normal = vt[2]
    deviations = np.abs((pts - centroid) @ normal)
    assert float(np.max(deviations)) < 0.05


def test_linear_molecule_has_five_zero_modes():
    # CO2 is linear: 3 translations + 2 rotations = 5 rigid-body modes
    # (rotation about the molecular axis is absent).
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C=O"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ff.optimize_rdkit_mol(mol, max_iter=1000)
    result = ff.check_minimum(mol)
    assert result["num_imaginary"] == 0
    assert result["num_zero"] >= 5


def test_optimized_water_is_a_true_minimum():
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ff.optimize_rdkit_mol(mol, max_iter=1000)
    result = ff.check_minimum(mol)
    assert result["is_minimum"] is True
    assert result["num_imaginary"] == 0
