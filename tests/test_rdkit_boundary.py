"""Tests for the RDKit boundary of the PMEFF engine."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pmeff_plugin import forcefield as ff


def _embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def test_topology_from_rdkit_ethanol():
    mol = _embed("CCO")
    topo = ff.topology_from_rdkit(mol)
    assert topo.num_atoms == mol.GetNumAtoms()
    assert len(topo.bonds) == mol.GetNumBonds()
    assert len(topo.angles) > 0


def test_compute_energy_returns_float():
    mol = _embed("CCO")
    energy = ff.compute_energy(mol)
    assert isinstance(energy, float)


def test_compute_energy_none_without_conformer():
    mol = Chem.MolFromSmiles("CCO")  # 2D-only, no conformer
    assert ff.compute_energy(mol) is None


def test_optimize_rdkit_mol_modifies_coordinates_in_place():
    mol = _embed("CCO")
    before = np.array(mol.GetConformer().GetPositions())
    success, result = ff.optimize_rdkit_mol(mol, max_iter=300)
    after = np.array(mol.GetConformer().GetPositions())
    assert success is True
    assert result is not None
    assert not np.allclose(before, after)


def test_optimize_rdkit_mol_lowers_energy():
    mol = _embed("CCO")
    e_before = ff.compute_energy(mol)
    ff.optimize_rdkit_mol(mol, max_iter=500)
    e_after = ff.compute_energy(mol)
    assert e_after <= e_before + 1e-6


def test_optimize_single_atom_is_noop_success():
    mol = _embed("[Ne]")
    success, result = ff.optimize_rdkit_mol(mol)
    assert success is True
    assert result is None


def test_optimize_no_conformer_fails_gracefully():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))  # no embed -> no conformer
    success, result = ff.optimize_rdkit_mol(mol)
    assert success is False
    assert result is None


def test_optimized_benzene_is_planar_with_uniform_aromatic_bonds():
    mol = _embed("c1ccccc1")
    success, result = ff.optimize_rdkit_mol(mol, max_iter=2000)
    assert success and result.converged
    coords = np.array(mol.GetConformer().GetPositions())
    ring = coords[:6]
    centroid = ring.mean(axis=0)
    normal = np.linalg.svd(ring - centroid)[2][2]
    # Torsions + out-of-plane terms must keep the ring flat.
    assert np.abs((ring - centroid) @ normal).max() < 0.01
    # Aromatic C-C bonds: uniform and between single (1.50) and double (1.34).
    cc = [
        float(np.linalg.norm(coords[b.GetBeginAtomIdx()] - coords[b.GetEndAtomIdx()]))
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() == 6
        and b.GetEndAtom().GetAtomicNum() == 6
    ]
    assert max(cc) - min(cc) < 0.01
    assert 1.34 < min(cc) and max(cc) < 1.50


def test_rdkit_bond_orders_shorten_double_bond():
    # Ethylene C=C must get a shorter rest length than ethane C-C.
    ethane = ff.topology_from_rdkit(Chem.AddHs(Chem.MolFromSmiles("CC")))
    ethylene = ff.topology_from_rdkit(Chem.AddHs(Chem.MolFromSmiles("C=C")))

    def cc_rest(topo):
        return next(
            r0 for i, j, r0, _k in topo.bonds
            if topo.atomic_numbers[i] == 6 and topo.atomic_numbers[j] == 6
        )

    assert cc_rest(ethylene) < cc_rest(ethane)


def test_energy_evaluation_speed_on_medium_molecule():
    # Vectorized energetics: a ~60-atom molecule must evaluate in well under
    # 10 ms (the old per-term Python loops took far longer).
    import time

    mol = _embed("CCCCCCCCCCCCCCCCCCCC")
    topo = ff.topology_from_rdkit(mol)
    coords = np.array(mol.GetConformer().GetPositions())
    ff.energy_and_gradient(coords, topo)  # warm the compiled-array cache
    t0 = time.perf_counter()
    for _ in range(50):
        ff.energy_and_gradient(coords, topo)
    per_eval = (time.perf_counter() - t0) / 50
    assert per_eval < 0.010


def test_lone_pair_centers_keep_their_shape():
    # Lone pairs are not explicit, but their steric effect enters through
    # the hybridization-derived angle targets: sp3 N stays pyramidal
    # (no out-of-plane term is applied to sp3 centers), sp2 amide N stays
    # planar, and sp3 O stays bent.
    def optimized(smiles):
        mol = _embed(smiles)
        assert ff.optimize_rdkit_mol(mol, max_iter=3000)[0]
        return mol, np.array(mol.GetConformer().GetPositions())

    def height_above_neighbors(mol, coords, center):
        nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()]
        assert len(nbrs) == 3
        n = np.cross(
            coords[nbrs[1]] - coords[nbrs[0]], coords[nbrs[2]] - coords[nbrs[0]]
        )
        n /= np.linalg.norm(n)
        return abs(float(np.dot(coords[center] - coords[nbrs[0]], n)))

    mol, coords = optimized("N")  # ammonia: pyramidal
    assert height_above_neighbors(mol, coords, 0) > 0.25

    mol, coords = optimized("CC(=O)N")  # acetamide: planar amide N
    assert height_above_neighbors(mol, coords, 3) < 0.02

    mol, coords = optimized("COC")  # ether: bent, not linear
    v1 = coords[0] - coords[1]
    v2 = coords[2] - coords[1]
    angle = np.degrees(
        np.arccos(
            float(np.dot(v1, v2))
            / float(np.linalg.norm(v1) * np.linalg.norm(v2))
        )
    )
    assert 100.0 < angle < 120.0


def test_handles_transition_metal_complex():
    # Ferrocene-like iron center: PMEFF must parameterize Fe without gaps.
    mol = Chem.AddHs(Chem.MolFromSmiles("[Fe]"))
    topo = ff.topology_from_rdkit(mol)
    # Fe is Z=26; its covalent radius must be a real, finite value.
    assert ff.covalent_radius(26) == pytest.approx(1.16)
    assert topo.num_atoms == 1
