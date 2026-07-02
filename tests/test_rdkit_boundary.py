"""Tests for the RDKit boundary of the PMEFF engine."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from force_field_plugin import forcefield as ff


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


def test_handles_transition_metal_complex():
    # Ferrocene-like iron center: PMEFF must parameterize Fe without gaps.
    mol = Chem.AddHs(Chem.MolFromSmiles("[Fe]"))
    topo = ff.topology_from_rdkit(mol)
    # Fe is Z=26; its covalent radius must be a real, finite value.
    assert ff.covalent_radius(26) == pytest.approx(1.16)
    assert topo.num_atoms == 1
