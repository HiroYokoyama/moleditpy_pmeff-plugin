"""Angles and fresh-charge convergence on macrocycles and bridged cages."""

from itertools import combinations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pmeff_plugin import forcefield as ff

_OPTIONS = dict(electronic_effects=True, use_morse=True, use_hbond=True)
_CASES = [
    ("adamantane", "C1C2CC3CC1CC(C2)C3", 109.47, 3.0),
    ("cubane", "C12C3C4C1C5C2C3C45", 90.0, 3.0),
    ("cyclododecane", "C1CCCCCCCCCCC1", 111.0, 8.0),
    ("18-crown-6", "C1COCCOCCOCCOCCOCCO1", 110.0, 8.0),
]


@pytest.mark.parametrize(
    "name,smiles,target,tolerance", _CASES, ids=[c[0] for c in _CASES]
)
@pytest.mark.parametrize("perturb", [False, True], ids=["embedded", "perturbed"])
def test_macrocycle_and_cage_angles(name, smiles, target, tolerance, perturb):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    if perturb:
        conf = mol.GetConformer()
        positions = conf.GetPositions() + np.random.default_rng(7).normal(
            0, 0.08, (mol.GetNumAtoms(), 3)
        )
        for i, xyz in enumerate(positions):
            conf.SetAtomPosition(i, tuple(xyz))
    success, result = ff.optimize_rdkit_mol(mol, max_iter=2000, **_OPTIONS)
    assert success and result.converged, (name, result)
    coords = mol.GetConformer().GetPositions()
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        if atom.GetAtomicNum() == 1:
            continue
        neighbors = [a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum() > 1]
        for i, k in combinations(neighbors, 2):
            a, b = coords[i] - coords[j], coords[k] - coords[j]
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1
                    )
                )
            )
            assert abs(angle - target) < tolerance, (name, (i, j, k), angle)
    # Rebuilding the topology recomputes charges from the final coordinates.
    topo = ff.topology_from_rdkit(mol, **_OPTIONS)
    energy, gradient = ff.energy_and_gradient(coords, topo)
    fresh_force = float(np.max(np.linalg.norm(gradient, axis=1)))
    assert fresh_force < 1e-3, (name, fresh_force, result.max_force)
    assert result.max_force == pytest.approx(fresh_force, abs=1e-7)
    assert result.energy == pytest.approx(energy, abs=1e-7)
    for i, j, r0, _k in topo.bonds:
        assert 0.9 < np.linalg.norm(coords[i] - coords[j]) / r0 < 1.1


@pytest.mark.parametrize("angle", [0, 45, 89, 90, 91, 135, 180])
def test_hbond_direction_and_analytical_gradient(angle):
    theta = np.radians(angle)
    coords = np.array(
        [[-1.0, 0, 0], [0, 0, 0], [-1.9 * np.cos(theta), 1.9 * np.sin(theta), 0]]
    )
    topo = ff.Topology([7, 1, 8], hbond_triplets=[(0, 1, 2, 2.0, 1.9)])
    energy, grad = ff.energy_and_gradient(coords, topo)
    expected = -2.0 * min(np.cos(theta), 0.0) ** 2
    assert energy == pytest.approx(expected, abs=1e-10)
    step = 1e-5
    numerical = np.zeros_like(coords)
    for i in range(coords.size):
        up, down = coords.copy(), coords.copy()
        up.flat[i] += step
        down.flat[i] -= step
        numerical.flat[i] = (
            ff.energy_and_gradient(up, topo)[0] - ff.energy_and_gradient(down, topo)[0]
        ) / (2 * step)
    np.testing.assert_allclose(grad, numerical, atol=1e-5)
