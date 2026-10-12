"""Cooperative cancellation must leave an RDKit conformer untouched."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pmeff_plugin import forcefield as ff


def test_progress_counts_across_optimizer_phases():
    coords = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    topo = ff.build_topology([6, 6], [(0, 1)])
    updates = []
    out, result = ff.optimize(
        coords, topo, max_iter=500, progress_callback=updates.append
    )
    assert result.converged
    assert updates[0].step == 0
    assert updates[-1].phase == "Done"
    assert updates[-1].step == result.steps
    assert updates[-1].energy == pytest.approx(result.energy)
    assert updates[-1].max_force == pytest.approx(result.max_force)
    assert {u.phase for u in updates} >= {"FIRE", "L-BFGS", "Done"}
    assert all(a.step <= b.step for a, b in zip(updates, updates[1:]))
    assert all(u.step <= u.max_iter for u in updates)
    reference, _ = ff.optimize(coords, ff.build_topology([6, 6], [(0, 1)]))
    np.testing.assert_allclose(out, reference)


def test_cancelled_run_does_not_write_partial_coordinates():
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    before = mol.GetConformer().GetPositions().copy()
    updates = []
    with pytest.raises(ff.OptimizationCancelled):
        ff.optimize_rdkit_mol(
            mol,
            progress_callback=updates.append,
            cancel_callback=lambda: bool(updates and updates[-1].step >= 3),
        )
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), before)
    assert updates[-1].step == 3


def test_cancel_before_initial_energy_evaluation(monkeypatch):
    def forbidden(*_args):
        pytest.fail("energy evaluation happened after cancellation")

    monkeypatch.setattr(ff, "energy_and_gradient", forbidden)
    with pytest.raises(ff.OptimizationCancelled):
        ff.optimize(
            np.zeros((0, 3)), ff.build_topology([], []), cancel_callback=lambda: True
        )


def test_vibrational_progress_and_cancellation():
    coords = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    topo = ff.build_topology([6, 6], [(0, 1)])
    updates = []
    with pytest.raises(ff.OptimizationCancelled):
        ff.hessian(
            coords,
            topo,
            progress_callback=lambda i, n: updates.append((i, n)),
            cancel_callback=lambda: len(updates) >= 2,
        )
    assert updates == [(1, 6), (2, 6)]
    updates.clear()
    result = ff.vibrational_analysis(
        coords, topo, progress_callback=lambda i, n: updates.append((i, n))
    )
    assert updates[-1] == (6, 6)
    assert len(result["eigenvalues"]) == 6
    assert result["max_force"] > 0  # Positive curvature alone is not stationarity.
