"""Tests for PMEFF's optional electronic-effects module.

Covers the derived electronic parameters (Slater Zeff, Allred-Rochow
electronegativity, hardness), QEq charges, the shielded Coulomb term, the
square-planar treatment of 4-coordinate d8 metals, and the settings.json
toggle in the plugin entry point.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import pmeff_plugin as plugin
from pmeff_plugin import forcefield as ff
from tests.conftest import make_context
from tests.test_forcefield import _numeric_gradient


# --- Derived electronic parameters -------------------------------------------


def test_slater_zeff_known_values():
    assert ff.slater_zeff(1) == pytest.approx(1.0)     # H: no screening
    assert ff.slater_zeff(6) == pytest.approx(3.25)    # C: 6 - 3*0.35 - 2*0.85
    assert ff.slater_zeff(9) == pytest.approx(5.20)    # F: 9 - 6*0.35 - 2*0.85


def test_electronegativity_trends():
    # Across a period: F > O > N > C; down a group: F > Cl; F >> alkali.
    assert (
        ff.electronegativity(9)
        > ff.electronegativity(8)
        > ff.electronegativity(7)
        > ff.electronegativity(6)
    )
    assert ff.electronegativity(9) > ff.electronegativity(17)
    assert ff.electronegativity(9) > 2 * ff.electronegativity(11)


def test_hardness_small_atoms_are_harder():
    assert ff.hardness(1) > ff.hardness(6) > ff.hardness(53)


# --- QEq charges --------------------------------------------------------------


def test_qeq_water_polarization_and_neutrality():
    # O at the vertex of a bent water geometry: O negative, H positive.
    coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    q = ff.qeq_charges([8, 1, 1], coords)
    assert q[0] < -0.05
    assert q[1] > 0.0 and q[2] > 0.0
    assert float(np.sum(q)) == pytest.approx(0.0, abs=1e-9)


def test_qeq_conserves_total_charge():
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    q = ff.qeq_charges([7, 6, 7], coords, total_charge=1.0)
    assert float(np.sum(q)) == pytest.approx(1.0, abs=1e-9)


def test_qeq_single_atom_gets_total_charge():
    q = ff.qeq_charges([26], np.zeros((1, 3)), total_charge=2.0)
    assert q[0] == pytest.approx(2.0)


# --- Coulomb term in the force field ------------------------------------------


def _fcf_topology_with_charges():
    # F-C-C-F: the (0,3) pair is 1-4, so it gets vdW + scaled Coulomb.
    atomic_numbers = [9, 6, 6, 9]
    coords = np.array(
        [[-1.4, 0.2, 0.0], [0.0, 0.0, 0.1], [1.5, 0.1, -0.1], [2.9, -0.2, 0.2]]
    )
    charges = ff.qeq_charges(atomic_numbers, coords)
    topo = ff.build_topology(
        atomic_numbers,
        [(0, 1), (1, 2), (2, 3)],
        ["SP3"] * 4,
        charges=charges,
    )
    return topo, coords


def test_elec_pairs_built_only_with_charges():
    topo, _coords = _fcf_topology_with_charges()
    assert len(topo.elec_pairs) == 1
    i, j, kqq, gamma = topo.elec_pairs[0]
    assert (i, j) == (0, 3)
    # Two like charges (both F negative) with the 1-4 scaling applied.
    assert kqq > 0.0
    assert gamma == pytest.approx(ff.covalent_radius(9))

    plain = ff.build_topology([9, 6, 6, 9], [(0, 1), (1, 2), (2, 3)], None)
    assert plain.elec_pairs == []


def test_gradient_matches_numeric_with_coulomb_term():
    topo, coords = _fcf_topology_with_charges()
    _, analytic = ff.energy_and_gradient(coords, topo)
    numeric = _numeric_gradient(coords, topo)
    assert np.allclose(analytic, numeric, atol=1e-4)


def test_like_charges_repel():
    topo, _coords = _fcf_topology_with_charges()
    _i, _j, kqq, gamma = topo.elec_pairs[0]
    close = kqq / math.sqrt(2.0**2 + gamma**2)
    far = kqq / math.sqrt(4.0**2 + gamma**2)
    assert close > far > 0.0


def test_hardness_scale_damps_polarization(monkeypatch):
    # The scaled hardness must shrink charge magnitudes (the anti-"skeleton
    # deformation" measure) while conserving the total charge exactly.
    coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    q_soft = ff.qeq_charges([8, 1, 1], coords)
    monkeypatch.setattr(ff, "_QEQ_HARDNESS_SCALE", 1.0)
    q_bare = ff.qeq_charges([8, 1, 1], coords)
    assert abs(q_soft[0]) < abs(q_bare[0])
    assert np.sign(q_soft[0]) == np.sign(q_bare[0])
    assert float(np.sum(q_soft)) == pytest.approx(0.0, abs=1e-9)


def test_refresh_qeq_charges_follows_geometry():
    topo, coords = _fcf_topology_with_charges()
    topo.qeq_total_charge = 0.0
    kqq_before = topo.elec_pairs[0][2]
    ff.energy_and_gradient(coords, topo)  # warm the compiled-array cache

    stretched = coords * 1.6
    ff.refresh_qeq_charges(topo, stretched)
    q_new = ff.qeq_charges([9, 6, 6, 9], stretched)
    expected = ff._K_COULOMB * q_new[0] * q_new[3] * ff._ELEC_14_SCALE
    i, j, kqq_after, _gamma = topo.elec_pairs[0]
    assert (i, j) == (0, 3)
    assert kqq_after == pytest.approx(expected)
    assert kqq_after != pytest.approx(kqq_before)
    # And the compiled cache must not serve the stale charges.
    e_ref, _ = ff.energy_and_gradient(stretched, topo)
    fresh = ff.build_topology(
        [9, 6, 6, 9], [(0, 1), (1, 2), (2, 3)], ["SP3"] * 4, charges=q_new
    )
    e_fresh, _ = ff.energy_and_gradient(stretched, fresh)
    assert e_ref == pytest.approx(e_fresh)


def test_refresh_qeq_charges_noop_for_static_topologies():
    topo, coords = _fcf_topology_with_charges()  # qeq_total_charge unset
    before = list(topo.elec_pairs)
    ff.refresh_qeq_charges(topo, coords * 2.0)
    assert topo.elec_pairs == before


def test_optimizer_resolves_dynamic_charges():
    # Two identical atoms given fake opposite charges: with dynamic QEq the
    # optimizer must re-solve on drift, discover q = 0 (equal
    # electronegativities), and drop the spurious attraction.
    atoms = [6, 6]
    start = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    topo = ff.build_topology(
        atoms, [], None, charges=[0.5, -0.5], coords=start, vdw_cutoff=12.0
    )
    topo.qeq_total_charge = 0.0
    assert len(topo.elec_pairs) == 1
    ff.optimize(start, topo, max_iter=200)
    assert topo.elec_pairs == []


# --- Square-planar d8 centers ---------------------------------------------------


def _ptcl4_topology(square_planar: bool):
    return ff.build_topology(
        [78, 17, 17, 17, 17],
        [(0, 1), (0, 2), (0, 3), (0, 4)],
        None,
        square_planar_metals=square_planar,
    )


def test_square_planar_sentinel_only_when_enabled():
    enabled = _ptcl4_topology(True)
    assert all(t0 == ff._SQ_PLANAR_T0 for *_ijk, t0 in enabled.angles)
    disabled = _ptcl4_topology(False)
    assert all(t0 > 0 for *_ijk, t0 in disabled.angles)


def test_flattened_tetrahedron_relaxes_to_square_planar():
    topo = _ptcl4_topology(True)
    r0 = topo.bonds[0][2]
    # Tetrahedral directions squashed along z: breaks the symmetry toward
    # planarity without starting anywhere near the answer.
    dirs = np.array(
        [[1, 1, 0.5], [1, -1, -0.5], [-1, 1, -0.5], [-1, -1, 0.5]], float
    )
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    coords = np.vstack([[0.0, 0.0, 0.0], r0 * dirs])
    out, result = ff.optimize(coords, topo, max_iter=3000)
    assert result.converged

    vecs = out[1:] - out[0]
    angles = []
    for a in range(4):
        for b in range(a + 1, 4):
            cos_t = float(
                np.dot(vecs[a], vecs[b])
                / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]))
            )
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos_t)))))
    angles.sort()
    # Square planar: four cis angles at 90 and two trans at 180.
    assert angles[:4] == pytest.approx([90.0] * 4, abs=3.0)
    assert angles[4:] == pytest.approx([180.0] * 2, abs=3.0)


# --- settings.json toggle in the plugin entry point ----------------------------


@pytest.fixture(name="settings_file")
def _settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(plugin, "_SETTINGS_FILE", path)
    return path


def test_electronic_effects_default_on(settings_file):
    assert not settings_file.exists()
    assert plugin.electronic_effects_enabled() is True


def test_save_load_roundtrip_all_settings_keys(settings_file):
    # save_settings / load_settings must persist and restore all known keys.
    expected = {
        "electronic_effects": False,
        "morse_bonds": False,
        "hbond": False,
        "dispersion": True,
        "polar_contraction": False,
    }
    plugin.save_settings(expected)
    loaded = plugin.load_settings()
    for key, val in expected.items():
        assert loaded[key] is val, f"mismatch for key {key!r}"
    assert json.loads(settings_file.read_text())["electronic_effects"] is False


def test_settings_menu_action_registered(settings_file):
    ctx = make_context()
    plugin.initialize(ctx)
    paths = [c.args[0] for c in ctx.add_menu_action.call_args_list]
    assert "Settings/PMEFF Setting" in paths


def test_corrupt_settings_file_falls_back_to_defaults(settings_file):
    settings_file.write_text("{not json")
    assert plugin.electronic_effects_enabled() is True


def test_optimizer_passes_electronic_effects_flag(settings_file, monkeypatch):
    ctx = make_context()
    plugin.initialize(ctx)
    # Default is on, so no toggle needed to exercise the enabled path.

    seen = {}

    def fake_optimize(mol, max_iter, **kwargs):
        seen.update(kwargs)
        return True, None

    monkeypatch.setattr(
        "pmeff_plugin.forcefield.optimize_rdkit_mol", fake_optimize
    )
    callback = ctx.register_optimization_method.call_args[0][1]
    assert callback(object()) is True
    assert seen["electronic_effects"] is True
    assert "use_morse" in seen
    assert "use_hbond" in seen
    assert "use_dispersion" in seen
    assert "use_polar_contraction" in seen
