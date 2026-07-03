"""Unit tests for the PMEFF engine (pmeff_plugin.forcefield)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pmeff_plugin import forcefield as ff


# --- Element parameter coverage --------------------------------------------


def test_covalent_radius_covers_whole_periodic_table():
    for z in range(1, 119):
        r = ff.covalent_radius(z)
        assert 0.2 < r < 2.6, f"implausible covalent radius for Z={z}: {r}"


def test_covalent_radius_known_values():
    assert ff.covalent_radius(1) == pytest.approx(0.32)   # H
    assert ff.covalent_radius(6) == pytest.approx(0.75)   # C
    assert ff.covalent_radius(8) == pytest.approx(0.63)   # O
    assert ff.covalent_radius(118) == pytest.approx(1.57)  # Og


def test_covalent_radius_fallback_for_dummy_atoms():
    assert ff.covalent_radius(0) == ff._DEFAULT_RADIUS_A
    assert ff.covalent_radius(200) == ff._DEFAULT_RADIUS_A


def test_vdw_radius_approximates_tabulated_values():
    # Derived vdW radii should land near the accepted tabulated values.
    assert ff.vdw_radius(1) == pytest.approx(1.20, abs=0.1)   # H ~1.20
    assert ff.vdw_radius(6) == pytest.approx(1.70, abs=0.1)   # C ~1.70
    assert ff.vdw_radius(8) == pytest.approx(1.52, abs=0.1)   # O ~1.52


def test_bond_order_factor_shortens_higher_orders():
    assert ff.bond_order_factor(1.0) == pytest.approx(1.0)
    assert ff.bond_order_factor(2.0) == pytest.approx(0.89)
    assert ff.bond_order_factor(3.0) == pytest.approx(0.78)
    # Aromatic order interpolates between single and double.
    assert 0.89 < ff.bond_order_factor(1.5) < 1.0
    # Out-of-range orders are clamped, never extrapolated.
    assert ff.bond_order_factor(0.5) == pytest.approx(1.0)
    assert ff.bond_order_factor(4.0) == pytest.approx(0.78)


# --- Topology construction --------------------------------------------------


def test_build_topology_water():
    # O(0) bonded to H(1) and H(2).
    topo = ff.build_topology(
        atomic_numbers=[8, 1, 1],
        bond_pairs=[(0, 1), (0, 2)],
        hybridizations=["SP3", None, None],
    )
    assert len(topo.bonds) == 2
    assert len(topo.angles) == 1          # H-O-H
    i, j, k, theta0 = topo.angles[0]
    assert j == 0                          # oxygen is the vertex
    assert math.degrees(theta0) == pytest.approx(109.47, abs=0.1)
    # The two H atoms are 1-3 to each other -> excluded from vdW.
    assert topo.vdw_pairs == []


def test_build_topology_deduplicates_bonds():
    topo = ff.build_topology([6, 6], [(0, 1), (1, 0)], None)
    assert len(topo.bonds) == 1


def test_build_topology_vdw_pairs_for_distant_atoms():
    # A linear 4-atom chain: atoms 0 and 3 are 1-4 -> a vdW pair.
    topo = ff.build_topology(
        [6, 6, 6, 6], [(0, 1), (1, 2), (2, 3)], None
    )
    pairs = {(i, j) for i, j, _, _ in topo.vdw_pairs}
    assert (0, 3) in pairs
    assert (0, 1) not in pairs   # bonded
    assert (0, 2) not in pairs   # 1-3


def _ethylene_topology() -> ff.Topology:
    # C0=C1 double bond, two H on each carbon, both carbons sp2.
    return ff.build_topology(
        atomic_numbers=[6, 6, 1, 1, 1, 1],
        bond_pairs=[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
        hybridizations=["SP2", "SP2", None, None, None, None],
        bond_orders=[2.0, 1.0, 1.0, 1.0, 1.0],
    )


def _ethane_topology() -> ff.Topology:
    return ff.build_topology(
        atomic_numbers=[6, 6, 1, 1, 1, 1, 1, 1],
        bond_pairs=[
            (0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)
        ],
        hybridizations=["SP3", "SP3"] + [None] * 6,
    )


def test_double_bond_rest_length_shorter_than_single():
    single = ff.build_topology([6, 6], [(0, 1)], None, bond_orders=[1.0])
    double = ff.build_topology([6, 6], [(0, 1)], None, bond_orders=[2.0])
    assert double.bonds[0][2] < single.bonds[0][2]
    assert double.bonds[0][2] == pytest.approx(1.50 * 0.89)


def test_torsions_assigned_for_sp2_and_sp3_bonds():
    ethylene = _ethylene_topology()
    # 4 H-C=C-H dihedrals, all 2-fold with the barrier split four ways.
    assert len(ethylene.torsions) == 4
    for _i, _j, _k, _l, v, n, gamma in ethylene.torsions:
        assert n == 2
        assert gamma == pytest.approx(math.pi)
        assert v == pytest.approx(ff._V_TORSION_SP2 / 4)

    ethane = _ethane_topology()
    assert len(ethane.torsions) == 9   # 3 x 3 H-C-C-H paths
    assert all(n == 3 for *_rest, n, _g in ethane.torsions)


def test_three_ring_angles_use_law_of_cosines():
    # Bare C3 ring: equal rest lengths -> equilateral -> 60 deg targets,
    # so bonds and angles share a single minimum instead of fighting.
    topo = ff.build_topology(
        [6, 6, 6], [(0, 1), (1, 2), (0, 2)], ["SP3"] * 3
    )
    assert len(topo.angles) == 3
    for *_ijk, theta0 in topo.angles:
        assert math.degrees(theta0) == pytest.approx(60.0, abs=1e-6)
    # At the exact equilateral geometry every bonded term is at rest.
    r0 = topo.bonds[0][2]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [r0, 0.0, 0.0],
            [r0 / 2, r0 * math.sqrt(3) / 2, 0.0],
        ]
    )
    energy, grad = ff.energy_and_gradient(coords, topo)
    assert energy == pytest.approx(0.0, abs=1e-9)
    assert np.abs(grad).max() < 1e-9


def test_linear_sp_center_stays_linear_with_finite_gradient():
    # CO2: sp carbon, theta0 = 180 deg. The cosine bending form must give a
    # zero (not divergent) gradient at the exactly linear geometry ...
    topo = ff.build_topology(
        [8, 6, 8], [(0, 1), (1, 2)], [None, "SP", None], bond_orders=[2, 2]
    )
    r0 = topo.bonds[0][2]
    linear = np.array([[-r0, 0.0, 0.0], [0.0, 0.0, 0.0], [r0, 0.0, 0.0]])
    energy, grad = ff.energy_and_gradient(linear, topo)
    assert np.all(np.isfinite(grad))
    assert np.abs(grad).max() < 1e-9
    assert energy == pytest.approx(0.0, abs=1e-9)

    # ... and a bent start must relax back to linear.
    bent = np.array(
        [
            [-r0 * 0.94, r0 * 0.34, 0.0],
            [0.0, 0.0, 0.0],
            [r0 * 0.94, r0 * 0.34, 0.0],
        ]
    )
    out, result = ff.optimize(bent, topo, max_iter=2000)
    assert result.converged
    v1 = out[0] - out[1]
    v2 = out[2] - out[1]
    angle = math.degrees(
        math.acos(
            float(np.dot(v1, v2))
            / float(np.linalg.norm(v1) * np.linalg.norm(v2))
        )
    )
    assert angle == pytest.approx(180.0, abs=1.0)


def test_gradient_matches_numeric_near_linear_angle():
    topo = ff.build_topology(
        [8, 6, 8], [(0, 1), (1, 2)], [None, "SP", None], bond_orders=[2, 2]
    )
    # Slightly bent, slightly stretched — off-minimum in every term.
    coords = np.array(
        [[-1.15, 0.06, 0.02], [0.0, 0.0, 0.0], [1.18, 0.05, -0.03]]
    )
    _, analytic = ff.energy_and_gradient(coords, topo)
    numeric = _numeric_gradient(coords, topo)
    assert np.allclose(analytic, numeric, atol=1e-4)


def test_vdw_14_pairs_get_half_epsilon():
    # Pentane-like chain: (0,3) is 1-4 (half eps), (0,4) is 1-5 (full eps).
    topo = ff.build_topology(
        [6] * 5, [(0, 1), (1, 2), (2, 3), (3, 4)], None
    )
    eps = {(i, j): e for i, j, _rmin, e in topo.vdw_pairs}
    assert eps[(0, 3)] == pytest.approx(ff._VDW_EPS * ff._VDW_14_SCALE)
    assert eps[(1, 4)] == pytest.approx(ff._VDW_EPS * ff._VDW_14_SCALE)
    assert eps[(0, 4)] == pytest.approx(ff._VDW_EPS)


def test_bond_stiffness_scales_with_order():
    single = ff.build_topology([6, 6], [(0, 1)], None, bond_orders=[1.0])
    triple = ff.build_topology([6, 6], [(0, 1)], None, bond_orders=[3.0])
    assert single.bonds[0][3] == pytest.approx(ff._K_BOND)
    assert triple.bonds[0][3] == pytest.approx(3.0 * ff._K_BOND)


def test_sp2_torsion_barrier_scales_with_pi_order():
    def cc_barrier(order):
        topo = ff.build_topology(
            [6, 6, 1, 1, 1, 1],
            [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
            ["SP2", "SP2", None, None, None, None],
            bond_orders=[order, 1, 1, 1, 1],
        )
        return sum(v for *_ijkl, v, _n, _g in topo.torsions)

    # Full barrier for a double bond, reduced for aromatic, weak but
    # non-zero for a conjugated sp2-sp2 single bond (biphenyl-like).
    assert cc_barrier(2.0) == pytest.approx(ff._V_TORSION_SP2)
    assert cc_barrier(1.0) < cc_barrier(1.5) < cc_barrier(2.0)
    assert cc_barrier(1.0) > 0.0


def test_vdw_epsilon_grows_with_atomic_size():
    assert ff.vdw_epsilon(6) == pytest.approx(ff._VDW_EPS)  # carbon anchor
    assert ff.vdw_epsilon(1) < ff.vdw_epsilon(6) < ff.vdw_epsilon(53)

    def end_pair_eps(z):
        topo = ff.build_topology(
            [z, 6, 6, 6, z], [(0, 1), (1, 2), (2, 3), (3, 4)], None
        )
        return next(e for i, j, _r, e in topo.vdw_pairs if (i, j) == (0, 4))

    assert end_pair_eps(53) > end_pair_eps(6) > end_pair_eps(1)


def test_vdw_cutoff_drops_distant_pairs_but_not_electrostatics():
    # Two well-separated methane-ish carbons: one vdW pair without a cutoff.
    atoms = [6, 6]
    coords = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    full = ff.build_topology(atoms, [], None)
    assert len(full.vdw_pairs) == 1

    cut = ff.build_topology(
        atoms, [], None, coords=coords, vdw_cutoff=12.0
    )
    assert cut.vdw_pairs == []

    # Electrostatics are long-range and must survive the vdW cutoff.
    charged = ff.build_topology(
        atoms, [], None, charges=[0.5, -0.5],
        coords=coords, vdw_cutoff=12.0,
    )
    assert charged.vdw_pairs == []
    assert len(charged.elec_pairs) == 1


def test_vdw_switching_tapers_smoothly_and_gradient_is_exact():
    atoms = [6, 6]
    cutoff = 12.0

    def topo_and_coords(r):
        coords = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        topo = ff.build_topology(
            atoms, [], None, coords=coords, vdw_cutoff=cutoff
        )
        return topo, coords

    # Inside the switching window (10-12 A) the analytical gradient must
    # still match numeric differentiation, i.e. the dS/dr term is correct.
    topo, coords = topo_and_coords(11.0)
    _, analytic = ff.energy_and_gradient(coords, topo)
    numeric = _numeric_gradient(coords, topo)
    assert np.allclose(analytic, numeric, atol=1e-7)

    # The energy vanishes (with no jump) right at the cutoff.
    topo_edge, coords_edge = topo_and_coords(11.999)
    e_edge, _ = ff.energy_and_gradient(coords_edge, topo_edge)
    assert abs(e_edge) < 1e-6

    # Below the switch-on radius the switched energy equals the untapered LJ.
    topo_in, coords_in = topo_and_coords(9.0)
    e_switched, _ = ff.energy_and_gradient(coords_in, topo_in)
    plain = ff.build_topology(atoms, [], None)  # no cutoff -> no switching
    e_plain, _ = ff.energy_and_gradient(coords_in, plain)
    assert e_switched == pytest.approx(e_plain)


def test_vdw_cutoff_keeps_close_pairs_unchanged():
    atoms = [6, 6]
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    full = ff.build_topology(atoms, [], None)
    cut = ff.build_topology(atoms, [], None, coords=coords, vdw_cutoff=12.0)
    assert len(cut.vdw_pairs) == len(full.vdw_pairs) == 1
    e_full, _ = ff.energy_and_gradient(coords, full)
    e_cut, _ = ff.energy_and_gradient(coords, cut)
    assert e_cut == pytest.approx(e_full)


def test_vdw_skin_shell_listed_but_energy_free():
    # Verlet list: pairs between the cutoff and cutoff + skin are listed
    # (so they are watched as atoms move) but the switching function zeroes
    # their energy; pairs beyond the skin are not listed at all.
    atoms = [6, 6]
    cutoff = 12.0
    in_skin = np.array([[0.0, 0.0, 0.0], [13.0, 0.0, 0.0]])
    topo = ff.build_topology(atoms, [], None, coords=in_skin, vdw_cutoff=cutoff)
    assert len(topo.vdw_pairs) == 1
    energy, grad = ff.energy_and_gradient(in_skin, topo)
    assert energy == pytest.approx(0.0, abs=1e-12)
    assert np.abs(grad).max() == pytest.approx(0.0, abs=1e-12)

    beyond = np.array([[0.0, 0.0, 0.0], [cutoff + ff._VDW_SKIN_A + 0.5, 0.0, 0.0]])
    topo = ff.build_topology(atoms, [], None, coords=beyond, vdw_cutoff=cutoff)
    assert topo.vdw_pairs == []


def test_refresh_vdw_pairs_tracks_moving_atoms():
    # Three free atoms: only the (0, 1) pair starts inside the list radius.
    atoms = [6, 6, 6]
    start = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    topo = ff.build_topology(atoms, [], None, coords=start, vdw_cutoff=12.0)
    assert {(i, j) for i, j, *_ in topo.vdw_pairs} == {(0, 1)}
    ff.energy_and_gradient(start, topo)  # warm the compiled-array cache

    # Atom 1 swaps places with atom 2: the pair list must follow, and the
    # compiled cache (same list length!) must be invalidated.
    moved = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    ff.refresh_vdw_pairs(topo, moved)
    assert {(i, j) for i, j, *_ in topo.vdw_pairs} == {(0, 2)}
    e_moved, _ = ff.energy_and_gradient(moved, topo)
    plain = ff.build_topology(atoms, [], None)
    e_ref, _ = ff.energy_and_gradient(moved, plain)
    # The no-cutoff reference also carries the ~1e-7 LJ tails of the 40 A
    # pairs, which the cutoff topology correctly omits.
    assert e_moved == pytest.approx(e_ref, abs=1e-6)


def test_refresh_vdw_pairs_keeps_14_scaling():
    topo = ff.build_topology(
        [6] * 5, [(0, 1), (1, 2), (2, 3), (3, 4)], None,
        coords=np.zeros((5, 3)), vdw_cutoff=12.0,
    )
    ff.refresh_vdw_pairs(topo, np.zeros((5, 3)))
    eps = {(i, j): e for i, j, _rmin, e in topo.vdw_pairs}
    assert eps[(0, 3)] == pytest.approx(ff._VDW_EPS * ff._VDW_14_SCALE)
    assert eps[(0, 4)] == pytest.approx(ff._VDW_EPS)


def test_refresh_is_noop_without_cutoff():
    topo = ff.build_topology([6, 6], [], None)
    assert len(topo.vdw_pairs) == 1
    ff.refresh_vdw_pairs(topo, np.array([[0.0, 0, 0], [50.0, 0, 0]]))
    assert len(topo.vdw_pairs) == 1  # full list: nothing to prune


def test_optimizer_refreshes_pair_list_as_atoms_approach():
    # Two opposite charges start outside the vdW list radius, so only the
    # (long-range) Coulomb term acts at first. As the attraction pulls them
    # in, the Verlet refresh must switch the LJ pair on — the shielded
    # Coulomb alone has its minimum at r = 0 and would let them collapse.
    atoms = [6, 6]
    start = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    topo = ff.build_topology(
        atoms, [], None, charges=[0.5, -0.5],
        coords=start, vdw_cutoff=12.0,
    )
    assert topo.vdw_pairs == []
    assert len(topo.elec_pairs) == 1

    out, _result = ff.optimize(start, topo, max_iter=5000)
    final = float(np.linalg.norm(out[0] - out[1]))
    assert len(topo.vdw_pairs) == 1        # the refresh picked the pair up
    assert final > 2.0                     # LJ repulsion prevented collapse
    assert final < 12.0                    # ... but they did bind


def test_no_torsions_without_hybridization():
    topo = ff.build_topology([6, 6, 6, 6], [(0, 1), (1, 2), (2, 3)], None)
    assert topo.torsions == []


def test_oop_assigned_only_to_three_coordinate_sp2_centers():
    ethylene = _ethylene_topology()
    assert sorted(o[0] for o in ethylene.oops) == [0, 1]
    ethane = _ethane_topology()
    assert ethane.oops == []


# --- Energy & analytical gradient -------------------------------------------


def _numeric_gradient(coords, topo, h=1e-5):
    grad = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for d in range(3):
            up = coords.copy()
            up[i, d] += h
            down = coords.copy()
            down[i, d] -= h
            e_up, _ = ff.energy_and_gradient(up, topo)
            e_down, _ = ff.energy_and_gradient(down, topo)
            grad[i, d] = (e_up - e_down) / (2 * h)
    return grad


def test_analytical_gradient_matches_numeric_water():
    topo = ff.build_topology([8, 1, 1], [(0, 1), (0, 2)], ["SP3", None, None])
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.80, 0.60, 0.0], [-0.80, 0.55, 0.10]]
    )
    _, analytic = ff.energy_and_gradient(coords, topo)
    numeric = _numeric_gradient(coords, topo)
    assert np.allclose(analytic, numeric, atol=1e-4)


def test_analytical_gradient_matches_numeric_chain():
    topo = ff.build_topology(
        [6, 6, 6, 6], [(0, 1), (1, 2), (2, 3)], None
    )
    rng = np.random.default_rng(1)
    coords = rng.normal(scale=1.2, size=(4, 3))
    _, analytic = ff.energy_and_gradient(coords, topo)
    numeric = _numeric_gradient(coords, topo)
    assert np.allclose(analytic, numeric, atol=1e-3)


def test_analytical_gradient_matches_numeric_with_torsions_and_oop():
    # Ethylene exercises every term: bonds, angles, torsions, oop, vdW.
    topo = _ethylene_topology()
    rng = np.random.default_rng(3)
    # Start from a roughly reasonable geometry, then perturb it.
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.33, 0.0, 0.0],
            [-0.5, 0.9, 0.1],
            [-0.5, -0.9, -0.1],
            [1.83, 0.9, 0.2],
            [1.83, -0.9, 0.0],
        ]
    ) + rng.normal(scale=0.15, size=(6, 3))
    _, analytic = ff.energy_and_gradient(coords, topo)
    numeric = _numeric_gradient(coords, topo)
    assert np.allclose(analytic, numeric, atol=1e-3)


def test_analytical_gradient_matches_numeric_ethane():
    topo = _ethane_topology()
    rng = np.random.default_rng(11)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [-0.4, 1.0, 0.0],
            [-0.4, -0.5, 0.9],
            [-0.4, -0.5, -0.9],
            [1.94, 1.0, 0.1],
            [1.94, -0.5, 0.9],
            [1.94, -0.5, -0.9],
        ]
    ) + rng.normal(scale=0.1, size=(8, 3))
    _, analytic = ff.energy_and_gradient(coords, topo)
    numeric = _numeric_gradient(coords, topo)
    assert np.allclose(analytic, numeric, atol=1e-3)


def _dihedral_deg(coords, i, j, k, l):
    b1 = coords[j] - coords[i]
    b2 = coords[k] - coords[j]
    b3 = coords[l] - coords[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    phi = math.atan2(
        float(np.dot(np.cross(n1, n2), b2)) / float(np.linalg.norm(b2)),
        float(np.dot(n1, n2)),
    )
    return math.degrees(phi)


def test_energy_components_decompose_the_total():
    topo = ff.build_topology(
        [8, 1, 1], [(0, 1), (0, 2)], ["SP3", None, None]
    )
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.10, 0.0, 0.0], [-0.20, 0.90, 0.0]]
    )  # stretched bonds, squeezed angle
    comp = ff.energy_components(coords, topo)
    assert set(comp) == {
        "bond", "angle", "torsion", "oop", "vdw", "elec", "total"
    }
    total, _ = ff.energy_and_gradient(coords, topo)
    parts = sum(v for k, v in comp.items() if k != "total")
    assert comp["total"] == pytest.approx(total)
    assert parts == pytest.approx(total)
    assert comp["bond"] > 0.0
    assert comp["angle"] > 0.0
    assert comp["elec"] == 0.0  # no charges given
    assert comp["torsion"] == 0.0 and comp["oop"] == 0.0


def test_energy_minimum_at_rest_length():
    # Two-atom bond: energy is minimal exactly at r0 and rises on either side.
    topo = ff.build_topology([6, 6], [(0, 1)], None)
    r0 = topo.bonds[0][2]
    e_rest, _ = ff.energy_and_gradient(
        np.array([[0.0, 0, 0], [r0, 0, 0]]), topo
    )
    e_short, _ = ff.energy_and_gradient(
        np.array([[0.0, 0, 0], [r0 - 0.2, 0, 0]]), topo
    )
    e_long, _ = ff.energy_and_gradient(
        np.array([[0.0, 0, 0], [r0 + 0.2, 0, 0]]), topo
    )
    assert e_rest < e_short
    assert e_rest < e_long
    assert e_rest == pytest.approx(0.0, abs=1e-9)


# --- Optimizer --------------------------------------------------------------


def test_optimize_relaxes_stretched_bond():
    topo = ff.build_topology([6, 6], [(0, 1)], None)
    r0 = topo.bonds[0][2]
    coords = np.array([[0.0, 0, 0], [r0 + 0.5, 0, 0]])
    out, result = ff.optimize(coords, topo, max_iter=500)
    final_len = float(np.linalg.norm(out[0] - out[1]))
    assert result.converged
    assert final_len == pytest.approx(r0, abs=1e-2)


def test_optimize_opens_up_bent_water():
    topo = ff.build_topology([8, 1, 1], [(0, 1), (0, 2)], ["SP3", None, None])
    # Start with an artificially tight H-O-H angle.
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.9, -0.1, 0.0]]
    )
    out, result = ff.optimize(coords, topo, max_iter=1000)
    v1 = out[1] - out[0]
    v2 = out[2] - out[0]
    angle = math.degrees(
        math.acos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )
    )
    assert angle == pytest.approx(109.47, abs=3.0)
    assert result.energy >= 0.0


def test_optimize_planarizes_twisted_ethylene():
    topo = _ethylene_topology()
    # Ethylene with the C1 end twisted ~35 degrees out of plane.
    twist = math.radians(35.0)
    c, s = math.cos(twist), math.sin(twist)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.33, 0.0, 0.0],
            [-0.55, 0.93, 0.0],
            [-0.55, -0.93, 0.0],
            [1.88, 0.93 * c, 0.93 * s],
            [1.88, -0.93 * c, -0.93 * s],
        ]
    )
    assert abs(_dihedral_deg(coords, 2, 0, 1, 4)) > 20.0
    out, _result = ff.optimize(coords, topo, max_iter=2000)
    # The 2-fold torsion should drive the H-C=C-H dihedral back to ~0/180.
    phi = abs(_dihedral_deg(out, 2, 0, 1, 4))
    assert min(phi, 180.0 - phi) < 5.0


def test_optimize_staggers_eclipsed_ethane():
    topo = _ethane_topology()
    # Nearly eclipsed ethane (15 deg twist — a perfect eclipse is a saddle
    # point with zero torsional force by symmetry).
    def ring(x, r, phase):
        return [
            [x, r * math.cos(phase + t), r * math.sin(phase + t)]
            for t in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
        ]

    coords = np.array(
        [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]]
        + ring(-0.36, 1.02, 0.0)
        + ring(1.90, 1.02, math.radians(15.0))
    )
    assert abs(_dihedral_deg(coords, 2, 0, 1, 5)) == pytest.approx(15.0, abs=1.0)
    out, _result = ff.optimize(coords, topo, max_iter=2000)
    phi = abs(_dihedral_deg(out, 2, 0, 1, 5))
    # 3-fold torsion: staggered minimum at 60 degrees.
    assert phi == pytest.approx(60.0, abs=5.0)


def test_optimize_flattens_pyramidal_sp2_center():
    # Formaldehyde-like fragment: sp2 C bonded to O (double) and two H,
    # started with the carbon pushed well out of the O-H-H plane.
    topo = ff.build_topology(
        [6, 8, 1, 1],
        [(0, 1), (0, 2), (0, 3)],
        ["SP2", None, None, None],
        bond_orders=[2.0, 1.0, 1.0],
    )
    assert len(topo.oops) == 1
    coords = np.array(
        [
            [0.0, 0.0, 0.6],       # C displaced from the plane
            [1.10, 0.0, 0.0],
            [-0.6, 0.95, 0.0],
            [-0.6, -0.95, 0.0],
        ]
    )
    out, _result = ff.optimize(coords, topo, max_iter=2000)
    # Height of C above the plane of its three substituents.
    n = np.cross(out[2] - out[1], out[3] - out[1])
    n /= np.linalg.norm(n)
    height = abs(float(np.dot(out[0] - out[1], n)))
    assert height < 0.05


def test_optimize_lowers_energy():
    topo = ff.build_topology(
        [6, 6, 6, 6], [(0, 1), (1, 2), (2, 3)], None
    )
    rng = np.random.default_rng(7)
    coords = rng.normal(scale=1.5, size=(4, 3))
    e_before, _ = ff.energy_and_gradient(coords, topo)
    out, result = ff.optimize(coords, topo, max_iter=1000)
    e_after, _ = ff.energy_and_gradient(out, topo)
    assert e_after <= e_before
    assert result.max_force < 1.0
