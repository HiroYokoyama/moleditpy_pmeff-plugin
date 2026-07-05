"""Metal-complex coordination-geometry benchmark.

PMEFF's headline claim is that it does not deform metal centers — it places /
keeps them in the right coordination geometry. The organic geometry benchmark
(``test_geometry_benchmark.py``) never exercises that, so this file validates
the metal geometry machinery directly on a set of textbook transition-metal
complexes spanning every supported coordination geometry:

* the auto-detected d-block targets (square-planar d8, octahedral), and
* the explicit per-atom overrides (linear, tetrahedral, trigonal-bipyramidal,
  square-pyramidal).

What is asserted is the *coordination geometry* (the L-M-L angle pattern), which
is what PMEFF's targets actually control — not exact M-L bond lengths, which are
radius-derived and not fit to metal data. Two things are checked per complex:

* **Recovery** — from a randomly perturbed start (0.12 A), the optimizer relaxes
  back to the target angle pattern (the targets genuinely drive the structure).
* **Stability** — from a small displacement (0.04 A), it returns to the ideal
  pattern, so the geometry is a real local minimum, not a saddle. (Starting from
  the exact ideal would be vacuous: a symmetric point is stationary under any
  potential.)

A companion ``test_benchmark_is_non_vacuous`` confirms that with the metal
targets switched off the recovery assertions fail — so the benchmark tests
PMEFF's metal machinery, not generic optimization.

Bonds are only checked for being finite, positive, and roughly equal among the
symmetry-equivalent M-L bonds (the center does not collapse or fly apart).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from pmeff_plugin.forcefield import build_topology, optimize


# ---------------------------------------------------------------------------
# Ideal coordination polyhedra (unit M-L distance, metal at the origin).
# ---------------------------------------------------------------------------
def _linear():
    return np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)


def _tetrahedral():
    return np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    ) / math.sqrt(3)


def _square_planar():
    return np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)


def _trigonal_bipyramidal():
    return np.array(
        [
            [1, 0, 0],
            [-0.5, math.sqrt(3) / 2, 0],
            [-0.5, -math.sqrt(3) / 2, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=float,
    )


def _square_pyramidal():
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float
    )


def _octahedral():
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Benchmark set: (label, metal Z, ligand Z, geometry generator, M-L start dist,
#                 override name or None, use_auto_metal, expected angle counts).
#
# Expected counts are the ideal L-M-L angle multiset (degrees, rounded), i.e.
# C(n, 2) angles per n-coordinate center.
# ---------------------------------------------------------------------------
CASES = [
    # [Ag(NH3)2]+  — linear d10, Ag-N ~2.1 A. CN2 default is already 180.
    ("[Ag(NH3)2]+ linear", 47, 7, _linear, 2.1, "linear", False, {180: 1}),
    # [CoCl4]2-  — tetrahedral, Co-Cl ~2.25 A (Co not in the d8 set → override).
    (
        "[CoCl4]2- tetrahedral",
        27,
        17,
        _tetrahedral,
        2.25,
        "tetrahedral",
        False,
        {109: 6},
    ),
    # [PtCl4]2-  — square-planar d8, Pt-Cl ~2.31 A (auto: Pt in the d8 set).
    (
        "[PtCl4]2- square planar",
        78,
        17,
        _square_planar,
        2.31,
        None,
        True,
        {90: 4, 180: 2},
    ),
    # [Ni(CN)4]2- — square-planar d8, Ni-C ~1.87 A (auto: Ni in the d8 set).
    (
        "[Ni(CN)4]2- square planar",
        28,
        6,
        _square_planar,
        1.87,
        None,
        True,
        {90: 4, 180: 2},
    ),
    # Fe(CO)5 — trigonal bipyramidal, Fe-C ~1.8 A (override).
    (
        "Fe(CO)5 trigonal bipyramidal",
        26,
        6,
        _trigonal_bipyramidal,
        1.8,
        "trigonal_bipyramidal",
        False,
        {90: 6, 120: 3, 180: 1},
    ),
    # [VO(H2O)4]-like square pyramidal, M-L ~2.0 A (override).
    (
        "square pyramidal",
        23,
        8,
        _square_pyramidal,
        2.0,
        "square_pyramidal",
        False,
        {90: 8, 180: 2},
    ),
    # [Fe(H2O)6]2+ — octahedral, Fe-O ~2.1 A (auto: Fe 6-coordinate).
    ("[Fe(H2O)6]2+ octahedral", 26, 8, _octahedral, 2.1, None, True, {90: 12, 180: 3}),
]


def _build(metal_z, ligand_z, geom, dist, override, use_auto):
    n_lig = len(geom())
    nums = [metal_z] + [ligand_z] * n_lig
    bonds = [(0, i) for i in range(1, n_lig + 1)]
    coords = np.vstack([[0.0, 0.0, 0.0], geom() * dist])
    overrides = {0: override} if override else None
    topo = build_topology(
        nums,
        bonds,
        coords=coords,
        square_planar_metals=use_auto,
        geometry_overrides=overrides,
    )
    return nums, bonds, coords, topo


def _lml_angles(coords):
    """All ligand-metal-ligand angles (degrees), metal = atom 0."""
    v = [coords[i] - coords[0] for i in range(1, len(coords))]
    out = []
    for a in range(len(v)):
        for b in range(a + 1, len(v)):
            na, nb = np.linalg.norm(v[a]), np.linalg.norm(v[b])
            ct = float(np.clip(np.dot(v[a], v[b]) / (na * nb), -1.0, 1.0))
            out.append(math.degrees(math.acos(ct)))
    return out


def _nearest_ideal(angle, ideals=(90, 109, 120, 180)):
    return min(ideals, key=lambda t: abs(angle - t))


def _histogram(angles):
    return Counter(_nearest_ideal(a) for a in angles)


# Tolerance: each optimized L-M-L angle must sit within this of its ideal.
_ANGLE_TOL_DEG = 8.0


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_metal_geometry_recovered_from_perturbation(case):
    label, mz, lz, geom, dist, override, use_auto, expected = case
    nums, bonds, coords, topo = _build(mz, lz, geom, dist, override, use_auto)

    rng = np.random.default_rng(abs(hash(label)) % (2**32))
    start = coords + rng.normal(0.0, 0.12, coords.shape)
    start[0] = coords[0]  # keep the metal at the origin frame

    final, result = optimize(start, topo, max_iter=800)
    assert np.all(np.isfinite(final)), f"{label}: non-finite coordinates"

    angles = _lml_angles(final)
    # The angle pattern must match the target coordination geometry.
    assert _histogram(angles) == expected, (
        f"{label}: angle pattern {_histogram(angles)} != expected {expected} "
        f"(angles={[round(a, 1) for a in sorted(angles)]})"
    )
    # And every angle must be close to its ideal, not merely nearest.
    worst = max(abs(a - _nearest_ideal(a)) for a in angles)
    assert worst < _ANGLE_TOL_DEG, f"{label}: worst angle deviation {worst:.1f} deg"


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_metal_geometry_is_a_stable_minimum(case):
    label, mz, lz, geom, dist, override, use_auto, expected = case
    nums, bonds, coords, topo = _build(mz, lz, geom, dist, override, use_auto)

    # A *small* displacement (not the ideal itself — a symmetric ideal is a
    # stationary point of any potential, so it would sit still regardless of the
    # targets). Relaxing back to the ideal pattern shows it is a genuine local
    # minimum, not a saddle the optimizer merely failed to leave.
    rng = np.random.default_rng((abs(hash(label)) % (2**32)) ^ 0x5F3759DF)
    start = coords + rng.normal(0.0, 0.04, coords.shape)
    start[0] = coords[0]

    final, result = optimize(start, topo, max_iter=800)
    assert np.all(np.isfinite(final)), f"{label}: non-finite coordinates"

    angles = _lml_angles(final)
    assert _histogram(angles) == expected, f"{label}: drifted to {_histogram(angles)}"
    worst = max(abs(a - _nearest_ideal(a)) for a in angles)
    assert worst < _ANGLE_TOL_DEG, f"{label}: drifted by {worst:.1f} deg"


def test_benchmark_is_non_vacuous():
    """Without the metal geometry targets the benchmark must FAIL.

    Guards against the whole benchmark silently passing on generic optimization:
    with the auto d-block targets switched off, a perturbed square-planar and a
    perturbed octahedral start relax to a *different* angle pattern, so the
    recovery assertions genuinely test PMEFF's metal machinery.
    """
    # PtCl4 without square-planar targets does not reach cis-90 / trans-180.
    _, _, coords, topo = _build(78, 17, _square_planar, 2.31, None, use_auto=False)
    rng = np.random.default_rng(123)
    start = coords + rng.normal(0.0, 0.12, coords.shape)
    start[0] = coords[0]
    final, _ = optimize(start, topo, max_iter=800)
    assert _histogram(_lml_angles(final)) != {90: 4, 180: 2}

    # Fe(H2O)6 without octahedral targets does not reach the octahedral pattern.
    _, _, coords, topo = _build(26, 8, _octahedral, 2.1, None, use_auto=False)
    start = coords + rng.normal(0.0, 0.12, coords.shape)
    start[0] = coords[0]
    final, _ = optimize(start, topo, max_iter=800)
    assert _histogram(_lml_angles(final)) != {90: 12, 180: 3}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_metal_bonds_stay_physical(case):
    """M-L bonds relax to finite, positive, mutually-consistent lengths."""
    label, mz, lz, geom, dist, override, use_auto, expected = case
    nums, bonds, coords, topo = _build(mz, lz, geom, dist, override, use_auto)

    final, _ = optimize(coords.copy(), topo, max_iter=800)
    lengths = [float(np.linalg.norm(final[i] - final[0])) for i in range(1, len(final))]
    assert all(math.isfinite(r) and r > 0.5 for r in lengths), f"{label}: {lengths}"
    # Symmetry-equivalent M-L bonds should agree closely (no one ligand ejected).
    assert max(lengths) - min(lengths) < 0.15, f"{label}: uneven M-L {lengths}"
