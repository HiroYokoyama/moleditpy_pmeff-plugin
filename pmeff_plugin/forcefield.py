"""PMEFF — a self-contained universal force field for MoleditPy.

PMEFF is a small, dependency-free molecular force field parameterized across
the **entire periodic table** (Z = 1..118). Unlike force fields that carry a
hand-tuned parameter table for a fixed subset of elements, every parameter here
is *derived* from a single per-element property — the Pyykko single-bond
covalent radius — so no element is ever missing:

* **Bonds** — harmonic, with the rest length taken as the sum of the two
  covalent radii, scaled down for double, triple and aromatic bonds and
  contracted for bond polarity (a capped electronegativity-difference term
  that fixes over-long polar bonds such as Si-O, P=O and the metal-oxides
  while leaving organic bonds untouched).
* **Angles** — harmonic in the bend angle, with the ideal angle inferred from
  the central atom's hybridization (falling back to its coordination number).
  sp3 pnictogen/chalcogen centers are compressed below tetrahedral by their
  lone pairs (VSEPR): mild for N/O (~107/104.5 deg), strong for the heavier
  congeners (~93 deg), so water, amines and thioethers bend correctly.
* **Torsions** — a cosine dihedral potential: 2-fold for sp2-sp2 bonds (keeps
  double bonds and conjugated systems planar), 3-fold for sp3-sp3 bonds
  (staggers single bonds). The per-bond barrier is split evenly over all
  dihedrals sharing that bond, UFF-style.
* **Out-of-plane** — a harmonic penalty on the pyramidalization of
  3-coordinate sp2 centers, expressed through the sum of the three bend
  angles (planar <=> sum = 360 deg).
* **van der Waals** — a Lennard-Jones 12-6 term whose per-atom radius is the
  covalent radius plus a fixed offset, which reproduces tabulated vdW radii of
  the common elements to within ~0.05 A.
* **Electrostatics (optional)** — QEq partial charges derived from Slater
  effective nuclear charges and Allred-Rochow electronegativities, feeding a
  shielded Coulomb term. Enabled together with square-planar angle targets
  for 4-coordinate d8 metal centers via the ``electronic_effects`` switch.

The only runtime dependencies are ``numpy`` (for the math) and, at the plugin
boundary, ``rdkit`` (to read connectivity and conformer coordinates). Geometry
optimization uses the FIRE algorithm, so no external optimizer or QM binary is
required.

All energy terms and their analytical gradients are evaluated with vectorized
numpy operations over precompiled index arrays, so the cost per iteration is
dominated by numpy kernels rather than Python loops.

The module is intentionally Qt-free and RDKit-free at its core: everything
operates on a :class:`Topology` of plain numbers, which makes it trivially
unit-testable without a GUI.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --- Element parameters -----------------------------------------------------

# Pyykko & Atsumi single-bond covalent radii (2009), in picometres, for every
# element Z = 1..118. These are the *only* per-element inputs PMEFF needs; all
# bonded and non-bonded parameters are derived from them, guaranteeing complete
# periodic-table coverage.
_COVALENT_RADII_PM: Tuple[int, ...] = (
    32,  46,                                                            # H  He
    133, 102,  85,  75,  71,  63,  64,  67,                             # Li..Ne
    155, 139, 126, 116, 111, 103,  99,  96,                             # Na..Ar
    196, 171,                                                           # K  Ca
    148, 136, 134, 122, 119, 116, 111, 110, 112, 118,                   # Sc..Zn
    124, 121, 121, 116, 114, 117,                                       # Ga..Kr
    210, 185,                                                           # Rb Sr
    163, 154, 147, 138, 128, 125, 125, 120, 128, 136,                   # Y..Cd
    142, 140, 140, 136, 133, 131,                                       # In..Xe
    232, 196,                                                           # Cs Ba
    180, 163, 176, 174, 173, 172, 168, 169, 168, 167, 166, 165, 164,   # La..Tm
    170, 162,                                                           # Yb Lu
    152, 146, 137, 131, 129, 122, 123, 124, 133,                        # Hf..Hg
    144, 144, 151, 145, 147, 142,                                       # Tl..Rn
    223, 201,                                                           # Fr Ra
    186, 175, 169, 170, 171, 172, 166, 166, 168, 168, 165, 167, 173,   # Ac..Md
    176, 161,                                                           # No Lr
    157, 149, 143, 141, 134, 129, 128, 121, 122,                        # Rf..Cn
    136, 143, 162, 175, 165, 157,                                       # Nh..Og
)

# Fallback radius (in Angstrom) for anything outside the table (e.g. dummy
# atoms with atomic number 0). Chosen as a mid-range covalent radius.
_DEFAULT_RADIUS_A = 1.50

# Offset (Angstrom) added to a covalent radius to approximate its vdW radius.
# covalent(C)=0.75 -> 1.65 (tabulated vdW 1.70); covalent(O)=0.63 -> 1.53
# (tabulated 1.52); covalent(H)=0.32 -> 1.22 (tabulated 1.20).
_VDW_OFFSET_A = 0.90

# Force constants (arbitrary but internally consistent energy units). Bonds are
# made much stiffer than angles, which are stiffer than the torsion, improper
# and weak vdW terms, so minimized geometries are dominated by the bonded
# topology.
_K_BOND = 700.0    # energy / Angstrom^2
_K_ANGLE = 120.0   # energy / radian^2
_K_OOP = 40.0      # energy / radian^2, on the angle-sum around sp2 centers
_VDW_EPS = 0.10    # LJ well depth (energy) for the reference atom (carbon)
_VDW_14_SCALE = 0.5  # conventional scaling of 1-4 LJ interactions
# Distance beyond which the (short-range) LJ term is dropped from the pair
# list. At 12 A a carbon-carbon LJ is ~1e-4 of the well depth, so truncation
# is negligible while it turns the O(N^2) vdW list near-linear for large
# molecules. Electrostatics are long-range and are never truncated.
_VDW_CUTOFF_A = 12.0
# Width over which the LJ term is smoothly switched off to reach zero (with
# zero slope) at the cutoff, so both energy and force stay continuous as a
# pair crosses the boundary during optimization.
_VDW_SWITCH_WIDTH_A = 2.0
# Verlet-list skin: the pair list is built out to cutoff + skin, so it stays
# valid until some atom has moved more than skin/2 since the list was built —
# only then does the optimizer rebuild it. Pairs inside the skin shell cost
# nothing energetically (the switching function is zero past the cutoff);
# the skin buys list validity while atoms move, not extra interactions.
_VDW_SKIN_A = 2.0
# Per-atom well depths scale with the covalent radius as a polarizability
# proxy: eps_i = _VDW_EPS * (r_i / r_C)^1.5, combined pairwise with the
# Lorentz-Berthelot geometric mean. Carbon (r = 0.75 A) is the anchor.
_EPS_RADIUS_REF = 0.75

# Per-bond torsional barriers (energy), split evenly across all dihedrals
# sharing the central bond, so the barrier does not grow with substitution.
_V_TORSION_SP3 = 2.0    # sp3-sp3: 3-fold, staggered minima
_V_TORSION_SP2 = 10.0   # sp2-sp2: 2-fold, planar minima (double/conjugated)
_V_TORSION_MIXED = 0.5  # sp2-sp3: 6-fold, nearly free rotation

# Bond-order rest-length shortening. Anchored to typical carbon homolytic
# lengths relative to the single-bond radius sum (1.50 A for C-C):
# C=C 1.33 A -> x0.89, C#C 1.20 A -> x0.78. Intermediate (e.g. aromatic 1.5)
# orders are interpolated linearly; aromatic C-C comes out at 1.42 A.
_BOND_ORDER_ANCHORS = ((1.0, 1.00), (2.0, 0.89), (3.0, 0.78))

# Polar-bond length contraction (Schomaker-Stevenson idea). Summing two
# covalent single-bond radii ignores the ionic contraction of a polar bond:
# a large electronegativity difference draws the atoms closer than the
# homonuclear-derived radii predict. The plain sum leaves e.g. Si-O 0.16 A
# too long (1.79 vs 1.63), P-O and the metal-oxides similarly long, and C-F
# ~0.04 A long. The correction is *quadratic above a threshold* in the
# Allred-Rochow electronegativity difference, so it is negligible for the
# mildly polar organic bonds the plain radii already handle well (C-O, C-N,
# C-H, O-H all move < 0.001 A) and only bites for genuinely polar bonds. It
# is capped so that even very ionic pairs (alkali halides) contract by a
# bounded amount rather than collapsing: Na-F lands at 1.99 A (gas 1.93),
# not the unphysical value an uncapped quadratic would give.
_POLAR_CONTRACTION_CHI0 = 2.0     # AR-chi difference below which no contraction
_POLAR_CONTRACTION_COEF = 0.157   # A per (delta-chi - chi0)^2; tuned to Si-O
_POLAR_CONTRACTION_CAP = 0.20     # max contraction (Angstrom)

# --- Optional "electronic effects" parameters --------------------------------

# Coulomb constant in the internal (kcal/mol-like) energy scale, per e^2/A.
_K_COULOMB = 332.07
_ELEC_14_SCALE = 0.5   # scaling of 1-4 electrostatic interactions
_EV_COULOMB = 14.4     # e^2/(4 pi eps0), in eV*Angstrom, for the QEq solve
# Approximate Pauling -> Mulliken (eV) electronegativity conversion.
_CHI_EV_PER_PAULING = 2.27
# Bare electronegativity equalization is known to over-polarize: it allows
# unlimited charge transfer at any distance, so electropositive centers
# (metals, boron) acquire charges large enough that their Coulomb pull on
# nearby H/heteroatoms deforms the bonded skeleton. Scaling the hardness up
# damps the charge transfer (roughly halving charges, quartering pair
# energies) while conserving the total charge *exactly* — unlike scaling the
# solved charges, which would break conservation for ions.
_QEQ_HARDNESS_SCALE = 2.0

# Metals that adopt a square-planar geometry when 4-coordinate (common d8
# centers). Used only when electronic effects are enabled.
_SQUARE_PLANAR_METALS = frozenset({28, 45, 46, 77, 78, 79})  # Ni Rh Pd Ir Pt Au

# Per-hybridization chi scaling factors (Bent's rule: more s-character →
# higher electronegativity). Applied before the QEq solve; only sp/sp2 atoms
# are shifted; everything else keeps the base Allred-Rochow value.
_CHI_HYB_SCALE: Dict[str, float] = {"SP": 1.08, "SP2": 1.04}
# Sentinel theta0 marking a square-planar angle term: the energy picks the
# nearest of the two ideal vertex angles (90 or 180 degrees) at runtime.
_SQ_PLANAR_T0 = -1.0

# Octahedral metals: all d-block transition metals (Z=21..30, 39..48, 57..80).
# 6-coordinate centers in this set get coordinate-based trans/cis angle targets
# (3 trans at π, 12 cis at π/2) rather than the flat 90° coordination default.
_OCTAHEDRAL_METALS = (
    frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81))
)

# Morse bond potential: V = D(1 − e^{−α(r−r₀)})²
# D = _MORSE_DEPTH_FACTOR * k * r₀  →  α = sqrt(k / 2D)
# Curvature at r₀ equals the harmonic (k = 2Dα²); energy is bounded above
# at D rather than growing without limit, improving robustness at large
# distortions. The factor 0.08 gives D ≈ 84 kcal/mol for a C-C single bond
# (k=700 kcal/mol/Å², r₀=1.50 Å), matching the experimental bond energy.
_MORSE_DEPTH_FACTOR = 0.08

# H-bond correction: D−H···A geometry-dependent attraction.
# Energy: eps * [(R₀/r)^12 − 2(R₀/r)^6] * cos²(θ_{DHA})
# where r = H···A distance, θ_{DHA} = angle at H.
_HBOND_DONORS = frozenset({7, 8, 9, 16})     # N  O  F  S as donors
_HBOND_ACCEPTORS = frozenset({7, 8, 9, 16})  # N  O  F  S as acceptors
_HBOND_CUTOFF_A = 4.5   # max H···A distance included in the triplet list (Å)
_HBOND_R0_A = 2.0       # equilibrium H···A distance (Å)
_HBOND_EPS = 3.0        # well depth at ideal (linear, r=R₀) H-bond (energy units)

# Dispersion correction: Becke-Johnson damped C₆/r⁶ added on top of the LJ.
# V_disp = −c₆ / (r⁶ + r₀⁶), where c₆ = _DISP_S6 · 2 · ε_LJ · rmin⁶
# and r₀ = rmin (BJ damping length = LJ equilibrium radius).
# At r = rmin this deepens the well by _DISP_S6 · ε; for r → ∞ it recovers
# the correct −C₆ / r⁶ London asymptotics. 15% extra is a light correction
# that improves aromatic stacking distances without over-binding.
_DISP_S6 = 0.15

# Aufbau filling order as (n, l, capacity), for Slater's rules.
_AUFBAU: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (4, 0, 2),
    (3, 2, 10), (4, 1, 6), (5, 0, 2), (4, 2, 10), (5, 1, 6), (6, 0, 2),
    (4, 3, 14), (5, 2, 10), (6, 1, 6), (7, 0, 2), (5, 3, 14), (6, 2, 10),
    (7, 1, 6),
)

# Ideal bond angle (degrees) by central-atom coordination number, used when the
# hybridization is unknown (common for metals).
_ANGLE_BY_COORDINATION = {
    1: 180.0,
    2: 109.47,
    3: 120.0,
    4: 109.47,
    5: 90.0,
    6: 90.0,
    7: 72.0,
    8: 72.0,
}
_DEFAULT_ANGLE_DEG = 109.47

# Lone-pair angle compression for sp3 centers (VSEPR). A bare tetrahedral
# 109.47 deg is wrong for the commonest heteroatoms: lone pairs occupy more
# angular space than bonding pairs and squeeze the inter-bond angles below
# tetrahedral. The effect is confined to the pnictogens (group 15) and
# chalcogens (group 16) — group 14 sp3 centers keep 109.47, group 17 sp3
# atoms are terminal (no angle), and group 13 sp3 centers are electron
# deficient (no lone pair). The number of lone pairs follows directly from
# the four sp3 electron domains: ``lone_pairs = 4 - coordination``, so this
# also does the right thing for charged centers (NH4+ has CN 4 -> 0 lone
# pairs -> 109.47; H3O+ has CN 3 -> 1 lone pair -> mildly pyramidal).
_PNICTOGENS = frozenset({7, 15, 33, 51, 83, 115})   # N  P  As Sb Bi Mc
_CHALCOGENS = frozenset({8, 16, 34, 52, 84, 116})   # O  S  Se Te Po Lv
_LONE_PAIR_ELEMENTS = _PNICTOGENS | _CHALCOGENS
# Period-2 centers (N, O) hybridize well: each lone pair costs ~2.5 deg,
# giving NH3 ~107 and H2O ~104.5, matching experiment. Heavier congeners
# bond through near-pure p orbitals (the s-p energy gap widens down a group),
# so their hydride angles collapse to ~92-93 deg (H2S 92, PH3 94, H2Se 91)
# largely independent of the lone-pair count — captured by a flat target.
_LP_COMPRESSION_PER_PAIR_DEG = 2.5
_HEAVY_P_BLOCK_ANGLE_DEG = 93.0


def _period(atomic_number: int) -> int:
    """Return the periodic-table period (row) of *atomic_number*."""
    for period, last_z in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if atomic_number <= last_z:
            return period
    return 7


def _sp3_lone_pair_angle(atomic_number: int, coordination: int) -> Optional[float]:
    """Return the lone-pair-compressed sp3 bond angle (deg), or None.

    None means "no compression applies" — the caller should keep the plain
    tetrahedral 109.47 deg. Only pnictogen/chalcogen centers carrying at
    least one lone pair (``lone_pairs = 4 - coordination > 0``) are shifted.
    """
    if atomic_number not in _LONE_PAIR_ELEMENTS:
        return None
    lone_pairs = 4 - coordination
    if lone_pairs <= 0:
        return None
    if _period(atomic_number) <= 2:
        return 109.47 - _LP_COMPRESSION_PER_PAIR_DEG * lone_pairs
    return _HEAVY_P_BLOCK_ANGLE_DEG


def _sp3_lone_pair_blend(
    atomic_number: int, coordination: int
) -> Optional[Tuple[float, float]]:
    """Return (open_deg, compressed_deg) for a period-2 sp3 lone-pair center.

    The compressed target from :func:`_sp3_lone_pair_angle` is calibrated on
    the *hydrides* (H2O 104.5, NH3 107). Bulkier substituents relieve the
    compression sterically and open the angle back toward the tetrahedral
    default — dimethyl ether is 111 deg, trimethylamine 111 deg, not 104-107.
    The caller blends between the two endpoints by the fraction of hydrogen
    substituents on each individual angle, so H-X-H stays fully compressed
    while C-X-C reverts to ~tetrahedral.

    Only period-2 centers (N, O) taper: heavier congeners bond through
    near-pure p orbitals and stay near 90-99 deg regardless of substituent
    (Me2S 99, H2S 92), so they keep the flat compressed target. Returns None
    when no lone-pair compression applies.
    """
    if _period(atomic_number) != 2:
        return None
    compressed = _sp3_lone_pair_angle(atomic_number, coordination)
    if compressed is None:
        return None
    return (109.47, compressed)


def covalent_radius(atomic_number: int) -> float:
    """Return the single-bond covalent radius (Angstrom) for *atomic_number*.

    Falls back to :data:`_DEFAULT_RADIUS_A` for atomic numbers outside 1..118
    (e.g. dummy atoms), so the force field never lacks a parameter.
    """
    if 1 <= atomic_number <= len(_COVALENT_RADII_PM):
        return _COVALENT_RADII_PM[atomic_number - 1] / 100.0
    return _DEFAULT_RADIUS_A


def vdw_radius(atomic_number: int) -> float:
    """Return the van der Waals radius (Angstrom) used for the LJ term."""
    return covalent_radius(atomic_number) + _VDW_OFFSET_A


def vdw_epsilon(atomic_number: int) -> float:
    """Return the per-atom LJ well depth, scaled with atomic size.

    Larger atoms are more polarizable and bind more strongly through
    dispersion; the covalent radius serves as the size/polarizability proxy,
    normalized so carbon keeps the reference well depth.
    """
    return _VDW_EPS * (covalent_radius(atomic_number) / _EPS_RADIUS_REF) ** 1.5


def bond_order_factor(order: float) -> float:
    """Return the rest-length scaling factor for a bond of the given order.

    Order 1 (and anything below) maps to 1.0; orders between the anchors are
    interpolated linearly; orders above 3 are clamped to the triple-bond
    factor. Aromatic bonds (order 1.5) land at 0.945 — C(ar)-C(ar) 1.42 A.
    """
    anchors = _BOND_ORDER_ANCHORS
    if order <= anchors[0][0]:
        return anchors[0][1]
    for (o_lo, f_lo), (o_hi, f_hi) in zip(anchors, anchors[1:]):
        if order <= o_hi:
            t = (order - o_lo) / (o_hi - o_lo)
            return f_lo + t * (f_hi - f_lo)
    return anchors[-1][1]


def polar_bond_contraction(z_i: int, z_j: int) -> float:
    """Return the polar-bond length contraction (Angstrom) for a Z_i-Z_j bond.

    A capped, quadratic-above-threshold function of the Allred-Rochow
    electronegativity difference (see :data:`_POLAR_CONTRACTION_COEF`). Zero
    for nonpolar and mildly polar bonds; grows for strongly polar bonds; never
    exceeds :data:`_POLAR_CONTRACTION_CAP`.
    """
    delta = abs(pauling_electronegativity(z_i) - pauling_electronegativity(z_j))
    excess = delta - _POLAR_CONTRACTION_CHI0
    if excess <= 0.0:
        return 0.0
    return min(_POLAR_CONTRACTION_COEF * excess * excess, _POLAR_CONTRACTION_CAP)


def bond_rest_length(z_i: int, z_j: int, order: float = 1.0) -> float:
    """Return the PMEFF rest length (Angstrom) of a Z_i-Z_j bond of *order*.

    The Pyykko single-bond covalent radii are summed, contracted for bond
    polarity (:func:`polar_bond_contraction`), then scaled for the bond order
    (:func:`bond_order_factor`). The polarity contraction is applied to the
    single-bond length before the order scaling, so a polar multiple bond
    (P=O, S=O) inherits both effects.
    """
    base = covalent_radius(z_i) + covalent_radius(z_j) - polar_bond_contraction(
        z_i, z_j
    )
    return base * bond_order_factor(order)


def slater_zeff(atomic_number: int) -> float:
    """Effective nuclear charge felt by the outermost s/p shell (Slater).

    Derived entirely from the atomic number via the Aufbau occupation and
    Slater's screening rules — no per-element table — so it covers the whole
    periodic table like every other PMEFF parameter.
    """
    z = min(max(int(atomic_number), 1), 118)
    per_n: Dict[int, int] = {}
    n_valence = 1
    remaining = z
    for n, _l, cap in _AUFBAU:
        if remaining <= 0:
            break
        fill = min(cap, remaining)
        per_n[n] = per_n.get(n, 0) + fill
        remaining -= fill
        n_valence = max(n_valence, n)
    same = per_n.get(n_valence, 0) - 1
    if n_valence == 1:
        return z - 0.30 * same
    inner = per_n.get(n_valence - 1, 0)
    deeper = sum(c for n, c in per_n.items() if n <= n_valence - 2)
    return z - 0.35 * same - 0.85 * inner - 1.0 * deeper


def pauling_electronegativity(atomic_number: int) -> float:
    """Allred-Rochow electronegativity on the (Pauling-like) chi scale.

    chi_AR = 0.359 * Zeff / r^2 + 0.744, from the Slater effective charge and
    the Pyykko covalent radius — derived, like every PMEFF parameter, from the
    atomic number alone. Used both for the QEq charge solve (after conversion
    to an energy scale) and for the polar-bond length contraction.
    """
    r = covalent_radius(atomic_number)
    return 0.359 * slater_zeff(atomic_number) / (r * r) + 0.744


def electronegativity(atomic_number: int) -> float:
    """Allred-Rochow electronegativity converted to the QEq energy (eV) scale."""
    return _CHI_EV_PER_PAULING * pauling_electronegativity(atomic_number)


def hardness(atomic_number: int) -> float:
    """Chemical hardness (eV): the self-Coulomb of a sphere of the covalent
    radius, which resists piling charge onto small atoms. Scaled up by
    :data:`_QEQ_HARDNESS_SCALE` to damp QEq's tendency to over-polarize."""
    return (
        _QEQ_HARDNESS_SCALE * _EV_COULOMB / (2.0 * covalent_radius(atomic_number))
    )


def qeq_charges(
    atomic_numbers: Sequence[int],
    coords: np.ndarray,
    total_charge: float = 0.0,
    hybridizations: Optional[Sequence[Optional[str]]] = None,
) -> np.ndarray:
    """Electronegativity-equalization (QEq-style) partial charges.

    Minimizes ``sum(chi_i q_i + 0.5 eta_i q_i^2) + sum_ij J_ij q_i q_j``
    subject to ``sum(q) = total_charge``, with an Ohno-shielded Coulomb
    interaction ``J_ij = 14.4 / sqrt(r^2 + gamma^2)`` that tends to the mean
    hardness at r = 0. One linear solve; charges are then held fixed.

    When *hybridizations* is supplied, the base Allred-Rochow electronegativity
    of each atom is scaled by :data:`_CHI_HYB_SCALE` before the solve (Bent's
    rule: higher s-character → higher electronegativity → more charge drawn in).
    """
    n = len(atomic_numbers)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([float(total_charge)])

    chi = np.array([electronegativity(z) for z in atomic_numbers])
    if hybridizations is not None:
        for i, hyb in enumerate(hybridizations):
            if hyb is not None:
                chi[i] *= _CHI_HYB_SCALE.get(str(hyb).upper(), 1.0)
    eta = np.array([hardness(z) for z in atomic_numbers])
    coords = np.asarray(coords, dtype=float)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    gamma = 2.0 * _EV_COULOMB / (eta[:, None] + eta[None, :])
    a = _EV_COULOMB / np.sqrt(d * d + gamma * gamma)
    np.fill_diagonal(a, eta)

    system = np.zeros((n + 1, n + 1))
    system[:n, :n] = a
    system[:n, n] = 1.0
    system[n, :n] = 1.0
    rhs = np.concatenate([-chi, [float(total_charge)]])
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:  # coincident atoms etc.
        logger.warning("PMEFF: QEq solve failed; using zero charges.")
        return np.zeros(n)
    return solution[:n]


@dataclass
class Topology:
    """A force-field problem stripped of any RDKit/Qt dependency.

    Attributes:
        atomic_numbers: Per-atom atomic numbers (length N).
        bonds: List of (i, j, r0, k) — atom indices, rest length (Angstrom)
            and harmonic force constant.
        angles: List of (i, j, k, theta0) — j is the vertex, theta0 in radians.
        torsions: List of (i, j, k, l, v, n, gamma) — dihedral i-j-k-l with
            energy ``0.5 * v * (1 + cos(n*phi - gamma))``.
        oops: List of (j, a, b, c) — 3-coordinate sp2 center j with neighbors
            a, b, c, kept planar by a harmonic on the sum of the three bend
            angles around j.
        vdw_pairs: List of (i, j, rmin, eps) — non-bonded LJ interactions.
        elec_pairs: List of (i, j, kqq, gamma) — shielded Coulomb
            interactions with energy ``kqq / sqrt(r^2 + gamma^2)``; kqq
            already contains the Coulomb constant, both charges and any 1-4
            scaling. Empty unless electronic effects are enabled.
    """

    atomic_numbers: Sequence[int]
    bonds: List[Tuple[int, int, float, float]] = field(default_factory=list)
    angles: List[Tuple[int, int, int, float]] = field(default_factory=list)
    torsions: List[Tuple[int, int, int, int, float, int, float]] = field(
        default_factory=list
    )
    oops: List[Tuple[int, int, int, int]] = field(default_factory=list)
    vdw_pairs: List[Tuple[int, int, float, float]] = field(default_factory=list)
    elec_pairs: List[Tuple[int, int, float, float]] = field(default_factory=list)
    # LJ switching cutoff (Angstrom), or None to evaluate every listed pair in
    # full. Set by :func:`build_topology` when a cutoff was applied.
    vdw_cutoff: Optional[float] = None
    # Connectivity-derived exclusion data (1-2/1-3 pairs and 1-4 pairs, both
    # as (i, j) with i < j), kept so :func:`refresh_vdw_pairs` can rebuild the
    # LJ pair list for new coordinates without the original bond list.
    excluded_pairs: Optional[frozenset] = None
    pairs14: Optional[frozenset] = None
    # Total molecular charge for dynamic QEq: when set, the optimizer
    # re-solves the charges (and rebuilds elec_pairs) whenever the geometry
    # has drifted enough to invalidate them. None = charges stay fixed.
    qeq_total_charge: Optional[float] = None
    # Per-atom hybridization labels stored for QEq refresh: the same
    # hybridization scaling applied at build time is re-applied on each
    # charge re-solve so the charges stay consistent. None = no scaling.
    hybridizations: Optional[List] = None
    # When True, bonds use the Morse potential D(1−e^{−α Δr})² instead of
    # the harmonic ½k Δr². Same curvature at the minimum; bounded above.
    use_morse: bool = False
    # H-bond triplets: (donor, H, acceptor, eps, r₀). Built when use_hbond=True
    # and coords are available. Empty list = H-bond term inactive.
    hbond_triplets: List[Tuple] = field(default_factory=list)
    # Dispersion pairs: (i, j, c₆, r₀⁶). Rebuilt alongside the Verlet LJ
    # list (same pair set, different coefficients). Empty = inactive.
    disp_pairs: List[Tuple] = field(default_factory=list)

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the topology."""
        return len(self.atomic_numbers)

    def compiled(self) -> Dict[str, np.ndarray]:
        """Return (and cache) the term lists as flat numpy index/param arrays.

        The cache is keyed on the term-list lengths, so a topology extended
        after a first energy evaluation is recompiled automatically.
        """
        key = (
            len(self.bonds),
            len(self.angles),
            len(self.torsions),
            len(self.oops),
            len(self.vdw_pairs),
            len(self.elec_pairs),
            len(self.hbond_triplets),
            len(self.disp_pairs),
        )
        cache = getattr(self, "_compiled_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]

        bonds = np.array(self.bonds, dtype=float).reshape(-1, 4)
        angles = np.array(self.angles, dtype=float).reshape(-1, 4)
        torsions = np.array(self.torsions, dtype=float).reshape(-1, 7)
        oops = np.array(self.oops, dtype=int).reshape(-1, 4)
        vdw = np.array(self.vdw_pairs, dtype=float).reshape(-1, 4)
        elec = np.array(self.elec_pairs, dtype=float).reshape(-1, 4)
        hbond = (
            np.array(self.hbond_triplets, dtype=float).reshape(-1, 5)
            if self.hbond_triplets
            else np.zeros((0, 5))
        )
        disp = (
            np.array(self.disp_pairs, dtype=float).reshape(-1, 4)
            if self.disp_pairs
            else np.zeros((0, 4))
        )
        arrays: Dict[str, np.ndarray] = {
            "bond_ij": bonds[:, :2].astype(int),
            "bond_r0": bonds[:, 2],
            "bond_k": bonds[:, 3],
            "angle_ijk": angles[:, :3].astype(int),
            "angle_t0": angles[:, 3],
            "tors_ijkl": torsions[:, :4].astype(int),
            "tors_v": torsions[:, 4],
            "tors_n": torsions[:, 5],
            "tors_gamma": torsions[:, 6],
            "oop_jabc": oops,
            "vdw_ij": vdw[:, :2].astype(int),
            "vdw_rmin": vdw[:, 2],
            "vdw_eps": vdw[:, 3],
            "elec_ij": elec[:, :2].astype(int),
            "elec_kqq": elec[:, 2],
            "elec_gamma": elec[:, 3],
            "hbond_dha": hbond[:, :3].astype(int),
            "hbond_eps": hbond[:, 3],
            "hbond_r0": hbond[:, 4],
            "disp_ij": disp[:, :2].astype(int),
            "disp_c6": disp[:, 2],
            "disp_r0_6": disp[:, 3],
        }
        self._compiled_cache = (key, arrays)  # type: ignore[attr-defined]
        return arrays


def _ideal_angle_deg(
    hybridization: Optional[str],
    coordination: int,
    atomic_number: Optional[int] = None,
) -> float:
    """Pick an ideal bond angle from hybridization, else coordination number.

    For sp3 centers, *atomic_number* (when given) enables lone-pair
    compression on pnictogen/chalcogen atoms (see :func:`_sp3_lone_pair_angle`)
    so water, amines, ethers and their heavier congeners bend correctly rather
    than sitting at the tetrahedral 109.47 deg.
    """
    hyb = (hybridization or "").upper()
    if hyb == "SP":
        return 180.0
    if hyb == "SP2":
        return 120.0
    if hyb == "SP3":
        if atomic_number is not None:
            compressed = _sp3_lone_pair_angle(atomic_number, coordination)
            if compressed is not None:
                return compressed
        return 109.47
    if hyb in ("SP3D", "SP2D"):
        return 90.0
    if hyb == "SP3D2":
        return 90.0
    return _ANGLE_BY_COORDINATION.get(coordination, _DEFAULT_ANGLE_DEG)


def _torsion_params(
    hyb_j: Optional[str], hyb_k: Optional[str]
) -> Optional[Tuple[float, int, float]]:
    """Return (barrier, periodicity, gamma) for a j-k central bond, or None.

    sp2-sp2 bonds get a 2-fold potential with minima at 0/180 deg (planar),
    sp3-sp3 a 3-fold one with staggered minima, and mixed sp2-sp3 a weak
    6-fold term. Anything involving sp, metals or unknown hybridization gets
    no torsion — the angle terms already fix those geometries.
    """
    a = (hyb_j or "").upper()
    b = (hyb_k or "").upper()
    if a == "SP2" and b == "SP2":
        # 0.5*v*(1 + cos(2*phi - pi)) = 0.5*v*(1 - cos(2*phi)): minima 0, 180.
        return _V_TORSION_SP2, 2, math.pi
    if a == "SP3" and b == "SP3":
        # 0.5*v*(1 + cos(3*phi)): minima at +-60, 180 (staggered).
        return _V_TORSION_SP3, 3, 0.0
    if {a, b} == {"SP2", "SP3"}:
        return _V_TORSION_MIXED, 6, math.pi
    return None


# Half-shell of neighbor-cell offsets for the cell-list pair search: the home
# cell plus the 13 lexicographically positive offsets, so every unordered pair
# of neighboring cells is scanned exactly once.
_HALF_SHELL: Tuple[Tuple[int, int, int], ...] = ((0, 0, 0),) + tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)


def _pairs_within(coords: np.ndarray, cutoff: float) -> List[Tuple[int, int]]:
    """Return every (i, j) pair (i < j) within *cutoff*, via cell lists.

    Atoms are binned into a grid of cutoff-sized cells and only the half-shell
    of neighboring cells is scanned, so for bounded density the cost is O(N)
    instead of the O(N^2) of an all-pairs distance check. The result is
    sorted, and identical to the brute-force pair list.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return []
    cells: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, key in enumerate(map(tuple, np.floor(coords / cutoff).astype(int))):
        cells.setdefault(key, []).append(idx)
    cutoff_sq = cutoff * cutoff
    pairs: List[Tuple[int, int]] = []
    for key, members in cells.items():
        home = np.array(members)
        for off in _HALF_SHELL:
            if off == (0, 0, 0):
                ia, ib = np.triu_indices(len(home), k=1)
                aa, bb = home[ia], home[ib]
            else:
                other = cells.get(
                    (key[0] + off[0], key[1] + off[1], key[2] + off[2])
                )
                if other is None:
                    continue
                aa = np.repeat(home, len(other))
                bb = np.tile(np.array(other), len(home))
            if not len(aa):
                continue
            d = coords[aa] - coords[bb]
            close = np.einsum("ij,ij->i", d, d) <= cutoff_sq
            pairs.extend(
                (int(a), int(b)) if a < b else (int(b), int(a))
                for a, b in zip(aa[close], bb[close])
            )
    pairs.sort()
    return pairs


def _vdw_pair(
    atomic_numbers: Sequence[int], i: int, j: int, is14: bool
) -> Tuple[int, int, float, float]:
    """Return the (i, j, rmin, eps) LJ parameters for one non-bonded pair."""
    rmin = vdw_radius(atomic_numbers[i]) + vdw_radius(atomic_numbers[j])
    eps = math.sqrt(
        vdw_epsilon(atomic_numbers[i]) * vdw_epsilon(atomic_numbers[j])
    )
    if is14:
        eps *= _VDW_14_SCALE
    return (i, j, rmin, eps)


def _disp_pair(
    i: int, j: int, rmin: float, eps: float
) -> Tuple[int, int, float, float]:
    """Return (i, j, c₆, r₀⁶) for one BJ-damped dispersion pair.

    c₆ = _DISP_S6 · 2 · ε · rmin⁶  (matches the LJ r⁻⁶ C₆ coefficient,
    scaled by _DISP_S6 so the correction is a fraction of the LJ well).
    r₀⁶ = rmin⁶  (BJ damping radius equals the LJ equilibrium separation).
    """
    r0_6 = rmin ** 6
    return (i, j, _DISP_S6 * 2.0 * eps * r0_6, r0_6)


def _elec_pair_list(
    atomic_numbers: Sequence[int],
    charges: Sequence[float],
    excluded: frozenset,
    pairs14: frozenset,
) -> List[Tuple[int, int, float, float]]:
    """Build the shielded-Coulomb pair list for the given charges.

    Every non-excluded pair (Coulomb is long-range — no cutoff), with the
    Coulomb constant and both charges baked into ``kqq``, 1-4 pairs scaled,
    and near-zero products dropped.
    """
    n = len(atomic_numbers)
    # Hardness-derived Ohno shielding: gamma_ij = 2*J_e/(eta_i+eta_j), the
    # same kernel used in the QEq solve matrix, ensuring the energy and the
    # charge derivation are internally consistent.
    eta = [hardness(z) for z in atomic_numbers]
    out: List[Tuple[int, int, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            kqq = _K_COULOMB * float(charges[i]) * float(charges[j])
            if (i, j) in pairs14:
                kqq *= _ELEC_14_SCALE
            if abs(kqq) > 1e-12:
                gamma = 2.0 * _EV_COULOMB / (eta[i] + eta[j])
                out.append((i, j, kqq, gamma))
    return out


def refresh_vdw_pairs(topo: Topology, coords: np.ndarray) -> None:
    """Rebuild the LJ pair list of *topo* for the current *coords*.

    A cutoff-built pair list is only valid near the geometry it was built
    from: atoms that drift to within the cutoff of each other would otherwise
    interact with no LJ term at all. This re-selects every non-excluded pair
    inside cutoff + skin — the Verlet-list refresh. No-op for topologies
    built without a cutoff (their list already contains every pair).
    """
    if topo.vdw_cutoff is None or topo.excluded_pairs is None:
        return
    coords = np.asarray(coords, dtype=float)
    topo.vdw_pairs = [
        _vdw_pair(topo.atomic_numbers, i, j, (i, j) in topo.pairs14)
        for i, j in _pairs_within(coords, topo.vdw_cutoff + _VDW_SKIN_A)
        if (i, j) not in topo.excluded_pairs
    ]
    # Rebuild dispersion pairs from the refreshed LJ list when active.
    if topo.disp_pairs:
        topo.disp_pairs = [
            _disp_pair(i, j, rmin, eps)
            for i, j, rmin, eps in topo.vdw_pairs
        ]
    # The compiled-array cache is keyed on term-list lengths only; a refresh
    # can swap pairs without changing the count, so drop it explicitly.
    topo._compiled_cache = None  # pylint: disable=protected-access


def refresh_qeq_charges(topo: Topology, coords: np.ndarray) -> None:
    """Re-solve the QEq charges for *coords* and rebuild the Coulomb pairs.

    The QEq solution depends on the geometry, so charges baked into the pair
    list at build time describe a geometry that no longer exists after a
    large relaxation. Because the charges *minimize* the QEq energy, the
    fixed-charge gradient stays exact at the re-solved charges (envelope
    theorem) — refreshing costs one linear solve and adds no gradient terms.
    No-op unless the topology carries a dynamic-charge total
    (:attr:`Topology.qeq_total_charge`).
    """
    if topo.qeq_total_charge is None or topo.excluded_pairs is None:
        return
    charges = qeq_charges(
        topo.atomic_numbers, coords, topo.qeq_total_charge, topo.hybridizations
    )
    topo.elec_pairs = _elec_pair_list(
        topo.atomic_numbers, charges, topo.excluded_pairs, topo.pairs14
    )
    # Pair count may change (near-zero products are dropped) or stay the
    # same with different kqq values; either way the cache must go.
    topo._compiled_cache = None  # pylint: disable=protected-access


def build_topology(
    atomic_numbers: Sequence[int],
    bond_pairs: Sequence[Tuple[int, int]],
    hybridizations: Optional[Sequence[Optional[str]]] = None,
    bond_orders: Optional[Sequence[float]] = None,
    charges: Optional[Sequence[float]] = None,
    square_planar_metals: bool = False,
    coords: Optional[np.ndarray] = None,
    vdw_cutoff: Optional[float] = None,
    use_morse: bool = False,
    use_dispersion: bool = False,
    use_hbond: bool = False,
    use_polar_contraction: bool = True,
) -> Topology:
    """Assemble a :class:`Topology` from connectivity alone.

    Args:
        atomic_numbers: Atomic number of every atom.
        bond_pairs: (i, j) index pairs describing covalent bonds.
        hybridizations: Optional per-atom hybridization labels ("SP", "SP2",
            "SP3", ...). Used to choose ideal bond angles, torsion potentials
            and out-of-plane terms; when omitted or unknown, the coordination
            number is used for angles and no torsions are assigned.
        bond_orders: Optional per-bond orders aligned with *bond_pairs*
            (1 single, 1.5 aromatic, 2 double, 3 triple). Rest lengths of
            higher-order bonds are shortened accordingly; omitted or
            unrecognized entries are treated as single bonds.
        charges: Optional per-atom partial charges (e.g. from
            :func:`qeq_charges`). When given, non-excluded pairs get a
            shielded Coulomb interaction (1-4 pairs scaled by 0.5).
        square_planar_metals: When True, both 4-coordinate d8 metal centers
            (Ni, Pd, Pt, Rh, Ir, Au — square-planar) and 6-coordinate d-block
            transition metals (octahedral) get coordinate-based trans/cis angle
            targets (π/2 cis, π trans) instead of coordination-number defaults.
        coords: Optional (N, 3) coordinates. Used for the Verlet LJ pair list
            when *vdw_cutoff* is given, coordinate-based metal angle targets,
            and H-bond partner detection.
        vdw_cutoff: Optional distance (Angstrom) beyond which LJ pairs are
            dropped. Requires *coords*. Electrostatic pairs are never
            truncated (Coulomb is long-range).
        use_morse: When True, store Morse-potential flag so the bond term uses
            D(1−e^{−α Δr})² instead of the harmonic ½k Δr².
        use_dispersion: When True, build Becke-Johnson damped dispersion pairs
            alongside the LJ pair list. Requires *coords* and *vdw_cutoff*.
        use_hbond: When True and *coords* are provided, detect D−H···A triplets
            (donors/acceptors N, O, F, S) and store them for the H-bond term.
        use_polar_contraction: When True (default), shorten polar bond rest
            lengths by the electronegativity-difference contraction
            (:func:`bond_rest_length`); when False, use the plain covalent
            radius sum.
    """
    n = len(atomic_numbers)
    neighbors: List[set] = [set() for _ in range(n)]
    for i, j in bond_pairs:
        if i == j:
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)

    def hyb(idx: int) -> Optional[str]:
        return hybridizations[idx] if hybridizations is not None else None

    topo = Topology(atomic_numbers=list(atomic_numbers))
    topo.hybridizations = list(hybridizations) if hybridizations is not None else None
    topo.use_morse = use_morse

    seen_bonds = set()
    order_by_pair = {}
    for b_idx, (i, j) in enumerate(bond_pairs):
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen_bonds:
            continue
        seen_bonds.add(key)
        order = 1.0
        if bond_orders is not None and b_idx < len(bond_orders):
            try:
                order = max(float(bond_orders[b_idx]), 0.5)
            except (TypeError, ValueError):
                order = 1.0
        order_by_pair[key] = order
        z_i, z_j = atomic_numbers[i], atomic_numbers[j]
        if use_polar_contraction:
            r0 = bond_rest_length(z_i, z_j, order)
        else:
            r0 = (covalent_radius(z_i) + covalent_radius(z_j)) * bond_order_factor(
                order
            )
        # Stretching stiffness grows roughly linearly with bond order
        # (C=C is ~2x, C#C ~3x as stiff as C-C).
        topo.bonds.append((key[0], key[1], r0, _K_BOND * order))

    rest_length = {(i, j): r0 for i, j, r0, _k in topo.bonds}
    _coords = np.asarray(coords, dtype=float) if coords is not None else None

    for j in range(n):
        nbrs = sorted(neighbors[j])
        if len(nbrs) < 2:
            continue
        is_sq_planar = (
            square_planar_metals
            and len(nbrs) == 4
            and atomic_numbers[j] in _SQUARE_PLANAR_METALS
        )
        is_octahedral = (
            square_planar_metals
            and len(nbrs) == 6
            and atomic_numbers[j] in _OCTAHEDRAL_METALS
        )
        is_special_metal = is_sq_planar or is_octahedral
        # Square-planar: 2 trans pairs (π); octahedral: 3 trans pairs (π).
        n_trans = 2 if is_sq_planar else (3 if is_octahedral else 0)
        _trans_pairs: Optional[frozenset] = None
        if is_special_metal and _coords is not None and n_trans > 0:
            # Geometry-based trans/cis assignment.
            #
            # Read the initial L-M-L angles from the coordinates. The n_trans
            # largest angles (atom-exclusive, greedy) are labelled trans (π);
            # all remaining pairs are cis (π/2). This avoids the symmetric-
            # gradient trap in near-tetrahedral/octahedral starting geometries,
            # where a nearest-target sentinel would assign the same ideal angle
            # to all pairs and the force network stalls.
            _sq: List[Tuple[float, int, int]] = []
            for _ai, _la in enumerate(nbrs):
                for _lb in nbrs[_ai + 1:]:
                    _r1 = _coords[_la] - _coords[j]
                    _r2 = _coords[_lb] - _coords[j]
                    _n1 = float(np.linalg.norm(_r1))
                    _n2 = float(np.linalg.norm(_r2))
                    if _n1 > 1e-9 and _n2 > 1e-9:
                        _ct = float(np.clip(
                            np.dot(_r1, _r2) / (_n1 * _n2), -1.0, 1.0
                        ))
                    else:
                        _ct = 0.0
                    _sq.append((math.acos(_ct), _la, _lb))
            _sq.sort(reverse=True)
            _selected: List[Tuple[int, int]] = []
            _used: set = set()
            for _, _la, _lb in _sq:
                if _la not in _used and _lb not in _used:
                    _selected.append((_la, _lb))
                    _used.update((_la, _lb))
                    if len(_selected) == n_trans:
                        break
            if len(_selected) == n_trans:
                _trans_pairs = frozenset(_selected)
        lp_blend: Optional[Tuple[float, float]] = None
        if not is_special_metal:
            theta0 = math.radians(
                _ideal_angle_deg(hyb(j), len(nbrs), atomic_numbers[j])
            )
            # Lone-pair compression is calibrated on hydrides; open it back up
            # per angle when the substituents are heavier (see
            # _sp3_lone_pair_blend). Only period-2 N/O centers taper.
            if (hyb(j) or "").upper() == "SP3":
                blend = _sp3_lone_pair_blend(atomic_numbers[j], len(nbrs))
                if blend is not None:
                    lp_blend = (math.radians(blend[0]), math.radians(blend[1]))
        else:
            theta0 = _SQ_PLANAR_T0  # sentinel: used only when _trans_pairs is None
        for a, atom_a in enumerate(nbrs):
            for atom_b in nbrs[a + 1:]:
                if is_special_metal:
                    if _trans_pairs is not None:
                        target = (
                            math.pi if (atom_a, atom_b) in _trans_pairs
                            else math.pi / 2
                        )
                    else:
                        target = _SQ_PLANAR_T0
                else:
                    target = theta0
                    if lp_blend is not None and atom_b not in neighbors[atom_a]:
                        # Blend open<->compressed by how many substituents are
                        # hydrogen: H-X-H fully compressed, C-X-C ~tetrahedral.
                        open_r, comp_r = lp_blend
                        n_h = (atomic_numbers[atom_a] == 1) + (
                            atomic_numbers[atom_b] == 1
                        )
                        f_h = n_h / 2.0
                        target = open_r - (open_r - comp_r) * f_h
                    if atom_b in neighbors[atom_a]:
                        # Three-membered ring: the hybridization-based angle
                        # would fight the three bond terms. Use the exact angle
                        # the rest lengths dictate (law of cosines) so bonds and
                        # angles share one minimum (e.g. 60 deg in cyclopropane).
                        r1 = rest_length[(min(atom_a, j), max(atom_a, j))]
                        r2 = rest_length[(min(atom_b, j), max(atom_b, j))]
                        r3 = rest_length[
                            (min(atom_a, atom_b), max(atom_a, atom_b))
                        ]
                        cos_t = (r1 * r1 + r2 * r2 - r3 * r3) / (2.0 * r1 * r2)
                        target = math.acos(max(-1.0, min(1.0, cos_t)))
                topo.angles.append((atom_a, j, atom_b, target))

    # Torsions: one term per i-j-k-l path around every j-k bond whose two
    # central atoms both have a recognized (sp2/sp3) hybridization. The
    # barrier is divided by the number of dihedrals on that bond so the
    # rotational barrier is a per-bond, not per-substituent, quantity.
    for j, k, _r0, _kf in topo.bonds:
        params = _torsion_params(hyb(j), hyb(k))
        if params is None:
            continue
        v_bond, periodicity, gamma = params
        if periodicity == 2:
            # The 2-fold barrier reflects the pi character of the central
            # bond: full for a double bond (order 2), reduced for aromatic
            # (1.5), weak for a conjugated sp2-sp2 single bond (biphenyl).
            order = order_by_pair.get((min(j, k), max(j, k)), 1.0)
            v_bond *= min(max(order - 1.0, 0.15), 1.0)
        ends_i = [a for a in sorted(neighbors[j]) if a != k]
        ends_l = [d for d in sorted(neighbors[k]) if d != j]
        paths = [
            (i, l) for i in ends_i for l in ends_l if i != l  # skip 3-rings
        ]
        if not paths:
            continue
        v_each = v_bond / len(paths)
        for i, l in paths:
            topo.torsions.append((i, j, k, l, v_each, periodicity, gamma))

    # Out-of-plane: every 3-coordinate sp2 center is kept planar.
    for j in range(n):
        if (hyb(j) or "").upper() != "SP2":
            continue
        nbrs = sorted(neighbors[j])
        if len(nbrs) == 3:
            topo.oops.append((j, nbrs[0], nbrs[1], nbrs[2]))

    # Non-bonded: every pair separated by more than two bonds (exclude 1-2,
    # 1-3); 1-4 pairs get the conventionally halved LJ well depth so torsional
    # profiles are not swamped by the vdW clash of the end atoms.
    excluded = set()
    pairs14 = set()
    for i in range(n):
        for j in neighbors[i]:
            excluded.add((min(i, j), max(i, j)))
            for k in neighbors[j]:
                if k == i:
                    continue
                excluded.add((min(i, k), max(i, k)))
                for l in neighbors[k]:
                    if l not in (i, j):
                        pairs14.add((min(i, l), max(i, l)))
    pairs14 -= excluded
    topo.excluded_pairs = frozenset(excluded)
    topo.pairs14 = frozenset(pairs14)
    if coords is not None and vdw_cutoff is not None:
        # List pairs out to cutoff + skin (Verlet list): the extra shell
        # contributes zero energy (switched off at the cutoff) but keeps the
        # list valid while atoms move up to skin/2 during optimization. The
        # cell-list search keeps this O(N) for bounded density.
        topo.vdw_cutoff = float(vdw_cutoff)
        coords = np.asarray(coords, dtype=float)
        for i, j in _pairs_within(coords, topo.vdw_cutoff + _VDW_SKIN_A):
            if (i, j) not in excluded:
                topo.vdw_pairs.append(
                    _vdw_pair(atomic_numbers, i, j, (i, j) in pairs14)
                )
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in excluded:
                    topo.vdw_pairs.append(
                        _vdw_pair(atomic_numbers, i, j, (i, j) in pairs14)
                    )

    if charges is not None:
        topo.elec_pairs = _elec_pair_list(
            atomic_numbers, charges, excluded, pairs14
        )

    # H-bond triplets: D−H···A where D and A are in _HBOND_DONORS/ACCEPTORS.
    # Detection uses coords for the distance filter; if coords are absent the
    # triplet list stays empty and the term is silently inactive.
    if use_hbond and _coords is not None:
        for _h in range(n):
            if atomic_numbers[_h] != 1:
                continue
            _donors = [
                _d for _d in sorted(neighbors[_h])
                if atomic_numbers[_d] in _HBOND_DONORS
            ]
            if not _donors:
                continue
            _donor = _donors[0]  # H has at most one donor in a valid molecule
            for _a in range(n):
                if atomic_numbers[_a] not in _HBOND_ACCEPTORS:
                    continue
                if _a == _donor or _a == _h:
                    continue
                if (min(_h, _a), max(_h, _a)) in excluded:
                    continue
                if float(np.linalg.norm(_coords[_h] - _coords[_a])) <= _HBOND_CUTOFF_A:
                    topo.hbond_triplets.append((_donor, _h, _a, _HBOND_EPS, _HBOND_R0_A))

    # Dispersion correction pairs: same pair set as the LJ Verlet list, with
    # BJ-damped C₆ coefficients computed from the LJ parameters. Rebuilt on
    # each Verlet refresh (see refresh_vdw_pairs).
    if use_dispersion:
        topo.disp_pairs = [
            _disp_pair(i, j, rmin, eps)
            for i, j, rmin, eps in topo.vdw_pairs
        ]

    return topo


# --- Energy & analytical gradient -------------------------------------------


def _switch(
    r: np.ndarray, r_on: float, r_off: float
) -> Tuple[np.ndarray, np.ndarray]:
    """CHARMM-style switching factor and its derivative for distances *r*.

    Returns (S, dS/dr) where ``S`` is 1 below *r_on*, falls smoothly to 0 at
    *r_off*, and has zero slope at both ends, so ``S * E`` and its force are
    continuous across the cutoff. Outside [r_on, r_off] the derivative is 0.
    """
    a = r_off * r_off
    b = r_on * r_on
    r2 = r * r
    denom = (a - b) ** 3
    s = np.ones_like(r)
    ds = np.zeros_like(r)
    mid = (r > r_on) & (r < r_off)
    am = a - r2[mid]
    s[mid] = am * am * (a + 2.0 * r2[mid] - 3.0 * b) / denom
    ds[mid] = 12.0 * r[mid] * am * (b - r2[mid]) / denom
    s[r >= r_off] = 0.0
    return s, ds


def _bend_terms(
    coords: np.ndarray, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized bend geometry for angles i-j-k.

    Returns (theta, cos_theta, dcos_di, dcos_dk); the vertex derivative is
    ``-(dcos_di + dcos_dk)`` by translational invariance. Terms with a
    degenerate (zero-length) arm get zero derivatives.
    """
    rij = coords[ii] - coords[jj]
    rkj = coords[kk] - coords[jj]
    nij = np.linalg.norm(rij, axis=1)
    nkj = np.linalg.norm(rkj, axis=1)
    safe = (nij > 1e-9) & (nkj > 1e-9)
    nij = np.where(safe, nij, 1.0)
    nkj = np.where(safe, nkj, 1.0)

    cos_t = np.clip(np.sum(rij * rkj, axis=1) / (nij * nkj), -1.0, 1.0)
    theta = np.arccos(cos_t)

    zero = np.where(safe, 1.0, 0.0)[:, None]
    dcos_di = zero * (
        rkj / (nij * nkj)[:, None] - (cos_t / nij**2)[:, None] * rij
    )
    dcos_dk = zero * (
        rij / (nij * nkj)[:, None] - (cos_t / nkj**2)[:, None] * rkj
    )
    return theta, cos_t, dcos_di, dcos_dk


def _angle_terms(
    coords: np.ndarray, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized bend angles i-j-k and their theta-gradients.

    Returns (theta, dtheta_di, dtheta_dk). Only valid away from theta = 0 or
    pi, where d(theta)/d(cos) diverges — linear angles must use the cosine
    form instead (see the angle section of :func:`energy_and_gradient`).
    """
    theta, cos_t, dcos_di, dcos_dk = _bend_terms(coords, ii, jj, kk)
    sin_t = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 1e-12))
    inv = (-1.0 / sin_t)[:, None]
    return theta, inv * dcos_di, inv * dcos_dk


def energy_and_gradient(
    coords: np.ndarray,
    topo: Topology,
    components: Optional[Dict[str, float]] = None,
) -> Tuple[float, np.ndarray]:
    """Return (energy, gradient) for *coords* (shape (N, 3)) under *topo*.

    The gradient (dE/dx, same shape as *coords*) is fully analytical, so the
    optimizer converges quickly without finite-difference noise. All five
    energy terms are evaluated with vectorized numpy operations.

    When *components* (a dict) is supplied it is filled with the per-term
    energy decomposition under the keys ``bond``, ``angle``, ``torsion``,
    ``oop``, ``vdw`` and ``elec`` — see :func:`energy_components`.
    """
    coords = np.asarray(coords, dtype=float)
    grad = np.zeros_like(coords)
    energy = 0.0
    arrays = topo.compiled()
    if components is not None:
        components.update(
            bond=0.0, angle=0.0, torsion=0.0, oop=0.0,
            vdw=0.0, elec=0.0, hbond=0.0, disp=0.0,
        )

    def _record(name: str, e_term: float) -> float:
        if components is not None:
            components[name] = e_term
        return e_term

    # --- Bonds: harmonic ½k(r−r₀)² or Morse D(1−e^{−α Δr})² ---
    bond_ij = arrays["bond_ij"]
    if len(bond_ij):
        d = coords[bond_ij[:, 0]] - coords[bond_ij[:, 1]]
        r = np.linalg.norm(d, axis=1)
        safe = r > 1e-9
        r = np.where(safe, r, 1.0)
        diff = np.where(safe, r - arrays["bond_r0"], 0.0)
        k = arrays["bond_k"]
        if topo.use_morse:
            # D = _MORSE_DEPTH_FACTOR · k · r₀  →  α = sqrt(k / 2D).
            # Same curvature as harmonic at the minimum; bounded above at D.
            r0 = arrays["bond_r0"]
            morse_d = _MORSE_DEPTH_FACTOR * k * r0
            alpha = np.sqrt(k / (2.0 * morse_d))
            x_term = np.exp(-alpha * diff)   # e^{−α Δr}
            energy += _record("bond", float(np.sum(morse_d * (1.0 - x_term) ** 2)))
            de_dr = 2.0 * morse_d * alpha * x_term * (1.0 - x_term)
        else:
            energy += _record("bond", 0.5 * float(np.sum(k * diff * diff)))
            de_dr = k * diff
        g = (de_dr / r)[:, None] * d
        np.add.at(grad, bond_ij[:, 0], g)
        np.add.at(grad, bond_ij[:, 1], -g)

    # --- Angles ---
    # Ordinary angles: 0.5 * k * (theta - theta0)^2. Linear targets
    # (theta0 = pi, sp centers): k * (1 + cos theta), which has the same
    # curvature at theta = pi but a finite gradient there — the harmonic
    # form's d(theta)/d(cos) diverges exactly at the linear minimum.
    angle_ijk = arrays["angle_ijk"]
    if len(angle_ijk):
        ii, jj, kk = angle_ijk[:, 0], angle_ijk[:, 1], angle_ijk[:, 2]
        theta, cos_t, dcos_di, dcos_dk = _bend_terms(coords, ii, jj, kk)
        t0 = arrays["angle_t0"]
        # Square-planar sentinel: pull toward whichever ideal vertex angle
        # (90 or 180 degrees) is closer, letting cis/trans assignment emerge
        # from the geometry itself.
        t0 = np.where(
            t0 < 0.0,
            np.where(theta < 0.75 * math.pi, 0.5 * math.pi, math.pi),
            t0,
        )
        linear = t0 > math.pi - 1e-6
        dtheta = theta - t0
        energy += _record(
            "angle",
            float(
                np.sum(
                    np.where(
                        linear,
                        _K_ANGLE * (1.0 + cos_t),
                        0.5 * _K_ANGLE * dtheta * dtheta,
                    )
                )
            ),
        )
        sin_t = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 1e-12))
        de_dcos = np.where(
            linear, _K_ANGLE, -_K_ANGLE * dtheta / sin_t
        )[:, None]
        gi = de_dcos * dcos_di
        gk = de_dcos * dcos_dk
        np.add.at(grad, ii, gi)
        np.add.at(grad, kk, gk)
        np.add.at(grad, jj, -(gi + gk))

    # --- Torsions: 0.5 * v * (1 + cos(n*phi - gamma)) ---
    tors = arrays["tors_ijkl"]
    if len(tors):
        ii, jj, kk, ll = tors[:, 0], tors[:, 1], tors[:, 2], tors[:, 3]
        b1 = coords[jj] - coords[ii]
        b2 = coords[kk] - coords[jj]
        b3 = coords[ll] - coords[kk]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1sq = np.sum(n1 * n1, axis=1)
        n2sq = np.sum(n2 * n2, axis=1)
        b2n = np.linalg.norm(b2, axis=1)
        safe = (n1sq > 1e-12) & (n2sq > 1e-12) & (b2n > 1e-9)
        n1sq = np.where(safe, n1sq, 1.0)
        n2sq = np.where(safe, n2sq, 1.0)
        b2n_s = np.where(safe, b2n, 1.0)

        phi = np.arctan2(
            np.sum(np.cross(n1, n2) * b2, axis=1) / b2n_s,
            np.sum(n1 * n2, axis=1),
        )
        v = arrays["tors_v"]
        n_per = arrays["tors_n"]
        gamma = arrays["tors_gamma"]
        arg = n_per * phi - gamma
        energy += _record(
            "torsion",
            0.5 * float(np.sum(np.where(safe, v * (1.0 + np.cos(arg)), 0.0))),
        )
        de_dphi = np.where(safe, -0.5 * v * n_per * np.sin(arg), 0.0)

        # Standard analytical dihedral derivatives (van Schaik et al.),
        # adapted to b1 = rj - ri, b3 = rl - rk. Verified against numeric
        # differentiation in the test suite.
        dphi_di = (-(b2n_s / n1sq))[:, None] * n1
        dphi_dl = (b2n_s / n2sq)[:, None] * n2
        s12 = (np.sum(b1 * b2, axis=1) / (b2n_s * b2n_s))[:, None]
        s32 = (np.sum(b3 * b2, axis=1) / (b2n_s * b2n_s))[:, None]
        dphi_dj = -(1.0 + s12) * dphi_di + s32 * dphi_dl
        dphi_dk = s12 * dphi_di - (1.0 + s32) * dphi_dl

        de = de_dphi[:, None]
        np.add.at(grad, ii, de * dphi_di)
        np.add.at(grad, jj, de * dphi_dj)
        np.add.at(grad, kk, de * dphi_dk)
        np.add.at(grad, ll, de * dphi_dl)

    # --- Out-of-plane: 0.5 * k * (theta_ab + theta_bc + theta_ac - 2*pi)^2 ---
    oops = arrays["oop_jabc"]
    if len(oops):
        jj = oops[:, 0]
        delta = -2.0 * math.pi * np.ones(len(oops))
        parts = []
        for c1, c2 in ((1, 2), (2, 3), (1, 3)):
            theta, dth_d1, dth_d2 = _angle_terms(
                coords, oops[:, c1], jj, oops[:, c2]
            )
            delta += theta
            parts.append((oops[:, c1], oops[:, c2], dth_d1, dth_d2))
        energy += _record("oop", 0.5 * _K_OOP * float(np.sum(delta * delta)))
        de = (_K_OOP * delta)[:, None]
        for a1, a2, dth_d1, dth_d2 in parts:
            g1 = de * dth_d1
            g2 = de * dth_d2
            np.add.at(grad, a1, g1)
            np.add.at(grad, a2, g2)
            np.add.at(grad, jj, -(g1 + g2))

    # --- van der Waals: eps * ((rmin/r)^12 - 2 (rmin/r)^6) ---
    # Smoothly switched off over the last _VDW_SWITCH_WIDTH_A before the
    # cutoff (when one is in effect), so a pair crossing the boundary during
    # optimization sees no energy or force jump.
    vdw_ij = arrays["vdw_ij"]
    if len(vdw_ij):
        d = coords[vdw_ij[:, 0]] - coords[vdw_ij[:, 1]]
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        r6 = (arrays["vdw_rmin"] / r) ** 6
        r12 = r6 * r6
        eps = arrays["vdw_eps"]
        e_lj = eps * (r12 - 2.0 * r6)
        de_lj = eps * 12.0 * (r6 - r12) / r
        if topo.vdw_cutoff is not None:
            r_on = max(topo.vdw_cutoff - _VDW_SWITCH_WIDTH_A, 0.0)
            s, ds = _switch(r, r_on, topo.vdw_cutoff)
            energy += _record("vdw", float(np.sum(s * e_lj)))
            de_dr = ds * e_lj + s * de_lj
        else:
            energy += _record("vdw", float(np.sum(e_lj)))
            de_dr = de_lj
        g = (de_dr / r)[:, None] * d
        np.add.at(grad, vdw_ij[:, 0], g)
        np.add.at(grad, vdw_ij[:, 1], -g)

    # --- Electrostatics: kqq / sqrt(r^2 + gamma^2) (shielded Coulomb) ---
    elec_ij = arrays["elec_ij"]
    if len(elec_ij):
        d = coords[elec_ij[:, 0]] - coords[elec_ij[:, 1]]
        kqq = arrays["elec_kqq"]
        gamma = arrays["elec_gamma"]
        inv = 1.0 / np.sqrt(np.sum(d * d, axis=1) + gamma * gamma)
        energy += _record("elec", float(np.sum(kqq * inv)))
        g = (-(kqq * inv**3))[:, None] * d
        np.add.at(grad, elec_ij[:, 0], g)
        np.add.at(grad, elec_ij[:, 1], -g)

    # --- H-bond: eps · [(R₀/r)^12 − 2(R₀/r)^6] · cos²(θ_{DHA}) ---
    # Minimum at r = R₀ (H···A distance), θ = 0° (linear D−H···A).
    # The LJ-12-6 radial profile is repulsive below R₀ and attractive above,
    # with the correct r⁻⁶ long-range tail. The cos² angular factor suppresses
    # bent H-bonds smoothly to zero. Donors and acceptors: N, O, F, S.
    hbond_dha = arrays["hbond_dha"]
    if len(hbond_dha):
        dd = hbond_dha[:, 0]   # donor D
        hh = hbond_dha[:, 1]   # hydrogen H (vertex of the D-H-A angle)
        aa = hbond_dha[:, 2]   # acceptor A
        eps_hb = arrays["hbond_eps"]
        r0_hb = arrays["hbond_r0"]
        d_HA = coords[hh] - coords[aa]
        r_HA = np.maximum(np.linalg.norm(d_HA, axis=1), 1e-6)
        r6_hb = (r0_hb / r_HA) ** 6
        r12_hb = r6_hb * r6_hb
        e_radial = r12_hb - 2.0 * r6_hb
        # D-H-A angle: D is atom-i, H is vertex (j), A is atom-k.
        _theta_dha, cos_t, dcos_di, dcos_dk = _bend_terms(coords, dd, hh, aa)
        cos2 = cos_t * cos_t
        energy += _record("hbond", float(np.sum(eps_hb * e_radial * cos2)))
        # Radial gradient (H and A move along the H-A vector).
        de_dr = eps_hb * cos2 * 12.0 * (r6_hb - r12_hb) / r_HA
        g_r = (de_dr / r_HA)[:, None] * d_HA
        np.add.at(grad, hh, g_r)
        np.add.at(grad, aa, -g_r)
        # Angular gradient (D, H, A all feel the cos² dependence).
        de_cos = (eps_hb * e_radial * 2.0 * cos_t)[:, None]
        g_D = de_cos * dcos_di
        g_A = de_cos * dcos_dk
        np.add.at(grad, dd, g_D)
        np.add.at(grad, aa, g_A)
        np.add.at(grad, hh, -(g_D + g_A))

    # --- Dispersion: −c₆ / (r⁶ + r₀⁶) (Becke-Johnson damped) ---
    # Added on top of the LJ term. At r = rmin it deepens the well by
    # _DISP_S6 · ε; for r → ∞ it recovers the correct −C₆/r⁶ asymptotics.
    # The BJ denominator prevents divergence at short range.
    disp_ij = arrays["disp_ij"]
    if len(disp_ij):
        d_disp = coords[disp_ij[:, 0]] - coords[disp_ij[:, 1]]
        r_disp = np.maximum(np.linalg.norm(d_disp, axis=1), 1e-6)
        r6_disp = r_disp ** 6
        c6 = arrays["disp_c6"]
        r0_6 = arrays["disp_r0_6"]
        denom = r6_disp + r0_6
        energy += _record("disp", float(np.sum(-c6 / denom)))
        de_dr = c6 * 6.0 * r_disp ** 5 / (denom * denom)
        g_d = (de_dr / r_disp)[:, None] * d_disp
        np.add.at(grad, disp_ij[:, 0], g_d)
        np.add.at(grad, disp_ij[:, 1], -g_d)

    return energy, grad


def energy_components(coords: np.ndarray, topo: Topology) -> Dict[str, float]:
    """Return the per-term energy decomposition of *coords* under *topo*.

    Keys: ``bond``, ``angle``, ``torsion``, ``oop``, ``vdw``, ``elec``,
    ``hbond``, ``disp`` and ``total`` (the sum). Terms inactive in the
    topology report 0.0, so all keys are always present.
    """
    comp: Dict[str, float] = {}
    total, _ = energy_and_gradient(coords, topo, components=comp)
    comp["total"] = total
    return comp


def hessian(
    coords: np.ndarray, topo: Topology, step: float = 1e-4
) -> np.ndarray:
    """Return the (3N, 3N) Hessian at *coords* under *topo*.

    Central finite differences of the *analytical* gradient (2 gradient
    evaluations per coordinate, 6N total), then symmetrized. With unit
    masses this is also the dynamical matrix.
    """
    coords = np.asarray(coords, dtype=float)
    flat = coords.ravel()
    n3 = flat.size
    hess = np.zeros((n3, n3))
    for a in range(n3):
        up = flat.copy()
        up[a] += step
        down = flat.copy()
        down[a] -= step
        _, g_up = energy_and_gradient(up.reshape(coords.shape), topo)
        _, g_down = energy_and_gradient(down.reshape(coords.shape), topo)
        hess[a] = (g_up - g_down).ravel() / (2.0 * step)
    return 0.5 * (hess + hess.T)


def vibrational_analysis(
    coords: np.ndarray, topo: Topology, zero_tol: float = 1e-2
) -> Dict[str, Any]:
    """Unit-mass normal-mode analysis of *coords* under *topo*.

    Diagonalizes the Hessian and classifies the eigenvalues: near-zero modes
    (|lambda| <= *zero_tol*) are the rigid-body translations/rotations (6 for
    a nonlinear molecule, 5 for a linear one, plus any genuinely soft modes),
    negative modes are imaginary frequencies — descent directions the
    optimizer converged *onto* rather than away from (a saddle point).

    Returns a dict with:
        ``frequencies`` — signed sqrt of the eigenvalues, ascending, in
        internal (unit-mass) units, **not** cm^-1; negative values mark
        imaginary modes.
        ``num_imaginary`` — count of eigenvalues below -*zero_tol*.
        ``num_zero`` — count of |eigenvalue| <= *zero_tol* (rigid body).
        ``is_minimum`` — True when no imaginary modes are present.
    """
    eigvals = np.linalg.eigvalsh(hessian(coords, topo))
    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals))
    return {
        "frequencies": freqs,
        "num_imaginary": int(np.sum(eigvals < -zero_tol)),
        "num_zero": int(np.sum(np.abs(eigvals) <= zero_tol)),
        "is_minimum": bool(np.all(eigvals >= -zero_tol)),
    }


@dataclass
class OptimizeResult:
    """Outcome of a :func:`optimize` run."""

    converged: bool
    energy: float
    steps: int
    max_force: float


# FIRE hands over to L-BFGS once the largest per-atom force drops below this
# value: FIRE is robust through clashes and rearrangements, L-BFGS converges
# superlinearly in the near-quadratic basin where FIRE crawls.
_LBFGS_CROSSOVER = 1.0
_LBFGS_HISTORY = 10       # stored (s, y) curvature pairs
_LBFGS_ARMIJO = 1e-4      # sufficient-decrease constant for the line search
_LBFGS_MAX_BACKTRACKS = 20


def _max_force(grad: np.ndarray) -> float:
    """Largest per-atom force magnitude (forces = -grad, same norms)."""
    return float(np.max(np.linalg.norm(grad, axis=1))) if len(grad) else 0.0


class _RefreshTracker:
    """Rebuilds geometry-dependent pair data when atoms drift past skin/2.

    Covers both the Verlet LJ list (cutoff topologies) and the QEq charges
    (dynamic-charge topologies). Calling the tracker with the current
    coordinates returns True when the pair data was rebuilt — i.e. the
    energy surface changed and any cached curvature information is stale.
    """

    def __init__(self, topo: Topology, x: np.ndarray):
        self._topo = topo
        active = topo.excluded_pairs is not None and (
            topo.vdw_cutoff is not None or topo.qeq_total_charge is not None
        )
        self._x_ref = x.copy() if (active and len(x)) else None

    def __call__(self, x: np.ndarray) -> bool:
        if self._x_ref is None:
            return False
        drift = float(np.max(np.linalg.norm(x - self._x_ref, axis=1)))
        if drift <= 0.5 * _VDW_SKIN_A:
            return False
        refresh_vdw_pairs(self._topo, x)
        refresh_qeq_charges(self._topo, x)
        self._x_ref = x.copy()
        return True


def _fire_phase(
    x: np.ndarray,
    topo: Topology,
    refresh: _RefreshTracker,
    energy: float,
    grad: np.ndarray,
    budget: int,
    stop_force: float,
    max_step: float,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """FIRE 2.0 iterations until max force < *stop_force* or budget is spent.

    Implements the Guenole et al. (2020) refinements: on an uphill step, half
    of the last applied (clamped) step is retracted before the velocity reset
    — using the clamped displacement, not dt*v, keeps the retraction bounded
    when forces were huge — and mixing follows the semi-implicit Euler
    update. Returns (x, energy, grad, steps_used).
    """
    v = np.zeros_like(x)
    # FIRE 2.0 tuning constants (Bitzek et al. 2006; Guenole et al. 2020).
    dt = 0.1
    dt_max = 0.5
    # Keep the floor far below the stability limit of the stiffest bond
    # (omega ~ sqrt(2 k_bond) ~ 50 rad per time unit): a floor near the limit
    # locks the step length at the displacement clamp and the optimizer
    # orbits the minimum instead of settling into it.
    dt_min = 1e-4
    n_min = 5
    f_inc = 1.1
    f_dec = 0.5
    alpha_start = 0.1
    f_alpha = 0.99

    alpha = alpha_start
    steps_since_neg = 0
    dx = np.zeros_like(x)  # last applied (clamped) displacement

    steps = 0
    while steps < budget:
        if _max_force(grad) < stop_force:
            break
        steps += 1
        forces = -grad

        power = float(np.sum(forces * v))
        if power > 0.0:
            steps_since_neg += 1
            if steps_since_neg > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
        else:
            x = x - 0.5 * dx
            v[:] = 0.0
            dt = max(dt * f_dec, dt_min)
            alpha = alpha_start
            steps_since_neg = 0

        # Semi-implicit Euler with unit masses, then FIRE velocity mixing.
        v = v + dt * forces
        fnorm = float(np.linalg.norm(forces))
        if fnorm > 1e-12:
            vnorm = float(np.linalg.norm(v))
            v = (1.0 - alpha) * v + alpha * (forces / fnorm) * vnorm
        dx = dt * v
        # Clamp the largest per-atom displacement for stability.
        step_norms = np.linalg.norm(dx, axis=1)
        largest = float(np.max(step_norms)) if len(step_norms) else 0.0
        if largest > max_step:
            dx *= max_step / largest
        x = x + dx

        refresh(x)
        energy, grad = energy_and_gradient(x, topo)

    return x, energy, grad, steps


def _lbfgs_phase(
    x: np.ndarray,
    topo: Topology,
    refresh: _RefreshTracker,
    energy: float,
    grad: np.ndarray,
    budget: int,
    f_tol: float,
    max_step: float,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """L-BFGS iterations until max force < *f_tol*, stall, or spent budget.

    Two-loop recursion over the last :data:`_LBFGS_HISTORY` curvature pairs,
    an Armijo backtracking line search, the same per-atom displacement clamp
    as FIRE, and a curvature-history reset whenever the refresh tracker
    rebuilds the pair data (the quadratic model no longer matches the
    surface). A failed line search returns early — the caller falls back to
    FIRE. Returns (x, energy, grad, steps_used).
    """
    s_hist: List[np.ndarray] = []
    y_hist: List[np.ndarray] = []
    rho: List[float] = []

    steps = 0
    while steps < budget:
        if _max_force(grad) < f_tol:
            break
        steps += 1

        # Two-loop recursion: q becomes the quasi-Newton direction H^-1 g.
        q = grad.ravel().copy()
        alphas: List[float] = []
        for s, y, r in zip(
            reversed(s_hist), reversed(y_hist), reversed(rho)
        ):
            a = r * float(s @ q)
            alphas.append(a)
            q -= a * y
        if y_hist:
            y_last = y_hist[-1]
            q *= float(s_hist[-1] @ y_last) / float(y_last @ y_last)
        else:
            # First step: steepest descent, normalized to unit total length.
            q /= max(float(np.linalg.norm(q)), 1e-12)
        for (s, y, r), a in zip(
            zip(s_hist, y_hist, rho), reversed(alphas)
        ):
            b = r * float(y @ q)
            q += s * (a - b)
        d = -q.reshape(x.shape)

        g_dot_d = float(np.sum(grad * d))
        if g_dot_d >= 0.0:
            # Not a descent direction (stale curvature): restart from
            # steepest descent.
            s_hist.clear()
            y_hist.clear()
            rho.clear()
            d = -grad
            g_dot_d = -float(np.sum(grad * grad))

        # Per-atom displacement clamp (scaling d scales the directional
        # derivative by the same factor).
        step_norms = np.linalg.norm(d, axis=1)
        largest = float(np.max(step_norms)) if len(step_norms) else 0.0
        if largest > max_step:
            scale = max_step / largest
            d *= scale
            g_dot_d *= scale

        # Armijo backtracking line search.
        t = 1.0
        for _ in range(_LBFGS_MAX_BACKTRACKS):
            x_new = x + t * d
            e_new, g_new = energy_and_gradient(x_new, topo)
            if e_new <= energy + _LBFGS_ARMIJO * t * g_dot_d:
                break
            t *= 0.5
        else:
            break  # line search stalled — hand back to the caller

        s_vec = (x_new - x).ravel()
        y_vec = (g_new - grad).ravel()
        sy = float(s_vec @ y_vec)
        if sy > 1e-12:  # keep the inverse-Hessian model positive definite
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            rho.append(1.0 / sy)
            if len(s_hist) > _LBFGS_HISTORY:
                s_hist.pop(0)
                y_hist.pop(0)
                rho.pop(0)
        x, energy, grad = x_new, e_new, g_new

        if refresh(x):
            # The pair data changed under us: the stored curvature describes
            # the old surface, so drop it and re-evaluate.
            s_hist.clear()
            y_hist.clear()
            rho.clear()
            energy, grad = energy_and_gradient(x, topo)

    return x, energy, grad, steps


def optimize(
    coords: np.ndarray,
    topo: Topology,
    max_iter: int = 500,
    f_tol: float = 1e-3,
    max_step: float = 0.20,
) -> Tuple[np.ndarray, OptimizeResult]:
    """Minimize the PMEFF energy of *coords*: FIRE 2.0, then L-BFGS.

    FIRE (Fast Inertial Relaxation Engine) handles the far-from-minimum
    regime — it is robust through steric clashes and large rearrangements —
    and hands over to an L-BFGS finisher once the largest per-atom force
    drops below :data:`_LBFGS_CROSSOVER`, where the quasi-Newton model
    converges superlinearly instead of FIRE's slow inertial crawl. If the
    L-BFGS line search stalls (e.g. on a kink in the surface), FIRE resumes
    with the remaining budget.

    Throughout both phases, geometry-dependent pair data (the Verlet LJ list
    and, for dynamic-charge topologies, the QEq charges) is refreshed
    whenever any atom has drifted more than half the list skin.

    Args:
        coords: Initial coordinates, shape (N, 3). Not modified in place.
        topo: The force-field topology.
        max_iter: Total iteration budget across all phases.
        f_tol: Convergence threshold on the largest per-atom force magnitude.
        max_step: Maximum distance (Angstrom) any atom may move in one step.

    Returns:
        (optimized_coords, OptimizeResult).
    """
    x = np.array(coords, dtype=float)
    refresh = _RefreshTracker(topo, x)
    energy, grad = energy_and_gradient(x, topo)

    steps = 0
    budget = max_iter
    x, energy, grad, used = _fire_phase(
        x, topo, refresh, energy, grad, budget,
        max(_LBFGS_CROSSOVER, f_tol), max_step,
    )
    steps += used
    budget -= used
    if budget > 0 and _max_force(grad) >= f_tol:
        x, energy, grad, used = _lbfgs_phase(
            x, topo, refresh, energy, grad, budget, f_tol, max_step
        )
        steps += used
        budget -= used
    if budget > 0 and _max_force(grad) >= f_tol:
        # L-BFGS stalled: finish with FIRE, now targeting f_tol directly.
        x, energy, grad, used = _fire_phase(
            x, topo, refresh, energy, grad, budget, f_tol, max_step
        )
        steps += used

    max_force = _max_force(grad)
    return x, OptimizeResult(max_force < f_tol, energy, steps, max_force)


# --- RDKit boundary ---------------------------------------------------------


def topology_from_rdkit(
    mol: Any,
    electronic_effects: bool = False,
    use_morse: bool = False,
    use_dispersion: bool = False,
    use_hbond: bool = False,
    use_polar_contraction: bool = True,
) -> Topology:
    """Build a :class:`Topology` from an RDKit molecule's connectivity.

    With *electronic_effects* enabled, QEq partial charges are derived from
    the conformer geometry (adding a shielded Coulomb term), 4-coordinate d8
    metal centers get square-planar angle targets, and 6-coordinate d-block
    transition metals get octahedral targets. *use_polar_contraction* (default
    True) shortens polar bond rest lengths by the electronegativity-difference
    contraction.
    """
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    bond_pairs = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    ]
    bond_orders: List[float] = []
    for b in mol.GetBonds():
        try:
            bond_orders.append(float(b.GetBondTypeAsDouble()))
        except Exception:  # pragma: no cover - defensive
            bond_orders.append(1.0)
    hybridizations: List[Optional[str]] = []
    for atom in mol.GetAtoms():
        try:
            hybridizations.append(str(atom.GetHybridization()))
        except Exception:  # pragma: no cover - defensive
            hybridizations.append(None)

    coords = _conformer_coords(mol)
    charges: Optional[np.ndarray] = None
    total = 0.0
    if electronic_effects and coords is not None:
        total = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        charges = qeq_charges(atomic_numbers, coords, total, hybridizations)

    topo = build_topology(
        atomic_numbers,
        bond_pairs,
        hybridizations,
        bond_orders,
        charges=charges,
        square_planar_metals=electronic_effects,
        coords=coords,
        vdw_cutoff=_VDW_CUTOFF_A,
        use_morse=use_morse,
        use_dispersion=use_dispersion,
        use_hbond=use_hbond,
        use_polar_contraction=use_polar_contraction,
    )
    if charges is not None:
        # Mark the charges as dynamic: the optimizer re-solves them as the
        # geometry moves, keeping electrostatics consistent with the
        # structure it is actually relaxing.
        topo.qeq_total_charge = total
    return topo


def _conformer_coords(mol: Any) -> Optional[np.ndarray]:
    """Extract conformer coordinates as an (N, 3) array, or None if absent."""
    try:
        conf = mol.GetConformer()
    except Exception:
        return None
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=float)
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        coords[i] = (pos.x, pos.y, pos.z)
    return coords


def compute_energy(mol: Any, electronic_effects: bool = False) -> Optional[float]:
    """Return the PMEFF single-point energy of *mol*, or None if unavailable."""
    components = compute_energy_components(
        mol, electronic_effects=electronic_effects
    )
    return None if components is None else components["total"]


def compute_energy_components(
    mol: Any,
    electronic_effects: bool = False,
    use_morse: bool = False,
    use_dispersion: bool = False,
    use_hbond: bool = False,
    use_polar_contraction: bool = True,
) -> Optional[Dict[str, float]]:
    """Return the per-term PMEFF energy decomposition of *mol*, or None.

    Same keys as :func:`energy_components`; None when the molecule has no
    conformer to evaluate.
    """
    coords = _conformer_coords(mol)
    if coords is None:
        return None
    topo = topology_from_rdkit(
        mol, electronic_effects=electronic_effects,
        use_morse=use_morse, use_dispersion=use_dispersion, use_hbond=use_hbond,
        use_polar_contraction=use_polar_contraction,
    )
    return energy_components(coords, topo)


def check_minimum(
    mol: Any,
    electronic_effects: bool = False,
    use_morse: bool = False,
    use_dispersion: bool = False,
    use_hbond: bool = False,
    use_polar_contraction: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run a vibrational analysis on *mol*'s current conformer.

    Returns the :func:`vibrational_analysis` dict, or None when the molecule
    has no conformer. Lets the user verify that an optimized structure is a
    true minimum rather than a saddle point (e.g. a symmetric geometry the
    optimizer could not break out of).
    """
    coords = _conformer_coords(mol)
    if coords is None:
        return None
    topo = topology_from_rdkit(
        mol, electronic_effects=electronic_effects,
        use_morse=use_morse, use_dispersion=use_dispersion, use_hbond=use_hbond,
        use_polar_contraction=use_polar_contraction,
    )
    return vibrational_analysis(coords, topo)


def optimize_rdkit_mol(
    mol: Any,
    max_iter: int = 500,
    f_tol: float = 1e-3,
    electronic_effects: bool = False,
    use_morse: bool = False,
    use_dispersion: bool = False,
    use_hbond: bool = False,
    use_polar_contraction: bool = True,
) -> Tuple[bool, Optional[OptimizeResult]]:
    """Optimize an RDKit molecule's conformer in place with PMEFF.

    Returns (success, result). Molecules with fewer than two atoms or without a
    conformer are treated as trivially successful (nothing to do) with a None
    result, matching the optimization-callback contract expected by MoleditPy.
    """
    if mol is None or mol.GetNumAtoms() < 2:
        return True, None

    coords = _conformer_coords(mol)
    if coords is None:
        logger.warning("PMEFF: molecule has no 3D conformer to optimize.")
        return False, None

    topo = topology_from_rdkit(
        mol, electronic_effects=electronic_effects,
        use_morse=use_morse, use_dispersion=use_dispersion, use_hbond=use_hbond,
        use_polar_contraction=use_polar_contraction,
    )
    new_coords, result = optimize(coords, topo, max_iter=max_iter, f_tol=f_tol)

    if not np.all(np.isfinite(new_coords)):
        logger.error("PMEFF: optimization produced non-finite coordinates.")
        return False, result

    conf = mol.GetConformer()
    try:
        from rdkit.Geometry import Point3D  # local import: keep core RDKit-free

        for i in range(mol.GetNumAtoms()):
            x, y, z = new_coords[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    except Exception:
        logger.exception("PMEFF: failed to write optimized coordinates back.")
        return False, result

    return True, result
