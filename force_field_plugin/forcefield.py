"""PMEFF — a self-contained universal force field for MoleditPy.

PMEFF is a small, dependency-free molecular force field parameterized across
the **entire periodic table** (Z = 1..118). Unlike force fields that carry a
hand-tuned parameter table for a fixed subset of elements, every parameter here
is *derived* from a single per-element property — the Pyykko single-bond
covalent radius — so no element is ever missing:

* **Bonds** — harmonic, with the rest length taken as the sum of the two
  covalent radii, scaled down for double, triple and aromatic bonds.
* **Angles** — harmonic in the bend angle, with the ideal angle inferred from
  the central atom's hybridization (falling back to its coordination number).
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

# --- Optional "electronic effects" parameters --------------------------------

# Coulomb constant in the internal (kcal/mol-like) energy scale, per e^2/A.
_K_COULOMB = 332.07
_ELEC_14_SCALE = 0.5   # scaling of 1-4 electrostatic interactions
_EV_COULOMB = 14.4     # e^2/(4 pi eps0), in eV*Angstrom, for the QEq solve
# Approximate Pauling -> Mulliken (eV) electronegativity conversion.
_CHI_EV_PER_PAULING = 2.27

# Metals that adopt a square-planar geometry when 4-coordinate (common d8
# centers). Used only when electronic effects are enabled.
_SQUARE_PLANAR_METALS = frozenset({28, 45, 46, 77, 78, 79})  # Ni Rh Pd Ir Pt Au
# Sentinel theta0 marking a square-planar angle term: the energy picks the
# nearest of the two ideal vertex angles (90 or 180 degrees) at runtime.
_SQ_PLANAR_T0 = -1.0

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


def electronegativity(atomic_number: int) -> float:
    """Allred-Rochow electronegativity (eV, Mulliken-like scale).

    chi_AR = 0.359 * Zeff / r^2 + 0.744 on the Pauling scale, using the
    Slater effective charge and the Pyykko covalent radius, then converted
    to an energy scale for the QEq charge solve.
    """
    r = covalent_radius(atomic_number)
    chi_pauling = 0.359 * slater_zeff(atomic_number) / (r * r) + 0.744
    return _CHI_EV_PER_PAULING * chi_pauling


def hardness(atomic_number: int) -> float:
    """Chemical hardness (eV): the self-Coulomb of a sphere of the covalent
    radius, which resists piling charge onto small atoms."""
    return _EV_COULOMB / (2.0 * covalent_radius(atomic_number))


def qeq_charges(
    atomic_numbers: Sequence[int],
    coords: np.ndarray,
    total_charge: float = 0.0,
) -> np.ndarray:
    """Electronegativity-equalization (QEq-style) partial charges.

    Minimizes ``sum(chi_i q_i + 0.5 eta_i q_i^2) + sum_ij J_ij q_i q_j``
    subject to ``sum(q) = total_charge``, with an Ohno-shielded Coulomb
    interaction ``J_ij = 14.4 / sqrt(r^2 + gamma^2)`` that tends to the mean
    hardness at r = 0. One linear solve; charges are then held fixed.
    """
    n = len(atomic_numbers)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([float(total_charge)])

    chi = np.array([electronegativity(z) for z in atomic_numbers])
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
        }
        self._compiled_cache = (key, arrays)  # type: ignore[attr-defined]
        return arrays


def _ideal_angle_deg(hybridization: Optional[str], coordination: int) -> float:
    """Pick an ideal bond angle from hybridization, else coordination number."""
    hyb = (hybridization or "").upper()
    if hyb == "SP":
        return 180.0
    if hyb == "SP2":
        return 120.0
    if hyb == "SP3":
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


def build_topology(
    atomic_numbers: Sequence[int],
    bond_pairs: Sequence[Tuple[int, int]],
    hybridizations: Optional[Sequence[Optional[str]]] = None,
    bond_orders: Optional[Sequence[float]] = None,
    charges: Optional[Sequence[float]] = None,
    square_planar_metals: bool = False,
    coords: Optional[np.ndarray] = None,
    vdw_cutoff: Optional[float] = None,
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
        square_planar_metals: When True, 4-coordinate common-d8 metal
            centers (Ni, Pd, Pt, Rh, Ir, Au) get square-planar angle terms
            (nearest of 90/180 degrees) instead of tetrahedral ones.
        coords: Optional (N, 3) coordinates. Only used to apply *vdw_cutoff*;
            without them the cutoff is ignored and every pair is kept.
        vdw_cutoff: Optional distance (Angstrom) beyond which LJ pairs are
            dropped. Requires *coords*. Electrostatic pairs are never
            truncated (Coulomb is long-range).
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

    seen_bonds = set()
    order_by_pair = {}
    for b_idx, (i, j) in enumerate(bond_pairs):
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen_bonds:
            continue
        seen_bonds.add(key)
        r0 = covalent_radius(atomic_numbers[i]) + covalent_radius(atomic_numbers[j])
        order = 1.0
        if bond_orders is not None and b_idx < len(bond_orders):
            try:
                order = max(float(bond_orders[b_idx]), 0.5)
            except (TypeError, ValueError):
                order = 1.0
        order_by_pair[key] = order
        r0 *= bond_order_factor(order)
        # Stretching stiffness grows roughly linearly with bond order
        # (C=C is ~2x, C#C ~3x as stiff as C-C).
        topo.bonds.append((key[0], key[1], r0, _K_BOND * order))

    rest_length = {(i, j): r0 for i, j, r0, _k in topo.bonds}

    for j in range(n):
        nbrs = sorted(neighbors[j])
        if len(nbrs) < 2:
            continue
        if (
            square_planar_metals
            and len(nbrs) == 4
            and atomic_numbers[j] in _SQUARE_PLANAR_METALS
        ):
            theta0 = _SQ_PLANAR_T0
        else:
            theta0 = math.radians(_ideal_angle_deg(hyb(j), len(nbrs)))
        for a, atom_a in enumerate(nbrs):
            for atom_b in nbrs[a + 1:]:
                target = theta0
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
    use_cutoff = coords is not None and vdw_cutoff is not None
    if use_cutoff:
        coords = np.asarray(coords, dtype=float)
        cutoff_sq = float(vdw_cutoff) * float(vdw_cutoff)
        topo.vdw_cutoff = float(vdw_cutoff)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            far = False
            if use_cutoff:
                d = coords[i] - coords[j]
                far = float(d @ d) > cutoff_sq
            if not far:
                rmin = vdw_radius(atomic_numbers[i]) + vdw_radius(atomic_numbers[j])
                eps = math.sqrt(
                    vdw_epsilon(atomic_numbers[i]) * vdw_epsilon(atomic_numbers[j])
                )
                if (i, j) in pairs14:
                    eps *= _VDW_14_SCALE
                topo.vdw_pairs.append((i, j, rmin, eps))
            if charges is not None:
                kqq = _K_COULOMB * float(charges[i]) * float(charges[j])
                if (i, j) in pairs14:
                    kqq *= _ELEC_14_SCALE
                if abs(kqq) > 1e-12:
                    gamma = 0.5 * (
                        covalent_radius(atomic_numbers[i])
                        + covalent_radius(atomic_numbers[j])
                    )
                    topo.elec_pairs.append((i, j, kqq, gamma))

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
    coords: np.ndarray, topo: Topology
) -> Tuple[float, np.ndarray]:
    """Return (energy, gradient) for *coords* (shape (N, 3)) under *topo*.

    The gradient (dE/dx, same shape as *coords*) is fully analytical, so the
    optimizer converges quickly without finite-difference noise. All five
    energy terms are evaluated with vectorized numpy operations.
    """
    coords = np.asarray(coords, dtype=float)
    grad = np.zeros_like(coords)
    energy = 0.0
    arrays = topo.compiled()

    # --- Bonds: 0.5 * k * (r - r0)^2 ---
    bond_ij = arrays["bond_ij"]
    if len(bond_ij):
        d = coords[bond_ij[:, 0]] - coords[bond_ij[:, 1]]
        r = np.linalg.norm(d, axis=1)
        safe = r > 1e-9
        r = np.where(safe, r, 1.0)
        diff = np.where(safe, r - arrays["bond_r0"], 0.0)
        k = arrays["bond_k"]
        energy += 0.5 * float(np.sum(k * diff * diff))
        g = (k * diff / r)[:, None] * d
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
        energy += float(
            np.sum(
                np.where(
                    linear,
                    _K_ANGLE * (1.0 + cos_t),
                    0.5 * _K_ANGLE * dtheta * dtheta,
                )
            )
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
        energy += 0.5 * float(np.sum(np.where(safe, v * (1.0 + np.cos(arg)), 0.0)))
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
        energy += 0.5 * _K_OOP * float(np.sum(delta * delta))
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
            energy += float(np.sum(s * e_lj))
            de_dr = ds * e_lj + s * de_lj
        else:
            energy += float(np.sum(e_lj))
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
        energy += float(np.sum(kqq * inv))
        g = (-(kqq * inv**3))[:, None] * d
        np.add.at(grad, elec_ij[:, 0], g)
        np.add.at(grad, elec_ij[:, 1], -g)

    return energy, grad


@dataclass
class OptimizeResult:
    """Outcome of a :func:`optimize` run."""

    converged: bool
    energy: float
    steps: int
    max_force: float


def optimize(
    coords: np.ndarray,
    topo: Topology,
    max_iter: int = 500,
    f_tol: float = 1e-3,
    max_step: float = 0.20,
) -> Tuple[np.ndarray, OptimizeResult]:
    """Minimize the PMEFF energy of *coords* using the FIRE algorithm.

    FIRE (Fast Inertial Relaxation Engine) is a robust, gradient-only optimizer
    that needs no external solver. A per-atom displacement clamp (*max_step*)
    keeps it stable regardless of the absolute force-constant scale.

    Args:
        coords: Initial coordinates, shape (N, 3). Not modified in place.
        topo: The force-field topology.
        max_iter: Maximum FIRE iterations.
        f_tol: Convergence threshold on the largest per-atom force magnitude.
        max_step: Maximum distance (Angstrom) any atom may move in one step.

    Returns:
        (optimized_coords, OptimizeResult).
    """
    x = np.array(coords, dtype=float)
    v = np.zeros_like(x)

    # FIRE tuning constants (Bitzek et al., 2006).
    dt = 0.1
    dt_max = 0.5
    n_min = 5
    f_inc = 1.1
    f_dec = 0.5
    alpha_start = 0.1
    f_alpha = 0.99

    alpha = alpha_start
    steps_since_neg = 0

    energy, grad = energy_and_gradient(x, topo)
    forces = -grad
    max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(x) else 0.0

    step = 0
    for step in range(1, max_iter + 1):
        if max_force < f_tol:
            return x, OptimizeResult(True, energy, step - 1, max_force)

        power = float(np.sum(forces * v))
        if power > 0.0:
            fnorm = float(np.linalg.norm(forces))
            vnorm = float(np.linalg.norm(v))
            if fnorm > 1e-12:
                v = (1.0 - alpha) * v + alpha * (forces / fnorm) * vnorm
            steps_since_neg += 1
            if steps_since_neg > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
        else:
            v[:] = 0.0
            dt *= f_dec
            alpha = alpha_start
            steps_since_neg = 0

        # Velocity-Verlet-style update with unit masses.
        v = v + dt * forces
        dx = dt * v
        # Clamp the largest per-atom displacement for stability.
        step_norms = np.linalg.norm(dx, axis=1)
        largest = float(np.max(step_norms)) if len(step_norms) else 0.0
        if largest > max_step:
            dx *= max_step / largest
        x = x + dx

        energy, grad = energy_and_gradient(x, topo)
        forces = -grad
        max_force = (
            float(np.max(np.linalg.norm(forces, axis=1))) if len(x) else 0.0
        )

    return x, OptimizeResult(max_force < f_tol, energy, step, max_force)


# --- RDKit boundary ---------------------------------------------------------


def topology_from_rdkit(mol: Any, electronic_effects: bool = False) -> Topology:
    """Build a :class:`Topology` from an RDKit molecule's connectivity.

    With *electronic_effects* enabled, QEq partial charges are derived from
    the conformer geometry (adding a shielded Coulomb term) and 4-coordinate
    d8 metal centers get square-planar angle targets.
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
    if electronic_effects and coords is not None:
        total = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        charges = qeq_charges(atomic_numbers, coords, total)

    return build_topology(
        atomic_numbers,
        bond_pairs,
        hybridizations,
        bond_orders,
        charges=charges,
        square_planar_metals=electronic_effects,
        coords=coords,
        vdw_cutoff=_VDW_CUTOFF_A,
    )


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
    coords = _conformer_coords(mol)
    if coords is None:
        return None
    topo = topology_from_rdkit(mol, electronic_effects=electronic_effects)
    energy, _ = energy_and_gradient(coords, topo)
    return energy


def optimize_rdkit_mol(
    mol: Any,
    max_iter: int = 500,
    f_tol: float = 1e-3,
    electronic_effects: bool = False,
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

    topo = topology_from_rdkit(mol, electronic_effects=electronic_effects)
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
