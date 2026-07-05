"""PMEFF — a self-contained universal force field (Z = 1..118).

``pmeff`` is the standalone, pip-installable distribution of the PMEFF engine
that ships inside the MoleditPy *PMEFF Plugin*. It is a small, NumPy-only
molecular force field whose every parameter is derived from a single
per-element property (the Pyykko covalent radius), so no element is ever
missing. It provides bonded (bond/angle/torsion/out-of-plane), van der Waals,
electrostatic (QEq), hydrogen-bond and dispersion terms, all with analytical
gradients, plus a dependency-free FIRE + L-BFGS optimizer.

Two ways to use it:

* **With RDKit** (``pip install "pmeff[rdkit]"``) — the convenient path. Hand it
  an RDKit ``Mol`` that has a 3D conformer; get the same ``Mol`` back with the
  conformer relaxed (bonds, charges and properties preserved):

      from pmeff import optimize_mol
      mol, result = optimize_mol(mol)            # relaxed in place, and returned

* **Without RDKit** (``pip install pmeff``) — the pure-NumPy path. Pass atomic
  numbers, a bond list and coordinates; get optimized coordinates back:

      from pmeff import optimize_coords
      coords, result = optimize_coords(atomic_numbers, bonds, coords)

The lower-level engine (``Topology``, ``build_topology``,
``energy_and_gradient``, ``optimize``, ``vibrational_analysis``, ...) is
re-exported for advanced use.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ._version import __version__
from .forcefield import (
    OptimizeResult,
    Topology,
    bond_rest_length,
    build_topology,
    energy_and_gradient,
    energy_components,
    optimize,
    qeq_charges,
    vibrational_analysis,
)

__all__ = [
    "__version__",
    "optimize_mol",
    "optimize_coords",
    "OptimizeResult",
    "Topology",
    "build_topology",
    "energy_and_gradient",
    "energy_components",
    "optimize",
    "qeq_charges",
    "vibrational_analysis",
    "bond_rest_length",
]


def optimize_mol(
    mol: Any,
    max_iter: int = 1000,
    f_tol: float = 1e-3,
    electronic_effects: bool = True,
    use_morse: bool = True,
    use_hbond: bool = True,
    use_dispersion: bool = False,
    use_polar_contraction: bool = True,
    geometry_overrides: Optional[dict] = None,
) -> Tuple[Any, Optional[OptimizeResult]]:
    """Relax an RDKit ``Mol``'s 3D conformer in place with PMEFF.

    Returns ``(mol, result)`` — the *same* molecule object (its conformer
    updated), so connectivity, bond orders, formal charges and properties are
    all preserved, plus an :class:`OptimizeResult` (``None`` for a single-atom
    molecule with nothing to do). Requires RDKit and a molecule that already
    carries a conformer; raises ``ImportError`` if RDKit is not installed and
    ``ValueError`` if the molecule has no 3D coordinates.

    The defaults mirror the shipped plugin (electronic effects, Morse bonds,
    H-bonds and the polar-bond contraction on; dispersion off).

    *geometry_overrides* optionally forces the coordination geometry of
    individual atoms — a ``{atom_index: name}`` mapping where *name* is one of
    ``"linear"``, ``"trigonal_planar"``, ``"square_planar"``, ``"tetrahedral"``
    or ``"octahedral"`` (chiefly for metal centers). Atoms not listed keep their
    default geometry.
    """
    try:
        from .forcefield import optimize_rdkit_mol
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError("pmeff.optimize_mol requires RDKit") from exc

    success, result = optimize_rdkit_mol(
        mol,
        max_iter=max_iter,
        f_tol=f_tol,
        electronic_effects=electronic_effects,
        use_morse=use_morse,
        use_hbond=use_hbond,
        use_dispersion=use_dispersion,
        use_polar_contraction=use_polar_contraction,
        geometry_overrides=geometry_overrides,
    )
    if not success:
        raise ValueError("PMEFF: molecule has no 3D conformer to optimize")
    return mol, result


def optimize_coords(
    atomic_numbers: Sequence[int],
    bonds: Sequence[Tuple[int, int]],
    coords: "np.ndarray",
    hybridizations: Optional[Sequence[Optional[str]]] = None,
    bond_orders: Optional[Sequence[float]] = None,
    charges: Optional[Sequence[float]] = None,
    max_iter: int = 1000,
    f_tol: float = 1e-3,
    use_morse: bool = True,
    use_hbond: bool = False,
    use_dispersion: bool = False,
    use_polar_contraction: bool = True,
    geometry_overrides: Optional[dict] = None,
) -> Tuple["np.ndarray", OptimizeResult]:
    """Relax a molecule described by plain arrays — no RDKit required.

    Args:
        atomic_numbers: Atomic number of every atom (length N).
        bonds: ``(i, j)`` index pairs describing the covalent bonds.
        coords: Initial coordinates, shape ``(N, 3)`` (not modified in place).
        hybridizations: Optional per-atom labels ("SP", "SP2", "SP3", ...) that
            improve angle/torsion assignment; omit for a coordination-based
            fallback.
        bond_orders: Optional per-bond orders aligned with *bonds*.
        charges: Optional per-atom partial charges enabling the Coulomb term;
            omit for none, or use :func:`qeq_charges` to derive them.
        geometry_overrides: Optional ``{atom_index: name}`` mapping forcing the
            coordination geometry of individual atoms ("linear",
            "trigonal_planar", "square_planar", "tetrahedral", "octahedral").

    Returns ``(optimized_coords, result)``.
    """
    coords = np.asarray(coords, dtype=float)
    topo = build_topology(
        atomic_numbers,
        bonds,
        hybridizations=hybridizations,
        bond_orders=bond_orders,
        charges=charges,
        coords=coords,
        use_morse=use_morse,
        use_hbond=use_hbond,
        use_dispersion=use_dispersion,
        use_polar_contraction=use_polar_contraction,
        geometry_overrides=geometry_overrides,
    )
    return optimize(coords, topo, max_iter=max_iter, f_tol=f_tol)
