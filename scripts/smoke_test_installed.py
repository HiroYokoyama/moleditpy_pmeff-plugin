#!/usr/bin/env python3
"""Smoke-test the *installed* ``pmeff`` package (e.g. from PyPI).

Run this from a directory that does NOT contain the repo's local ``pmeff/``
folder, so ``import pmeff`` resolves to the pip-installed distribution rather
than the source tree. The CI workflow copies this file into a temp directory
before running it for exactly that reason.

    python smoke_test_installed.py --extras core     # NumPy-only checks
    python smoke_test_installed.py --extras rdkit     # also the Mol path

Set ``PMEFF_SMOKE_ALLOW_LOCAL=1`` to skip the "must be an installed copy"
guard when trying the script out from the source tree locally.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import pmeff


def _check_installed_origin() -> None:
    path = os.path.abspath(pmeff.__file__)
    if os.environ.get("PMEFF_SMOKE_ALLOW_LOCAL") == "1":
        print(f"[info] importing pmeff from {path} (local allowed)")
        return
    if "site-packages" not in path.replace(os.sep, "/"):
        raise SystemExit(
            f"pmeff was imported from {path}, which is not an installed "
            "copy — run this from a clean directory so the installed package "
            "is under test (or set PMEFF_SMOKE_ALLOW_LOCAL=1)."
        )
    print(f"[ok] installed pmeff {pmeff.__version__} at {path}")


def _check_core() -> None:
    # 1) Pure-NumPy optimization: a stretched water relaxes to ~104.5 deg.
    coords = np.array([[0.0, 0, 0], [1.3, 0, 0], [-0.4, 1.2, 0.0]])
    out, result = pmeff.optimize_coords(
        [8, 1, 1], [(0, 1), (0, 2)], coords, hybridizations=["SP3", None, None]
    )
    assert result.converged, "water optimization did not converge"
    v1, v2 = out[1] - out[0], out[2] - out[0]
    ang = math.degrees(
        math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    )
    assert 100.0 < ang < 108.0, f"H-O-H angle {ang:.1f} deg out of range"
    print(f"[ok] optimize_coords: water H-O-H = {ang:.1f} deg")

    # 2) Energy decomposition returns finite numbers with all keys.
    topo = pmeff.build_topology([6, 6], [(0, 1)])
    comp = pmeff.energy_components(np.array([[0.0, 0, 0], [1.5, 0, 0]]), topo)
    assert math.isfinite(comp["total"])
    for key in ("bond", "angle", "torsion", "oop", "vdw", "elec"):
        assert key in comp
    print(f"[ok] energy_components: total = {comp['total']:.3f}")

    # 3) QEq charges conserve the total charge.
    q = pmeff.qeq_charges([8, 6, 7], coords, total_charge=0.0)
    assert abs(float(np.sum(q))) < 1e-6, "QEq charges do not sum to zero"
    print("[ok] qeq_charges: conserves total charge")

    # 4) Vibrational analysis runs on an optimized diatomic.
    di = pmeff.build_topology([1, 1], [(0, 1)])
    r0 = di.bonds[0][2]
    vib = pmeff.vibrational_analysis(np.array([[0.0, 0, 0], [r0, 0, 0]]), di)
    assert "is_minimum" in vib
    print("[ok] vibrational_analysis: ran")


def _check_rdkit() -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("O[SiH3]"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    returned, result = pmeff.optimize_mol(mol)
    assert returned is mol, "optimize_mol should return the same Mol object"
    assert result.converged, "silanol optimization did not converge"
    conf = returned.GetConformer()
    si_o = float(
        np.linalg.norm(
            np.array(conf.GetAtomPosition(0)) - np.array(conf.GetAtomPosition(1))
        )
    )
    assert 1.58 < si_o < 1.68, f"Si-O = {si_o:.3f} A out of range"
    print(f"[ok] optimize_mol: silanol Si-O = {si_o:.3f} A")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        choices=["core", "rdkit"],
        default="core",
        help="'core' = NumPy-only checks; 'rdkit' also exercises optimize_mol.",
    )
    args = parser.parse_args()

    _check_installed_origin()
    _check_core()
    if args.extras == "rdkit":
        _check_rdkit()
    else:
        print("[info] skipping RDKit checks (core install)")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
