# PMEFF 窶・Python Molecular Editor Force Field

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21151897.svg)](https://doi.org/10.5281/zenodo.21151897)
[![Tests](https://github.com/HiroYokoyama/moleditpy_pmeff-plugin/actions/workflows/test.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_pmeff-plugin/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pmeff.svg)](https://pypi.org/project/pmeff/)
[![Downloads](https://img.shields.io/github/downloads/HiroYokoyama/moleditpy_pmeff-plugin/total)](https://github.com/HiroYokoyama/moleditpy_pmeff-plugin/releases)

A [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor) plugin that
adds **PMEFF**, a self-contained *universal* molecular force field covering the
**entire periodic table** (Z = 1 窶・118). It needs no external QM binary 窶・just
`numpy` and `rdkit`.

## Why another force field?

The classic force fields shipped with molecular editors are excellent for
organic chemistry but leave gaps: MMFF94 only parameterizes a fixed subset of
main-group elements, and even UFF-style tables are hand-tuned per element.

PMEFF takes a different approach. Every parameter is **derived from a single
per-element property** 窶・the Pyykkﾃｶ single-bond covalent radius 窶・so *no element
is ever missing a parameter*. Drop in a lanthanide, an actinide, a transition
metal, or a superheavy element and PMEFF still produces a sensible geometry.

## What it does

Once installed, PMEFF registers:

| Feature | Where | Description |
|---|---|---|
| **PMEFF** | Right-click the *Optimize 3D* button | Relaxes the current 3D geometry with a dependency-free FIRE 2.0 + L-BFGS optimizer. |
| **PMEFF Single-Point Energy** | *Analysis* menu | Reports the force-field energy (with per-term decomposition) without modifying the geometry. |
| **PMEFF Minimum Check (Vibrational)** | *Analysis* menu | Diagonalizes the Hessian at the current geometry and reports whether it is a true minimum or a saddle point. |
| **PMEFF Settings窶ｦ** | *Settings 竊・PMEFF Setting* menu | Opens a dialog to toggle individual physics options (persisted to `settings.json`). |

> **Scope:** PMEFF is a pre-DFT *geometry-cleanup* force field. Its goal is
> **initial structure preparation** 窶・removing clashes, correcting bond lengths,
> angles and torsions, and placing metal centers in the right coordination
> geometry. It is not designed for high-accuracy thermochemistry. For accurate
> energies, pair it with the ORCA or PySCF plugins.

## The physics

For the full specification 窶・functional forms, parameter derivations,
gradient formulas, non-bonded bookkeeping and the optimizer 窶・see the
[technical reference](docs/TECHNICAL.md). In brief, PMEFF has these energy terms:

- **Bonds** 窶・by default **Morse potential** `D(1 竏・e^{竏槻ｱ ﾎ排})ﾂｲ` (bounded above
  at the dissociation energy D; same curvature as harmonic at the minimum). The
  harmonic `ﾂｽﾂｷkﾂｷ(r 竏・r竄)ﾂｲ` is available as a fallback. Rest lengths and force
  constants follow the same covalent-radius rules in both cases, with a
  **polar-bond contraction** (a capped, quadratic function of the
  electronegativity difference) that pulls in bonds the plain radius sum leaves
  too long 窶・Si窶徹 1.79竊・.63 ﾃ・ P=O, B窶徹, the metal-oxides and C窶擢 窶・while
  leaving organic C窶鼎/C窶徹/C窶哲/C窶滴 unchanged. Toggleable in settings.
- **Angles** 窶・harmonic in the bend angle, `ﾂｽﾂｷkﾂｷ(ﾎｸ 竏・ﾎｸ竄)ﾂｲ`, with the ideal angle
  `ﾎｸ竄` inferred from the central atom's hybridization (falling back to its
  coordination number for metals and other cases where hybridization is
  ambiguous). spﾂｳ pnictogens and chalcogens are compressed below tetrahedral
  by their lone pairs (`4 竏・coordination` of them): mildly for period-2 atoms
  (NH竄・竕・107ﾂｰ, H竄０ 竕・104.5ﾂｰ), strongly for heavier congeners that bond through
  near-pure p orbitals (H竄４/PH竄・竕・93ﾂｰ). The period-2 compression is calibrated
  on hydrides and opens back toward tetrahedral for bulkier substituents
  (dimethyl ether's C-O-C 竕・110ﾂｰ, not 104.5ﾂｰ). Two more special cases: in-ring angles
  of three-membered rings take their target from the law of cosines over the
  bond rest lengths (60ﾂｰ in cyclopropane), so bonds and angles share one
  minimum; and linear sp centers use `kﾂｷ(1 + cos ﾎｸ)`, which matches the
  harmonic curvature at 180ﾂｰ but keeps the gradient finite at the linear
  minimum.
- **Torsions** 窶・a cosine dihedral potential `ﾂｽﾂｷVﾂｷ(1 + cos(nﾂｷﾏ・竏・ﾎｳ))`: 2-fold
  for spﾂｲ窶都pﾂｲ bonds (keeps double bonds and conjugated systems planar), 3-fold
  for spﾂｳ窶都pﾂｳ bonds (staggered minima), and a weak 6-fold term for mixed
  spﾂｲ窶都pﾂｳ bonds. The per-bond barrier is split evenly over all dihedrals
  sharing the bond, UFF-style, so it doesn't grow with substitution, and the
  2-fold barrier scales with the ﾏ character of the central bond 窶・full for a
  double bond, reduced for aromatic, weak for a conjugated single bond, so
  biphenyl can twist while ethylene stays rigid.
- **Out-of-plane** 窶・a harmonic penalty on the pyramidalization of
  3-coordinate spﾂｲ centers, expressed through the sum of the three bend angles
  around the center (planar 竍・360ﾂｰ).
- **van der Waals** 窶・a Lennard-Jones 12-6 term whose per-atom radius is the
  covalent radius plus a fixed 0.90 ﾃ・offset. This reproduces the tabulated vdW
  radii of the common elements to within ~0.05 ﾃ・(C 竊・1.65 ﾃ・vs. 1.70 ﾃ・
  O 竊・1.53 ﾃ・vs. 1.52 ﾃ・ H 竊・1.22 ﾃ・vs. 1.20 ﾃ・. Well depths are derived per
  atom from the covalent radius as a polarizability proxy (carbon anchors the
  scale) and combined with the Lorentz窶釘erthelot geometric mean. 1-2 and 1-3
  pairs are excluded; 1-4 pairs get the conventional half well depth.

### Optional physics terms

**Settings 竊・PMEFF Setting** opens a dialog. All settings are persisted in
`pmeff_plugin/settings.json`. The optional terms (defaults shown) are:

- **Electronic effects** *(on)* 窶・QEq partial charges from Slater Zeff and
  Allred窶迭ochow electronegativities feed a shielded Coulomb term; charges are
  re-solved as the geometry relaxes (envelope theorem keeps the gradient exact).
  Also enables square-planar angle targets for 4-coordinate Ni/Pd/Pt/Rh/Ir/Au
  and octahedral targets for 6-coordinate d-block transition metals; cis/trans
  assignment is read from the starting geometry so tetrahedral Pdﾂｲ竅ｺ converges
  to square-planar.
- **Morse bond stretching** *(on)* 窶・replaces harmonic bonds with the Morse
  potential `D(1竏弾^{竏槻ｱ ﾎ排})ﾂｲ`; bounded above at D, same curvature at the
  minimum. Improves robustness for severely distorted starting geometries at
  negligible computational cost.
- **Hydrogen bond correction** *(on)* 窶・a geometry-dependent D竏辿ﾂｷﾂｷﾂｷA term
  (donors/acceptors: N, O, F, S) with a 12-6 radial profile and cosﾂｲ(竏DHA)
  angular dependence. Improves H-bond distances and linearity. Fast.
- **Dispersion correction** *(off by default)* 窶・Becke-Johnson damped C竄・r竅ｶ
  London dispersion added on top of the LJ term. Improves aromatic stacking
  distances and hydrophobic contacts. Can slow optimization of large molecules.

Lone pairs are not explicit particles, but their steric effect enters through
the hybridization-derived angle targets: spﾂｳ N stays pyramidal, spﾂｳ O stays
bent, and a conjugated (spﾂｲ) amide nitrogen stays planar.

Geometry optimization uses **FIRE 2.0** (Fast Inertial Relaxation Engine) for
the far-from-minimum regime and hands over to an **L-BFGS finisher** in the
quadratic basin, with a per-atom displacement clamp for stability and fully
**analytical gradients** for all energy terms 窶・including the dihedral and
Coulomb derivatives, which are verified against numeric differentiation in
the test suite. All terms are evaluated with vectorized numpy over
precompiled index arrays, so evaluation cost is dominated by numpy kernels
rather than Python loops. The short-range van der Waals term is truncated at
a 12 ﾃ・cutoff (electrostatics, being long-range, are not) with the pair list
built by an O(N) cell-list search and maintained as a Verlet list during
optimization; a CHARMM-style switching function tapers the LJ term to zero
over the last 2 ﾃ・with zero slope at the cutoff, so both energy and force
stay continuous as a pair crosses the boundary. A finite-difference Hessian
over the analytic gradient powers the vibrational minimum check.

> **Note:** PMEFF energies are in internal, consistent units meant for
> **relative comparison and geometry guidance**, not thermochemistry. The goal is
> to give the DFT optimizer a sensible starting geometry 窶・not to replace it.

## Installation

Copy the `pmeff_plugin/` folder into your MoleditPy user plugin
directory:

- **Windows:** `C:\Users\<YourName>\.moleditpy\plugins\`
- **Linux / macOS:** `~/.moleditpy/plugins/`

Then restart MoleditPy, or use **Plugins 竊・Reload All Plugins**.

## Standalone Python package (`pmeff`)

The force-field engine is also published on PyPI as **`pmeff`** 窶・the same
physics, usable from any Python script without MoleditPy. The core has **no
dependency beyond NumPy**; RDKit is an optional extra for the convenience layer.

```bash
pip install pmeff            # core: NumPy only
pip install "pmeff[rdkit]"   # adds the RDKit convenience layer (optimize_mol)
```

### With RDKit 窶・`Mol` in, relaxed `Mol` out

The recommended path: hand it an RDKit molecule that has a 3D conformer, and get
the **same molecule back with its geometry relaxed**. Connectivity, bond orders,
formal charges and properties are all preserved (a raw coordinate list would
throw them away).

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import pmeff

mol = Chem.AddHs(Chem.MolFromSmiles("O[SiH3]"))   # silanol
AllChem.EmbedMolecule(mol, randomSeed=1)          # give it a 3D conformer

mol, result = pmeff.optimize_mol(mol)             # relaxed in place, and returned
print(result.converged, result.energy, result.max_force)
print(Chem.MolToXYZBlock(mol))                    # -> optimized coordinates
```

`optimize_mol` takes the same physics switches as the plugin
(`electronic_effects`, `use_morse`, `use_hbond`, `use_dispersion`,
`use_polar_contraction`) plus `max_iter` and `f_tol`.

### Without RDKit 窶・plain arrays in, coordinates out

For pipelines that don't use RDKit, describe the molecule with atomic numbers, a
bond list and coordinates:

```python
import numpy as np
import pmeff

atomic_numbers = [8, 1, 1]                         # water
bonds = [(0, 1), (0, 2)]
coords = np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.3, 0.9, 0.0]])

coords, result = pmeff.optimize_coords(
    atomic_numbers, bonds, coords, hybridizations=["SP3", None, None]
)
print(result.converged)
print(coords)                                      # optimized (N, 3) array
```

Optional charges (`pmeff.qeq_charges(...)`), per-bond orders and hybridizations
sharpen the result. The lower-level engine 窶・`build_topology`,
`energy_and_gradient`, `energy_components`, `optimize`, `vibrational_analysis` 窶・is exported too for custom workflows.

For the full programmatic guide 窶・every option, the low-level engine, worked
recipes and an API reference 窶・see **[docs/USAGE.md](docs/USAGE.md)**.

> The engine is a copy of `pmeff_plugin/forcefield.py` (the single source of
> truth); see [PACKAGING.md](PACKAGING.md) for the build/release process.

## Development

```bash
# Run the test suite (uses real numpy + rdkit)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=pmeff_plugin --cov-report=term-missing

# Lint
pylint pmeff_plugin/
```

The engine (`pmeff_plugin/forcefield.py`) is deliberately Qt-free and
RDKit-free at its core 窶・it operates on a plain-number `Topology` 窶・which makes
it fully unit-testable without a GUI. Only the thin boundary functions touch
RDKit.

## License

GPL-3.0. See [LICENSE](LICENSE).
