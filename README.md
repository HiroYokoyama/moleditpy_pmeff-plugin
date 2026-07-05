# PMEFF — Python Molecular Editor Force Field

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21151897.svg)](https://doi.org/10.5281/zenodo.21151897)
[![Tests](https://github.com/HiroYokoyama/moleditpy_pmeff-plugin/actions/workflows/test.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_pmeff-plugin/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pmeff.svg)](https://pypi.org/project/pmeff/)
[![Downloads](https://img.shields.io/github/downloads/HiroYokoyama/moleditpy_pmeff-plugin/total)](https://github.com/HiroYokoyama/moleditpy_pmeff-plugin/releases)


A [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor) plugin that
adds **PMEFF**, a self-contained *universal* molecular force field covering the
**entire periodic table** (Z = 1 – 118). It needs no external QM binary — just
`numpy` and `rdkit`.

## Why another force field?

The classic force fields shipped with molecular editors are excellent for
organic chemistry but leave gaps: MMFF94 only parameterizes a fixed subset of
main-group elements, and even UFF-style tables are hand-tuned per element.

PMEFF takes a different approach. Every parameter is **derived from a single
per-element property** — the Pyykkö single-bond covalent radius — so *no element
is ever missing a parameter*. Drop in a lanthanide, an actinide, a transition
metal, or a superheavy element and PMEFF still produces a sensible geometry.

## What it does

Once installed, PMEFF registers:

| Feature | Where | Description |
|---|---|---|
| **PMEFF** | Right-click the *Optimize 3D* button | Relaxes the current 3D geometry with a dependency-free FIRE 2.0 + L-BFGS optimizer. |
| **PMEFF Single-Point Energy** | *Analysis* menu | Reports the force-field energy (with per-term decomposition) without modifying the geometry. |
| **PMEFF Minimum Check (Vibrational)** | *Analysis* menu | Diagonalizes the Hessian at the current geometry and reports whether it is a true minimum or a saddle point. |
| **Metal Geometry Override…** | *3D Edit* menu (also a button in *PMEFF Settings*) | Opens a modeless table to force the coordination geometry of individual atoms (see below). |
| **PMEFF Settings…** | *Settings → PMEFF Setting* menu | Opens a dialog to toggle individual physics options (persisted to `settings.json`). |

> **Scope:** PMEFF is a pre-DFT *geometry-cleanup* force field. Its goal is
> **initial structure preparation** — removing clashes, correcting bond lengths,
> angles and torsions, and placing metal centers in the right coordination
> geometry. It is not designed for high-accuracy thermochemistry. For accurate
> energies, pair it with the ORCA or PySCF plugins.

## The physics

For the full specification — functional forms, parameter derivations,
gradient formulas, non-bonded bookkeeping and the optimizer — see the
[technical reference](docs/TECHNICAL.md). In brief, PMEFF has these energy terms:

- **Bonds** — by default **Morse potential** `D(1 − e^{−α Δr})²` (bounded above
  at the dissociation energy D; same curvature as harmonic at the minimum). The
  harmonic `½·k·(r − r₀)²` is available as a fallback. Rest lengths and force
  constants follow the same covalent-radius rules in both cases, with a
  **polar-bond contraction** (a capped, quadratic function of the
  electronegativity difference) that pulls in bonds the plain radius sum leaves
  too long — Si–O 1.79→1.63 Å, P=O, B–O, the metal-oxides and C–F — while
  leaving organic C–C/C–O/C–N/C–H unchanged. Toggleable in settings.
- **Angles** — harmonic in the bend angle, `½·k·(θ − θ₀)²`, with the ideal angle
  `θ₀` inferred from the central atom's hybridization (falling back to its
  coordination number for metals and other cases where hybridization is
  ambiguous). sp³ pnictogens and chalcogens are compressed below tetrahedral
  by their lone pairs (`4 − coordination` of them): mildly for period-2 atoms
  (NH₃ ≈ 107°, H₂O ≈ 104.5°), strongly for heavier congeners that bond through
  near-pure p orbitals (H₂S/PH₃ ≈ 93°). The period-2 compression is calibrated
  on hydrides and opens back toward tetrahedral for bulkier substituents
  (dimethyl ether's C-O-C ≈ 110°, not 104.5°). Two more special cases: in-ring angles
  of three-membered rings take their target from the law of cosines over the
  bond rest lengths (60° in cyclopropane), so bonds and angles share one
  minimum; and linear sp centers use `k·(1 + cos θ)`, which matches the
  harmonic curvature at 180° but keeps the gradient finite at the linear
  minimum.
- **Torsions** — a cosine dihedral potential `½·V·(1 + cos(n·φ − γ))`: 2-fold
  for sp²–sp² bonds (keeps double bonds and conjugated systems planar), 3-fold
  for sp³–sp³ bonds (staggered minima), and a weak 6-fold term for mixed
  sp²–sp³ bonds. The per-bond barrier is split evenly over all dihedrals
  sharing the bond, UFF-style, so it doesn't grow with substitution, and the
  2-fold barrier scales with the π character of the central bond — full for a
  double bond, reduced for aromatic, weak for a conjugated single bond, so
  biphenyl can twist while ethylene stays rigid.
- **Out-of-plane** — a harmonic penalty on the pyramidalization of
  3-coordinate sp² centers, expressed through the sum of the three bend angles
  around the center (planar ⇔ 360°).
- **van der Waals** — a Lennard-Jones 12-6 term whose per-atom radius is the
  covalent radius plus a fixed 0.90 Å offset. This reproduces the tabulated vdW
  radii of the common elements to within ~0.05 Å (C → 1.65 Å vs. 1.70 Å,
  O → 1.53 Å vs. 1.52 Å, H → 1.22 Å vs. 1.20 Å). Well depths are derived per
  atom from the covalent radius as a polarizability proxy (carbon anchors the
  scale) and combined with the Lorentz–Berthelot geometric mean. 1-2 and 1-3
  pairs are excluded; 1-4 pairs get the conventional half well depth.

### Optional physics terms

**Settings → PMEFF Setting** opens a dialog. All settings are persisted in
`pmeff_plugin/settings.json`. The optional terms (defaults shown) are:

- **Electronic effects** *(on)* — QEq partial charges from Slater Zeff and
  Allred–Rochow electronegativities feed a shielded Coulomb term; charges are
  re-solved as the geometry relaxes (envelope theorem keeps the gradient exact).
  Also enables square-planar angle targets for 4-coordinate Ni/Pd/Pt/Rh/Ir/Au
  and octahedral targets for 6-coordinate d-block transition metals; cis/trans
  assignment is read from the starting geometry so tetrahedral Pd²⁺ converges
  to square-planar.
- **Morse bond stretching** *(on)* — replaces harmonic bonds with the Morse
  potential `D(1−e^{−α Δr})²`; bounded above at D, same curvature at the
  minimum. Improves robustness for severely distorted starting geometries at
  negligible computational cost.
- **Hydrogen bond correction** *(on)* — a geometry-dependent D−H···A term
  (donors/acceptors: N, O, F, S) with a 12-6 radial profile and cos²(∠DHA)
  angular dependence. Improves H-bond distances and linearity. Fast.
- **Dispersion correction** *(off by default)* — Becke-Johnson damped C₆/r⁶
  London dispersion added on top of the LJ term. Improves aromatic stacking
  distances and hydrophobic contacts. Can slow optimization of large molecules.

Lone pairs are not explicit particles, but their steric effect enters through
the hybridization-derived angle targets: sp³ N stays pyramidal, sp³ O stays
bent, and a conjugated (sp²) amide nitrogen stays planar.

### Per-atom geometry override

Connectivity alone cannot always determine a coordination geometry — a
4-coordinate metal may be square-planar *or* tetrahedral, and a bare metal
centre may have no reliable hybridization at all. **3D Edit → Metal Geometry
Override** (also a button in *PMEFF Settings*) opens a modeless table where you
can force the geometry of individual atoms:

| Override | Ideal L–M–L angles |
|---|---|
| **Auto** *(default)* | Hybridization / auto-metal detection (unchanged) |
| **Linear** | 180° |
| **Trigonal Planar** | 120°, kept planar |
| **Square Planar** | cis 90° / trans 180° |
| **Tetrahedral** | 109.47° |
| **Octahedral** | cis 90° / trans 180° |

The table lists the current molecule's atoms with a *Show metals only* filter
(on by default — uncheck it to override carbon or any other element). Click an
atom in the 3D view to jump to its row; selected rows are highlighted in 3D.
Overrides are **entirely opt-in**: an atom left on *Auto* behaves exactly as
before, so the shipped defaults (including the d8 square-planar / octahedral
auto-detection) are unchanged. They are also **independent of the electronic-
effects setting** — a forced geometry applies whether or not electronic effects
are on. Press **Apply**, then run *Optimize 3D (PMEFF)* to relax under the new
targets. Overrides are saved with the project and restored when it is reopened.

Geometry optimization uses **FIRE 2.0** (Fast Inertial Relaxation Engine) for
the far-from-minimum regime and hands over to an **L-BFGS finisher** in the
quadratic basin, with a per-atom displacement clamp for stability and fully
**analytical gradients** for all energy terms — including the dihedral and
Coulomb derivatives, which are verified against numeric differentiation in
the test suite. All terms are evaluated with vectorized numpy over
precompiled index arrays, so evaluation cost is dominated by numpy kernels
rather than Python loops. The short-range van der Waals term is truncated at
a 12 Å cutoff (electrostatics, being long-range, are not) with the pair list
built by an O(N) cell-list search and maintained as a Verlet list during
optimization; a CHARMM-style switching function tapers the LJ term to zero
over the last 2 Å with zero slope at the cutoff, so both energy and force
stay continuous as a pair crosses the boundary. A finite-difference Hessian
over the analytic gradient powers the vibrational minimum check.

> **Note:** PMEFF energies are in internal, consistent units meant for
> **relative comparison and geometry guidance**, not thermochemistry. The goal is
> to give the DFT optimizer a sensible starting geometry — not to replace it.

## Installation

Copy the `pmeff_plugin/` folder into your MoleditPy user plugin
directory:

- **Windows:** `C:\Users\<YourName>\.moleditpy\plugins\`
- **Linux / macOS:** `~/.moleditpy/plugins/`

Then restart MoleditPy, or use **Plugins → Reload All Plugins**.

## Standalone Python package (`pmeff`)

The force-field engine is also published on PyPI as **`pmeff`** — the same
physics, usable from any Python script without MoleditPy. The core has **no
dependency beyond NumPy**; RDKit is an optional extra for the convenience layer.

```bash
pip install pmeff            # core: NumPy only
pip install "pmeff[rdkit]"   # adds the RDKit convenience layer (optimize_mol)
```

### With RDKit — `Mol` in, relaxed `Mol` out

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
`use_polar_contraction`) plus `max_iter` and `f_tol`. It also accepts
`geometry_overrides` — a `{atom_index: name}` mapping (`"linear"`,
`"trigonal_planar"`, `"square_planar"`, `"tetrahedral"`, `"octahedral"`) that
forces the coordination geometry of individual atoms:

```python
# Force atom 1 to square-planar regardless of its detected hybridization.
mol, result = pmeff.optimize_mol(mol, geometry_overrides={1: "square_planar"})
```

### Without RDKit — plain arrays in, coordinates out

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
sharpen the result; `optimize_coords` also accepts the same `geometry_overrides`
mapping as `optimize_mol`. The lower-level engine — `build_topology`,
`energy_and_gradient`, `energy_components`, `optimize`, `vibrational_analysis` —
is exported too for custom workflows.

For the full programmatic guide — every option, the low-level engine, worked
recipes and an API reference — see **[docs/USAGE.md](docs/USAGE.md)**.

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
RDKit-free at its core — it operates on a plain-number `Topology` — which makes
it fully unit-testable without a GUI. Only the thin boundary functions touch
RDKit.

## License

GPL-3.0. See [LICENSE](LICENSE).
