# PMEFF — Python Molecular Editor Force Field

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
| **PMEFF (Universal)** | Right-click the *Optimize 3D* button | Relaxes the current 3D geometry with a dependency-free FIRE optimizer. |
| **PMEFF Single-Point Energy** | *Analysis* menu | Reports the force-field energy of the current geometry without modifying it. |

## The physics

PMEFF is a five-term force field:

- **Bonds** — harmonic, `½·k·(r − r₀)²`, with the rest length `r₀` taken as the
  sum of the two atoms' covalent radii, scaled down by bond order (double
  ×0.89, triple ×0.78, aromatic interpolated → C(ar)–C(ar) 1.42 Å).
- **Angles** — harmonic in the bend angle, `½·k·(θ − θ₀)²`, with the ideal angle
  `θ₀` inferred from the central atom's hybridization (falling back to its
  coordination number for metals and other cases where hybridization is
  ambiguous). Two special cases: in-ring angles of three-membered rings take
  their target from the law of cosines over the bond rest lengths (60° in
  cyclopropane), so bonds and angles share one minimum; and linear sp centers
  use `k·(1 + cos θ)`, which matches the harmonic curvature at 180° but keeps
  the gradient finite at the linear minimum.
- **Torsions** — a cosine dihedral potential `½·V·(1 + cos(n·φ − γ))`: 2-fold
  for sp²–sp² bonds (keeps double bonds and conjugated systems planar), 3-fold
  for sp³–sp³ bonds (staggered minima), and a weak 6-fold term for mixed
  sp²–sp³ bonds. The per-bond barrier is split evenly over all dihedrals
  sharing the bond, UFF-style, so it doesn't grow with substitution.
- **Out-of-plane** — a harmonic penalty on the pyramidalization of
  3-coordinate sp² centers, expressed through the sum of the three bend angles
  around the center (planar ⇔ 360°).
- **van der Waals** — a Lennard-Jones 12-6 term whose per-atom radius is the
  covalent radius plus a fixed 0.90 Å offset. This reproduces the tabulated vdW
  radii of the common elements to within ~0.05 Å (C → 1.65 Å vs. 1.70 Å,
  O → 1.53 Å vs. 1.52 Å, H → 1.22 Å vs. 1.20 Å). 1-2 and 1-3 pairs are
  excluded; 1-4 pairs get the conventional half well depth.

Geometry optimization uses **FIRE** (Fast Inertial Relaxation Engine) with a
per-atom displacement clamp for stability, plus fully **analytical gradients**
for all five energy terms — including the dihedral derivatives, which are
verified against numeric differentiation in the test suite. All terms are
evaluated with vectorized numpy over precompiled index arrays, so evaluation
cost is dominated by numpy kernels rather than Python loops.

> **Note:** PMEFF is a fast, universal *geometry-cleanup* force field, not a
> replacement for quantum-chemical optimization. Its energies are in internal,
> consistent units and are meant for relative comparison and clash removal, not
> for reporting thermochemistry. For accurate energies, pair it with the ORCA or
> PySCF plugins.

## Installation

Copy the `force_field_plugin/` folder into your MoleditPy user plugin
directory:

- **Windows:** `C:\Users\<YourName>\.moleditpy\plugins\`
- **Linux / macOS:** `~/.moleditpy/plugins/`

Then restart MoleditPy, or use **Plugins → Reload All Plugins**.

## Development

```bash
# Run the test suite (uses real numpy + rdkit)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=force_field_plugin --cov-report=term-missing

# Lint
pylint force_field_plugin/
```

The engine (`force_field_plugin/forcefield.py`) is deliberately Qt-free and
RDKit-free at its core — it operates on a plain-number `Topology` — which makes
it fully unit-testable without a GUI. Only the thin boundary functions touch
RDKit.

## License

GPL-3.0. See [LICENSE](LICENSE).
