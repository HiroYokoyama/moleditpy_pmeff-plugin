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

PMEFF is a three-term force field:

- **Bonds** — harmonic, `½·k·(r − r₀)²`, with the rest length `r₀` taken as the
  sum of the two atoms' covalent radii.
- **Angles** — harmonic in the bend angle, `½·k·(θ − θ₀)²`, with the ideal angle
  `θ₀` inferred from the central atom's hybridization (falling back to its
  coordination number for metals and other cases where hybridization is
  ambiguous).
- **van der Waals** — a Lennard-Jones 12-6 term whose per-atom radius is the
  covalent radius plus a fixed 0.90 Å offset. This reproduces the tabulated vdW
  radii of the common elements to within ~0.05 Å (C → 1.65 Å vs. 1.70 Å,
  O → 1.53 Å vs. 1.52 Å, H → 1.22 Å vs. 1.20 Å).

Geometry optimization uses **FIRE** (Fast Inertial Relaxation Engine) with a
per-atom displacement clamp for stability, plus fully **analytical gradients**
for all three energy terms, so it converges quickly and without external
solvers.

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
