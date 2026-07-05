# PMEFF — Usage Guide (`pmeff` Python package)

This is the practical, example-driven guide to the standalone **`pmeff`** pip
package — the same force-field engine that ships inside the MoleditPy PMEFF
plugin, usable from any Python script.

- For the **physics specification** (functional forms, parameter derivations,
  gradient formulas, optimizer internals) see [TECHNICAL.md](TECHNICAL.md).
- For the **plugin/GUI** side see the top-level [README](../README.md).

> **Scope reminder.** PMEFF is a *pre-DFT geometry-cleanup* force field. Its job
> is to turn a rough structure into a clean one — remove clashes, fix bond
> lengths/angles/torsions, place metal centers in sensible coordination
> geometries — so a DFT optimizer starts from something reasonable. Energies are
> internally consistent for **relative comparison and geometry guidance**, not
> thermochemistry.

---

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [The high-level API](#3-the-high-level-api)
4. [Physics options](#4-physics-options)
5. [Reading the result](#5-reading-the-result-optimizeresult)
6. [Single-point energy & decomposition](#6-single-point-energy--decomposition)
7. [Vibrational / minimum check](#7-vibrational--minimum-check)
8. [QEq partial charges](#8-qeq-partial-charges)
9. [The low-level engine](#9-the-low-level-engine)
10. [Worked recipes](#10-worked-recipes)
11. [Units & interpretation](#11-units--interpretation)
12. [Troubleshooting / FAQ](#12-troubleshooting--faq)
13. [API reference summary](#13-api-reference-summary)

---

## 1. Installation

```bash
pip install pmeff            # core: NumPy only
pip install "pmeff[rdkit]"   # adds the RDKit convenience layer (optimize_mol)
```

The core engine depends on **NumPy only** (`numpy>=1.20`, Python ≥ 3.9). RDKit is
an *optional* extra needed solely for the `optimize_mol` convenience function —
everything else (arrays in, coordinates out) works without it.

Check the installed version:

```python
import pmeff
print(pmeff.__version__)
```

---

## 2. Quick start

### With RDKit — `Mol` in, relaxed `Mol` out (recommended)

Hand it an RDKit molecule that already has a 3D conformer and get the **same
molecule back** with its geometry relaxed. Connectivity, bond orders, formal
charges and properties are all preserved.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import pmeff

mol = Chem.AddHs(Chem.MolFromSmiles("O[SiH3]"))   # silanol
AllChem.EmbedMolecule(mol, randomSeed=1)          # give it a 3D conformer

mol, result = pmeff.optimize_mol(mol)             # relaxed in place, and returned

print(result.converged, result.energy, result.max_force)
print(Chem.MolToXYZBlock(mol))                    # optimized coordinates
```

### Without RDKit — plain arrays in, coordinates out

Describe the molecule with atomic numbers, a bond list and coordinates:

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

---

## 3. The high-level API

Two convenience entry points cover almost every use case.

### `optimize_mol(mol, ...)` — the RDKit path

```python
mol, result = pmeff.optimize_mol(
    mol,
    max_iter=1000,
    f_tol=1e-3,
    electronic_effects=True,
    use_morse=True,
    use_hbond=True,
    use_dispersion=False,
    use_polar_contraction=True,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `mol` | — | An RDKit `Mol` that **already carries a 3D conformer**. |
| `max_iter` | `1000` | Total optimizer iteration budget (FIRE + L-BFGS combined). |
| `f_tol` | `1e-3` | Convergence threshold on the largest per-atom force. |
| `electronic_effects` | `True` | QEq dynamic charges + square-planar/octahedral metal angle targets. |
| `use_morse` | `True` | Morse bond potential instead of harmonic. |
| `use_hbond` | `True` | Explicit D−H···A hydrogen-bond term. |
| `use_dispersion` | `False` | Becke-Johnson damped dispersion. |
| `use_polar_contraction` | `True` | Shorten polar bonds by the electronegativity-difference contraction. |

**Returns** `(mol, result)` — the *same* `Mol` object with its conformer updated,
and an [`OptimizeResult`](#5-reading-the-result-optimizeresult) (`None` for a
single-atom molecule with nothing to do).

**Raises**
- `ImportError` if RDKit is not installed.
- `ValueError` if the molecule has no 3D conformer.

> The defaults deliberately mirror the shipped plugin: electronic effects, Morse
> bonds, H-bonds and polar contraction **on**; dispersion **off**.

### `optimize_coords(atomic_numbers, bonds, coords, ...)` — the array path

```python
coords, result = pmeff.optimize_coords(
    atomic_numbers,
    bonds,
    coords,
    hybridizations=None,
    bond_orders=None,
    charges=None,
    max_iter=1000,
    f_tol=1e-3,
    use_morse=True,
    use_hbond=False,
    use_dispersion=False,
    use_polar_contraction=True,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `atomic_numbers` | — | Atomic number of every atom, length *N*. |
| `bonds` | — | `(i, j)` index pairs describing covalent bonds. |
| `coords` | — | Initial coordinates, shape `(N, 3)`. **Not** modified in place. |
| `hybridizations` | `None` | Per-atom labels `"SP"`, `"SP2"`, `"SP3"`, … — sharpen angle/torsion assignment. Omit for a coordination-number fallback. |
| `bond_orders` | `None` | Per-bond orders aligned with `bonds` (1, 1.5 aromatic, 2, 3). |
| `charges` | `None` | Per-atom partial charges enabling the Coulomb term. Omit for none, or derive with [`qeq_charges`](#8-qeq-partial-charges). |
| `max_iter`, `f_tol` | `1000`, `1e-3` | As above. |
| `use_*` | see signature | Physics switches (see [§4](#4-physics-options)). Note `use_hbond` defaults to **False** here. |

**Returns** `(optimized_coords, result)` — a new `(N, 3)` array and an
`OptimizeResult`.

> Note the two entry points have slightly different defaults: `optimize_mol`
> turns `electronic_effects` and `use_hbond` **on** (matching the plugin);
> `optimize_coords` leaves charges/H-bonds off unless you opt in, because with
> plain arrays there is no formal-charge information to seed QEq from.

---

## 4. Physics options

Each switch toggles one physics term. Defaults are chosen for general-purpose
geometry cleanup; tune them for your system.

| Option | Effect | Turn it on when… |
|---|---|---|
| `electronic_effects` | Derives QEq partial charges from the geometry (adds a shielded Coulomb term, re-solved as the geometry moves) and gives d⁸ / d-block metals square-planar / octahedral angle targets. | You have charged species, polar molecules, or transition-metal complexes. |
| `use_morse` | Morse bond `D(1 − e^{−α Δr})²` instead of harmonic `½k Δr²`. Same curvature at the minimum, but bounded above at *D* — robust through badly stretched starting bonds. | Almost always (default on). |
| `use_hbond` | Explicit D−H···A term (donors/acceptors N, O, F, S). Requires coordinates. | Systems with intra/intermolecular hydrogen bonds. |
| `use_dispersion` | Becke-Johnson damped dispersion pairs alongside the LJ list. | π-stacking, large flexible molecules, van der Waals complexes. |
| `use_polar_contraction` | Shortens polar bond rest lengths by a capped quadratic in the electronegativity difference (Si–O 1.79→1.63 Å, P=O, B–O, metal-oxides, C–F). Organic C–C/C–O/C–N/C–H unchanged. | Default on; leave it unless you specifically want plain covalent-radius sums. |

Both `optimize_mol` and `optimize_coords` also accept `geometry_overrides` — a
`{atom_index: name}` mapping forcing the coordination geometry of individual
atoms, where *name* is one of `"linear"`, `"trigonal_planar"`,
`"square_planar"`, `"tetrahedral"` or `"octahedral"`. It is independent of the
switches above (it works with `electronic_effects` off) and applies to any atom,
not only metals; atoms not listed keep their default geometry.

```python
mol, result = pmeff.optimize_mol(mol, geometry_overrides={0: "square_planar"})
```

---

## 5. Reading the result (`OptimizeResult`)

Both optimizers return an `OptimizeResult` dataclass:

```python
@dataclass
class OptimizeResult:
    converged: bool     # True if max per-atom force fell below f_tol
    energy: float       # final PMEFF energy (internal units)
    steps: int          # optimizer iterations actually taken
    max_force: float    # largest per-atom force magnitude at exit
```

```python
mol, result = pmeff.optimize_mol(mol)
if not result.converged:
    print(f"stalled after {result.steps} steps, |F|max={result.max_force:.4f}")
```

A `False` `converged` usually means the iteration budget ran out — raise
`max_iter`, or check for a pathological starting geometry (overlapping atoms).

---

## 6. Single-point energy & decomposition

To evaluate energy **without** moving atoms, use the engine directly. Build a
topology, then call `energy_components`:

```python
import numpy as np
from pmeff import build_topology, energy_components

topo = build_topology(atomic_numbers, bonds, coords=coords, charges=charges)
comp = energy_components(coords, topo)

for term, e in comp.items():
    print(f"{term:>8}: {e:12.4f}")
```

`energy_components` returns a dict with **all keys always present** (inactive
terms report `0.0`):

| Key | Term |
|---|---|
| `bond` | Bond stretch (Morse or harmonic) |
| `angle` | Angle bend |
| `torsion` | Dihedral torsion |
| `oop` | Out-of-plane / improper |
| `vdw` | Lennard-Jones van der Waals |
| `elec` | Shielded Coulomb electrostatics |
| `hbond` | Hydrogen bond |
| `disp` | Dispersion |
| `total` | Sum of the above |

For energy **and** the analytical gradient in one call:

```python
from pmeff import energy_and_gradient

energy, grad = energy_and_gradient(coords, topo)   # grad has shape (N, 3)
```

Pass a dict as `components=` to fill it with the per-term decomposition while you
compute the gradient.

### RDKit shortcut — `compute_energy` / `compute_energy_components`

If you already have an RDKit `Mol` with a conformer, skip the manual topology
build. These live in `pmeff.forcefield` (not re-exported at the top level):

```python
from pmeff.forcefield import compute_energy, compute_energy_components

e = compute_energy(mol)                                    # bare-default single point
e = compute_energy(mol, electronic_effects=True,           # …or under the same
                   use_morse=True, use_hbond=True)         #   force field you optimize with
comp = compute_energy_components(mol, use_morse=True)      # per-term dict, or None
```

Both accept the **full physics switch set** — `electronic_effects`, `use_morse`,
`use_hbond`, `use_dispersion`, `use_polar_contraction` — so a single point can be
evaluated under exactly the force field `optimize_mol` will use, rather than the
bare harmonic default. They return `None` when the molecule has no conformer.

> The switches all default to *off* (except `use_polar_contraction`), so a plain
> `compute_energy(mol)` is the bare-default single point. To match `optimize_mol`'s
> defaults, pass `electronic_effects=True, use_morse=True, use_hbond=True`.

---

## 7. Vibrational / minimum check

After optimizing, verify the structure is a true minimum rather than a saddle
point:

```python
from pmeff import build_topology, vibrational_analysis

topo = build_topology(atomic_numbers, bonds, coords=coords, charges=charges)
vib = vibrational_analysis(coords, topo)

print("is minimum:", vib["is_minimum"])
print("imaginary modes:", vib["num_imaginary"])
```

Returned dict:

| Key | Meaning |
|---|---|
| `frequencies` | Signed √eigenvalue of the Hessian, ascending. Internal (unit-mass) units, **not** cm⁻¹. Negative values mark imaginary modes. |
| `num_imaginary` | Count of clearly negative eigenvalues (descent directions the optimizer landed *on*). |
| `num_zero` | Count of near-zero modes (rigid-body translations/rotations, plus any genuinely soft modes). |
| `is_minimum` | `True` when no imaginary modes are present. |

This is a **unit-mass** normal-mode analysis over a finite-difference Hessian of
the analytical gradient — good for classifying stationary points, not for
predicting real IR frequencies.

For an RDKit `Mol`, `pmeff.forcefield.check_minimum(mol, ...)` is the same
shortcut as above — it takes the full physics switch set and returns the same
dict (or `None` without a conformer):

```python
from pmeff.forcefield import check_minimum

vib = check_minimum(mol, electronic_effects=True, use_morse=True)
```

---

## 8. QEq partial charges

When you take the array path and want electrostatics, derive charges from the
geometry with the electronegativity-equalization solver:

```python
import numpy as np
from pmeff import qeq_charges, optimize_coords

charges = qeq_charges(
    atomic_numbers,
    coords,
    total_charge=0.0,               # net molecular charge
    hybridizations=hybridizations,  # optional; applies a Bent's-rule scaling
)

coords, result = optimize_coords(
    atomic_numbers, bonds, coords, charges=charges
)
```

`qeq_charges` does a single linear solve minimizing
`Σ(χ_i q_i + ½ η_i q_i²) + Σ_ij J_ij q_i q_j` subject to `Σq = total_charge`,
with an Ohno-shielded Coulomb kernel. It returns a length-*N* NumPy array.

> With the **RDKit path** (`optimize_mol(..., electronic_effects=True)`) this is
> done for you — and the charges are *dynamic*: the optimizer re-solves them as
> the geometry changes. `qeq_charges` on its own gives you a one-shot, fixed set.

---

## 9. The low-level engine

For custom workflows the full engine is re-exported from the top-level package.

### `build_topology(...)` → `Topology`

Assembles a `Topology` — a plain-number, RDKit/Qt-free force-field problem — from
connectivity. Key parameters beyond the ones in `optimize_coords`:

| Parameter | Meaning |
|---|---|
| `square_planar_metals` | Give 4-coordinate d⁸ metals square-planar and 6-coordinate d-block metals octahedral angle targets (needs `coords`). |
| `coords` | Enables the Verlet LJ pair list, coordinate-based metal angles, and H-bond partner detection. |
| `vdw_cutoff` | Distance (Å) beyond which LJ pairs are dropped (needs `coords`). Electrostatics are never truncated. |
| `use_morse`, `use_dispersion`, `use_hbond`, `use_polar_contraction` | Physics switches, as in [§4](#4-physics-options). |

### `optimize(coords, topo, ...)` → `(coords, OptimizeResult)`

The dependency-free minimizer: **FIRE 2.0** far from the minimum (robust through
clashes and rearrangements), handing over to an **L-BFGS** finisher once the
largest per-atom force drops below the crossover, for superlinear convergence.
Geometry-dependent pair data (Verlet LJ list, dynamic QEq charges) is refreshed
whenever an atom drifts more than half the list skin.

| Parameter | Default | Meaning |
|---|---|---|
| `max_iter` | `500` | Total iteration budget across both phases. |
| `f_tol` | `1e-3` | Convergence threshold on the largest per-atom force. |
| `max_step` | `0.20` | Maximum distance (Å) any atom may move in one step. |

### A full custom pipeline

```python
import numpy as np
from pmeff import (
    build_topology, qeq_charges, energy_components,
    optimize, vibrational_analysis,
)

atomic_numbers = [...]
bonds = [...]
coords = np.array([...])
hybs = [...]

# 1. charges from the starting geometry
q = qeq_charges(atomic_numbers, coords, total_charge=0.0, hybridizations=hybs)

# 2. build the force-field problem
topo = build_topology(
    atomic_numbers, bonds,
    hybridizations=hybs, charges=q, coords=coords,
    vdw_cutoff=12.0, use_morse=True, use_hbond=True,
)

# 3. energy before
print("E before:", energy_components(coords, topo)["total"])

# 4. relax
coords, result = optimize(coords, topo, max_iter=1000, f_tol=1e-3)

# 5. verify it is a minimum
print("converged:", result.converged, "min:", vibrational_analysis(coords, topo)["is_minimum"])
```

---

## 10. Worked recipes

### Batch-optimize a set of RDKit molecules

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import pmeff

def relax(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) != 0:
        return None                       # embedding failed
    mol, result = pmeff.optimize_mol(mol)
    return mol, result.energy

for smi in ["CCO", "c1ccccc1", "O[SiH3]", "[Fe]"]:
    out = relax(smi)
    if out:
        _, e = out
        print(f"{smi:12} E = {e:.3f}")
```

### Rank conformers by PMEFF energy

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import pmeff

mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
cids = AllChem.EmbedMultipleConfs(mol, numConfs=20, randomSeed=1)

energies = []
for cid in cids:
    single = Chem.Mol(mol, False, cid)    # isolate one conformer
    single.RemoveAllConformers()
    single.AddConformer(mol.GetConformer(cid), assignId=True)
    _, result = pmeff.optimize_mol(single)
    energies.append((result.energy, cid))

for e, cid in sorted(energies):
    print(f"conf {cid:2}: {e:.3f}")
```

### Optimize a metal complex

```python
import pmeff
# electronic_effects gives the metal its octahedral/square-planar angle targets
# and switches on geometry-derived QEq charges.
mol, result = pmeff.optimize_mol(mol, electronic_effects=True, use_dispersion=True)
```

---

## 11. Units & interpretation

- **Lengths** are in Ångström; **coordinates** are `(N, 3)` NumPy arrays.
- **Energies** are internal, self-consistent units meant for *relative*
  comparison and geometry guidance — **not** kcal/mol thermochemistry. Do not
  compare PMEFF energies against experiment or DFT in absolute terms.
- **Forces / gradients** are dE/dx in the matching internal units; `max_force` is
  the largest per-atom force magnitude.
- **Frequencies** from `vibrational_analysis` are unit-mass (signed √eigenvalue),
  not cm⁻¹ — use them to *classify* stationary points, not to predict spectra.
- The intended workflow: PMEFF cleans up a rough geometry → hand the result to a
  real QM engine (e.g. the ORCA or PySCF MoleditPy plugins) for energies.

---

## 12. Troubleshooting / FAQ

**`ImportError: pmeff.optimize_mol requires RDKit`**
Install the extra: `pip install "pmeff[rdkit]"`. Or use the array path
(`optimize_coords`), which needs only NumPy.

**`ValueError: PMEFF: molecule has no 3D conformer to optimize`**
Embed a conformer first: `AllChem.EmbedMolecule(mol)`. A SMILES-only `Mol` has no
coordinates.

**`result.converged` is `False`.**
The iteration budget ran out. Raise `max_iter`, and check for overlapping/coincident
atoms in the input geometry, which produce huge forces the optimizer cannot resolve.

**Charges look wrong / electrostatics seem missing (array path).**
`optimize_coords` does *not* add charges unless you pass them. Supply
`charges=qeq_charges(...)`, or use `optimize_mol(..., electronic_effects=True)`.

**A metal center collapses to the wrong geometry.**
Enable `electronic_effects=True` so d⁸ / d-block metals get square-planar /
octahedral angle targets instead of coordination-number defaults.

**Optimization is slow on a large system.**
Evaluation is dominated by the non-bonded terms. Keep `vdw_cutoff` at its default
(12 Å) and avoid `use_dispersion` unless you need it.

---

## 13. API reference summary

Everything below is importable from the top-level `pmeff` package.

| Symbol | Kind | Purpose |
|---|---|---|
| `optimize_mol(mol, ...)` | function | RDKit path: relax a `Mol`'s conformer in place. |
| `optimize_coords(nums, bonds, coords, ...)` | function | Array path: relax plain coordinates. |
| `optimize(coords, topo, ...)` | function | Low-level FIRE + L-BFGS minimizer. |
| `build_topology(nums, bonds, ...)` | function | Assemble a `Topology` from connectivity. |
| `energy_and_gradient(coords, topo)` | function | `(energy, gradient)` — analytical. |
| `energy_components(coords, topo)` | function | Per-term energy decomposition dict. |
| `vibrational_analysis(coords, topo)` | function | Normal-mode minimum check. |
| `qeq_charges(nums, coords, ...)` | function | Electronegativity-equalization partial charges. |
| `bond_rest_length(z_i, z_j, order)` | function | Polar-contracted bond rest length. |
| `Topology` | dataclass | RDKit/Qt-free force-field problem. |
| `OptimizeResult` | dataclass | `converged`, `energy`, `steps`, `max_force`. |
| `__version__` | str | Installed package version. |

For deeper internals (parameter tables, gradient derivations, optimizer tuning
constants) read the source of `pmeff/forcefield.py` and
[TECHNICAL.md](TECHNICAL.md).
