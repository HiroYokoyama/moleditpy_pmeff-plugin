# PMEFF Technical Reference

The physics and numerics of the PMEFF engine
(`pmeff_plugin/forcefield.py`). The [README](../README.md) gives the
user-level overview; this document specifies the model precisely enough to
reimplement it: every functional form, every parameter derivation, the
gradient formulas, the non-bonded bookkeeping and the optimizer.

Everything below the RDKit boundary operates on a `Topology` of plain numbers
— no Qt, no RDKit — so each statement here is checked directly by the unit
tests in `tests/test_forcefield.py`.

## Units and conventions

| Quantity | Unit |
|---|---|
| Length | Ångström (Å) |
| Angle | radian internally (degrees in this text where noted) |
| Energy | internal, kcal/mol-like (self-consistent, not calibrated) |
| Charge | elementary charge *e* |
| Mass | unit mass per atom (optimizer only; no dynamics are produced) |

Energies are meant for relative comparison and geometry relaxation, not
thermochemistry. The Coulomb constant `_K_COULOMB = 332.07` sets the
electrostatic scale so that, *if* the energy unit is read as kcal/mol, charges
are in *e* and distances in Å.

Atom indices in every term tuple are 0-based. Pair keys are always stored as
`(i, j)` with `i < j`.

## Parameter philosophy

Every parameter derives from **one per-element input**: the Pyykkö & Atsumi
(2009) single-bond covalent radius, tabulated for Z = 1–118
(`_COVALENT_RADII_PM`). Atomic numbers outside 1–118 (dummy atoms) fall back
to `_DEFAULT_RADIUS_A = 1.50 Å`. Because everything else is a formula of the
radius (and, for QEq, of Z via Slater's rules), no element can ever be
missing a parameter — the defining property of the force field.

Derived per-atom quantities:

- **vdW radius** — `R_i = r_i + 0.90 Å` (`vdw_radius`). The offset reproduces
  tabulated vdW radii of common elements to ~0.05 Å (C 1.65 vs 1.70, O 1.53
  vs 1.52, H 1.22 vs 1.20).
- **LJ well depth** — `ε_i = 0.10 · (r_i / 0.75)^1.5` (`vdw_epsilon`); the
  covalent radius acts as a polarizability proxy and carbon (r = 0.75 Å)
  anchors the scale.
- **Effective nuclear charge** — Slater's rules over the Aufbau occupation
  (`slater_zeff`): same-shell electrons screen 0.30 (n = 1) or 0.35, the
  (n−1) shell screens 0.85, deeper shells screen 1.00.
- **Electronegativity** — Allred–Rochow,
  `χ_Pauling = 0.359 · Z_eff / r² + 0.744`, converted to an energy scale by
  ×2.27 eV per Pauling unit (`electronegativity`).
- **Hardness** — the self-Coulomb of a sphere of the covalent radius,
  `η_i = 14.4 / (2 r_i)` eV (`hardness`).

## Energy terms

The total energy is a sum of six terms. `energy_and_gradient(coords, topo)`
returns the total and its fully **analytical** gradient; passing a dict as
`components=` (or calling `energy_components`) yields the per-term
decomposition under the keys `bond`, `angle`, `torsion`, `oop`, `vdw`,
`elec` (all keys always present; absent terms report 0.0).

### 1. Bond stretch

```
E = ½ k (r − r₀)²          k = 700 · (bond order)
```

`r₀` is the sum of the two covalent radii scaled by the bond-order factor:
linear interpolation through the anchors (1.0 → ×1.00, 2.0 → ×0.89,
3.0 → ×0.78), clamped outside; anchored to C–C 1.50 Å / C=C 1.33 Å /
C≡C 1.20 Å, so aromatic order 1.5 gives C(ar)–C(ar) = 1.42 Å
(`bond_order_factor`). Stiffness grows linearly with order (C≡C ≈ 3× C–C).

### 2. Angle bend

Ordinary angles are harmonic in θ with `k = 120`:

```
E = ½ k (θ − θ₀)²
```

θ₀ comes from the central atom's hybridization (sp 180°, sp² 120°, sp³
109.47°, sp³d/sp³d² 90°), falling back to the coordination number for metals
and unknown cases (`_ideal_angle_deg`, `_ANGLE_BY_COORDINATION`).

Three special cases:

- **Linear centers (θ₀ = π).** `d θ / d cos θ` diverges at θ = π, so linear
  targets use `E = k (1 + cos θ)` instead — identical curvature at 180° but a
  finite (zero) gradient exactly at the minimum.
- **Three-membered rings.** When the two outer atoms of an angle are
  themselves bonded, the hybridization target would fight the three bond
  terms. The target is instead the law-of-cosines angle implied by the three
  bond rest lengths (60° for cyclopropane), so bonds and angles share one
  exact minimum.
- **Square-planar d⁸ metals** (electronic effects only). Angles at
  4-coordinate Ni/Pd/Pt/Rh/Ir/Au carry the sentinel θ₀ = −1; at evaluation
  time each such angle pulls toward whichever ideal vertex angle (90° or
  180°) is nearer (threshold 135°), so the cis/trans assignment emerges from
  the starting geometry.

### 3. Torsion

```
E = ½ V (1 + cos(n φ − γ))
```

assigned per central bond from the hybridizations of its two atoms
(`_torsion_params`):

| central bond | n | γ | barrier V (per bond) |
|---|---|---|---|
| sp²–sp² | 2 | π | 10.0 × π-scaling |
| sp³–sp³ | 3 | 0 | 2.0 |
| sp²–sp³ | 6 | π | 0.5 |
| anything else | — | — | no torsion (angles fix the geometry) |

The per-bond barrier is split evenly over all i–j–k–l paths sharing the bond
(UFF-style), so it is a per-bond, not per-substituent, quantity. Paths where
i = l (three-membered rings) are skipped. The 2-fold barrier is scaled by the
π character of the central bond, `clamp(order − 1, 0.15, 1.0)` — full for a
double bond, ~0.5 for aromatic, weak but nonzero for a conjugated sp²–sp²
single bond, so biphenyl can twist while ethylene stays rigid.

### 4. Out-of-plane

Every 3-coordinate sp² center j with neighbors a, b, c gets

```
E = ½ k (θ_ab + θ_bc + θ_ac − 2π)²        k = 40
```

on the sum of the three bend angles around j: the sum is 2π exactly when j is
in the plane of its neighbors, and decreases with pyramidalization.

### 5. van der Waals (Lennard-Jones 12-6)

```
E = ε [ (r_min / r)¹² − 2 (r_min / r)⁶ ]
r_min = R_i + R_j        ε = √(ε_i ε_j)      (Lorentz–Berthelot)
```

1-2 and 1-3 pairs are excluded; 1-4 pairs use ε × 0.5 so torsional profiles
are not swamped by the end-atom clash.

**Cutoff and switching.** At the RDKit boundary the pair list is truncated at
`_VDW_CUTOFF_A = 12 Å` (a C–C interaction there is ~10⁻⁴ of the well depth).
A CHARMM-style switching function `S(r)` (`_switch`) multiplies the LJ energy
between `r_on = cutoff − 2 Å` and the cutoff:

```
S(r) = (r_off² − r²)² (r_off² + 2r² − 3r_on²) / (r_off² − r_on²)³
```

S = 1 below r_on, S = 0 with **zero slope** at r_off, so both energy and
force are continuous as a pair crosses the boundary. The gradient includes
the `dS/dr · E_LJ` cross term. Topologies built without a cutoff
(`Topology.vdw_cutoff is None`) evaluate every listed pair in full.

**Verlet list.** The pair list is built out to `cutoff + skin`
(`_VDW_SKIN_A = 2 Å`). Pairs in the skin shell cost nothing (S = 0 past the
cutoff) — the skin buys *validity*: the list remains correct until some atom
has moved more than skin/2 from where the list was built. The optimizer
tracks that drift and calls `refresh_vdw_pairs(topo, coords)` when the bound
is crossed, which re-selects all non-excluded pairs inside `cutoff + skin`
(vectorized over `triu_indices`) and invalidates the compiled-array cache.
Without this, a pair drifting into the cutoff during optimization would feel
no LJ force at all. `Topology.excluded_pairs` / `Topology.pairs14` store the
connectivity-derived exclusion data that makes the rebuild possible without
the original bond list.

### 6. Electrostatics (electronic effects only)

Charges come from a one-shot electronegativity equalization (`qeq_charges`):
minimize `Σ (χ_i q_i + ½ η_i q_i²) + Σ_ij J_ij q_i q_j` subject to
`Σ q_i = Q_total` (the molecule's total formal charge), with the Ohno-shielded
interaction

```
J_ij = 14.4 / √(r_ij² + γ_ij²)        γ_ij = 2 · 14.4 / (η_i + η_j)
```

which tends to the mean hardness at r = 0. This is a single (N+1)×(N+1)
linear solve with a Lagrange multiplier; the charges are then **held fixed**
during optimization. A singular system (coincident atoms) falls back to zero
charges with a warning.

The pair energy is a shielded Coulomb term

```
E = k q_i q_j / √(r² + γ²)        γ = ½ (r_i + r_j)
```

with the same 1-2/1-3 exclusions and 0.5× 1-4 scaling as the LJ term. Pairs
with `|k q_i q_j| ≤ 10⁻¹²` are dropped. Electrostatics are long-range and are
**never** distance-truncated — only the LJ list has a cutoff.

## Gradients

All gradients are analytical; the test suite verifies every term against
central finite differences (including inside the switching window and near
linear angles).

- **Bends** are differentiated through cos θ (`_bend_terms`); the vertex
  derivative is `−(∂cos/∂i + ∂cos/∂k)` by translational invariance, and
  degenerate (zero-length) arms get zero derivatives. The harmonic form
  converts via `dθ/dcos = −1/sin θ` with sin θ floored at 10⁻⁶.
- **Dihedrals** use the van Schaik et al. analytical formulas with
  `φ = atan2((n₁×n₂)·b̂₂, n₁·n₂)`. **Sign convention warning:** the published
  middle-atom formulas assume `b₁ = r_i − r_j`; this code uses
  `b₁ = r_j − r_i`, which flips the cross terms to
  `∂φ/∂r_j = −(1+s₁₂)·∂φ/∂r_i + s₃₂·∂φ/∂r_l` and
  `∂φ/∂r_k = s₁₂·∂φ/∂r_i − (1+s₃₂)·∂φ/∂r_l`, where
  `s₁₂ = (b₁·b₂)/|b₂|²`, `s₃₂ = (b₃·b₂)/|b₂|²`. A sign error here still
  passes translational-invariance checks — only a numeric-gradient test
  catches it. Degenerate torsions (collinear arms) contribute zero energy and
  gradient.
- All terms are evaluated with vectorized numpy over precompiled index
  arrays; per-atom accumulation uses `np.add.at` (unbuffered, so repeated
  indices are correct).

### Compiled-array cache

`Topology.compiled()` flattens the Python term lists into numpy arrays once
and caches them, keyed on the six term-list *lengths*. Growing a topology
recompiles automatically; `refresh_vdw_pairs` can swap pairs *without*
changing the count, so it clears the cache explicitly.

## Optimizer — FIRE 2.0

`optimize(coords, topo, max_iter, f_tol, max_step)` minimizes with FIRE
(Bitzek et al. 2006) including the FIRE 2.0 refinements (Guénolé et al.
2020), with unit masses:

1. Convergence check: largest per-atom force magnitude < `f_tol`
   (default 10⁻³).
2. Power `P = F · v`. If `P > 0` for more than `n_min = 5` consecutive
   steps: `dt ← min(1.1 dt, 0.5)`, `α ← 0.99 α`.
3. If `P ≤ 0` (uphill): **retract half of the last applied step**
   (`x ← x − ½ Δx_prev`), zero the velocity, `dt ← max(0.5 dt, 10⁻⁴)`,
   reset `α = 0.1`. Using the clamped displacement — not `½ dt v` — keeps
   the retraction bounded when forces were huge (an unclamped retraction can
   teleport atoms after a steric clash).
4. Semi-implicit Euler + velocity mixing:
   `v ← v + dt F`, then `v ← (1−α) v + α F̂ |v|`.
5. Displacement clamp: if any atom would move farther than
   `max_step = 0.20 Å`, the whole step is rescaled so the largest per-atom
   displacement equals `max_step`. This makes stability independent of the
   absolute force-constant scale.
6. Verlet-list drift check (see above), then re-evaluate energy/forces.

Two numerically motivated choices deserve emphasis:

- **The `dt` floor must sit far below the stiffest mode's stability limit.**
  The stiffest bonds (k = 700 × order) give ω ≈ √(2k) ≈ 50 in internal time
  units, i.e. a critical `dt ≈ 0.04`. A floor near that value pins the step
  length at the displacement clamp and the optimizer *orbits* the minimum in
  a limit cycle instead of settling; the floor is therefore 10⁻⁴, which only
  guards against `dt` collapsing to exactly zero.
- **A perfectly symmetric saddle is a stationary point.** An exactly eclipsed
  or planar-symmetric start has zero force along the symmetry-breaking mode;
  behavioral tests (and users judging optimizer quality) must start from
  symmetry-broken geometries.

`OptimizeResult` reports `converged`, final `energy`, `steps`, and
`max_force`.

## RDKit boundary and public API

Only four functions touch RDKit, and `Point3D` is imported lazily so the
core stays importable without it:

| Function | Purpose |
|---|---|
| `topology_from_rdkit(mol, electronic_effects)` | connectivity, bond orders (`GetBondTypeAsDouble`), hybridizations, formal-charge total → `build_topology` with the 12 Å cutoff; QEq charges from the conformer when electronic effects are on |
| `optimize_rdkit_mol(mol, max_iter, f_tol, electronic_effects)` | optimize the conformer in place; `(True, None)` for < 2 atoms, `(False, …)` without a conformer or on non-finite output |
| `compute_energy(mol, electronic_effects)` | single-point total, or `None` without a conformer |
| `compute_energy_components(mol, electronic_effects)` | per-term decomposition dict (adds `total`), or `None` |

Core (RDKit-free) entry points: `build_topology`, `refresh_vdw_pairs`,
`energy_and_gradient`, `energy_components`, `optimize`, `qeq_charges`, plus
the per-element parameter functions (`covalent_radius`, `vdw_radius`,
`vdw_epsilon`, `slater_zeff`, `electronegativity`, `hardness`,
`bond_order_factor`).

`build_topology` accepts `coords` + `vdw_cutoff` to prune the LJ list at
build time; without coordinates the cutoff is ignored and every non-excluded
pair is listed (and `Topology.vdw_cutoff` stays `None`, disabling both the
switching function and the Verlet refresh).

## Testing strategy

`tests/` runs against real numpy + RDKit, headless:

- **Numeric-gradient checks** for every term and special case (linear
  angles, switching window, torsion + OOP combinations) — the only reliable
  guard against dihedral sign errors.
- **Single-minimum consistency**: geometries where all bonded terms should be
  simultaneously at rest (equilateral 3-ring, exact linear CO₂) must give
  E = 0 and zero gradient.
- **Optimizer behavior**: stretched bonds relax, water opens to 109.47°,
  twisted ethylene planarizes, eclipsed ethane staggers (started 15° off the
  saddle), pyramidal sp² centers flatten, benzene converges planar.
- **Verlet-list behavior**: skin-shell pairs are listed but energy-free;
  refresh follows moving atoms and invalidates the compiled cache; two
  opposite charges starting outside the list radius bind but do not collapse
  (the refresh switches the LJ pair on mid-optimization).
- **Electronic effects**: QEq charge signs/conservation, square-planar
  relaxation, lone-pair shape regressions (pyramidal NH₃, planar amide N).
