# Changelog

## Unreleased

## 1.5.0 — 2026-10-12

### Scope

- PMEFF is only for initial structure preparation before DFT. Optimized
  geometries, energies and vibrational modes are approximate, not quantitatively
  accurate substitutes for DFT results. This note appears in every result dialog.
- The imaginary-mode checker reports unit-mass internal curvature modes,
  not physical wavenumbers in cm⁻¹ or an accurate IR/Raman spectrum.

### Fixed

- Final-charge convergence checks prevent reporting a minimum with stale QEq
  charges. Residual forces now agree with a freshly rebuilt force-field topology.
- Hydrogen-bond angular taper favors the linear 180-degree geometry and
  removes the spurious attraction at acute/reverse-facing angles. Analytical
  gradients are continuous at the 90-degree taper boundary.
- Macrocycle and cage regressions cover embedded and perturbed adamantane,
  cubane, cyclododecane and 18-crown-6: heavy-atom angles, bond lengths and
  freshly recomputed residual forces.

### Added

- Configurable maximum optimization steps in PMEFF Settings (default 1,000),
  and progress/cancellation callbacks in the standalone `pmeff` convenience API.

- PMEFF optimization, single-point energy and vibrational checks run on a
  worker thread with a calculating/cancel dialog. Calculations use molecular
  snapshots; the GUI thread applies accepted optimization results.
- Persistent, copyable result dialogs for energy decomposition, optimization
  convergence and every vibrational mode. Unconverged optimizations require
  an explicit choice to apply the geometry; keeping the original is the default.
- Compact result summaries with a **Details…** button that opens the full
  calculation data and a copy button in a separate window.

- Cooperative cancellation and progress callbacks for geometry optimization
  and the vibrational Hessian calculation. Cancellation never writes partial
  optimized coordinates to the input RDKit conformer.
- Vibrational results include eigenvalues, the classification threshold and
  the residual maximum force, so positive curvature is not confused with
  convergence to a stationary point.
