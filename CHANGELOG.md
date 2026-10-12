# Changelog

## Unreleased

### Added

- PMEFF optimization, single-point energy and vibrational checks run on a
  worker thread with a calculating/cancel dialog. Calculations use molecular
  snapshots; the GUI thread applies accepted optimization results.
- Persistent, copyable result dialogs for energy decomposition, optimization
  convergence and every vibrational mode. Unconverged optimizations require
  an explicit choice to apply the geometry; keeping the original is the default.

- Cooperative cancellation and progress callbacks for geometry optimization
  and the vibrational Hessian calculation. Cancellation never writes partial
  optimized coordinates to the input RDKit conformer.
- Vibrational results include eigenvalues, the classification threshold and
  the residual maximum force, so positive curvature is not confused with
  convergence to a stationary point.
