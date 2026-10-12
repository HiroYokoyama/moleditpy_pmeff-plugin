# Changelog

## Unreleased

### Added

- Cooperative cancellation and progress callbacks for geometry optimization
  and the vibrational Hessian calculation. Cancellation never writes partial
  optimized coordinates to the input RDKit conformer.
- Vibrational results include eigenvalues, the classification threshold and
  the residual maximum force, so positive curvature is not confused with
  convergence to a stationary point.
