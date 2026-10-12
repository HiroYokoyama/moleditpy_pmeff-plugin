# PMEFF 1.5.0 validation

These changes are general engine and UI changes, not parameters fitted to
particular molecules. The named structures below are regression cases.

PMEFF is only for initial structure preparation before DFT. These validation
checks do not make its geometries, energies or modes quantitatively accurate
substitutes for DFT results.

## Geometry and angle checks

RDKit embedding with seed 42, explicit hydrogens, electronic effects, Morse
bonds and hydrogen-bond correction enabled; at most 2,000 steps, force
tolerance 0.001. Angles are heavy-atom neighbor angles. Residual forces are
recomputed from a newly built topology at the final coordinates.

| Structure | Heavy-atom angle range (degrees) | Steps | Fresh maximum force | Imaginary modes |
|---|---:|---:|---:|---:|
| Adamantane | 109.449–109.515 | 106 | 0.0003202 | 0 |
| Cubane | 90.000–90.000 | 69 | 0.0003955 | 0 |
| Cyclododecane | 109.850–113.398 | 333 | 0.0009768 | 0 |
| 18-crown-6 | 108.722–111.839 | 535 | 0.0008235 | 0 |

Automated regressions also perturb all coordinates by Gaussian noise with
standard deviation 0.08 Å (seed 7). Both starts preserve plausible bonds and
heavy-atom angles and converge with freshly recomputed forces below tolerance.
The existing organic and metal geometry benchmarks are included in the suite.

Hydrogen-bond direction and analytical derivatives are checked at 0, 45,
89, 90, 91, 135 and 180 degrees, including the taper boundary.

## What these checks establish

The checked cages retain their constrained angles; the checked macrocycles
avoid gross angle/bond deformation. Final-charge convergence prevents the
optimizer claiming convergence with charges derived from an earlier geometry.
There are no molecule-name rules in these changes.

These are not a proof of accuracy for every molecule or a conformational
search. PMEFF has a known short C–C rest-length bias (adamantane approximately
1.501 Å, versus approximately 1.539 Å from MMFF in the same comparison).
The QEq charge-relaxation model remains approximate. Vibrational checks use
the fixed-charge Hessian at the reported geometry and unit masses, not quantum
frequencies in cm⁻¹. Flexible macrocycles can have multiple local minima;
the particular user-reported distorted molecule was not supplied, so that
specific conformation could not be validated.

## UI and cancellation checks

Six real Qt tests confirm worker thread execution, GUI timer responsiveness,
window-close cancellation, error delivery, preservation of the live conformer
on cancellation, copyable result tables and review of unconverged geometry.
Offscreen snapshots of the progress and vibrational result dialogs were
inspected for readable labels, mode tables and units.

The full numerical/plugin suite passed 350 tests. Pylint scored 9.28/10 before
the version stamp. The standalone distribution is derived from the same engine
and its API/export/source-sync checks are included.
