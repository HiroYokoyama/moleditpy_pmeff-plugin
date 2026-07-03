# Packaging & release

This repo ships two artifacts from one version tag:

1. **The MoleditPy plugin** — a `pmeff_plugin_<version>.zip` attached to a
   GitHub Release (and the `moleditpy-plugins` registry is notified), matching
   the other MoleditPy plugin repos.
2. **The standalone `pmeff` PyPI package** — a NumPy-only force field usable
   outside MoleditPy.

## Single source of truth

`pmeff_plugin/forcefield.py` is the canonical engine. The `pmeff` package is
*derived* from it:

- `pmeff/forcefield.py` is a verbatim copy (with a generated header).
- `pmeff/_version.py` is stamped from `PLUGIN_VERSION`.

Both are produced by:

```bash
python scripts/sync_forcefield.py
```

The committed copies let the repo build as-is; the test suite
(`test_pmeff_package.py::test_copied_engine_matches_source`) fails if they
drift, so **edit `pmeff_plugin/forcefield.py`, then re-run the sync script**.

## Public API (`pmeff`)

```python
import pmeff

# RDKit path (pip install "pmeff[rdkit]") — Mol in, same Mol back, relaxed:
mol, result = pmeff.optimize_mol(mol)

# Pure-NumPy path (pip install pmeff) — arrays in, coords out:
coords, result = pmeff.optimize_coords(atomic_numbers, bonds, coords)
```

## Cutting a release

Bump `PLUGIN_VERSION` in `pmeff_plugin/__init__.py`, then either:

- **push a tag**: `git tag v1.2.3 && git push origin v1.2.3`, or
- **one click**: run the *Release* workflow from the Actions tab with the
  version as input (it creates and pushes the tag for you).

The `release.yml` workflow verifies the tag matches `PLUGIN_VERSION`, builds
the plugin zip + GitHub Release, then re-syncs and publishes `pmeff` to PyPI.

## Building / uploading locally

```bash
python scripts/sync_forcefield.py
python -m build            # -> dist/pmeff-<version>.tar.gz + .whl
python -m twine check dist/*
python -m twine upload dist/*
```

## PyPI trusted publishing (one-time)

The workflow publishes via OIDC (no API token stored). Before the first CI
release, add a **pending publisher** on PyPI for project `pmeff`:

- Owner / repo: `HiroYokoyama/moleditpy_pmeff-plugin`
- Workflow filename: `release.yml`
- Environment: `pypi`
