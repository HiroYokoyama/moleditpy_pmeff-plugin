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

The `release.yml` workflow then, in order:

1. **release** — verifies the tag matches `PLUGIN_VERSION`, builds the plugin
   zip, creates the GitHub Release, and notifies the `moleditpy-plugins`
   registry.
2. **pypi** — re-syncs the engine and publishes `pmeff` to PyPI.
3. **sync-back** — re-runs the sync script on `main` and, if the committed
   `pmeff/forcefield.py` / `pmeff/_version.py` don't already match the release
   (e.g. the tag was cut without syncing locally first), commits the refreshed
   copies back to `main` (`[skip ci]`, so it doesn't re-trigger). In the normal
   case — where you synced before tagging — this no-ops.

So `PLUGIN_VERSION` stays the single source of truth, and the repo's derived
package files are kept in sync automatically at release time.

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
