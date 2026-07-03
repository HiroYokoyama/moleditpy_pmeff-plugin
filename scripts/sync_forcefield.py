#!/usr/bin/env python3
"""Sync the canonical PMEFF engine into the ``pmeff`` PyPI package.

``pmeff_plugin/forcefield.py`` is the single source of truth for the force
field. The standalone ``pmeff`` distribution is *derived* from it: this script
copies that file into ``pmeff/forcefield.py`` (prefixing a "generated" header)
and stamps ``pmeff/_version.py`` from the plugin's ``PLUGIN_VERSION``.

Run it before building the package (the publish workflow does this on every
release, so the published artifact is always fresh); it is also handy locally:

    python scripts/sync_forcefield.py
    python -m build
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pmeff_plugin" / "forcefield.py"
PLUGIN_INIT = ROOT / "pmeff_plugin" / "__init__.py"
DEST = ROOT / "pmeff" / "forcefield.py"
VERSION_FILE = ROOT / "pmeff" / "_version.py"

_HEADER = (
    "# AUTO-GENERATED — DO NOT EDIT.\n"
    "# Copied verbatim from pmeff_plugin/forcefield.py (the single source of\n"
    "# truth) by scripts/sync_forcefield.py. Edit the source, then re-run the\n"
    "# script; the publish workflow re-syncs on every release.\n\n"
)


def _plugin_version() -> str:
    text = PLUGIN_INIT.read_text(encoding="utf-8")
    match = re.search(r'^PLUGIN_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        raise SystemExit("could not find PLUGIN_VERSION in pmeff_plugin/__init__.py")
    return match.group(1)


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"source engine not found: {SOURCE}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(_HEADER + SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    version = _plugin_version()
    VERSION_FILE.write_text(
        '"""Version of the pmeff distribution (kept in sync with the plugin)."""\n\n'
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    print(f"synced {SOURCE.name} -> {DEST.relative_to(ROOT)} (version {version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
