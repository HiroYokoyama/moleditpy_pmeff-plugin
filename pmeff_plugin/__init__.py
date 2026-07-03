"""PMEFF — Python Molecular Editor Force Field.

A MoleditPy plugin providing PMEFF, a self-contained universal force field
that covers the entire periodic table (Z = 1..118). PMEFF is a pre-DFT
geometry-cleanup force field: its purpose is *initial structure preparation*
— removing clashes, fixing bond lengths and angles, placing metal centers in
the correct coordination geometry — not high-accuracy thermochemistry. It
registers:

* an **Optimize 3D** method ("PMEFF") that relaxes the current 3D
  geometry with a dependency-free FIRE + L-BFGS optimizer,
* an **Analysis** tool ("PMEFF Single-Point Energy") that reports the current
  force-field energy (with per-term decomposition) without modifying the
  molecule,
* an **Analysis** tool ("PMEFF Minimum Check (Vibrational)") that verifies
  the current geometry is a true minimum, and
* a **Settings/PMEFF Setting** menu entry that opens a settings dialog.

See ``forcefield.py`` for the physics; this module only wires it into the host
via the stable ``PluginContext`` API.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

PLUGIN_NAME = "PMEFF Plugin"
PLUGIN_VERSION = "0.4.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "PMEFF (Python Molecular Editor Force Field) — a self-contained universal "
    "force field covering the entire periodic table (Z=1..118). Adds a "
    "geometry optimizer, a single-point energy tool with per-term "
    "decomposition, and a vibrational minimum check. No external QM binary "
    "required."
)
PLUGIN_CATEGORY = "Optimization"
PLUGIN_TAGS = ["Optimization", "Force Field", "3D"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"
PLUGIN_DEPENDENCIES = ["numpy", "rdkit"]

_OPT_METHOD_NAME = f"PMEFF (v{PLUGIN_VERSION})"
_MAX_ITER = 1000

# Plugin options live in a JSON file next to the plugin package, so they
# travel with the installed plugin and need no host settings API.
_SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
_DEFAULT_SETTINGS = {
    # QEq charges + shielded Coulomb + square-planar/octahedral metal targets.
    "electronic_effects": True,
    # Morse bond potential replaces harmonic; no overhead, on by default.
    "morse_bonds": True,
    # D-H...A geometry-dependent attraction (N/O/F/S donors and acceptors).
    "hbond": True,
    # BJ-damped C6/r^6 dispersion on top of LJ. Off by default (can be slow
    # for large molecules with many non-bonded pairs).
    "dispersion": False,
}

logger = logging.getLogger(__name__)


def load_settings() -> dict:
    """Read settings.json, falling back to defaults on any problem."""
    settings = dict(_DEFAULT_SETTINGS)
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            settings.update(data)
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logger.warning("PMEFF: could not read %s; using defaults.", _SETTINGS_FILE)
    return settings


def save_settings(settings: dict) -> None:
    """Write settings.json atomically (temp file + replace)."""
    try:
        fd, tmp = tempfile.mkstemp(
            dir=str(_SETTINGS_FILE.parent), suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
        os.replace(tmp, _SETTINGS_FILE)
    except OSError:
        logger.exception("PMEFF: failed to write %s", _SETTINGS_FILE)


def electronic_effects_enabled() -> bool:
    """Whether the optional electronic-effects terms are switched on."""
    return bool(load_settings().get("electronic_effects", False))


def _settings_kwargs() -> dict:
    """Return forcefield keyword arguments derived from the current settings."""
    s = load_settings()
    return {
        "electronic_effects": bool(s.get("electronic_effects", True)),
        "use_morse": bool(s.get("morse_bonds", True)),
        "use_hbond": bool(s.get("hbond", True)),
        "use_dispersion": bool(s.get("dispersion", False)),
    }


def initialize(context):
    """Register PMEFF's optimization method, energy tool and settings."""
    context.register_optimization_method(
        _OPT_METHOD_NAME, lambda mol: _optimize(mol, context)
    )
    context.add_analysis_tool(
        "PMEFF Single-Point Energy", lambda: _show_energy(context)
    )
    context.add_analysis_tool(
        "PMEFF Minimum Check (Vibrational)", lambda: _check_minimum(context)
    )
    context.add_menu_action(
        "Settings/PMEFF Setting",
        lambda: _open_settings_dialog(context),
        text="PMEFF Settings…",
    )


def _open_settings_dialog(context) -> None:
    """Open the PMEFF settings dialog and save any changes."""
    from .settings_dialog import open_settings_dialog

    parent = getattr(context, "main_window", None)
    current = load_settings()
    updated = open_settings_dialog(parent, current)
    if updated is None:
        return  # cancelled or headless
    # Merge: preserve unknown keys the dialog doesn't know about.
    current.update(updated)
    save_settings(current)
    _status(context, "PMEFF settings saved.", 3000)


def _optimize(mol, context) -> bool:
    """Relax *mol* in place with PMEFF. Returns True on success."""
    from .forcefield import optimize_rdkit_mol

    try:
        success, result = optimize_rdkit_mol(
            mol, max_iter=_MAX_ITER, **_settings_kwargs()
        )
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF optimization failed")
        _status(context, f"PMEFF optimization failed: {exc}", 5000)
        return False

    if not success:
        _status(context, "PMEFF: no 3D geometry to optimize.", 4000)
        return False

    if result is not None:
        state = "converged" if result.converged else "stopped (max iterations)"
        _status(
            context,
            f"PMEFF {state} in {result.steps} steps "
            f"(E = {result.energy:.2f}, |F|max = {result.max_force:.4f}).",
            5000,
        )
    else:
        _status(context, "PMEFF: nothing to optimize (single atom).", 3000)
    return True


def _show_energy(context) -> None:
    """Report the PMEFF single-point energy of the current molecule."""
    from .forcefield import compute_energy_components

    mol = context.current_molecule
    if mol is None:
        _status(context, "PMEFF: no molecule loaded.", 3000)
        return

    try:
        comp = compute_energy_components(mol, **_settings_kwargs())
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF energy evaluation failed")
        _status(context, f"PMEFF energy failed: {exc}", 5000)
        return

    if comp is None:
        _status(context, "PMEFF: molecule has no 3D coordinates.", 4000)
        return

    base = (
        f"bond {comp['bond']:.2f}, angle {comp['angle']:.2f}, "
        f"torsion {comp['torsion']:.2f}, oop {comp['oop']:.2f}, "
        f"vdW {comp['vdw']:.2f}, elec {comp['elec']:.2f}"
    )
    extras = "".join(
        f", {k} {comp[k]:.2f}"
        for k in ("hbond", "disp")
        if abs(comp.get(k, 0.0)) > 1e-6
    )
    _status(
        context,
        f"PMEFF single-point energy: {comp['total']:.4f} ({base}{extras})",
        8000,
    )


def _check_minimum(context) -> None:
    """Report whether the current geometry is a true PMEFF minimum."""
    from .forcefield import check_minimum

    mol = context.current_molecule
    if mol is None:
        _status(context, "PMEFF: no molecule loaded.", 3000)
        return

    try:
        result = check_minimum(mol, **_settings_kwargs())
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF vibrational analysis failed")
        _status(context, f"PMEFF vibrational analysis failed: {exc}", 5000)
        return

    if result is None:
        _status(context, "PMEFF: molecule has no 3D coordinates.", 4000)
        return

    if result["is_minimum"]:
        _status(
            context,
            "PMEFF vibrational check: true minimum "
            f"({result['num_zero']} rigid-body modes, 0 imaginary).",
            8000,
        )
    else:
        lowest = float(result["frequencies"][0])
        _status(
            context,
            f"PMEFF vibrational check: NOT a minimum — "
            f"{result['num_imaginary']} imaginary mode(s), lowest "
            f"{lowest:.3f}. Re-optimize from a perturbed geometry.",
            8000,
        )


def _status(context, message: str, timeout: int = 3000) -> None:
    """Show a status-bar message, tolerating a minimal/mock context."""
    show = getattr(context, "show_status_message", None)
    if callable(show):
        show(message, timeout)
    else:  # pragma: no cover - only hit with a bare context
        logger.info(message)
