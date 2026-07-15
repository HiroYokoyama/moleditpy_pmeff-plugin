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
PLUGIN_VERSION = "1.3.1"
# Must equal the GitHub username (the moleditpy registry enforces this).
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
    # Electronegativity-difference shortening of polar bond rest lengths
    # (fixes over-long Si-O, P-O, metal-oxide and C-F bonds; organic bonds
    # are left untouched). On by default.
    "polar_contraction": True,
}

logger = logging.getLogger(__name__)

# The physics options used the last time PMEFF actually optimized the current
# document, or None if it never has. Saved into (and restored from) the project
# file, and cleared on File > New, so a project remembers what it was last
# optimized with. This is a per-document record only; it does not override the
# global settings.json read by _settings_kwargs().
_last_opt_settings = None

# Per-atom coordination-geometry overrides for the current document, mapping
# ``atom_index -> geometry name`` (see forcefield._VALID_GEOMETRIES). Set via
# the Metal Geometry Override table, applied by every PMEFF optimize / energy /
# minimum-check call, persisted into the project file and restored on load,
# and cleared on File > New. Empty means "no overrides" — default behavior.
_geometry_overrides: dict = {}


def _overrides_kwarg() -> "dict | None":
    """Return the current geometry overrides for the engine, or None if empty."""
    return dict(_geometry_overrides) if _geometry_overrides else None


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
        fd, tmp = tempfile.mkstemp(dir=str(_SETTINGS_FILE.parent), suffix=".tmp")
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
        "use_polar_contraction": bool(s.get("polar_contraction", True)),
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
    # One menu entry under 3D Edit; also reachable from a button in PMEFF
    # Settings. Both open the same modeless override table.
    context.add_menu_action(
        "3D Edit/PMEFF Metal Geometry Override",
        lambda: _open_geometry_override_window(context),
        text="PMEFF Metal Geometry Override…",
    )
    # Persist the last-used optimization options and the per-atom geometry
    # overrides with the project; restore the overrides on load; forget both on
    # File > New.
    context.register_save_handler(_save_project_state)
    context.register_load_handler(_load_project_state)
    context.register_document_reset_handler(_reset_project_state)


def _save_project_state() -> dict:
    """Return the PMEFF state to persist in the project (``.pmeprj``) file.

    ``last_opt_settings`` is the physics options of the last optimization (None
    when PMEFF has not optimized this document yet); ``geometry_overrides`` is
    the per-atom coordination-geometry override table (keyed by atom index as a
    string, since JSON object keys are strings).
    """
    return {
        "last_opt_settings": _last_opt_settings,
        "geometry_overrides": {
            str(idx): name for idx, name in _geometry_overrides.items()
        },
    }


def _load_project_state(data: object) -> None:
    """Restore the per-atom geometry overrides saved with the project."""
    global _geometry_overrides
    restored: dict = {}
    if isinstance(data, dict):
        raw = data.get("geometry_overrides")
        if isinstance(raw, dict):
            for key, name in raw.items():
                try:
                    restored[int(key)] = str(name)
                except (TypeError, ValueError):
                    continue
    _geometry_overrides = restored


def _reset_project_state() -> None:
    """Forget the last-optimization snapshot and geometry overrides on File > New."""
    global _last_opt_settings, _geometry_overrides
    _last_opt_settings = None
    _geometry_overrides = {}


def _store_geometry_overrides(overrides: dict) -> int:
    """Replace the document's geometry overrides; return how many are set."""
    global _geometry_overrides
    cleaned: dict = {}
    for idx, name in (overrides or {}).items():
        try:
            cleaned[int(idx)] = str(name)
        except (TypeError, ValueError):
            continue
    _geometry_overrides = cleaned
    return len(cleaned)


def _apply_geometry_overrides(context, overrides: dict) -> None:
    """Store the geometry-override table (Apply — does not re-optimize)."""
    n = _store_geometry_overrides(overrides)
    if n:
        _status(
            context,
            f"PMEFF: {n} atom geometry override(s) set — run "
            "Optimize 3D (PMEFF) to apply them.",
            6000,
        )
    else:
        _status(context, "PMEFF: geometry overrides cleared.", 4000)


def _apply_and_optimize_geometry(context, overrides: dict) -> None:
    """Store the overrides and immediately re-optimize the current molecule."""
    _store_geometry_overrides(overrides)
    mol = getattr(context, "current_molecule", None)
    if mol is None:
        _status(context, "PMEFF: no molecule loaded to optimize.", 3000)
        return
    if _optimize(mol, context):
        # Redraw the 3D view so the relaxed geometry is visible immediately.
        refresh = getattr(context, "refresh_3d_view", None)
        if callable(refresh):
            refresh()


def _open_geometry_override_window(context) -> None:
    """Open (or re-show) the modeless Metal Geometry Override table."""
    from .geometry_override_dialog import open_override_window

    open_override_window(
        context,
        dict(_geometry_overrides),
        lambda ov: _apply_geometry_overrides(context, ov),
        lambda ov: _apply_and_optimize_geometry(context, ov),
    )


def _host_window(context):
    """Return the host main window, tolerating either PluginContext API.

    The stable ``PluginContext`` exposes ``get_main_window()`` (there is no
    ``main_window`` attribute); ``getattr`` fallback keeps a minimal mock happy.
    """
    getter = getattr(context, "get_main_window", None)
    if callable(getter):
        return getter()
    return getattr(context, "main_window", None)


def _open_settings_dialog(context) -> None:
    """Open the PMEFF settings dialog and save any changes."""
    from .settings_dialog import open_settings_dialog

    parent = _host_window(context)
    current = load_settings()
    updated = open_settings_dialog(
        parent,
        current,
        defaults=_DEFAULT_SETTINGS,
        on_open_geometry=lambda: _open_geometry_override_window(context),
    )
    if updated is None:
        return  # cancelled or headless
    # Merge: preserve unknown keys the dialog doesn't know about.
    current.update(updated)
    save_settings(current)
    _status(context, "PMEFF settings saved.", 3000)


def _optimize(mol, context) -> bool:
    """Relax *mol* in place with PMEFF. Returns True on success."""
    from .forcefield import optimize_rdkit_mol

    kwargs = _settings_kwargs()
    try:
        success, result = optimize_rdkit_mol(
            mol,
            max_iter=_MAX_ITER,
            geometry_overrides=_overrides_kwarg(),
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF optimization failed")
        _status(context, f"PMEFF optimization failed: {exc}", 5000)
        return False

    if not success:
        _status(context, "PMEFF: no 3D geometry to optimize.", 4000)
        return False

    # Record the options this successful run used, for project persistence.
    global _last_opt_settings
    _last_opt_settings = kwargs

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
        comp = compute_energy_components(
            mol, geometry_overrides=_overrides_kwarg(), **_settings_kwargs()
        )
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
        result = check_minimum(
            mol, geometry_overrides=_overrides_kwarg(), **_settings_kwargs()
        )
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
