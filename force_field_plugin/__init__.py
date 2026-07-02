"""PMEFF — Python Molecular Editor Force Field.

A MoleditPy plugin providing PMEFF, a self-contained universal force field
that covers the entire periodic table (Z = 1..118). It registers:

* an **Optimize 3D** method ("PMEFF (Universal)") that relaxes the current 3D
  geometry with a dependency-free FIRE optimizer, and
* an **Analysis** tool ("PMEFF Single-Point Energy") that reports the current
  force-field energy without modifying the molecule.

See ``forcefield.py`` for the physics; this module only wires it into the host
via the stable ``PluginContext`` API.
"""

import logging

PLUGIN_NAME = "PMEFF Force Field"
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "PMEFF (Python Molecular Editor Force Field) — a self-contained universal "
    "force field covering the entire periodic table (Z=1..118). Adds a "
    "geometry optimizer and a single-point energy tool. No external QM binary "
    "required."
)
PLUGIN_CATEGORY = "Optimization"
PLUGIN_TAGS = ["Optimization", "Force Field", "3D"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"
PLUGIN_DEPENDENCIES = ["numpy", "rdkit"]

_OPT_METHOD_NAME = "PMEFF (Universal)"
_MAX_ITER = 1000

logger = logging.getLogger(__name__)


def initialize(context):
    """Register PMEFF's optimization method and energy tool."""
    context.register_optimization_method(
        _OPT_METHOD_NAME, lambda mol: _optimize(mol, context)
    )
    context.add_analysis_tool(
        "PMEFF Single-Point Energy", lambda: _show_energy(context)
    )


def _optimize(mol, context) -> bool:
    """Relax *mol* in place with PMEFF. Returns True on success."""
    from .forcefield import optimize_rdkit_mol

    try:
        success, result = optimize_rdkit_mol(mol, max_iter=_MAX_ITER)
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
    from .forcefield import compute_energy

    mol = context.current_molecule
    if mol is None:
        _status(context, "PMEFF: no molecule loaded.", 3000)
        return

    try:
        energy = compute_energy(mol)
    except Exception as exc:  # pragma: no cover - defensive GUI guard
        logger.exception("PMEFF energy evaluation failed")
        _status(context, f"PMEFF energy failed: {exc}", 5000)
        return

    if energy is None:
        _status(context, "PMEFF: molecule has no 3D coordinates.", 4000)
        return

    _status(context, f"PMEFF single-point energy: {energy:.4f}", 6000)


def _status(context, message: str, timeout: int = 3000) -> None:
    """Show a status-bar message, tolerating a minimal/mock context."""
    show = getattr(context, "show_status_message", None)
    if callable(show):
        show(message, timeout)
    else:  # pragma: no cover - only hit with a bare context
        logger.info(message)
