"""PMEFF settings dialog (PyQt6).

Opens a modal dialog with checkboxes for each PMEFF physics option.
The PyQt6 import is guarded so this module can be imported in headless
(test) environments without raising ImportError — callers check the
return value of :func:`open_settings_dialog` for None.
"""

from __future__ import annotations

from typing import Optional


def open_settings_dialog(parent: object, current: dict) -> Optional[dict]:
    """Show the PMEFF settings dialog on top of *parent*.

    Returns the updated settings dict when the user clicks OK, or None if
    they cancel or if PyQt6 is not available (headless environment).
    """
    try:
        from PyQt6.QtWidgets import (  # type: ignore[import]
            QDialog,
            QDialogButtonBox,
            QLabel,
            QVBoxLayout,
            QCheckBox,
        )
        from PyQt6.QtCore import Qt  # type: ignore[import]
    except ImportError:
        return None

    dlg = QDialog(parent)  # type: ignore[arg-type]
    dlg.setWindowTitle("PMEFF Settings")
    dlg.setMinimumWidth(430)
    layout = QVBoxLayout(dlg)
    layout.setSpacing(2)

    intro = QLabel(
        "<b>PMEFF</b> is a pre-DFT geometry-cleanup force field. "
        "Its goal is <i>initial structure preparation</i> — removing "
        "clashes, correcting bond lengths and angles, and giving the DFT "
        "optimizer a sensible starting point — not high-accuracy energetics."
    )
    intro.setWordWrap(True)
    intro.setContentsMargins(0, 0, 0, 8)
    layout.addWidget(intro)

    checks: dict[str, QCheckBox] = {}

    def _add(key: str, label: str, desc: str, default: bool = True) -> None:
        cb = QCheckBox(label)
        cb.setChecked(bool(current.get(key, default)))
        lbl = QLabel(desc)
        lbl.setWordWrap(True)
        lbl.setContentsMargins(22, 0, 0, 6)
        lbl.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(cb)
        layout.addWidget(lbl)
        checks[key] = cb

    _add(
        "electronic_effects",
        "Electronic effects",
        "QEq partial charges (Allred-Rochow χ, Slater Zeff, Ohno-shielded "
        "Coulomb) with Bent's-rule χ scaling. Square-planar angle targets for "
        "4-coordinate d8 metals (Ni, Pd, Pt, Rh, Ir, Au) and octahedral targets "
        "for 6-coordinate d-block metals. On by default.",
        default=True,
    )
    _add(
        "morse_bonds",
        "Morse bond stretching",
        "Replaces the harmonic bond term with the Morse potential "
        "D(1 − e^{−α Δr})². Same curvature at the minimum; energy is "
        "bounded above at D (≈ 84 kcal/mol for C-C) rather than diverging. "
        "Improves robustness for severely distorted starting geometries. "
        "No performance overhead.",
        default=True,
    )
    _add(
        "hbond",
        "Hydrogen bond correction",
        "Geometry-dependent D−H···A attraction (donors / acceptors: N, O, F, S). "
        "Energy: ε · [(R₀/r)¹² − 2(R₀/r)⁶] · cos²(∠DHA). "
        "Improves H-bond distances and linearity. Fast.",
        default=True,
    )
    _add(
        "dispersion",
        "Dispersion correction  [may slow large molecules]",
        "Becke-Johnson damped C₆/r⁶ London dispersion added on top of the "
        "Lennard-Jones term: −c₆/(r⁶ + rmin⁶). Improves aromatic stacking "
        "distances and hydrophobic contacts. Off by default.",
        default=False,
    )

    layout.addSpacing(8)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() == QDialog.DialogCode.Accepted:
        return {key: cb.isChecked() for key, cb in checks.items()}
    return None
