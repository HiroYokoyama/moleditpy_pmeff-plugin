"""PMEFF settings dialog (PyQt6).

Opens a modal dialog with checkboxes for each PMEFF physics option.
The PyQt6 import is guarded so this module can be imported in headless
(test) environments without raising ImportError — callers check the
return value of :func:`open_settings_dialog` for None.
"""

from __future__ import annotations

from typing import Callable, Optional


def open_settings_dialog(
    parent: object,
    current: dict,
    defaults: Optional[dict] = None,
    on_open_geometry: Optional[Callable[[], None]] = None,
) -> Optional[dict]:
    """Show the PMEFF settings dialog on top of *parent*.

    Returns the updated settings dict when the user clicks OK, or None if
    they cancel or if PyQt6 is not available (headless environment).

    *defaults* supplies the values the "Restore Defaults" button resets each
    option to; when omitted, the dialog's own per-option defaults are used.

    *on_open_geometry*, when given, is invoked by a "Metal Geometry Override…"
    button that opens the per-atom coordination-geometry table.
    """
    try:
        from PyQt6.QtWidgets import (  # type: ignore[import]
            QDialog,
            QDialogButtonBox,
            QLabel,
            QVBoxLayout,
            QCheckBox,
            QPushButton,
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
    option_defaults: dict[str, bool] = {}

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
        option_defaults[key] = default

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
    _add(
        "polar_contraction",
        "Polar bond contraction",
        "Shortens polar bond rest lengths by a capped, quadratic function of "
        "the Allred-Rochow electronegativity difference (Schomaker-Stevenson "
        "idea). Fixes bonds the plain covalent-radius sum leaves too long — "
        "Si-O 1.79→1.63 Å, P=O, B-O, metal-oxides, C-F — while leaving "
        "organic C-C/C-O/C-N/C-H untouched. On by default.",
        default=True,
    )

    # Per-atom coordination-geometry override table (chiefly for metals).
    if on_open_geometry is not None:
        layout.addSpacing(4)
        geom_btn = QPushButton("Metal Geometry Override…")
        geom_btn.setToolTip(
            "Force the coordination geometry of individual atoms "
            "(linear, trigonal/square planar, tetrahedral, octahedral)."
        )
        geom_btn.clicked.connect(lambda: on_open_geometry())
        geom_desc = QLabel(
            "Override the auto-detected geometry of individual metal centers. "
            "Applied on the next Optimize 3D (PMEFF) and saved with the project."
        )
        geom_desc.setWordWrap(True)
        geom_desc.setContentsMargins(0, 2, 0, 0)
        geom_desc.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(geom_btn)
        layout.addWidget(geom_desc)

    layout.addSpacing(8)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel
        | QDialogButtonBox.StandardButton.RestoreDefaults
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)

    # "Restore Defaults" resets every checkbox to its default (the caller-
    # supplied *defaults*, falling back to each option's own default).
    reset_to = {**option_defaults, **(defaults or {})}

    def _restore_defaults() -> None:
        for key, cb in checks.items():
            cb.setChecked(bool(reset_to.get(key, True)))

    restore_btn = buttons.button(
        QDialogButtonBox.StandardButton.RestoreDefaults
    )
    restore_btn.clicked.connect(_restore_defaults)
    layout.addWidget(buttons)

    if dlg.exec() == QDialog.DialogCode.Accepted:
        return {key: cb.isChecked() for key, cb in checks.items()}
    return None
