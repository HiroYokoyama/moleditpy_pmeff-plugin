"""Tests for the plugin entry point (pmeff_plugin.__init__)."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from rdkit import Chem
from rdkit.Chem import AllChem

import pmeff_plugin as plugin
from tests.conftest import make_context


def _embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def test_metadata_present():
    assert plugin.PLUGIN_NAME == "PMEFF Plugin"
    # Shape only, not a pinned literal: version bumps must not break tests.
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)
    assert "numpy" in plugin.PLUGIN_DEPENDENCIES
    assert "rdkit" in plugin.PLUGIN_DEPENDENCIES


def _analysis_tools(ctx):
    return {c.args[0]: c.args[1] for c in ctx.add_analysis_tool.call_args_list}


def test_initialize_registers_method_and_tools():
    ctx = make_context()
    plugin.initialize(ctx)
    ctx.register_optimization_method.assert_called_once()
    name = ctx.register_optimization_method.call_args[0][0]
    assert name == f"PMEFF (v{plugin.PLUGIN_VERSION})"
    assert set(_analysis_tools(ctx)) == {
        "PMEFF Single-Point Energy",
        "PMEFF Minimum Check (Vibrational)",
    }


def test_settings_toggle_lives_under_settings_menu():
    ctx = make_context()
    plugin.initialize(ctx)
    ctx.add_menu_action.assert_called_once()
    assert ctx.add_menu_action.call_args[0][0] == "Settings/PMEFF Setting"


def test_registered_optimizer_callback_runs():
    ctx = make_context()
    plugin.initialize(ctx)
    callback = ctx.register_optimization_method.call_args[0][1]
    mol = _embed("CCO")
    assert callback(mol) is True
    ctx.show_status_message.assert_called()


def test_optimizer_callback_reports_failure_without_conformer():
    ctx = make_context()
    plugin.initialize(ctx)
    callback = ctx.register_optimization_method.call_args[0][1]
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))  # no conformer
    assert callback(mol) is False


def test_energy_tool_reports_energy():
    ctx = make_context()
    ctx.current_molecule = _embed("CCO")
    plugin.initialize(ctx)
    _analysis_tools(ctx)["PMEFF Single-Point Energy"]()
    msg = ctx.show_status_message.call_args[0][0]
    assert "energy" in msg.lower()


def test_energy_tool_handles_no_molecule():
    ctx = make_context()
    ctx.current_molecule = None
    plugin.initialize(ctx)
    _analysis_tools(ctx)["PMEFF Single-Point Energy"]()
    msg = ctx.show_status_message.call_args[0][0]
    assert "no molecule" in msg.lower()


def test_minimum_check_tool_reports_verdict():
    ctx = make_context()
    mol = _embed("CO")
    plugin.initialize(ctx)
    callback = ctx.register_optimization_method.call_args[0][1]
    assert callback(mol) is True
    ctx.current_molecule = mol
    _analysis_tools(ctx)["PMEFF Minimum Check (Vibrational)"]()
    msg = ctx.show_status_message.call_args[0][0]
    assert "minimum" in msg.lower()


def test_minimum_check_tool_handles_no_conformer():
    ctx = make_context()
    ctx.current_molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    plugin.initialize(ctx)
    _analysis_tools(ctx)["PMEFF Minimum Check (Vibrational)"]()
    msg = ctx.show_status_message.call_args[0][0]
    assert "no 3d" in msg.lower()


def test_initialize_registers_save_handler_but_not_load():
    ctx = make_context()
    plugin.initialize(ctx)
    # Write-only persistence: a save handler and a document-reset handler are
    # registered, but deliberately no load handler.
    ctx.register_save_handler.assert_called_once()
    ctx.register_document_reset_handler.assert_called_once()
    ctx.register_load_handler.assert_not_called()


def test_save_handler_is_none_before_any_optimization():
    plugin._last_opt_settings = None
    ctx = make_context()
    plugin.initialize(ctx)
    save_cb = ctx.register_save_handler.call_args[0][0]
    assert save_cb() == {"last_opt_settings": None}


def test_save_handler_snapshots_last_optimization_settings():
    plugin._last_opt_settings = None
    ctx = make_context()
    plugin.initialize(ctx)
    optimize = ctx.register_optimization_method.call_args[0][1]
    save_cb = ctx.register_save_handler.call_args[0][0]

    assert optimize(_embed("CCO")) is True
    snap = save_cb()["last_opt_settings"]
    # The saved snapshot is the exact kwargs the run used.
    assert snap == plugin._settings_kwargs()
    assert set(snap) == {
        "electronic_effects",
        "use_morse",
        "use_hbond",
        "use_dispersion",
        "use_polar_contraction",
    }


def test_document_reset_forgets_last_optimization():
    ctx = make_context()
    plugin.initialize(ctx)
    optimize = ctx.register_optimization_method.call_args[0][1]
    save_cb = ctx.register_save_handler.call_args[0][0]
    reset_cb = ctx.register_document_reset_handler.call_args[0][0]

    assert optimize(_embed("CCO")) is True
    assert save_cb()["last_opt_settings"] is not None
    reset_cb()
    assert save_cb()["last_opt_settings"] is None


def test_optimizer_callback_survives_engine_exception(monkeypatch):
    ctx = make_context()
    plugin.initialize(ctx)
    callback = ctx.register_optimization_method.call_args[0][1]

    def boom(*_args, **_kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(plugin, "_optimize", plugin._optimize)
    monkeypatch.setattr(
        "pmeff_plugin.forcefield.optimize_rdkit_mol", boom
    )
    assert callback(_embed("CCO")) is False
