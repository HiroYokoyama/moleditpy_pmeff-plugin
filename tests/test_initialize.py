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
