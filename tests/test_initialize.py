"""Tests for the plugin entry point (pmeff_plugin.__init__)."""

from __future__ import annotations

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
    assert plugin.PLUGIN_VERSION == "0.1.0"
    assert "numpy" in plugin.PLUGIN_DEPENDENCIES
    assert "rdkit" in plugin.PLUGIN_DEPENDENCIES


def test_initialize_registers_method_and_tool():
    ctx = make_context()
    plugin.initialize(ctx)
    ctx.register_optimization_method.assert_called_once()
    name = ctx.register_optimization_method.call_args[0][0]
    assert name == "PMEFF (Universal)"
    ctx.add_analysis_tool.assert_called_once()


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
    tool_callback = ctx.add_analysis_tool.call_args[0][1]
    tool_callback()
    msg = ctx.show_status_message.call_args[0][0]
    assert "energy" in msg.lower()


def test_energy_tool_handles_no_molecule():
    ctx = make_context()
    ctx.current_molecule = None
    plugin.initialize(ctx)
    tool_callback = ctx.add_analysis_tool.call_args[0][1]
    tool_callback()
    msg = ctx.show_status_message.call_args[0][0]
    assert "no molecule" in msg.lower()


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
