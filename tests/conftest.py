"""Shared test infrastructure for moleditpy_pmeff-plugin.

The PMEFF engine (``pmeff_plugin.forcefield``) uses real ``numpy`` and,
at its boundary, real ``rdkit`` — both are required to exercise the actual
math and are available in the test environment. Only the heavy *host* imports
(PyQt6, moleditpy) are mocked, and only for the plugin-entry (``__init__``)
tests via :func:`load_plugin_entry`.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Only the host GUI stack is mocked; numpy/rdkit stay real.
BLOCKED_TOPS: frozenset[str] = frozenset({"PyQt6", "moleditpy"})


class _MagicLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> MagicMock:
        m = MagicMock()
        m.__name__ = spec.name
        m.__spec__ = spec
        m.__path__ = []
        m.__package__ = spec.name.split(".")[0]
        return m  # type: ignore[return-value]

    def exec_module(self, module: object) -> None:
        pass


class _MagicFinder(importlib.abc.MetaPathFinder):
    _loader = _MagicLoader()

    def find_spec(
        self,
        fullname: str,
        path: object,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".")[0] in BLOCKED_TOPS:
            return importlib.machinery.ModuleSpec(fullname, self._loader)
        return None


@contextlib.contextmanager
def mock_host_imports() -> Generator[None, None, None]:
    removed = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k.split(".")[0] in BLOCKED_TOPS
    }
    finder = _MagicFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(removed)
        for k in list(sys.modules):
            if k.split(".")[0] in BLOCKED_TOPS and k not in removed:
                del sys.modules[k]


def make_context() -> MagicMock:
    """Return a stub PluginContext with a non-None main window."""
    ctx = MagicMock()
    ctx.get_main_window.return_value = MagicMock()
    return ctx
