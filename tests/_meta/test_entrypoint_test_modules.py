"""Invariant: the canonical entrypoint test modules stay entrypoint-only.

``test_train.py`` / ``test_eval.py`` / ``test_generate_dataset.py`` each compose
one Hydra ``cfg`` and drive its in-process entrypoint; helper/unit tests for the
entrypoint module's private functions belong in a sibling ``test_<name>_*.py``.

A reference to a private (``_``-prefixed) ``synth_setter.cli`` member is the tell
one crept back in. This static AST check catches both the
``from synth_setter.cli.x import _helper`` import and the ``import ... as m`` alias
form followed by ``m._helper`` access; it neither imports the modules nor runs them.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_PACKAGE = "synth_setter.cli"

_CANONICAL_ENTRYPOINT_TEST_FILES = (
    "tests/test_train.py",
    "tests/test_eval.py",
    "tests/test_generate_dataset.py",
)


def _private_cli_references(tree: ast.AST) -> list[str]:
    """Collect references to private ``synth_setter.cli`` members in a parsed module.

    Covers a direct ``from synth_setter.cli.x import _helper`` import and a
    ``synth_setter.cli`` module bound to an alias (``import ... as m`` /
    ``from synth_setter.cli import x as m``) then read as ``m._helper``.

    :param tree: Parsed AST of a test module.
    :returns: Sorted dotted names of every private (``_``-prefixed)
        ``synth_setter.cli`` member the module reaches.
    """
    cli_aliases: dict[str, str] = {}
    refs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(_CLI_PACKAGE):
            for alias in node.names:
                if alias.name.startswith("_"):
                    refs.append(f"{node.module}.{alias.name}")
                else:
                    cli_aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_CLI_PACKAGE):
                    cli_aliases[alias.asname or alias.name] = alias.name
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and isinstance(node.value, ast.Name)
            and node.value.id in cli_aliases
        ):
            refs.append(f"{cli_aliases[node.value.id]}.{node.attr}")
    return sorted(set(refs))


@pytest.mark.parametrize("test_file", _CANONICAL_ENTRYPOINT_TEST_FILES)
def test_canonical_entrypoint_module_references_no_private_cli_helper(test_file: str) -> None:
    """A canonical entrypoint test module references no private ``synth_setter.cli`` member.

    :param test_file: Repo-relative path to a canonical entrypoint test module.
    """
    tree = ast.parse((_REPO_ROOT / test_file).read_text())
    private = _private_cli_references(tree)
    assert not private, (
        f"{test_file} references private synth_setter.cli helpers {private}. "
        "Keep this module to cfg-entrypoint tests; move helper/unit tests into a "
        "sibling test_<entrypoint>_<concern>.py module instead."
    )
