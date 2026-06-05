"""Invariant: canonical entrypoint test modules do not contain config-layer tests.

``test_generate_dataset.py`` and ``test_train.py`` are for tests that drive the
real CLI entrypoint (``from_hydra``, ``train``, or ``subprocess``). Config-level
tests — Hydra compose + ``spec_from_cfg`` / schema validation without running the
entrypoint — belong in ``tests/pipeline/configs/``.

The tell is a direct ``initialize_config_module`` import: it indicates the test
manages its own Hydra lifecycle, meaning it is doing config-layer work rather
than driving the entrypoint through a fixture. Tests that exercise the real
entrypoint receive a composed ``cfg`` via fixture and call ``from_hydra(cfg)`` /
``train(cfg)`` / ``subprocess.run([..., "synth-setter-..."])``.

Note: ``test_eval.py`` is excluded — it legitimately composes a cfg inline and
immediately calls ``evaluate(cfg)``, which is e2e.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ENTRYPOINT_ONLY_TEST_FILES = (
    "tests/test_generate_dataset.py",
    "tests/test_train.py",
)

_BANNED_HYDRA_IMPORTS = frozenset(
    {
        "initialize_config_module",
        "initialize",
        "initialize_config_dir",
    }
)


def _direct_hydra_compose_imports(tree: ast.AST) -> list[str]:
    """Return names of hydra config-initializer functions imported at module level.

    :param tree: Parsed AST of the test module.
    :returns: Sorted list of banned hydra function names imported in the module.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hydra"):
            for alias in node.names:
                if alias.name in _BANNED_HYDRA_IMPORTS:
                    found.append(alias.name)
    return sorted(set(found))


@pytest.mark.parametrize("test_file", _ENTRYPOINT_ONLY_TEST_FILES)
def test_entrypoint_module_does_not_contain_config_layer_imports(test_file: str) -> None:
    """Banned Hydra config-initializer imports must not appear in canonical entrypoint tests.

    Config-layer tests (bare Hydra compose + spec validation without running the
    entrypoint) belong in ``tests/pipeline/configs/``. A direct import of
    ``initialize_config_module`` (or the older ``initialize`` / ``initialize_config_dir``
    variants) in these files is the tell.

    :param test_file: Repo-relative path to a canonical entrypoint test module.
    """
    path = _REPO_ROOT / test_file
    tree = ast.parse(path.read_text(), filename=test_file)
    banned = _direct_hydra_compose_imports(tree)
    assert not banned, (
        f"{test_file} imports hydra config-initializer(s) {banned}. "
        "Config-composition tests (compose + spec_from_cfg/schema-only assertions) "
        "belong in tests/pipeline/configs/. "
        "Entrypoint tests receive a composed cfg via fixture and call "
        "from_hydra(cfg) / train(cfg) / subprocess.run([...])."
    )
