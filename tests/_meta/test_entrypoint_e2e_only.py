"""Invariant: canonical entrypoint test modules contain only e2e tests.

``test_generate_dataset.py`` and ``test_train.py`` are for tests that drive the
real CLI entrypoint (``from_hydra``, ``train``, or ``subprocess``). Config-layer
tests (Hydra compose + ``spec_from_cfg`` without running the entrypoint) belong in
``tests/pipeline/configs/``. The tell is a direct ``initialize_config_module``
import — it means the test manages its own Hydra lifecycle.

``test_eval.py`` is excluded: it composes a cfg inline and immediately calls
``evaluate(cfg)``, which is e2e. Refs #1345.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ENTRYPOINT_ONLY_TEST_FILES: tuple[str, ...] = (
    "tests/test_generate_dataset.py",
    "tests/test_train.py",
)

_BANNED_HYDRA_IMPORTS = frozenset(
    {
        "initialize",  # legacy Hydra 1.x alias; same lifecycle semantics as initialize_config_module
        "initialize_config_dir",
        "initialize_config_module",
    }
)


def _direct_hydra_compose_imports(tree: ast.AST) -> list[str]:
    """Return sorted banned hydra config-initializer names imported in ``tree``.

    :param tree: Parsed AST of the test module.
    :returns: Sorted list of banned hydra function names found in the module.
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
