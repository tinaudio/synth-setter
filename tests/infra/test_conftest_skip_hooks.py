"""Tests for conftest.py auto-skips for native and remote prerequisites.

Exercises ``pytest_collection_modifyitems`` directly with a lightweight item double
to verify both the skip-inserted and run-through branches for each marker.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import tests.conftest as conftest_module


class _FakeConfig:
    """Minimal pytest config double exposing the marker expression."""

    def __init__(self, mark_expression: str) -> None:
        """Store the marker expression returned for pytest's ``-m`` option.

        :param mark_expression: Value supplied to pytest's ``-m`` option.
        """
        self.mark_expression = mark_expression

    def getoption(self, name: str) -> str:
        """Return the configured marker expression.

        :param name: Pytest option name requested by the hook.
        :returns: The configured marker expression.
        """
        assert name == "-m"
        return self.mark_expression


class _FakeItem:
    """Minimal pytest item double sufficient for the skip hook."""

    def __init__(self, keywords: dict[str, pytest.MarkDecorator]) -> None:
        """Initialise with a keyword dict that mimics ``item.keywords``.

        :param keywords: marker-name-to-marker mapping the hook inspects.
        """
        self.keywords = keywords
        self.added_markers: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        """Record the marker — mirrors ``pytest.Item.add_marker``.

        :param marker: appended to ``added_markers`` for assertion.
        """
        self.added_markers.append(marker)


@pytest.mark.infra
def test_same_e2e_selected_loader_item_without_vst_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected loader-only SAME item does not require a VST plugin.

    :param monkeypatch: Simulates an unavailable VST plugin.
    """
    monkeypatch.setattr(conftest_module, "VST_AVAILABLE", False)
    item = _FakeItem({"same_e2e": pytest.mark.same_e2e})

    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("same_e2e")),
        items=cast(list[pytest.Item], [item]),
    )

    assert item.added_markers == []


@pytest.mark.infra
def test_same_e2e_marker_expression_excluding_vst_collects_encoder_tests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real pytest selection excludes VST-backed SAME tests before prerequisite validation.

    :param monkeypatch: Sets a guaranteed-absent plugin path for the child process.
    :param tmp_path: Parent of the guaranteed-absent plugin path.
    """
    monkeypatch.setenv("SYNTH_SETTER_PLUGIN_PATH", str(tmp_path / "absent.vst3"))
    repo_root = Path(__file__).parents[2]

    result = subprocess.run(  # noqa: S603 — interpreter and arguments are test-controlled
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "same_e2e and not requires_vst",
            "tests/pipeline/data/test_same_encoder_e2e.py",
            "tests/test_eval.py",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "test_same_hydra_main_writes_legacy_matching_lance_column[same_s-12]" in result.stdout
    assert "test_same_hydra_main_writes_legacy_matching_lance_column[same_l-11]" in result.stdout
    assert "test_train_eval_same_conditioning_real_e2e" not in result.stdout


@pytest.mark.infra
def test_same_e2e_selected_requires_vst_item_without_vst_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected SAME item marked requires_vst retains fail-fast behavior.

    :param monkeypatch: Simulates an unavailable VST plugin.
    """
    monkeypatch.setattr(conftest_module, "VST_AVAILABLE", False)
    item = _FakeItem(
        {
            "requires_vst": pytest.mark.requires_vst,
            "same_e2e": pytest.mark.same_e2e,
        }
    )

    with pytest.raises(pytest.UsageError, match="SYNTH_SETTER_PLUGIN_PATH") as error:
        conftest_module.pytest_collection_modifyitems(
            config=cast(pytest.Config, _FakeConfig("same_e2e")),
            items=cast(list[pytest.Item], [item]),
        )

    assert conftest_module.PLUGIN_PATH in str(error.value)


@pytest.mark.infra
def test_same_e2e_selected_requires_vst_item_with_vst_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected SAME VST item remains runnable when its plugin exists.

    :param monkeypatch: Simulates an available VST plugin.
    """
    monkeypatch.setattr(conftest_module, "VST_AVAILABLE", True)
    item = _FakeItem(
        {
            "requires_vst": pytest.mark.requires_vst,
            "same_e2e": pytest.mark.same_e2e,
        }
    )

    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("slow and same_e2e")),
        items=cast(list[pytest.Item], [item]),
    )

    assert item.added_markers == []


@pytest.mark.infra
def test_requires_vst_item_skipped_when_vst_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """requires_vst item gets a skip marker with a path-specific reason when VST is absent.

    :param monkeypatch: rebinds ``VST_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "VST_AVAILABLE", False)
    item = _FakeItem({"requires_vst": pytest.mark.requires_vst})
    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("")),
        items=cast(list[pytest.Item], [item]),
    )
    assert len(item.added_markers) == 1
    assert "VST plugin not found" in item.added_markers[0].kwargs["reason"]


@pytest.mark.infra
def test_requires_surgepy_item_skipped_when_extension_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requires_surgepy items skip when the platform-specific extension is absent.

    :param monkeypatch: Rebinds the cached SurgePy capability.
    """
    monkeypatch.setattr(conftest_module, "_SURGEPY_AVAILABLE", False)
    item = _FakeItem({"requires_surgepy": pytest.mark.requires_surgepy})

    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("")),
        items=cast(list[pytest.Item], [item]),
    )

    assert len(item.added_markers) == 1
    assert "surgepy native extension is unavailable" in item.added_markers[0].kwargs["reason"]


@pytest.mark.infra
def test_requires_surgepy_item_not_skipped_when_extension_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requires_surgepy items run when the native extension is installed.

    :param monkeypatch: Rebinds the cached SurgePy capability.
    """
    monkeypatch.setattr(conftest_module, "_SURGEPY_AVAILABLE", True)
    item = _FakeItem({"requires_surgepy": pytest.mark.requires_surgepy})

    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("")),
        items=cast(list[pytest.Item], [item]),
    )

    assert item.added_markers == []


@pytest.mark.infra
def test_integration_r2_item_skipped_when_r2_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """integration_r2 item gets a probe hint when R2 is absent.

    :param monkeypatch: rebinds ``_R2_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "_R2_AVAILABLE", False)
    item = _FakeItem({"integration_r2": pytest.mark.integration_r2})
    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("")),
        items=cast(list[pytest.Item], [item]),
    )
    assert len(item.added_markers) == 1
    assert "rclone lsd r2:" in item.added_markers[0].kwargs["reason"]


@pytest.mark.infra
def test_requires_vst_item_not_skipped_when_vst_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """requires_vst item receives no skip marker when VST is present.

    :param monkeypatch: rebinds ``VST_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "VST_AVAILABLE", True)
    item = _FakeItem({"requires_vst": pytest.mark.requires_vst})
    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("")),
        items=cast(list[pytest.Item], [item]),
    )
    assert item.added_markers == []


@pytest.mark.infra
def test_integration_r2_item_not_skipped_when_r2_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """integration_r2 item receives no skip marker when R2 is reachable.

    :param monkeypatch: rebinds ``_R2_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "_R2_AVAILABLE", True)
    item = _FakeItem({"integration_r2": pytest.mark.integration_r2})
    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("")),
        items=cast(list[pytest.Item], [item]),
    )
    assert item.added_markers == []


@pytest.mark.infra
def test_unmarked_item_receives_no_skip_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """An item with no VST/R2 markers is untouched regardless of resource availability.

    :param monkeypatch: rebinds both ``VST_AVAILABLE`` and ``_R2_AVAILABLE`` to False.
    """
    monkeypatch.setattr(conftest_module, "VST_AVAILABLE", False)
    monkeypatch.setattr(conftest_module, "_R2_AVAILABLE", False)
    item = _FakeItem({"slow": pytest.mark.slow})
    conftest_module.pytest_collection_modifyitems(
        config=cast(pytest.Config, _FakeConfig("")),
        items=cast(list[pytest.Item], [item]),
    )
    assert item.added_markers == []
