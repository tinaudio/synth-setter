"""Tests for the conftest.py auto-skip hooks for requires_vst and integration_r2.

Exercises ``pytest_collection_modifyitems`` directly with a lightweight item double
to verify both the skip-inserted and run-through branches for each marker.
"""

from __future__ import annotations

from typing import cast

import pytest

import tests.conftest as conftest_module


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
def test_requires_vst_item_skipped_when_vst_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """requires_vst item gets a skip marker with a path-specific reason when VST is absent.

    :param monkeypatch: rebinds ``_VST_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "_VST_AVAILABLE", False)
    item = _FakeItem({"requires_vst": pytest.mark.requires_vst})
    conftest_module.pytest_collection_modifyitems(items=cast(list[pytest.Item], [item]))
    assert len(item.added_markers) == 1
    assert "Surge XT VST not found" in item.added_markers[0].kwargs["reason"]


@pytest.mark.infra
def test_integration_r2_item_skipped_when_r2_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """integration_r2 item gets a skip marker with a credential hint when R2 is absent.

    :param monkeypatch: rebinds ``_R2_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "_R2_AVAILABLE", False)
    item = _FakeItem({"integration_r2": pytest.mark.integration_r2})
    conftest_module.pytest_collection_modifyitems(items=cast(list[pytest.Item], [item]))
    assert len(item.added_markers) == 1
    assert "RCLONE_CONFIG_R2_ACCESS_KEY_ID" in item.added_markers[0].kwargs["reason"]


@pytest.mark.infra
def test_requires_vst_item_not_skipped_when_vst_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """requires_vst item receives no skip marker when VST is present.

    :param monkeypatch: rebinds ``_VST_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "_VST_AVAILABLE", True)
    item = _FakeItem({"requires_vst": pytest.mark.requires_vst})
    conftest_module.pytest_collection_modifyitems(items=cast(list[pytest.Item], [item]))
    assert item.added_markers == []


@pytest.mark.infra
def test_integration_r2_item_not_skipped_when_r2_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """integration_r2 item receives no skip marker when R2 credentials are present.

    :param monkeypatch: rebinds ``_R2_AVAILABLE`` on ``conftest_module``.
    """
    monkeypatch.setattr(conftest_module, "_R2_AVAILABLE", True)
    item = _FakeItem({"integration_r2": pytest.mark.integration_r2})
    conftest_module.pytest_collection_modifyitems(items=cast(list[pytest.Item], [item]))
    assert item.added_markers == []


@pytest.mark.infra
def test_unmarked_item_receives_no_skip_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """An item with no VST/R2 markers is untouched regardless of resource availability.

    :param monkeypatch: rebinds both ``_VST_AVAILABLE`` and ``_R2_AVAILABLE`` to False.
    """
    monkeypatch.setattr(conftest_module, "_VST_AVAILABLE", False)
    monkeypatch.setattr(conftest_module, "_R2_AVAILABLE", False)
    item = _FakeItem({"slow": pytest.mark.slow})
    conftest_module.pytest_collection_modifyitems(items=cast(list[pytest.Item], [item]))
    assert item.added_markers == []
