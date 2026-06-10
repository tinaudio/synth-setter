"""Unit tests for the citest checkpoint-artifact helper's offline-testable seams.

The live round-trip (publish → resolve → cleanup) is exercised by the slow,
``WANDB_API_KEY``-gated tests in ``test_train`` / ``test_eval``. These cover the
two pieces that need no network: run-scoped naming and best-effort deletion.
"""

from __future__ import annotations

import pytest

from tests.helpers import wandb_artifacts


def test_run_scoped_name_distinct_runs_distinct_names() -> None:
    """Distinct run ids yield distinct, base-prefixed collection names."""
    base = "model-citest-ffn_full-resume"
    name_a = wandb_artifacts._run_scoped_name(base, "abc123")
    name_b = wandb_artifacts._run_scoped_name(base, "def456")

    assert name_a != name_b, "same name across runs would collide on the shared collection"
    assert name_a.startswith(f"{base}-") and name_b.startswith(f"{base}-")
    assert "abc123" in name_a and "def456" in name_b


def test_delete_artifact_best_effort_api_error_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing delete is swallowed — scratch cleanup must never fail the test.

    :param monkeypatch: Stubs ``wandb.Api`` to raise on construction.
    """
    import wandb

    def _raise() -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(wandb, "Api", _raise)
    # Must return normally despite the API blowing up.
    wandb_artifacts._delete_artifact_best_effort("entity/proj/name:latest")


def test_delete_artifact_best_effort_passes_delete_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The version carries a ``:latest`` alias, so deletion must pass ``delete_aliases=True``.

    :param monkeypatch: Stubs ``wandb.Api`` with a fake recording the delete call.
    """
    import wandb

    calls: dict[str, object] = {}

    class _FakeArtifact:
        def delete(self, delete_aliases: bool = False) -> None:
            calls["delete_aliases"] = delete_aliases

    class _FakeApi:
        def artifact(self, ref: str) -> _FakeArtifact:
            calls["ref"] = ref
            return _FakeArtifact()

    monkeypatch.setattr(wandb, "Api", _FakeApi)
    wandb_artifacts._delete_artifact_best_effort("entity/proj/name:latest")

    assert calls["ref"] == "entity/proj/name:latest"
    assert calls["delete_aliases"] is True
