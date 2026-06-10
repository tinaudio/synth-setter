"""Unit tests for the citest checkpoint-artifact helper's offline-testable seams.

The live round-trip (publish → resolve → cleanup) is exercised by the slow,
``WANDB_API_KEY``-gated tests in ``test_train`` / ``test_eval``. These cover the
pieces that need no network: run-scoped naming, cleanup-on-exit, and best-effort
deletion — using ``MagicMock`` to stand in for the W&B publish/delete surface.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


def test_published_checkpoint_artifact_deletes_version_when_body_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cleanup deletes the run-scoped version even when the ``with`` body raises.

    Pins the context manager's core guarantee — and that the ref it yields is run-scoped —
    without touching the network, by faking the W&B publish + delete surface.

    :param monkeypatch: Stubs the ``wandb`` init/Artifact/Api surface with mocks.
    :param tmp_path: Source for the (faked) checkpoint file the helper embeds.
    """
    import wandb

    fake_run = MagicMock(id="run42", entity="ent")
    fake_api = MagicMock()
    monkeypatch.setattr(wandb, "init", lambda **_kwargs: fake_run)
    monkeypatch.setattr(wandb, "Artifact", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(wandb, "Api", lambda: fake_api)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"x")
    expected_ref = "ent/synth-setter-citest/model-citest-x-run42:latest"

    def _boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):  # noqa: PT012 - the with-body is the scenario
        with wandb_artifacts.published_checkpoint_artifact(
            ckpt, "model-citest-x", tmp_path
        ) as ref:
            assert ref == expected_ref
            _boom()

    fake_api.artifact.assert_called_once_with(expected_ref)
    fake_api.artifact.return_value.delete.assert_called_once_with(delete_aliases=True)


def test_delete_artifact_best_effort_api_error_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing delete is swallowed — scratch cleanup must never fail the test.

    :param monkeypatch: Stubs ``wandb.Api`` to raise on construction.
    """
    import wandb

    monkeypatch.setattr(wandb, "Api", MagicMock(side_effect=RuntimeError("network down")))
    # Must return normally despite the API blowing up.
    wandb_artifacts._delete_artifact_best_effort("entity/proj/name:latest")


def test_delete_artifact_best_effort_passes_delete_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The version carries a ``:latest`` alias, so deletion must pass ``delete_aliases=True``.

    :param monkeypatch: Stubs ``wandb.Api`` with a mock recording the delete call.
    """
    import wandb

    fake_api = MagicMock()
    monkeypatch.setattr(wandb, "Api", lambda: fake_api)

    wandb_artifacts._delete_artifact_best_effort("entity/proj/name:latest")

    fake_api.artifact.assert_called_once_with("entity/proj/name:latest")
    fake_api.artifact.return_value.delete.assert_called_once_with(delete_aliases=True)
