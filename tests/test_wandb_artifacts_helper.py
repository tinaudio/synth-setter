"""Unit tests for the citest checkpoint-artifact publisher's naming and teardown.

``publish_checkpoint_artifact`` uploads a real checkpoint to W&B for the live
``${wandb:...}`` resolver round-trip tests. These stub the ``wandb`` SDK so the
unique-naming and delete-on-exit contract — which keeps the round-trips off the
shared W&B storage budget — is pinned without a network call or a live key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    fake = mock.MagicMock()
    run = fake.init.return_value
    run.entity = "ent"
    run.id = "run123"
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _raise_inside_context() -> None:
    """Fail inside the publisher context so teardown-on-exception can be asserted.

    :raises RuntimeError: Always.
    """
    raise RuntimeError("boom")


def test_publish_checkpoint_artifact_embeds_ckpt_and_yields_suffixed_latest_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The yielded ref pins ``:latest`` on a per-call-suffixed citest name; ckpt is ``model.ckpt``.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")

    with publish_checkpoint_artifact(ckpt, "model-citest-ffn_full", tmp_path) as ref:
        assert ref.startswith("ent/synth-setter-citest/model-citest-ffn_full-")
        assert ref.endswith(":latest")
        fake.Artifact.return_value.add_file.assert_called_once_with(str(ckpt), name="model.ckpt")
        # Nothing is deleted while the context is still open.
        fake.Api.return_value.artifact.assert_not_called()

    name = fake.Artifact.call_args.kwargs["name"]
    assert name.startswith("model-citest-ffn_full-") and name != "model-citest-ffn_full"


def test_publish_checkpoint_artifact_deletes_artifact_and_run_on_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Leaving the context finishes the run and deletes both the artifact and the run.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")

    with publish_checkpoint_artifact(ckpt, "model-citest-ffn_full", tmp_path) as ref:
        artifact_name = ref.split("/")[-1].removesuffix(":latest")

    api = fake.Api.return_value
    fake.init.return_value.finish.assert_called_once()
    api.artifact.assert_called_once()
    assert api.artifact.call_args.args[0] == f"ent/synth-setter-citest/{artifact_name}:latest"
    api.artifact.return_value.delete.assert_called_once()
    api.run.assert_called_once_with("ent/synth-setter-citest/run123")
    api.run.return_value.delete.assert_called_once()


def test_publish_checkpoint_artifact_deletes_even_when_body_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing context body still triggers artifact and run deletion (cleanup is in ``finally``).

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")

    with (
        pytest.raises(RuntimeError, match="boom"),
        publish_checkpoint_artifact(ckpt, "model-citest-ffn_full", tmp_path),
    ):
        _raise_inside_context()

    fake.Api.return_value.artifact.return_value.delete.assert_called_once()
    fake.Api.return_value.run.return_value.delete.assert_called_once()


def test_publish_checkpoint_artifact_swallows_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A W&B error during teardown is swallowed so it never masks the test's own result.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    fake.Api.return_value.artifact.side_effect = RuntimeError("wandb 503")
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")

    # Exiting the context must not re-raise the artifact-delete failure.
    with publish_checkpoint_artifact(ckpt, "model-citest-ffn_full", tmp_path) as ref:
        assert ref

    # The run is still finished and the run-delete is still attempted — a failed artifact
    # delete must not short-circuit the rest of teardown.
    fake.init.return_value.finish.assert_called_once()
    fake.Api.return_value.run.return_value.delete.assert_called_once()
