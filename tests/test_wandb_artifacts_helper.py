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


def test_publish_checkpoint_artifact_retries_with_fresh_artifact_after_upload_files_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recognized upload failure retries the full publish with a fresh name after one second.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    failed_artifact = mock.MagicMock()
    failed_artifact.wait.side_effect = ValueError(
        "ArtifactSaver.uploadFiles: most remaining uploads (1/1) have failed, giving up"
    )
    successful_artifact = mock.MagicMock()
    fake.Artifact.side_effect = [failed_artifact, successful_artifact]
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")
    sleeps: list[float] = []

    with publish_checkpoint_artifact(
        ckpt, "model-citest-ffn_full", tmp_path, sleep=sleeps.append
    ) as ref:
        assert ref.endswith(":latest")

    assert fake.Artifact.call_count == 2
    first_name = fake.Artifact.call_args_list[0].kwargs["name"]
    second_name = fake.Artifact.call_args_list[1].kwargs["name"]
    assert first_name != second_name
    assert ref == f"ent/synth-setter-citest/{second_name}:latest"
    failed_artifact.add_file.assert_called_once_with(str(ckpt), name="model.ckpt")
    successful_artifact.add_file.assert_called_once_with(str(ckpt), name="model.ckpt")
    assert fake.init.return_value.log_artifact.call_args_list == [
        mock.call(failed_artifact),
        mock.call(successful_artifact),
    ]
    assert sleeps == [1]


@pytest.mark.parametrize(
    "message",
    [
        "ArtifactSaver.uploadFiles: most remaining uploads (1/1) have failed, giving up",
        (
            "ArtifactSaver.uploadManifest: file transfer: upload: failed to upload: "
            "status: 403 Forbidden"
        ),
    ],
)
def test_publish_checkpoint_artifact_retries_each_recognized_wait_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, message: str
) -> None:
    """Each recognized ``Artifact.wait`` failure prefix is retryable.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    :param message: Recognized W&B upload failure text.
    """
    fake = _install_fake_wandb(monkeypatch)
    failed_artifact = mock.MagicMock()
    failed_artifact.wait.side_effect = ValueError(message)
    fake.Artifact.side_effect = [failed_artifact, mock.MagicMock()]
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")
    sleeps: list[float] = []

    with publish_checkpoint_artifact(ckpt, "model-citest-ffn_full", tmp_path, sleep=sleeps.append):
        pass

    assert fake.Artifact.call_count == 2
    assert sleeps == [1]


def test_publish_checkpoint_artifact_three_failures_preserve_final_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Three recognized failures stop after three attempts and re-raise the final error.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    errors = [
        ValueError("ArtifactSaver.uploadFiles: most remaining uploads failure 1"),
        ValueError("ArtifactSaver.uploadFiles: most remaining uploads failure 2"),
        ValueError("ArtifactSaver.uploadFiles: most remaining uploads failure 3"),
    ]
    first_artifact = mock.MagicMock()
    first_artifact.wait.side_effect = errors[0]
    second_artifact = mock.MagicMock()
    second_artifact.wait.side_effect = errors[1]
    third_artifact = mock.MagicMock()
    third_artifact.wait.side_effect = errors[2]
    fake.Artifact.side_effect = [first_artifact, second_artifact, third_artifact]
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")
    sleeps: list[float] = []

    with pytest.raises(ValueError) as exc_info:
        with publish_checkpoint_artifact(
            ckpt, "model-citest-ffn_full", tmp_path, sleep=sleeps.append
        ):
            pass

    assert exc_info.value is errors[-1]
    assert fake.Artifact.call_count == 3
    assert sleeps == [1, 2]


def test_publish_checkpoint_artifact_unrelated_value_error_fails_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unrelated ``ValueError`` propagates immediately without sleeping.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    artifact = fake.Artifact.return_value
    unrelated_error = ValueError("download: status: 403 Forbidden")
    artifact.wait.side_effect = unrelated_error
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")
    sleeps: list[float] = []

    with pytest.raises(ValueError) as exc_info:
        with publish_checkpoint_artifact(
            ckpt, "model-citest-ffn_full", tmp_path, sleep=sleeps.append
        ):
            pass

    assert exc_info.value is unrelated_error
    fake.Artifact.assert_called_once()
    assert sleeps == []


def test_publish_checkpoint_artifact_failed_attempts_use_independent_names_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Failed publish attempts receive distinct names and each artifact is cleaned up.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    failed_artifact = mock.MagicMock()
    failed_artifact.wait.side_effect = ValueError(
        "ArtifactSaver.uploadManifest: file transfer: upload: failed to upload: "
        "status: 403 Forbidden"
    )
    fake.Artifact.side_effect = [failed_artifact, mock.MagicMock()]
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")

    with publish_checkpoint_artifact(
        ckpt, "model-citest-ffn_full", tmp_path, sleep=lambda _: None
    ):
        pass

    names = [call.kwargs["name"] for call in fake.Artifact.call_args_list]
    assert len(set(names)) == 2
    cleaned_refs = [call.args[0] for call in fake.Api.return_value.artifact.call_args_list]
    assert cleaned_refs == [
        f"ent/synth-setter-citest/{names[0]}:latest",
        f"ent/synth-setter-citest/{names[1]}:latest",
    ]
    fake.Api.return_value.run.assert_called_once_with("ent/synth-setter-citest/run123")


def test_publish_checkpoint_artifact_finish_failure_does_not_skip_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``run.finish()`` error is swallowed and still reaches the artifact and run deletion.

    :param monkeypatch: Installs the fake ``wandb`` SDK in ``sys.modules``.
    :param tmp_path: Holds the dummy checkpoint and the run dir.
    """
    fake = _install_fake_wandb(monkeypatch)
    fake.init.return_value.finish.side_effect = RuntimeError("wandb comm error")
    from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("weights")

    # A finish() failure must neither propagate out of the context nor skip cleanup.
    with publish_checkpoint_artifact(ckpt, "model-citest-ffn_full", tmp_path) as ref:
        assert ref

    fake.Api.return_value.artifact.return_value.delete.assert_called_once()
    fake.Api.return_value.run.return_value.delete.assert_called_once()
