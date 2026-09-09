"""Behavior tests for dataset oracle-probe archival."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pytest

from synth_setter.evaluation.oracle_probe import (
    OracleProbeProvenance,
    new_oracle_probe_launch_id,
    upload_oracle_probe,
)
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig, Split
from synth_setter.utils.logging_utils import resolve_git_sha


def _write_eval_artifacts(eval_dir: Path, *, metric: float = 1.0) -> None:
    """Create the uploadable eval outputs plus excluded and unsafe files.

    :param eval_dir: Root directory for the synthetic eval run.
    :param metric: Distinguishable audio metric persisted by the synthetic run.
    """
    for relative_path, contents in (
        (".hydra/config.yaml", "task_name: eval\n"),
        ("audio/sample_0/pred.wav", "audio"),
        ("metrics/metrics.json", f'{{"audio/rms_mean": {metric}}}\n'),
        ("predictions/pred-0.pt", "tensor"),
        ("eval.log", "raw process output"),
        ("wandb/debug.log", "credential-adjacent client log"),
    ):
        path = eval_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def _provenance(
    spec: DatasetSpec,
    split: Split = "test",
    *,
    candidate_render: RenderConfig | None = None,
) -> OracleProbeProvenance:
    """Build provenance from one generated dataset and an oracle render.

    :param spec: Source dataset specification.
    :param split: Source split represented by the probe.
    :param candidate_render: Evaluated render, or the source render when omitted.
    :returns: Validated provenance for the probe upload.
    """
    return OracleProbeProvenance(
        source_dataset_uri=spec.r2.split_lance_uri(split),
        source_dataset_task=spec.task_name,
        source_split=split,
        source_run_id=spec.run_id,
        source_render=spec.render,
        candidate_render=candidate_render or spec.render,
    )


def test_upload_oracle_probe_materializes_only_probe_artifacts(
    tmp_path: Path,
    fake_r2_remote: Path,
    valid_dataset_spec_kwargs: dict[str, object],
) -> None:
    """Real rclone upload keeps config, audio, metrics, and strict provenance only.

    :param tmp_path: Temporary eval-run root.
    :param fake_r2_remote: Local rclone remote root.
    :param valid_dataset_spec_kwargs: Valid source dataset fields.
    """
    spec = DatasetSpec.model_validate(valid_dataset_spec_kwargs)
    eval_dir = tmp_path / "eval"
    _write_eval_artifacts(eval_dir)

    uri = upload_oracle_probe(
        eval_dir,
        r2=spec.r2,
        launch_id="launch-1",
        provenance=_provenance(spec),
    )

    assert uri == (
        f"r2://{spec.r2.bucket}/probes/dataset-oracle/{spec.task_name}/{spec.run_id}/launch-1/test"
    )
    landed = (
        fake_r2_remote
        / spec.r2.bucket
        / "probes"
        / "dataset-oracle"
        / spec.task_name
        / spec.run_id
        / "launch-1"
        / "test"
    )
    uploaded = sorted(
        path.relative_to(landed).as_posix() for path in landed.rglob("*") if path.is_file()
    )
    assert uploaded == [
        ".hydra/config.yaml",
        "audio/sample_0/pred.wav",
        "metrics/metrics.json",
        "provenance.json",
    ]
    payload = json.loads((landed / "provenance.json").read_text())
    assert OracleProbeProvenance.model_validate(payload) == _provenance(spec)
    assert payload["evaluation_git_sha"] == resolve_git_sha()
    assert payload["source_render"] == payload["candidate_render"]


def test_upload_oracle_probe_roles_preserve_both_renderer_results(
    tmp_path: Path,
    fake_r2_remote: Path,
    valid_dataset_spec_kwargs: dict[str, object],
) -> None:
    """Source and candidate archives retain separate metrics and render provenance.

    :param tmp_path: Temporary eval-run root.
    :param fake_r2_remote: Local rclone remote root.
    :param valid_dataset_spec_kwargs: Valid source dataset fields.
    """
    spec = DatasetSpec.model_validate(valid_dataset_spec_kwargs)
    source_eval_dir = tmp_path / "source"
    candidate_eval_dir = tmp_path / "candidate"
    _write_eval_artifacts(source_eval_dir, metric=1.0)
    _write_eval_artifacts(candidate_eval_dir, metric=2.0)
    candidate_render = spec.render.model_copy(update={"renderer_backend": "pedalboard"})

    source_uri = upload_oracle_probe(
        source_eval_dir,
        r2=spec.r2,
        launch_id="launch-roles",
        role="source",
        provenance=_provenance(spec),
    )
    candidate_uri = upload_oracle_probe(
        candidate_eval_dir,
        r2=spec.r2,
        launch_id="launch-roles",
        role="candidate",
        provenance=_provenance(spec, candidate_render=candidate_render),
    )

    assert source_uri.endswith("/launch-roles/source/test")
    assert candidate_uri.endswith("/launch-roles/candidate/test")
    probe_root = (
        fake_r2_remote
        / spec.r2.bucket
        / "probes"
        / "dataset-oracle"
        / spec.task_name
        / spec.run_id
        / "launch-roles"
    )
    assert json.loads((probe_root / "source/test/metrics/metrics.json").read_text()) == {
        "audio/rms_mean": 1.0
    }
    assert json.loads((probe_root / "candidate/test/metrics/metrics.json").read_text()) == {
        "audio/rms_mean": 2.0
    }
    candidate_payload = json.loads((probe_root / "candidate/test/provenance.json").read_text())
    candidate_provenance = OracleProbeProvenance.model_validate(candidate_payload)
    assert candidate_provenance.candidate_render == candidate_render
    assert candidate_provenance.source_render == spec.render


def test_upload_oracle_probe_payload_failure_leaves_no_provenance_record(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_dataset_spec_kwargs: dict[str, object],
) -> None:
    """A partial payload transfer cannot publish the provenance commit record.

    :param tmp_path: Temporary eval-run root.
    :param fake_r2_remote: Local rclone remote root.
    :param monkeypatch: Patches only the payload-failure seam.
    :param valid_dataset_spec_kwargs: Valid source dataset fields.
    """
    spec = DatasetSpec.model_validate(valid_dataset_spec_kwargs)
    eval_dir = tmp_path / "eval"
    _write_eval_artifacts(eval_dir)
    landed = (
        fake_r2_remote
        / spec.r2.bucket
        / "probes"
        / "dataset-oracle"
        / spec.task_name
        / spec.run_id
        / "launch-failed"
        / "test"
    )

    real_upload_dir = r2_io.upload_dir

    def upload_payload_then_report_failure(
        local_dir: Path, destination: str, *, exclude: str | None = None
    ) -> None:
        real_upload_dir(local_dir, destination, exclude=exclude)
        raise subprocess.CalledProcessError(returncode=1, cmd="rclone copy")

    monkeypatch.setattr(r2_io, "upload_dir", upload_payload_then_report_failure)

    with pytest.raises(subprocess.CalledProcessError):
        upload_oracle_probe(
            eval_dir,
            r2=spec.r2,
            launch_id="launch-failed",
            provenance=_provenance(spec),
        )

    assert not (landed / "provenance.json").exists()


def test_oracle_probe_provenance_non_r2_source_rejected(
    valid_dataset_spec_kwargs: dict[str, object],
) -> None:
    """Reject source locations outside the finalized R2 dataset namespace.

    :param valid_dataset_spec_kwargs: Valid source dataset fields.
    """
    spec = DatasetSpec.model_validate(valid_dataset_spec_kwargs)
    payload = _provenance(spec).model_dump()
    payload["source_dataset_uri"] = "https://example.com/test.lance"

    with pytest.raises(ValueError, match="source_dataset_uri must be an r2:// URI"):
        OracleProbeProvenance.model_validate(payload)


def test_new_oracle_probe_launch_id_returns_unique_names() -> None:
    """Independent inline invocations receive distinct destination components."""
    assert new_oracle_probe_launch_id() != new_oracle_probe_launch_id()


@pytest.mark.integration_r2
@pytest.mark.r2
def test_upload_oracle_probe_real_r2_round_trip(
    tmp_path: Path,
    valid_dataset_spec_kwargs: dict[str, object],
) -> None:
    """A probe uploaded to real R2 can be downloaded with its provenance intact.

    :param tmp_path: Temporary eval and download root.
    :param valid_dataset_spec_kwargs: Valid source dataset fields.
    """
    nonce = uuid.uuid4().hex
    valid_dataset_spec_kwargs["task_name"] = f"oracle-probe-test-{nonce}"
    valid_dataset_spec_kwargs["run_id"] = f"run-{nonce}"
    spec = DatasetSpec.model_validate(valid_dataset_spec_kwargs)
    eval_dir = tmp_path / "eval"
    _write_eval_artifacts(eval_dir)
    launch_id = new_oracle_probe_launch_id()
    probe_prefix = f"probes/dataset-oracle/{spec.task_name}/{spec.run_id}/{launch_id}/"

    try:
        uri = upload_oracle_probe(
            eval_dir,
            r2=spec.r2,
            launch_id=launch_id,
            provenance=_provenance(spec),
        )
        downloaded = tmp_path / "downloaded"
        r2_io.download_dir_no_overwrite(uri, downloaded)

        payload = json.loads((downloaded / "provenance.json").read_text())
        assert OracleProbeProvenance.model_validate(payload) == _provenance(spec)
        assert (downloaded / ".hydra" / "config.yaml").is_file()
        assert (downloaded / "audio" / "sample_0" / "pred.wav").is_file()
        assert (downloaded / "metrics" / "metrics.json").is_file()
        assert not (downloaded / "predictions").exists()
    finally:
        r2_io.purge_prefix(spec.r2.bucket, probe_prefix)
