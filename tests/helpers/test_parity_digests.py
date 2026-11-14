"""Stage attribution for the MeanAudio parity comparison."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

from tests.helpers.parity_digests import (
    DIGEST_PRELUDE,
    DIGEST_STAGES,
    describe_digest_divergence,
)

_AGREEING = {"checkpoint_file": "aaa", "encoder_weights": "bbb", "mel": "ccc"}


def test_describe_digest_divergence_differing_weights_names_the_weight_stage() -> None:
    """A weight-only difference is reported as the diverging stage, with both digests."""
    summary = describe_digest_divergence(_AGREEING, {**_AGREEING, "encoder_weights": "zzz"})

    assert "encoder_weights" in summary
    assert "bbb" in summary
    assert "zzz" in summary


def test_describe_digest_divergence_differing_weights_excludes_agreeing_stages() -> None:
    """Stages that agree are named as agreeing, so attribution is unambiguous."""
    summary = describe_digest_divergence(_AGREEING, {**_AGREEING, "encoder_weights": "zzz"})

    assert "ccc" not in summary
    assert "mel" in summary


def test_describe_digest_divergence_identical_digests_reports_a_downstream_divergence() -> None:
    """Latents that differ while every recorded stage agrees point past the last digest."""
    summary = describe_digest_divergence(_AGREEING, dict(_AGREEING))

    assert "agree" in summary
    for stage in DIGEST_STAGES:
        assert stage in summary


def test_describe_digest_divergence_missing_stage_is_reported_not_skipped() -> None:
    """A path that failed to record a stage is named, rather than read as agreement."""
    summary = describe_digest_divergence(_AGREEING, {"checkpoint_file": "aaa"})

    assert "missing" in summary
    assert "encoder_weights" in summary


def test_describe_digest_divergence_rejects_an_empty_recording() -> None:
    """Comparing nothing is a programming error, not a silent pass."""
    with pytest.raises(ValueError, match="no digests"):
        describe_digest_divergence({}, {})


def _run_prelude(
    checkpoint: Path, module: torch.nn.Module, mel: torch.Tensor, out: Path
) -> dict[str, str]:
    """Execute the injected prelude the probe programs run and return what it recorded.

    :param checkpoint: File standing in for the VAE state the path opened.
    :param module: Model standing in for the loaded encoder.
    :param mel: Tensor standing in for the mel handed to the encoder.
    :param out: Destination the prelude writes its digests to.
    :returns: The recorded digest mapping.
    """
    namespace: dict[str, object] = {"torch": torch}
    exec(DIGEST_PRELUDE, namespace)  # noqa: S102 - the prelude under test is repo-owned source
    record = cast(
        "Callable[[str, str, torch.nn.Module, torch.Tensor], None]", namespace["record_digests"]
    )
    record(str(out), str(checkpoint), module, mel)
    return cast("dict[str, str]", json.loads(out.read_text()))


def test_digest_prelude_perturbed_weights_attribute_the_failure_to_the_encoder(
    tmp_path: Path,
) -> None:
    """A path whose weights changed is named by the weight stage, not by the checkpoint or mel.

    :param tmp_path: Location of the stand-in checkpoint and digest files.
    """
    checkpoint = tmp_path / "v1-16.pth"
    checkpoint.write_bytes(b"pinned-state")
    mel = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    encoder = torch.nn.Linear(3, 2)
    direct = _run_prelude(checkpoint, encoder, mel, tmp_path / "direct.json")
    with torch.no_grad():
        encoder.weight[0, 0] += 1.0
    adapter = _run_prelude(checkpoint, encoder, mel, tmp_path / "adapter.json")

    summary = describe_digest_divergence(direct, adapter)

    assert summary.startswith("encoder_weights:")
    assert "agreeing stages: checkpoint_file, mel" in summary


def test_digest_prelude_identical_inputs_record_identical_digests(tmp_path: Path) -> None:
    """Two paths in the same state digest every stage the same, so parity reads as agreement.

    :param tmp_path: Location of the stand-in checkpoint and digest files.
    """
    checkpoint = tmp_path / "v1-16.pth"
    checkpoint.write_bytes(b"pinned-state")
    mel = torch.ones(1, 2, 3)
    encoder = torch.nn.Linear(3, 2)

    direct = _run_prelude(checkpoint, encoder, mel, tmp_path / "direct.json")
    adapter = _run_prelude(checkpoint, encoder, mel, tmp_path / "adapter.json")

    assert direct == adapter
    assert set(direct) == set(DIGEST_STAGES)
