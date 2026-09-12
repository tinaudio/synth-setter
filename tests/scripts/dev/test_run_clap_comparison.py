"""Behavior tests for paired CLAP comparison reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.dev.run_clap_comparison import (
    CANDIDATE_SOURCE_CLAP,
    CANDIDATE_SOURCE_MEANAUDIO_S_FULL,
    CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED,
    build_candidate_identity,
    build_paired_row,
    ensure_resume_identity,
    write_aggregate_comparison,
)
from synth_setter.pipeline.data.meanaudio_generation import (
    MEANAUDIO_DURATION_SECONDS,
    MEANAUDIO_S_FULL_CHECKPOINT_NAME,
    MEANAUDIO_S_FULL_CHECKPOINT_SHA256,
    MEANAUDIO_STEPS,
)


def test_build_paired_row_candidate_lower_distance_marks_candidate_win() -> None:
    """A lower candidate cosine distance wins the paired prompt."""
    baseline = {
        "prompt": "frog croak",
        "cosine_similarity": "0.2",
        "cosine_distance": "0.8",
    }
    candidate = {
        "prompt": "frog croak",
        "cosine_similarity": "0.3",
        "cosine_distance": "0.7",
    }

    row = build_paired_row(1, baseline, candidate)

    assert row == {
        "index": 1,
        "prompt": "frog croak",
        "baseline_cosine_similarity": 0.2,
        "baseline_cosine_distance": 0.8,
        "candidate_cosine_similarity": 0.3,
        "candidate_cosine_distance": 0.7,
        "distance_delta_candidate_minus_baseline": -0.1,
        "winner": "candidate",
    }


def test_build_candidate_identity_meanaudio_records_every_reproducibility_input() -> None:
    """MeanAudio cache identity includes generation, weight, and inverse identities."""
    identity = build_candidate_identity(
        CANDIDATE_SOURCE_MEANAUDIO_S_FULL,
        "r2://checkpoints/candidate/last.ckpt",
        steps=17,
        duration_seconds=MEANAUDIO_DURATION_SECONDS,
        seed=23,
    )

    assert identity["candidate_source"] == CANDIDATE_SOURCE_MEANAUDIO_S_FULL
    assert identity["candidate_checkpoint"] == "r2://checkpoints/candidate/last.ckpt"
    assert identity["candidate_checkpoint_identity"] == (
        "uri:r2://checkpoints/candidate/last.ckpt"
    )
    assert identity["meanaudio_steps"] == 17
    assert identity["meanaudio_duration_seconds"] == MEANAUDIO_DURATION_SECONDS
    assert identity["seed"] == 23
    assert identity["meanaudio_model_checkpoint_name"] == MEANAUDIO_S_FULL_CHECKPOINT_NAME
    assert identity["meanaudio_model_checkpoint_sha256"] == MEANAUDIO_S_FULL_CHECKPOINT_SHA256
    assert "meanaudio_upstream_revision" in identity
    assert "meanaudio_checkpoint_revision" in identity


def test_build_candidate_identity_reencoded_meanaudio_records_projection() -> None:
    """The aligned arm cannot reuse direct-latent candidate artifacts."""
    identity = build_candidate_identity(
        CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED,
        "r2://checkpoints/candidate/last.ckpt",
        steps=25,
        duration_seconds=MEANAUDIO_DURATION_SECONDS,
        seed=0,
    )

    assert identity["candidate_source"] == CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED
    assert identity["meanaudio_projection"] == "vae-decode-vocode-encode-mode"
    assert "meanaudio_vae_checkpoint_sha256" in identity
    assert "meanaudio_vocoder_checkpoint_sha256" in identity


def test_build_candidate_identity_clap_preserves_checkpoint_and_seed_contract() -> None:
    """The existing CLAP backend keeps its checkpoint and flow seed identity."""
    identity = build_candidate_identity(
        CANDIDATE_SOURCE_CLAP,
        "r2://checkpoints/clap/last.ckpt",
        steps=MEANAUDIO_STEPS,
        duration_seconds=MEANAUDIO_DURATION_SECONDS,
        seed=5,
    )

    assert identity == {
        "candidate_source": CANDIDATE_SOURCE_CLAP,
        "candidate_checkpoint": "r2://checkpoints/clap/last.ckpt",
        "candidate_checkpoint_identity": "uri:r2://checkpoints/clap/last.ckpt",
        "clap_cfg_strength": "checkpoint-default",
        "clap_sample_steps": "checkpoint-default",
        "seed": 5,
    }


def test_build_candidate_identity_clap_records_inference_overrides() -> None:
    """CLAP cache identity separates solver and guidance experiments."""
    identity = build_candidate_identity(
        CANDIDATE_SOURCE_CLAP,
        "r2://checkpoints/clap/last.ckpt",
        steps=MEANAUDIO_STEPS,
        duration_seconds=MEANAUDIO_DURATION_SECONDS,
        seed=0,
        clap_sample_steps=200,
        clap_cfg_strength=8.0,
    )

    assert identity["clap_sample_steps"] == 200
    assert identity["clap_cfg_strength"] == 8.0


def test_ensure_resume_identity_changed_steps_rejects_cached_candidates(tmp_path: Path) -> None:
    """A generation-setting change cannot reuse candidate artifacts.

    :param tmp_path: Isolated resume-manifest location.
    """
    path = tmp_path / "candidate-identity.json"
    original = build_candidate_identity(
        CANDIDATE_SOURCE_MEANAUDIO_S_FULL,
        "r2://checkpoints/candidate/last.ckpt",
        steps=25,
        duration_seconds=MEANAUDIO_DURATION_SECONDS,
        seed=0,
    )
    changed = original | {"meanaudio_steps": 24}
    ensure_resume_identity(path, original)

    with pytest.raises(ValueError, match="does not match"):
        ensure_resume_identity(path, changed)

    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_ensure_resume_identity_unidentified_candidate_cache_rejects_reuse(tmp_path: Path) -> None:
    """Candidate artifacts cannot be grandfathered into an unknown identity.

    :param tmp_path: Isolated unidentified candidate cache.
    """
    candidate = tmp_path / "audio" / "sample_001_frog" / "candidate.wav"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"old candidate")

    with pytest.raises(ValueError, match="has no identity"):
        ensure_resume_identity(
            tmp_path / "candidate-identity.json",
            {
                "candidate_source": CANDIDATE_SOURCE_CLAP,
                "candidate_checkpoint": "r2://checkpoints/clap/last.ckpt",
            },
        )


def test_write_aggregate_comparison_two_arms_writes_statistics_and_wins(
    tmp_path: Path,
) -> None:
    """Aggregate output reports each arm's distances and paired wins.

    :param tmp_path: Isolates the generated aggregate CSV.
    """
    rows = [
        {
            "baseline_cosine_distance": 0.8,
            "candidate_cosine_distance": 0.7,
            "winner": "candidate",
        },
        {
            "baseline_cosine_distance": 0.6,
            "candidate_cosine_distance": 0.65,
            "winner": "baseline",
        },
    ]
    destination = tmp_path / "aggregate.csv"

    write_aggregate_comparison(destination, rows)

    with destination.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert written[0]["arm"] == "baseline"
    assert written[0]["mean"] == "0.7"
    assert written[0]["wins"] == "1"
    assert written[1]["arm"] == "candidate"
    assert written[1]["mean"] == "0.675"
    assert written[1]["wins"] == "1"
