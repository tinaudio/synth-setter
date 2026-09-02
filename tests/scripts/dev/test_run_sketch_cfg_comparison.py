"""Behavior tests for the sketch CFG suite runner."""

from pathlib import Path

import pytest

from scripts.dev import run_sketch_cfg_comparison as suite
from scripts.dev.run_sketch_cfg_comparison import aggregate_metrics, select_vocal_rows


def test_select_vocal_rows_mixed_corpus_returns_only_included_imitations() -> None:
    """Freesound originals and excluded imitations cannot enter the sketch suite."""
    rows = [
        {"row_type": "freesound_original", "included": None, "audio_decode_status": "decoded"},
        {"row_type": "imitation", "included": False, "audio_decode_status": "decoded"},
        {"row_type": "imitation", "included": True, "audio_decode_status": "decoded"},
        {"row_type": "imitation", "included": True, "audio_decode_status": "decoded"},
    ]

    assert select_vocal_rows(rows, 2) == ((2, rows[2]), (3, rows[3]))


def test_select_vocal_rows_failed_included_imitation_raises() -> None:
    """A decode failure cannot silently substitute a later curated imitation."""
    rows = [
        {"row_type": "imitation", "included": True, "audio_decode_status": "failed"},
        {"row_type": "imitation", "included": True, "audio_decode_status": "decoded"},
    ]

    with pytest.raises(ValueError, match="row 0"):
        select_vocal_rows(rows, 1)


def test_require_fresh_run_nonempty_r2_destination_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a remote prefix to prevent relabeling existing artifacts.

    :param tmp_path: Supplies the required empty local side of the run contract.
    :param monkeypatch: Simulates an occupied remote prefix without mutating R2.
    """
    monkeypatch.setattr(suite.r2_io, "r2_directory_exists", lambda *args: True)

    with pytest.raises(FileExistsError, match="R2 destination"):
        suite._require_fresh_run(tmp_path, "r2://bucket/existing")


def test_require_fresh_run_nonempty_local_output_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing local artifact cannot be deleted or reused implicitly.

    :param tmp_path: Nonempty local suite root.
    :param monkeypatch: R2 probe guard fixture.
    """
    (tmp_path / "diagnostic.log").write_text("trace", encoding="utf-8")
    monkeypatch.setattr(
        suite.r2_io,
        "r2_directory_exists",
        lambda *args: pytest.fail("local collision must fail first"),
    )

    with pytest.raises(FileExistsError, match="output directory"):
        suite._require_fresh_run(tmp_path, "r2://bucket/new")


def test_render_pair_fresh_output_explicitly_enables_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publishing suite opts into the single-pair CLI's R2 side effect.

    :param tmp_path: Temporary suite output root.
    :param monkeypatch: Subprocess command patch fixture.
    """
    command: list[str] = []

    class RecordingCommand:
        def __init__(self, executable: str) -> None:
            command.append(executable)

        def __call__(self, *args: str, **kwargs: object) -> None:
            del kwargs
            command.extend(args)

    monkeypatch.setattr(suite, "Command", RecordingCommand)

    suite._render_pair(
        {"pair_index": 0, "sketch_path": "sketch.wav", "content_path": "content.wav"},
        output_dir=tmp_path,
        destination="r2://bucket/run",
        checkpoint="r2://bucket/model.ckpt",
        checkpoint_sha256="a" * 64,
        stats="r2://bucket/stats.npz",
        stats_sha256="b" * 64,
        content_cfg=(0.0,),
        sketch_cfg=(0.0,),
        sample_steps=2,
        seed=0,
        device="cpu",
    )

    assert command[command.index("--checkpoint") + 1] == "r2://bucket/model.ckpt"
    assert command[command.index("--checkpoint-sha256") + 1] == "a" * 64
    assert command[command.index("--stats") + 1] == "r2://bucket/stats.npz"
    assert command[command.index("--stats-sha256") + 1] == "b" * 64
    assert "--upload" in command


def test_render_pair_existing_output_fails_without_deleting_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun cannot recursively delete an interrupted pair.

    :param tmp_path: Temporary suite output root.
    :param monkeypatch: Subprocess command patch fixture.
    """
    diagnostic = tmp_path / "audio" / "sample_000" / "failure.log"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text("trace", encoding="utf-8")
    monkeypatch.setattr(
        suite,
        "Command",
        lambda *args: pytest.fail("renderer should not run"),
    )

    with pytest.raises(FileExistsError, match="fresh output"):
        suite._render_pair(
            {"pair_index": 0, "sketch_path": "sketch.wav", "content_path": "content.wav"},
            output_dir=tmp_path,
            destination="r2://bucket/run",
            checkpoint="r2://bucket/model.ckpt",
            checkpoint_sha256="a" * 64,
            stats="r2://bucket/stats.npz",
            stats_sha256="b" * 64,
            content_cfg=(0.0,),
            sketch_cfg=(0.0,),
            sample_steps=2,
            seed=0,
            device="cpu",
        )

    assert diagnostic.read_text(encoding="utf-8") == "trace"


def test_aggregate_metrics_nonfinite_value_raises() -> None:
    """A non-finite per-pair metric cannot publish aggregate completion."""
    rows = [{"arm": "cfg-c0-s0", "mss": float("nan"), "wmfcc": 2.0, "sot": 3.0, "rms": 0.2}]

    with pytest.raises(ValueError, match="finite.*mss"):
        aggregate_metrics(rows)


def test_aggregate_metrics_multiple_arms_reports_per_metric_mean() -> None:
    """Each CFG arm receives independent aggregate audio statistics."""
    rows = [
        {"arm": "cfg-c0-s0", "mss": 1.0, "wmfcc": 2.0, "sot": 3.0, "rms": 0.2},
        {"arm": "cfg-c0-s0", "mss": 3.0, "wmfcc": 4.0, "sot": 5.0, "rms": 0.4},
        {"arm": "cfg-c2-s2", "mss": 8.0, "wmfcc": 7.0, "sot": 6.0, "rms": 0.9},
    ]

    aggregates = aggregate_metrics(rows)

    assert aggregates[0]["arm"] == "cfg-c0-s0"
    assert aggregates[0]["mss_mean"] == 2.0
    assert aggregates[0]["mss_std_population"] == 1.0
    assert aggregates[0]["rms_mean"] == pytest.approx(0.3)
    assert aggregates[1]["arm"] == "cfg-c2-s2"
    assert aggregates[1]["mss_mean"] == 8.0
