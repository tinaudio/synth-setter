"""Behavior tests for the bounded NSynth-to-Lance importer."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from synth_setter.pipeline.data.nsynth_import import (
    OFFICIAL_EXPECTED_COUNTS,
    build_local_import,
    download_and_verify_nsynth,
    ingest_nsynth,
    verify_local_import,
)
from tests.helpers.nsynth_fixtures import (
    SPLITS,
    example,
    wav_bytes,
    write_tiny_source,
)

_TINY_COUNTS = {"train": 1, "valid": 1, "test": 1}


def test_official_expected_counts_match_published_splits() -> None:
    """Default validation pins the published NSynth split cardinalities."""
    assert OFFICIAL_EXPECTED_COUNTS == {
        "train": 289205,
        "valid": 12678,
        "test": 4096,
    }


def _rewrite_train_row(output_root: Path, overrides: dict[str, pa.Array]) -> None:
    """Replace the train dataset with one deliberately altered real Lance row.

    :param output_root: Completed tiny import root.
    :param overrides: Replacement Arrow arrays keyed by column name.
    """
    dataset_path = output_root / "train.lance"
    dataset = lance.dataset(dataset_path)
    projected_names = [name for name in dataset.schema.names if name != "audio"]
    metadata = dataset.scanner(columns=projected_names).to_table()
    columns = [
        overrides[field.name]
        if field.name in overrides
        else lance.blob_array([wav_bytes()])
        if field.name == "audio"
        else metadata[field.name].combine_chunks()
        for field in dataset.schema
    ]
    corrupted_path = output_root / "corrupted-train.lance"
    lance.write_dataset(
        pa.record_batch(columns, schema=dataset.schema),
        corrupted_path,
        schema=dataset.schema,
        data_storage_version="2.2",
    )
    shutil.rmtree(dataset_path)
    corrupted_path.rename(dataset_path)


def test_build_local_import_valid_source_preserves_rows_blobs_and_sidecars(
    tmp_path: Path,
) -> None:
    """A complete tiny source becomes three typed Blob-v2 Lance datasets.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    notes = write_tiny_source(source_root)
    output_root = tmp_path / "output"

    manifest = build_local_import(
        source_root,
        output_root,
        expected_counts=_TINY_COUNTS,
        batch_size=1,
        remote_root="r2://experiments/third_party/NSynth",
    )

    assert manifest.total_count == 3
    assert (output_root / "manifest.json").is_file()
    for split in SPLITS:
        source_json = source_root / f"nsynth-{split}" / "examples.json"
        sidecar = output_root / "metadata" / f"nsynth-{split}.examples.json"
        assert sidecar.read_bytes() == source_json.read_bytes()

        dataset = lance.dataset(output_root / f"{split}.lance")
        assert dataset.count_rows() == 1
        table = dataset.scanner(columns=["note_str", "wav_size", "wav_sha256"]).to_table()
        assert table["note_str"][0].as_py() == notes[split]
        assert table["wav_size"][0].as_py() == len(wav_bytes())
        assert table["wav_sha256"][0].as_py() == hashlib.sha256(wav_bytes()).hexdigest()
        assert dataset.take_blobs("audio", indices=[0])[0].read() == wav_bytes()
        assert dataset.schema.field("audio").type.extension_name == "lance.blob.v2"
        assert set(dataset.schema.metadata) == {
            b"synth_setter.nsynth.license",
            b"synth_setter.nsynth.provenance",
            b"synth_setter.nsynth.source",
        }
        assert dataset.data_storage_version == "2.2"
        assert example(notes[split])["qualities"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_build_local_import_top_level_key_mismatch_refuses_output(tmp_path: Path) -> None:
    """A top-level key differing from its record's ``note_str`` is rejected.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    notes = write_tiny_source(source_root)
    examples_path = source_root / "nsynth-train" / "examples.json"
    examples_path.write_text(
        json.dumps({"different-key": example(notes["train"])}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not equal note_str"):
        build_local_import(source_root, tmp_path / "output", expected_counts=_TINY_COUNTS)


def test_build_local_import_nonbinary_qualities_refuses_output(tmp_path: Path) -> None:
    """The qualities vector must contain exactly ten binary integers.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    notes = write_tiny_source(source_root)
    malformed = example(notes["train"])
    malformed["qualities"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    examples_path = source_root / "nsynth-train" / "examples.json"
    examples_path.write_text(json.dumps({notes["train"]: malformed}), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 10 binary"):
        build_local_import(source_root, tmp_path / "output", expected_counts=_TINY_COUNTS)


def test_build_local_import_unsafe_note_str_refuses_output(tmp_path: Path) -> None:
    """A path-like note identifier cannot escape the audio root.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    unsafe = "../escape"
    examples_path = source_root / "nsynth-train" / "examples.json"
    examples_path.write_text(json.dumps({unsafe: example(unsafe)}), encoding="utf-8")

    with pytest.raises(ValueError, match="only letters, digits"):
        build_local_import(source_root, tmp_path / "output", expected_counts=_TINY_COUNTS)


def test_build_local_import_nested_orphan_wav_refuses_output(tmp_path: Path) -> None:
    """A WAV outside ``audio/<note_str>.wav`` prevents atomic publication.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    nested = source_root / "nsynth-train" / "audio" / "nested"
    nested.mkdir()
    (nested / "orphan.wav").write_bytes(wav_bytes())
    output_root = tmp_path / "output"

    with pytest.raises(ValueError, match="orphan WAV"):
        build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)

    assert not output_root.exists()
    assert not list(tmp_path.glob(".output.partial-*"))


def test_build_local_import_duplicate_metadata_key_refuses_orphan_wav(
    tmp_path: Path,
) -> None:
    """Duplicate JSON keys cannot hide an otherwise orphaned WAV.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    notes = write_tiny_source(source_root)
    train_root = source_root / "nsynth-train"
    record_json = json.dumps(example(notes["train"]))
    (train_root / "examples.json").write_text(
        f'{{"{notes["train"]}":{record_json},"{notes["train"]}":{record_json}}}',
        encoding="utf-8",
    )
    (train_root / "audio" / "orphan.wav").write_bytes(wav_bytes())

    with pytest.raises(ValueError, match="duplicate metadata key"):
        build_local_import(
            source_root,
            tmp_path / "output",
            expected_counts={"train": 2, "valid": 1, "test": 1},
        )


def test_build_local_import_missing_wav_refuses_output(tmp_path: Path) -> None:
    """A metadata row without its named WAV fails before publication.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    notes = write_tiny_source(source_root)
    (source_root / "nsynth-valid" / "audio" / f"{notes['valid']}.wav").unlink()
    output_root = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="missing WAV"):
        build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)

    assert not output_root.exists()


def test_build_local_import_extra_metadata_field_refuses_output(tmp_path: Path) -> None:
    """External records containing non-official fields fail strict validation.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    note_str = "bass_synthetic_007-033-100-train"
    malformed = example(note_str)
    malformed["unexpected"] = "value"
    examples_path = source_root / "nsynth-train" / "examples.json"
    examples_path.write_text(json.dumps({note_str: malformed}), encoding="utf-8")

    with pytest.raises(ValueError, match="extra_forbidden"):
        build_local_import(source_root, tmp_path / "output", expected_counts=_TINY_COUNTS)


def test_build_local_import_wrong_expected_count_refuses_output(tmp_path: Path) -> None:
    """A split count differing from the explicit contract fails closed.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)

    with pytest.raises(ValueError, match="train expected 2"):
        build_local_import(
            source_root,
            tmp_path / "output",
            expected_counts={"train": 2, "valid": 1, "test": 1},
        )


def test_build_local_import_existing_output_refuses_overwrite(tmp_path: Path) -> None:
    """An existing output root is untouched.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    output_root = tmp_path / "output"
    output_root.mkdir()
    sentinel = output_root / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_ingest_nsynth_remote_conflict_does_not_publish_manifest(
    fake_r2_remote: Path,
    tmp_path: Path,
) -> None:
    """An immutable artifact conflict fails before the remote completion marker.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    remote_root = fake_r2_remote / "experiments" / "third_party" / "NSynth"
    collision = remote_root / "metadata" / "nsynth-train.examples.json"
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"conflict")

    with pytest.raises(subprocess.CalledProcessError):
        ingest_nsynth(
            source_root,
            tmp_path / "output",
            expected_counts=_TINY_COUNTS,
            batch_size=1,
        )

    assert not (remote_root / "manifest.json").exists()
    assert collision.read_bytes() == b"conflict"


def test_download_and_verify_nsynth_manifest_remote_root_mismatch_raises(
    fake_r2_remote: Path,
    tmp_path: Path,
) -> None:
    """The downloaded manifest must identify the prefix that supplied it.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    ingest_nsynth(
        source_root,
        tmp_path / "output",
        expected_counts=_TINY_COUNTS,
        batch_size=1,
    )
    manifest_path = fake_r2_remote / "experiments" / "third_party" / "NSynth" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["remote_root"] = "r2://other/prefix"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest remote_root mismatch"):
        download_and_verify_nsynth(
            source_root,
            tmp_path / "download",
            expected_counts=_TINY_COUNTS,
            batch_size=1,
        )


def test_verify_local_import_valid_round_trip_reports_zero_mismatches(
    tmp_path: Path,
) -> None:
    """Verification streams every source row and WAV through its Lance consumer.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    output_root = tmp_path / "output"
    build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS, batch_size=1)

    summary = verify_local_import(
        source_root,
        output_root,
        expected_counts=_TINY_COUNTS,
        batch_size=1,
    )

    assert summary.total_count == 3
    assert summary.split_counts == _TINY_COUNTS
    assert summary.mismatches == 0


def test_verify_local_import_corrupted_lance_metadata_raises(tmp_path: Path) -> None:
    """A typed Lance metadata value differing from source is reported.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    output_root = tmp_path / "output"
    build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)
    _rewrite_train_row(output_root, {"pitch": pa.array([34], type=pa.int64())})

    with pytest.raises(ValueError, match="metadata mismatch for pitch"):
        verify_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)


def test_verify_local_import_corrupted_lance_blob_raises(tmp_path: Path) -> None:
    """A Blob-v2 payload differing from source is reported byte-for-byte.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    output_root = tmp_path / "output"
    build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)
    corrupted = bytearray(wav_bytes())
    corrupted[-1] ^= 1
    _rewrite_train_row(output_root, {"audio": lance.blob_array([bytes(corrupted)])})

    with pytest.raises(ValueError, match="WAV blob bytes mismatch"):
        verify_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)


def test_verify_local_import_corrupted_source_wav_raises(tmp_path: Path) -> None:
    """A source WAV byte differing from the imported Blob-v2 payload is reported.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    notes = write_tiny_source(source_root)
    output_root = tmp_path / "output"
    build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)
    wav_path = source_root / "nsynth-test" / "audio" / f"{notes['test']}.wav"
    corrupted = bytearray(wav_path.read_bytes())
    corrupted[-1] ^= 1
    wav_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="WAV SHA-256 mismatch"):
        verify_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)


def test_verify_local_import_corrupted_sidecar_raises(tmp_path: Path) -> None:
    """A downloaded examples sidecar differing by one byte is reported.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    output_root = tmp_path / "output"
    build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)
    sidecar = output_root / "metadata" / "nsynth-valid.examples.json"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")

    with pytest.raises(ValueError, match="sidecar bytes differ"):
        verify_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)


def test_verify_local_import_manifest_extra_field_raises(tmp_path: Path) -> None:
    """A manifest containing an unrecognized field fails strict parsing.

    :param tmp_path: Pytest temporary directory.
    """
    source_root = tmp_path / "source"
    write_tiny_source(source_root)
    output_root = tmp_path / "output"
    build_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="extra_forbidden"):
        verify_local_import(source_root, output_root, expected_counts=_TINY_COUNTS)
