"""Behavior tests for the RIR corpus publication tooling (no network, no R2)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.data_publication.rir_corpora import rirpub, specs


def _wav_bytes(*, sample_rate: int, channels: int, frames: int) -> bytes:
    """Encode a silent PCM-16 WAV in memory.

    :param sample_rate: Sample rate in Hz.
    :param channels: Channel count.
    :param frames: Frame count.
    :returns: WAV container bytes.
    """
    buffer = io.BytesIO()
    samples = np.zeros((frames, channels), dtype=np.float32)
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.mark.parametrize("name", sorted(specs.SPECS))
def test_every_spec_pins_source_license_and_citation(name: str) -> None:
    """Every registered corpus carries the provenance fields the card requires.

    :param name: Corpus name.
    """
    spec = specs.SPECS[name]
    assert spec.pin and spec.license and spec.citation and spec.source_urls


@pytest.mark.parametrize(
    "relative",
    [".DS_Store", "Audio/.DS_Store", "__MACOSX/x/._a.wav", "a/__MACOSX/._b.wav", "Thumbs.db"],
)
def test_excluded_drops_macos_and_windows_litter(relative: str) -> None:
    """Archive litter is excluded regardless of nesting depth.

    :param relative: Litter path relative to the payload root.
    """
    assert specs.SPECS["MITIRSurvey"].excluded(relative)


def test_excluded_keeps_regular_payload_and_spec_globs_compose() -> None:
    """Corpus-specific globs add to the litter rules without displacing them."""
    spec = specs.SPECS["OpenAIR"]
    assert not spec.excluded("IRs/maes-howe/stereo/maes-howe.wav")
    assert spec.excluded("IRs/maes-howe/index.html")


def test_is_ancillary_matches_documentation_but_not_audio() -> None:
    """Documentation is copied as plain files; audio is not."""
    spec = specs.SPECS["EchoThief"]
    assert spec.is_ancillary("a/EchoThief License.pdf")
    assert not spec.is_ancillary("a/Brutalism/GalbraithHall.wav")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("x/a.wav", "audio/wav"),
        ("x/a.WAV", "audio/wav"),
        ("x/a.sofa", "application/x-sofa-hdf5"),
        ("x/S1_Mrir.npy", "application/x-numpy"),
        ("x/rirs.mat", "application/x-matlab-data"),
        ("x/unknown.xyz", "application/octet-stream"),
    ],
)
def test_media_type_is_inferred_from_extension(path: str, expected: str) -> None:
    """Media types come from the extension, case-insensitively, with an octet-stream fallback.

    :param path: Source object path.
    :param expected: Expected MIME type.
    """
    assert rirpub.media_type(Path(path)) == expected


def test_audio_info_reports_header_fields_for_decodable_wav() -> None:
    """A decodable WAV yields its libsndfile header fields."""
    payload = _wav_bytes(sample_rate=32_000, channels=1, frames=640)
    info = rirpub.audio_info(payload, Path("ir.wav"))
    assert info == {
        "audio_decodable": True,
        "sample_rate": 32_000,
        "channels": 1,
        "frames": 640,
        "subtype": "WAV/PCM_16",
    }


def test_audio_info_marks_non_audio_extension_undecodable_without_reading() -> None:
    """Container formats are never handed to libsndfile."""
    info = rirpub.audio_info(b"not audio", Path("rirs.mat"))
    assert info["audio_decodable"] is False
    assert info["sample_rate"] is None


def test_audio_info_marks_corrupt_wav_undecodable() -> None:
    """A corrupt WAV is reported undecodable instead of raising."""
    info = rirpub.audio_info(b"RIFF garbage", Path("broken.wav"))
    assert info["audio_decodable"] is False


def test_corpus_root_honours_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``RIR_CORPORA_ROOT`` relocates the corpora root.

    :param monkeypatch: Environment patcher.
    :param tmp_path: Override root.
    """
    monkeypatch.setenv("RIR_CORPORA_ROOT", str(tmp_path))
    assert rirpub.corpora_root() == tmp_path


def _tiny_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[specs.Spec, Path]:
    """Lay out a three-object corpus (two WAVs, one licence) under a temp root.

    :param tmp_path: Temp root.
    :param monkeypatch: Environment patcher.
    :returns: The descriptor and the corpus root.
    """
    monkeypatch.setenv("RIR_CORPORA_ROOT", str(tmp_path))
    root = tmp_path / "Tiny"
    source = root / "source"
    source.mkdir(parents=True)
    payload = source / "loose"
    (payload / "b").mkdir(parents=True)
    (payload / "a.wav").write_bytes(_wav_bytes(sample_rate=8_000, channels=1, frames=16))
    (payload / "b" / "c.wav").write_bytes(_wav_bytes(sample_rate=8_000, channels=2, frames=8))
    (payload / "LICENSE.txt").write_text("cc by\n", encoding="utf-8")
    (payload / ".DS_Store").write_bytes(b"junk")
    (source / "acquisition.json").write_text("{}\n", encoding="utf-8")
    spec = specs.Spec(
        name="Tiny",
        title="Tiny corpus",
        summary="s",
        source_urls=["https://example.invalid"],
        pin="pinned",
        license="CC BY 4.0",
        license_note="n",
        citation="c",
        acquisition_commands=["echo fetched"],
        payload_root="source/loose",
    )
    return spec, root


def test_extract_and_build_publish_exact_bytes_in_row_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extract/build stages write one blob row per non-litter object with exact bytes.

    :param tmp_path: Temp root.
    :param monkeypatch: Environment patcher.
    """
    import lance

    spec, root = _tiny_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(rirpub, "MIN_FREE_BYTES", 0)
    corpus = rirpub.Corpus(spec)
    corpus.extract()
    corpus.build()
    dataset = lance.dataset(str(root / "publication" / "all.lance"))
    rows = dataset.scanner(columns=rirpub.META_COLUMNS).to_table().to_pylist()
    assert [r["source_path"] for r in rows] == ["LICENSE.txt", "a.wav", "b/c.wav"]
    assert [r["audio_decodable"] for r in rows] == [False, True, True]
    assert [r["channels"] for r in rows] == [None, 1, 2]
    blobs = dataset.take_blobs("source_bytes", indices=[0, 1, 2])
    payload = root / "source" / "loose"
    assert [b.read() for b in blobs] == [
        (payload / "LICENSE.txt").read_bytes(),
        (payload / "a.wav").read_bytes(),
        (payload / "b" / "c.wav").read_bytes(),
    ]
    assert (root / "publication" / "source" / "LICENSE.txt").read_text() == "cc by\n"
    assert dataset.schema.metadata[b"source_pin"] == b"pinned"
    assert "## Schema" in (root / "publication" / "README.md").read_text()


def test_batches_split_at_row_bound_and_keep_oversized_object_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batches close at the row bound and a single object may exceed the byte bound.

    :param tmp_path: Temp root.
    :param monkeypatch: Environment patcher.
    """
    spec, root = _tiny_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(rirpub, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(rirpub, "BATCH_ROWS", 2)
    monkeypatch.setattr(rirpub, "BATCH_BYTES", 1)
    corpus = rirpub.Corpus(spec)
    corpus.extract()
    inventory = corpus.load_inventory()
    sizes = [b.num_rows for b in corpus.batches(inventory, corpus.schema([]))]
    assert sizes == [1, 1, 1]


def test_build_rejects_source_object_changed_after_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload edited between extract and build fails the build instead of publishing.

    :param tmp_path: Temp root.
    :param monkeypatch: Environment patcher.
    """
    spec, root = _tiny_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(rirpub, "MIN_FREE_BYTES", 0)
    corpus = rirpub.Corpus(spec)
    corpus.extract()
    (root / "source" / "loose" / "LICENSE.txt").write_text("changed\n", encoding="utf-8")
    # Lance surfaces the generator's RuntimeError through its C-data bridge as OSError.
    with pytest.raises((RuntimeError, OSError), match="changed since inventory"):
        corpus.build()


def test_main_rejects_unknown_stage_and_corpus() -> None:
    """The dispatcher refuses unknown stages and corpora before touching disk."""
    with pytest.raises(SystemExit, match="usage"):
        rirpub.main(["publish", "MITIRSurvey"])
    with pytest.raises(SystemExit, match="unknown corpus"):
        rirpub.main(["extract", "Nope"])
