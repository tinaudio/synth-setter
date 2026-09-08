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
