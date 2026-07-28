"""Small official-layout NSynth fixtures for importer tests."""

from __future__ import annotations

import json
import wave
from pathlib import Path

SPLITS = ("train", "valid", "test")


def example(note_str: str) -> dict[str, object]:
    """Return one complete NSynth metadata record.

    :param note_str: Safe note identifier stored in the record.
    :returns: Metadata with every official NSynth field.
    """
    return {
        "instrument": 7,
        "instrument_family": 0,
        "instrument_family_str": "bass",
        "instrument_source": 2,
        "instrument_source_str": "synthetic",
        "instrument_str": "bass_synthetic_007",
        "note": 100,
        "note_str": note_str,
        "pitch": 33,
        "qualities": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "qualities_str": ["percussive"],
        "sample_rate": 16000,
        "velocity": 100,
    }


def wav_bytes() -> bytes:
    """Return a valid mono PCM WAV containing four samples.

    :returns: Complete RIFF/WAV bytes.
    """
    from io import BytesIO

    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00\x01\x00\xff\xff\x00\x00")
    return output.getvalue()


def write_source_split(
    source_root: Path,
    split: str,
    records: dict[str, dict[str, object]],
    *,
    audio: dict[str, bytes] | None = None,
) -> Path:
    """Write one extracted NSynth split in the official layout.

    :param source_root: Parent of the extracted split directories.
    :param split: Official split name.
    :param records: Top-level examples mapping.
    :param audio: WAV payloads keyed by note string; defaults to one valid WAV per record.
    :returns: Extracted split directory.
    """
    split_root = source_root / f"nsynth-{split}"
    audio_root = split_root / "audio"
    audio_root.mkdir(parents=True)
    examples_bytes = json.dumps(records, indent=1).encode("utf-8") + b"\n"
    (split_root / "examples.json").write_bytes(examples_bytes)
    payloads = {name: wav_bytes() for name in records} if audio is None else audio
    for note_str, payload in payloads.items():
        (audio_root / f"{note_str}.wav").write_bytes(payload)
    return split_root


def write_tiny_source(source_root: Path) -> dict[str, str]:
    """Write one valid example in each official split.

    :param source_root: Parent directory to populate.
    :returns: Note string keyed by split.
    """
    notes = {
        "test": "bass_synthetic_007-033-100-test",
        "train": "bass_synthetic_007-033-100-train",
        "valid": "bass_synthetic_007-033-100-valid",
    }
    for split in SPLITS:
        note_str = notes[split]
        write_source_split(source_root, split, {note_str: example(note_str)})
    return notes
