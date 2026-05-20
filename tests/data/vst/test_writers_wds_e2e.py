"""End-to-end tests for the wds branch of ``synth_setter.data.vst.writers``.

Renders real audio through the Surge XT VST3 plugin (``@pytest.mark.requires_vst``)
and validates the on-disk shape of the produced tar shard. The load-bearing
parity test ``test_h5_and_wds_outputs_are_equivalent`` is the original
acceptance criterion in issue #874's checklist — same params, identical bytes.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers Blosc2 for h5py reads
import numpy as np
import pydantic
import pytest
import webdataset as wds

from synth_setter.data.vst.writers import make_hdf5_dataset, make_wds_dataset
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

_ = hdf5plugin  # keep type checkers from flagging the side-effect import

_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"

skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)

# Hardcoded loudness-passing patch and h5↔h5 phase-robust comparison helpers
# are reused verbatim from ``test_generate_vst_dataset.py`` so this module's
# h5↔wds parity check uses the same thresholds the h5↔h5 round-trip tests
# already pin. Imported (not copied) on purpose: a future spec change updates
# the canonical patch in one place, and both test modules track it.
from tests.data.vst.test_generate_vst_dataset import (
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
    _MEL_MEAN_ABS_MAX,
    _assert_audio_metrics_within_thresholds,
    _render_cfg,
)


def _tar_members(tar_path: Path) -> dict[str, bytes]:
    """Return a mapping of every tar member's name to its raw bytes.

    :param tar_path: Filesystem path to the tar archive to read.
    :return: Mapping of tar member name to raw bytes.
    """
    members: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r") as tar:
        for entry in tar:
            if not entry.isfile():
                continue
            fobj = tar.extractfile(entry)
            assert fobj is not None
            members[entry.name] = fobj.read()
    return members


def _load_npy_bytes(raw: bytes) -> np.ndarray:
    """Decode a ``.npy`` payload from raw bytes.

    :param raw: Raw bytes of a ``.npy`` payload.
    :return: Decoded numpy array.
    """
    return np.load(io.BytesIO(raw), allow_pickle=False)


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_wds_dataset_writes_per_batch_npy_members(tmp_path: Path) -> None:
    """A 4-sample shard with batch_size=2 emits two batches of three npy members plus metadata.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out = tmp_path / "shard-000000.tar"
    num_samples = 4
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples

    make_wds_dataset(
        wds_file=out,
        render_cfg=_render_cfg(num_samples, samples_per_render_batch=2),
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )

    members = _tar_members(out)
    expected = {
        "00000000.audio.npy",
        "00000000.mel_spec.npy",
        "00000000.param_array.npy",
        "00000002.audio.npy",
        "00000002.mel_spec.npy",
        "00000002.param_array.npy",
        "metadata.json",
    }
    assert expected <= set(members), (
        f"missing members: {expected - set(members)}; got {sorted(members)}"
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_wds_dataset_metadata_json_is_strict_shard_metadata(tmp_path: Path) -> None:
    """The shard's ``metadata.json`` member parses as a strict ``ShardMetadata`` matching the cfg.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out = tmp_path / "shard-000000.tar"
    num_samples = 2
    render_cfg = _render_cfg(num_samples)

    make_wds_dataset(
        wds_file=out,
        render_cfg=render_cfg,
        fixed_synth_params_list=[_HARDCODED_SYNTH_PARAMS] * num_samples,
        fixed_note_params_list=[_HARDCODED_NOTE_PARAMS] * num_samples,
    )

    members = _tar_members(out)
    meta = ShardMetadata.model_validate_json(members["metadata.json"])
    assert meta.velocity == render_cfg.velocity
    assert meta.signal_duration_seconds == render_cfg.signal_duration_seconds
    assert meta.sample_rate == render_cfg.sample_rate
    assert meta.channels == render_cfg.channels
    assert meta.min_loudness == render_cfg.min_loudness


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_wds_dataset_audio_is_float16(tmp_path: Path) -> None:
    """Audio members are ``float16`` so they match the h5 path's storage precision.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out = tmp_path / "shard-000000.tar"
    num_samples = 2

    make_wds_dataset(
        wds_file=out,
        render_cfg=_render_cfg(num_samples),
        fixed_synth_params_list=[_HARDCODED_SYNTH_PARAMS] * num_samples,
        fixed_note_params_list=[_HARDCODED_NOTE_PARAMS] * num_samples,
    )

    members = _tar_members(out)
    audio = _load_npy_bytes(members["00000000.audio.npy"])
    mel = _load_npy_bytes(members["00000000.mel_spec.npy"])
    params = _load_npy_bytes(members["00000000.param_array.npy"])
    assert audio.dtype == np.float16
    assert mel.dtype == np.float32
    assert params.dtype == np.float32


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_h5_and_wds_outputs_are_equivalent(tmp_path: Path) -> None:
    """Same params written through both writers produce equivalent on-disk arrays.

    Load-bearing test from the original #874 acceptance checklist. Pins the
    invariant that downstream consumers see equivalent data regardless of
    which writer produced the shard — the wds path can't drift from the h5
    path without this test failing.

    Exact byte equality is asserted for ``param_array`` (deterministic — same
    params → same encoded vector) and for ``ShardMetadata``-derived sidecar
    values. The renderer's phase-init nondeterminism (#489) means two renders
    of the same params can differ at the sample level even with
    ``fixed_synth_params_list`` pinned, so audio and mel are compared via the
    phase-robust metrics this repo already pins for h5↔h5 round-trips.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    num_samples = 2
    fixed_synth = [_HARDCODED_SYNTH_PARAMS] * num_samples
    fixed_note = [_HARDCODED_NOTE_PARAMS] * num_samples
    render_cfg = _render_cfg(num_samples)

    h5_path = tmp_path / "shard.h5"
    tar_path = tmp_path / "shard.tar"

    make_hdf5_dataset(
        hdf5_file=h5_path,
        render_cfg=render_cfg,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )
    make_wds_dataset(
        wds_file=tar_path,
        render_cfg=render_cfg,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )

    # Iterate the tar via webdataset so we exercise the consumer-side path.
    wds_rows_audio: list[np.ndarray] = []
    wds_rows_mel: list[np.ndarray] = []
    wds_rows_params: list[np.ndarray] = []
    metadata_json_bytes: bytes | None = None
    for sample in wds.WebDataset(str(tar_path)):  # pyright: ignore[reportAttributeAccessIssue]
        if sample.get("__key__") == "metadata":
            metadata_json_bytes = sample["json"]
            continue
        wds_rows_audio.append(_load_npy_bytes(sample["audio.npy"]))
        wds_rows_mel.append(_load_npy_bytes(sample["mel_spec.npy"]))
        wds_rows_params.append(_load_npy_bytes(sample["param_array.npy"]))

    assert metadata_json_bytes is not None, "missing metadata.json member in wds tar"

    wds_audio = np.concatenate(wds_rows_audio, axis=0)
    wds_mel = np.concatenate(wds_rows_mel, axis=0)
    wds_params = np.concatenate(wds_rows_params, axis=0)

    with h5py.File(h5_path, "r") as f:
        audio_ds = f["audio"]
        mel_ds = f["mel_spec"]
        params_ds = f["param_array"]
        assert isinstance(audio_ds, h5py.Dataset)
        assert isinstance(mel_ds, h5py.Dataset)
        assert isinstance(params_ds, h5py.Dataset)
        h5_audio = audio_ds[...]
        h5_mel = mel_ds[...]
        h5_params = params_ds[...]
        h5_attrs = dict(audio_ds.attrs)

    # Dtypes are pinned by the writers; both formats must agree.
    assert h5_audio.dtype == wds_audio.dtype == np.float16
    assert h5_mel.dtype == wds_mel.dtype == np.float32
    assert h5_params.dtype == wds_params.dtype == np.float32

    # Shapes are pinned by the shapes helpers — both writers must produce them.
    assert h5_audio.shape == wds_audio.shape
    assert h5_mel.shape == wds_mel.shape
    assert h5_params.shape == wds_params.shape

    # param_array is byte-equal: same params in, same encoded vector out.
    np.testing.assert_array_equal(h5_params, wds_params)

    # Audio is phase-perturbed across renders (#489); compare per-row with
    # phase-robust metrics that the h5↔h5 round-trip tests already use.
    h5_audio_f32 = h5_audio.astype(np.float32)
    wds_audio_f32 = wds_audio.astype(np.float32)
    for i in range(num_samples):
        _assert_audio_metrics_within_thresholds(
            h5_audio_f32[i], wds_audio_f32[i], label=f"h5↔wds sample {i}"
        )

    # Mel is deterministic given audio; the audio phase difference produces a
    # small mel difference, bounded by the same _MEL_MEAN_ABS_MAX threshold
    # the h5↔h5 tests use.
    mel_dist = float(np.mean(np.abs(h5_mel - wds_mel)))
    assert mel_dist < _MEL_MEAN_ABS_MAX, (
        f"mel mean abs diff {mel_dist:.4f} exceeds {_MEL_MEAN_ABS_MAX}"
    )

    # Same ShardMetadata projection underlies both — sidecar values match.
    wds_meta = ShardMetadata.model_validate_json(metadata_json_bytes)
    # h5py stores ints/floats; pydantic preserves Python types — compare per field.
    # h5py.AttributeManager returns numpy scalars; ``int()``/``float()`` coerce them at runtime.
    assert int(h5_attrs["velocity"]) == wds_meta.velocity  # pyright: ignore[reportArgumentType]
    assert (
        float(h5_attrs["signal_duration_seconds"])  # pyright: ignore[reportArgumentType]
        == wds_meta.signal_duration_seconds
    )
    assert int(h5_attrs["sample_rate"]) == wds_meta.sample_rate  # pyright: ignore[reportArgumentType]
    assert int(h5_attrs["channels"]) == wds_meta.channels  # pyright: ignore[reportArgumentType]
    assert (
        float(h5_attrs["min_loudness"])  # pyright: ignore[reportArgumentType]
        == wds_meta.min_loudness
    )


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_make_wds_dataset_metadata_json_strict_rejects_extra(tmp_path: Path) -> None:
    """The metadata.json member round-trips through strict ``ShardMetadata`` validation.

    Mirrors the trust-boundary contract used by the consumer: a corrupt
    sidecar would surface as a pydantic ``ValidationError`` rather than
    silently passing through with wrong values.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    out = tmp_path / "shard-000000.tar"
    num_samples = 2

    make_wds_dataset(
        wds_file=out,
        render_cfg=_render_cfg(num_samples),
        fixed_synth_params_list=[_HARDCODED_SYNTH_PARAMS] * num_samples,
        fixed_note_params_list=[_HARDCODED_NOTE_PARAMS] * num_samples,
    )

    members = _tar_members(out)
    raw = json.loads(members["metadata.json"])
    # ``extra="forbid"`` on ShardMetadata means unknown fields raise.
    raw["unexpected_field"] = "boom"
    with pytest.raises(pydantic.ValidationError, match="unexpected_field"):
        ShardMetadata.model_validate(raw)
