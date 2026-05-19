"""Tar/wds branch tests for ``synth_setter.pipeline.ci.validate_shard``.

These tests pin the new dispatch-by-suffix surface that ``validate_shard``
grew when the tar/wds branch landed alongside the existing HDF5 path:

- ``.h5`` shards continue to run the HDF5 checks (regression guard).
- ``.tar`` shards run the new tar/wds branch — ``metadata.json`` presence
  and strict parsing, every ``<batch_key>.<field>.npy`` member loadable as
  numpy, the summed row count per field equal to ``samples_per_shard``, and
  each batch's inner shape (``arr.shape[1:]``) equal to the writer's
  source-of-truth shape helpers in ``synth_setter.data.vst.shapes``.
- Any other suffix surfaces a clear error naming the supported set.

This file is intentionally *not* on the pydoclint exclude list — every
helper and test below carries a full sphinx-style docstring (``:param:`` /
``:returns:`` / ``:rtype:``) so the new defs land with full coverage.
"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.schemas.spec import DatasetSpec

_VALID_AUDIO_CHANNELS = 2
_VALID_AUDIO_SAMPLES_PER_ROW = 64000
_VALID_MEL_INNER_SHAPE: tuple[int, int, int] = (2, 128, 401)
_VALID_PARAM_LENGTH = 92
_VALID_METADATA: dict[str, object] = {
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "sample_rate": 16000,
    "channels": _VALID_AUDIO_CHANNELS,
    "min_loudness": -55.0,
}


@pytest.fixture()
def real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetSpec:
    """Build a ``DatasetSpec`` whose render config matches the module's ``_VALID_*`` constants.

    The sample rate (``16000``) and duration (``4.0``) make audio time-samples
    ``64000`` and mel ``n_frames`` ``401`` — lining up with ``_VALID_*`` so each
    test can override exactly one dim while leaving the others correct.

    :param tmp_path: pytest-provided temp directory used for the fake VST3 bundle.
    :param monkeypatch: pytest monkeypatch used to freeze the git/timestamp factories
        the spec validators consult — keeps spec construction deterministic on machines
        without the repo's git metadata available to the schema layer.
    :returns: Spec whose render fields match the ``_VALID_*`` constants in this module.
    :rtype: DatasetSpec
    """
    fixed_now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: fixed_now)

    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')

    return DatasetSpec(
        task_name="test-dataset",
        output_format="wds",
        train_val_test_sizes=(10, 0, 0),
        base_seed=42,
        r2_bucket="intermediate-data",
        render={
            "plugin_path": str(tmp_path / "FakePlugin.vst3"),
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": _VALID_AUDIO_CHANNELS,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10,
            "gui_toggle_cadence": "never",
        },  # type: ignore[arg-type]
    )


def _npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to ``.npy`` bytes via an in-memory buffer.

    :param arr: Array to serialize. Any dtype/shape numpy accepts is fine.
    :returns: Raw ``.npy`` payload bytes — what ``np.load`` would read back.
    :rtype: bytes
    """
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _make_tar_with(path: Path, members: dict[str, bytes]) -> None:
    """Write an uncompressed tar at ``path`` with one member per ``members`` entry.

    :param path: Filesystem path where the tar archive will be written.
    :param members: Mapping from in-tar member name to raw bytes; iteration order
        is preserved as insertion order.
    :returns: ``None``.
    :rtype: None
    """
    with tarfile.open(path, mode="w:") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _valid_batch_arrays(
    n_rows: int,
) -> dict[str, np.ndarray]:
    """Return canonical-shape per-field arrays the writer would emit for one batch.

    :param n_rows: Number of rows for this batch (sum across batches must equal
        ``spec.render.samples_per_shard``).
    :returns: Mapping from field name to a zero-filled ``float32`` array at the writer's
        canonical inner shape.
    :rtype: dict[str, np.ndarray]
    """
    return {
        "audio": np.zeros(
            (n_rows, _VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW), dtype=np.float32
        ),
        "mel_spec": np.zeros((n_rows, *_VALID_MEL_INNER_SHAPE), dtype=np.float32),
        "param_array": np.zeros((n_rows, _VALID_PARAM_LENGTH), dtype=np.float32),
    }


def _make_valid_tar_bytes(path: Path, batches: list[tuple[int, int]]) -> None:
    """Write a fully-valid tar shard at ``path`` with the given per-batch row layout.

    ``batches`` is a list of ``(batch_start_idx, n_rows)`` pairs — the writer's
    naming convention is ``<batch_start_idx:08d>.<field>.npy``. Members are
    written in writer-style insertion order with ``metadata.json`` last.

    :param path: Filesystem path where the tar archive will be written.
    :param batches: List of ``(batch_start_idx, n_rows)`` pairs describing each
        per-batch group of ``.npy`` members in the tar.
    :returns: ``None``.
    :rtype: None
    """
    members: dict[str, bytes] = {}
    for batch_start_idx, n_rows in batches:
        for field, arr in _valid_batch_arrays(n_rows).items():
            members[f"{batch_start_idx:08d}.{field}.npy"] = _npy_bytes(arr)
    members["metadata.json"] = json.dumps(_VALID_METADATA).encode("utf-8")
    _make_tar_with(path, members)


def _make_valid_h5_shard(path: Path, n_rows: int) -> None:
    """Write an HDF5 shard at ``path`` whose datasets match the writer's canonical shapes.

    :param path: Filesystem path where the HDF5 shard will be written.
    :param n_rows: Row count for every dataset; matched against ``samples_per_shard``.
    :returns: ``None``.
    :rtype: None
    """
    shapes: dict[str, tuple[int, ...]] = {
        "audio": (n_rows, _VALID_AUDIO_CHANNELS, _VALID_AUDIO_SAMPLES_PER_ROW),
        "mel_spec": (n_rows, *_VALID_MEL_INNER_SHAPE),
        "param_array": (n_rows, _VALID_PARAM_LENGTH),
    }
    with h5py.File(path, "w") as f:
        for name, shape in shapes.items():
            f.create_dataset(name, shape=shape, dtype=np.float32)


class TestSuffixDispatch:
    """``validate_shard`` dispatches on the shard's filename suffix."""

    def test_validate_shard_dispatches_on_h5_suffix(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A valid ``.h5`` shard runs the HDF5 path and returns no errors.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.h5"
        _make_valid_h5_shard(shard_path, real_spec.render.samples_per_shard)

        assert validate_shard(shard_path, real_spec) == []

    def test_validate_shard_dispatches_on_tar_suffix(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A valid ``.tar`` shard runs the tar/wds path and returns no errors.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        _make_valid_tar_bytes(shard_path, [(0, real_spec.render.samples_per_shard)])

        assert validate_shard(shard_path, real_spec) == []

    def test_validate_shard_rejects_unsupported_suffix(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A ``.bin`` suffix surfaces a clear error naming the registered set.

        :param real_spec: Spec; the suffix dispatch runs before any shape check.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.bin"
        shard_path.write_bytes(b"not a shard")

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert ".bin" in errors[0]
        assert ".h5" in errors[0]
        assert ".tar" in errors[0]


class TestTarShardValidation:
    """Tar/wds branch validates metadata, members, row counts, and inner shapes."""

    def test_validate_shard_tar_rejects_missing_metadata_json(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A tar without ``metadata.json`` surfaces a missing-member error.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        members: dict[str, bytes] = {}
        for field, arr in _valid_batch_arrays(real_spec.render.samples_per_shard).items():
            members[f"00000000.{field}.npy"] = _npy_bytes(arr)
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any("metadata.json" in err for err in errors)

    def test_validate_shard_tar_rejects_invalid_metadata_json(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A tar whose ``metadata.json`` has an unknown extra field surfaces a ValidationError.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        members: dict[str, bytes] = {}
        for field, arr in _valid_batch_arrays(real_spec.render.samples_per_shard).items():
            members[f"00000000.{field}.npy"] = _npy_bytes(arr)
        bad_metadata = {**_VALID_METADATA, "unexpected_key": "boom"}
        members["metadata.json"] = json.dumps(bad_metadata).encode("utf-8")
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any("metadata.json" in err and "ShardMetadata" in err for err in errors)

    def test_validate_shard_tar_rejects_missing_field_npy(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A tar missing one field's ``.npy`` members surfaces a missing-field error.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        members: dict[str, bytes] = {}
        for field, arr in _valid_batch_arrays(real_spec.render.samples_per_shard).items():
            if field == "param_array":
                continue
            members[f"00000000.{field}.npy"] = _npy_bytes(arr)
        members["metadata.json"] = json.dumps(_VALID_METADATA).encode("utf-8")
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any("param_array" in err for err in errors)

    def test_validate_shard_tar_rejects_wrong_row_count(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A tar whose summed rows-per-field is not ``samples_per_shard`` surfaces an error.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        wrong_n = real_spec.render.samples_per_shard - 1
        _make_valid_tar_bytes(shard_path, [(0, wrong_n)])

        errors = validate_shard(shard_path, real_spec)

        assert any(
            "audio" in err
            and str(real_spec.render.samples_per_shard) in err
            and str(wrong_n) in err
            for err in errors
        )

    def test_validate_shard_tar_rejects_wrong_inner_shape(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A tar with audio's trailing time-samples dim wrong surfaces an inner-shape error.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        n_rows = real_spec.render.samples_per_shard
        wrong_time = _VALID_AUDIO_SAMPLES_PER_ROW + 1
        members: dict[str, bytes] = {
            "00000000.audio.npy": _npy_bytes(
                np.zeros((n_rows, _VALID_AUDIO_CHANNELS, wrong_time), dtype=np.float32)
            ),
            "00000000.mel_spec.npy": _npy_bytes(
                np.zeros((n_rows, *_VALID_MEL_INNER_SHAPE), dtype=np.float32)
            ),
            "00000000.param_array.npy": _npy_bytes(
                np.zeros((n_rows, _VALID_PARAM_LENGTH), dtype=np.float32)
            ),
            "metadata.json": json.dumps(_VALID_METADATA).encode("utf-8"),
        }
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any(
            "audio" in err and "inner shape" in err and str(wrong_time) in err for err in errors
        )

    def test_validate_shard_tar_accepts_multi_batch_layout(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """Multiple per-batch ``.npy`` groups that sum to ``samples_per_shard`` pass.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        total = real_spec.render.samples_per_shard
        first = total // 2
        second = total - first
        _make_valid_tar_bytes(shard_path, [(0, first), (first, second)])

        assert validate_shard(shard_path, real_spec) == []

    def test_validate_shard_tar_rejects_malformed_member_name(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A ``.npy`` member without the ``<batch:08d>.<field>.npy`` prefix is rejected.

        Guards against silently coercing names like ``audio.npy`` (no batch key)
        or ``foo.audio.npy`` (non-numeric prefix) into the audio bucket. Even
        though the trailing field token matches, the writer never emits such
        names — accepting them would let a malformed shard through.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        n_rows = real_spec.render.samples_per_shard
        arrays = _valid_batch_arrays(n_rows)
        members: dict[str, bytes] = {
            "audio.npy": _npy_bytes(arrays["audio"]),
            "00000000.mel_spec.npy": _npy_bytes(arrays["mel_spec"]),
            "00000000.param_array.npy": _npy_bytes(arrays["param_array"]),
            "metadata.json": json.dumps(_VALID_METADATA).encode("utf-8"),
        }
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any("malformed tar member name" in err and "audio.npy" in err for err in errors)

    def test_validate_shard_tar_rejects_within_batch_row_mismatch(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A batch key whose per-field row counts disagree surfaces a mismatch error.

        The writer's invariant is that all three fields under one batch key
        share the same N. A tar where ``audio`` has N rows under one batch
        key but ``mel_spec``/``param_array`` have different N under that same
        key would produce misaligned WebDataset samples even if the per-field
        sums match.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        total = real_spec.render.samples_per_shard
        batch_a_audio = 3
        batch_a_other = total - batch_a_audio
        batch_b_audio = total - batch_a_audio
        batch_b_other = batch_a_audio
        members: dict[str, bytes] = {
            "00000000.audio.npy": _npy_bytes(_valid_batch_arrays(batch_a_audio)["audio"]),
            "00000000.mel_spec.npy": _npy_bytes(_valid_batch_arrays(batch_a_other)["mel_spec"]),
            "00000000.param_array.npy": _npy_bytes(
                _valid_batch_arrays(batch_a_other)["param_array"]
            ),
            f"{batch_a_audio:08d}.audio.npy": _npy_bytes(
                _valid_batch_arrays(batch_b_audio)["audio"]
            ),
            f"{batch_a_audio:08d}.mel_spec.npy": _npy_bytes(
                _valid_batch_arrays(batch_b_other)["mel_spec"]
            ),
            f"{batch_a_audio:08d}.param_array.npy": _npy_bytes(
                _valid_batch_arrays(batch_b_other)["param_array"]
            ),
            "metadata.json": json.dumps(_VALID_METADATA).encode("utf-8"),
        }
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any("row-count mismatch" in err for err in errors)

    def test_validate_shard_tar_rejects_batch_key_missing_field(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A batch key missing one of the three field ``.npy`` files surfaces an error.

        The writer emits one ``.npy`` per ``DATASET_FIELD_NAMES`` for every
        batch key. A shard with two batch keys where one batch is missing
        ``param_array`` (yet the field's overall total still hits
        ``samples_per_shard`` via the other batch) must still be flagged.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        total = real_spec.render.samples_per_shard
        first = total // 2
        second = total - first
        members: dict[str, bytes] = {
            "00000000.audio.npy": _npy_bytes(_valid_batch_arrays(first)["audio"]),
            "00000000.mel_spec.npy": _npy_bytes(_valid_batch_arrays(first)["mel_spec"]),
            f"{first:08d}.audio.npy": _npy_bytes(_valid_batch_arrays(second)["audio"]),
            f"{first:08d}.mel_spec.npy": _npy_bytes(_valid_batch_arrays(second)["mel_spec"]),
            f"{first:08d}.param_array.npy": _npy_bytes(_valid_batch_arrays(second)["param_array"]),
            "metadata.json": json.dumps(_VALID_METADATA).encode("utf-8"),
        }
        members[f"{first + second:08d}.param_array.npy"] = _npy_bytes(
            _valid_batch_arrays(first)["param_array"]
        )
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any(
            "00000000" in err and "missing field" in err and "param_array" in err for err in errors
        )

    def test_validate_shard_tar_rejects_npz_payload_under_npy_name(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A ``.npy`` member whose bytes are actually ``.npz`` (NpzFile) is rejected cleanly.

        Without this guard ``np.load`` would return an ``NpzFile`` which has
        no ``.shape`` attribute, and the per-batch ``arr.shape[0]`` access
        would crash with an opaque ``AttributeError`` instead of surfacing
        the malformed payload as a validation error.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        n_rows = real_spec.render.samples_per_shard
        arrays = _valid_batch_arrays(n_rows)
        npz_buf = io.BytesIO()
        np.savez(npz_buf, audio=arrays["audio"])
        members: dict[str, bytes] = {
            "00000000.audio.npy": npz_buf.getvalue(),
            "00000000.mel_spec.npy": _npy_bytes(arrays["mel_spec"]),
            "00000000.param_array.npy": _npy_bytes(arrays["param_array"]),
            "metadata.json": json.dumps(_VALID_METADATA).encode("utf-8"),
        }
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any(
            "00000000.audio.npy" in err and "expected a single ndarray" in err for err in errors
        )

    def test_validate_shard_tar_rejects_zero_d_scalar_npy(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A ``.npy`` member that loads as a 0-d scalar is rejected cleanly.

        A 0-d ndarray has ``.shape == ()`` so ``arr.shape[0]`` would raise
        ``IndexError`` — we want a targeted validation error instead.

        :param real_spec: Spec whose render config matches the canonical valid shapes.
        :param tmp_path: pytest-provided temp directory for the shard file.
        :returns: ``None``.
        :rtype: None
        """
        shard_path = tmp_path / "shard-000000.tar"
        n_rows = real_spec.render.samples_per_shard
        arrays = _valid_batch_arrays(n_rows)
        members: dict[str, bytes] = {
            "00000000.audio.npy": _npy_bytes(np.array(0.0, dtype=np.float32)),
            "00000000.mel_spec.npy": _npy_bytes(arrays["mel_spec"]),
            "00000000.param_array.npy": _npy_bytes(arrays["param_array"]),
            "metadata.json": json.dumps(_VALID_METADATA).encode("utf-8"),
        }
        _make_tar_with(shard_path, members)

        errors = validate_shard(shard_path, real_spec)

        assert any("00000000.audio.npy" in err and "0-d scalar" in err for err in errors)
