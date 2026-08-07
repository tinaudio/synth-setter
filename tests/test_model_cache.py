"""Tests for shared model-cache path resolution."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from synth_setter.model_cache import (
    cache_r2_file,
    checkpoint_files_sha256,
    embedding_model_dir,
    synth_setter_cache_dir,
)


def test_synth_setter_cache_dir_without_xdg_uses_home_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use the conventional home cache when XDG_CACHE_HOME is absent.

    :param monkeypatch: Isolates environment and home-directory discovery.
    :param tmp_path: Home directory for the assertion.
    """
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert synth_setter_cache_dir() == tmp_path / ".cache" / "synth-setter"


def test_synth_setter_cache_dir_with_xdg_uses_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prefer XDG_CACHE_HOME over the conventional home cache.

    :param monkeypatch: Sets the XDG cache root.
    :param tmp_path: Parent of the configured cache root.
    """
    xdg_cache = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))

    assert synth_setter_cache_dir() == xdg_cache / "synth-setter"


def test_checkpoint_files_sha256_ignores_unselected_sibling_artifacts(tmp_path: Path) -> None:
    """A shared snapshot hashes only files owned by the selected model variant.

    :param tmp_path: Snapshot root containing two model variants.
    """
    selected = tmp_path / "tiny" / "args.json"
    selected.parent.mkdir()
    selected.write_text("tiny")
    sibling = tmp_path / "large" / "args.json"
    sibling.parent.mkdir()
    sibling.write_text("large-v1")

    expected = checkpoint_files_sha256(tmp_path, [selected])
    sibling.write_text("large-v2")

    assert checkpoint_files_sha256(tmp_path, [selected]) == expected


def test_embedding_model_dir_places_model_under_shared_embedding_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Place named embedding models under the canonical shared hierarchy.

    :param monkeypatch: Sets the XDG cache root.
    :param tmp_path: Parent of the configured cache root.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert embedding_model_dir("same-s") == (
        tmp_path / "synth-setter" / "models" / "embeddings" / "same-s"
    )


@pytest.mark.parametrize("namespace", ["", ".", "..", "nested/path"])
def test_cache_r2_file_invalid_namespace_raises(
    namespace: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cache namespaces cannot escape the artifact root.

    :param namespace: Malformed cache namespace.
    :param monkeypatch: Isolates the XDG cache root.
    :param tmp_path: Parent of the isolated cache.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(ValueError, match="namespace"):
        cache_r2_file(
            "r2://bucket/models/weights.ckpt",
            namespace,
            hashlib.sha256(b"expected").hexdigest(),
        )


def test_cache_r2_file_replaces_corrupt_cached_bytes_via_real_rclone(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A digest-invalid cache entry is replaced only after a complete transfer.

    :param fake_r2_remote: Local filesystem backing the real rclone transport.
    :param monkeypatch: Isolates the XDG cache root.
    :param tmp_path: Holds the source object and cache.
    """
    source = fake_r2_remote / "bucket" / "models" / "weights.ckpt"
    source.parent.mkdir(parents=True)
    payload = b"complete checkpoint bytes"
    source.write_bytes(payload)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    cached = cache_r2_file("r2://bucket/models/weights.ckpt", "surge-sketch", expected_sha256)
    cached.write_bytes(b"partial")
    repaired = cache_r2_file("r2://bucket/models/weights.ckpt", "surge-sketch", expected_sha256)

    assert repaired == cached
    assert repaired.read_bytes() == payload


def test_cache_r2_file_digest_mismatch_never_publishes_final_file(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject wrong remote bytes without publishing them under the requested pin.

    :param fake_r2_remote: Local filesystem backing the real rclone transport.
    :param monkeypatch: Isolates the XDG cache root.
    :param tmp_path: Holds the source object and cache.
    """
    source = fake_r2_remote / "bucket" / "models" / "weights.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"wrong bytes")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(ValueError, match="SHA-256"):
        cache_r2_file(
            "r2://bucket/models/weights.ckpt",
            "surge-sketch",
            hashlib.sha256(b"expected bytes").hexdigest(),
        )

    cache_root = tmp_path / "cache" / "synth-setter" / "models" / "artifacts"
    assert list(cache_root.rglob("weights.ckpt")) == []


def test_cache_r2_file_failed_transfer_never_publishes_partial_file(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real rclone failure leaves no destination that a retry could accept.

    :param fake_r2_remote: Activates the local-backed real rclone process.
    :param monkeypatch: Isolates the XDG cache root.
    :param tmp_path: Parent of the isolated cache.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with pytest.raises(subprocess.CalledProcessError):
        cache_r2_file(
            "r2://bucket/missing/weights.ckpt",
            "surge-sketch",
            hashlib.sha256(b"expected").hexdigest(),
        )

    cache_root = tmp_path / "cache" / "synth-setter" / "models" / "artifacts"
    assert list(cache_root.rglob("weights.ckpt")) == []
    assert list(cache_root.rglob("*.partial-*")) == []
