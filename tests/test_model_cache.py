"""Tests for shared model-cache path resolution."""

from pathlib import Path

import pytest

from synth_setter.model_cache import embedding_model_dir, synth_setter_cache_dir


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
