"""Behavioral tests for the encoder-weights R2 mirror and hydration CLI."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from synth_setter.pipeline.data.hydrate_encoders import (
    DEFAULT_CLAP_CHECKPOINT,
    hydrate,
    hydrated_clap_checkpoint,
    main,
    mirror,
    music2latent_weights_path,
)

_ENCODERS_URI = "r2://intermediate-data/models/encoders"
_SNAPSHOT_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")


def _write_clap_snapshot(clap_dir: Path) -> None:
    """Write a minimal flat CLAP snapshot directory.

    :param clap_dir: Destination directory for the snapshot files.
    """
    clap_dir.mkdir(parents=True, exist_ok=True)
    for name in _SNAPSHOT_FILES:
        (clap_dir / name).write_text(f"content of {name}")


def _seed_fake_remote(fake_r2_remote: Path) -> Path:
    """Populate the fake R2 remote with a mirrored encoder layout.

    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    :returns: The seeded encoders prefix directory.
    """
    prefix = fake_r2_remote / "intermediate-data" / "models" / "encoders"
    _write_clap_snapshot(prefix / "clap-htsat-unfused")
    m2l = prefix / "music2latent" / "music2latent.pt"
    m2l.parent.mkdir(parents=True)
    m2l.write_bytes(b"m2l-weights")
    return prefix


def test_mirror_uploads_clap_snapshot_and_m2l_weights(
    tmp_path: Path, fake_r2_remote: Path
) -> None:
    """Mirroring lands the flat CLAP snapshot and the m2l weights under the prefix.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    clap_source = tmp_path / "clap-src"
    _write_clap_snapshot(clap_source)
    m2l_source = tmp_path / "music2latent.pt"
    m2l_source.write_bytes(b"m2l-weights")

    mirror(_ENCODERS_URI, clap_source=clap_source, m2l_source=m2l_source)

    prefix = fake_r2_remote / "intermediate-data" / "models" / "encoders"
    for name in _SNAPSHOT_FILES:
        assert (prefix / "clap-htsat-unfused" / name).read_text() == f"content of {name}"
    assert (prefix / "music2latent" / "music2latent.pt").read_bytes() == b"m2l-weights"


def test_mirror_incomplete_clap_source_raises(tmp_path: Path, fake_r2_remote: Path) -> None:
    """A CLAP source missing required snapshot files is rejected before upload.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    clap_source = tmp_path / "clap-src"
    clap_source.mkdir()
    (clap_source / "config.json").write_text("{}")
    m2l_source = tmp_path / "music2latent.pt"
    m2l_source.write_bytes(b"m2l-weights")

    with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
        mirror(_ENCODERS_URI, clap_source=clap_source, m2l_source=m2l_source)


def test_hydrate_materializes_clap_dir_and_m2l_weights(
    tmp_path: Path, fake_r2_remote: Path
) -> None:
    """Hydration copies the mirrored snapshot and weights to the local targets.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    _seed_fake_remote(fake_r2_remote)
    clap_dir = tmp_path / "clap"
    m2l_path = tmp_path / "pkg" / "models" / "music2latent.pt"

    hydrate(_ENCODERS_URI, clap_dir=clap_dir, m2l_path=m2l_path)

    for name in _SNAPSHOT_FILES:
        assert (clap_dir / name).read_text() == f"content of {name}"
    assert m2l_path.read_bytes() == b"m2l-weights"


def test_hydrate_rerun_is_idempotent(tmp_path: Path, fake_r2_remote: Path) -> None:
    """A second hydration run succeeds and leaves the targets unchanged.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    _seed_fake_remote(fake_r2_remote)
    clap_dir = tmp_path / "clap"
    m2l_path = tmp_path / "pkg" / "models" / "music2latent.pt"
    hydrate(_ENCODERS_URI, clap_dir=clap_dir, m2l_path=m2l_path)

    hydrate(_ENCODERS_URI, clap_dir=clap_dir, m2l_path=m2l_path)

    for name in _SNAPSHOT_FILES:
        assert (clap_dir / name).read_text() == f"content of {name}"
    assert m2l_path.read_bytes() == b"m2l-weights"


def test_hydrate_preserves_existing_m2l_weights(tmp_path: Path, fake_r2_remote: Path) -> None:
    """Weights already installed by the package's own downloader are left untouched.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    _seed_fake_remote(fake_r2_remote)
    clap_dir = tmp_path / "clap"
    m2l_path = tmp_path / "pkg" / "models" / "music2latent.pt"
    m2l_path.parent.mkdir(parents=True)
    m2l_path.write_bytes(b"already-installed")

    hydrate(_ENCODERS_URI, clap_dir=clap_dir, m2l_path=m2l_path)

    assert m2l_path.read_bytes() == b"already-installed"


def test_hydrated_clap_checkpoint_complete_dir_returns_dir(tmp_path: Path) -> None:
    """A complete hydrated snapshot is preferred over the HF checkpoint name.

    :param tmp_path: Per-test scratch directory.
    """
    clap_dir = tmp_path / "clap"
    _write_clap_snapshot(clap_dir)

    assert hydrated_clap_checkpoint(clap_dir) == str(clap_dir)


def test_hydrated_clap_checkpoint_incomplete_dir_falls_back(tmp_path: Path) -> None:
    """A missing or partial local snapshot falls back to the HF checkpoint name.

    :param tmp_path: Per-test scratch directory.
    """
    assert hydrated_clap_checkpoint(tmp_path / "absent") == DEFAULT_CLAP_CHECKPOINT
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    assert hydrated_clap_checkpoint(partial) == DEFAULT_CLAP_CHECKPOINT


def test_music2latent_weights_path_resolves_inside_package() -> None:
    """The hydration target is the package-native models directory."""
    pytest.importorskip("music2latent", reason="music2latent not installed")

    path = music2latent_weights_path()

    assert path.name == "music2latent.pt"
    assert path.parent.name == "models"
    assert "music2latent" in path.parent.parent.name


def test_cli_hydrates_to_explicit_targets(tmp_path: Path, fake_r2_remote: Path) -> None:
    """The console entrypoint hydrates both encoders to the given targets.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    _seed_fake_remote(fake_r2_remote)
    clap_dir = tmp_path / "clap"
    m2l_path = tmp_path / "pkg" / "models" / "music2latent.pt"

    result = CliRunner().invoke(
        main,
        [
            "--encoders-uri",
            _ENCODERS_URI,
            "--clap-dir",
            str(clap_dir),
            "--m2l-path",
            str(m2l_path),
        ],
    )

    assert result.exit_code == 0, result.output
    for name in _SNAPSHOT_FILES:
        assert (clap_dir / name).is_file()
    assert m2l_path.read_bytes() == b"m2l-weights"


def test_cli_mirror_uploads_from_explicit_sources(tmp_path: Path, fake_r2_remote: Path) -> None:
    """The console entrypoint's mirror mode seeds the R2 prefix from local sources.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    clap_source = tmp_path / "clap-src"
    _write_clap_snapshot(clap_source)
    m2l_source = tmp_path / "music2latent.pt"
    m2l_source.write_bytes(b"m2l-weights")

    result = CliRunner().invoke(
        main,
        [
            "--encoders-uri",
            _ENCODERS_URI,
            "--mirror",
            "--clap-source",
            str(clap_source),
            "--m2l-source",
            str(m2l_source),
        ],
    )

    assert result.exit_code == 0, result.output
    prefix = fake_r2_remote / "intermediate-data" / "models" / "encoders"
    assert (prefix / "music2latent" / "music2latent.pt").read_bytes() == b"m2l-weights"


def test_cli_mirror_missing_m2l_source_exits_nonzero(
    tmp_path: Path, fake_r2_remote: Path
) -> None:
    """Mirror mode fails fast with a clear error when the local weights are absent.

    :param tmp_path: Per-test scratch directory.
    :param fake_r2_remote: Local root backing the ``r2:`` remote.
    """
    clap_source = tmp_path / "clap-src"
    _write_clap_snapshot(clap_source)

    result = CliRunner().invoke(
        main,
        [
            "--encoders-uri",
            _ENCODERS_URI,
            "--mirror",
            "--clap-source",
            str(clap_source),
            "--m2l-source",
            str(tmp_path / "missing.pt"),
        ],
    )

    assert result.exit_code == 1
    assert "missing.pt" in result.output
