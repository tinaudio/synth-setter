"""Behavior tests for the RIR source acquisition module (no network)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.data_publication.rir_corpora import fetch_sources


def _record(files: list[tuple[str, bytes]]) -> dict[str, object]:
    """Build a minimal Zenodo record with md5 checksums for the given payloads.

    :param files: ``(key, bytes)`` pairs.
    :returns: Record JSON.
    """
    return {
        "doi": "10.5281/zenodo.1",
        "files": [
            {
                "key": key,
                "checksum": "md5:" + hashlib.md5(data, usedforsecurity=False).hexdigest(),
            }
            for key, data in files
        ],
    }


def test_zenodo_verifies_checksums_and_writes_acquisition_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fetched file is md5-checked against the record and hashed into the provenance log.

    :param tmp_path: Temp corpora root.
    :param monkeypatch: Patcher for the root and the downloader.
    """
    monkeypatch.setattr(fetch_sources, "ROOT", tmp_path)
    payloads = {"a.zip": b"aaa", "b.csv": b"bbb"}
    source = tmp_path / "X" / "source"
    source.mkdir(parents=True)
    (source / "zenodo-record-1.json").write_text(json.dumps(_record(list(payloads.items()))))
    fetched: list[str] = []

    def fake_download(url: str, out: Path) -> None:
        fetched.append(url)
        out.write_bytes(payloads[out.name])

    monkeypatch.setattr(fetch_sources, "download", fake_download)
    fetch_sources.zenodo("X", 1)
    record = json.loads((source / "acquisition.json").read_text())
    assert [f["key"] for f in record["files"]] == ["a.zip", "b.csv"]
    assert record["files"][0]["sha256"] == hashlib.sha256(b"aaa").hexdigest()
    assert fetched == [
        "https://zenodo.org/api/records/1/files/a.zip/content",
        "https://zenodo.org/api/records/1/files/b.csv/content",
    ]


def test_zenodo_skips_files_already_matching_and_rejects_bad_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present, matching file is not re-fetched; a mismatching download raises.

    :param tmp_path: Temp corpora root.
    :param monkeypatch: Patcher for the root and the downloader.
    """
    monkeypatch.setattr(fetch_sources, "ROOT", tmp_path)
    source = tmp_path / "X" / "source"
    source.mkdir(parents=True)
    (source / "zenodo-record-1.json").write_text(
        json.dumps(_record([("ok.zip", b"ok"), ("bad.zip", b"good")]))
    )
    (source / "ok.zip").write_bytes(b"ok")
    fetched: list[str] = []

    def fake_download(url: str, out: Path) -> None:
        fetched.append(out.name)
        out.write_bytes(b"corrupt")

    monkeypatch.setattr(fetch_sources, "download", fake_download)
    with pytest.raises(RuntimeError, match="md5 mismatch bad.zip"):
        fetch_sources.zenodo("X", 1)
    assert fetched == ["bad.zip"]


def test_zenodo_skip_rule_drops_tau_noise_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The TAU-SRIR rule excludes the ambient-noise archive from acquisition.

    :param tmp_path: Temp corpora root.
    :param monkeypatch: Patcher for the root and the downloader.
    """
    monkeypatch.setattr(fetch_sources, "ROOT", tmp_path)
    source = tmp_path / "TAUSRIR" / "source"
    source.mkdir(parents=True)
    (source / "zenodo-record-6408611.json").write_text(
        json.dumps(_record([("TAU-SNoise_DB.zip", b"n"), ("TAU-SRIR_DB.zip", b"r")]))
    )
    monkeypatch.setattr(fetch_sources, "download", lambda url, out: out.write_bytes(b"r"))
    fetch_sources.zenodo("TAUSRIR", 6408611)
    record = json.loads((source / "acquisition.json").read_text())
    assert [f["key"] for f in record["files"]] == ["TAU-SRIR_DB.zip"]


def test_main_rejects_unknown_family() -> None:
    """An unknown acquisition family exits before any download."""
    with pytest.raises(SystemExit, match="unknown acquisition family"):
        fetch_sources.main(["ftp"])


def _git(repo: Path, args: list[str]) -> str:
    """Run git in a throwaway repository with a fixed identity.

    :param repo: Repository directory.
    :param args: git arguments.
    :returns: Stripped stdout.
    """
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(  # noqa: S603 — args are literal strings
        [fetch_sources.executable("git"), "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def test_require_clean_checkout_rejects_wrong_revision(tmp_path: Path) -> None:
    """A checkout at another commit or with local edits is refused.

    :param tmp_path: Temp directory for a throwaway repository.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, ["init", "-q"])
    (repo / "f").write_text("1")
    _git(repo, ["add", "f"])
    _git(repo, ["commit", "-q", "-m", "one"])
    head = _git(repo, ["rev-parse", "HEAD"])
    fetch_sources._require_clean_checkout(repo, head)
    with pytest.raises(RuntimeError, match="expected a clean tree"):
        fetch_sources._require_clean_checkout(repo, "0" * 40)
    (repo / "f").write_text("2")
    with pytest.raises(RuntimeError, match="local changes"):
        fetch_sources._require_clean_checkout(repo, head)
