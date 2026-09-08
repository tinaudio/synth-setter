#!/usr/bin/env python3
"""Acquire pinned RIR corpus sources with checksum verification and provenance logs.

Invocation (from the repository root)::

    uv run python -m scripts.data_publication.rir_corpora.fetch_sources zenodo [NAME ...]
    uv run python -m scripts.data_publication.rir_corpora.fetch_sources direct
    uv run python -m scripts.data_publication.rir_corpora.fetch_sources ashir
    uv run python -m scripts.data_publication.rir_corpora.fetch_sources openair
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from scripts.data_publication.rir_corpora.specs import executable

log = logging.getLogger(__name__)

ROOT = Path(os.environ.get("RIR_CORPORA_ROOT", "~/datasets/rir-corpora")).expanduser()
ZENODO = {
    "MultiRoomTransition": 13341566,
    "MPRIR": 11148712,
    "THKoelnSRIR": 5031335,
    "TAUSRIR": 6408611,
    "Arni": 6985104,
}
# Only impulse responses are needed; the TAU ambient-noise archive is skipped.
SKIP: dict[str, Callable[[str], bool]] = {"TAUSRIR": lambda key: key.startswith("TAU-SNoise")}
DIRECT = {
    "MITIRSurvey": ["https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"],
    "AachenIR": ["https://www2.iks.rwth-aachen.de/air/air_database_release_1_4.zip"],
    "EchoThief": [
        "https://www.echothief.com/wp-content/uploads/2024/07/EchoThiefImpulseResponseLibrary.zip"
    ],
}
# Zenodo throttles a single stream to ~1 MB/s; aria2 with 8 connections reaches ~14 MiB/s.
ARIA2_ROOT = Path(
    os.environ.get("RIR_ARIA2_ROOT", "~/datasets/guitarset/tools/aria2-root")
).expanduser()
ARIA2C = os.environ.get("RIR_ARIA2C", str(ARIA2_ROOT / "usr/bin/aria2c"))
ARIA2_LIB = os.environ.get("RIR_ARIA2_LIB", str(ARIA2_ROOT / "usr/lib/x86_64-linux-gnu"))
ASHIR_REPOSITORY = "https://github.com/ShanonPearce/ASH-IR-Dataset.git"
ASHIR_REV = "af0d3f51e19cf74dc77a603fd3135a14d8aeb29e"
OPENAIR_STORE = "https://webfiles.york.ac.uk/OPENAIR/"


def now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    :returns: Timestamp for provenance records.
    """
    return datetime.now(UTC).isoformat()


def _digest(path: Path, algorithm: str) -> str:
    """Hash a file in 1 MiB blocks.

    :param path: File to hash.
    :param algorithm: ``sha256`` or ``md5`` (md5 only matches provider checksums).
    :returns: Hex digest.
    """
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return _digest(path, "sha256")


def md5(path: Path) -> str:
    """Return the MD5 of a file, used only to match Zenodo's published checksums.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return _digest(path, "md5")


def download(url: str, out: Path) -> None:
    """Fetch one URL with a multi-connection resumable aria2 download.

    :param url: Source URL.
    :param out: Destination path.
    :raises RuntimeError: Every attempt failed.
    """
    for attempt in range(6):
        result = subprocess.run(  # noqa: S603 — args are literal strings
            [
                ARIA2C,
                "-x",
                "8",
                "-s",
                "8",
                "-k",
                "8M",
                "-c",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--summary-interval=0",
                "--console-log-level=warn",
                "-d",
                str(out.parent),
                "-o",
                out.name,
                url,
            ],
            env={**os.environ, "LD_LIBRARY_PATH": ARIA2_LIB},
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def _zenodo_record(record: int, meta: Path) -> dict[str, object]:
    """Fetch and cache a Zenodo record's JSON, retrying through rate limits.

    :param record: Zenodo record id.
    :param meta: Cache path for the record JSON.
    :returns: Parsed record.
    :raises RuntimeError: The record could not be fetched.
    """
    if meta.exists():
        return json.loads(meta.read_bytes())
    for _ in range(10):
        result = subprocess.run(  # noqa: S603 — args are literal strings
            [
                executable("curl"),
                "-sSL",
                "-H",
                "Accept: application/json",
                f"https://zenodo.org/api/records/{record}",
            ],
            capture_output=True,
            check=False,
        )
        try:
            data = json.loads(result.stdout)
        except ValueError:
            data = {}
        if "files" in data:
            meta.write_bytes(result.stdout)
            return data
        time.sleep(60)
    raise RuntimeError(f"zenodo record fetch failed: {record}")


def zenodo(name: str, record: int) -> None:
    """Download every file of a Zenodo record, verifying the record's MD5 checksums.

    :param name: Corpus name (directory under the corpora root).
    :param record: Zenodo record id.
    :raises RuntimeError: A downloaded file does not match its published checksum.
    """
    directory = ROOT / name / "source"
    directory.mkdir(parents=True, exist_ok=True)
    data = _zenodo_record(record, directory / f"zenodo-record-{record}.json")
    files: list[dict[str, str]] = data["files"]  # type: ignore[assignment]
    records = []
    skip = SKIP.get(name)
    for entry in files:
        key = entry["key"]
        if skip is not None and skip(key):
            continue
        out = directory / key
        want = entry["checksum"].split(":", 1)[1]
        if not (out.exists() and md5(out) == want):
            download(f"https://zenodo.org/api/records/{record}/files/{key}/content", out)
            got = md5(out)
            if got != want:
                raise RuntimeError(f"md5 mismatch {key}: {got} != {want}")
        records.append(
            {
                "key": key,
                "size": out.stat().st_size,
                "zenodo_md5": want,
                "sha256": sha256(out),
                "fetched_at": now(),
            }
        )
        log.info("%s %s ok", name, key)
    (directory / "acquisition.json").write_text(
        json.dumps({"record": record, "doi": data.get("doi"), "files": records}, indent=2) + "\n"
    )


def direct(name: str, urls: list[str]) -> None:
    """Download plain archive URLs, recording response headers and hashes.

    :param name: Corpus name.
    :param urls: Archive URLs.
    """
    directory = ROOT / name / "source"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for url in urls:
        out = directory / url.rsplit("/", 1)[1]
        headers = subprocess.run(  # noqa: S603 — args are literal strings
            [executable("curl"), "-sIL", url],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if not out.exists():
            download(url, out)
        records.append(
            {
                "url": url,
                "file": out.name,
                "size": out.stat().st_size,
                "sha256": sha256(out),
                "fetched_at": now(),
                "response_headers": headers,
            }
        )
        log.info("%s %s ok", name, out.name)
    (directory / "acquisition.json").write_text(json.dumps({"files": records}, indent=2) + "\n")


def ashir() -> None:
    """Mirror the ASH-IR repository and check out the pinned revision."""
    directory = ROOT / "ASHIR"
    directory.mkdir(parents=True, exist_ok=True)
    mirror = directory / "source.git"
    checkout = directory / "source"
    if not mirror.exists():
        subprocess.run(  # noqa: S603 — args are literal strings
            [executable("git"), "clone", "--mirror", ASHIR_REPOSITORY, str(mirror)],
            check=True,
        )
    if not checkout.exists():
        subprocess.run(  # noqa: S603 — args are literal strings
            [
                executable("git"),
                f"--git-dir={mirror}",
                "worktree",
                "add",
                "--detach",
                str(checkout),
                ASHIR_REV,
            ],
            check=True,
        )
    (directory / "acquisition.json").write_text(
        json.dumps(
            {"repository": ASHIR_REPOSITORY, "revision": ASHIR_REV, "fetched_at": now()},
            indent=2,
        )
        + "\n"
    )
    log.info("ASHIR ok")


def openair() -> None:
    """Crawl the York OpenAIR file store recursively (openairlib.net is suspended)."""
    directory = ROOT / "OpenAIR" / "source"
    directory.mkdir(parents=True, exist_ok=True)
    started = now()
    subprocess.run(  # noqa: S603 — args are literal strings
        [
            executable("wget"),
            "-q",
            "-nH",
            "--cut-dirs=1",
            "-m",
            "-np",
            "-e",
            "robots=off",
            "-c",
            "--no-parent",
            "-P",
            str(directory),
            OPENAIR_STORE,
        ],
        check=False,
    )
    (ROOT / "OpenAIR" / "acquisition.json").write_text(
        json.dumps(
            {
                "url": OPENAIR_STORE,
                "method": "wget -m -np",
                "started_at": started,
                "finished_at": now(),
            },
            indent=2,
        )
        + "\n"
    )
    log.info("OpenAIR ok")


def main(argv: list[str]) -> None:
    """Dispatch one acquisition family.

    :param argv: ``[family, *names]`` where family is zenodo, direct, ashir, or openair.
    :raises SystemExit: The family is unknown.
    """
    family = argv[0]
    if family == "zenodo":
        only = argv[1:]
        for name, record in ZENODO.items():
            if not only or name in only:
                zenodo(name, record)
    elif family == "direct":
        for name, urls in DIRECT.items():
            direct(name, urls)
    elif family == "ashir":
        ashir()
    elif family == "openair":
        openair()
    else:
        raise SystemExit(f"unknown acquisition family: {family}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
