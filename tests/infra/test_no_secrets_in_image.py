"""Invariant 5: no secrets are baked into image layers or scripts.

Static checks: regex-scan Dockerfile and post-create.sh for `KEY = "literal..."`
assignments that look like real secrets. Variable references (`$VAR`, `${VAR}`)
and unset env-var defaults are fine — they don't bake values into layers.

Optional check: if `docker` is on PATH, run `docker history` on the base image
and grep for secret keywords. Skipped when docker is missing or the image
isn't present locally.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SECRET_KEYWORD_PATTERN = re.compile(
    r"(?i)(?:token|secret|password|api[_-]?key|access[_-]?key)\s*=\s*"
    r"['\"]([A-Za-z0-9_\-/+=]{16,})['\"]"
)

ALLOWLISTED_VARIABLE_NAMES: frozenset[str] = frozenset(
    {
        "RESTRICTED_AGENT_GIT_PAT",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)


def _read_base_image_from_dockerfile(dockerfile_path: Path) -> str:
    """Return the first `FROM` instruction's image reference from a Dockerfile."""
    for line in dockerfile_path.read_text().splitlines():
        match = re.match(r"^\s*FROM\s+(?P<image>\S+)", line, flags=re.IGNORECASE)
        if match:
            return match.group("image")
    raise RuntimeError(f"No FROM instruction found in {dockerfile_path}")


def _scan_for_baked_secrets(text: str, path_label: str) -> list[str]:
    """Return per-line findings where a secret-keyword assigns a literal value."""
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in SECRET_KEYWORD_PATTERN.finditer(line):
            value = match.group(1)
            if value in ALLOWLISTED_VARIABLE_NAMES:
                continue
            findings.append(f"{path_label}:{lineno}: {line.strip()!r} (matched {value!r})")
    return findings


@pytest.mark.infra
def test_dockerfile_contains_no_baked_secret_values_no_secrets_in_image(
    dockerfile_path: Path,
) -> None:
    """Dockerfile must not assign literal secret values to env vars/build args."""
    text = dockerfile_path.read_text()
    findings = _scan_for_baked_secrets(text, dockerfile_path.name)
    assert not findings, "Possible baked secrets in Dockerfile:\n" + "\n".join(findings)


@pytest.mark.infra
def test_post_create_contains_no_baked_secret_values_no_secrets_in_image(
    post_create_script: Path,
) -> None:
    """post-create.sh must not assign literal secret values."""
    text = post_create_script.read_text()
    findings = _scan_for_baked_secrets(text, post_create_script.name)
    assert not findings, "Possible baked secrets in post-create.sh:\n" + "\n".join(findings)


@pytest.mark.infra
def test_initialize_contains_no_baked_secret_values_no_secrets_in_image(
    initialize_script: Path,
) -> None:
    """initialize.sh must not assign literal secret values."""
    text = initialize_script.read_text()
    findings = _scan_for_baked_secrets(text, initialize_script.name)
    assert not findings, "Possible baked secrets in initialize.sh:\n" + "\n".join(findings)


@pytest.mark.infra
def test_docker_history_contains_no_secret_keywords_no_secrets_in_image(
    dockerfile_path: Path,
) -> None:
    """`docker history` for the Dockerfile's FROM image must not surface secret-keyword tokens."""
    if shutil.which("docker") is None:
        pytest.skip("'docker' binary not available on PATH")

    base_image = _read_base_image_from_dockerfile(dockerfile_path)

    inspect = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["docker", "image", "inspect", base_image],  # noqa: S607 — docker on PATH
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        pytest.skip(f"base image {base_image!r} not pulled locally")

    history = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [  # noqa: S607 — docker on PATH
            "docker",
            "history",
            "--no-trunc",
            "--format",
            "{{.CreatedBy}}",
            base_image,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert history.returncode == 0, (
        f"docker history failed: exit {history.returncode}\nstderr: {history.stderr}"
    )

    findings: list[str] = []
    for line in history.stdout.splitlines():
        for match in SECRET_KEYWORD_PATTERN.finditer(line):
            value = match.group(1)
            if value in ALLOWLISTED_VARIABLE_NAMES:
                continue
            findings.append(f"{line.strip()!r} (matched {value!r})")
    assert not findings, (
        f"`docker history {base_image}` surfaced possible secrets in layers:\n"
        + "\n".join(findings)
    )
