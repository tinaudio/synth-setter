"""Invariant 2: `gh auth status` works inside the container without a stale `.env`.

The auth flow in post-create.sh must use the env-passed `RESTRICTED_AGENT_GIT_PAT`
and NOT source `.env` (which would override a fresh token with a stale value).
initialize.sh must only `touch` `.env` when absent — never overwrite contents.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TOKEN_ENV_VAR = "RESTRICTED_AGENT_GIT_PAT"  # noqa: S105 — env-var name, not a password value


@pytest.mark.infra
def test_post_create_references_token_env_var_for_gh_auth_flow(
    post_create_script: Path,
) -> None:
    """post-create.sh must reference RESTRICTED_AGENT_GIT_PAT for gh auth."""
    text = post_create_script.read_text()
    assert TOKEN_ENV_VAR in text, (
        f"post-create.sh must reference {TOKEN_ENV_VAR} to authenticate gh"
    )


@pytest.mark.infra
def test_post_create_does_not_source_dotenv_for_gh_auth_flow(
    post_create_script: Path,
) -> None:
    """post-create.sh must NOT `source .env`; that would clobber the env-passed token."""
    text = post_create_script.read_text()
    dotenv_arg = r"""['"]?(?:\.?/)?(?:\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/)?\.env['"]?"""
    forbidden_patterns = [
        rf"^\s*source\s+{dotenv_arg}\b",
        rf"^\s*\.\s+{dotenv_arg}\b",
    ]
    for pattern in forbidden_patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        assert match is None, (
            f"post-create.sh must not source .env (would override the env-passed "
            f"{TOKEN_ENV_VAR}); matched pattern {pattern!r} at: {match.group(0) if match else ''!r}"
        )


@pytest.mark.infra
def test_post_create_pipes_token_to_gh_auth_login_for_gh_auth_flow(
    post_create_script: Path,
) -> None:
    """post-create.sh must pipe the token to `gh auth login --with-token`."""
    text = post_create_script.read_text()
    assert "gh auth login --with-token" in text, (
        "post-create.sh must call `gh auth login --with-token` to register the token"
    )
    assert "gh auth setup-git" in text, (
        "post-create.sh must call `gh auth setup-git` after login so git uses the token"
    )


@pytest.mark.infra
def test_post_create_strips_quotes_around_token_for_gh_auth_flow(
    post_create_script: Path,
) -> None:
    """post-create.sh must strip surrounding quotes from the token value.

    Docker's `--env-file` does not strip quotes the way `source` would, so
    tokens read from `.env` lines like `KEY="value"` carry the quotes through.
    """
    text = post_create_script.read_text()
    has_double_quote_strip = f"{TOKEN_ENV_VAR}=" in text and '%\\"' in text and '#\\"' in text
    has_single_quote_strip = "%\\'" in text and "#\\'" in text
    assert has_double_quote_strip and has_single_quote_strip, (
        "post-create.sh must strip both single and double quotes around the token "
        f"({TOKEN_ENV_VAR}) before piping it to `gh auth login`"
    )


@pytest.mark.infra
def test_initialize_only_touches_dotenv_for_gh_auth_flow(
    initialize_script: Path,
) -> None:
    """initialize.sh must only `touch .env` when absent — never write or overwrite."""
    text = initialize_script.read_text()
    assert re.search(r"\btouch\s+\.env\b", text), (
        "initialize.sh must `touch .env` so Docker --env-file finds the file"
    )
    forbidden = [
        r">\s*\.env\b",
        r">>\s*\.env\b",
        r"^\s*cp\s+.*\s+\.env\b",
        r"^\s*mv\s+.*\s+\.env\b",
        r"tee\s+.*\.env\b",
        r"cat\s+.*>\s*\.env\b",
    ]
    for pattern in forbidden:
        match = re.search(pattern, text, flags=re.MULTILINE)
        assert match is None, (
            f"initialize.sh must not overwrite .env (would clobber existing credentials); "
            f"matched pattern {pattern!r}: {match.group(0) if match else ''!r}"
        )
