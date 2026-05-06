"""Tests for scripts/skypilot_write_provider_creds.sh — provider cred bootstrap.

The script writes per-provider SkyPilot credentials before `sky check` / `sky.launch`:

- R2 (always, regardless of provider): bridges bare ``R2_*`` env vars into
  rclone-prefixed ``RCLONE_CONFIG_R2_*`` form (printed to stdout for the launcher
  to source) AND writes ``~/.cloudflare/r2.credentials`` + ``~/.cloudflare/accountid``
  for SkyPilot's R2 file_mounts adaptor.
- RunPod (when --provider runpod): ``~/.runpod/config.toml``.
- OCI (when --provider oci): ``~/.oci/config`` + ``~/.oci/oci_api_key.pem`` +
  ``~/.sky/config.yaml``.

Tests run the script in a tmp ``HOME`` so they exercise the real bash without
touching the developer's actual cred files. Stdout is captured to verify the
rclone env-var bridging output the launcher will source.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "skypilot_write_provider_creds.sh"

R2_ENV: dict[str, str] = {
    "R2_ACCESS_KEY_ID": "AK_TEST",
    "R2_SECRET_ACCESS_KEY": "SK_TEST",
    "R2_ENDPOINT": "https://acct-test.r2.cloudflarestorage.com",
    "R2_ACCOUNT_ID": "acct-test-id",
}

RUNPOD_ENV: dict[str, str] = {"RUNPOD_API_KEY": "rp-test-key"}

# Build a fake PEM at runtime — the source must not contain literal PEM
# headers, otherwise the detect-private-key pre-commit hook flags this file.
_PEM_HEADER = "-----BEGIN " + "PRIVATE" + " KEY-----"
_PEM_FOOTER = "-----END " + "PRIVATE" + " KEY-----"
_FAKE_PEM = f"{_PEM_HEADER}\nFAKE\n{_PEM_FOOTER}"

OCI_ENV: dict[str, str] = {
    "OCI_USER_OCID": "ocid1.user.oc1..xxxx",
    "OCI_TENANCY_OCID": "ocid1.tenancy.oc1..yyyy",
    "OCI_FINGERPRINT": "aa:bb:cc:dd",
    "OCI_REGION": "us-ashburn-1",
    "OCI_COMPARTMENT_OCID": "ocid1.compartment.oc1..zzzz",
    "OCI_API_KEY_PEM": _FAKE_PEM,
}


def _run(
    home: Path,
    env_extra: dict[str, str],
    *args: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run the bootstrap script in a clean env with HOME=home and the given extras."""
    base_env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
    }
    base_env.update(env_extra)
    result = subprocess.run(  # noqa: S603 — controlled args, hermetic env
        ["bash", str(SCRIPT), *args],  # noqa: S607 — bash on PATH
        env=base_env,
        capture_output=True,
        text=True,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"script failed rc={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _file_mode(path: Path) -> int:
    """Return the permission bits (lower 9) of ``path`` as an int."""
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# R2 cred writing — runs unconditionally regardless of provider
# ---------------------------------------------------------------------------


class TestR2CredentialsFile:
    """``~/.cloudflare/r2.credentials`` + ``~/.cloudflare/accountid`` are written for SkyPilot."""

    def test_writes_credentials_file_with_aws_keys(self, tmp_path: Path) -> None:
        """The credentials file uses the ``[r2]`` profile with AWS-style key names so SkyPilot's R2
        adaptor (`sky/adaptors/cloudflare.py`) can boto-load it."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")

        creds = (tmp_path / ".cloudflare" / "r2.credentials").read_text()
        assert "[r2]" in creds
        assert "aws_access_key_id = AK_TEST" in creds
        assert "aws_secret_access_key = SK_TEST" in creds

    def test_credentials_file_is_mode_600(self, tmp_path: Path) -> None:
        """File contains long-lived secrets — must be mode 600 (owner read/write only)."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert _file_mode(tmp_path / ".cloudflare" / "r2.credentials") == 0o600

    def test_writes_accountid_file_plain_text(self, tmp_path: Path) -> None:
        """``~/.cloudflare/accountid`` holds the bare account id (no INI/JSON wrapping)."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert (tmp_path / ".cloudflare" / "accountid").read_text().strip() == "acct-test-id"

    def test_accountid_file_is_mode_600(self, tmp_path: Path) -> None:
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert _file_mode(tmp_path / ".cloudflare" / "accountid") == 0o600

    def test_runs_unconditionally_for_oci_provider(self, tmp_path: Path) -> None:
        """R2 is shared across compute providers; OCI runs still get R2 creds written."""
        _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")
        assert (tmp_path / ".cloudflare" / "r2.credentials").is_file()
        assert (tmp_path / ".cloudflare" / "accountid").is_file()

    def test_missing_r2_account_id_fails(self, tmp_path: Path) -> None:
        """``R2_ACCOUNT_ID`` is required and not derivable from other R2 vars."""
        env = {**R2_ENV, **RUNPOD_ENV}
        del env["R2_ACCOUNT_ID"]
        result = _run(tmp_path, env, "--provider", "runpod", expect_success=False)
        assert result.returncode != 0
        assert "R2_ACCOUNT_ID" in result.stderr


class TestR2RcloneEnvBridging:
    """Stdout emits ``RCLONE_CONFIG_R2_*=...`` lines the launcher sources/parses."""

    def test_stdout_emits_rclone_prefixed_keys(self, tmp_path: Path) -> None:
        """The launcher reads stdout to populate rclone env vars in its own subprocess env.

        Format: one ``KEY=VALUE`` line per rclone-prefixed key, no quoting (consumers
        treat values literally).
        """
        result = _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        lines = set(result.stdout.splitlines())
        assert "RCLONE_CONFIG_R2_TYPE=s3" in lines
        assert "RCLONE_CONFIG_R2_PROVIDER=Cloudflare" in lines
        assert "RCLONE_CONFIG_R2_ACCESS_KEY_ID=AK_TEST" in lines
        assert "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=SK_TEST" in lines
        assert "RCLONE_CONFIG_R2_ENDPOINT=https://acct-test.r2.cloudflarestorage.com" in lines

    def test_stdout_only_emits_env_lines_no_log_chatter(self, tmp_path: Path) -> None:
        """Diagnostic output goes to stderr; stdout must be machine-parseable KEY=VALUE only."""
        result = _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        for line in result.stdout.splitlines():
            assert "=" in line, f"non-KEY=VALUE line on stdout: {line!r}"


# ---------------------------------------------------------------------------
# Idempotency — don't clobber operator-managed cred files
# ---------------------------------------------------------------------------


class TestIdempotency:
    """The bootstrap must not silently overwrite existing non-empty cred files.

    Local-dev operators frequently hand-manage ``~/.cloudflare/r2.credentials`` and
    ``~/.oci/config``; a re-run that clobbers them is a regression. Re-runs must
    no-op when files already exist non-empty unless ``--force`` is passed.
    """

    def test_existing_r2_credentials_file_is_not_overwritten(self, tmp_path: Path) -> None:
        """A pre-existing non-empty ``~/.cloudflare/r2.credentials`` survives a re-run."""
        cf_dir = tmp_path / ".cloudflare"
        cf_dir.mkdir()
        existing = cf_dir / "r2.credentials"
        existing.write_text("[r2]\naws_access_key_id = MANUALLY_MANAGED\n")
        existing.chmod(0o600)
        (cf_dir / "accountid").write_text("manual-acct-id\n")
        (cf_dir / "accountid").chmod(0o600)

        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")

        assert "MANUALLY_MANAGED" in existing.read_text()

    def test_force_flag_overwrites_existing_r2_credentials_file(self, tmp_path: Path) -> None:
        """``--force`` is the explicit opt-in for clobbering hand-managed files."""
        cf_dir = tmp_path / ".cloudflare"
        cf_dir.mkdir()
        existing = cf_dir / "r2.credentials"
        existing.write_text("[r2]\naws_access_key_id = MANUALLY_MANAGED\n")
        existing.chmod(0o600)

        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod", "--force")

        contents = existing.read_text()
        assert "MANUALLY_MANAGED" not in contents
        assert "AK_TEST" in contents

    def test_existing_runpod_config_is_not_overwritten(self, tmp_path: Path) -> None:
        runpod_dir = tmp_path / ".runpod"
        runpod_dir.mkdir()
        config = runpod_dir / "config.toml"
        config.write_text('[default]\napi_key = "MANUALLY_MANAGED"\n')
        config.chmod(0o600)

        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")

        assert "MANUALLY_MANAGED" in config.read_text()

    def test_empty_existing_file_is_overwritten(self, tmp_path: Path) -> None:
        """A zero-byte file from a previous failed run should not block the bootstrap."""
        cf_dir = tmp_path / ".cloudflare"
        cf_dir.mkdir()
        empty = cf_dir / "r2.credentials"
        empty.touch()

        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")

        assert "AK_TEST" in empty.read_text()

    def test_repeat_run_on_clean_home_is_idempotent(self, tmp_path: Path) -> None:
        """Two clean-home runs produce identical files — the bootstrap is deterministic."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        first = (tmp_path / ".cloudflare" / "r2.credentials").read_text()
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        second = (tmp_path / ".cloudflare" / "r2.credentials").read_text()
        assert first == second


# ---------------------------------------------------------------------------
# Provider gating — RunPod vs OCI is gated; R2 is always written
# ---------------------------------------------------------------------------


class TestProviderGating:
    """Only the per-provider compute-cred write is gated on ``--provider``; R2 always runs."""

    def test_runpod_provider_writes_runpod_config(self, tmp_path: Path) -> None:
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        config = tmp_path / ".runpod" / "config.toml"
        assert config.is_file()
        assert "rp-test-key" in config.read_text()

    def test_runpod_provider_does_not_write_oci_config(self, tmp_path: Path) -> None:
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert not (tmp_path / ".oci" / "config").exists()

    def test_oci_provider_writes_oci_config_and_key(self, tmp_path: Path) -> None:
        _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")
        assert (tmp_path / ".oci" / "config").is_file()
        assert (tmp_path / ".oci" / "oci_api_key.pem").is_file()
        assert (tmp_path / ".sky" / "config.yaml").is_file()

    def test_oci_provider_does_not_write_runpod_config(self, tmp_path: Path) -> None:
        _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")
        assert not (tmp_path / ".runpod" / "config.toml").exists()

    def test_unknown_provider_fails(self, tmp_path: Path) -> None:
        result = _run(tmp_path, R2_ENV, "--provider", "aws", expect_success=False)
        assert result.returncode != 0
        assert "aws" in result.stderr.lower() or "unknown" in result.stderr.lower()

    def test_missing_provider_fails(self, tmp_path: Path) -> None:
        result = _run(tmp_path, R2_ENV, expect_success=False)
        assert result.returncode != 0


class TestRequiredVarValidation:
    """Each provider's required vars are validated before any file is written."""

    def test_runpod_missing_api_key_fails(self, tmp_path: Path) -> None:
        result = _run(tmp_path, R2_ENV, "--provider", "runpod", expect_success=False)
        assert result.returncode != 0
        assert "RUNPOD_API_KEY" in result.stderr

    def test_oci_missing_user_ocid_fails(self, tmp_path: Path) -> None:
        env = {**R2_ENV, **OCI_ENV}
        del env["OCI_USER_OCID"]
        result = _run(tmp_path, env, "--provider", "oci", expect_success=False)
        assert result.returncode != 0
        assert "OCI_USER_OCID" in result.stderr


@pytest.mark.parametrize(
    "missing_key", ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_ACCOUNT_ID"]
)
def test_missing_r2_var_fails(tmp_path: Path, missing_key: str) -> None:
    """Each R2 var is required (no graceful degradation — R2 is always needed)."""
    env = {**R2_ENV, **RUNPOD_ENV}
    del env[missing_key]
    result = _run(tmp_path, env, "--provider", "runpod", expect_success=False)
    assert result.returncode != 0
    assert missing_key in result.stderr
