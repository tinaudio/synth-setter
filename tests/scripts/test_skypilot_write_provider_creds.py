"""Tests for scripts/skypilot_write_provider_creds.sh — provider cred bootstrap.

The script writes per-provider SkyPilot credentials before `sky check` /
`sky.launch`:

- R2 (always, regardless of provider): writes ``~/.cloudflare/r2.credentials``
  + ``~/.cloudflare/accountid`` for SkyPilot's R2 storage adaptor.
- RunPod (when --provider runpod): ``~/.runpod/config.toml``.
- OCI (when --provider oci): ``~/.oci/config`` + ``~/.oci/oci_api_key.pem`` +
  upserts the ``oci:`` block into ``~/.sky/config.yaml``.
- Local (when --provider local): upserts the ``jobs:`` (controller resources)
  block into ``~/.sky/config.yaml`` so the managed-jobs controller pod fits
  on the kind cluster ``sky local up`` provisions in CI.

``~/.sky/config.yaml`` is shared by both OCI and local: writes are per-key
upserts so running for one provider after another preserves the other's keys.

Tests run the script in a tmp ``HOME`` so they exercise the real bash without
touching the developer's actual cred files. **Critical no-leak invariant**:
the script must emit zero bytes on stdout. Tests pin this so future edits
can't reintroduce a leak surface.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "skypilot_write_provider_creds.sh"

R2_ENV: dict[str, str] = {
    "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "AK_TEST",
    "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "SK_TEST",
    "RCLONE_CONFIG_R2_ENDPOINT": "https://acct-test.r2.cloudflarestorage.com",
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
# No-leak invariant — script must emit zero bytes on stdout, regardless of
# provider. This is the structural guard against the secret-leak class of
# regressions: any future edit that prints cred values to stdout will fail
# this test.
# ---------------------------------------------------------------------------


class TestNoStdoutLeak:
    """Stdout is empty under every successful provider invocation."""

    def test_runpod_emits_zero_stdout_bytes(self, tmp_path: Path) -> None:
        """RunPod-mode invocation emits zero bytes on stdout (no leak surface)."""
        result = _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert result.stdout == ""

    def test_oci_emits_zero_stdout_bytes(self, tmp_path: Path) -> None:
        """OCI-mode invocation emits zero bytes on stdout (no leak surface)."""
        result = _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")
        assert result.stdout == ""

    def test_local_emits_zero_stdout_bytes(self, tmp_path: Path) -> None:
        """Local-mode invocation emits zero bytes on stdout (no leak surface)."""
        result = _run(tmp_path, R2_ENV, "--provider", "local")
        assert result.stdout == ""

    def test_no_secret_byte_appears_on_stderr_either(self, tmp_path: Path) -> None:
        """Stderr is for status/notice messages only — never for cred values."""
        result = _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        for value in (
            R2_ENV["RCLONE_CONFIG_R2_ACCESS_KEY_ID"],
            R2_ENV["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"],
            RUNPOD_ENV["RUNPOD_API_KEY"],
        ):
            assert value not in result.stderr, (
                f"secret value {value!r} leaked to stderr: {result.stderr!r}"
            )

    def test_oci_compartment_ocid_never_passed_as_subprocess_argv(self, tmp_path: Path) -> None:
        """The `upsert_sky_config_key` helper must NOT pass a secret-bearing YAML fragment to
        ``python3`` via argv. ``/proc/<pid>/cmdline`` is world-readable on Linux, so a secret
        carried in argv can be observed by any other user on the runner via ``ps``. Stdin/env-var
        carriage is required.

        Test approach: shim ``python3`` with a wrapper that logs its argv to a
        file before exec'ing the real interpreter; run ``--provider oci``; assert
        the OCI_COMPARTMENT_OCID value never appears anywhere in the logged argv.
        """
        real_python = shutil.which("python3") or sys.executable
        log_path = tmp_path / "argv.log"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        shim = bin_dir / "python3"
        shim.write_text(
            "#!/usr/bin/env bash\n"
            f'{{ printf "argv:\\n"; printf "  %s\\n" "$@"; echo "---"; }} >> {log_path}\n'
            f'exec {real_python} "$@"\n'
        )
        shim.chmod(0o755)

        env_overrides = {
            **R2_ENV,
            **OCI_ENV,
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        }
        _run(tmp_path, env_overrides, "--provider", "oci")

        logged = log_path.read_text() if log_path.exists() else ""
        # Sanity: shim was actually invoked (otherwise the test would pass vacuously).
        assert "argv:" in logged, f"python3 shim was not invoked: log={logged!r}"
        assert OCI_ENV["OCI_COMPARTMENT_OCID"] not in logged, (
            f"OCI_COMPARTMENT_OCID leaked into a subprocess argv: {logged!r}"
        )


# ---------------------------------------------------------------------------
# R2 cred writing — runs unconditionally regardless of provider
# ---------------------------------------------------------------------------


class TestR2CredentialsFile:
    """R2 cred-file writes: contents, modes, and unconditional invocation across providers."""

    def test_writes_credentials_file_with_aws_keys(self, tmp_path: Path) -> None:
        """R2 credentials file contains the AWS-style profile with the supplied access key +
        secret."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        creds = tmp_path / ".cloudflare" / "r2.credentials"
        assert creds.is_file()
        content = creds.read_text()
        assert "[r2]" in content
        assert f"aws_access_key_id = {R2_ENV['RCLONE_CONFIG_R2_ACCESS_KEY_ID']}" in content
        assert f"aws_secret_access_key = {R2_ENV['RCLONE_CONFIG_R2_SECRET_ACCESS_KEY']}" in content

    def test_credentials_file_is_mode_600(self, tmp_path: Path) -> None:
        """R2 credentials file is mode 600 (explicit chmod, not umask-dependent)."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert _file_mode(tmp_path / ".cloudflare" / "r2.credentials") == 0o600

    def test_writes_accountid_file(self, tmp_path: Path) -> None:
        """R2 account-id file contains the account ID and is mode 600."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        accountid = tmp_path / ".cloudflare" / "accountid"
        assert accountid.is_file()
        assert R2_ENV["R2_ACCOUNT_ID"] in accountid.read_text()
        assert _file_mode(accountid) == 0o600

    def test_runs_unconditionally_for_oci_provider(self, tmp_path: Path) -> None:
        """R2 cred files are written even when --provider is oci (R2 is shared across
        providers)."""
        _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")
        assert (tmp_path / ".cloudflare" / "r2.credentials").is_file()
        assert (tmp_path / ".cloudflare" / "accountid").is_file()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Existing non-empty cred files are preserved by default and overwritten with --force."""

    def test_existing_non_empty_file_preserved(self, tmp_path: Path) -> None:
        """An existing non-empty cred file is left alone unless --force is passed."""
        creds = tmp_path / ".cloudflare" / "r2.credentials"
        creds.parent.mkdir(parents=True)
        creds.write_text("HAND_MANAGED\n")
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert creds.read_text() == "HAND_MANAGED\n"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """Passing --force overwrites the existing cred file."""
        creds = tmp_path / ".cloudflare" / "r2.credentials"
        creds.parent.mkdir(parents=True)
        creds.write_text("HAND_MANAGED\n")
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod", "--force")
        assert "[r2]" in creds.read_text()

    def test_skip_path_tightens_loose_permissions(self, tmp_path: Path) -> None:
        """A pre-existing cred file with mode 0644 should be tightened to 0600 even when its
        contents are preserved (no-leak posture: never leave creds world-readable)."""
        creds = tmp_path / ".cloudflare" / "r2.credentials"
        creds.parent.mkdir(parents=True)
        creds.write_text("HAND_MANAGED\n")
        creds.chmod(0o644)
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        assert creds.read_text() == "HAND_MANAGED\n"
        assert _file_mode(creds) == 0o600


# ---------------------------------------------------------------------------
# Provider gating
# ---------------------------------------------------------------------------


class TestProviderGating:
    """`--provider <name>` gates which compute-cred files (if any) get written."""

    def test_runpod_writes_runpod_config(self, tmp_path: Path) -> None:
        """RunPod provider writes ~/.runpod/config.toml with the supplied API key (mode 600)."""
        _run(tmp_path, {**R2_ENV, **RUNPOD_ENV}, "--provider", "runpod")
        config = tmp_path / ".runpod" / "config.toml"
        assert config.is_file()
        assert "rp-test-key" in config.read_text()
        assert _file_mode(config) == 0o600

    def test_oci_writes_oci_config_and_key(self, tmp_path: Path) -> None:
        """OCI provider writes the three OCI cred files (config, oci_api_key.pem, sky/config.yaml)
        all mode 600."""
        _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")
        assert (tmp_path / ".oci" / "config").is_file()
        assert (tmp_path / ".oci" / "oci_api_key.pem").is_file()
        assert (tmp_path / ".sky" / "config.yaml").is_file()
        assert _file_mode(tmp_path / ".oci" / "config") == 0o600
        assert _file_mode(tmp_path / ".oci" / "oci_api_key.pem") == 0o600
        assert _file_mode(tmp_path / ".sky" / "config.yaml") == 0o600

    def test_local_writes_r2_files_and_shrunken_sky_jobs_controller_config(
        self, tmp_path: Path
    ) -> None:
        """Local provider writes the R2 cred files plus a `~/.sky/config.yaml` that shrinks the
        managed-jobs controller's default resource request so the controller pod schedules on the
        kind cluster `sky local up` provisions in CI.

        The default `cpus=4+, mem=4x, disk_size=50` does not fit on kind (k8s rejects
        `disk_size`; CPU/memory floor exceeds the runner's pod-level capacity).
        """
        _run(tmp_path, R2_ENV, "--provider", "local")
        assert (tmp_path / ".cloudflare" / "r2.credentials").is_file()
        assert (tmp_path / ".cloudflare" / "accountid").is_file()
        assert not (tmp_path / ".runpod").exists()
        assert not (tmp_path / ".oci").exists()

        sky_config = tmp_path / ".sky" / "config.yaml"
        assert sky_config.is_file()
        assert _file_mode(sky_config) == 0o600
        config_text = sky_config.read_text()
        assert "jobs:" in config_text
        assert "controller:" in config_text
        # Resource floor must be lower than the kind cluster's per-pod ceiling — the values
        # don't matter precisely, but `4+` (the SkyPilot default) must NOT appear, and
        # `disk_size` must not be set (k8s rejects it).
        assert "4+" not in config_text
        assert "disk_size" not in config_text

    def test_local_does_not_require_compute_provider_env(self, tmp_path: Path) -> None:
        """Local provider succeeds with R2 vars alone — no RUNPOD_API_KEY, no OCI_*."""
        result = _run(tmp_path, R2_ENV, "--provider", "local")
        assert result.returncode == 0

    def test_oci_then_local_preserves_both_keys_in_sky_config(self, tmp_path: Path) -> None:
        """`~/.sky/config.yaml` is shared by OCI (`oci:` block) and local (`jobs:` block).

        Running `--provider oci` then `--provider local` must end with both top-level keys
        present — neither write may clobber the other. Defends the multi-provider local-dev
        flow from regressing back to "first writer wins, second is silently skipped".
        """
        import yaml as _yaml

        _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")
        _run(tmp_path, R2_ENV, "--provider", "local")

        sky_config = tmp_path / ".sky" / "config.yaml"
        assert sky_config.is_file()
        assert _file_mode(sky_config) == 0o600
        doc = _yaml.safe_load(sky_config.read_text())
        assert "oci" in doc, f"OCI key clobbered by local write: {doc!r}"
        assert "jobs" in doc, f"local key missing after OCI write: {doc!r}"
        assert doc["oci"]["default"]["compartment_ocid"] == OCI_ENV["OCI_COMPARTMENT_OCID"]
        assert doc["jobs"]["controller"]["resources"]["cpus"] == "1+"

    def test_local_then_oci_preserves_both_keys_in_sky_config(self, tmp_path: Path) -> None:
        """Symmetric to test_oci_then_local: order doesn't matter — both keys must coexist."""
        import yaml as _yaml

        _run(tmp_path, R2_ENV, "--provider", "local")
        _run(tmp_path, {**R2_ENV, **OCI_ENV}, "--provider", "oci")

        sky_config = tmp_path / ".sky" / "config.yaml"
        doc = _yaml.safe_load(sky_config.read_text())
        assert "oci" in doc and "jobs" in doc, f"local→oci write order dropped a key: {doc!r}"

    def test_malformed_sky_config_fails_with_clear_error(self, tmp_path: Path) -> None:
        """A pre-existing ``~/.sky/config.yaml`` whose top level is not a mapping (e.g. a list,
        left over from a hand-edit gone wrong) must fail the upsert with a clear, named error — not
        a bare ``TypeError`` traceback from ``existing[key] = ...``."""
        sky_dir = tmp_path / ".sky"
        sky_dir.mkdir()
        sky_config = sky_dir / "config.yaml"
        sky_config.write_text("- accidentally\n- a list\n")

        result = _run(tmp_path, R2_ENV, "--provider", "local", expect_success=False)
        assert result.returncode != 0
        assert "not a YAML mapping" in result.stderr
        assert str(sky_config) in result.stderr
        assert "Traceback" not in result.stderr

    def test_unknown_provider_fails(self, tmp_path: Path) -> None:
        """An unknown --provider value (e.g. aws) is rejected with a clear error."""
        result = _run(tmp_path, R2_ENV, "--provider", "aws", expect_success=False)
        assert result.returncode != 0
        assert "aws" in result.stderr.lower() or "unknown" in result.stderr.lower()

    def test_missing_provider_fails(self, tmp_path: Path) -> None:
        """Invoking the script without --provider exits non-zero with a clear error."""
        result = _run(tmp_path, R2_ENV, expect_success=False)
        assert result.returncode != 0

    def test_provider_flag_without_value_fails_cleanly(self, tmp_path: Path) -> None:
        """`--provider` as the trailing arg (no value) fails with a clear error rather than a bash
        `shift` failure under `set -e`."""
        result = _run(tmp_path, R2_ENV, "--provider", expect_success=False)
        assert result.returncode != 0
        assert "--provider" in result.stderr
        assert "value" in result.stderr.lower() or "required" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Required-var validation
# ---------------------------------------------------------------------------


class TestRequiredVarValidation:
    """Per-provider env vars are validated (`require_var`) before any file is written."""

    def test_runpod_missing_api_key_fails(self, tmp_path: Path) -> None:
        """RunPod mode without RUNPOD_API_KEY exits non-zero and names the missing var."""
        result = _run(tmp_path, R2_ENV, "--provider", "runpod", expect_success=False)
        assert result.returncode != 0
        assert "RUNPOD_API_KEY" in result.stderr

    def test_oci_missing_user_ocid_fails(self, tmp_path: Path) -> None:
        """OCI mode without OCI_USER_OCID exits non-zero and names the missing var."""
        env = {**R2_ENV, **OCI_ENV}
        del env["OCI_USER_OCID"]
        result = _run(tmp_path, env, "--provider", "oci", expect_success=False)
        assert result.returncode != 0
        assert "OCI_USER_OCID" in result.stderr


@pytest.mark.parametrize(
    "missing_key",
    [
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
        "RCLONE_CONFIG_R2_ENDPOINT",
        "R2_ACCOUNT_ID",
    ],
)
def test_missing_r2_var_fails(tmp_path: Path, missing_key: str) -> None:
    """Each R2 var is required (no graceful degradation — R2 is always needed)."""
    env = {**R2_ENV, **RUNPOD_ENV}
    del env[missing_key]
    result = _run(tmp_path, env, "--provider", "runpod", expect_success=False)
    assert result.returncode != 0
    assert missing_key in result.stderr
