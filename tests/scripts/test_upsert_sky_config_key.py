"""Unit tests for ``scripts.upsert_sky_config_key`` — the per-key YAML upsert helper.

The module owns the parser branches (UTF-8 / YAML / mapping checks) that the
bash entry point ``scripts/skypilot_write_provider_creds.sh::upsert_sky_config_key``
delegates to. These tests exercise the contract directly via Python imports so
each branch is covered by a focused unit test, not a subprocess invocation.

End-to-end subprocess coverage of the bash wrapper still lives in
``tests/scripts/test_skypilot_write_provider_creds.py`` — those tests pin the
bash entry point's external contract; these tests pin the Python internals.
Both must pass simultaneously.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.upsert_sky_config_key import UpsertError, upsert_sky_config_key

MODULE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "upsert_sky_config_key.py"
)


def _file_mode(path: Path) -> int:
    """Return the permission bits (lower 9) of ``path`` as an int.

    :param path: Filesystem path whose mode bits to extract.
    :returns: permission bits (``stat.S_IMODE`` of ``path.stat().st_mode``).
    :rtype: int
    """
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestUpsertHappyPath:
    """Happy-path semantics: file is created/updated and other top-level keys are preserved."""

    def test_writes_fragment_when_file_does_not_exist(self, tmp_path: Path) -> None:
        """Upserting into a non-existent path creates the file with just the fragment's mapping.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        upsert_sky_config_key("oci", "oci:\n  default:\n    compartment_ocid: ocid-z\n", path)
        doc = yaml.safe_load(path.read_text())
        assert doc == {"oci": {"default": {"compartment_ocid": "ocid-z"}}}

    def test_writes_fragment_when_file_is_empty(self, tmp_path: Path) -> None:
        """An empty pre-existing file is populated with just the fragment's mapping.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text("")
        upsert_sky_config_key(
            "jobs", "jobs:\n  controller:\n    resources:\n      cpus: 1+\n", path
        )
        doc = yaml.safe_load(path.read_text())
        assert doc == {"jobs": {"controller": {"resources": {"cpus": "1+"}}}}

    def test_preserves_other_top_level_keys(self, tmp_path: Path) -> None:
        """Keys other than the managed one are preserved verbatim across the upsert.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text(
            "oci:\n  default:\n    compartment_ocid: ocid-existing\nother_key: leave-me\n"
        )
        upsert_sky_config_key(
            "jobs", "jobs:\n  controller:\n    resources:\n      cpus: 1+\n", path
        )
        doc = yaml.safe_load(path.read_text())
        assert doc["oci"] == {"default": {"compartment_ocid": "ocid-existing"}}
        assert doc["other_key"] == "leave-me"
        assert doc["jobs"] == {"controller": {"resources": {"cpus": "1+"}}}

    def test_replaces_managed_key_wholesale(self, tmp_path: Path) -> None:
        """The managed top-level key is replaced wholesale — nested keys under the managed key that
        were hand-added (and not present in the fragment) are NOT preserved.

        This pins the documented contract from PR #876 commit 89cc74a: within the managed top-level
        key the entire mapping is replaced; the merge semantics are intentionally coarse-grained
        (top-level-key granularity, not leaf-merge).

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text(
            "oci:\n"
            "  default:\n"
            "    compartment_ocid: ocid-existing\n"
            "    hand_added_nested: should-disappear\n"
        )
        upsert_sky_config_key("oci", "oci:\n  default:\n    compartment_ocid: ocid-new\n", path)
        doc = yaml.safe_load(path.read_text())
        assert doc["oci"] == {"default": {"compartment_ocid": "ocid-new"}}
        assert "hand_added_nested" not in doc["oci"]["default"]


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------


class TestFileMode:
    """The output file lands at mode 0o600 regardless of pre-existing perms or the umask."""

    def test_file_mode_after_write_is_0600(self, tmp_path: Path) -> None:
        """A freshly-written file lands at mode 0o600 (explicit chmod, not umask-dependent).

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        upsert_sky_config_key("jobs", "jobs:\n  controller: {}\n", path)
        assert _file_mode(path) == 0o600

    def test_file_mode_tightens_when_existing_was_world_readable(self, tmp_path: Path) -> None:
        """A pre-existing file with looser perms is tightened to 0o600 after upsert.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text("oci:\n  default:\n    compartment_ocid: x\n")
        path.chmod(0o644)
        upsert_sky_config_key("oci", "oci:\n  default:\n    compartment_ocid: y\n", path)
        assert _file_mode(path) == 0o600

    def test_file_lands_at_0600_under_loose_umask(self, tmp_path: Path) -> None:
        """Even with a permissive umask (0o022 — world-readable default), the output is 0o600.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        prev_umask = os.umask(0o022)
        try:
            upsert_sky_config_key("jobs", "jobs:\n  controller: {}\n", path)
        finally:
            os.umask(prev_umask)
        assert _file_mode(path) == 0o600


# ---------------------------------------------------------------------------
# Error paths — pre-existing file is malformed
# ---------------------------------------------------------------------------


class TestExistingFileErrors:
    """Pre-existing file is unreadable / unparsable / not-a-mapping — raise UpsertError."""

    def test_non_utf8_existing_file_raises(self, tmp_path: Path) -> None:
        """A pre-existing file containing non-UTF-8 bytes raises UpsertError with a clear reason.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_bytes(b"\xff\xfe oci: foo\n")
        with pytest.raises(UpsertError, match="not valid UTF-8"):
            upsert_sky_config_key("oci", "oci: {}\n", path)

    def test_unparsable_yaml_existing_file_raises(self, tmp_path: Path) -> None:
        """A pre-existing file whose contents fail ``yaml.safe_load`` raises UpsertError.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text("oci: [unterminated\n")
        with pytest.raises(UpsertError, match="not valid YAML"):
            upsert_sky_config_key("oci", "oci: {}\n", path)

    def test_non_mapping_top_level_existing_file_raises(self, tmp_path: Path) -> None:
        """A pre-existing file whose top-level YAML is not a mapping (e.g. a list) raises.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text("- accidentally\n- a list\n")
        with pytest.raises(UpsertError, match="not a YAML mapping"):
            upsert_sky_config_key("oci", "oci: {}\n", path)

    def test_error_message_includes_key_prefix(self, tmp_path: Path) -> None:
        """Every error message starts with ``upsert_sky_config_key[<key>]:`` for grep-ability.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text("- a list\n")
        with pytest.raises(UpsertError, match=r"^upsert_sky_config_key\[oci\]:"):
            upsert_sky_config_key("oci", "oci: {}\n", path)

    def test_error_message_includes_path(self, tmp_path: Path) -> None:
        """Error messages mention the offending path so operators can locate the file.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        path.write_text("- a list\n")
        with pytest.raises(UpsertError, match=str(path)):
            upsert_sky_config_key("oci", "oci: {}\n", path)


# ---------------------------------------------------------------------------
# Error paths — fragment is malformed
# ---------------------------------------------------------------------------


class TestFragmentErrors:
    """The caller-supplied fragment text is unparsable / not-a-mapping / missing the key."""

    def test_unparsable_fragment_raises(self, tmp_path: Path) -> None:
        """A fragment that fails ``yaml.safe_load`` raises UpsertError naming the fragment.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        with pytest.raises(UpsertError, match="fragment is not valid YAML"):
            upsert_sky_config_key("oci", "oci: [unterminated\n", path)

    def test_fragment_not_a_mapping_raises(self, tmp_path: Path) -> None:
        """A fragment that parses to a scalar (not a mapping) raises UpsertError.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        with pytest.raises(UpsertError, match="fragment is not a YAML mapping"):
            upsert_sky_config_key("oci", "just-a-scalar\n", path)

    def test_fragment_missing_named_key_raises(self, tmp_path: Path) -> None:
        """A fragment that's a mapping but lacks the named top-level key raises UpsertError.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        with pytest.raises(UpsertError, match="fragment missing top-level 'oci'"):
            upsert_sky_config_key("oci", "jobs:\n  controller: {}\n", path)


# ---------------------------------------------------------------------------
# CLI wrapper — exit codes, env var carriage, no-traceback contract
# ---------------------------------------------------------------------------


class TestCliEntryPoint:
    """The ``python -m scripts.upsert_sky_config_key`` CLI wrapper around the public function."""

    def _run_cli(
        self,
        *args: str,
        fragment: str,
        env_extra: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Invoke the module as a script with SYNTH_UPSERT_FRAGMENT set.

        :param args: CLI positional arguments forwarded to the module.
        :param fragment: YAML text passed through the ``SYNTH_UPSERT_FRAGMENT`` env var.
        :param env_extra: Optional extra env vars merged on top of the parent env + fragment.
        :returns: ``subprocess.CompletedProcess`` with ``stdout`` / ``stderr`` captured as text.
        :rtype: subprocess.CompletedProcess[str]
        """
        env = {
            **os.environ,
            "SYNTH_UPSERT_FRAGMENT": fragment,
            **(env_extra or {}),
        }
        return subprocess.run(  # noqa: S603 — controlled args
            [sys.executable, str(MODULE_PATH), *args],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_cli_succeeds_with_exit_zero(self, tmp_path: Path) -> None:
        """A successful upsert via the CLI exits 0 and writes the file.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        result = self._run_cli("oci", str(path), fragment="oci:\n  default: {}\n")
        assert result.returncode == 0, result.stderr
        assert path.is_file()

    def test_cli_emits_single_line_error_on_stderr_no_traceback(self, tmp_path: Path) -> None:
        """On malformed fragment input, the CLI exits non-zero with a single-line error on stderr
        and no Python traceback — matches the existing bash-driven contract.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        result = self._run_cli("oci", str(path), fragment="just-a-scalar\n")
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "upsert_sky_config_key[oci]:" in result.stderr
        assert "fragment is not a YAML mapping" in result.stderr

    def test_cli_missing_fragment_env_var_fails(self, tmp_path: Path) -> None:
        """If SYNTH_UPSERT_FRAGMENT is unset, the CLI fails with a clear error (no traceback).

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        env = {k: v for k, v in os.environ.items() if k != "SYNTH_UPSERT_FRAGMENT"}
        result = subprocess.run(  # noqa: S603 — controlled args
            [sys.executable, str(MODULE_PATH), "oci", str(path)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "SYNTH_UPSERT_FRAGMENT" in result.stderr

    def test_cli_writes_file_at_mode_0600(self, tmp_path: Path) -> None:
        """The CLI's successful write leaves the file at mode 0o600.

        :param tmp_path: pytest tmp dir fixture.
        """
        path = tmp_path / "config.yaml"
        result = self._run_cli("oci", str(path), fragment="oci:\n  default: {}\n")
        assert result.returncode == 0, result.stderr
        assert _file_mode(path) == 0o600
