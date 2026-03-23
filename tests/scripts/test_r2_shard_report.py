"""Tests for scripts/r2_shard_report.py — R2 shard analysis and reporting.

Tests are organized around the PUBLIC typed API:
- analyze_shards(): pure function that returns a ShardReport TypedDict
- format_report(): renders a ShardReport to plain text
- format_size(): stable utility with well-defined contract
- run_rclone_ls(): integration test against real R2 (auto-skipped without access)
"""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from scripts.r2_shard_report import (
    RcloneFile,
    ShardReport,
    analyze_shards,
    format_report,
    format_size,
    parse_rclone_ls_output,
    run_rclone_ls,
)

# ---------------------------------------------------------------------------
# Fixtures / test data
# ---------------------------------------------------------------------------

SAMPLE_FILES: list[RcloneFile] = [
    RcloneFile(4_650_191_447, "shard-0t7qvi3i3wyzfb-de99e3c2-0000.h5"),
    RcloneFile(4_622_813_770, "shard-0t7qvi3i3wyzfb-de99e3c2-0001.h5"),
    RcloneFile(4_650_191_447, "shard-20f21kt2vpztws-077d464c-0000.h5"),
    RcloneFile(4_622_813_770, "shard-20f21kt2vpztws-077d464c-0001.h5"),
    RcloneFile(2296, "shard-9iwsm829v7a92z-14ce64ee-0000.h5"),
    RcloneFile(2296, "shard-9iwsm829v7a92z-14ce64ee-0001.h5"),
    RcloneFile(457, "metadata-0t7qvi3i3wyzfb-de99e3c2.json"),
    RcloneFile(457, "metadata-20f21kt2vpztws-077d464c.json"),
]

SAMPLE_RCLONE_OUTPUT = """\
  4650191447 shard-0t7qvi3i3wyzfb-de99e3c2-0000.h5
  4622813770 shard-0t7qvi3i3wyzfb-de99e3c2-0001.h5
  4650191447 shard-20f21kt2vpztws-077d464c-0000.h5
  4622813770 shard-20f21kt2vpztws-077d464c-0001.h5
       2296 shard-9iwsm829v7a92z-14ce64ee-0000.h5
       2296 shard-9iwsm829v7a92z-14ce64ee-0001.h5
        457 metadata-0t7qvi3i3wyzfb-de99e3c2.json
        457 metadata-20f21kt2vpztws-077d464c.json
"""

DEFAULT_PREFIX = "r2:intermediate-data/10k_size_205_shard/shards/"
DEFAULT_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# analyze_shards — counts
# ---------------------------------------------------------------------------


class TestAnalyzeShardsCounts:
    """analyze_shards returns correct file counts, totals, and logical shard counts."""

    def test_h5_count_equals_six(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        assert report["h5_count"] == 6

    def test_json_count_equals_two(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        assert report["json_count"] == 2

    def test_logical_shard_count_equals_three(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        assert report["logical_shard_count"] == 3

    def test_other_count_equals_zero(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        assert report["other_count"] == 0

    def test_total_h5_bytes(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        assert report["total_h5_bytes"] == 18_546_015_026

    def test_total_h5_human(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        assert report["total_h5_human"] == "17.27 GiB"


# ---------------------------------------------------------------------------
# analyze_shards — suspect file detection
# ---------------------------------------------------------------------------


class TestAnalyzeShardsSuspect:
    """analyze_shards correctly identifies suspect files below the threshold."""

    def test_suspect_count_with_default_threshold(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        assert len(report["suspect_files"]) == 2

    def test_suspect_filenames_are_the_small_ones(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        suspect_names = {f["filename"] for f in report["suspect_files"]}
        assert suspect_names == {
            "shard-9iwsm829v7a92z-14ce64ee-0000.h5",
            "shard-9iwsm829v7a92z-14ce64ee-0001.h5",
        }

    def test_suspect_files_have_is_suspect_true(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        for suspect in report["suspect_files"]:
            assert suspect["is_suspect"] is True

    def test_suspect_files_have_correct_size_bytes(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)

        for suspect in report["suspect_files"]:
            assert suspect["size_bytes"] == 2296

    def test_all_big_files_no_suspects(self) -> None:
        all_big: list[RcloneFile] = [
            RcloneFile(4_650_191_447, "shard-aaa-111-0000.h5"),
            RcloneFile(4_622_813_770, "shard-aaa-111-0001.h5"),
        ]
        report: ShardReport = analyze_shards(all_big, threshold_gib=DEFAULT_THRESHOLD)

        assert len(report["suspect_files"]) == 0

    def test_custom_threshold_flags_medium_file(self) -> None:
        files: list[RcloneFile] = [
            RcloneFile(4_650_191_447, "shard-aaa-111-0000.h5"),  # 4.33 GiB
        ]
        report: ShardReport = analyze_shards(files, threshold_gib=5.0)

        assert len(report["suspect_files"]) == 1
        assert report["suspect_files"][0]["filename"] == "shard-aaa-111-0000.h5"


# ---------------------------------------------------------------------------
# analyze_shards — logical shard counting
# ---------------------------------------------------------------------------


class TestAnalyzeShardsLogicalShards:
    """analyze_shards correctly counts logical shards from chunk filenames."""

    def test_single_chunk_counts_as_one_logical_shard(self) -> None:
        files: list[RcloneFile] = [
            RcloneFile(4_650_191_447, "shard-aaa-111-0000.h5"),
        ]
        report: ShardReport = analyze_shards(files, threshold_gib=DEFAULT_THRESHOLD)

        assert report["logical_shard_count"] == 1

    def test_two_chunks_same_id_count_as_one_logical_shard(self) -> None:
        files: list[RcloneFile] = [
            RcloneFile(4_650_191_447, "shard-aaa-111-0000.h5"),
            RcloneFile(4_622_813_770, "shard-aaa-111-0001.h5"),
        ]
        report: ShardReport = analyze_shards(files, threshold_gib=DEFAULT_THRESHOLD)

        assert report["logical_shard_count"] == 1

    def test_empty_input_all_counts_zero(self) -> None:
        report: ShardReport = analyze_shards([], threshold_gib=DEFAULT_THRESHOLD)

        assert report["h5_count"] == 0
        assert report["json_count"] == 0
        assert report["other_count"] == 0
        assert report["logical_shard_count"] == 0
        assert report["total_h5_bytes"] == 0
        assert report["suspect_files"] == []
        assert report["h5_files"] == []


# ---------------------------------------------------------------------------
# format_report — thin smoke tests (formatting is not the contract)
# ---------------------------------------------------------------------------


class TestFormatReport:
    """format_report produces a human-readable string from a ShardReport."""

    def test_output_contains_prefix(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)
        text = format_report(report, prefix=DEFAULT_PREFIX)

        assert DEFAULT_PREFIX in text

    def test_output_contains_title(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)
        text = format_report(report, prefix=DEFAULT_PREFIX)

        assert "R2 Shard Report" in text

    def test_output_contains_suspect_when_suspects_present(self) -> None:
        report: ShardReport = analyze_shards(SAMPLE_FILES, threshold_gib=DEFAULT_THRESHOLD)
        text = format_report(report, prefix=DEFAULT_PREFIX)

        assert "SUSPECT" in text


# ---------------------------------------------------------------------------
# format_size — stable utility with well-defined contract
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_format_size_bytes_small_value(self) -> None:
        assert format_size(500) == "500 B"

    def test_format_size_kib(self) -> None:
        assert format_size(2296) == "2.24 KiB"

    def test_format_size_mib(self) -> None:
        assert format_size(5_242_880) == "5.00 MiB"

    def test_format_size_gib(self) -> None:
        assert format_size(4_650_191_447) == "4.33 GiB"

    def test_format_size_zero(self) -> None:
        assert format_size(0) == "0 B"


# ---------------------------------------------------------------------------
# run_rclone_ls — integration test against real R2 (auto-skipped without access)
# ---------------------------------------------------------------------------

_has_rclone = shutil.which("rclone") is not None


def _r2_reachable() -> bool:
    """Check if rclone can reach the r2: remote."""
    if not _has_rclone:
        return False
    try:
        subprocess.run(
            ["rclone", "lsd", "r2:"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


_skip_no_r2 = pytest.mark.skipif(not _r2_reachable(), reason="R2 not reachable")


@_skip_no_r2
@pytest.mark.r2
@pytest.mark.slow
class TestRunRcloneLsIntegration:
    """Integration test: write real files to r2:test-bucket, run report, clean up."""

    def test_end_to_end_report_against_r2(self) -> None:
        test_prefix = f"r2:test-bucket/test-{uuid.uuid4().hex[:8]}/"
        test_content = b"fake shard data"

        try:
            # Upload two small fake h5 files and one metadata json
            for name in [
                "shard-test-abc123-0000.h5",
                "shard-test-abc123-0001.h5",
                "metadata-test-abc123.json",
            ]:
                subprocess.run(
                    ["rclone", "rcat", f"{test_prefix}{name}"],
                    input=test_content,
                    check=True,
                )

            # Run the actual function under test
            output = run_rclone_ls(test_prefix)
            files = parse_rclone_ls_output(output)
            report = analyze_shards(files, threshold_gib=1.0)

            assert report["h5_count"] == 2
            assert report["json_count"] == 1
            assert report["logical_shard_count"] == 1
            # Both h5 files are 15 bytes — well under 1 GiB threshold
            assert len(report["suspect_files"]) == 2

        finally:
            # Clean up: delete the test prefix
            subprocess.run(
                ["rclone", "purge", test_prefix],
                capture_output=True,
            )
