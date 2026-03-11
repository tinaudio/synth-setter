"""Unit tests for scripts/generate_shards.py.

The shard generator creates N shard files in output_dir/shards/, each with
a configurable number of samples. These tests mock the subprocess calls to
generate_vst_dataset.py so they run without VST plugins or audio hardware.

To run:
    pytest tests/test_generate_shards.py -v
"""

import json
from pathlib import Path
from unittest.mock import patch

import h5py
import hdf5plugin
import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.generate_shards import SHARD_SUBDIR, _resolve_parallel, generate_shards
from src.data.uploader import LocalFakeUploader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PARAMS = 92
SAMPLE_RATE = 44100
CHANNELS = 2
SIGNAL_DURATION = 4.0


def _make_fake_shard(path: Path, n_samples: int = 100) -> None:
    """Write a minimal HDF5 shard matching generate_vst_dataset output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        audio = f.create_dataset(
            "audio",
            data=np.zeros(
                (n_samples, CHANNELS, int(SAMPLE_RATE * SIGNAL_DURATION)),
                dtype=np.float16,
            ),
            compression=hdf5plugin.Blosc2(),
        )
        audio.attrs["velocity"] = 100
        audio.attrs["signal_duration_seconds"] = SIGNAL_DURATION
        audio.attrs["sample_rate"] = float(SAMPLE_RATE)
        audio.attrs["channels"] = CHANNELS
        audio.attrs["min_loudness"] = -55.0

        f.create_dataset(
            "mel_spec",
            data=np.zeros((n_samples, 2, 128, 401), dtype=np.float32),
            compression=hdf5plugin.Blosc2(),
        )
        f.create_dataset(
            "param_array",
            data=np.zeros((n_samples, N_PARAMS), dtype=np.float32),
            compression=hdf5plugin.Blosc2(),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_vst_subprocess():
    """Mock subprocess.run to create fake HDF5 shard files."""
    call_log = []

    def _fake_run(cmd, check=True):
        call_log.append(cmd)
        if "generate_vst_dataset.py" in " ".join(cmd):
            h5_path = None
            num_samples = 100
            for i, arg in enumerate(cmd):
                if arg.endswith(".h5"):
                    h5_path = Path(arg)
                    if i + 1 < len(cmd):
                        try:
                            num_samples = int(cmd[i + 1])
                        except ValueError:
                            pass
            if h5_path is not None:
                _make_fake_shard(h5_path, num_samples)

    with patch("scripts.generate_shards.subprocess.run", side_effect=_fake_run):
        yield call_log


# ---------------------------------------------------------------------------
# Tests — generate_shards
# ---------------------------------------------------------------------------


class TestGenerateShards:
    """Tests for the shard generation orchestration."""

    def test_generates_correct_number_of_shards(self, tmp_path, fake_vst_subprocess):
        """generate_shards creates the expected number of shard-*.h5 files."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=4,
            shard_size=100,
            param_spec="surge_simple",
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert len(shards) == 4

    def test_shard_naming_with_prefix(self, tmp_path, fake_vst_subprocess):
        """With a prefix, shard names are shard-{prefix}-{8hex}-{seq}.h5."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=2,
            shard_size=100,
            param_spec="surge_simple",
            instance_id_prefix="mypod",
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert len(shards) == 2
        # Pattern: shard-mypod-<8hex>-0000.h5
        for i, shard in enumerate(shards):
            parts = shard.stem.split("-")
            assert parts[0] == "shard"
            assert parts[1] == "mypod"
            assert len(parts[2]) == 8  # 8-char hex UUID suffix
            assert parts[3] == f"{i:04d}"

    def test_subprocess_calls(self, tmp_path, fake_vst_subprocess):
        """Mock subprocess is invoked once per shard with correct args."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=3,
            shard_size=100,
            param_spec="surge_simple",
        )
        generate_calls = [
            c for c in fake_vst_subprocess if "generate_vst_dataset.py" in " ".join(c)
        ]
        assert len(generate_calls) == 3

    def test_writes_worker_metadata_with_prefix(self, tmp_path, fake_vst_subprocess):
        """Metadata file uses the full instance_id (prefix-uuid) and records it."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=2,
            shard_size=100,
            param_spec="surge_simple",
            instance_id_prefix="worker01",
        )
        # Metadata filename: {prefix}-{8hex}-metadata.json
        meta_files = list(shard_dir.glob("worker01-*-metadata.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["instance_id"].startswith("worker01-")
        assert len(meta["instance_id"]) == len("worker01-") + 8
        assert meta["num_shards"] == 2
        assert meta["shard_size"] == 100
        assert meta["param_spec"] == "surge_simple"

    def test_no_prefix_auto_generates_id(self, tmp_path, fake_vst_subprocess):
        """When no prefix is provided, instance_id is an auto-generated 8-char hex."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        # Name pattern: shard-<8-char-hex>-0000.h5
        name = shards[0].stem  # e.g. "shard-a3f2b1c9-0000"
        parts = name.split("-")
        assert len(parts) == 3
        assert parts[0] == "shard"
        assert len(parts[1]) == 8  # 8-char hex from uuid4

    def test_same_prefix_produces_different_ids(self, tmp_path, fake_vst_subprocess):
        """Two calls with the same prefix produce different instance IDs (retry detection)."""
        dir_a = tmp_path / "a" / "shards"
        dir_b = tmp_path / "b" / "shards"
        generate_shards(
            shard_dir=dir_a,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
            instance_id_prefix="samepod",
        )
        generate_shards(
            shard_dir=dir_b,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
            instance_id_prefix="samepod",
        )

        name_a = list(dir_a.glob("shard-*.h5"))[0].name
        name_b = list(dir_b.glob("shard-*.h5"))[0].name
        # Same prefix but different UUID suffix
        assert name_a != name_b
        assert name_a.startswith("shard-samepod-")
        assert name_b.startswith("shard-samepod-")

    def test_passes_plugin_options(self, tmp_path, fake_vst_subprocess):
        """Plugin path and preset path are forwarded to generate_vst_dataset."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
            plugin_path="/custom/plugin.vst3",
            preset_path="/custom/preset.vstpreset",
        )
        cmd = fake_vst_subprocess[0]
        assert "/custom/plugin.vst3" in cmd
        assert "/custom/preset.vstpreset" in cmd

    def test_headless_false_omits_wrapper(self, tmp_path, fake_vst_subprocess):
        """Without --headless, the command starts with python (no Xvfb wrapper)."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
        )
        cmd = fake_vst_subprocess[0]
        assert cmd[0] == "python"
        assert cmd[1] == "src/data/vst/generate_vst_dataset.py"

    def test_headless_true_includes_wrapper(self, tmp_path, fake_vst_subprocess):
        """With --headless, the command is prefixed with the Xvfb wrapper script."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
            headless=True,
        )
        cmd = fake_vst_subprocess[0]
        assert cmd[0] == "scripts/run-linux-vst-headless.sh"
        assert cmd[1] == "python"
        assert cmd[2] == "src/data/vst/generate_vst_dataset.py"


# ---------------------------------------------------------------------------
# Tests — parallel mode
# ---------------------------------------------------------------------------


class TestGenerateShardsParallel:
    """Tests for parallel shard generation."""

    def test_parallel_generates_correct_number_of_shards(self, tmp_path, fake_vst_subprocess):
        """Parallel mode creates the expected number of shard files."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=4,
            shard_size=100,
            param_spec="surge_simple",
            parallel=True,
            max_workers=2,
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert len(shards) == 4

    def test_parallel_subprocess_calls(self, tmp_path, fake_vst_subprocess):
        """All N subprocess calls happen with parallel=True."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=3,
            shard_size=100,
            param_spec="surge_simple",
            parallel=True,
            max_workers=3,
        )
        generate_calls = [
            c for c in fake_vst_subprocess if "generate_vst_dataset.py" in " ".join(c)
        ]
        assert len(generate_calls) == 3

    def test_parallel_writes_worker_metadata(self, tmp_path, fake_vst_subprocess):
        """Metadata is written correctly after parallel generation."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=2,
            shard_size=100,
            param_spec="surge_simple",
            instance_id_prefix="par01",
            parallel=True,
            max_workers=2,
        )
        meta_files = list(shard_dir.glob("par01-*-metadata.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["instance_id"].startswith("par01-")
        assert meta["num_shards"] == 2


# ---------------------------------------------------------------------------
# Tests — _resolve_parallel
# ---------------------------------------------------------------------------


class TestResolveParallel:
    """Tests for auto-detection of parallel workers."""

    def test_true_resolves_to_cpu_count(self):
        """Parallel=True uses os.cpu_count() when cpu_count < num_shards."""
        with patch("scripts.generate_shards.os.cpu_count", return_value=4):
            assert _resolve_parallel(True, num_shards=10) == 4

    def test_true_caps_at_num_shards(self):
        """Parallel=True is capped at num_shards when cpu_count > num_shards."""
        with patch("scripts.generate_shards.os.cpu_count", return_value=16):
            assert _resolve_parallel(True, num_shards=3) == 3

    def test_true_cpu_count_none_falls_back_to_one(self):
        """Parallel=True falls back to 1 when os.cpu_count() returns None."""
        with patch("scripts.generate_shards.os.cpu_count", return_value=None):
            assert _resolve_parallel(True, num_shards=10) == 1

    def test_true_with_max_workers_caps(self):
        """Parallel=True with max_workers uses min(max_workers, num_shards)."""
        with patch("scripts.generate_shards.os.cpu_count", return_value=16):
            assert _resolve_parallel(True, num_shards=10, max_workers=4) == 4

    def test_true_with_max_workers_capped_by_num_shards(self):
        """max_workers is still capped at num_shards."""
        with patch("scripts.generate_shards.os.cpu_count", return_value=16):
            assert _resolve_parallel(True, num_shards=2, max_workers=8) == 2

    def test_false_returns_one(self):
        """Parallel=False returns 1 (sequential)."""
        assert _resolve_parallel(False, num_shards=10) == 1

    def test_false_ignores_max_workers(self):
        """Parallel=False returns 1 even if max_workers is set."""
        assert _resolve_parallel(False, num_shards=10, max_workers=4) == 1


# ---------------------------------------------------------------------------
# Tests — parallel=0 integration
# ---------------------------------------------------------------------------


class TestAutoParallel:
    """Integration tests: parallel=True auto-detects worker count."""

    def test_parallel_true_generates_all_shards(self, tmp_path, fake_vst_subprocess):
        """Parallel=True (auto) produces the correct number of shards."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=4,
            shard_size=100,
            param_spec="surge_simple",
            parallel=True,
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert len(shards) == 4

    def test_max_workers_generates_all_shards(self, tmp_path, fake_vst_subprocess):
        """Parallel=True with max_workers produces the correct number of shards."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=4,
            shard_size=100,
            param_spec="surge_simple",
            parallel=True,
            max_workers=2,
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert len(shards) == 4

    def test_default_parallel_generates_all_shards(self, tmp_path, fake_vst_subprocess):
        """Default (no parallel arg) produces the correct number of shards."""
        shard_dir = tmp_path / SHARD_SUBDIR
        generate_shards(
            shard_dir=shard_dir,
            num_shards=3,
            shard_size=100,
            param_spec="surge_simple",
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert len(shards) == 3


# ---------------------------------------------------------------------------
# Tests — R2 upload
# ---------------------------------------------------------------------------


class TestGenerateShardsUpload:
    """Tests for optional R2 upload after shard generation."""

    def test_upload_copies_shards_and_metadata(self, tmp_path, fake_vst_subprocess):
        """When uploader + r2_prefix are provided, shards and metadata are uploaded."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / SHARD_SUBDIR
        r2_dest = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(dest_dir=r2_dest)

        generate_shards(
            shard_dir=shard_dir,
            num_shards=2,
            shard_size=100,
            param_spec="surge_simple",
            instance_id_prefix="w1",
            uploader=uploader,
            r2_prefix="runs/batch42",
        )

        upload_dest = r2_dest / "runs/batch42" / SHARD_SUBDIR

        # Shards uploaded under runs/batch42/shards/
        uploaded_shards = sorted(upload_dest.glob("shard-*.h5"))
        assert len(uploaded_shards) == 2

        # Worker metadata uploaded alongside shards
        uploaded_meta = list(upload_dest.glob("w1-*-metadata.json"))
        assert len(uploaded_meta) == 1
        meta = json.loads(uploaded_meta[0].read_text())
        assert meta["instance_id"].startswith("w1-")

    def test_no_upload_without_uploader(self, tmp_path, fake_vst_subprocess):
        """When uploader is None, generate_shards completes without uploading."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / SHARD_SUBDIR

        # Should not raise — upload is simply skipped
        generate_shards(
            shard_dir=shard_dir,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
            r2_prefix="runs/batch42",
        )
        assert len(list(shard_dir.glob("shard-*.h5"))) == 1

    def test_no_upload_without_r2_prefix(self, tmp_path, fake_vst_subprocess):
        """When r2_prefix is None, generate_shards completes without uploading."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / SHARD_SUBDIR
        r2_dest = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(dest_dir=r2_dest)

        generate_shards(
            shard_dir=shard_dir,
            num_shards=1,
            shard_size=100,
            param_spec="surge_simple",
            uploader=uploader,
        )
        # Nothing uploaded
        assert not r2_dest.exists() or not list(r2_dest.rglob("*"))


# ---------------------------------------------------------------------------
# Tests — CLI with R2 options
# ---------------------------------------------------------------------------


class TestGenerateShardsCLI:
    """Tests for the Click CLI entry point, including R2 upload options."""

    def test_cli_r2_options_construct_uploader(self, tmp_path, fake_vst_subprocess):
        """CLI with --r2-bucket and --r2-prefix creates RcloneUploader and uploads."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from click.testing import CliRunner

        from scripts.generate_shards import main

        runner = CliRunner()
        mock_uploader_cls = MagicMock()

        with mock_patch("scripts.generate_shards.RcloneUploader", mock_uploader_cls):
            result = runner.invoke(
                main,
                [
                    "--num-shards",
                    "1",
                    "--shard-size",
                    "100",
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--param-spec",
                    "surge_simple",
                    "--r2-bucket",
                    "my-bucket",
                    "--r2-prefix",
                    "runs/batch42",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_uploader_cls.assert_called_once_with(bucket="my-bucket", dry_run=False)

    def test_cli_local_skips_upload(self, tmp_path, fake_vst_subprocess):
        """CLI with --local skips upload even if R2 env vars are set."""
        from click.testing import CliRunner

        from scripts.generate_shards import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--num-shards",
                "1",
                "--shard-size",
                "100",
                "--output-dir",
                str(tmp_path / "out"),
                "--param-spec",
                "surge_simple",
                "--local",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "uploaded" not in result.output

    def test_cli_missing_r2_bucket_fails(self, tmp_path, fake_vst_subprocess, monkeypatch):
        """CLI without --local and missing --r2-bucket exits with an error."""
        from click.testing import CliRunner

        from scripts.generate_shards import main

        monkeypatch.delenv("R2_BUCKET", raising=False)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--num-shards",
                "1",
                "--shard-size",
                "100",
                "--output-dir",
                str(tmp_path / "out"),
                "--param-spec",
                "surge_simple",
                "--r2-prefix",
                "runs/batch42",
            ],
        )
        assert result.exit_code != 0

    def test_cli_parallel_flag_succeeds(self, tmp_path, fake_vst_subprocess):
        """CLI with --parallel flag generates shards in parallel."""
        from click.testing import CliRunner

        from scripts.generate_shards import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--num-shards",
                "2",
                "--shard-size",
                "100",
                "--output-dir",
                str(tmp_path / "out"),
                "--param-spec",
                "surge_simple",
                "--local",
                "--parallel",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_cli_max_workers_succeeds(self, tmp_path, fake_vst_subprocess):
        """CLI with --parallel --max-workers generates shards with capped workers."""
        from click.testing import CliRunner

        from scripts.generate_shards import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--num-shards",
                "4",
                "--shard-size",
                "100",
                "--output-dir",
                str(tmp_path / "out"),
                "--param-spec",
                "surge_simple",
                "--local",
                "--parallel",
                "--max-workers",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_cli_max_workers_without_parallel_ignored(self, tmp_path, fake_vst_subprocess):
        """CLI with --max-workers but without --parallel runs sequentially."""
        from click.testing import CliRunner

        from scripts.generate_shards import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--num-shards",
                "2",
                "--shard-size",
                "100",
                "--output-dir",
                str(tmp_path / "out"),
                "--param-spec",
                "surge_simple",
                "--local",
                "--max-workers",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_cli_missing_r2_prefix_fails(self, tmp_path, fake_vst_subprocess):
        """CLI without --local and missing --r2-prefix exits with an error."""
        from click.testing import CliRunner

        from scripts.generate_shards import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--num-shards",
                "1",
                "--shard-size",
                "100",
                "--output-dir",
                str(tmp_path / "out"),
                "--param-spec",
                "surge_simple",
                "--r2-bucket",
                "my-bucket",
            ],
        )
        assert result.exit_code != 0
