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

from scripts.generate_shards import generate_shards
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
        shard_dir = tmp_path / "shards"
        generate_shards(
            shard_dir=shard_dir,
            num_shards=4,
            shard_size=100,
            param_spec="surge_simple",
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert len(shards) == 4

    def test_shard_naming_includes_instance_id(self, tmp_path, fake_vst_subprocess):
        """Shard filenames include the instance_id: shard-{id}-{seq}.h5."""
        shard_dir = tmp_path / "shards"
        generate_shards(
            shard_dir=shard_dir,
            num_shards=2,
            shard_size=100,
            param_spec="surge_simple",
            instance_id="abc12345",
        )
        shards = sorted(shard_dir.glob("shard-*.h5"))
        assert shards[0].name == "shard-abc12345-0000.h5"
        assert shards[1].name == "shard-abc12345-0001.h5"

    def test_subprocess_calls(self, tmp_path, fake_vst_subprocess):
        """Mock subprocess is invoked once per shard with correct args."""
        shard_dir = tmp_path / "shards"
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

    def test_writes_worker_metadata(self, tmp_path, fake_vst_subprocess):
        """generate_shards writes {instance_id}-metadata.json in the parent dir."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / "shards"
        generate_shards(
            shard_dir=shard_dir,
            num_shards=2,
            shard_size=100,
            param_spec="surge_simple",
            instance_id="worker01",
        )
        meta_path = output_dir / "worker01-metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["instance_id"] == "worker01"
        assert meta["num_shards"] == 2
        assert meta["shard_size"] == 100
        assert meta["param_spec"] == "surge_simple"

    def test_instance_id_auto_generated(self, tmp_path, fake_vst_subprocess):
        """When instance_id is not provided, a UUID-based ID is auto-generated."""
        shard_dir = tmp_path / "shards"
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

    def test_auto_id_differs_between_calls(self, tmp_path, fake_vst_subprocess):
        """Two separate generate_shards calls produce different auto-generated IDs."""
        dir_a = tmp_path / "a" / "shards"
        dir_b = tmp_path / "b" / "shards"
        generate_shards(shard_dir=dir_a, num_shards=1, shard_size=100, param_spec="surge_simple")
        generate_shards(shard_dir=dir_b, num_shards=1, shard_size=100, param_spec="surge_simple")

        name_a = list(dir_a.glob("shard-*.h5"))[0].name
        name_b = list(dir_b.glob("shard-*.h5"))[0].name
        assert name_a != name_b

    def test_passes_plugin_options(self, tmp_path, fake_vst_subprocess):
        """Plugin path and preset path are forwarded to generate_vst_dataset."""
        shard_dir = tmp_path / "shards"
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


# ---------------------------------------------------------------------------
# Tests — R2 upload
# ---------------------------------------------------------------------------


class TestGenerateShardsUpload:
    """Tests for optional R2 upload after shard generation."""

    def test_upload_copies_shards_and_metadata(self, tmp_path, fake_vst_subprocess):
        """When uploader + r2_prefix are provided, shards and metadata are uploaded."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / "shards"
        r2_dest = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(dest_dir=r2_dest)

        generate_shards(
            shard_dir=shard_dir,
            num_shards=2,
            shard_size=100,
            param_spec="surge_simple",
            instance_id="w1",
            uploader=uploader,
            r2_prefix="runs/batch42",
        )

        # Shards uploaded under runs/batch42/shards/
        uploaded_shards = sorted((r2_dest / "runs/batch42/shards").glob("shard-*.h5"))
        assert len(uploaded_shards) == 2

        # Metadata uploaded under runs/batch42/
        uploaded_meta = r2_dest / "runs/batch42" / "w1-metadata.json"
        assert uploaded_meta.exists()
        meta = json.loads(uploaded_meta.read_text())
        assert meta["instance_id"] == "w1"

    def test_no_upload_without_uploader(self, tmp_path, fake_vst_subprocess):
        """When uploader is None, generate_shards completes without uploading."""
        output_dir = tmp_path / "dataset"
        shard_dir = output_dir / "shards"

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
        shard_dir = output_dir / "shards"
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

    def test_cli_no_r2_options_skips_upload(self, tmp_path, fake_vst_subprocess):
        """CLI without --r2-bucket/--r2-prefix does not attempt upload."""
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
            ],
        )
        assert result.exit_code == 0, result.output
        assert "uploaded" not in result.output
