"""Cross-platform integration test for ``RenderConfig.parallel`` dispatch.

Exercises ``run(spec)`` with ``render.parallel=True`` through the real
``subprocess.check_call`` boundary on every OS / CI host — no Linux Xvfb
wrapper, no Surge VST3 bundle, no R2 credentials. A fake renderer script
(written into ``tmp_path``) replaces ``generate_vst_dataset.py``: it just
writes the expected output file and exits 0, so the test crosses the real
``ThreadPoolExecutor`` → ``subprocess.check_call`` → fresh Python interpreter
hop that the unit-tier tests (which patch ``subprocess.check_call``) cannot.

Companion to ``test_parallel_shard_render_linux.py``: that one stress-tests
the X11 wrapper against the real plugin (gated Linux+VST); this one keeps
parallel-dispatch coverage in CI on macOS and on Linux runners without a
plugin installed, so the dispatcher path is never coverage-blind.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from synth_setter.cli.generate_dataset import run
from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec

_NUM_SHARDS = 4
_SAMPLES_PER_SHARD = 8

_FAKE_RENDERER_SOURCE = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"")
    """
).strip()


def _write_fake_renderer(tmp_path: Path) -> Path:
    """Write a no-op renderer script that materializes the requested output path.

    The script takes the same first positional arg as the real renderer (the
    output shard path), writes an empty file there, and exits 0 — enough to
    satisfy ``_render_and_upload_shard``'s post-render ``is_file()`` check.
    Subsequent ``--<key> <value>`` flags from ``build_generate_args`` are
    accepted and ignored.

    :param tmp_path: Per-test tmp dir; the script is dropped here.
    :returns: Path to the fake renderer ``.py`` file.
    """
    script_path = tmp_path / "fake_renderer.py"
    script_path.write_text(_FAKE_RENDERER_SOURCE)
    return script_path


def _build_spec() -> DatasetSpec:
    """Return a ``DatasetSpec`` with ``num_shards=4`` and ``parallel=True``.

    Pins the renderer/plugin paths to placeholders — the real check is bypassed
    by monkeypatching ``extract_renderer_version`` in the test body, so the
    paths only need to round-trip through Pydantic.

    :returns: Spec yielding 4 shards of ``_SAMPLES_PER_SHARD`` rows each.
    """
    kwargs: dict[str, object] = {
        "task_name": "parallel-dispatch-xplat",
        "run_id": "parallel-dispatch-xplat-20260520T000000000Z",
        "created_at": datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc),
        "git_sha": "0" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [_SAMPLES_PER_SHARD * _NUM_SHARDS, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "parallel-dispatch-bucket",
            "prefix": "data/parallel-dispatch-xplat/parallel-dispatch-xplat-20260520T000000000Z/",
        },
        "render": {
            "plugin_path": "plugins/fake.vst3",
            "preset_path": "presets/fake.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "0.0.0-fake",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 1.0,
            "min_loudness": -60.0,
            "samples_per_render_batch": _SAMPLES_PER_SHARD,
            "samples_per_shard": _SAMPLES_PER_SHARD,
            "parallel": True,
            "gui_toggle_cadence": "never",
        },
    }
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


def test_parallel_dispatch_crosses_real_subprocess_boundary(
    fake_r2_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``parallel=True`` + 4 shards uploads every shard via real subprocess + real rclone.

    Stubs the renderer-version probe (no real VST3 on disk) and the R2
    skip-probe (every shard is "absent"), pins ``available_cpus`` to 8 so
    pool size is deterministic across CI hosts, and swaps the renderer args
    for a ``sys.executable <fake_renderer.py> <output_path>`` invocation.
    The dispatch path then runs end-to-end: ``ThreadPoolExecutor`` →
    ``subprocess.check_call`` → fresh Python → empty file written → real
    ``rclone copy`` → ``fake_r2_remote``.

    State-based assertion: every shard's filename exists under the fake R2
    remote when ``run(spec)`` returns.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Per-test tmp dir for the fake renderer script.
    :param monkeypatch: Used to pin pool size and swap version/probe/args.
    """
    spec = _build_spec()
    assert len(spec.shards) == _NUM_SHARDS

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    monkeypatch.setattr("synth_setter.cli.generate_dataset.available_cpus", lambda: 8)
    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.extract_renderer_version",
        lambda _path: spec.render.renderer_version,
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: None)
    # Skip the Linux Xvfb-wrapper prefix in ``_render_and_upload_shard`` — the
    # fake renderer needs no X11, and the wrapper isn't on PATH from the test
    # CWD (``fake_r2_remote`` chdirs into tmp_path). The wrapper itself is
    # stress-tested by ``test_parallel_shard_render_linux.py``.
    monkeypatch.setattr("synth_setter.cli.generate_dataset.sys.platform", "darwin")

    fake_renderer = _write_fake_renderer(tmp_path)

    def _fake_build_args(_spec: DatasetSpec, shard: ShardSpec, output_dir: Path) -> list[str]:
        return [sys.executable, str(fake_renderer), str(output_dir / shard.filename)]

    monkeypatch.setattr("synth_setter.cli.generate_dataset.build_generate_args", _fake_build_args)

    run(spec)

    bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    for shard in spec.shards:
        assert (bucket_prefix / shard.filename).is_file(), (
            f"shard {shard.shard_id} did not land in fake R2: {shard.filename}"
        )
