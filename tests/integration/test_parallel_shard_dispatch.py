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

import shutil
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


def _wire_run_into_fake_renderer(
    spec: DatasetSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire ``run(spec)`` to cross the real subprocess boundary into a fake renderer.

    Stubs the renderer-version probe (no real VST3 on disk), the R2 skip-probe
    (every shard is "absent"), pins ``available_cpus`` to 8 so pool size is
    deterministic across CI hosts, swaps the renderer args for a
    ``sys.executable <fake_renderer.py> <output_path>`` invocation, and skips
    the Linux Xvfb wrapper (fake renderer needs no X11; the wrapper is
    stress-tested by ``test_parallel_shard_render_linux.py``).

    Partition env (``SYNTH_SETTER_WORKER_RANK`` / ``SYNTH_SETTER_NUM_WORKERS``)
    is the caller's responsibility — leave both unset for default-mode tests
    or call ``monkeypatch.setenv`` for explicit-rank tests.

    :param spec: Spec whose ``render.renderer_version`` is mirrored by the stub probe.
    :param tmp_path: Per-test tmp dir; the fake renderer is dropped here.
    :param monkeypatch: Pytest fixture used for all the stubs.
    """
    monkeypatch.setattr("synth_setter.cli.generate_dataset.available_cpus", lambda: 8)
    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset.extract_renderer_version",
        lambda _path: spec.render.renderer_version,
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: None)
    monkeypatch.setattr("synth_setter.cli.generate_dataset.sys.platform", "darwin")

    fake_renderer = _write_fake_renderer(tmp_path)

    def _fake_build_args(
        _spec: DatasetSpec,
        shard: ShardSpec,
        output_dir: Path,
        _script_path: Path,
    ) -> list[str]:
        return [sys.executable, str(fake_renderer), str(output_dir / shard.filename)]

    monkeypatch.setattr("synth_setter.cli.generate_dataset.build_generate_args", _fake_build_args)


def test_parallel_dispatch_crosses_real_subprocess_boundary(
    fake_r2_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``parallel=True`` + 4 shards uploads every shard via real subprocess + real rclone.

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
    _wire_run_into_fake_renderer(spec, tmp_path, monkeypatch)

    run(spec)

    bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    for shard in spec.shards:
        assert (bucket_prefix / shard.filename).is_file(), (
            f"shard {shard.shard_id} did not land in fake R2: {shard.filename}"
        )


def test_two_ranks_render_disjoint_complete_shard_partition(
    fake_r2_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ranks (0/2 + 1/2) over the same spec produce a disjoint, complete shard set.

    Cross-checks the launcher/worker env-var contract end-to-end: rank-0 and
    rank-1 each render their slice through the real subprocess boundary; the
    union must cover every shard and the intersection must be empty. Catches
    drift between ``WORKER_RANK_ENV_VAR`` / ``NUM_WORKERS_ENV_VAR`` (the
    launcher-side injection) and the names ``read_rank_world_from_env`` reads.
    If anyone renames one side without the other, both ranks fall back to the
    single-worker default and render every shard, breaking the disjointness
    assertion.

    :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
    :param tmp_path: Per-test tmp dir for the fake renderer script.
    :param monkeypatch: Used by the shared setup helper plus per-rank env injection.
    """
    spec = _build_spec()
    assert len(spec.shards) == _NUM_SHARDS

    _wire_run_into_fake_renderer(spec, tmp_path, monkeypatch)
    bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix

    def _landed_filenames() -> set[str]:
        return {
            shard.filename for shard in spec.shards if (bucket_prefix / shard.filename).is_file()
        }

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
    run(spec)
    rank_0_landed = _landed_filenames()

    # Wipe the bucket between ranks so rank-1's landed set is observed
    # independently — otherwise the intersection check would be vacuously
    # empty by set-difference construction and couldn't catch a regression
    # where rank-1 silently re-renders rank-0's shards.
    if bucket_prefix.exists():
        shutil.rmtree(bucket_prefix)

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "1")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
    run(spec)
    rank_1_landed = _landed_filenames()

    every_shard = {shard.filename for shard in spec.shards}
    assert rank_0_landed | rank_1_landed == every_shard, "ranks left a shard unrendered"
    assert not (rank_0_landed & rank_1_landed), (
        f"ranks rendered overlapping shards: {rank_0_landed & rank_1_landed}"
    )
    assert rank_0_landed and rank_1_landed, "one rank rendered nothing — env-var injection broke"
