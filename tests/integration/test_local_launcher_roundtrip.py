"""End-to-end launcher roundtrip against real R2, with the VST renderer stubbed.

Exercises the ~85% of the dataset launcher's surface that ``test.yml``'s unit
tests do not: the full ``cli.generate_dataset.main`` orchestration writing the
canonical spec to R2, partitioning shards across (a single-worker) rank/world,
the per-shard render → ``rclone copy`` upload → skip-existing probe loop,
and the ``synth_setter.pipeline.ci`` validate-spec / validate-shard helpers
fetching everything back from R2.

What this test deliberately does NOT cover:
  - ``sky.launch`` / kind / file_mounts / dev-snapshot image pull — the
    ``test-dataset-generation.yml`` workflow keeps that coverage nightly.
  - The real Surge VST3 subprocess — replaced by a deterministic dummy that
    writes a validation-passing shard (HDF5 or tar) of the right shape.

The test is gated on ``rclone lsd r2:`` succeeding so a contributor's bare
clone or a fork PR without R2 secrets auto-skips. A unique R2 prefix per
``GITHUB_RUN_ID`` / ``GITHUB_RUN_ATTEMPT`` (plus a uuid suffix) keeps concurrent
CI runs isolated; a best-effort ``rclone purge`` finalizer cleans the prefix
even when the test body raises.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
    param_array_dataset_shape,
)
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.ci.validate_shard import validate_all_shards_from_r2
from synth_setter.pipeline.ci.validate_spec import validate_structure
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import (
    EXTENSION_TO_OUTPUT_FORMAT,
    DatasetSpec,
)
from tests.helpers.subprocess_args import find_script_index

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

# Mirror tests/pipeline/test_entrypoints/test_generate_dataset.py: a real
# VST3 bundle with a deterministic Contents/moduleinfo.json so
# extract_renderer_version (called by run) returns "1.0.0-test" without
# loading any .so via pedalboard. The Hydra override below pins
# render.plugin_path to this path so the constraint check passes.
TEST_PLUGIN_VST3 = (
    Path(__file__).resolve().parent.parent / "pipeline" / "fixtures" / "TestPlugin.vst3"
)
TEST_PLUGIN_VERSION = "1.0.0-test"


# Captured at import time so the rclone-passthrough side-effect can call the
# real subprocess.check_call without recursing through any patch that targets
# the same symbol the production code uses.
_REAL_CHECK_CALL = subprocess.check_call


def _write_dummy_h5_shard(output_path: Path, spec: DatasetSpec) -> None:
    """Write a validation-passing HDF5 shard with zeroed datasets.

    The datasets' full shapes match the writer's source-of-truth helpers in
    ``synth_setter.data.vst.shapes`` so ``validate_shard`` accepts the output.
    Values are all zeros — the validator only checks structure and shape, not
    content; determinism is preserved because no RNG is involved.

    :param output_path: Destination ``.h5`` file path; parent dir must exist.
    :param spec: Dataset spec whose ``render`` config and ``num_params`` drive
        the per-field array shapes the writer must reproduce.
    """
    render = spec.render
    n = render.samples_per_shard
    audio_shape = audio_dataset_shape(
        n, render.channels, render.sample_rate, render.signal_duration_seconds
    )
    mel_shape = mel_dataset_shape(
        n, render.channels, render.sample_rate, render.signal_duration_seconds
    )
    param_shape = param_array_dataset_shape(n, spec.num_params)
    with h5py.File(output_path, "w") as f:
        f.create_dataset(AUDIO_FIELD, data=np.zeros(audio_shape, dtype=np.float16))
        f.create_dataset(MEL_SPEC_FIELD, data=np.zeros(mel_shape, dtype=np.float32))
        f.create_dataset(PARAM_ARRAY_FIELD, data=np.zeros(param_shape, dtype=np.float32))


def _write_dummy_tar_shard(output_path: Path, spec: DatasetSpec) -> None:
    """Write a validation-passing WDS tar shard.

    A single batch keyed by ``00000000`` holds ``samples_per_shard`` rows for
    every writer field; ``metadata.json`` mirrors the ``RenderConfig`` fields
    that ``ShardMetadata`` requires. Validation is structural so all-zero
    arrays are accepted.

    :param output_path: Destination ``.tar`` file path; parent dir must exist.
    :param spec: Dataset spec whose ``render`` config and ``num_params`` drive
        the per-field array shapes and the ``ShardMetadata`` field values.
    """
    render = spec.render
    n = render.samples_per_shard
    audio = np.zeros(
        audio_dataset_shape(
            n, render.channels, render.sample_rate, render.signal_duration_seconds
        ),
        dtype=np.float16,
    )
    mel = np.zeros(
        mel_dataset_shape(n, render.channels, render.sample_rate, render.signal_duration_seconds),
        dtype=np.float32,
    )
    params = np.zeros(param_array_dataset_shape(n, spec.num_params), dtype=np.float32)
    metadata = ShardMetadata(
        velocity=render.velocity,
        signal_duration_seconds=render.signal_duration_seconds,
        sample_rate=render.sample_rate,
        channels=render.channels,
        min_loudness=render.min_loudness,
    )
    with tarfile.open(output_path, mode="w") as tar:
        for field_name, arr in (
            (AUDIO_FIELD, audio),
            (MEL_SPEC_FIELD, mel),
            (PARAM_ARRAY_FIELD, params),
        ):
            assert field_name in DATASET_FIELD_NAMES
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            payload = buf.getvalue()
            info = tarfile.TarInfo(name=f"00000000.{field_name}.npy")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        payload = metadata.model_dump_json().encode("utf-8")
        info = tarfile.TarInfo(name="metadata.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def _stub_renderer(spec: DatasetSpec) -> Callable[[list[str]], int]:
    """Return a subprocess.check_call side-effect that writes dummy shards.

    Dispatches on the renderer's output-path suffix via
    ``EXTENSION_TO_OUTPUT_FORMAT``, so the same factory backs both the hdf5
    and wds parametrizations. ``rclone`` invocations fall through to the real
    binary so the R2 upload, the skip-existing probe, and the finalize purge
    all hit real R2.

    :param spec: Dataset spec the launcher will materialize; threaded into
        the dummy-shard writers so shapes match the validator's expectations.
    :returns: A callable matching ``subprocess.check_call``'s side-effect contract.
    """

    def _side_effect(args: list[str]) -> int:
        if args and args[0] == "rclone":
            return _REAL_CHECK_CALL(args)  # noqa: S603 — passthrough to real rclone
        script_idx = find_script_index(args)
        output_file = Path(args[script_idx + 1])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fmt = EXTENSION_TO_OUTPUT_FORMAT.get(output_file.suffix)
        if fmt == "hdf5":
            _write_dummy_h5_shard(output_file, spec)
        elif fmt == "wds":
            _write_dummy_tar_shard(output_file, spec)
        else:
            raise AssertionError(
                f"stubbed renderer cannot write output with suffix {output_file.suffix!r}"
            )
        return 0

    return _side_effect


def _unique_r2_prefix() -> str:
    """Build a ``ci-roundtrip/<run_id>/<run_attempt>/<uuid>/`` R2 prefix.

    The full triple keeps concurrent CI runs isolated even on re-runs: the
    same ``run_id`` paired with a bumped ``run_attempt`` lands under a fresh
    leaf, and the trailing uuid nonce keeps the two parametrizations
    (``hdf5`` / ``wds``) in the same run apart. Locally (no ``GITHUB_*``
    env) the placeholders ``local`` / ``0`` keep dev artifacts grouped
    under a recognizable ``ci-roundtrip/local/0/`` parent that can be
    purged in bulk; each ``pytest`` invocation still gets a fresh uuid
    leaf so concurrent local runs don't collide.

    :returns: Trailing-slash-terminated R2 prefix string suitable for use as
        a ``r2.prefix=`` Hydra override and as an ``rclone purge`` target.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    nonce = uuid.uuid4().hex[:8]
    return f"ci-roundtrip/{run_id}/{run_attempt}/{nonce}/"


@pytest.fixture()
def ci_r2_prefix() -> Iterator[str]:
    """Yield a unique R2 prefix; purge it on teardown (best-effort).

    The yielded string is suitable for use as a Hydra override
    (``r2.prefix=<value>``). After the test body completes (success or
    failure), ``rclone purge`` removes the entire prefix. A non-zero exit
    from the purge is logged but never raised — leaking a few kilobytes of
    test artifacts is preferable to obscuring a real test failure with a
    cleanup error.

    :yields str: Trailing-slash-terminated unique R2 prefix string.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    prefix = _unique_r2_prefix()
    try:
        yield prefix
    finally:
        # Read bucket from the same config the test composed against. R2 v3
        # purges return non-zero on an empty prefix on some endpoints; swallow
        # any rclone exit so a cleanup hiccup doesn't mask a real test failure.
        from hydra import compose, initialize_config_module

        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="dataset",
                overrides=["experiment=generate_dataset/smoke-shard"],
            )
        bucket = cfg.r2.bucket
        purge_target = f"r2:{bucket}/{prefix}"
        result = subprocess.run(  # noqa: S603 — args built from validated config
            ["rclone", "purge", purge_target, "--contimeout=10s", "--timeout=60s"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            sys.stderr.write(
                f"WARN: rclone purge of {purge_target} exited {result.returncode}: "
                f"{result.stderr.strip()[:300]}\n"
            )


@pytest.mark.parametrize(
    "experiment",
    [
        "generate_dataset/smoke-shard",
        "generate_dataset/smoke-shard-wds",
    ],
)
def test_launcher_roundtrip_with_stubbed_renderer(
    experiment: str,
    ci_r2_prefix: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run ``cli.generate_dataset.main`` end-to-end against real R2.

    Covers: Hydra compose → ``DatasetSpec`` materialize → spec upload to R2,
    rank/world partitioning (single worker), per-shard skip-existing probe,
    per-shard render (stubbed to write a deterministic dummy shard) → rclone
    upload, and finally the existing ``pipeline.ci`` validators downloading
    everything back from R2 and confirming structural correctness.

    :param experiment: Hydra experiment id (parametrized to cover both
        ``hdf5`` and ``wds`` formats).
    :param ci_r2_prefix: Unique R2 prefix per CI run; finalizer purges it.
    :param monkeypatch: Pytest fixture used to set env vars and patch
        ``sys.argv`` for the ``main()`` invocation.
    """
    import synth_setter.cli.generate_dataset as gd

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    # ``main()`` reads sys.argv to build Hydra overrides. Pin:
    #   - experiment (drives shard count + render config)
    #   - render.plugin_path → TestPlugin.vst3 so the renderer_version probe
    #     resolves "1.0.0-test" without loading a real .so
    #   - render.renderer_version → same value so the constraint check passes
    #   - r2.prefix → the unique ci_r2_prefix so cleanup can purge a tight scope
    #     (uses ``+`` because ``prefix`` is not in ``configs/r2/default.yaml``;
    #     ``DatasetSpec``'s ``_normalize_r2_input`` then promotes the nested
    #     ``r2`` dict into the ``R2Location`` field, prefix included).
    #   - created_at → fixed timestamp for determinism (``+`` because the key
    #     is not in ``configs/dataset.yaml``; matches how the production
    #     launcher pins it on the worker side in ``_build_worker_cmd``).
    fixed_created_at = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    argv = [
        "synth-setter-generate-dataset",
        f"experiment={experiment}",
        f"render.plugin_path={TEST_PLUGIN_VST3}",
        f"render.renderer_version={TEST_PLUGIN_VERSION}",
        f"+r2.prefix={ci_r2_prefix}",
        f"+created_at={fixed_created_at}",
    ]
    monkeypatch.setattr("sys.argv", argv)

    # Compose once so we have the spec the launcher will materialize; this
    # is also what the downstream validate helpers expect to be on R2.
    from hydra import compose, initialize_config_module

    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=argv[1:])
    # Mirror main()'s pre-spec-construction shim for the unresolved
    # ${hydra:runtime.output_dir} interpolation.
    cfg.paths.root_dir = str(repo_root)
    cfg.paths.output_dir = str(repo_root)
    cfg.paths.work_dir = str(repo_root)
    expected_spec = gd.spec_from_cfg(cfg)

    side_effect = _stub_renderer(expected_spec)
    with patch(
        "synth_setter.cli.generate_dataset.subprocess.check_call",
        side_effect=side_effect,
    ):
        gd.main()

    # Re-read the canonical spec from R2 (the launcher just uploaded it) and
    # validate its structure with the same helper the production CI uses.
    from synth_setter.pipeline.r2_io import downloaded_to_tempfile

    spec_uri = expected_spec.r2.input_spec_uri()
    with downloaded_to_tempfile(spec_uri) as local_spec:
        spec_dict = json.loads(local_spec.read_text())
    spec_errors = validate_structure(spec_dict)
    assert spec_errors == [], f"spec validation failed: {spec_errors}"

    # And validate every shard by downloading each from R2 — this is the same
    # function .github/workflows/validate-dataset-shards.yaml runs.
    shard_errors = validate_all_shards_from_r2(expected_spec)
    assert shard_errors == [], f"shard validation failed: {shard_errors}"

    # Skip-existing path: re-run main() and assert that no renderer call
    # fires, because every shard is already in R2 from the first pass. This
    # exercises the production resumability invariant (#750) end-to-end.
    renderer_invocations = 0

    def _no_renderer_side_effect(args: list[str]) -> int:
        nonlocal renderer_invocations
        if args and args[0] == "rclone":
            return _REAL_CHECK_CALL(args)  # noqa: S603 — passthrough to real rclone
        renderer_invocations += 1
        return side_effect(args)

    with patch(
        "synth_setter.cli.generate_dataset.subprocess.check_call",
        side_effect=_no_renderer_side_effect,
    ):
        gd.main()
    assert renderer_invocations == 0, (
        f"skip-existing path failed: renderer was invoked {renderer_invocations}x on resume"
    )


def test_subprocess_writes_spec_under_hydra_output_dir(
    ci_r2_prefix: str,
    tmp_path: Path,
) -> None:
    """Spawn the CLI as a real subprocess and pin the spec-mirror path.

    Complements ``test_launcher_roundtrip_with_stubbed_renderer`` (which
    runs ``gd.main()`` in-process) by exercising the real CLI binary
    across a process boundary, which an in-process ``patch`` of
    ``subprocess.check_call`` cannot reach. ``hydra.run.dir`` is pinned to
    a known location under ``tmp_path`` so the spec-mirror file path is
    deterministic; the negative assertion catches a silent re-introduction
    of the operator-workspace anchor. The render subprocess crashes here
    because the fixture ``TestPlugin.vst3`` is a moduleinfo-only bundle
    with no loadable .so — the test pins behavior that happens before
    render dispatch, so the returncode is intentionally not asserted.

    :param ci_r2_prefix: Unique R2 prefix; finalizer purges it.
    :param tmp_path: Pytest tmp dir; pinned as the Hydra run dir so the
        timestamped default doesn't make filesystem assertions brittle.
    """
    from hydra import compose, initialize_config_module

    import synth_setter.cli.generate_dataset as gd

    fixed_created_at = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    hydra_run_dir = tmp_path / "run"
    overrides = [
        "experiment=generate_dataset/smoke-shard",
        f"render.plugin_path={TEST_PLUGIN_VST3}",
        f"render.renderer_version={TEST_PLUGIN_VERSION}",
        f"+r2.prefix={ci_r2_prefix}",
        f"+created_at={fixed_created_at}",
        f"hydra.run.dir={hydra_run_dir}",
    ]

    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=overrides)
    cfg.paths.root_dir = str(repo_root)
    cfg.paths.output_dir = str(hydra_run_dir)
    cfg.paths.work_dir = str(repo_root)
    expected_spec = gd.spec_from_cfg(cfg)

    repo_data_path = repo_root / "data" / expected_spec.task_name / expected_spec.run_id
    assert not repo_data_path.exists(), (
        f"precondition: {repo_data_path} must not exist before the run "
        "(otherwise the negative-pin assertion below is meaningless)"
    )

    env = {**os.environ, "SYNTH_SETTER_WORKER_RANK": "0", "SYNTH_SETTER_NUM_WORKERS": "1"}
    result = subprocess.run(  # noqa: S603 — fixed argv; overrides are validated by Hydra.
        [sys.executable, "-m", "synth_setter.cli.generate_dataset", *overrides],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    spec_mirror = (
        hydra_run_dir
        / "data"
        / expected_spec.task_name
        / expected_spec.run_id
        / "metadata"
        / "input_spec.json"
    )
    # Spec mirror is written before generate() runs, so it must exist even if
    # the subprocess later crashes during render — the fixture TestPlugin.vst3
    # has no loadable .so. The diagnostic dump fires only when the mirror is
    # missing, surfacing the real failure (e.g. spec write itself raised).
    assert spec_mirror.is_file(), (
        f"spec mirror not at expected path: {spec_mirror}\n"
        f"subprocess returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    loaded = json.loads(spec_mirror.read_text())
    assert loaded["task_name"] == expected_spec.task_name
    assert loaded["run_id"] == expected_spec.run_id

    assert not repo_data_path.exists(), (
        f"operator-workspace anchor regression: subprocess wrote under {repo_data_path}"
    )
