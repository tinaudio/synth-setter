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

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.ci.validate_shard import validate_all_shards_from_r2
from synth_setter.pipeline.ci.validate_spec import validate_structure
from tests.helpers.dummy_shards import stub_renderer

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

# Mirror tests/pipeline/entrypoints/test_generate_dataset_unit.py: a real
# VST3 bundle with a deterministic Contents/moduleinfo.json so
# extract_renderer_version (called by run) returns "1.0.0-test" without
# loading any .so via pedalboard. The Hydra override below pins
# render.plugin_path to this path so the constraint check passes.
TEST_PLUGIN_VST3 = (
    Path(__file__).resolve().parent.parent / "pipeline" / "fixtures" / "TestPlugin.vst3"
)
TEST_PLUGIN_VERSION = "1.0.0-test"


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

    side_effect = stub_renderer(expected_spec)
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
        if not (args and args[0] == "rclone"):
            renderer_invocations += 1
        # ``side_effect`` already passes rclone through to the real binary and
        # writes a dummy shard otherwise; we only need to count the renderer arm.
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
    operator_workspace_dir = tmp_path / "workspace"
    operator_workspace_dir.mkdir()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=overrides)
    cfg.paths.root_dir = str(repo_root)
    cfg.paths.output_dir = str(hydra_run_dir)
    cfg.paths.work_dir = str(repo_root)
    expected_spec = gd.spec_from_cfg(cfg)

    # Redirect ``operator_workspace()`` into ``tmp_path`` so the regression
    # check below — "spec must not land under the operator-workspace anchor"
    # — never touches files outside ``tmp_path`` on a developer's machine.
    env = {
        **os.environ,
        "SYNTH_SETTER_WORKER_RANK": "0",
        "SYNTH_SETTER_NUM_WORKERS": "1",
        "SYNTH_SETTER_WORKSPACE": str(operator_workspace_dir),
    }
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

    workspace_data_path = (
        operator_workspace_dir / "data" / expected_spec.task_name / expected_spec.run_id
    )
    assert not workspace_data_path.exists(), (
        f"operator-workspace anchor regression: subprocess wrote under {workspace_data_path}"
    )
