"""End-to-end CLI subprocess test for ``generate_dataset`` wandb tracking.

Exercises the manual-verification items from
``thoughts/shared/plans/2026-05-27-track-generate-dataset-in-wandb.md``:

- **Phase 1** — ``synth-setter-generate-dataset experiment=generate_dataset/smoke-shard``
  under ``WANDB_MODE=offline`` creates an offline run dir with a
  ``run_id`` matching ``smoke-shard-<timestamp>``.
- **Phase 2** — the offline ``.wandb`` binary carries one history row per
  shard (``shard/bytes`` + ``shard/render_seconds``) plus a terminal summary
  row (``shards/{rendered,skipped,total}`` + ``generation/{elapsed_seconds,samples,samples_per_second}``).
- **Phase 3** — the four cadence cells in ``sweeps/generate_dataset_cadence.yaml``
  (``plugin_reload_cadence`` × ``gui_toggle_cadence``) all run end-to-end as
  individual subprocess invocations and each lands the cadence override in
  the wandb run config — the same per-cell contract ``wandb agent`` would
  exercise.

The CLI subprocess hits real R2 and the real Surge VST3 plugin (same scope
as CI's ``Generate dataset shards -> real R2 (Docker + VST)`` job). The
test auto-skips when:

- not on Linux (``vst_headless_wrapper`` needs Xvfb),
- Surge VST3 is not at ``/usr/lib/vst3/Surge XT.vst3``,
- R2 is unreachable via :func:`r2_io.is_r2_reachable`.

A unique ``r2.prefix`` per test run keeps concurrent CI runs isolated, and a
best-effort ``rclone purge`` finalizer cleans the prefix even if the test
body raises.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from synth_setter.pipeline import r2_io
from tests.helpers.wandb_offline import read_history_rows

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_SURGE_VST3 = Path("/usr/lib/vst3/Surge XT.vst3")
_SMOKE_NUM_SHARDS = 3  # smoke-shard config: 12 samples / 4 per-shard = 3 shards


def _unique_r2_prefix() -> str:
    """Build a ``ci-cli-wandb-e2e/<run_id>/<run_attempt>/<uuid>/`` R2 prefix.

    Matches the isolation pattern in
    ``tests/integration/test_local_launcher_roundtrip.py`` so concurrent CI
    runs and re-runs don't collide on the same R2 prefix.

    :returns: Trailing-slash-terminated R2 prefix string suitable for use as
        an ``r2.prefix=`` Hydra override and as an ``rclone purge`` target.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    nonce = uuid.uuid4().hex[:8]
    return f"ci-cli-wandb-e2e/{run_id}/{run_attempt}/{nonce}/"


def _purge_r2_prefix(prefix: str) -> None:
    """Best-effort ``rclone purge`` of an R2 prefix; never raises.

    Reads ``r2.bucket`` from the same Hydra composition the test ran against.
    A non-zero purge exit is logged but swallowed — leaking a few kilobytes
    of test artifacts is preferable to obscuring a real test failure with a
    cleanup error.

    :param prefix: Trailing-slash-terminated R2 prefix, as built by
        :func:`_unique_r2_prefix`.
    """
    from hydra import compose, initialize_config_module

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=["experiment=generate_dataset/smoke-shard"])
    purge_target = f"r2:{cfg.r2.bucket}/{prefix}"
    result = subprocess.run(  # noqa: S603 — args from validated config
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


@pytest.fixture()
def cli_env(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(hydra_run_dir, r2_prefix)`` for one CLI subprocess invocation.

    Skips the test when prerequisites are missing (Linux + Surge VST3 + R2).
    Purges the R2 prefix on teardown. The hydra-run-dir doubles as the wandb
    ``save_dir`` via ``logger/wandb.yaml``'s ``save_dir: "${paths.output_dir}"``
    interpolation, so the offline run lands inside the test's ``tmp_path``.

    :param tmp_path: Pytest tmp dir; parent of the test-controlled Hydra
        run dir, so wandb output is fully isolated per test.
    :yields: ``(hydra_run_dir, r2_prefix)`` — the run dir wandb writes
        into, plus the unique R2 prefix the fixture purges on teardown.
    :ytype: tuple[Path, str]
    """
    if sys.platform != "linux":
        pytest.skip(f"requires Linux + Xvfb for vst_headless_wrapper (got {sys.platform})")
    if not _SURGE_VST3.is_dir():
        pytest.skip(f"Surge VST3 not at {_SURGE_VST3}; install in dev container or skip")
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")

    hydra_run_dir = tmp_path / "hydra_run"
    prefix = _unique_r2_prefix()
    try:
        yield (hydra_run_dir, prefix)
    finally:
        _purge_r2_prefix(prefix)


def _run_cli(
    *,
    hydra_run_dir: Path,
    r2_prefix: str,
    extra_overrides: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Subprocess-invoke ``synth-setter-generate-dataset`` with offline wandb.

    Pins the smoke-shard experiment, the system Surge VST3 (so the
    ``renderer_version`` probe and the actual render-subprocess have a real
    plugin to load), a unique ``r2.prefix``, and the Hydra run dir (which
    flows into the wandb logger's ``save_dir`` via
    ``${paths.output_dir}``). ``WANDB_MODE=offline`` keeps the run hermetic.

    :param hydra_run_dir: Test-controlled Hydra output dir; the wandb logger
        nests its ``wandb/offline-run-*/`` under here.
    :param r2_prefix: Unique R2 prefix for this invocation; the fixture
        purges it on teardown.
    :param extra_overrides: Additional Hydra overrides appended after the
        baseline pins (e.g. cadence overrides for Phase 3).
    :param extra_env: Merged on top of the base env (``WANDB_MODE=offline``
        + ``PYTHONPATH``); use to pin ``SYNTH_SETTER_WORKSPACE`` when the
        test cares where ``operator_workspace()`` lands on disk.
    :param timeout: Wall-clock seconds for the whole CLI; lift above the
        300s default when the invocation also runs inline finalize / oracle
        eval phases.
    :returns: The completed subprocess; the caller asserts on
        ``returncode`` + reads the wandb dir under ``hydra_run_dir``.
    """
    overrides = [
        "experiment=generate_dataset/smoke-shard",
        f"render.plugin_path={_SURGE_VST3}",
        f"+r2.prefix={r2_prefix}",
        f"hydra.run.dir={hydra_run_dir}",
    ]
    if extra_overrides:
        overrides += extra_overrides
    # Prepend the worktree's ``src/`` to ``PYTHONPATH`` so the subprocess
    # imports the same ``synth_setter`` source (and ``configs/``) the test
    # was collected from, even when ``pip install -e .`` is pinned to a
    # sibling checkout (the editable-install / worktree split otherwise
    # makes the subprocess silently use the wrong ``dataset.yaml``).
    worktree_src = Path(__file__).resolve().parents[2] / "src"
    pythonpath = f"{worktree_src}:{os.environ.get('PYTHONPATH', '')}"
    env = {**os.environ, "WANDB_MODE": "offline", "PYTHONPATH": pythonpath}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603 — args built from validated config + test-controlled paths
        [sys.executable, "-m", "synth_setter.cli.generate_dataset", *overrides],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _find_offline_run_dir(hydra_run_dir: Path) -> Path:
    """Locate the single ``wandb/offline-run-*`` dir under the Hydra run dir.

    The wandb logger's ``save_dir`` resolves to ``paths.output_dir`` which
    Hydra populates from ``hydra.run.dir``, so the offline run materializes
    at ``<hydra_run_dir>/wandb/offline-run-*``.

    :param hydra_run_dir: The Hydra run dir passed to ``_run_cli``.
    :returns: The single offline-run directory; fails the assertion if zero
        or more than one matches.
    """
    candidates = sorted(hydra_run_dir.glob("wandb/offline-run-*"))
    assert len(candidates) == 1, (
        f"expected exactly one offline run dir under {hydra_run_dir}/wandb/; got {candidates}"
    )
    return candidates[0]


def _assert_cli_succeeded(result: subprocess.CompletedProcess[str], context: str) -> None:
    """Fail loudly if the CLI exited non-zero, surfacing the tail of both streams.

    :param result: ``subprocess.run`` return value.
    :param context: Human-readable label for the failure message
        (e.g. ``"Phase 1+2 invocation"`` or ``"cadence reload=once gui=never"``).
    """
    assert result.returncode == 0, (
        f"CLI exited {result.returncode} for {context}\n"
        f"--- STDOUT (tail) ---\n{result.stdout[-2000:]}\n"
        f"--- STDERR (tail) ---\n{result.stderr[-2000:]}"
    )


def test_phase_1_and_2_cli_emits_run_id_plus_shard_and_summary_history(
    cli_env: tuple[Path, str],
) -> None:
    """Phase 1 + Phase 2 manual-verification items, exercised as a true subprocess.

    One ``synth-setter-generate-dataset experiment=generate_dataset/smoke-shard``
    invocation under ``WANDB_MODE=offline``. Asserts the offline run dir's
    ``wandb-metadata.json`` carries a ``run_id`` matching the
    ``make_dataset_wandb_run_id`` shape, then decodes the ``.wandb`` binary
    and confirms three per-shard rows plus one terminal summary row with
    every key the PR claims.

    :param cli_env: ``(hydra_run_dir, r2_prefix)`` from the fixture — gates
        skip conditions and owns R2 cleanup.
    """
    hydra_run_dir, prefix = cli_env
    result = _run_cli(hydra_run_dir=hydra_run_dir, r2_prefix=prefix)
    _assert_cli_succeeded(result, context="Phase 1+2 smoke-shard invocation")

    run_dir = _find_offline_run_dir(hydra_run_dir)

    # wandb offline mode (0.26.x) does not materialize ``files/wandb-metadata.json``
    # or ``files/config.yaml`` until ``wandb sync`` runs — everything is in the
    # ``run-*.wandb`` protobuf binary. The offline-run dir's trailing slug
    # carries the run id, so this is the load-bearing assertion that
    # ``wandb.run.id`` matched ``make_dataset_wandb_run_id(spec)``.
    dir_run_id = run_dir.name.split("-", 3)[-1]
    assert dir_run_id.startswith("smoke-shard-"), (
        f"expected run_id like smoke-shard-<timestamp>; got {dir_run_id!r} from {run_dir.name!r}"
    )

    wandb_binaries = list(run_dir.glob("run-*.wandb"))
    assert len(wandb_binaries) == 1, (
        f"expected exactly one .wandb binary under {run_dir}; got {wandb_binaries}"
    )
    rows = read_history_rows(wandb_binaries[0])

    shard_rows = [r for r in rows if "shard/bytes" in r]
    assert len(shard_rows) == _SMOKE_NUM_SHARDS, (
        f"expected {_SMOKE_NUM_SHARDS} per-shard rows; got {len(shard_rows)}: {shard_rows}"
    )
    for r in shard_rows:
        assert json.loads(r["shard/bytes"]) > 0, r
        assert json.loads(r["shard/render_seconds"]) >= 0.0, r

    summary_rows = [r for r in rows if "shards/rendered" in r]
    assert len(summary_rows) == 1, (
        f"expected exactly one summary row; got {len(summary_rows)}: {summary_rows}"
    )
    summary = summary_rows[0]
    for required_key in (
        "shards/rendered",
        "shards/skipped",
        "shards/total",
        "generation/elapsed_seconds",
        "generation/samples",
        "generation/samples_per_second",
    ):
        assert required_key in summary, f"summary row missing {required_key!r}: {summary}"
    # The run completed without raising, so the summary should report all 3
    # shards as either rendered or skipped (fail-fast contract — see
    # docs/reference/wandb-integration.md §5c).
    assert json.loads(summary["shards/total"]) == _SMOKE_NUM_SHARDS, summary
    assert json.loads(summary["generation/samples_per_second"]) >= 0.0, summary


@pytest.mark.parametrize(
    "reload_cadence,gui_cadence",
    [
        ("once", "never"),
        ("once", "once"),
        ("render", "never"),
        ("render", "once"),
    ],
)
def test_phase_3_sweep_cadence_cell_runs_end_to_end(
    cli_env: tuple[Path, str], reload_cadence: str, gui_cadence: str
) -> None:
    """Phase 3 manual-verification: each cell of the cadence sweep grid runs.

    The wandb sweep agent invokes the CLI per cell with cadence overrides
    appended via ``${args_no_hyphens}``. This test exercises each of the
    four cells (``plugin_reload_cadence`` ∈ {once, render} ×
    ``gui_toggle_cadence`` ∈ {never, once}) as a true subprocess, asserting
    the override flows into the wandb run config — the same end-to-end
    contract ``wandb agent`` exercises, without needing the wandb sweep
    backend (covered by the manual sweep validation step).

    :param cli_env: ``(hydra_run_dir, r2_prefix)`` from the fixture — gates
        skip conditions and owns R2 cleanup.
    :param reload_cadence: ``render.plugin_reload_cadence`` override.
    :param gui_cadence: ``render.gui_toggle_cadence`` override.
    """
    hydra_run_dir, prefix = cli_env
    overrides = [
        f"render.plugin_reload_cadence={reload_cadence}",
        f"render.gui_toggle_cadence={gui_cadence}",
    ]
    result = _run_cli(hydra_run_dir=hydra_run_dir, r2_prefix=prefix, extra_overrides=overrides)
    _assert_cli_succeeded(
        result, context=f"cadence cell reload={reload_cadence} gui={gui_cadence}"
    )

    run_dir = _find_offline_run_dir(hydra_run_dir)
    dir_run_id = run_dir.name.split("-", 3)[-1]
    assert dir_run_id.startswith("smoke-shard-"), (
        f"expected run_id like smoke-shard-<timestamp> for cell "
        f"reload={reload_cadence} gui={gui_cadence}; got {dir_run_id!r}"
    )
    # The cadence override is JSON-encoded inside ``Record.config`` entries in
    # the ``.wandb`` protobuf binary (wandb offline mode does not write
    # ``config.yaml`` until ``wandb sync``). A byte-substring scan over the
    # binary is a coarse but stable contract — the values ``"once"`` /
    # ``"render"`` / ``"never"`` are JSON-quoted and unambiguous against the
    # rest of the spec payload. If the override never reached
    # ``log_hyperparams``, neither byte sequence would appear.
    wandb_binaries = list(run_dir.glob("run-*.wandb"))
    assert len(wandb_binaries) == 1, (
        f"expected exactly one .wandb binary under {run_dir} for cell "
        f"reload={reload_cadence} gui={gui_cadence}; got {wandb_binaries}"
    )
    payload = wandb_binaries[0].read_bytes()
    for key, value in (
        ("plugin_reload_cadence", reload_cadence),
        ("gui_toggle_cadence", gui_cadence),
    ):
        needle = f'"{key}": "{value}"'.encode()
        assert needle in payload, (
            f"cadence override {key}={value!r} not found in wandb binary at "
            f"{wandb_binaries[0]} (cell reload={reload_cadence} gui={gui_cadence})"
        )


def _single_wandb_binary(run_dir: Path) -> Path:
    """Return the sole ``run-*.wandb`` binary under one offline-run dir.

    :param run_dir: a single ``offline-run-*`` dir; must hold exactly one
        ``run-*.wandb`` binary.
    :returns: the lone matching binary path.
    """
    binaries = list(run_dir.glob("run-*.wandb"))
    assert len(binaries) == 1, (
        f"expected exactly one .wandb binary under {run_dir}; got {binaries}"
    )
    return binaries[0]


def test_oracle_eval_inline_resumes_generate_wandb_run(
    cli_env: tuple[Path, str],
) -> None:
    """Inline oracle eval resumes the generate-phase wandb run id.

    One CLI invocation with ``oracle_eval_inline=true`` +
    ``finalize_inline=true`` under ``WANDB_MODE=offline``. After the CLI
    exits, the launcher's hydra dir and the operator workspace's
    ``oracle_eval/<run_id>/`` dir each hold one ``offline-run-*-<run_id>``
    directory; the two ``<run_id>`` slugs must match (the eval child
    resumed the same wandb run). Generate's dir must carry shard +
    summary history rows; eval's dir must carry at least one ``test/*``
    row from Lightning's ``trainer.test``.

    :param cli_env: ``(hydra_run_dir, r2_prefix)`` from the fixture — gates
        skip conditions and owns R2 cleanup.
    """
    hydra_run_dir, prefix = cli_env
    # ``operator_workspace()`` honors ``$SYNTH_SETTER_WORKSPACE`` and
    # decides where ``main()`` writes ``oracle_eval/<run_id>/``; pin it to
    # the fixture's tmp_path so the eval's wandb dir is reachable from the
    # test without depending on the checkout-root walk.
    workspace = hydra_run_dir.parent
    result = _run_cli(
        hydra_run_dir=hydra_run_dir,
        r2_prefix=prefix,
        extra_overrides=[
            "experiment=generate_dataset/smoke-shard-with-oracle-eval",
            "finalize_inline=true",
            "oracle_eval_inline=true",
        ],
        extra_env={"SYNTH_SETTER_WORKSPACE": str(workspace)},
        timeout=600,
    )
    _assert_cli_succeeded(result, context="oracle_eval_inline shared-wandb-run")

    generate_run_dir = _find_offline_run_dir(hydra_run_dir)
    generate_run_id = generate_run_dir.name.split("-", 3)[-1]

    eval_workspace_dir = workspace / "oracle_eval" / generate_run_id
    assert eval_workspace_dir.is_dir(), (
        f"expected eval workspace dir at {eval_workspace_dir}; "
        f"main()'s oracle_eval_inline branch did not run with the launcher's run_id"
    )
    eval_run_dir = _find_offline_run_dir(eval_workspace_dir)
    eval_run_id = eval_run_dir.name.split("-", 3)[-1]
    assert eval_run_id == generate_run_id, (
        f"eval wandb run id {eval_run_id!r} does not match generate's "
        f"{generate_run_id!r} — the eval subprocess opened a fresh run "
        f"instead of resuming"
    )

    generate_binary = _single_wandb_binary(generate_run_dir)
    generate_rows = read_history_rows(generate_binary)
    assert any("shard/bytes" in r for r in generate_rows), (
        f"generate dir missing per-shard history rows in {generate_binary}"
    )
    assert any("shards/rendered" in r for r in generate_rows), (
        f"generate dir missing summary history row in {generate_binary}"
    )

    eval_binary = _single_wandb_binary(eval_run_dir)
    eval_rows = read_history_rows(eval_binary)
    assert any(any(k.startswith("test/") for k in r) for r in eval_rows), (
        f"eval dir has no test/* history rows in {eval_binary}; "
        f"oracle eval did not log Lightning test_step metrics to the resumed run"
    )
