"""End-to-end CLI test for ``synth-setter-generate-dataset-from-spec-uri``.

Exercises the spec-URI operator CLI on the 100% production path — no fakes,
no mocks, no stubbed renderer:

1. Compose the ``generate_dataset/smoke-shard`` experiment and materialize a
   real ``DatasetSpec`` (the same construction the Hydra launcher performs).
2. Upload it to real Cloudflare R2 via the production ``spec_io.upload_spec``.
3. Invoke the real CLI subprocess with the spec's ``r2://`` URI as its only
   argument: the CLI downloads the spec over the network, renders every shard
   through the real Surge XT VST3 plugin, and uploads the shards to R2.
4. Probe R2 for every shard object (real ``rclone lsf`` round-trip).
5. Re-invoke with the equivalent ``s3://`` URI and assert the resumability
   probe (#750) skips all already-uploaded shards — pinning the s3-scheme
   rewrite end-to-end as well.

Same prerequisites as ``test_generate_dataset_cli_wandb_e2e.py`` (auto-skip
otherwise): Linux (Xvfb for the headless VST host), Surge XT VST3 at
``/usr/lib/vst3/Surge XT.vst3``, and reachable R2 creds. A unique
``r2.prefix`` per run keeps concurrent CI runs isolated; a best-effort
``rclone purge`` finalizer cleans the prefix even when the test body raises.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import lance
import pytest
from omegaconf import DictConfig

from synth_setter.cli.generate_dataset import spec_from_cfg
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_shard import read_shard_metadata
from synth_setter.pipeline.data.lance_staging import shard_has_complete_attempt
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import load_spec_from_uri, upload_spec
from synth_setter.plugin_manager import (
    ArtifactLock,
    PluginManifest,
    adopt_plugin_bundle,
    link_plugin,
    managed_plugin_digest,
)

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_SYSTEM_SURGE_VST3 = Path("/usr/lib/vst3/Surge XT.vst3")
_SURGE_VST3 = Path(
    os.environ.get(
        "SYNTH_SETTER_PLUGIN_PATH",
        str(
            _SYSTEM_SURGE_VST3
            if _SYSTEM_SURGE_VST3.is_dir()
            else _CHECKOUT_ROOT / "plugins/Surge XT.vst3"
        ),
    )
)
_SMOKE_NUM_SHARDS = 3  # smoke-shard config: 12 samples / 4 per-shard = 3 shards

# Bounded so a hung VST init or stalled R2 transfer can't wedge CI; the skip
# re-run only probes R2 (no render), so it gets a much tighter bound.
_RENDER_TIMEOUT_S = 600
_SKIP_RUN_TIMEOUT_S = 180


def _unique_r2_prefix() -> str:
    """Build a ``ci-spec-uri/<run_id>/<run_attempt>/<uuid>/`` R2 prefix.

    Same isolation pattern as ``test_generate_dataset_cli_wandb_e2e.py`` so
    concurrent CI runs and re-runs never collide on one prefix.

    :returns: Trailing-slash-terminated prefix, used both as the spec's
        ``r2.prefix`` and as the ``rclone purge`` target.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    nonce = uuid.uuid4().hex[:8]
    return f"ci-spec-uri/{run_id}/{run_attempt}/{nonce}/"


def _compose_smoke_shard_cfg(prefix: str, plugin_path: Path) -> DictConfig:
    """Compose the smoke-shard dataset cfg pinned to this test's R2 prefix.

    :param prefix: Unique R2 prefix from :func:`_unique_r2_prefix`.
    :param plugin_path: Real manager-owned Surge alias used by launcher and worker.
    :returns: A composed dataset cfg ready for launcher spec materialization.
    """
    from hydra import compose, initialize_config_module

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/smoke-shard",
                f"synth.plugin_path={plugin_path}",
                f"+r2.prefix={prefix}",
            ],
        )


def _purge_r2_prefix(bucket: str, prefix: str) -> None:
    """Best-effort ``rclone purge`` of the test's R2 prefix; never raises.

    :param bucket: R2 bucket holding the prefix.
    :param prefix: Trailing-slash-terminated prefix to purge.
    """
    result = subprocess.run(  # noqa: S603 — args from validated spec fields
        ["rclone", "purge", f"r2:{bucket}/{prefix}", "--contimeout=10s", "--timeout=60s"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"WARN: rclone purge of r2:{bucket}/{prefix} exited {result.returncode}: "
            f"{result.stderr.strip()[:300]}\n"
        )


@pytest.fixture()
def managed_surge_alias(tmp_path: Path) -> Path:
    """Adopt real Surge into an isolated manager-owned stable alias.

    :param tmp_path: Scratch root for managed state and the stable alias.
    :returns: Stable alias backed by the repository manifest and lock.
    """
    if sys.platform != "linux":
        pytest.skip(f"requires Linux + Xvfb for vst_headless_wrapper (got {sys.platform})")
    if not _SURGE_VST3.is_dir():
        pytest.skip(f"Surge VST3 not at {_SURGE_VST3}; install in dev container or skip")
    manifest = PluginManifest.load(_CHECKOUT_ROOT / "studiorack.json")
    lock = ArtifactLock.load(_CHECKOUT_ROOT / "studiorack.lock.json", manifest)
    plugin = manifest.resolve("surge-synthesizer/surge")
    managed = tmp_path / "managed"
    adopt_plugin_bundle(
        plugin,
        plugins_dir=managed,
        bundle=_SURGE_VST3,
        locked_package=lock.package_for(plugin),
    )
    return link_plugin(
        plugin,
        artifact_lock=lock,
        plugins_dir=managed,
        links_dir=tmp_path / "plugins",
    )


@pytest.fixture()
def uploaded_spec(managed_surge_alias: Path) -> Iterator[DatasetSpec]:
    """Materialize a smoke-shard spec and upload it to real R2; purge on teardown.

    Skips when prerequisites are missing (Linux + Surge VST3 + reachable R2).

    :param managed_surge_alias: Isolated manager-owned alias for real Surge.
    :yields DatasetSpec: The frozen spec whose ``input_spec.json`` now lives
        at ``spec.r2.input_spec_uri()`` in real R2.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")

    prefix = _unique_r2_prefix()
    cfg = _compose_smoke_shard_cfg(prefix, managed_surge_alias)
    spec = spec_from_cfg(cfg)
    assert spec.render.synth.managed_plugin_digest is not None
    r2_io.ensure_r2_env_loaded()
    upload_spec(spec)
    uploaded = load_spec_from_uri(spec.r2.input_spec_uri())
    assert uploaded.render.synth.managed_plugin_digest == spec.render.synth.managed_plugin_digest
    try:
        yield uploaded
    finally:
        _purge_r2_prefix(spec.r2.bucket, prefix)


# The CLI's documented invocation contract (all argv forms): run from the
# checkout root, where the render subprocess script and the smoke-shard
# spec's relative preset path resolve. The CWD-relative work dir therefore
# lands under <checkout>/logs/ (gitignored), keyed by the unique run_id.


def _run_cli_with_uri(spec_uri: str, timeout: int) -> subprocess.CompletedProcess[str]:
    """Invoke the real generate-dataset CLI with ``spec_uri`` as its only argument.

    :param spec_uri: The spec URI positional handed to the CLI.
    :param timeout: Wall-clock ceiling in seconds.
    :returns: The completed subprocess; caller asserts on ``returncode``.
    """
    # Prepend this worktree's src/ so the subprocess imports the same
    # synth_setter source the test was collected from (editable-install /
    # worktree split — see test_generate_dataset_cli_wandb_e2e.py).
    worktree_src = _CHECKOUT_ROOT / "src"
    env = {
        **os.environ,
        "PYTHONPATH": f"{worktree_src}:{os.environ.get('PYTHONPATH', '')}",
        "SYNTH_SETTER_NUM_WORKERS": "1",
        "SYNTH_SETTER_WORKER_RANK": "0",
        "WANDB_MODE": "disabled",
    }
    entrypoint = (
        Path(sys.executable)
        .with_name("synth-setter-generate-dataset-from-spec-uri")
        .resolve(strict=True)
    )
    return subprocess.run(  # noqa: S603 — resolved console script and validated spec URI
        [str(entrypoint), "--no-wandb", spec_uri],
        cwd=_CHECKOUT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _assert_cli_succeeded(result: subprocess.CompletedProcess[str], context: str) -> None:
    """Fail loudly on a non-zero CLI exit, surfacing both stream tails.

    :param result: ``subprocess.run`` return value.
    :param context: Human-readable label for the failure message.
    """
    assert result.returncode == 0, (
        f"CLI exited {result.returncode} for {context}\n"
        f"--- STDOUT (tail) ---\n{result.stdout[-2000:]}\n"
        f"--- STDERR (tail) ---\n{result.stderr[-2000:]}"
    )


def test_cli_renders_spec_fetched_from_r2_uri_end_to_end(uploaded_spec: DatasetSpec) -> None:
    """``synth-setter-generate-dataset-from-spec-uri r2://…`` renders + uploads for real.

    The subprocess downloads the spec from R2 over the network, renders all three smoke shards
    through the real Surge plugin, uploads them to R2, and retains them under the CWD-relative work
    dir. Every shard is then probed in real R2 for a complete staged attempt — the same worker-side
    resumability contract used by generation.

    :param uploaded_spec: Spec already uploaded to real R2 by the fixture.
    """
    spec = uploaded_spec
    assert spec.num_shards == _SMOKE_NUM_SHARDS
    assert spec.render.synth.managed_plugin_digest is not None

    result = _run_cli_with_uri(spec.r2.input_spec_uri(), timeout=_RENDER_TIMEOUT_S)
    _assert_cli_succeeded(result, context=f"spec-URI render of {spec.r2.input_spec_uri()}")

    work_dir = _CHECKOUT_ROOT / "logs" / "generate_dataset" / "from_spec_uri" / spec.run_id
    for shard in spec.shards:
        assert shard_has_complete_attempt(spec, shard.shard_id), (
            f"shard {shard.filename} has no complete staged attempt at "
            f"{spec.r2.shard_staging_dir_uri(shard.shard_id)}"
        )
        local_shard = work_dir / shard.filename
        assert local_shard.is_dir(), f"shard {shard.filename} not retained under {work_dir}"
        metadata = read_shard_metadata(lance.dataset(str(local_shard)).schema)
        assert metadata.managed_plugin_digest == spec.render.synth.managed_plugin_digest


@pytest.fixture()
def launcher_uploaded_spec(
    managed_surge_alias: Path,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> DatasetSpec:
    """Run the real launcher console and return its managed spec from R2.

    :param managed_surge_alias: Isolated manager-owned alias for real Surge.
    :param tmp_path: Scratch root for launcher output.
    :param request: Registers remote-prefix cleanup even when fixture setup fails.
    :returns: Canonical launcher-pinned spec fetched through R2.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or rclone lsd r2: failed)")
    prefix = _unique_r2_prefix()
    request.addfinalizer(lambda: _purge_r2_prefix("intermediate-data", prefix))
    run_id = f"managed-digest-{uuid.uuid4().hex[:8]}"
    launcher_output = tmp_path / "launcher"
    launcher = Path(sys.executable).with_name("synth-setter-generate-dataset").resolve(strict=True)
    overrides = [
        "experiment=generate_dataset/smoke-shard",
        f"synth.plugin_path={managed_surge_alias}",
        "train_val_test_sizes=[1,0,0]",
        "render.samples_per_render_batch=1",
        "render.samples_per_shard=1",
        f"run_id={run_id}",
        f"+r2.prefix={prefix}",
        f"hydra.run.dir={launcher_output}",
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": f"{_CHECKOUT_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        "SYNTH_SETTER_NUM_WORKERS": "2",
        "SYNTH_SETTER_WORKER_RANK": "1",
        "WANDB_MODE": "disabled",
    }
    launched = subprocess.run(  # noqa: S603 — resolved console script and fixed overrides
        [str(launcher), *overrides],
        cwd=_CHECKOUT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=_SKIP_RUN_TIMEOUT_S,
    )
    _assert_cli_succeeded(launched, context="managed launcher materialization")
    spec_paths = list(launcher_output.rglob("input_spec.json"))
    assert len(spec_paths) == 1
    launcher_spec = DatasetSpec.model_validate_json(spec_paths[0].read_text())
    expected_digest = managed_plugin_digest(managed_surge_alias)
    assert expected_digest is not None
    assert launcher_spec.render.synth.managed_plugin_digest == expected_digest
    worker_spec = load_spec_from_uri(launcher_spec.r2.input_spec_uri())
    assert worker_spec.render.synth.managed_plugin_digest == expected_digest
    return worker_spec


def _spec_uri_work_dir(spec: DatasetSpec) -> Path:
    """Locate worker output outside Hydra's launcher directory.

    :param spec: Frozen worker spec whose run ID selects the output directory.
    :returns: CWD-relative spec-URI work directory.
    """
    return _CHECKOUT_ROOT / "logs/generate_dataset/from_spec_uri" / spec.run_id


@pytest.mark.requires_vst
def test_launcher_digest_reaches_real_worker_rendered_shard_metadata(
    launcher_uploaded_spec: DatasetSpec,
) -> None:
    """The real worker preserves launcher-pinned managed identity in its shard.

    :param launcher_uploaded_spec: Canonical managed spec uploaded by the real launcher.
    """
    spec = launcher_uploaded_spec
    expected_digest = spec.render.synth.managed_plugin_digest
    assert expected_digest is not None
    work_dir = _spec_uri_work_dir(spec)
    shutil.rmtree(work_dir, ignore_errors=True)

    worker = _run_cli_with_uri(spec.r2.input_spec_uri(), timeout=_RENDER_TIMEOUT_S)

    _assert_cli_succeeded(worker, context="managed worker render")
    assert shard_has_complete_attempt(spec, spec.shards[0].shard_id)
    shard_path = work_dir / spec.shards[0].filename
    metadata = read_shard_metadata(lance.dataset(str(shard_path)).schema)
    assert metadata.managed_plugin_digest == expected_digest
    shutil.rmtree(work_dir)


@pytest.mark.requires_vst
def test_worker_local_managed_digest_mismatch_fails_before_output(
    launcher_uploaded_spec: DatasetSpec,
    tmp_path: Path,
) -> None:
    """A real local worker rejects mismatched managed identity before shard output.

    :param launcher_uploaded_spec: Canonical managed spec uploaded by the real launcher.
    :param tmp_path: Scratch root for the deliberately mismatched local spec.
    """
    spec = launcher_uploaded_spec
    mismatched_synth = spec.render.synth.model_copy(update={"managed_plugin_digest": "0" * 64})
    mismatched_spec = spec.model_copy(
        update={"render": spec.render.model_copy(update={"synth": mismatched_synth})}
    )
    mismatch_path = tmp_path / "mismatch-input-spec.json"
    mismatch_path.write_text(mismatched_spec.model_dump_json())
    work_dir = _spec_uri_work_dir(spec)
    shutil.rmtree(work_dir, ignore_errors=True)

    mismatch = _run_cli_with_uri(str(mismatch_path), timeout=_SKIP_RUN_TIMEOUT_S)

    assert mismatch.returncode != 0
    assert "Managed plugin digest mismatch" in mismatch.stderr
    assert not work_dir.exists()


def test_cli_s3_uri_rerun_skips_already_uploaded_shards(uploaded_spec: DatasetSpec) -> None:
    """An ``s3://`` spelling of the spec URI works and the re-run skips extant shards.

    First run renders + uploads via the ``r2://`` URI; the second invocation
    addresses the same spec through ``s3://`` (the scheme W&B references
    record) and must exit 0 with every shard skipped by the resumability
    probe (#750) — no re-render.

    :param uploaded_spec: Spec already uploaded to real R2 by the fixture.
    """
    spec = uploaded_spec

    first = _run_cli_with_uri(spec.r2.input_spec_uri(), timeout=_RENDER_TIMEOUT_S)
    _assert_cli_succeeded(first, context="initial r2:// spec-URI render")

    s3_uri = r2_io.to_s3_uri(spec.r2.input_spec_uri())
    second = _run_cli_with_uri(s3_uri, timeout=_SKIP_RUN_TIMEOUT_S)
    _assert_cli_succeeded(second, context=f"s3:// spec-URI skip re-run of {s3_uri}")

    summary = f"rendered=0 skipped={_SMOKE_NUM_SHARDS} of {_SMOKE_NUM_SHARDS}"
    assert summary in second.stderr, (
        f"expected skip summary {summary!r} in re-run stderr; the s3:// re-run "
        f"re-rendered shards it should have skipped\n"
        f"--- STDERR (tail) ---\n{second.stderr[-2000:]}"
    )
