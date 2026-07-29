"""Contract tests for the focused managed-plugin integrity boundary."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

import synth_setter.plugin_integrity as plugin_integrity
import synth_setter.plugin_runtime as plugin_runtime
from synth_setter import plugin_manager
from synth_setter.cli import generate_dataset
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.plugin_integrity import (
    ArtifactLock,
    BundleEntry,
    BundleSeal,
    LockedArtifact,
    LockedPackage,
    ManagedBundleRecord,
    ManagedBundleStorage,
    PluginIntegrityError,
    bundle_entries,
    bundle_is_sealed,
    locked_package_digest,
    seal_plugin_bundle,
)
from synth_setter.plugin_manager import (
    PluginManifest,
    adopt_plugin_bundle,
    link_plugin,
    resolve_plugin_bundle,
)
from synth_setter.plugin_runtime import (
    ManagedAliasRecord,
    managed_plugin_digest,
    plugin_bundle_version,
    validate_plugin_bundle_for_runtime,
)
from tests.plugin_manager_test_support import (
    _artifact_lock,
    _binary_path,
    _bundle,
    _cross_platform_lock,
    _example_lock,
    _example_plugin,
    _manifest,
    _pe_test_binary_magic,
    _seal_bundle,
    _test_binary_magic,
)
from tests.plugin_manager_test_support import (
    _platform_binary_environment as _platform_binary_environment,
)


def test_plugin_integrity_public_api_imports_from_focused_module() -> None:
    """Lock and seal APIs are owned by the focused integrity module."""
    assert ArtifactLock.__module__ == "synth_setter.plugin_integrity"
    assert BundleEntry.__module__ == "synth_setter.plugin_integrity"
    assert BundleSeal.__module__ == "synth_setter.plugin_integrity"
    assert LockedArtifact.__module__ == "synth_setter.plugin_integrity"
    assert LockedPackage.__module__ == "synth_setter.plugin_integrity"
    assert ManagedAliasRecord.__module__ == "synth_setter.plugin_runtime"
    assert ManagedBundleRecord.__module__ == "synth_setter.plugin_integrity"
    assert PluginIntegrityError.__module__ == "synth_setter.plugin_integrity"
    assert bundle_entries.__module__ == "synth_setter.plugin_integrity"
    assert bundle_is_sealed.__module__ == "synth_setter.plugin_integrity"
    assert managed_plugin_digest.__module__ == "synth_setter.plugin_runtime"
    assert plugin_bundle_version.__module__ == "synth_setter.plugin_runtime"
    assert seal_plugin_bundle.__module__ == "synth_setter.plugin_integrity"
    assert validate_plugin_bundle_for_runtime.__module__ == "synth_setter.plugin_runtime"


def test_managed_bundle_storage_validates_and_discards_integrity_records(
    tmp_path: Path,
) -> None:
    """The public storage facade owns managed bundle record access.

    :param tmp_path: Scratch root for one sealed managed bundle.
    """
    plugin = _example_plugin()
    bundle = _bundle(tmp_path / "Example Synth.vst3")
    _seal_bundle(bundle, plugin)
    storage = ManagedBundleStorage(bundle)

    ownership = storage.read_ownership()
    resolved, seal = storage.validate()

    assert ownership is not None
    assert ownership.package == plugin.package
    assert resolved == bundle.resolve(strict=True)
    assert seal.package_reference == plugin.reference
    assert storage.has_integrity_record()

    storage.discard()

    assert storage.read_ownership() is None
    assert not storage.has_integrity_record()


def test_managed_alias_paths_are_owned_by_plugin_runtime(tmp_path: Path) -> None:
    """One runtime API owns alias ownership and transaction filenames.

    :param tmp_path: Scratch root for a representative stable alias.
    """
    alias = tmp_path / "plugins/Example Synth.vst3"

    ownership, transaction = plugin_runtime.managed_alias_paths(alias)

    assert ownership == alias.parent / ".Example Synth.vst3.synth-setter-managed.json"
    assert transaction == alias.parent / ".Example Synth.vst3.synth-setter-publication.json"
    assert not hasattr(plugin_integrity, "_alias_record_path")
    assert not hasattr(plugin_integrity, "_alias_transaction_path")


def test_plugin_manager_compatibility_reexports_integrity_public_api() -> None:
    """Existing public plugin_manager imports retain object identity."""
    assert plugin_manager.ArtifactLock is ArtifactLock
    assert plugin_manager.ManagedBundleRecord is ManagedBundleRecord
    assert plugin_manager.PluginIntegrityError is PluginIntegrityError
    assert plugin_manager.managed_plugin_digest is managed_plugin_digest
    assert plugin_manager.plugin_bundle_version is plugin_bundle_version
    assert plugin_manager.seal_plugin_bundle is seal_plugin_bundle
    assert plugin_manager.validate_plugin_bundle_for_runtime is validate_plugin_bundle_for_runtime


def test_locked_package_digest_is_canonical_across_artifact_and_selector_order() -> None:
    """Equivalent lock JSON ordering produces one package identity."""
    first = LockedPackage.model_validate(
        {
            "artifacts": [
                {
                    "architectures": ["x64", "arm64"],
                    "sha256": "a" * 64,
                    "systems": ["linux", "mac"],
                    "type": "archive",
                    "url": "https://example.test/a.zip",
                },
                {
                    "architectures": ["arm64"],
                    "sha256": "b" * 64,
                    "systems": ["mac"],
                    "type": "installer",
                    "url": "https://example.test/b.pkg",
                },
            ]
        }
    )
    second_payload = first.model_dump(mode="json")
    second_payload["artifacts"].reverse()
    second_payload["artifacts"][1]["architectures"].reverse()
    second_payload["artifacts"][1]["systems"].reverse()
    second = LockedPackage.model_validate(second_payload)

    assert locked_package_digest("example/synth@1.2.3", first) == locked_package_digest(
        "example/synth@1.2.3", second
    )


def test_managed_plugin_digest_cross_platform_contents_share_package_identity(
    tmp_path: Path,
) -> None:
    """Host-specific bytes under one exact lock produce one dataset identity.

    :param tmp_path: Scratch root for platform-specific bundle contents.
    """
    artifact_lock = _cross_platform_lock()
    plugin = _example_plugin()
    mac_bundle = _bundle(tmp_path / "mac/Example Synth.vst3", payload=b"mac-arm64")
    linux_bundle = _bundle(tmp_path / "linux/Example Synth.vst3", payload=b"linux-x64")
    _seal_bundle(mac_bundle, plugin, artifact_lock)
    _seal_bundle(linux_bundle, plugin, artifact_lock)

    assert managed_plugin_digest(mac_bundle) == managed_plugin_digest(linux_bundle)


def test_managed_plugin_digest_lock_rotation_changes_package_identity(tmp_path: Path) -> None:
    """Any repository lock rotation changes the dataset identity.

    :param tmp_path: Scratch root for bundles sealed under two lock revisions.
    """
    plugin = _example_plugin()
    first_bundle = _bundle(tmp_path / "first/Example Synth.vst3")
    second_bundle = _bundle(tmp_path / "second/Example Synth.vst3")
    _seal_bundle(first_bundle, plugin, _cross_platform_lock(linux_sha256="a" * 64))
    _seal_bundle(second_bundle, plugin, _cross_platform_lock(linux_sha256="c" * 64))

    assert managed_plugin_digest(first_bundle) != managed_plugin_digest(second_bundle)


def test_managed_plugin_digest_local_content_mutation_fails(tmp_path: Path) -> None:
    """Package provenance never substitutes for local byte validation.

    :param tmp_path: Scratch root for the mutated managed bundle.
    """
    plugin_bundle = _bundle(tmp_path / "Example Synth.vst3", payload=b"original")
    _seal_bundle(plugin_bundle, _example_plugin(), _cross_platform_lock())
    _binary_path(plugin_bundle).write_bytes(_test_binary_magic() + b"mutated")

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        managed_plugin_digest(plugin_bundle)


def test_adopt_plugin_bundle_records_distinct_explicit_source_identity(tmp_path: Path) -> None:
    """Explicit adoption combines its package lock with sealed source content.

    :param tmp_path: Scratch root for explicit and artifact-backed bundles.
    """
    artifact_lock = _cross_platform_lock()
    plugin = _example_plugin()
    source = _bundle(tmp_path / "source/Example Synth.vst3")

    managed = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "managed",
        bundle=source,
        locked_package=artifact_lock.package_for(plugin),
    )
    registry_bundle = _bundle(tmp_path / "linux/Example Synth.vst3")
    _seal_bundle(registry_bundle, plugin, artifact_lock)

    seal = BundleSeal.model_validate_json(
        (managed.parent / ".synth-setter-complete.json").read_text()
    )
    assert seal.source_kind == "explicit"
    assert managed_plugin_digest(managed) != managed_plugin_digest(registry_bundle)


def test_adopt_plugin_bundle_distinct_source_trees_have_distinct_identities(
    tmp_path: Path,
) -> None:
    """Two explicit builds under one package lock cannot share provenance.

    :param tmp_path: Scratch root for two adopted source trees.
    """
    plugin = _example_plugin()
    artifact_lock = _cross_platform_lock()
    first = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "first-managed",
        bundle=_bundle(tmp_path / "first/Example Synth.vst3", payload=b"first-build"),
        locked_package=artifact_lock.package_for(plugin),
    )
    second = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "second-managed",
        bundle=_bundle(tmp_path / "second/Example Synth.vst3", payload=b"second-build"),
        locked_package=artifact_lock.package_for(plugin),
    )

    assert managed_plugin_digest(first) != managed_plugin_digest(second)


def test_adopt_plugin_bundle_lock_rotation_changes_explicit_identity(tmp_path: Path) -> None:
    """An explicit build retains source identity while package-lock rotation changes its pin.

    :param tmp_path: Scratch root for the adopted source and two managed roots.
    """
    plugin = _example_plugin()
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"same-build")
    first_lock = _cross_platform_lock(linux_sha256="a" * 64)
    second_lock = _cross_platform_lock(linux_sha256="c" * 64)
    first = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "first-managed",
        bundle=source,
        locked_package=first_lock.package_for(plugin),
    )
    second = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "second-managed",
        bundle=source,
        locked_package=second_lock.package_for(plugin),
    )

    assert managed_plugin_digest(first) != managed_plugin_digest(second)


def test_remote_worker_linux_content_accepts_mac_launch_identity(
    tmp_path: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Linux worker renders a spec pinned from the lock-equivalent macOS seal.

    :param tmp_path: Scratch root for cross-platform bundles and worker output.
    :param dataset_spec_factory: Builds the worker's canonical dataset spec.
    :param monkeypatch: Selects an idle distributed worker rank.
    """
    artifact_lock = _cross_platform_lock()
    plugin = _example_plugin()
    mac_bundle = _bundle(tmp_path / "mac/Example Synth.vst3", payload=b"mac-arm64")
    linux_bundle = _bundle(tmp_path / "linux/Example Synth.vst3", payload=b"linux-x64")
    _seal_bundle(mac_bundle, plugin, artifact_lock)
    _seal_bundle(linux_bundle, plugin, artifact_lock)
    launch_digest = managed_plugin_digest(mac_bundle)
    assert launch_digest is not None

    base = dataset_spec_factory(
        task_name="cross-platform-plugin",
        run_id="cross-platform-plugin-run",
        train_val_test_sizes=(1, 0, 0),
        r2={"bucket": "intermediate-data", "prefix": "cross-platform-plugin/"},
        render={"samples_per_shard": 1},
    )
    synth = base.render.synth.model_copy(
        update={
            "managed_plugin_digest": launch_digest,
            "plugin_path": str(linux_bundle),
            "synth_version": "1.2.3",
        }
    )
    worker_spec = base.model_copy(
        update={"render": base.render.model_copy(update={"synth": synth})}
    )
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "1")

    generate_dataset.generate(worker_spec, tmp_path / "work", [])

    assert managed_plugin_digest(linux_bundle) == launch_digest
    assert list((tmp_path / "work").iterdir()) == []


def test_artifact_lock_load_exact_manifest_coverage_returns_lock(tmp_path: Path) -> None:
    """The artifact lock covers every manifest package at its exact version.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))

    lock = ArtifactLock.load(_artifact_lock(tmp_path / "studiorack.lock.json"), manifest)

    assert list(lock.root) == ["example/synth@1.2.3"]


def test_artifact_lock_load_unknown_artifact_field_rejected(tmp_path: Path) -> None:
    """Artifact-lock schema rejects registry fields outside trusted identity.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    payload = json.loads(lock_path.read_text())
    payload["example/synth@1.2.3"]["artifacts"][0]["size"] = 123
    lock_path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArtifactLock.load(lock_path, manifest)


def test_artifact_lock_load_http_artifact_url_rejected(tmp_path: Path) -> None:
    """Repository locks cannot downgrade artifact transport from HTTPS.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    payload = json.loads(lock_path.read_text())
    payload["example/synth@1.2.3"]["artifacts"][0]["url"] = "http://example.test/synth.zip"
    lock_path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="artifact URL must use HTTPS"):
        ArtifactLock.load(lock_path, manifest)


def test_artifact_lock_load_missing_manifest_reference_rejected(tmp_path: Path) -> None:
    """A lock cannot omit a manifest package version.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    lock_path = tmp_path / "studiorack.lock.json"
    lock_path.write_text("{}")

    with pytest.raises(ValueError, match="exactly cover"):
        ArtifactLock.load(lock_path, manifest)


def test_resolve_plugin_bundle_markerless_bundle_rejected(tmp_path: Path) -> None:
    """A realistic bundle without a completion record is not installed.

    :param tmp_path: Scratch root for markerless managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")

    with pytest.raises(FileNotFoundError, match="integrity"):
        resolve_plugin_bundle(plugin, tmp_path / "managed", artifact_lock=_example_lock())


@pytest.mark.parametrize(
    ("host_platform", "machine", "relative_binary", "signature"),
    [
        ("linux", "x86_64", "Contents/x86_64-linux/Example Synth.so", b"\x7fELF"),
        ("darwin", "arm64", "Contents/MacOS/Example Synth", b"\xcf\xfa\xed\xfe"),
        (
            "win32",
            "AMD64",
            "Contents/x86_64-win/Example Synth.vst3",
            _pe_test_binary_magic(),
        ),
    ],
)
def test_seal_plugin_bundle_platform_binary_signature_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_platform: str,
    machine: str,
    relative_binary: str,
    signature: bytes,
) -> None:
    """Platform VST3 locations require their native executable signature.

    :param tmp_path: Scratch root for one platform-shaped bundle.
    :param monkeypatch: Selects the platform and architecture under validation.
    :param host_platform: Platform selector used by the validator.
    :param machine: Host architecture reported to the validator.
    :param relative_binary: Platform VST3 executable location.
    :param signature: Minimal native executable header.
    """
    monkeypatch.setattr(plugin_integrity.sys, "platform", host_platform)
    monkeypatch.setattr(plugin_integrity.platform, "machine", lambda: machine)
    bundle = tmp_path / "Example Synth.vst3"
    binary = bundle / relative_binary
    binary.parent.mkdir(parents=True)
    binary.write_bytes(signature + b"plugin")
    (bundle / "Contents/moduleinfo.json").write_text('{"Version":"1.2.3"}')

    marker = _seal_bundle(bundle)

    assert marker.is_file()


def test_seal_plugin_bundle_empty_bundle_rejected(tmp_path: Path) -> None:
    """An empty VST3 directory cannot receive a completion record.

    :param tmp_path: Scratch root for empty managed state.
    """
    bundle = tmp_path / "Example Synth.vst3"
    bundle.mkdir()

    with pytest.raises(ValueError, match="platform VST3 binary"):
        _seal_bundle(bundle)


def test_seal_plugin_bundle_escaping_symlink_rejected(tmp_path: Path) -> None:
    """A bundle cannot seal a symlink whose target escapes the bundle.

    :param tmp_path: Scratch root for bundle and external content.
    """
    bundle = _bundle(tmp_path / "Example Synth.vst3")
    external = tmp_path / "external.txt"
    external.write_text("outside")
    (bundle / "Contents/external.txt").symlink_to(external)

    with pytest.raises(ValueError, match="escapes"):
        _seal_bundle(bundle)


def test_seal_plugin_bundle_records_exact_package_lock_provenance(tmp_path: Path) -> None:
    """Registry seals bind content to the canonical exact-package lock digest.

    :param tmp_path: Scratch root for managed bundle state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")

    marker = _seal_bundle(bundle, plugin)

    seal = json.loads(marker.read_text())
    ownership = json.loads((bundle.parent / ".synth-setter-managed.json").read_text())
    assert ownership == {
        "bundle": "Example Synth.vst3",
        "package": "example/synth",
        "schema": 1,
        "source_kind": "artifact-lock",
        "version": "1.2.3",
    }
    assert seal["package_reference"] == "example/synth@1.2.3"
    assert seal["source_kind"] == "artifact-lock"
    assert seal["locked_package_sha256"] == (
        "ce68cc987663810c48dcd1de66953f8f278a569dc4aef0747a68b18044502c46"
    )


def test_dataset_spec_managed_digest_matches_same_package_across_content(
    tmp_path: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """Validated host-specific contents produce the same package identity.

    :param tmp_path: Scratch root for two sealed managed bundles.
    :param dataset_spec_factory: Builds the common dataset-spec payload.
    """
    first_bundle = _bundle(tmp_path / "first/Example Synth.vst3", payload=b"first")
    second_bundle = _bundle(tmp_path / "second/Example Synth.vst3", payload=b"second")
    _seal_bundle(first_bundle)
    _seal_bundle(second_bundle)
    first_stable = tmp_path / "first-system/Example Synth.vst3"
    first_stable.parent.mkdir()
    first_stable.symlink_to(first_bundle.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(first_stable, first_bundle)
    first_config_path = tmp_path / "first-checkout/Example Synth.vst3"
    first_config_path.parent.mkdir()
    first_config_path.symlink_to(first_stable.absolute(), target_is_directory=True)
    second_stable = tmp_path / "second-system/Example Synth.vst3"
    second_stable.parent.mkdir()
    second_stable.symlink_to(second_bundle.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(second_stable, second_bundle)
    second_config_path = tmp_path / "second-checkout/Example Synth.vst3"
    second_config_path.parent.mkdir()
    second_config_path.symlink_to(second_stable.absolute(), target_is_directory=True)
    base = dataset_spec_factory(
        task_name="managed-content",
        run_id="managed-content-run",
        train_val_test_sizes=(1, 0, 0),
        r2={"bucket": "intermediate-data", "prefix": "managed-content/"},
        render={"samples_per_shard": 1},
    )
    raw = base.model_dump(
        mode="json",
        exclude={"shards", "num_shards", "num_params"},
    )
    raw["synth"] = raw["render"].pop("synth")

    first_raw = json.loads(json.dumps(raw))
    first_raw["synth"]["plugin_path"] = str(first_config_path)
    second_raw = json.loads(json.dumps(raw))
    second_raw["synth"]["plugin_path"] = str(second_config_path)
    from synth_setter.cli.generate_dataset import spec_from_cfg

    first_spec = spec_from_cfg(cast(DictConfig, OmegaConf.create(first_raw)))
    second_spec = spec_from_cfg(cast(DictConfig, OmegaConf.create(second_raw)))

    assert first_spec.render.synth.synth_version == second_spec.render.synth.synth_version
    assert first_spec.render.synth.managed_plugin_digest == managed_plugin_digest(
        first_config_path
    )
    assert second_spec.render.synth.managed_plugin_digest == managed_plugin_digest(
        second_config_path
    )
    assert (
        first_spec.render.synth.managed_plugin_digest
        == second_spec.render.synth.managed_plugin_digest
    )
    assert first_spec.render.shard_metadata() == second_spec.render.shard_metadata()


def test_spec_uri_worker_rejects_rotated_package_lock_before_staging(
    tmp_path: Path,
    fake_r2_remote: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker consumes launcher lock A and rejects local package lock B.

    :param tmp_path: Scratch root for managed state and worker files.
    :param fake_r2_remote: Local-typed R2 remote holding the canonical launcher spec.
    :param dataset_spec_factory: Builds the common dataset-spec payload.
    :param monkeypatch: Bypasses external auth while retaining real local rclone I/O.
    """
    from synth_setter.cli.generate_dataset import spec_from_cfg
    from synth_setter.cli.generate_dataset_from_spec_uri import run_from_spec_uri
    from synth_setter.pipeline.spec_io import upload_spec

    monkeypatch.setattr(
        "synth_setter.cli.generate_dataset_from_spec_uri.r2_io.ensure_r2_env_loaded",
        lambda: None,
    )
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "system-vst3/Example Synth.vst3", payload=b"artifact-a")
    managed_dir = tmp_path / "managed"
    artifact_lock = _example_lock()
    adopt_plugin_bundle(
        plugin,
        plugins_dir=managed_dir,
        bundle=source,
        locked_package=artifact_lock.package_for(plugin),
    )
    alias = link_plugin(
        plugin,
        artifact_lock=artifact_lock,
        plugins_dir=managed_dir,
        links_dir=tmp_path / "plugins",
    )
    base = dataset_spec_factory(
        task_name="managed-worker",
        run_id="managed-worker-run",
        train_val_test_sizes=(1, 0, 0),
        r2={"bucket": "intermediate-data", "prefix": "managed-worker/"},
        render={"samples_per_shard": 1},
    )
    raw = base.model_dump(mode="json", exclude={"shards", "num_shards", "num_params"})
    raw["synth"] = raw["render"].pop("synth")
    raw["synth"]["plugin_path"] = str(alias)
    launcher_spec = spec_from_cfg(OmegaConf.create(raw))
    digest_a = launcher_spec.render.synth.managed_plugin_digest
    assert digest_a is not None
    upload_spec(launcher_spec)

    _binary_path(source).write_bytes(_test_binary_magic() + b"artifact-b")
    rotated_payload = artifact_lock.model_dump(mode="json")
    rotated_payload[plugin.reference]["artifacts"][0]["sha256"] = "b" * 64
    rotated_lock = ArtifactLock.model_validate(rotated_payload)
    adopt_plugin_bundle(
        plugin,
        plugins_dir=managed_dir,
        bundle=source,
        locked_package=rotated_lock.package_for(plugin),
    )
    assert managed_plugin_digest(alias) != digest_a

    with pytest.raises(RuntimeError, match="Managed plugin digest mismatch"):
        run_from_spec_uri(launcher_spec.r2.input_spec_uri(), enable_wandb=False)

    run_prefix = fake_r2_remote / launcher_spec.r2.bucket / launcher_spec.r2.prefix
    assert (run_prefix / "input_spec.json").is_file()
    assert not (run_prefix / "metadata/workers").exists()


def test_worker_rejects_managed_plugin_without_digest_before_output(
    tmp_path: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """A worker cannot render manager-owned content as unmanaged provenance.

    :param tmp_path: Scratch root for managed state and render output.
    :param dataset_spec_factory: Builds the common dataset-spec payload.
    """
    from synth_setter.cli.generate_dataset import generate

    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "system-vst3/Example Synth.vst3")
    managed_dir = tmp_path / "managed"
    artifact_lock = _example_lock()
    adopt_plugin_bundle(
        plugin,
        plugins_dir=managed_dir,
        bundle=source,
        locked_package=artifact_lock.package_for(plugin),
    )
    alias = link_plugin(
        plugin,
        artifact_lock=artifact_lock,
        plugins_dir=managed_dir,
        links_dir=tmp_path / "plugins",
    )
    base = dataset_spec_factory(
        task_name="managed-worker-missing",
        run_id="managed-worker-missing-run",
        train_val_test_sizes=(1, 0, 0),
        r2={"bucket": "intermediate-data", "prefix": "managed-worker-missing/"},
        render={"samples_per_shard": 1},
    )
    synth = base.render.synth.model_copy(
        update={
            "managed_plugin_digest": None,
            "plugin_path": str(alias),
            "synth_version": "1.2.3",
        }
    )
    spec = base.model_copy(update={"render": base.render.model_copy(update={"synth": synth})})
    work_dir = tmp_path / "work"

    with pytest.raises(RuntimeError, match="missing managed plugin digest"):
        generate(spec, work_dir, [])

    assert not work_dir.exists()


def test_resolve_plugin_bundle_modified_after_seal_rejected(tmp_path: Path) -> None:
    """Content drift invalidates an otherwise complete managed bundle.

    :param tmp_path: Scratch root for sealed managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle, plugin)
    _binary_path(bundle).write_bytes(b"modified")

    with pytest.raises(FileNotFoundError, match="integrity"):
        resolve_plugin_bundle(plugin, tmp_path / "managed", artifact_lock=_example_lock())
