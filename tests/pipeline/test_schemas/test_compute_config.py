"""Tests for src/synth_setter/pipeline/schemas/compute.py — SkyPilot compute config.

Mirrors the dataset-spec / image-config trust-boundary pattern: every compute
YAML under configs/compute/ flows Hydra → ComputeConfig → sky.Task.from_yaml_config.
These tests pin the public typed API:

- ``ComputeConfig`` (Pydantic model): validates a SkyPilot Task YAML dict.
- ``load_compute_config_yaml(path)``: helper that loads + validates a YAML file.
- ``compute_config_from_cfg(cfg, *, compute_dir)``: resolves ``cfg.compute_template``
  (a name) to a YAML under ``compute_dir`` and constructs the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.pipeline.schemas.compute import (
    ComputeConfig,
    compute_config_from_cfg,
    load_compute_config_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPUTE_DIR = REPO_ROOT / "configs" / "compute"

MINIMAL_RUNPOD_RAW: dict = {
    "resources": {"cloud": "runpod", "accelerators": "RTX3090:1"},
    "envs": {"WORKER_SPEC_URI": ""},
    "setup": "echo setup",
    "run": "echo run",
}


class TestComputeConfigConstruction:
    """ComputeConfig accepts the minimal/required SkyPilot Task fields."""

    def test_minimal_runpod_validates(self) -> None:
        """A minimal RunPod-shaped dict constructs cleanly."""
        cfg = ComputeConfig(**MINIMAL_RUNPOD_RAW)

        assert cfg.resources["cloud"] == "runpod"
        assert cfg.envs == {"WORKER_SPEC_URI": ""}
        assert cfg.setup == "echo setup"
        assert cfg.run == "echo run"

    def test_resources_required(self) -> None:
        """Missing `resources` raises ValidationError."""
        bad = {k: v for k, v in MINIMAL_RUNPOD_RAW.items() if k != "resources"}

        with pytest.raises(ValidationError, match="resources"):
            ComputeConfig(**bad)

    def test_run_required(self) -> None:
        """Missing `run` raises ValidationError; a SkyPilot Task without `run:` is useless here."""
        bad = {k: v for k, v in MINIMAL_RUNPOD_RAW.items() if k != "run"}

        with pytest.raises(ValidationError, match="run"):
            ComputeConfig(**bad)

    def test_envs_optional_defaults_empty(self) -> None:
        """`envs:` is optional; SkyPilot's ``task.update_envs`` can add keys at launch time."""
        without_envs = {k: v for k, v in MINIMAL_RUNPOD_RAW.items() if k != "envs"}

        cfg = ComputeConfig(**without_envs)

        assert cfg.envs == {}

    def test_setup_optional(self) -> None:
        """`setup:` is optional — not every template needs a pre-run install step."""
        without_setup = {k: v for k, v in MINIMAL_RUNPOD_RAW.items() if k != "setup"}

        cfg = ComputeConfig(**without_setup)

        assert cfg.setup is None

    def test_extra_fields_permitted(self) -> None:
        """SkyPilot accepts many optional Task fields (num_nodes, file_mounts, workdir, ...).

        ComputeConfig must round-trip them so the launcher can hand SkyPilot a faithful dict.
        """
        with_extras = {
            **MINIMAL_RUNPOD_RAW,
            "num_nodes": 1,
            "file_mounts": {"/remote": "/local"},
            "workdir": "./work",
        }

        cfg = ComputeConfig(**with_extras)
        dumped = cfg.model_dump()

        assert dumped["num_nodes"] == 1
        assert dumped["file_mounts"] == {"/remote": "/local"}
        assert dumped["workdir"] == "./work"

    def test_model_is_frozen(self) -> None:
        """ComputeConfig is immutable post-construction — trust-boundary discipline."""
        cfg = ComputeConfig(**MINIMAL_RUNPOD_RAW)

        with pytest.raises(ValidationError):
            cfg.run = "echo other"  # type: ignore[misc]


class TestExistingComputeYamlsValidate:
    """Every YAML under configs/compute/ validates as a ComputeConfig.

    Pins the migration's correctness: if a new template adds a field ComputeConfig rejects, this
    test fails at PR time rather than at first-launch time.
    """

    @pytest.mark.parametrize(
        "yaml_path",
        sorted(COMPUTE_DIR.glob("*.yaml")),
        ids=lambda p: p.name,
    )
    def test_template_validates_via_pydantic(  # noqa: DOC101,DOC103
        self, yaml_path: Path
    ) -> None:
        """Each existing template loads + validates without raising."""
        cfg = load_compute_config_yaml(yaml_path)

        assert isinstance(cfg, ComputeConfig)
        assert cfg.run, f"{yaml_path.name}: run block is required"
        assert cfg.envs is not None


class TestComputeConfigFromCfg:
    """compute_config_from_cfg resolves cfg.compute_template (a name) to a ComputeConfig."""

    def test_resolves_template_name_to_yaml(self) -> None:
        """Hydra-composed cfg.compute_template="runpod-template" loads the runpod template."""
        cfg = OmegaConf.create({"compute_template": "runpod-template"})

        result = compute_config_from_cfg(cfg, compute_dir=COMPUTE_DIR)

        assert isinstance(result, ComputeConfig)
        assert result.resources["cloud"] == "runpod"

    def test_override_selects_different_template(self) -> None:
        """A different `cfg.compute_template` value resolves to a different YAML file."""
        cfg = OmegaConf.create({"compute_template": "oci-cpu-template"})

        result = compute_config_from_cfg(cfg, compute_dir=COMPUTE_DIR)

        # OCI templates use the `any_of` shape, not a flat resources dict.
        assert "any_of" in result.resources
        assert result.resources["any_of"][0]["cloud"] == "oci"

    def test_missing_compute_key_raises(self) -> None:
        """A cfg without a `compute_template` field is a programming error — surface loudly."""
        cfg = OmegaConf.create({"other": "thing"})

        with pytest.raises(KeyError, match="compute_template"):
            compute_config_from_cfg(cfg, compute_dir=COMPUTE_DIR)

    def test_empty_name_raises(self) -> None:
        """An empty `cfg.compute_template` is rejected, not silently resolved to ``.yaml``."""
        cfg = OmegaConf.create({"compute_template": ""})

        with pytest.raises(ValueError, match="compute"):
            compute_config_from_cfg(cfg, compute_dir=COMPUTE_DIR)

    def test_unknown_template_raises_filenotfound(  # noqa: DOC101,DOC103
        self, tmp_path: Path
    ) -> None:
        """Naming a missing template raises FileNotFoundError instead of silently falling back."""
        cfg = OmegaConf.create({"compute_template": "no-such-template"})

        with pytest.raises(FileNotFoundError):
            compute_config_from_cfg(cfg, compute_dir=tmp_path)

    @pytest.mark.parametrize(
        "name",
        [
            "../etc/passwd",
            "subdir/template",
            "subdir\\template",
            ".hidden",
            "..",
        ],
    )
    def test_path_traversing_name_rejected(  # noqa: DOC101,DOC103
        self, tmp_path: Path, name: str
    ) -> None:
        """Names with path separators or leading dots are rejected, not joined to compute_dir."""
        cfg = OmegaConf.create({"compute_template": name})

        with pytest.raises(ValueError, match="filename stem"):
            compute_config_from_cfg(cfg, compute_dir=tmp_path)

    def test_trailing_yaml_extension_stripped(self) -> None:
        """``compute_template=runpod-template.yaml`` resolves the same as without the suffix."""
        cfg = OmegaConf.create({"compute_template": "runpod-template.yaml"})

        result = compute_config_from_cfg(cfg, compute_dir=COMPUTE_DIR)

        assert isinstance(result, ComputeConfig)
        assert result.resources["cloud"] == "runpod"


class TestComputeConfigDictRoundTrip:
    """model_dump() of a ComputeConfig is acceptable to sky.Task.from_yaml_config."""

    def test_round_trip_preserves_raw_yaml(self) -> None:
        """Loading a YAML file and dumping the model yields the same dict shape."""
        path = COMPUTE_DIR / "runpod-template.yaml"
        raw = yaml.safe_load(path.read_text())

        cfg = ComputeConfig(**raw)
        dumped = cfg.model_dump(exclude_none=True)

        assert dumped["resources"] == raw["resources"]
        assert dumped["envs"] == raw["envs"]
        assert dumped["run"] == raw["run"]
        assert dumped["setup"] == raw["setup"]
