"""Post-draft verification battery behind the introspect CLI's ``--verify``.

Re-runs, in one pass, the checks that previously required a manual sweep
after every draft (issue #1596 verification loop): the repo's pre-commit
gates (codespell labels, oversized files), spec-text sanity (size, sweep-wide
onehots, duplicate names), registry import + ``sample()``, Hydra
``render=<name>`` composition into a strict ``RenderConfig``, provenance
portability, and a classifier audit against the live plugin. Findings carry a
BLOCK / WARN / PASS severity and render to a markdown report.

The import and Hydra checks run in a clean subprocess with ``PYTHONPATH``
pointed at the target checkout — the CLI process imported
``param_spec_registry`` before ``--register`` rewrote it, so an in-process
import would see the stale registry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from synth_setter.data.vst.introspect import (
    MAX_NUMERIC_CATEGORY_VALUES,
    MAX_STR_CATEGORY_VALUES,
)
from synth_setter.data.vst.registration import registration_paths

if TYPE_CHECKING:
    from synth_setter.data.vst.introspect import IntrospectablePlugin

# Mirrors the check-added-large-files pre-commit hook's limit.
_LARGE_FILE_LIMIT_BYTES = 500 * 1024


@dataclass
class VerificationReport:
    """Accumulates per-check findings for one registered synth.

    .. attribute :: spec_name

       Registry key of the verified synth.

    .. attribute :: blocks

       Findings that make the output uncommittable.

    .. attribute :: warns

       Findings that need hand-tuning but do not block a commit.

    .. attribute :: passes

       Checks that passed.
    """

    spec_name: str
    blocks: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)

    def block(self, msg: str) -> None:
        """Record a finding that makes the output uncommittable.

        :param msg: Description of the blocking finding.
        """
        self.blocks.append(msg)

    def warn(self, msg: str) -> None:
        """Record a finding that needs hand-tuning but does not block a commit.

        :param msg: Description of the warning.
        """
        self.warns.append(msg)

    def ok(self, msg: str) -> None:
        """Record a passing check.

        :param msg: Description of the passing check.
        """
        self.passes.append(msg)

    def verdict(self) -> str:
        """Summarize the recorded findings into one verdict line.

        :returns: ``BLOCKED`` if any block, else ``COMMITTABLE`` if any warn,
            else ``CLEAN``.
        """
        if self.blocks:
            return "BLOCKED — not committable as-is"
        if self.warns:
            return "COMMITTABLE with WARN findings (hand-tuning needed)"
        return "CLEAN"

    def to_markdown(self, files: list[Path]) -> str:
        """Render the findings as a markdown report.

        :param files: Emitted artifact paths listed in the report header.
        :returns: The markdown report body.
        """
        lines = [
            f"# Introspection verification — `{self.spec_name}`",
            "",
            f"**Verdict: {self.verdict()}**",
            "",
            "Emitted artifacts:",
            *[f"- `{f}`" for f in files],
            "",
        ]
        for title, items in (("BLOCK", self.blocks), ("WARN", self.warns), ("PASS", self.passes)):
            if items:
                lines.append(f"## {title} ({len(items)})")
                lines.extend(f"- {item}" for item in items)
                lines.append("")
        return "\n".join(lines)


def verify_registration(
    root: Path, spec_name: str, plugin: IntrospectablePlugin
) -> VerificationReport:
    """Run the full battery against a just-registered synth.

    :param root: Checkout root the artifacts were registered into.
    :param spec_name: Registry key of the registered synth.
    :param plugin: The still-loaded plugin, for the classifier audit.
    :returns: The accumulated findings.
    """
    paths = registration_paths(root, spec_name)
    report = VerificationReport(spec_name)
    _check_precommit(root, [paths.spec_module, paths.render_config, paths.csv], report)
    check_spec_text(paths.spec_module, report)
    _check_runtime(root, spec_name, report)
    check_classifier_against_plugin(plugin, paths.spec_module, report)
    return report


def _check_precommit(root: Path, files: list[Path], report: VerificationReport) -> None:
    """Run the repo pre-commit hooks on the emitted files.

    Commit-time failures (codespell on host labels, oversized files) are the
    costliest to discover late, so they surface here as BLOCKs. Environments
    where the hooks cannot run at all (no ``pre-commit``, ``root`` is not a
    git repo) degrade to a WARN instead of a false BLOCK.

    :param root: Checkout root the hooks run in.
    :param files: Emitted files to check.
    :param report: Accumulator updated in place.
    """
    present = [str(f) for f in files if f.exists()]
    exe = shutil.which("pre-commit")
    if exe is None:
        report.warn("pre-commit gate skipped — pre-commit not on PATH")
        return
    proc = subprocess.run(  # noqa: S603 — fixed argv, checkout-internal paths only
        [exe, "run", "--files", *present],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        report.ok(f"pre-commit gate passed on {len(present)} emitted file(s)")
        return
    failed = re.findall(r"^(.*?)\.{3,}Failed$", proc.stdout, re.M)
    if not failed:
        detail = next(iter((proc.stderr or proc.stdout).strip().splitlines()), "unknown error")
        report.warn(f"pre-commit gate skipped — could not run in {root}: {detail}")
        return
    hooks = ", ".join(name.strip() for name in failed)
    report.block(f"pre-commit gate FAILED ({hooks}) — output cannot be committed")
    for line in proc.stdout.splitlines():
        if "==>" in line or "exceeds" in line:
            report.block(f"  └ {line.strip()}")


def check_spec_text(spec_module: Path, report: VerificationReport) -> None:
    """Analyze the emitted module statically: size, sweep-wide onehots, duplicate names.

    :param spec_module: Path of the drafted spec module.
    :param report: Accumulator updated in place.
    """
    if not spec_module.exists():
        report.block(f"spec module missing: {spec_module}")
        return
    src = spec_module.read_text(encoding="utf-8")
    size = len(src.encode("utf-8"))
    if size > _LARGE_FILE_LIMIT_BYTES:
        report.block(
            f"spec module is {size // 1024} KB (> {_LARGE_FILE_LIMIT_BYTES // 1024} KB "
            "check-added-large-files limit)"
        )
    for name, body in re.findall(r'name="([^"]+)",\s*\n\s*values=\[(.*?)\],', src, re.S):
        if body.count(",") > MAX_STR_CATEGORY_VALUES:
            report.warn(
                f"categorical '{name}' has ~{body.count(',')} onehot values — "
                "likely a continuous knob"
            )
    name_counts = Counter(re.findall(r'name="([^"]+)"', src))
    for name in sorted(n for n, count in name_counts.items() if count > 1):
        report.warn(f"duplicate parameter name emitted: '{name}'")


def _check_runtime(root: Path, spec_name: str, report: VerificationReport) -> None:
    """Import the registry, sample the spec, and Hydra-compose the render config.

    Runs in a clean subprocess with ``PYTHONPATH`` pointed at ``root``'s
    ``src`` so the probe sees the just-rewritten registry, exactly the way
    ``generate_dataset`` will.

    :param root: Checkout root holding the registry and Hydra configs.
    :param spec_name: Registry key to import and compose.
    :param report: Accumulator updated in place.
    """
    probe = textwrap.dedent(
        f"""
        import json

        from synth_setter.data.vst.param_spec_registry import param_specs, preset_paths

        spec = param_specs[{spec_name!r}]
        assert {spec_name!r} in preset_paths, "missing preset_paths entry"
        spec.sample()

        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from synth_setter.pipeline.schemas.spec import RenderConfig

        with initialize_config_dir(config_dir={str(root / "src/synth_setter/configs")!r},
                                   version_base="1.3"):
            cfg = compose(overrides=["+render=" + {spec_name!r}])
        raw = OmegaConf.to_container(cfg.render, resolve=True)
        render = RenderConfig(**{{k: v for k, v in raw.items() if isinstance(k, str)}})
        print(json.dumps({{
            "encoded_width": len(spec),
            "plugin_path": render.plugin_path,
            "renderer_version": render.renderer_version,
        }}))
        """
    )
    proc = subprocess.run(  # noqa: S603 — fixed argv, runs our own probe source
        [sys.executable, "-c", probe],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no stderr"
        report.block(f"registry import / sample() / Hydra compose FAILED: {detail}")
        return
    probe_result = json.loads(proc.stdout)
    report.ok(f"registry import + sample() OK (encoded width {probe_result['encoded_width']})")
    report.ok(f"Hydra render={spec_name} composes into a valid RenderConfig")
    if Path(probe_result["plugin_path"]).is_absolute():
        report.warn(
            f"render plugin_path is absolute and host-specific: {probe_result['plugin_path']!r} "
            "(expected repo-relative 'plugins/<x>.vst3'); not portable across machines"
        )
    if probe_result["renderer_version"] == "unknown":
        report.warn("renderer_version is 'unknown' — generate cross-checks this; pin it by hand")


def check_classifier_against_plugin(
    plugin: IntrospectablePlugin, spec_module: Path, report: VerificationReport
) -> None:
    """Cross-check drafted classes against the host's ``valid_values`` cardinality.

    Flags numeric switches/selectors drafted as continuous — invisible to spec-text analysis and
    the regression the cardinality caps exist to stop. str-typed parameters drafted continuous are
    the cap working as intended (formatted numeric sweeps) and are not flagged.

    :param plugin: The loaded plugin providing ground-truth cardinality.
    :param spec_module: Path of the drafted spec module.
    :param report: Accumulator updated in place.
    """
    src = spec_module.read_text(encoding="utf-8")
    continuous = set(re.findall(r'ContinuousParameter\(\s*name="([^"]+)"', src))
    binary: list[str] = []
    discrete: list[tuple[str, int]] = []
    for name, param in plugin.parameters.items():
        # Broad catch: one unreadable parameter must not abort the audit.
        try:
            is_numeric = param.type not in (str, bool)
            n_valid = len(param.valid_values)
        except Exception:  # noqa: BLE001, S112
            continue
        if name in continuous and is_numeric:
            if n_valid == 2:
                binary.append(name)
            elif n_valid <= MAX_NUMERIC_CATEGORY_VALUES:
                discrete.append((name, n_valid))
    if binary:
        shown = ", ".join(binary[:8]) + (" …" if len(binary) > 8 else "")
        report.warn(
            f"{len(binary)} numeric param(s) with exactly 2 valid_values drafted as "
            f"continuous (likely on/off switches): {shown}"
        )
    if discrete:
        shown = ", ".join(f"{name}({n})" for name, n in discrete[:8])
        report.warn(
            f"{len(discrete)} numeric param(s) with 3-{MAX_NUMERIC_CATEGORY_VALUES} "
            f"valid_values drafted as continuous (likely discrete selectors): {shown}"
        )
