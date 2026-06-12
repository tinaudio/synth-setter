"""Post-draft verification harness for ``synth-setter-introspect-plugin`` output.

Runs the validation battery against a just-registered synth and writes a
findings ``.md`` report to disk. Catches the failure modes surfaced while
verifying PR #1662 — codespell/large-file commit blockers, non-portable
absolute ``plugin_path``, and classifier mistakes (continuous knobs drafted as
huge onehots; binary/discrete float switches drafted as full-range continuous).

Usage::

    python agent/verify/verify_introspected_spec.py <spec_name> \
        [--plugin-path plugins/<x>.vst3] [--out <report.md>]

Static checks (always): pre-commit gate on the emitted files, registry import +
``sample()``, Hydra ``render=<name>`` compose into a strict ``RenderConfig``,
spec file size, oversized categoricals, duplicate names, absolute/unknown
provenance. Deep classifier audit (only with ``--plugin-path``): cross-checks
each drafted parameter against the host's ``valid_values`` cardinality.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from synth_setter.data.vst.registration import find_repo_root, registration_paths

_LARGE_FILE_LIMIT_BYTES = 500 * 1024
_ONEHOT_CARDINALITY_WARN = (
    32  # above this, a onehot categorical is almost certainly a numeric sweep
)


@dataclass
class Report:
    """Accumulates per-check verdicts for one synth.

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

        :param msg: Human-readable description of the blocking finding.
        """
        self.blocks.append(msg)

    def warn(self, msg: str) -> None:
        """Record a finding that needs hand-tuning but does not block a commit.

        :param msg: Human-readable description of the warning.
        """
        self.warns.append(msg)

    def ok(self, msg: str) -> None:
        """Record a passing check.

        :param msg: Human-readable description of the passing check.
        """
        self.passes.append(msg)

    def verdict(self) -> str:
        """Return the overall verdict string from the recorded findings.

        :returns: ``BLOCKED`` if any block, else ``COMMITTABLE`` if any warn, else ``CLEAN``.
        """
        if self.blocks:
            return "BLOCKED — not committable as-is"
        if self.warns:
            return "COMMITTABLE with WARN findings (hand-tuning needed)"
        return "CLEAN"


def _check_precommit(repo_root: Path, files: list[Path], report: Report) -> None:
    """Run the repo pre-commit hooks on the emitted files; record any failures.

    :param repo_root: Checkout root the hooks run in.
    :param files: Emitted files to check.
    :param report: Accumulator updated in place.
    """
    present = [str(f) for f in files if f.exists()]
    proc = subprocess.run(  # noqa: S603 — fixed argv, internal checkout paths only
        [shutil.which("pre-commit") or "pre-commit", "run", "--files", *present],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        report.ok(f"pre-commit gate passed on {len(present)} emitted file(s)")
        return
    failed = re.findall(r"^(.*?)\.{3,}Failed$", proc.stdout, re.M)
    detail = ", ".join(h.strip() for h in failed) or "see output"
    report.block(f"pre-commit gate FAILED ({detail}) — output cannot be committed")
    for line in proc.stdout.splitlines():
        if "==>" in line or "exceeds" in line:
            report.block(f"  └ {line.strip()}")


def _check_import_and_sample(spec_name: str, report: Report) -> None:
    """Import the registry, fetch the spec, and draw one sample.

    :param spec_name: Registry key of the verified synth.
    :param report: Accumulator updated in place.
    """
    try:
        from synth_setter.data.vst.param_spec_registry import param_specs, preset_paths

        spec = param_specs[spec_name]
        assert spec_name in preset_paths, "missing preset_paths entry"
        spec.sample()
        report.ok(f"registry import + sample() OK (encoded width {len(spec)})")
    except Exception as exc:  # noqa: BLE001 — surface any failure as a block
        report.block(f"registry import / sample() FAILED: {exc!r}")


def _check_render_compose(repo_root: Path, spec_name: str, report: Report) -> None:
    """Hydra-compose ``render=<name>`` and validate into a strict ``RenderConfig``.

    :param repo_root: Checkout root holding ``src/synth_setter/configs``.
    :param spec_name: Render group key to compose.
    :param report: Accumulator updated in place.
    """
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        from synth_setter.pipeline.schemas.spec import RenderConfig

        GlobalHydra.instance().clear()
        cfg_dir = str((repo_root / "src/synth_setter/configs").resolve())
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
            cfg = compose(overrides=[f"+render={spec_name}"])
        raw = OmegaConf.to_container(cfg.render, resolve=True)
        assert isinstance(raw, dict), "render group did not resolve to a mapping"
        render = RenderConfig(**{k: v for k, v in raw.items() if isinstance(k, str)})
    except Exception as exc:  # noqa: BLE001
        report.block(f"Hydra compose / RenderConfig FAILED: {exc!r}")
        return
    report.ok(f"Hydra render={spec_name} composes into a valid RenderConfig")
    if Path(render.plugin_path).is_absolute():
        report.warn(
            f"render plugin_path is absolute and host-specific: {render.plugin_path!r} "
            "(expected repo-relative 'plugins/<x>.vst3'); not portable across machines"
        )
    if render.renderer_version == "unknown":
        report.warn("renderer_version is 'unknown' — generate cross-checks this; pin it by hand")


def _check_spec_text(spec_module: Path, report: Report) -> None:
    """Analyze the emitted module statically: size, oversized categoricals, dup names.

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
    oversized = [
        (name, body.count(","))
        for name, body in re.findall(r'name="([^"]+)",\s*\n\s*values=\[(.*?)\],', src, re.S)
        if body.count(",") > _ONEHOT_CARDINALITY_WARN
    ]
    for name, n in oversized:
        report.warn(f"categorical '{name}' has ~{n} onehot values — likely a continuous knob")
    names = re.findall(r'name="([^"]+)"', src)
    dupes = {n for n in names if names.count(n) > 1}
    for n in sorted(dupes):
        report.warn(f"duplicate parameter name emitted: '{n}'")


def _check_classifier_against_plugin(plugin_path: str, spec_module: Path, report: Report) -> None:
    """Cross-check drafted types against the host's ``valid_values`` cardinality.

    Catches binary/discrete float params flattened to ``ContinuousParameter`` —
    the inverse of the oversized-onehot case, and invisible to spec-text analysis.

    :param plugin_path: Path of the ``.vst3`` to load for ground-truth cardinality.
    :param spec_module: Path of the drafted spec module.
    :param report: Accumulator updated in place.
    """
    try:
        from pedalboard import VST3Plugin

        from synth_setter.data.vst.introspect import IntrospectablePlugin

        # Cast: pedalboard's plugin surface is dynamic, so VST3Plugin's stubs
        # don't declare the attributes IntrospectablePlugin pins structurally.
        plugin = cast(IntrospectablePlugin, VST3Plugin(plugin_path))
    except Exception as exc:  # noqa: BLE001
        report.warn(f"deep classifier audit skipped — plugin load failed: {exc!r}")
        return
    src = spec_module.read_text(encoding="utf-8")
    continuous = set(re.findall(r'ContinuousParameter\(name="([^"]+)"', src))
    binary, discrete = [], []
    for name, par in plugin.parameters.items():
        try:
            is_float = par.type is float
            n_valid = len(par.valid_values)
        except Exception:  # noqa: BLE001, S112 — one bad param must not abort the audit
            continue
        if name in continuous and is_float:
            if n_valid == 2:
                binary.append(name)
            elif 3 <= n_valid <= 16:
                discrete.append((name, n_valid))
    if binary:
        report.warn(
            f"{len(binary)} float param(s) with exactly 2 valid_values drafted as continuous "
            f"(likely on/off switches): {', '.join(binary[:8])}"
            + (" …" if len(binary) > 8 else "")
        )
    if discrete:
        shown = ", ".join(f"{n}({k})" for n, k in discrete[:8])
        report.warn(
            f"{len(discrete)} float param(s) with 3-16 valid_values drafted as continuous "
            f"(likely discrete selectors): {shown}"
        )


def _render_markdown(report: Report, files: list[Path]) -> str:
    """Render the accumulated verdicts as a markdown report.

    :param report: The accumulated findings.
    :param files: Emitted artifact paths listed in the report header.
    :returns: The markdown report body.
    """
    lines = [
        f"# Introspection verification — `{report.spec_name}`",
        "",
        f"**Verdict: {report.verdict()}**",
        "",
        "Emitted artifacts:",
        *[f"- `{f}`" for f in files],
        "",
    ]
    for title, items in (
        ("BLOCK", report.blocks),
        ("WARN", report.warns),
        ("PASS", report.passes),
    ):
        if items:
            lines.append(f"## {title} ({len(items)})")
            lines.extend(f"- {it}" for it in items)
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run the verification battery and write the findings report.

    :returns: ``1`` if any BLOCK finding was recorded, else ``0``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_name", help="Registry key of the just-drafted synth.")
    parser.add_argument("--plugin-path", default=None, help="Enables the deep classifier audit.")
    parser.add_argument("--out", default=None, help="Report path (default: verify-<name>.md).")
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    if repo_root is None:
        sys.stderr.write("not inside a synth-setter checkout\n")
        return 2
    paths = registration_paths(repo_root, args.spec_name)
    files = [paths.spec_module, paths.preset, paths.csv, paths.render_config, paths.registry]

    report = Report(args.spec_name)
    _check_precommit(repo_root, [paths.spec_module, paths.render_config, paths.csv], report)
    _check_spec_text(paths.spec_module, report)
    _check_import_and_sample(args.spec_name, report)
    _check_render_compose(repo_root, args.spec_name, report)
    if args.plugin_path:
        _check_classifier_against_plugin(args.plugin_path, paths.spec_module, report)

    out = Path(args.out or repo_root / f"verify-{args.spec_name}.md")
    out.write_text(_render_markdown(report, files), encoding="utf-8")
    sys.stdout.write(f"{report.verdict()} — report: {out}\n")
    return 1 if report.blocks else 0


if __name__ == "__main__":
    raise SystemExit(main())
