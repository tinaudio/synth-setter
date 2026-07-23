#!/usr/bin/env python3
"""Supervise deferred Pi review passes outside the foreground host lifetime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

if __package__:
    from agent._shared.pi_review_routing import (
        PINNED_REVIEW_MODELS,
        WorkerFinding,
        WorkerReport,
        extract_report,
        parse_worker_report,
    )
else:
    from pi_review_routing import (
        PINNED_REVIEW_MODELS,
        WorkerFinding,
        WorkerReport,
        extract_report,
        parse_worker_report,
    )

_AFTERCARE_MODEL = "gpt-5.6-terra"
_AFTERCARE_PROVIDER = "openai-codex"
_AFTERCARE_THINKING = "medium"
_FOREGROUND_STOPPED_ENV = "SYNTH_SETTER_PI_REVIEW_FOREGROUND_STOPPED"
_RUNTIME_MANIFEST_ENV = "PI_REVIEW_AFTERCARE_RUNTIME_MANIFEST"
_MAX_LOG_BYTES = 64 * 1024
_LOG_TAIL_CHARS = 16 * 1024
_CAPACITY_MARKERS = (
    "resourceexhausted",
    "resource exhausted",
    "worker limit",
    "maximum concurrent workers",
)

type AttemptStatus = Literal[
    "adopted-foreground-result",
    "terminated-original-worker",
    "success",
    "failed",
    "stale",
    "verified",
    "rejected",
    "malformed-report",
]
type DiagnosticCategory = Literal[
    "capacity",
    "child-exit",
    "invalid-result",
    "missing-result",
    "ownership",
    "supervisor-error",
]


class DeferredPass(BaseModel, strict=True, extra="forbid"):
    """One model pass transferred from foreground review to aftercare.

    .. attribute :: skill

        Assigned checklist.

    .. attribute :: pass_name

        Logical independent review pass.

    .. attribute :: origin

        Whether the pass uses its primary provider or a Codex fallback.

    .. attribute :: model

        Exact pinned model selector.

    .. attribute :: verification_model

        Foreground Codex model used to verify free-pool findings.

    .. attribute :: thinking

        Thinking level selected by the routing plan.

    .. attribute :: agent_id

        Optional Tintin foreground worker identifier.

    .. attribute :: output_path

        Optional Tintin foreground transcript path.
    """

    skill: str = Field(min_length=1)
    pass_name: Literal["codex", "free-pool"]
    origin: Literal["primary", "codex-fallback"]
    model: str = Field(min_length=1)
    verification_model: str = Field(min_length=1)
    thinking: Literal["low", "medium", "high"]
    agent_id: str | None = Field(default=None, min_length=1)
    output_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _require_valid_ownership_and_model(self) -> DeferredPass:
        """Reject incomplete ownership handles and models outside the routing pool.

        :returns: Validated deferred pass.
        :raises ValueError: If ownership or model provenance is inconsistent.
        """
        if (self.agent_id is None) != (self.output_path is None):
            raise ValueError("Deferred pass agent_id and output_path must be provided together")
        if self.model not in PINNED_REVIEW_MODELS:
            raise ValueError("Deferred pass model is outside the pinned review pool")
        is_codex = self.model.startswith("openai-codex/")
        verification_is_codex = self.verification_model.startswith("openai-codex/")
        if self.verification_model not in PINNED_REVIEW_MODELS or not verification_is_codex:
            raise ValueError("Deferred pass requires a pinned Codex verification model")
        codex_label = "codex"
        expected_codex_origin = self.pass_name == codex_label or self.origin == "codex-fallback"
        if is_codex != expected_codex_origin:
            raise ValueError("Deferred pass model origin does not match its provider family")
        if self.pass_name == codex_label and self.origin != "primary":
            raise ValueError("Deferred Codex pass cannot be labeled as a fallback")
        return self


class AftercareManifest(BaseModel, strict=True, extra="forbid"):
    """Durable handoff from foreground review to deferred review work.

    .. attribute :: version

        Manifest schema version.

    .. attribute :: mode

        Foreground delivery mode.

    .. attribute :: repo

        GitHub repository in ``owner/name`` form.

    .. attribute :: pr_number

        Pull request receiving late findings.

    .. attribute :: base_sha

        Reviewed base commit.

    .. attribute :: head_sha

        Reviewed PR head.

    .. attribute :: target

        Worker target label.

    .. attribute :: deferred_passes

        Incomplete independent passes.

    .. attribute :: foreground_fingerprints

        Stable identities of findings already delivered.
    """

    version: Literal[1]
    mode: Literal["full", "no-comments"]
    repo: str = Field(min_length=1)
    pr_number: int = Field(gt=0)
    base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    target: str = Field(min_length=1)
    deferred_passes: tuple[DeferredPass, ...] = Field(min_length=1)
    foreground_fingerprints: tuple[str, ...]


class AftercareAttempt(BaseModel, strict=True, extra="forbid"):
    """One auditable foreground-ownership or aftercare attempt row.

    .. attribute :: skill

        Assigned checklist.

    .. attribute :: pass_name

        Logical pass or verification label.

    .. attribute :: model

        Effective model selector.

    .. attribute :: status

        Attempt outcome.

    .. attribute :: agent_id

        Tintin worker identifier when available.

    .. attribute :: output_path

        Worker transcript path when available.

    .. attribute :: detail

        Exact outcome diagnostic or validation evidence.
    """

    skill: str = Field(min_length=1)
    pass_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    status: AttemptStatus
    agent_id: str | None = Field(default=None, min_length=1)
    output_path: str | None = Field(default=None, min_length=1)
    detail: str = Field(min_length=1)


class AftercareDiagnostic(BaseModel, strict=True, extra="forbid"):
    """One supervisor or provider diagnostic.

    .. attribute :: category

        Machine-readable failure class.

    .. attribute :: message

        Actionable diagnostic text.
    """

    category: DiagnosticCategory
    message: str = Field(min_length=1)


class AftercareResult(BaseModel, strict=True, extra="forbid"):
    """Strict result atomically published by the aftercare supervisor.

    .. attribute :: status

        Overall aftercare outcome.

    .. attribute :: attempts

        Foreground ownership and aftercare audit rows.

    .. attribute :: diagnostics

        Supervisor and provider diagnostics.

    .. attribute :: late_findings

        Validated late findings retained by aftercare.

    .. attribute :: posted_review_url

        Posted review URL in full mode, if any.

    .. attribute :: child_exit_code

        Pi child exit code, including negative signal numbers.

    .. attribute :: log_tail

        Bounded tail of persisted child output.

    .. attribute :: completed_at

        UTC completion timestamp.
    """

    status: Literal["complete", "stale", "failed"]
    attempts: tuple[AftercareAttempt, ...]
    diagnostics: tuple[AftercareDiagnostic, ...]
    late_findings: tuple[WorkerFinding, ...]
    posted_review_url: str | None
    child_exit_code: int | None
    log_tail: str = Field(max_length=_LOG_TAIL_CHARS)
    completed_at: datetime

    @model_validator(mode="after")
    def _require_failure_diagnostic(self) -> AftercareResult:
        """Reject failed results without actionable diagnostics.

        :returns: Validated aftercare result.
        :raises ValueError: If a failed result omits diagnostics.
        """
        if self.status == "failed" and not self.diagnostics:
            raise ValueError("failed result requires a diagnostic")
        return self


@dataclass(frozen=True, slots=True)
class _OwnershipPlan:
    """Validated ownership state before a new Pi process may launch.

    .. attribute :: deferred_passes

        Rows requiring a fresh owner after foreground shutdown.

    .. attribute :: adopted_passes

        Rows with valid foreground reports.

    .. attribute :: attempts

        Supervisor-owned audit rows.

    .. attribute :: adopted_reports

        Validated foreground reports paired with adopted passes.

    .. attribute :: blocked

        Whether ownership could not be transferred safely.
    """

    deferred_passes: tuple[DeferredPass, ...]
    adopted_passes: tuple[DeferredPass, ...]
    attempts: tuple[AftercareAttempt, ...]
    adopted_reports: tuple[WorkerReport, ...]
    blocked: bool


@dataclass(frozen=True, slots=True)
class _SupervisorPaths:
    """Sidecars owned by one supervisor invocation.

    .. attribute :: canonical_result

        Atomically published result path.

    .. attribute :: log

        Bounded child output path.

    .. attribute :: runtime_manifest

        Supervisor-generated child manifest path.

    .. attribute :: runtime_result

        Model-written result path awaiting validation.
    """

    canonical_result: Path
    log: Path
    runtime_manifest: Path
    runtime_result: Path


def _sidecar_path(manifest_path: Path, suffix: str) -> Path:
    """Append a suffix without replacing the manifest extension.

    :param manifest_path: Foreground manifest path.
    :param suffix: Complete suffix appended to the filename.
    :returns: Derived sidecar path.
    """
    return Path(f"{manifest_path}{suffix}")


def load_manifest(path: Path) -> AftercareManifest:
    """Load a strict manifest confined to the current worktree review directory.

    :param path: Manifest path under the current worktree's ``.agent-reviews``.
    :returns: Strict deferred-review manifest.
    :raises ValueError: If the path escapes ``.agent-reviews``.
    """
    resolved = path.resolve()
    review_dir = (Path.cwd() / ".agent-reviews").resolve()
    if not resolved.is_relative_to(review_dir):
        raise ValueError("Aftercare manifest must be under .agent-reviews")
    return AftercareManifest.model_validate_json(resolved.read_text())


def build_command(manifest_path: Path, adopted_passes: tuple[DeferredPass, ...] = ()) -> list[str]:
    """Build the pinned Pi child command for one supervisor-owned runtime manifest.

    :param manifest_path: Validated runtime manifest path.
    :param adopted_passes: Rows whose valid foreground reports must not be relaunched.
    :returns: Argument vector for Pi's headless aftercare session.
    :raises RuntimeError: If Pi is unavailable.
    """
    pi = shutil.which("pi")
    if pi is None:
        raise RuntimeError("pi executable not found on PATH")
    adopted = ", ".join(f"{row.skill}/{row.pass_name}" for row in adopted_passes)
    ownership_instruction = ""
    if adopted:
        ownership_instruction = (
            f" The supervisor adopted foreground reports for: {adopted}. "
            "Use each row's output_path and do not launch those passes again."
        )
    prompt = (
        "Execute the deferred PR-review procedure in "
        "agent/skills/_shared/repo-review-aftercare.md using manifest "
        f"{manifest_path}. Process only its deferred passes, then exit."
        f"{ownership_instruction}"
    )
    return [
        pi,
        "-p",
        "--approve",
        "--provider",
        _AFTERCARE_PROVIDER,
        "--model",
        _AFTERCARE_MODEL,
        "--thinking",
        _AFTERCARE_THINKING,
        "--no-session",
        prompt,
    ]


def build_supervisor_command(manifest_path: Path) -> list[str]:
    """Build the detached Python supervisor command.

    :param manifest_path: Validated foreground manifest path.
    :returns: Argument vector rooted at the active Python interpreter.
    """
    return [sys.executable, str(Path(__file__).resolve()), "--supervise", str(manifest_path)]


def _atomic_write(path: Path, content: str) -> None:
    """Replace a sidecar only after its complete payload reaches disk.

    :param path: Final sidecar path.
    :param content: Complete text payload.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_result(path: Path, result: AftercareResult) -> None:
    """Validate and atomically publish the canonical result.

    :param path: Canonical result path.
    :param result: Result constructed or validated by the supervisor.
    """
    validated = AftercareResult.model_validate_json(result.model_dump_json())
    _atomic_write(path, f"{validated.model_dump_json(indent=2)}\n")


def _attempt(deferred: DeferredPass, status: AttemptStatus, detail: str) -> AftercareAttempt:
    """Build one ownership audit row from a deferred pass.

    :param deferred: Pass supplying assignment identity.
    :param status: Supervisor-observed lifecycle outcome.
    :param detail: Exact evidence supporting the outcome.
    :returns: Strict attempt row.
    """
    return AftercareAttempt(
        skill=deferred.skill,
        pass_name=deferred.pass_name,
        model=deferred.model,
        status=status,
        agent_id=deferred.agent_id,
        output_path=deferred.output_path,
        detail=detail,
    )


def _adopt_report(deferred: DeferredPass, target: str) -> WorkerReport | None:
    """Return a valid foreground report when its transcript is complete.

    :param deferred: Pass carrying the foreground transcript path.
    :param target: Expected worker assignment target.
    :returns: Valid report, or ``None`` when adoption is unsafe.
    """
    if deferred.output_path is None:
        return None
    output_path = Path(deferred.output_path)
    if not output_path.is_file():
        return None
    try:
        report = extract_report(output_path)
        return parse_worker_report(report, expected_skill=deferred.skill, expected_target=target)
    except (OSError, ValueError):
        return None


def _plan_ownership(manifest: AftercareManifest) -> _OwnershipPlan:
    """Adopt reports and fail closed unless shutdown ended other owners.

    :param manifest: Validated foreground ownership handoff.
    :returns: Adoption, termination, and relaunch plan.
    """
    host_stopped = os.environ.get(_FOREGROUND_STOPPED_ENV) == "1"
    remaining: list[DeferredPass] = []
    adopted_passes: list[DeferredPass] = []
    attempts: list[AftercareAttempt] = []
    adopted: list[WorkerReport] = []
    blocked = False
    for deferred in manifest.deferred_passes:
        report = _adopt_report(deferred, manifest.target)
        if report is not None:
            attempts.append(
                _attempt(
                    deferred,
                    "adopted-foreground-result",
                    "foreground transcript contains a valid report for this pass",
                )
            )
            adopted_passes.append(deferred)
            adopted.append(report)
            continue
        if not host_stopped:
            attempts.append(
                _attempt(
                    deferred,
                    "failed",
                    "foreground owner cannot be stopped through the model-facing Tintin handles",
                )
            )
            blocked = True
            continue
        attempts.append(
            _attempt(
                deferred,
                "terminated-original-worker",
                "foreground Pi exited and Tintin session shutdown aborted remaining workers",
            )
        )
        remaining.append(deferred)
    return _OwnershipPlan(
        deferred_passes=tuple(remaining),
        adopted_passes=tuple(adopted_passes),
        attempts=tuple(attempts),
        adopted_reports=tuple(adopted),
        blocked=blocked,
    )


def _bounded_log_tail(log_path: Path) -> str:
    """Decode the configured character tail from persisted child output.

    :param log_path: Bounded aftercare log path.
    :returns: Latest decoded child-output characters.
    """
    if not log_path.exists():
        return ""
    return log_path.read_bytes()[-_MAX_LOG_BYTES:].decode(errors="replace")[-_LOG_TAIL_CHARS:]


def _run_child(command: list[str], environment: dict[str, str], log_path: Path) -> int:
    """Run Pi while persisting only the newest bounded output bytes.

    :param command: Trusted Pi argument vector.
    :param environment: Child environment with aftercare ownership metadata.
    :param log_path: Bounded stdout and stderr destination.
    :returns: Pi child exit code, including negative signal numbers.
    :raises RuntimeError: If the child output pipe is unavailable.
    """
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=Path.cwd(),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.stdout is None:
        raise RuntimeError("Pi child output pipe was not created")
    tail = bytearray()
    with log_path.open("wb") as log_file:
        while chunk := os.read(process.stdout.fileno(), 8192):
            tail.extend(chunk)
            del tail[:-_MAX_LOG_BYTES]
            log_file.seek(0)
            log_file.truncate()
            log_file.write(tail)
            log_file.flush()
    return process.wait()


def _diagnostics_for_failure(
    *, exit_code: int | None, result_problem: DiagnosticCategory, log_tail: str
) -> tuple[AftercareDiagnostic, ...]:
    """Build diagnostics with a distinct provider-capacity row.

    :param exit_code: Observed Pi exit code when available.
    :param result_problem: Primary supervisor failure category.
    :param log_tail: Bounded child output used for capacity classification.
    :returns: Primary diagnostic followed by optional capacity evidence.
    """
    diagnostics = [
        AftercareDiagnostic(
            category=result_problem,
            message={
                "child-exit": f"Pi aftercare child exited with code {exit_code}",
                "invalid-result": "Pi aftercare child wrote a result that failed strict validation",
                "missing-result": "Pi aftercare child did not write its runtime result",
                "ownership": "A foreground owner could not be proven stopped; duplicate launch refused",
                "supervisor-error": "Pi aftercare supervisor raised before obtaining a valid result",
                "capacity": "Pi aftercare child exhausted provider or worker capacity",
            }[result_problem],
        )
    ]
    lowered = log_tail.lower()
    if any(marker in lowered for marker in _CAPACITY_MARKERS):
        diagnostics.append(
            AftercareDiagnostic(
                category="capacity",
                message="Pi aftercare child exhausted provider or worker capacity",
            )
        )
    return tuple(diagnostics)


def _failed_result(
    *,
    manifest: AftercareManifest | None,
    attempts: tuple[AftercareAttempt, ...],
    diagnostics: tuple[AftercareDiagnostic, ...],
    exit_code: int | None,
    log_tail: str,
) -> AftercareResult:
    """Build a strict failure with a row for every recoverable pass.

    :param manifest: Handoff supplying recoverable pass identity.
    :param attempts: Ownership rows already observed.
    :param diagnostics: Terminal failure evidence.
    :param exit_code: Pi child exit code when available.
    :param log_tail: Bounded persisted child output.
    :returns: Strict failed result.
    """
    failed_attempts = list(attempts)
    attempted_keys = {(row.skill, row.pass_name, row.status) for row in failed_attempts}
    if manifest is not None:
        for deferred in manifest.deferred_passes:
            has_terminal_failure = (deferred.skill, deferred.pass_name, "failed") in attempted_keys
            if not has_terminal_failure:
                failed_attempts.append(_attempt(deferred, "failed", diagnostics[0].message))
    return AftercareResult(
        status="failed",
        attempts=tuple(failed_attempts),
        diagnostics=diagnostics,
        late_findings=(),
        posted_review_url=None,
        child_exit_code=exit_code,
        log_tail=log_tail,
        completed_at=datetime.now(UTC),
    )


def _merge_child_result(
    child_result: AftercareResult,
    ownership_attempts: tuple[AftercareAttempt, ...],
    *,
    exit_code: int,
    log_tail: str,
) -> AftercareResult:
    """Attach supervisor lifecycle evidence to a valid model result.

    :param child_result: Strict model-written result.
    :param ownership_attempts: Supervisor-owned adoption and termination rows.
    :param exit_code: Observed successful Pi exit code.
    :param log_tail: Bounded persisted child output.
    :returns: Revalidated canonical result.
    """
    merged = child_result.model_copy(
        update={
            "attempts": (*ownership_attempts, *child_result.attempts),
            "child_exit_code": exit_code,
            "log_tail": log_tail,
        }
    )
    return AftercareResult.model_validate(merged.model_dump())


def _load_child_result(path: Path) -> AftercareResult:
    """Validate model output with the canonical result model.

    :param path: Model-written runtime result path.
    :returns: Strict aftercare result.
    """
    return AftercareResult.model_validate_json(path.read_text())


def _supervisor_paths(manifest_path: Path) -> _SupervisorPaths:
    """Derive sidecars from the immutable foreground manifest path.

    :param manifest_path: Foreground manifest path.
    :returns: Canonical, runtime, and log sidecar paths.
    """
    runtime_manifest = _sidecar_path(manifest_path, ".supervised.json")
    return _SupervisorPaths(
        canonical_result=_sidecar_path(manifest_path, ".result.json"),
        log=_sidecar_path(manifest_path, ".aftercare.log"),
        runtime_manifest=runtime_manifest,
        runtime_result=_sidecar_path(runtime_manifest, ".result.json"),
    )


def _prelaunch_result(
    manifest: AftercareManifest, ownership: _OwnershipPlan
) -> AftercareResult | None:
    """Resolve ownership outcomes before child launch.

    :param manifest: Validated foreground handoff.
    :param ownership: Supervisor ownership plan.
    :returns: Terminal prelaunch result, or ``None`` when Pi must run.
    """
    if ownership.blocked:
        diagnostics = _diagnostics_for_failure(
            exit_code=None,
            result_problem="ownership",
            log_tail="",
        )
        return _failed_result(
            manifest=manifest,
            attempts=ownership.attempts,
            diagnostics=diagnostics,
            exit_code=None,
            log_tail="",
        )
    adopted_findings = any(report.findings for report in ownership.adopted_reports)
    if ownership.deferred_passes or adopted_findings:
        return None
    return AftercareResult(
        status="complete",
        attempts=ownership.attempts,
        diagnostics=(),
        late_findings=(),
        posted_review_url=None,
        child_exit_code=None,
        log_tail="",
        completed_at=datetime.now(UTC),
    )


def _supervise_child(
    manifest: AftercareManifest,
    ownership: _OwnershipPlan,
    paths: _SupervisorPaths,
) -> AftercareResult:
    """Convert every remaining child outcome into a strict result.

    :param manifest: Validated foreground handoff.
    :param ownership: Adopted and terminated pass ownership plan.
    :param paths: Supervisor-owned sidecar paths.
    :returns: Valid child result or strict synthesized failure.
    """
    paths.runtime_result.unlink(missing_ok=True)
    runtime_passes = (*ownership.deferred_passes, *ownership.adopted_passes)
    runtime_manifest = manifest.model_copy(update={"deferred_passes": runtime_passes})
    _atomic_write(paths.runtime_manifest, f"{runtime_manifest.model_dump_json(indent=2)}\n")
    environment = os.environ.copy()
    environment.pop("SYNTH_SETTER_PI_REVIEW", None)
    environment["SYNTH_SETTER_PI_REVIEW_AFTERCARE"] = "1"
    environment[_RUNTIME_MANIFEST_ENV] = str(paths.runtime_manifest)
    command = build_command(paths.runtime_manifest, ownership.adopted_passes)
    exit_code = _run_child(command, environment, paths.log)
    log_tail = _bounded_log_tail(paths.log)

    problem: DiagnosticCategory | None = None
    if exit_code != 0:
        problem = "child-exit"
    elif not paths.runtime_result.exists():
        problem = "missing-result"
    else:
        try:
            child_result = _load_child_result(paths.runtime_result)
        except (OSError, ValueError):
            problem = "invalid-result"
        else:
            return _merge_child_result(
                child_result,
                ownership.attempts,
                exit_code=exit_code,
                log_tail=log_tail,
            )

    diagnostics = _diagnostics_for_failure(
        exit_code=exit_code,
        result_problem=problem,
        log_tail=log_tail,
    )
    return _failed_result(
        manifest=manifest,
        attempts=ownership.attempts,
        diagnostics=diagnostics,
        exit_code=exit_code,
        log_tail=log_tail,
    )


def supervise_aftercare(manifest_path: Path) -> int:
    """Run one Pi child and atomically guarantee the canonical result sidecar.

    :param manifest_path: Original foreground handoff manifest.
    :returns: Zero for complete or stale aftercare, otherwise one.
    """
    paths = _supervisor_paths(manifest_path)
    manifest: AftercareManifest | None = None
    attempts: tuple[AftercareAttempt, ...] = ()
    result: AftercareResult | None = None
    try:
        manifest = load_manifest(manifest_path)
        ownership = _plan_ownership(manifest)
        attempts = ownership.attempts
        result = _prelaunch_result(manifest, ownership)
        if result is None:
            result = _supervise_child(manifest, ownership, paths)
    except (OSError, RuntimeError, ValueError) as error:
        log_tail = _bounded_log_tail(paths.log)
        result = _failed_result(
            manifest=manifest,
            attempts=attempts,
            diagnostics=(
                AftercareDiagnostic(
                    category="supervisor-error",
                    message=str(error) or type(error).__name__,
                ),
            ),
            exit_code=None,
            log_tail=log_tail,
        )
    finally:
        if result is None:
            result = _failed_result(
                manifest=manifest,
                attempts=attempts,
                diagnostics=(
                    AftercareDiagnostic(
                        category="supervisor-error",
                        message="Supervisor exited without constructing a result",
                    ),
                ),
                exit_code=None,
                log_tail=_bounded_log_tail(paths.log),
            )
        _atomic_write_result(paths.canonical_result, result)
        paths.runtime_result.unlink(missing_ok=True)
        paths.runtime_manifest.unlink(missing_ok=True)
    return 1 if result.status == "failed" else 0


def launch_aftercare(manifest_path: Path) -> int:
    """Start a detached Python supervisor and return its PID.

    :param manifest_path: Absolute path to a valid aftercare manifest.
    :returns: Detached supervisor process identifier.
    """
    load_manifest(manifest_path)
    environment = os.environ.copy()
    environment[_FOREGROUND_STOPPED_ENV] = "1"
    log_path = _sidecar_path(manifest_path, ".aftercare.log")
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(  # noqa: S603
            build_supervisor_command(manifest_path),
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


def _build_parser() -> argparse.ArgumentParser:
    """Build the supervisor and detached-launch argument parser.

    :returns: CLI parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Validate a manifest, then supervise it or print the detached PID.

    :returns: Process exit status.
    """
    args = _build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    if args.supervise:
        return supervise_aftercare(manifest_path)
    if args.dry_run:
        load_manifest(manifest_path)
        sys.stdout.write(f"{json.dumps(build_supervisor_command(manifest_path))}\n")
        return 0
    sys.stdout.write(f"{launch_aftercare(manifest_path)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
