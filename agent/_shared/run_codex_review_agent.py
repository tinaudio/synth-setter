#!/usr/bin/env python3
# Callers run this outside the project environment, so it declares its own deps.
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///
"""Launch a Codex PR-review role with its project-pinned runtime policy."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_TIMEOUT_S = 900
MAX_TIMEOUT_S = 86_400
REVIEW_ROLES = (
    "pr-review-orchestrator",
    "pr-review-worker-deep",
    "pr-review-worker-fast",
)
TERMINATION_GRACE_S = 2
WORKER_TIMEOUT_S = 720


class _AgentConfig(BaseModel):
    """Validated execution policy for one Codex review role.

    .. attribute :: model_config

       Enforces strict input types while allowing future Codex agent fields.
    .. attribute :: name

       Must match the role requested by the launcher.
    .. attribute :: model

       Exact model slug passed to ``codex exec``.
    .. attribute :: model_reasoning_effort

       Exact reasoning effort passed to ``codex exec``.
    .. attribute :: developer_instructions

       Prepended to the worker prompt because subprocess launches do not select
       project custom-agent roles natively.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    name: str
    model: str
    model_reasoning_effort: str
    developer_instructions: str


def _parse_args() -> argparse.Namespace:
    """Parse exactly one prompt source for one review role.

    :returns: Validated launcher arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=REVIEW_ROLES)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    prompt.add_argument("--skill-brief", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_agent(role: str) -> _AgentConfig:
    """Load a role and guard against filename/name drift.

    :param role: Project role whose pinned execution policy is required.
    :returns: Validated model, effort, and instruction settings.
    :raises ValueError: If the file's declared name does not match ``role``.
    """
    path = REPO_ROOT / ".codex" / "agents" / f"{role}.toml"
    with path.open("rb") as file:
        config = _AgentConfig.model_validate(tomllib.load(file))
    if config.name != role:
        raise ValueError(f"agent name {config.name!r} does not match role {role!r}")
    return config


def _extract_skill_brief(path: Path, target: str | None) -> str:
    """Preserve the skill-owned orchestrator prompt across subprocess launch.

    :param path: Full-review skill containing the orchestrator brief.
    :param target: Explicit PR number to substitute, or ``None`` for auto-resolution.
    :returns: The exact orchestrator section, with only the target substituted.
    """
    text = path.read_text()
    marker = "\n## Orchestrator agent brief\n"
    start = text.index(marker) + 1
    end = text.find("\n## ", start + len(marker))
    brief = text[start:] if end == -1 else text[start:end]
    if target is not None:
        brief = brief.replace("<N>", target)
    return brief


def _resolve_prompt(args: argparse.Namespace) -> str:
    """Read prompt text from the mutually exclusive selected source.

    :param args: Parsed launcher arguments.
    :returns: Prompt text for the selected review role.
    """
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return args.prompt_file.read_text()
    return _extract_skill_brief(args.skill_brief, args.target)


def _resolve_timeout(role: str) -> int:
    """Resolve a positive deadline from the role default or environment.

    :param role: Review role being launched.
    :returns: Deadline in seconds.
    :raises ValueError: If ``CODEX_REVIEW_TIMEOUT`` is not a positive integer.
    """
    default = ORCHESTRATOR_TIMEOUT_S if role == "pr-review-orchestrator" else WORKER_TIMEOUT_S
    raw_timeout = os.environ.get("CODEX_REVIEW_TIMEOUT", str(default))
    if not raw_timeout.isascii() or not raw_timeout.isdecimal():
        raise ValueError(f"CODEX_REVIEW_TIMEOUT must be a positive integer, got: {raw_timeout}")
    timeout_s = int(raw_timeout)
    if not 1 <= timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(
            f"CODEX_REVIEW_TIMEOUT must be between 1 and {MAX_TIMEOUT_S}, got: {raw_timeout}"
        )
    return timeout_s


def _wait_for_pid(pid: int, deadline: float) -> int | None:
    """Poll a child until it exits or a monotonic deadline passes.

    :param pid: Direct child process to reap.
    :param deadline: Absolute ``time.monotonic`` deadline.
    :returns: Exit code, or ``None`` when the deadline passes first.
    """
    while True:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid:
            return os.waitstatus_to_exitcode(status)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.05, remaining))


def _terminate_process_group(pid: int) -> None:
    """Stop and reap a timed-out process group.

    :param pid: Process-group leader and direct child to reap.
    """
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + TERMINATION_GRACE_S
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            time.sleep(0.05)
            continue
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _interrupt_process_group(pid: int, signum: int) -> None:
    """Terminate the active review group before unwinding a launcher signal.

    :param pid: Process-group leader and direct child to reap.
    :param signum: Signal received by the launcher.
    :raises InterruptedError: Always, after the child group is reaped.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _terminate_process_group(pid)
    raise InterruptedError(signum)


@contextlib.contextmanager
def _forward_termination_signals(pid: int) -> Iterator[None]:
    """Forward launcher interrupts to an isolated review process group.

    :param pid: Process-group leader that receives termination signals. :yields: Control while
        forwarding is installed.
    """
    forwarded = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in forwarded}
    for signum in forwarded:
        signal.signal(
            signum,
            lambda received, _frame: _interrupt_process_group(pid, received),
        )
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _extract_report(output: str) -> str:
    """Extract the last completed agent message from Codex JSON events.

    :param output: Newline-delimited JSON events from ``codex exec``.
    :returns: Final agent report text.
    :raises ValueError: If no completed agent message exists.
    """
    reports = []
    for line in output.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            reports.append(item["text"])
    if not reports:
        raise ValueError("missing final agent message")
    return reports[-1]


def _execute_agent(command: list[str], prompt: str, timeout_s: int) -> tuple[int | None, str, str]:
    """Execute Codex and capture output with bounded process-group cleanup.

    :param command: Pinned ``codex exec`` command arguments.
    :param prompt: Fully resolved agent prompt.
    :param timeout_s: Positive execution deadline in seconds.
    :returns: Exit status (``None`` on timeout), standard output, and standard error.
    """
    argv = [*command, "--json", prompt]
    with (
        open(os.devnull) as devnull,
        tempfile.TemporaryFile(mode="w+") as output_file,
        tempfile.TemporaryFile(mode="w+") as error_file,
    ):
        pid = os.posix_spawnp(
            command[0],
            argv,
            os.environ,
            file_actions=[
                (os.POSIX_SPAWN_DUP2, devnull.fileno(), 0),
                (os.POSIX_SPAWN_DUP2, output_file.fileno(), 1),
                (os.POSIX_SPAWN_DUP2, error_file.fileno(), 2),
            ],
            setpgroup=0,
        )
        try:
            with _forward_termination_signals(pid):
                status = _wait_for_pid(pid, time.monotonic() + timeout_s)
        except InterruptedError as error:
            status = 128 + int(error.args[0])
        if status is None:
            _terminate_process_group(pid)
        output_file.seek(0)
        error_file.seek(0)
        output = output_file.read()
        errors = error_file.read()
    return status, output, errors


def _run_agent(command: list[str], prompt: str, timeout_s: int) -> int:
    """Print the final report from one bounded Codex execution.

    :param command: Pinned ``codex exec`` command arguments.
    :param prompt: Fully resolved agent prompt.
    :param timeout_s: Positive execution deadline in seconds.
    :returns: Zero after printing a final report, otherwise one.
    """
    status, output, errors = _execute_agent(command, prompt, timeout_s)
    if status is None:
        sys.stderr.write(errors)
        sys.stderr.write(f"codex exec timed out after {timeout_s}s\n")
        return 1
    if status != 0:
        sys.stderr.write(errors)
        return status
    try:
        report = _extract_report(output)
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        sys.stderr.write(errors)
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stdout.write(report)
    return 0


def _main() -> int:
    """Resolve the review policy and run Codex unless dry-run was requested.

    :returns: Process exit status.
    """
    args = _parse_args()
    agent = _load_agent(args.role)
    prompt = _resolve_prompt(args)
    prompt = f"{agent.developer_instructions.strip()}\n\n{prompt}"
    command = [
        "codex",
        "exec",
        "--model",
        agent.model,
        "--config",
        f'model_reasoning_effort="{agent.model_reasoning_effort}"',
    ]
    try:
        timeout_s = _resolve_timeout(args.role)
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    if args.dry_run:
        sys.stdout.write(
            json.dumps(
                {"command": command, "prompt": prompt, "dry_run": True, "timeout_s": timeout_s}
            )
        )
        return 0
    return _run_agent(command, prompt, timeout_s)


if __name__ == "__main__":
    sys.exit(_main())
