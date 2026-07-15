#!/usr/bin/env python3
"""Launch a Codex PR-review role with its project-pinned runtime policy."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_ROLES = (
    "pr-review-orchestrator",
    "pr-review-worker-deep",
    "pr-review-worker-fast",
)


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


def _main() -> int:
    """Emit the resolved command for the shell launcher.

    :returns: Zero after emitting valid JSON.
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
    sys.stdout.write(json.dumps({"command": command, "prompt": prompt, "dry_run": args.dry_run}))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
