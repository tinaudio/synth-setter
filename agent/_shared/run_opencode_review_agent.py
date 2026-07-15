#!/usr/bin/env python3
# Callers run this outside the project environment, so it declares its own deps.
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///
"""Launch an OpenCode PR-review pass with its project-pinned runtime policy."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
# Only workers run the cross-model pass; the orchestrator has no OpenCode policy.
REVIEW_ROLES = (
    "pr-review-worker-deep",
    "pr-review-worker-fast",
)


class _AgentConfig(BaseModel):
    """Validated execution policy for one OpenCode review role.

    .. attribute :: model_config

       Enforces strict input types while allowing future OpenCode agent fields.
    .. attribute :: name

       Must match the role requested by the launcher.
    .. attribute :: model

       Exact ``provider/model`` slug passed to ``opencode run``.
    .. attribute :: variant

       Provider-specific reasoning effort, omitted when the model default applies.
    .. attribute :: developer_instructions

       Prepended to the worker prompt because subprocess launches do not select
       project custom-agent roles natively.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    name: str
    model: str
    variant: str | None = None
    developer_instructions: str


def _parse_args() -> argparse.Namespace:
    """Parse exactly one prompt source for one review worker role.

    :returns: Validated launcher arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=REVIEW_ROLES)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_agent(role: str) -> _AgentConfig:
    """Load a role and guard against filename/name drift.

    :param role: Project role whose pinned execution policy is required.
    :returns: Validated model, variant, and instruction settings.
    :raises ValueError: If the file's declared name does not match ``role``.
    """
    path = REPO_ROOT / ".opencode" / "agents" / f"{role}.toml"
    with path.open("rb") as file:
        config = _AgentConfig.model_validate(tomllib.load(file))
    if config.name != role:
        raise ValueError(f"agent name {config.name!r} does not match role {role!r}")
    return config


def _resolve_prompt(args: argparse.Namespace) -> str:
    """Read prompt text from the mutually exclusive selected source.

    :param args: Parsed launcher arguments.
    :returns: Prompt text for the selected review role.
    """
    if args.prompt is not None:
        return args.prompt
    return args.prompt_file.read_text()


def _main() -> int:
    """Emit the resolved command for the shell launcher.

    :returns: Zero after emitting valid JSON.
    """
    args = _parse_args()
    agent = _load_agent(args.role)
    prompt = f"{agent.developer_instructions.strip()}\n\n{_resolve_prompt(args)}"
    command = [
        "opencode",
        "run",
        "-m",
        agent.model,
        "--agent",
        "pr-reviewer",
        "--format",
        "json",
    ]
    if agent.variant is not None:
        command += ["--variant", agent.variant]
    sys.stdout.write(json.dumps({"command": command, "prompt": prompt, "dry_run": args.dry_run}))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
