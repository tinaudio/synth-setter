#!/usr/bin/env python3
"""Launch deferred Pi review passes outside the foreground host lifetime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

if __package__:
    from agent._shared.pi_review_routing import PINNED_REVIEW_MODELS
else:
    from pi_review_routing import PINNED_REVIEW_MODELS

_AFTERCARE_MODEL = "gpt-5.6-terra"
_AFTERCARE_PROVIDER = "openai-codex"
_AFTERCARE_THINKING = "medium"


class DeferredPass(BaseModel, strict=True, extra="forbid"):
    """One model pass deferred by the foreground review.

    .. attribute :: skill
        :type: str

        Assigned checklist.

    .. attribute :: pass_name
        :type: Literal["codex", "free-pool"]

        Logical review pass.

    .. attribute :: origin
        :type: Literal["primary", "codex-fallback"]

        Whether this model is the pass's independent provider or an exhausted-pool fallback.

    .. attribute :: model
        :type: str

        Exact pinned model selector.

    .. attribute :: verification_model
        :type: str

        Effective foreground Codex model used to verify free-pool findings.

    .. attribute :: thinking
        :type: str

        Thinking level selected by the routing plan.
    """

    skill: str = Field(min_length=1)
    pass_name: Literal["codex", "free-pool"]
    origin: Literal["primary", "codex-fallback"]
    model: str = Field(min_length=1)
    verification_model: str = Field(min_length=1)
    thinking: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def _require_pinned_model_family(self) -> DeferredPass:
        """Reject model selectors outside the reviewed routing pool.

        :returns: Validated deferred pass.
        :raises ValueError: If model provenance conflicts with the logical pass.
        """
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
        :type: Literal[1]

        Manifest schema version.

    .. attribute :: mode
        :type: Literal["full", "no-comments"]

        Foreground delivery mode.

    .. attribute :: repo
        :type: str

        GitHub repository in ``owner/name`` form.

    .. attribute :: pr_number
        :type: int

        Existing pull request receiving late findings.

    .. attribute :: base_sha
        :type: str

        Reviewed base commit.

    .. attribute :: head_sha
        :type: str

        Reviewed PR head; aftercare suppresses stale results.

    .. attribute :: target
        :type: str

        Worker target label.

    .. attribute :: deferred_passes
        :type: tuple[DeferredPass, ...]

        Incomplete independent passes.

    .. attribute :: foreground_fingerprints
        :type: tuple[str, ...]

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


def load_manifest(path: Path) -> AftercareManifest:
    """Load and validate one foreground aftercare handoff.

    :param path: Absolute manifest path under the current worktree's review directory.
    :returns: Strict deferred-review manifest.
    :raises ValueError: If the path escapes ``.agent-reviews``.
    """
    resolved = path.resolve()
    review_dir = (Path.cwd() / ".agent-reviews").resolve()
    if not resolved.is_relative_to(review_dir):
        raise ValueError("Aftercare manifest must be under .agent-reviews")
    return AftercareManifest.model_validate_json(resolved.read_text())


def build_command(manifest_path: Path) -> list[str]:
    """Build the pinned detached Pi command for one aftercare manifest.

    :param manifest_path: Validated absolute manifest path.
    :returns: Argument vector for Pi's headless aftercare session.
    :raises RuntimeError: If Pi is unavailable.
    """
    pi = shutil.which("pi")
    if pi is None:
        raise RuntimeError("pi executable not found on PATH")
    prompt = (
        "Execute the deferred PR-review procedure in "
        "agent/skills/_shared/repo-review-aftercare.md using manifest "
        f"{manifest_path}. Process only its deferred passes, then exit."
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


def launch_aftercare(manifest_path: Path) -> int:
    """Start a detached Pi aftercare process and return its PID.

    :param manifest_path: Absolute path to a valid aftercare manifest.
    :returns: Detached process identifier.
    """
    load_manifest(manifest_path)
    command = build_command(manifest_path)
    environment = os.environ.copy()
    environment.pop("SYNTH_SETTER_PI_REVIEW", None)
    environment["SYNTH_SETTER_PI_REVIEW_AFTERCARE"] = "1"
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=Path.cwd(),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def _build_parser() -> argparse.ArgumentParser:
    """Build the detached-launcher argument parser.

    :returns: CLI parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Validate a manifest, then print its command or detached PID.

    :returns: Process exit status.
    """
    args = _build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    if args.dry_run:
        load_manifest(manifest_path)
        sys.stdout.write(f"{json.dumps(build_command(manifest_path))}\n")
        return 0
    sys.stdout.write(f"{launch_aftercare(manifest_path)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
