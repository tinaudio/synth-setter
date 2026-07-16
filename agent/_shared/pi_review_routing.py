"""Build auditable model plans for Pi PR-review workers."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import sh

DEEP_SKILLS = frozenset({"correctness-review", "lance-review"})
MECHANICAL_SKILLS = frozenset({"comment-hygiene", "python-style", "shell-style"})
SUPPORTED_SKILLS = frozenset(
    {
        "code-health",
        "comment-hygiene",
        "correctness-review",
        "gha-workflow-validator",
        "lance-review",
        "ml-data-pipeline",
        "ml-test",
        "python-style",
        "shell-style",
        "synth-setter-project-standards",
        "tdd-implementation",
        "tdd-refactor",
    }
)
REQUIRED_PROVIDERS = {
    "openai-codex": "authenticate with `/login openai-codex`",
    "openrouter": "set OPENROUTER_API_KEY before starting Pi or use `/login openrouter`",
}

_DEEP_CANDIDATES = {
    "codex": (
        "openai-codex/gpt-5.6-sol",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/openrouter/free",
    ),
    "openrouter": (
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/openrouter/free",
        "openai-codex/gpt-5.6-sol",
    ),
}
_STANDARD_CANDIDATES = {
    "codex": (
        "openai-codex/gpt-5.6-terra",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/openrouter/free",
    ),
    "openrouter": (
        "openrouter/cohere/north-mini-code:free",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openai-codex/gpt-5.6-terra",
    ),
}
_REQUIRED_REPORT_HEADINGS = (
    "### BLOCK findings",
    "### WARN findings",
    "### What looks good",
)
_REPORT_TITLE = re.compile(r"^## (?P<skill>[a-z0-9-]+) review — .+$")
_FINDING = re.compile(r"^\d+\. \*\*.+:\d+\*\* — \S.+$")


@dataclass(frozen=True, slots=True)
class ReviewPass:
    """One skill/model-family pass and its available candidate sequence.

    .. attribute :: skill
        :type: str

        Authoritative checklist name.

    .. attribute :: pass_name
        :type: str

        Logical Codex or OpenRouter pass.

    .. attribute :: candidates
        :type: tuple[str, ...]

        Available models in attempt order.

    .. attribute :: unavailable
        :type: tuple[str, ...]

        Configured models absent from Pi's registry.

    .. attribute :: thinking
        :type: str

        Pi thinking level for every attempt.

    .. attribute :: reason
        :type: str

        Auditable explanation for the thinking allocation.
    """

    skill: str
    pass_name: str
    candidates: tuple[str, ...]
    unavailable: tuple[str, ...]
    thinking: str
    reason: str


def parse_available_models(output: str) -> set[str]:
    """Parse ``pi --list-models`` output into canonical selectors.

    :param output: Whitespace-delimited Pi model table.
    :returns: Available ``provider/model-id`` selectors.
    """
    models: set[str] = set()
    for line in output.splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[0] != "provider":
            models.add(f"{columns[0]}/{columns[1]}")
    return models


def provenance_for_model(model: str) -> str:
    """Return finding provenance from the model that produced it.

    :param model: Canonical ``provider/model-id`` selector.
    :returns: ``codex`` or ``openrouter``.
    :raises ValueError: If the provider is outside the review policy.
    """
    provider = model.split("/", 1)[0]
    if provider == "openai-codex":
        return "codex"
    if provider == "openrouter":
        return "openrouter"
    raise ValueError(f"Unsupported Pi review provider: {provider}")


def report_is_parseable(report: str, *, expected_skill: str) -> bool:
    """Return whether a worker report satisfies the merge contract.

    :param report: Worker Markdown output.
    :param expected_skill: Checklist the worker was assigned.
    :returns: Whether the title and ordered sections are structurally valid.
    """
    lines = report.strip().splitlines()
    if not lines:
        return False
    title = _REPORT_TITLE.fullmatch(lines[0])
    if title is None or title.group("skill") != expected_skill:
        return False
    if any(lines.count(heading) != 1 for heading in _REQUIRED_REPORT_HEADINGS):
        return False
    indices = [lines.index(heading) for heading in _REQUIRED_REPORT_HEADINGS]
    if indices != sorted(indices) or any(line.strip() for line in lines[1 : indices[0]]):
        return False

    block_lines = lines[indices[0] + 1 : indices[1]]
    warn_lines = lines[indices[1] + 1 : indices[2]]
    good_lines = [line for line in lines[indices[2] + 1 :] if line.strip()]
    return (
        _findings_section_is_valid(block_lines)
        and _findings_section_is_valid(warn_lines)
        and bool(good_lines)
        and all(line.startswith("- ") and len(line) > 2 for line in good_lines)
    )


def _findings_section_is_valid(lines: Sequence[str]) -> bool:
    content = [line for line in lines if line.strip()]
    if content in (["None."], ["None"], ["- None."], ["- None"]):
        return True
    if not content:
        return False

    return all(_FINDING.fullmatch(line) for line in content)


def build_review_plan(
    skills: Sequence[str],
    *,
    changed_lines: int,
    risk_reasons: Sequence[str],
    available_models: set[str],
) -> list[ReviewPass]:
    """Allocate model candidates and thinking to selected review skills.

    :param skills: Selected authoritative review checklists.
    :param changed_lines: Total added and deleted lines in the diff.
    :param risk_reasons: Named risk signals detected in the diff.
    :param available_models: Canonical selectors from Pi's model registry.
    :returns: Two ordered passes per skill, preserving the supplied skill order.
    :raises ValueError: If provider authentication or all candidates are unavailable.
    """
    if changed_lines < 0:
        raise ValueError("changed_lines must be non-negative")
    unknown = sorted(set(skills) - SUPPORTED_SKILLS)
    if unknown:
        raise ValueError(f"Unknown review skill(s): {', '.join(unknown)}")
    _require_providers(available_models)
    plan: list[ReviewPass] = []
    for skill in skills:
        is_deep = skill in DEEP_SKILLS
        candidates_by_pass = _DEEP_CANDIDATES if is_deep else _STANDARD_CANDIDATES
        thinking, reason = _thinking_for(
            skill,
            changed_lines=changed_lines,
            risk_reasons=risk_reasons,
            is_deep=is_deep,
        )
        for pass_name in ("codex", "openrouter"):
            configured = candidates_by_pass[pass_name]
            candidates = tuple(model for model in configured if model in available_models)
            unavailable = tuple(model for model in configured if model not in available_models)
            if not candidates:
                raise ValueError(f"No available models remain for {skill}/{pass_name}")
            plan.append(
                ReviewPass(
                    skill=skill,
                    pass_name=pass_name,
                    candidates=candidates,
                    unavailable=unavailable,
                    thinking=thinking,
                    reason=reason,
                )
            )
    return plan


def _require_providers(available_models: set[str]) -> None:
    for provider, setup in REQUIRED_PROVIDERS.items():
        prefix = f"{provider}/"
        if not any(model.startswith(prefix) for model in available_models):
            raise ValueError(f"No {provider} models available; {setup}; credentials required")


def _thinking_for(
    skill: str,
    *,
    changed_lines: int,
    risk_reasons: Sequence[str],
    is_deep: bool,
) -> tuple[str, str]:
    if is_deep:
        return "high", "deep checklist"

    thinking = "medium"
    reason = "standard checklist"
    if skill in MECHANICAL_SKILLS and changed_lines < 200:
        thinking = "low"
        reason = "mechanical checklist on diff under 200 lines"

    risks = list(risk_reasons)
    if changed_lines > 800:
        risks.insert(0, "diff over 800 lines")
    if risks:
        thinking = {"low": "medium", "medium": "high", "high": "high"}[thinking]
        reason = f"risk: {', '.join(risks)}"
    return thinking, reason


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the available review plan as JSON")
    plan.add_argument("--skill", action="append", required=True)
    plan.add_argument("--changed-lines", type=int, required=True)
    plan.add_argument("--risk", action="append", default=[])
    validate = subparsers.add_parser(
        "validate-report", help="check a worker report's section contract"
    )
    validate.add_argument("path", type=Path)
    validate.add_argument("--skill", required=True, choices=sorted(SUPPORTED_SKILLS))
    provenance = subparsers.add_parser(
        "provenance", help="print provenance for an effective model"
    )
    provenance.add_argument("model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the routing CLI.

    :param argv: Optional command arguments for tests or embedding.
    :returns: Process exit status.
    :raises RuntimeError: If Pi is unavailable when building a plan.
    :raises AssertionError: If argument parsing returns an unknown command.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        pi_executable = shutil.which("pi")
        if pi_executable is None:
            raise RuntimeError("pi executable not found on PATH")
        model_output = str(sh.Command(pi_executable)("--list-models"))
        plan = build_review_plan(
            args.skill,
            changed_lines=args.changed_lines,
            risk_reasons=args.risk,
            available_models=parse_available_models(model_output),
        )
        sys.stdout.write(f"{json.dumps([asdict(item) for item in plan], indent=2)}\n")
        return 0
    if args.command == "validate-report":
        return 0 if report_is_parseable(args.path.read_text(), expected_skill=args.skill) else 1
    if args.command == "provenance":
        sys.stdout.write(f"{provenance_for_model(args.model)}\n")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
