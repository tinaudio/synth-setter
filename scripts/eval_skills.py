#!/usr/bin/env python3
"""Skill evaluation runner with A/B comparator methodology.

Benchmarks Claude Code skills by running two versions (A: skill-enabled, B: skill-disabled
or candidate) against eval cases, then using a blind comparator to judge quality.

Usage:
    # Dry run (no API calls)
    python scripts/eval_skills.py --dry-run --skill tdd-implementation

    # Run evals for a single skill
    python scripts/eval_skills.py --skill tdd-implementation --max-evals 2

    # Run all Tier 1 skills
    python scripts/eval_skills.py --skill all

    # Compare current skill against a candidate file
    python scripts/eval_skills.py --skill tdd-implementation --candidate path/to/SKILL.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(".claude/skills")
RESULTS_DIR = Path("eval-results")

TIER1_SKILLS = [
    "tdd-implementation",
    "pr-checkbox",
    "ml-test",
    "code-health",
    "review",
]


@dataclass
class EvalCase:
    """A single evaluation case loaded from YAML."""

    name: str
    skill: str
    description: str
    prompt: str
    success_criteria: list[str]
    failure_criteria: list[str]
    comparator_focus: str
    context_files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result from evaluating a single case."""

    case_name: str
    skill: str
    winner: str  # "A" (skill-on), "B" (skill-off/candidate), or "tie"
    score_a: float
    score_b: float
    comparator_reasoning: str
    criteria_scores: dict[str, dict[str, bool]]
    timestamp: str
    duration_seconds: float


def load_eval_cases(skill_name: str) -> list[EvalCase]:
    """Load eval cases from a skill's evals/ directory."""
    evals_dir = SKILLS_DIR / skill_name / "evals"
    if not evals_dir.exists():
        logger.warning("No evals directory for skill: %s", skill_name)
        return []

    cases = []
    for yaml_path in sorted(evals_dir.glob("eval_*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        cases.append(
            EvalCase(
                name=data["name"],
                skill=data["skill"],
                description=data["description"],
                prompt=data["prompt"],
                success_criteria=data["success_criteria"],
                failure_criteria=data["failure_criteria"],
                comparator_focus=data["comparator_focus"],
                context_files=data.get("context_files", []),
                tags=data.get("tags", []),
            )
        )
    return cases


def load_skill_content(skill_name: str) -> str:
    """Load a skill's SKILL.md content."""
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_path.exists():
        msg = f"SKILL.md not found: {skill_path}"
        raise FileNotFoundError(msg)
    return skill_path.read_text()


def build_task_prompt(case: EvalCase) -> str:
    """Build the task prompt with context files if specified."""
    parts = [case.prompt]
    for ctx_file in case.context_files:
        ctx_path = Path(ctx_file)
        if ctx_path.exists():
            parts.append(f"\n--- {ctx_file} ---\n{ctx_path.read_text()}")
    return "\n".join(parts)


def call_claude(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Call Claude API and return the response text.

    Uses claude-sonnet-4-20250514 by default for cost efficiency on evals.
    """
    try:
        import anthropic  # noqa: S404
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def build_comparator_prompt(
    case: EvalCase,
    output_x: str,
    output_y: str,
) -> str:
    """Build the blind comparator prompt."""
    criteria_list = "\n".join(f"  - {c}" for c in case.success_criteria)
    failure_list = "\n".join(f"  - {c}" for c in case.failure_criteria)

    return f"""You are an expert evaluator comparing two AI assistant outputs for the same task.
You do NOT know which output used a skill prompt and which did not. Judge purely on quality.

## Task Given to Both
{case.prompt}

## Success Criteria (output SHOULD exhibit these)
{criteria_list}

## Failure Criteria (output MUST NOT exhibit these)
{failure_list}

## Comparator Focus
{case.comparator_focus}

## Output X
{output_x}

## Output Y
{output_y}

## Instructions
1. Score each output on each success criterion (met=1, partial=0.5, not met=0).
2. Penalize each output for each failure criterion exhibited (-1 per violation).
3. Compute a total score for each output.
4. Declare a winner: X, Y, or tie.

Respond in this exact JSON format:
{{
  "criteria_scores": {{
    "X": {{"<criterion>": true/false, ...}},
    "Y": {{"<criterion>": true/false, ...}}
  }},
  "score_x": <float>,
  "score_y": <float>,
  "winner": "X" or "Y" or "tie",
  "reasoning": "<brief explanation>"
}}"""


def run_eval_case(
    case: EvalCase,
    skill_content: str,
    candidate_content: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> EvalResult:
    """Run A/B eval for a single case.

    Version A: skill_content injected as system prompt.
    Version B: no skill (or candidate_content if provided).
    """
    start = time.monotonic()
    task_prompt = build_task_prompt(case)

    # Version A: skill-enabled
    output_a = call_claude(
        system_prompt=f"Follow these guidelines:\n\n{skill_content}",
        user_prompt=task_prompt,
        model=model,
    )

    # Version B: skill-disabled or candidate
    if candidate_content:
        system_b = f"Follow these guidelines:\n\n{candidate_content}"
    else:
        system_b = "You are a helpful coding assistant."

    output_b = call_claude(
        system_prompt=system_b,
        user_prompt=task_prompt,
        model=model,
    )

    # Randomize presentation order to avoid position bias
    if random.random() < 0.5:  # noqa: S311
        output_x, output_y = output_a, output_b
        mapping = {"X": "A", "Y": "B"}
    else:
        output_x, output_y = output_b, output_a
        mapping = {"X": "B", "Y": "A"}

    # Comparator judges blindly
    comparator_prompt = build_comparator_prompt(case, output_x, output_y)
    comparator_response = call_claude(
        system_prompt="You are an expert code quality evaluator. Respond only in valid JSON.",
        user_prompt=comparator_prompt,
        model=model,
    )

    # Parse comparator response
    try:
        result_data = json.loads(comparator_response)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", comparator_response, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group(1))
        else:
            result_data = {
                "score_x": 0,
                "score_y": 0,
                "winner": "tie",
                "reasoning": f"Failed to parse comparator response: {comparator_response[:200]}",
                "criteria_scores": {"X": {}, "Y": {}},
            }

    # Map X/Y back to A/B
    raw_winner = result_data.get("winner", "tie")
    if raw_winner in mapping:
        winner = mapping[raw_winner]
    else:
        winner = "tie"

    duration = time.monotonic() - start

    return EvalResult(
        case_name=case.name,
        skill=case.skill,
        winner=winner,
        score_a=result_data.get(f"score_{mapping.get('X', 'x').lower()}", result_data.get("score_x", 0)),
        score_b=result_data.get(f"score_{mapping.get('Y', 'y').lower()}", result_data.get("score_y", 0)),
        comparator_reasoning=result_data.get("reasoning", ""),
        criteria_scores=result_data.get("criteria_scores", {}),
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        duration_seconds=round(duration, 2),
    )


def run_skill_evals(
    skill_name: str,
    max_evals: int = 5,
    candidate_path: Optional[Path] = None,
    dry_run: bool = False,
    model: str = "claude-sonnet-4-20250514",
) -> list[EvalResult]:
    """Run all eval cases for a skill."""
    cases = load_eval_cases(skill_name)
    if not cases:
        logger.info("No eval cases found for skill: %s", skill_name)
        return []

    cases = cases[:max_evals]
    skill_content = load_skill_content(skill_name)

    candidate_content = None
    if candidate_path:
        candidate_content = candidate_path.read_text()

    if dry_run:
        logger.info("[DRY RUN] Would run %d eval(s) for skill: %s", len(cases), skill_name)
        for case in cases:
            logger.info("  - %s: %s", case.name, case.description)
        return []

    logger.info("Running %d eval(s) for skill: %s", len(cases), skill_name)
    results = []
    for i, case in enumerate(cases, 1):
        logger.info("  [%d/%d] %s ...", i, len(cases), case.name)
        result = run_eval_case(case, skill_content, candidate_content, model=model)
        results.append(result)
        logger.info(
            "    Winner: %s (A=%.1f, B=%.1f) - %s",
            result.winner,
            result.score_a,
            result.score_b,
            result.comparator_reasoning[:80],
        )

    return results


def save_results(results: list[EvalResult], skill_name: str) -> Path:
    """Save eval results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{skill_name}_{timestamp}.json"

    data = {
        "skill": skill_name,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "num_cases": len(results),
        "wins_a": sum(1 for r in results if r.winner == "A"),
        "wins_b": sum(1 for r in results if r.winner == "B"),
        "ties": sum(1 for r in results if r.winner == "tie"),
        "results": [
            {
                "case_name": r.case_name,
                "winner": r.winner,
                "score_a": r.score_a,
                "score_b": r.score_b,
                "reasoning": r.comparator_reasoning,
                "criteria_scores": r.criteria_scores,
                "timestamp": r.timestamp,
                "duration_seconds": r.duration_seconds,
            }
            for r in results
        ],
    }

    output_path.write_text(json.dumps(data, indent=2))
    return output_path


def print_summary(all_results: dict[str, list[EvalResult]]) -> None:
    """Print a summary table of all eval results."""
    print("\n" + "=" * 60)
    print("SKILL EVALUATION SUMMARY")
    print("=" * 60)

    total_a = 0
    total_b = 0
    total_ties = 0

    for skill_name, results in all_results.items():
        wins_a = sum(1 for r in results if r.winner == "A")
        wins_b = sum(1 for r in results if r.winner == "B")
        ties = sum(1 for r in results if r.winner == "tie")
        total_a += wins_a
        total_b += wins_b
        total_ties += ties

        status = "PASS" if wins_a >= wins_b else "NEEDS IMPROVEMENT"
        print(f"\n  {skill_name}:")
        print(f"    Skill wins: {wins_a}  |  Baseline wins: {wins_b}  |  Ties: {ties}")
        print(f"    Status: {status}")

        for r in results:
            marker = {"A": "+", "B": "-", "tie": "="}[r.winner]
            print(f"      [{marker}] {r.case_name}: A={r.score_a:.1f} B={r.score_b:.1f}")

    print(f"\n  TOTAL: Skill={total_a}  Baseline={total_b}  Ties={total_ties}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Claude Code skills via A/B comparator",
    )
    parser.add_argument(
        "--skill",
        required=True,
        help="Skill to evaluate (or 'all' for all Tier 1 skills)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=5,
        help="Maximum eval cases per skill (default: 5)",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Path to candidate SKILL.md for A/B comparison",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print what would run without making API calls (default: True)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run evals (overrides --dry-run default)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use for evals (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    dry_run = args.dry_run and not args.run

    skills = TIER1_SKILLS if args.skill == "all" else [args.skill]

    all_results: dict[str, list[EvalResult]] = {}

    for skill_name in skills:
        results = run_skill_evals(
            skill_name=skill_name,
            max_evals=args.max_evals,
            candidate_path=args.candidate,
            dry_run=dry_run,
            model=args.model,
        )

        if results:
            output_path = save_results(results, skill_name)
            logger.info("Results saved to: %s", output_path)
            all_results[skill_name] = results

    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
