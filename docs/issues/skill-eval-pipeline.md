# GitHub Issue: feat(evaluation): CI skill evaluation pipeline with A/B comparator

**Type:** Feature
**Domain label:** `evaluation`
**Milestone:** `evaluation v1.0.0`
**Priority:** P1

---

## Summary

Set up a CI evaluation pipeline that benchmarks Claude Code skills using A/B comparator
methodology, enabling iterative improvement through the native Skills Creator and prompt
engineering best practices.

## Motivation

We have 11 Claude Code skills in `.claude/skills/` governing code review, TDD, ML testing, and
project standards -- but no way to **measure** whether they actually improve Claude's output.
Without measurement, skill iteration is guesswork.

## Methodology

Based on [Anthropic: Test, Measure, and Refine Agent Skills](https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills)
and the [Claude Prompt Engineering Guide](https://github.com/ThamJiaHe/claude-prompt-engineering-guide/blob/main/Claude-Prompt-Guide.md):

- **A/B comparator evals**: Run two versions of a skill (current vs candidate) in parallel
  isolated contexts. A comparator agent blindly judges which output is better against defined
  success/failure criteria.
- **Fallback**: If A/B proves too costly or inconsistent, fall back to prompt + golden-output
  evals (deterministic grading, 1 API call per case instead of 3).

## Skill Priority Tiers

| Tier | Skills | Focus |
|------|--------|-------|
| **Tier 1** (now) | `tdd-implementation`, `pr-checkbox`, `ml-test` | Code testing, implementation, feature verification |
| **Tier 1.5** (now) | `code-health`, `review` | Enhance code review quality |
| **Tier 2** (next) | `project-standards`, `ml-data-pipeline` | Narrow domain: DSP coding, ML pipeline standards |
| **Tier 3** (later) | `python-style`, `shell-style`, `github-taxonomy`, `design-doc` | Style and process skills |

## Implementation Plan

### Eval Case Format (co-located YAML)

Each skill gets an `evals/` subfolder with eval cases as YAML files:

```
.claude/skills/<skill-name>/evals/
  eval_<scenario>.yaml
```

Each eval case defines: `name`, `skill`, `description`, `prompt`, `context_files`,
`success_criteria`, `failure_criteria`, `comparator_focus`.

### Eval Runner (`scripts/eval_skills.py`)

- Parses eval YAML files from a skill's `evals/` directory
- Runs A/B Claude API calls (skill-on vs skill-off/candidate)
- Blind comparator scores outputs against criteria
- Results written to `eval-results/` (gitignored)
- Cost guard: `--max-evals N` flag, `--dry-run` by default

### CI Workflow (`.github/workflows/skill-eval.yml`)

- **Auto-trigger**: on PRs that modify `.claude/skills/**/SKILL.md`
- **Manual trigger**: `workflow_dispatch` for on-demand evaluation
- Uploads results as GitHub Actions artifacts

### Code Review Enhancement

- Planted-bug test fixtures for the `review` skill
- Measure false positive/negative rates
- Verify all 7 sub-checklists are invoked

## Deliverables

- [ ] Eval case YAML schema (`.claude/skills/eval-schema.yaml`)
- [ ] Eval runner script (`scripts/eval_skills.py`)
- [ ] CI workflow (`.github/workflows/skill-eval.yml`)
- [ ] ~15 eval cases across 5 Tier 1/1.5 skills
- [ ] Makefile target (`make eval-skills`)

## References

- [Anthropic: Improving Skill Creator](https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills)
- [Claude Prompt Engineering Guide](https://github.com/ThamJiaHe/claude-prompt-engineering-guide/blob/main/Claude-Prompt-Guide.md)
