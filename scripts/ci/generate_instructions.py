#!/usr/bin/env python3
"""Generate CLAUDE.md, GEMINI.md, .gemini/settings.json, and .gemini/commands/*.toml."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

CLAUDE_HEADER = """# CLAUDE.md

synth-setter: synth inversion, sound matching, and preset-exploration tools — Python 3.10+, PyTorch Lightning, Hydra, with a distributed data pipeline on SkyPilot-managed compute (RunPod + OCI) stored in Cloudflare R2.

Shared agent instructions for Claude and Codex; AGENTS.md is the canonical source. Architecture: [docs/architecture.md](docs/architecture.md).
"""

GEMINI_CONTENT = """# GEMINI.md

synth-setter: synth inversion, sound matching, and preset-exploration tools — Python 3.10+, PyTorch Lightning, Hydra, with a distributed data pipeline on SkyPilot-managed compute (RunPod + OCI) stored in Cloudflare R2.

Shared agent instructions for Claude, Codex, and Gemini; AGENTS.md is the canonical source. Architecture: [docs/architecture.md](docs/architecture.md).

Read and follow [AGENTS.md](AGENTS.md). That file is the canonical project
instruction source for Claude, Codex, and Gemini. Keep Gemini-specific compatibility
notes in `.gemini/`; keep shared hooks and review skills under `agent/`.
"""

CLAUDE_SECTION_ORDER = [
    "Always",
    "Commands",
    "Writing code",
    "Comment hygiene",
    "Design defaults",
    "Mandatory skills for code changes",
    "Testing",
    "Commits",
    "Lint exceptions are append-frozen",
    "YAML `run:` block scalars are bash",
    "Refactoring",
    "PRs",
    "Code review",
    "GPU verification",
    "VST and R2 verification",
]

CLAUDE_CONDITIONS = {
    "Always": "you are about to edit, write, or commit any file",
    "Commands": "you need to run commands to build, test, lint, or format",
    "Writing code": "you are writing or modifying Python code",
    "Comment hygiene": "you are writing inline comments or docstrings",
    "Design defaults": "you are starting any non-trivial change",
    "Mandatory skills for code changes": "you are modifying non-documentation code (anything other than .md / docs/ edits)",
    "Testing": "you are writing or running tests",
    "Commits": "you are committing",
    "Lint exceptions are append-frozen": "a lint, pydoclint, or pyright check fails on a file your change touches",
    "YAML `run:` block scalars are bash": "you are editing GitHub Actions workflows (.github/workflows/*.yml) or SkyPilot compute configs (src/synth_setter/configs/compute/*.yaml)",
    "Refactoring": "you are moving, renaming, or restructuring code",
    "PRs": "you are opening or driving a pull request",
    "Code review": "you are reviewing code",
    "GPU verification": "you are about to claim no GPU is available",
    "VST and R2 verification": "you are about to claim no VST or R2 is available",
}

SKILL_COMMANDS = {
    "repo-review": "repo/review",
    "repo-review-full": "repo/review-full",
    "repo-review-full-no-comments": "repo/review-full-no-comments",
    "fix-review-comments": "fix/review-comments",
    "pr-readiness": "pr/readiness",
}

SHIM_COMMANDS = {
    "qrspi-design": "qrspi/design",
    "qrspi-implement": "qrspi/implement",
    "qrspi-plan": "qrspi/plan",
    "qrspi-pr": "qrspi/pr",
    "qrspi-questions": "qrspi/questions",
    "qrspi-research": "qrspi/research",
    "qrspi-structure": "qrspi/structure",
    "qrspi-worktree": "qrspi/worktree",
    "lint-cleanup": "lint/cleanup",
}

GEMINI_HOOKS_CONFIG = {
    "SessionStart": [
        {
            "description": "Worktree-status banner: prints cwd, current branch, and whether the agent is sitting in the primary checkout (which AGENTS.md marks read-only). Fires on every session start, resume, /clear, and /compact so post-compaction agents see the same context.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/session-start-cwd-banner.sh",
                }
            ],
        }
    ],
    "BeforeTool": [
        {
            "matcher": "replace_file_content|multi_replace_file_content|write_to_file",
            "description": "Credential protection: blocks edits to .env, .pem, and .key files. Goal: prevent secrets from being modified or leaked into commits.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/edit-write.sh credential-protect",
                }
            ],
        },
        {
            "matcher": "replace_file_content|multi_replace_file_content|write_to_file",
            "description": "Worktree guard: on Edit/Write inside the primary checkout, prints a WARNING with the exact `git worktree add` command.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/worktree-guard.sh",
                }
            ],
        },
        {
            "matcher": "replace_file_content|multi_replace_file_content|write_to_file",
            "description": "No-baseline-additions: on Edit/Write of .pydoclint-baseline.txt, blocks the call if the post-edit line count exceeds the current count.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/no-baseline-additions.sh",
                }
            ],
        },
        {
            "matcher": "replace_file_content|multi_replace_file_content|write_to_file",
            "description": "No-yaml-run-comments: on Edit/Write of .github/workflows/*.{yml,yaml} or configs/compute/*.{yml,yaml}, blocks the call if the post-edit content has any `#`-comment line inside a `run: |` / `setup: |` block scalar.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/no-yaml-run-comments.sh",
                }
            ],
        },
        {
            "matcher": "run_command",
            "description": "Branch safety: echoes the current branch name before any git commit.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/branch-safety.sh",
                }
            ],
        },
        {
            "matcher": "run_command",
            "description": "Git-commit-trailer-check: blocks `git commit` (and other forms) that carry --no-verify / -n, a Co-Authored-By trailer, or an agent-attribution footer.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/git-commit-trailer-check.sh",
                }
            ],
        },
        {
            "matcher": "run_command",
            "description": "Pre-PR review gate: blocks gh pr create unless REVIEW_FULL=<path> points at a sentinel review file.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/pre-pr-review-gate.sh",
                }
            ],
        },
    ],
    "AfterAgent": [
        {
            "description": "PR-readiness gate: on Stop, if there's an open PR for the current branch whose readiness gates don't hold, blocks the turn from ending (exit 2).",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/pr-readiness-stop.sh",
                }
            ],
        }
    ],
    "AfterTool": [
        {
            "matcher": "replace_file_content|multi_replace_file_content|write_to_file",
            "description": "Auto-format on every Edit/Write: ruff (Python), mdformat (Markdown), prettier (YAML).",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/edit-write.sh format",
                }
            ],
        },
        {
            "matcher": "replace_file_content|multi_replace_file_content|write_to_file",
            "description": "Auto-test: runs the matching pytest file when src/, scripts/, or tests/ Python files are edited.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/edit-write.sh test",
                }
            ],
        },
        {
            "matcher": "run_command",
            "description": "Worktree post-setup: after `git worktree add`, automatically runs `make link-plugins && make link-thoughts` in the new worktree.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/worktree-post-setup.sh",
                }
            ],
        },
        {
            "matcher": "run_command",
            "description": "Taxonomy verification: validates GitHub metadata after gh issue create, gh pr create, and addSubIssue mutations.",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/verify-gh-taxonomy.sh",
                }
            ],
        },
        {
            "matcher": "run_command",
            "description": "PR checkbox trigger: detects gh pr create commands and prompts the agent to invoke the /pr-checkbox skill.",
            "hooks": [
                {
                    "type": "command",
                    "command": 'jq -r \'.tool_input.command // ""\' | grep -q \'gh pr create\' && echo \'{"hookSpecificOutput":{"hookEventName":"AfterTool","additionalContext":"A PR was just created. You MUST now invoke the /pr:checkbox command to add verification checkboxes to this PR."}}\' || true',
                }
            ],
        },
        {
            "description": "Doc-drift advisory review: on gh pr create, runs a headless agent session invoking the doc-drift skill.",
            "matcher": "run_command",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/doc-drift.sh",
                }
            ],
        },
        {
            "description": "PR review resolver: on git push (excluding main/master), resolves the branch's PR up front.",
            "matcher": "run_command",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash agent/hooks/pr-review-resolver.sh",
                }
            ],
        },
    ],
}


def parse_agents_md(agents_path: Path) -> dict[str, str]:
    """Parse sections from AGENTS.md.

    :param agents_path: Path to AGENTS.md.
    :returns: Map from section header to body text.
    """
    content = agents_path.read_text(encoding="utf-8")
    sections = {}

    parts = content.split("\n## ")
    for part in parts[1:]:
        lines = part.split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:])
        sections[title] = body

    return sections


def parse_markdown_with_frontmatter(file_path: Path) -> tuple[dict[str, str], str]:
    """Parse markdown file that may contain YAML frontmatter.

    :param file_path: Path to the markdown file.
    :returns: Tuple of frontmatter dict and body text.
    """
    content = file_path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return {}, content.strip()

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content.strip()

    frontmatter_text = parts[1]
    body = parts[2].strip()

    try:
        metadata = yaml.safe_load(frontmatter_text) or {}
    except Exception:
        metadata = {}
        for line in frontmatter_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip()

    return metadata, body


def generate_toml_command(description: str, prompt: str) -> str:
    """Generate the contents of a Gemini CLI TOML command file.

    :param description: Description of the command.
    :param prompt: Prompt/instructions body.
    :returns: TOML string content.
    """
    description_clean = description.replace("\n", " ").replace('"', '\\"')
    return f"description = \"{description_clean}\"\nprompt = '''\n{prompt}\n'''\n"


def format_via_pre_commit(content: str, filename: str, project_root: Path) -> str:
    """Format the markdown content using the pre-commit mdformat hook.

    :param content: Unformatted markdown string.
    :param filename: Name of the file (e.g. 'CLAUDE.md').
    :param project_root: Path to the repository root.
    :returns: Formatted markdown string.
    """
    temp_path = project_root / f"{filename}.tmp.md"
    temp_path.write_text(content, encoding="utf-8")

    try:
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pre_commit", "run", "mdformat", "--files", str(temp_path)],
            cwd=str(project_root),
            capture_output=True,
            check=False,
        )
        formatted = temp_path.read_text(encoding="utf-8")
    except Exception:
        formatted = content
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return formatted


def main() -> int:
    """Generate or check the instruction and config files.

    :returns: Exit status code.
    """
    parser = argparse.ArgumentParser(
        description="Generate CLAUDE.md/GEMINI.md, .gemini/settings.json, and .gemini/commands/*.toml"
    )
    parser.add_argument(
        "--check", action="store_true", help="Only check for differences without writing"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    agents_path = project_root / "AGENTS.md"
    claude_path = project_root / "CLAUDE.md"
    gemini_path = project_root / "GEMINI.md"
    mcp_config_path = project_root / ".mcp.json"
    gemini_settings_path = project_root / ".gemini" / "settings.json"

    if not agents_path.exists():
        sys.stderr.write(f"Error: AGENTS.md not found at {agents_path}\n")
        return 1

    sections = parse_agents_md(agents_path)

    # Build CLAUDE.md content
    claude_parts = [CLAUDE_HEADER.strip()]
    for title in CLAUDE_SECTION_ORDER:
        if title not in sections:
            sys.stderr.write(f"Error: Section {title!r} not found in AGENTS.md\n")
            return 1
        body = sections[title].strip()
        condition = CLAUDE_CONDITIONS[title]
        claude_parts.append(f'<important if="{condition}">\n\n{body}\n</important>')

    generated_claude = "\n\n".join(claude_parts) + "\n"
    generated_gemini = GEMINI_CONTENT.strip() + "\n"

    # Format markdown contents through pre-commit
    generated_claude = format_via_pre_commit(generated_claude, "CLAUDE.md", project_root)
    generated_gemini = format_via_pre_commit(generated_gemini, "GEMINI.md", project_root)

    # Build .gemini/settings.json content
    mcp_servers = {}
    if mcp_config_path.exists():
        try:
            mcp_data = json.loads(mcp_config_path.read_text(encoding="utf-8"))
            mcp_servers = mcp_data.get("mcpServers", {})
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to parse .mcp.json: {e}\n")

    settings_dict = {
        "permissions": {"defaultMode": "auto"},
        "hooks": GEMINI_HOOKS_CONFIG,
        "mcpServers": mcp_servers,
    }
    generated_settings = json.dumps(settings_dict, indent=2) + "\n"

    # Map command projections
    projected_commands: dict[Path, str] = {}

    # 1. Project skills
    for skill_name, relative_path in SKILL_COMMANDS.items():
        skill_file = project_root / "agent" / "skills" / skill_name / "SKILL.md"
        if not skill_file.exists():
            sys.stderr.write(f"Error: Skill file not found at {skill_file}\n")
            return 1
        metadata, body = parse_markdown_with_frontmatter(skill_file)
        desc = metadata.get("description", "").strip() or f"Run the {skill_name} skill."
        toml_content = generate_toml_command(desc, body)
        dest_path = project_root / ".gemini" / "commands" / f"{relative_path}.toml"
        projected_commands[dest_path] = toml_content

    # 2. Project shims
    for shim_name, relative_path in SHIM_COMMANDS.items():
        shim_file = project_root / ".claude" / "commands" / f"{shim_name}.md"
        if not shim_file.exists():
            sys.stderr.write(f"Error: Claude command file not found at {shim_file}\n")
            return 1
        metadata, body = parse_markdown_with_frontmatter(shim_file)
        desc = metadata.get("description", "").strip() or f"Run the {shim_name} command."
        body_with_args = body.replace("$ARGUMENTS", "{{args}}")
        toml_content = generate_toml_command(desc, body_with_args)
        dest_path = project_root / ".gemini" / "commands" / f"{relative_path}.toml"
        projected_commands[dest_path] = toml_content

    if args.check:
        drift = False
        if not claude_path.exists():
            sys.stderr.write("CLAUDE.md does not exist.\n")
            drift = True
        elif claude_path.read_text(encoding="utf-8") != generated_claude:
            sys.stderr.write("CLAUDE.md is out of sync with AGENTS.md.\n")
            drift = True

        if not gemini_path.exists():
            sys.stderr.write("GEMINI.md does not exist.\n")
            drift = True
        elif gemini_path.read_text(encoding="utf-8") != generated_gemini:
            sys.stderr.write("GEMINI.md is out of sync.\n")
            drift = True

        if not gemini_settings_path.exists():
            sys.stderr.write(".gemini/settings.json does not exist.\n")
            drift = True
        elif gemini_settings_path.read_text(encoding="utf-8") != generated_settings:
            sys.stderr.write(".gemini/settings.json is out of sync.\n")
            drift = True

        # Check projected command TOML files
        for dest_path, expected_content in projected_commands.items():
            if not dest_path.exists():
                sys.stderr.write(
                    f"Projected command {dest_path.relative_to(project_root)} does not exist.\n"
                )
                drift = True
            elif dest_path.read_text(encoding="utf-8") != expected_content:
                sys.stderr.write(
                    f"Projected command {dest_path.relative_to(project_root)} is out of sync.\n"
                )
                drift = True

        if drift:
            return 1
        sys.stdout.write("All instruction and command files are in sync.\n")
        return 0

    # Write instruction files
    claude_path.write_text(generated_claude, encoding="utf-8")
    sys.stdout.write(f"Updated {claude_path}\n")
    gemini_path.write_text(generated_gemini, encoding="utf-8")
    sys.stdout.write(f"Updated {gemini_path}\n")

    # Write settings file
    gemini_settings_path.parent.mkdir(parents=True, exist_ok=True)
    gemini_settings_path.write_text(generated_settings, encoding="utf-8")
    sys.stdout.write(f"Updated {gemini_settings_path}\n")

    # Write projected TOML command files
    for dest_path, content in projected_commands.items():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(content, encoding="utf-8")
        sys.stdout.write(f"Updated {dest_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
