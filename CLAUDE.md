# CLAUDE.md

Claude-specific entry point for the shared agent instructions.

Read and follow [AGENTS.md](AGENTS.md). That file is the canonical project
instruction source for both Claude and Codex. Keep Claude-only compatibility
notes in `.claude/`; keep shared hooks and review skills under `agent/`.

## Comment hygiene (do this when writing inline text)

Code says **what**; comments say **why**. Lean on names and types first; add
prose only when it carries something they can't. Full rules live in the
`comment-hygiene` skill — these are the load-bearing actions for inline
writing:

- **Earn each comment.** Before adding one, ask: "would a future reader be
  confused without this?" If no, skip it. Trust the names and types.
- **Stay at one line; cap at two.** When you need more context, write a
  one-line pointer (`# <why> — see #N`) and put the detail in the issue,
  commit message, or design doc.
- **Make every line carry new information.** Comments and `:param:` lines
  should add a constraint (`must be sorted`), a unit (`seconds`), a
  semantic (`dB, not linear`), or a rationale — never the function name,
  signature, type, or literal value the reader can already see.
- **Describe current behavior only.** Write docstrings and comments as if
  the code has always looked this way. Put renames, migrations, and
  removals in the commit message; git log carries history.
- **Open docstrings with the contract.** Lead with what the function
  promises or constrains. Skip generic openers ("This module provides…",
  "Helper function that…", "Utility for…", "Wrapper around…") — if the
  next sentence adds nothing real, the whole docstring goes.
- **Let blank lines separate sections.** No `# ===== SECTION =====` or
  `# ----- foo -----` banners.
- **Put YAML comments above the `- name:` step.** Inside `run: |` /
  `setup: |` block scalars the body is bash; a `PreToolUse` hook blocks
  `#`-lines there.
- **Reach for `:param:` / `:returns:` only when they add what the type
  hint can't.** Pydoclint still requires field lists for Pydantic classes
  and validators with `raise` paths — follow the skill's syntax notes
  (`.. attribute ::`, no `:param cls:` / `:param self:`, no `:rtype:`).

When in doubt, write less. Reviewers will ask for a comment if they need
one; they rarely ask for fewer.
