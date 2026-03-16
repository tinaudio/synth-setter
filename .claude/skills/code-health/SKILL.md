---
name: code-health
description: >-
  Code health and implementation quality skill for writing clean, readable, well-structured code.
  Use this skill whenever Claude is asked to review code, review a PR, improve code quality,
  refactor or clean up code, reduce complexity, simplify logic, restructure a module, or discuss
  architecture and design. Also trigger on "review this", "what do you think of this code",
  "how should I structure this", "make this more readable", "reduce nesting", "naming suggestions",
  or any request involving code quality, readability, or structural improvement. This skill
  complements tdd-implementation — use both when writing new code. Use this one alone for
  review-only or refactoring-only tasks. Even if the user doesn't mention "code health" explicitly,
  trigger on any code-related request where structural quality matters.
---

# Code Health — Implementation Quality

This skill governs how Claude evaluates and improves code quality. It covers readability,
structure, naming, comments, domain modeling, defensive design, and PR practices. These
principles are distilled from Google's Tech on the Toilet (Code Health) series and industry
best practices, generalized for open-source development.

**When to use this skill vs. tdd-implementation**: This skill handles *structural quality* —
how code reads, how it's organized, how it communicates intent. The tdd-implementation skill
handles *testing methodology* — Red-Green-Refactor, test design, mutation testing. When
implementing new code, read both. When reviewing or refactoring existing code, this skill
alone may suffice.

______________________________________________________________________

## Core Principles

### 1. Reduce Nesting, Reduce Complexity

Deeply nested code is hard to follow. Each level of nesting adds cognitive load because the
reader must mentally track every condition that led to this point.

**Fix it with guard clauses and early returns:**

```python
# Bad: deeply nested
def process_order(order):
    if order is not None:
        if order.is_valid():
            if order.has_items():
                total = calculate_total(order)
                if total > 0:
                    return charge(order, total)
    return None

# Good: flat with guard clauses
def process_order(order):
    if order is None:
        return None
    if not order.is_valid():
        return None
    if not order.has_items():
        return None
    total = calculate_total(order)
    if total <= 0:
        return None
    return charge(order, total)
```

**Techniques:**

- Invert conditions and return/continue early
- Extract deeply nested blocks into named functions
- Replace `else` after `return` with top-level code
- Limit nesting to 2–3 levels maximum

### 2. Arrange Code to Communicate Data Flow

Readers should be able to follow data flow linearly from top to bottom. Declare variables
close to first use, not at the top of the function. Group related operations together.

```python
# Bad: declarations far from use
def build_report(data):
    header = ""
    rows = []
    footer = ""
    total = 0
    # ... 20 lines later, header is finally used ...

# Good: declare near first use, group related operations
def build_report(data):
    header = format_header(data.title)

    rows = [format_row(item) for item in data.items]
    total = sum(item.amount for item in data.items)

    footer = format_footer(total)
    return Report(header, rows, footer)
```

**The principle:** If you have to scroll up to find where a variable was defined, or
jump around to trace the data flow, the arrangement needs work.

### 3. Write Clean Code to Reduce Cognitive Load

Every line of code imposes a mental cost on the reader. Minimize the effort required to
understand what code does. This applies at every level — individual expressions, function
bodies, module organization.

**Practical rules:**

- Functions should do one thing at one level of abstraction
- If a function requires a comment to explain what it does, consider renaming it or splitting it
- Avoid clever one-liners that require mental unpacking — prefer clear over concise
- Limit functions to a size that fits in your working memory (~20 lines is a good target)

### 4. Use Positive Booleans

Negative and double-negative booleans are hard to reason about. Flip them.

```python
# Bad: double negation
if not is_not_valid:
if not disable_logging:

# Good: positive form
if is_valid:
if enable_logging:
```

Name boolean variables and functions as positive assertions: `is_valid`, `has_permission`,
`should_retry`, `can_proceed`. When you must negate, negate a positive:
`if not is_valid` reads better than `if is_invalid`.

### 5. Comment the Why, Not the What

Good code is self-documenting for *what* it does. Comments should explain *why* —
the business reason, the constraint, the non-obvious decision.

```python
# Bad: restates the code
x = x + 1  # increment x

# Bad: explains what
# Sort users by age
users.sort(key=lambda u: u.age)

# Good: explains why
# Users must be sorted by age for the sliding-window dedup to work correctly.
# The dedup assumes adjacent entries are the most likely duplicates.
users.sort(key=lambda u: u.age)
```

**When to comment:**

- Non-obvious "why" decisions
- Workarounds for bugs or edge cases (include issue/ticket links)
- Public API contracts — what callers can expect
- Performance-critical sections where the "obvious" approach was too slow

**When NOT to comment:**

- What the next line of code does (rename it instead)
- Changelog entries in the code (that's what git log is for)
- Commented-out code (delete it; version control remembers)

### 6. Write Focused, Descriptive Comments for Public APIs

For public functions, classes, and modules, write docstrings that describe the contract:
what it accepts, what it returns, what can go wrong. Describe the *public API contract*,
not implementation details.

```python
def retry_with_backoff(fn, max_attempts=3, base_delay=1.0):
    """Call fn, retrying on transient failures with exponential backoff.

    :param fn: Callable that may raise TransientError.
    :param max_attempts: Maximum number of tries (including the initial call).
    :param base_delay: Initial delay in seconds; doubled after each retry.
    :returns: The return value of fn on success.
    :raises TransientError: If all attempts fail, re-raises the last error.
    :raises PermanentError: Immediately, without retrying.
    """
```

### 7. Eliminate YAGNI Smells

Don't build features, abstractions, or extension points that aren't needed yet. "You Aren't
Gonna Need It" is a refactoring principle: code that doesn't exist has zero bugs and zero
maintenance cost.

**Smell indicators:**

- Empty interfaces/abstract classes with a single implementation
- Configuration options that no caller uses
- "Flexible" data structures that handle cases that don't exist
- Comment saying "in case we need this later"

**The fix:** Delete it. Write the simplest code that solves today's problem. When
tomorrow's problem arrives, you'll know what extension point you actually need.

### 8. Functional Core, Imperative Shell

Separate pure business logic from I/O and side effects. The pure "core" is trivially
testable, easy to reason about, and composable. The "shell" handles I/O, talks to
databases, reads files — but contains minimal logic.

```python
# Imperative shell — thin, handles I/O
def process_csv_file(path: str) -> None:
    raw_data = read_csv(path)                    # I/O
    results = transform_and_validate(raw_data)   # Pure core
    write_results(results)                       # I/O

# Functional core — pure, testable
def transform_and_validate(data: list[dict]) -> list[Result]:
    return [
        validate(transform(row))
        for row in data
        if row.get("active")
    ]
```

**Benefits:** The core can be unit tested with plain data (no mocking files/databases).
The shell is thin enough that integration tests cover it adequately.

### 9. Sort Lines in Source Code

When you have lists of items with no inherent ordering — imports, config keys, enum
members, dependency lists — sort them alphabetically.

**Why:**

- Prevents accidental duplicates (adjacent lines are easy to spot)
- Merge conflicts are less likely (additions go to predictable positions)
- Diffs are cleaner (one-line additions, not reshuffled blocks)
- No arguments about "where should I put this"

### 10. Set Safe Defaults for Flags and Configuration

When code accepts flags or configuration, default to the safest option. Require explicit
opt-in for anything destructive or risky.

```python
# Bad: destructive by default
def cleanup(directory, dry_run=False):
    ...

# Good: safe by default
def cleanup(directory, dry_run=True):
    ...
```

**Rules:**

- Dry-run mode on by default; require `--execute` or `--no-dry-run` to actually act
- Require environment-specific settings (prod credentials, endpoints) to be explicit
- Never default to deleting, overwriting, or sending without confirmation
- Fail closed: when in doubt, refuse rather than proceed

### 11. Replace Magic Numbers with Named Constants

Never scatter literal values through code. Extract them into named constants that
communicate intent.

```python
# Bad: magic numbers
if retry_count > 3:
    time.sleep(0.5)
if len(audio) > 16000 * 4:

# Good: named constants
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 0.5
MAX_AUDIO_DURATION_SAMPLES = SAMPLE_RATE * MAX_DURATION_SECONDS
```

For audio/ML pipelines: sample rates, hop lengths, mel bins, and buffer sizes should
live in a config object, not as literals in function bodies.

### 12. Prefer Fewer Function Arguments

Functions with many arguments are hard to understand, test, and call correctly. Three
arguments is a practical maximum for most functions.

**Techniques to reduce arguments:**

- Group related arguments into a dataclass or config object
- If a function takes a boolean flag that changes behavior, split it into two functions
- If several functions always take the same cluster of arguments, they belong in a class

```python
# Bad: many arguments, flag argument
def render_audio(sample_rate, channels, duration, normalize, preset, velocity, path):
    if normalize:
        ...

# Good: config object, no flag — split into two functions
def render_audio(config: AudioConfig, preset: Preset, output_path: Path):
    ...

def render_and_normalize(config: AudioConfig, preset: Preset, output_path: Path):
    audio = render_audio(config, preset, output_path)
    return normalize(audio)
```

**Don't use flag arguments.** A boolean that selects between two behaviors means the
function does two things. Split it into two functions with descriptive names.

### 13. Use Explanatory Variables

Extract complex expressions into named variables that explain what the expression means.
This is especially important for boolean conditions and index calculations.

```python
# Bad: opaque condition
if audio.shape[0] > sr * 4 and np.max(np.abs(audio)) > 0.01:

# Good: explanatory variables
exceeds_max_duration = audio.shape[0] > sr * MAX_DURATION_SECONDS
is_not_silence = np.max(np.abs(audio)) > SILENCE_THRESHOLD
if exceeds_max_duration and is_not_silence:
```

### 14. Encapsulate Boundary Conditions

Boundary conditions (off-by-one, start/end of range, empty input) are hard to track when
scattered across multiple locations. Put the processing for them in one place.

```python
# Bad: boundary logic duplicated
end_index = len(data) - 1
if index >= len(data) - 1:
for i in range(len(data) - 1):

# Good: encapsulated
last_index = len(data) - 1
if index >= last_index:
for i in range(last_index):
```

### 15. Follow the Law of Demeter

A function should only call methods on: its own object, its parameters, objects it creates,
and its direct component objects. Don't reach through chains of objects.

```python
# Bad: reaching through object chains
user.get_account().get_settings().get_theme().get_color()

# Good: ask, don't dig
user.preferred_color()  # user knows how to get it
```

This reduces coupling — if the internal structure of `Account` changes, only `Account`
needs to update, not every caller that reached through it.

### 16. Be Consistent

If you do something a certain way, do all similar things the same way. Consistency reduces
cognitive load because readers can predict patterns.

- If some functions return `None` on failure and others raise exceptions, pick one pattern
- If some configs use snake_case keys and others use camelCase, pick one
- If some modules use `import X` and others use `from X import Y`, be consistent within a module

### 17. Boy Scout Rule

Leave the code cleaner than you found it. When you touch a file to fix a bug or add a
feature, clean up one small thing nearby — a confusing name, an unnecessary comment, a
duplicated condition. Don't do a full rewrite, just leave it slightly better.

### 18. Don't Repeat Yourself (DRY)

Every piece of knowledge should have a single, unambiguous, authoritative representation.
Duplication means every change requires updating multiple locations — and missing one creates
bugs.

**What to look for:**

- Copy-pasted logic that differs only in a variable name or constant
- The same validation check in multiple places
- Parallel data structures that must stay in sync manually
- String literals (paths, keys, error messages) repeated across files

**What NOT to do:** Don't create premature abstractions to eliminate trivial duplication.
Three similar lines of code is better than a premature abstraction. DRY applies to
*knowledge* and *logic*, not to superficially similar code.

### 19. Use Dependency Injection

Pass dependencies as arguments rather than creating them internally. This makes code
testable, composable, and decoupled from concrete implementations.

```python
# Bad: hard-coded dependency
def upload_shard(shard_path: Path, run_id: str) -> None:
    uploader = RcloneUploader()  # hard-coded — can't test without rclone
    uploader.upload(shard_path, f"{run_id}/data/")

# Good: injected dependency
def upload_shard(shard_path: Path, run_id: str, storage: StorageBackend) -> None:
    storage.upload_file(shard_path, run_id, f"data/{shard_path.name}")
```

- Define interfaces as `Protocol` classes, inject implementations
- In tests, inject fakes or mocks. In production, inject real implementations.
- Wire dependencies at the top level (CLI entry point, `main()`), not deep in the call stack

### 20. Recognize Code Smells

Watch for these structural problems that indicate deeper design issues:

- **Rigidity:** A small change forces a cascade of changes in unrelated modules
- **Fragility:** A change in one place breaks something in a seemingly unrelated place
- **Immobility:** Code can't be reused in another context because it drags in too many dependencies
- **Opacity:** Code is hard to understand even after reading it carefully — poor naming, deep nesting, or convoluted logic
- **Needless repetition:** Same logic in multiple places (violates DRY)

These are symptoms, not root causes. When you spot them, look for the underlying design
problem (missing abstraction, wrong boundary, coupled responsibilities) rather than
patching the symptom.

### 21. Write Change-Resilient Code with Domain Objects

Model fundamental concepts, not current requirements. Use domain objects — types that
represent real ideas in the problem space — rather than passing around primitives.

```python
# Bad: primitive obsession
def create_user(name: str, email: str, age: int, role: str): ...
# Easy to swap email and name, role can be any string

# Good: domain objects
@dataclass(frozen=True)
class Email:
    value: str
    def __post_init__(self):
        if "@" not in self.value:
            raise ValueError(f"Invalid email: {self.value}")

@dataclass(frozen=True)
class Role:
    value: str  # validated against known roles

def create_user(name: str, email: Email, age: int, role: Role): ...
# Type system prevents mixing up arguments
```

**This connects to two related principles:**

**Avoid Primitive Obsession:** When you find yourself passing a `str` that represents
something specific (user ID, currency code, file path), wrap it in a type. This prevents
accidental mixing (passing a user_id where an order_id is expected).

**Use Typed Identifiers:** Wrap IDs in typed wrappers to prevent mixing different ID types:

```python
@dataclass(frozen=True)
class UserId:
    value: int

@dataclass(frozen=True)
class OrderId:
    value: int

# Now the type system catches: find_order(user_id)  # type error
```

______________________________________________________________________

## PR Linked Issue Check (Hard Fail)

Every pull request **must** have a linked GitHub issue before merge. This is
a **hard check** — CI will block merge if missing.

| #   | Requirement              | How to satisfy                                                    |
| --- | ------------------------ | ----------------------------------------------------------------- |
| H1  | **Linked GitHub Issue**  | Include `closes #N` / `fixes #N` in PR body, or link via sidebar  |

This check is enforced by the `pr-metadata-gate` GitHub Actions workflow
(`.github/workflows/pr-metadata-gate.yaml`). The workflow runs on every PR targeting
`main`, `release/*`, or `dev` and produces `::error::` annotations so failures are
visible inline on the PR.

______________________________________________________________________

## PR and Code Review Practices

### 22. Prefer Small, Focused Pull Requests

Large PRs are harder to review, riskier to merge, and slower to get approved. Break work
into small, focused changes.

**Guidelines:**

- One logical change per PR — a refactor, a feature, a bug fix — not all three
- If a PR touches more than ~400 lines, ask if it can be split
- Separate mechanical changes (renames, formatting) from behavioral changes
- Each PR should be independently reviewable and (ideally) independently deployable

### 23. Write Commit Messages That Explain Why

The diff shows *what* changed. The commit message should explain *why*.

Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#specification)
specification. The commit message format is:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Common types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `ci`, `style`, `perf`.
Use `!` after the type/scope for breaking changes. Scopes are optional but encouraged.

```
# Bad
Fix bug

# Bad
Update user_service.py

# Good
fix(auth): prevent race condition in user creation

When two requests create users with the same email simultaneously,
both could pass the uniqueness check before either insert completes.
Add a database-level unique constraint and handle IntegrityError
with a retry.

Fixes #1234
```

**Structure:** First line is a Conventional Commits summary (\<72 chars). Body explains
motivation, approach, and trade-offs. Include ticket/issue references.

**The topline MUST be a valid Conventional Commit.** This means:

- Starts with a type: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `ci`, `style`, `perf`
- Optional scope in parentheses: `fix(auth):`, `feat(pipeline):`
- Colon + space + lowercase imperative description
- Description is meaningful — describes the *why* or the *what changed*, not just "update file"
- Bad: `fix: stuff`, `chore: updates`, `feat: changes` — these say nothing
- Good: `fix(validation): catch NaN values in mel spectrograms before upload`

**Commits should be as small as possible** while keeping the codebase in a working state.
Prefer many small, focused commits over one large commit. Each commit should be one logical
change. However, never split a commit such that intermediate states have untested code or
broken tests — every commit should leave the repo green.

**Do NOT** append `Co-Authored-By` tags or similar attribution trailers to commit messages.

### 24. Use Conventional Branch Names

Branch names should follow the pattern `<type>/<short-description>`:

```
feat/add-wds-output-format
fix/nan-validation-in-mel
refactor/storage-backend-protocol
ci/claude-review-skills
docs/update-design-doc
test/add-finalize-integration-tests
chore/update-dependencies
dev/distributed-pipeline
```

**Rules:**

- Prefix with type: `feat/`, `fix/`, `refactor/`, `ci/`, `docs/`, `test/`, `chore/`, `dev/`
- Use lowercase kebab-case for the description
- Keep it short but descriptive
- Bad: `my-branch`, `wip`, `test123`, `ktinubu/stuff`
- Good: `fix/pre-commit-docformatter`, `feat/pipeline-cli-generate`

### 25. Code Review Etiquette

When reviewing others' code (or when Claude suggests changes):

- Focus on the code, not the person — "This function could be simplified" not "You wrote this wrong"
- Distinguish "must fix" from "nice to have" — prefix optional suggestions with "nit:" or "optional:"
- Ask questions rather than making demands — "What if we used X here?" invites discussion
- Acknowledge good code — reviewers often only comment on problems
- If a PR attracts many comments, that's a signal the PR is too large or the code needs structural work

### 26. Reduce Code Review Friction

If your PRs consistently attract many review comments:

- Break them into smaller changes
- Add a PR description explaining the approach
- Self-review before requesting review — catch the obvious issues yourself
- Include test coverage so reviewers can verify behavior
- Address all comments or explicitly note disagreements with rationale

______________________________________________________________________

## Applying These Principles

### During Code Review

When reviewing code, evaluate against these principles in priority order:

1. **Correctness**: Does the code do what it should? (not covered here — see tdd-implementation)
2. **Readability**: Can a new team member understand this code without the author explaining it?
3. **Structure**: Is nesting shallow? Is data flow linear? Are functions focused?
4. **Naming**: Do names communicate intent? Are booleans positive? Are types meaningful?
5. **Comments**: Do comments explain *why*, not *what*? Are public APIs documented?
6. **Design**: Are domain concepts modeled? Is I/O separated from logic?
7. **Defensive design**: Are defaults safe? Are destructive operations guarded?

### During Refactoring

When the user asks to "clean up" or "simplify" code:

1. Read the entire file/module first to understand the data flow
2. Identify the worst structural problem (usually deep nesting or long functions)
3. Fix one thing at a time — don't rewrite everything at once
4. After each change, verify the code still works (or explain what tests to run)
5. Name the principle behind each change so the user learns the pattern

### When Writing New Code

Apply these principles proactively:

1. Start with the data flow — what goes in, what comes out, what steps in between
2. Write function signatures and types before implementations
3. Separate pure logic from I/O early
4. Use domain types for anything that isn't truly a generic int/string
5. Default all flags to safe values
6. Keep nesting shallow from the start — it's easier than flattening later

______________________________________________________________________

## Code Health Review Report

After completing a code review or refactoring task, produce a brief report. This helps the
user see what was evaluated and what to prioritize.

### Findings Table

| #   | Principle                                                                              | Status   | Finding    |
| --- | -------------------------------------------------------------------------------------- | -------- | ---------- |
| H1  | **GATE** Milestone assigned                                                            | ✅/❌    | Brief note |
| H2  | **GATE** Linked GitHub issue                                                           | ✅/❌    | Brief note |
| H3  | **GATE** GitHub Project assigned                                                       | ✅/❌    | Brief note |
| 1   | Nesting depth                                                                          | ✅/⚠️/❌ | Brief note |
| 2   | Data flow clarity                                                                      | ✅/⚠️/❌ | Brief note |
| 3   | Cognitive load                                                                         | ✅/⚠️/❌ | Brief note |
| 4   | Positive booleans                                                                      | ✅/⚠️/❌ | Brief note |
| 5   | Comment quality (why > what)                                                           | ✅/⚠️/❌ | Brief note |
| 6   | Public API documentation                                                               | ✅/⚠️/❌ | Brief note |
| 7   | No YAGNI / unnecessary abstractions                                                    | ✅/⚠️/❌ | Brief note |
| 8   | Functional core / imperative shell                                                     | ✅/⚠️/❌ | Brief note |
| 9   | Sorted lists/imports                                                                   | ✅/⚠️/❌ | Brief note |
| 10  | Safe defaults                                                                          | ✅/⚠️/❌ | Brief note |
| 11  | No magic numbers — named constants used                                                | ✅/⚠️/❌ | Brief note |
| 12  | Few function arguments (\<=3), no flag args                                            | ✅/⚠️/❌ | Brief note |
| 13  | Explanatory variables for complex expressions                                          | ✅/⚠️/❌ | Brief note |
| 14  | Boundary conditions encapsulated                                                       | ✅/⚠️/❌ | Brief note |
| 15  | Law of Demeter — no object chain reaching                                              | ✅/⚠️/❌ | Brief note |
| 16  | Consistency — similar things done the same way                                         | ✅/⚠️/❌ | Brief note |
| 17  | Domain objects / no primitive obsession                                                | ✅/⚠️/❌ | Brief note |
| 18  | DRY — no duplicated logic or knowledge                                                 | ✅/⚠️/❌ | Brief note |
| 19  | Dependencies injected, not hard-coded                                                  | ✅/⚠️/❌ | Brief note |
| 20  | No code smells (rigidity, fragility, opacity)                                          | ✅/⚠️/❌ | Brief note |
| 21  | PR size and focus                                                                      | ✅/⚠️/❌ | Brief note |
| 22  | Commit topline is valid Conventional Commit with meaningful description                | ✅/⚠️/❌ | Brief note |
| 23  | Commit is smallest possible while keeping repo green (no untested intermediate states) | ✅/⚠️/❌ | Brief note |
| 24  | Branch name follows conventional format: `<type>/<short-description>`                  | ✅/⚠️/❌ | Brief note |

Use ✅ for good, ⚠️ for needs attention with explanation, ❌ for significant issue.
**H1–H3 are hard gates**: they only use ✅ or ❌ BLOCK — never ⚠️. A missing gate is always BLOCK.
Skip rows that aren't applicable to the task (e.g., skip PR rows for single-file reviews).

### Summary

After the table, write 2–4 sentences covering:

- The most impactful issue found (or "code is in good shape" if so)
- Recommended priority for fixes
- Any patterns that suggest deeper structural problems

______________________________________________________________________

## TotT Episode Reference

For the full catalog of all Google Tech on the Toilet episodes — including the
implementation-focused episodes that informed this skill — see the tott-catalog bundled
with the `tdd-implementation` skill. The 🔨-marked episodes there are the source material
for the principles above.

Key episodes by principle:

- Nesting: [Reduce Nesting, Reduce Complexity](https://testing.googleblog.com/2017/06/code-health-reduce-nesting-reduce.html)
- Data flow: [Arrange Your Code to Communicate Data Flow](https://testing.googleblog.com/2025/01/arrange-your-code-to-communicate-data.html)
- Cognitive load: [Write Clean Code to Reduce Cognitive Load](https://testing.googleblog.com/2023/11/write-clean-code-to-reduce-cognitive.html)
- Booleans: [Improve Readability With Positive Booleans](https://testing.googleblog.com/2023/10/improve-readability-with-positive.html)
- Comments: [Less Is More](https://testing.googleblog.com/2024/08/less-is-more-principles-for-simple.html), [To Comment or Not](https://testing.googleblog.com/2017/07/code-health-to-comment-or-not-to-comment.html)
- YAGNI: [Eliminate YAGNI Smells](https://testing.googleblog.com/2017/08/code-health-eliminate-yagni-smells.html)
- Functional core: [Simplify Your Code](https://testing.googleblog.com/2025/10/simplify-your-code-functional-core.html)
- Sorting: [Sort Lines in Source Code](https://testing.googleblog.com/2025/09/sort-lines-in-source-code.html)
- Safe defaults: [Set Safe Defaults for Flags](https://testing.googleblog.com/2026/03/set-safe-defaults-for-flags.html)
- Domain objects: [Write Change-Resilient Code](https://testing.googleblog.com/2024/09/write-change-resilient-code-with-domain.html)
- Primitives: [Obsessed With Primitives?](https://testing.googleblog.com/2017/11/obsessed-with-primitives.html)
- Identifiers: [Identifierify](https://testing.googleblog.com/2017/10/code-health-identifierify.html)
- Small PRs: [Prefer Small Focused Pull Requests](https://testing.googleblog.com/2024/04/prefer-small-focused-pull-requests.html)
- Commit messages: [Providing Context with Commit Messages](https://testing.googleblog.com/2017/09/code-health-providing-context-with.html)
- Code review: [Code Review Etiquette](https://testing.googleblog.com/2019/11/code-health-respectful-reviews-useful.html), [Too Many Comments?](https://testing.googleblog.com/2016/11/code-health-too-many-comments-on-your.html)
- Metrics: [Code Health: Now With Metrics!](https://testing.googleblog.com/2023/12/code-health-now-with-metrics.html)
