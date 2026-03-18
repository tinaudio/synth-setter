---
name: shell-style
description: >-
  Google Shell Style Guide adapted for this project. Covers bash conventions, formatting,
  naming, error handling, quoting, arrays, and all shell scripting rules.
  Used by the /review skill for shell script review.
---

# Shell Style Guide

Adapted from the [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html).
Applies to all `.sh` files and bash scripts in `scripts/`.

______________________________________________________________________

## 1. When to Use Shell

- Shell is OK for small utilities, wrapper scripts, and glue code
- If a script exceeds ~100 lines or has complex control flow, rewrite in Python
- Use shell when mostly calling other utilities with minimal data manipulation
- Avoid shell when performance matters

## 2. File Conventions

- Bash only: `#!/bin/bash` (not sh, dash, zsh)
- Use `set -euo pipefail` at the top of scripts for strict error handling
- Executables: no extension or `.sh`. Libraries: must have `.sh` extension
- SUID/SGID is forbidden on shell scripts — use `sudo` instead
- Run ShellCheck on all scripts

## 3. Output Conventions

- All error messages go to STDERR:
  ```bash
  err() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
  }
  ```
- Normal output goes to STDOUT

## 4. Comments

### File Headers

Every script starts with a description:

```bash
#!/bin/bash
#
# Perform hot backups of Oracle databases.
```

### Function Comments

Required for any function that isn't both obvious and short. Include:

- Description
- `Globals:` — listed and noted if modified
- `Arguments:` — what the function takes
- `Outputs:` — STDOUT/STDERR output
- `Returns:` — return values other than last command's exit status

```bash
#######################################
# Delete a file in a sophisticated manner.
# Arguments:
#   File to delete, a path.
# Returns:
#   0 if deleted, non-zero on error.
#######################################
del_thing() {
  rm "$1"
}
```

### Implementation Comments

Comment tricky, non-obvious, or important code. Don't comment everything.

### TODO Comments

Format: `# TODO(name): explanation (bug ####)`

## 5. Formatting

### Indentation

- 2 spaces. No tabs.
- Exception: tabs only for `<<-` here-documents

### Line Length

- Maximum 80 characters
- Long strings: use here-documents or embedded newlines
- Long file paths and URLs can exceed on their own line

### Pipelines

```bash
# Fits on one line — keep together
command1 | command2

# Doesn't fit — split with pipe on newline, 2-space indent
command1 \
  | command2 \
  | command3
```

### Control Flow

`; then` and `; do` on same line as `if`/`for`/`while`. `else`, `fi`, `done` on own lines:

```bash
for dir in "${dirs[@]}"; do
  if [[ -d "${dir}" ]]; then
    rm -rf "${dir}"
  else
    mkdir -p "${dir}"
  fi
done
```

### Case Statements

```bash
case "${expression}" in
  a)
    variable="value"
    ;;
  b) short_action ;;
  *)
    error "Unexpected: '${expression}'"
    ;;
esac
```

### Variable Expansion

- Prefer `"${var}"` over `"$var"` (except single-char specials: `$?`, `$#`, `$$`, `$!`)
- Always double-quote variables, command substitutions, and strings with spaces/metacharacters
- Use arrays for safe quoting of lists: `"${FLAGS[@]}"`
- `"$@"` almost always (not `$*`)

### Quoting Rules (priority order)

1. Always quote strings with variables, command substitutions, spaces, or metacharacters
2. Use arrays for lists of elements (especially command-line flags)
3. Shell-internal integer specials (`$?`, `$#`, `$$`) may be unquoted
4. Prefer quoting words; don't quote literal integers

## 6. Features

### Command Substitution

- Use `$(command)` not backticks
- Nesting: `var="$(command "$(inner)")"`

### Test Expressions

- Use `[[ ... ]]` not `[ ... ]` or `test`
- String testing: use `-z` (zero length) and `-n` (non-zero length) explicitly
- Use `==` for equality (not `=`)
- Numeric comparison: use `(( ... ))` or `-lt`/`-gt`, not `<`/`>` in `[[ ]]`

```bash
if [[ -z "${my_var}" ]]; then
  echo "empty"
fi

if (( count > 3 )); then
  echo "many"
fi
```

### Wildcard Expansion

- Use `./*` instead of `*` to avoid filenames starting with `-`

### Eval

- **Never use `eval`**

### Arrays

- Use arrays to store lists of elements safely:
  ```bash
  declare -a flags
  flags=(--foo --bar='baz')
  flags+=(--greeting="Hello ${name}")
  mybinary "${flags[@]}"
  ```
- Never use strings for sequences of arguments
- Use `"${array[@]}"` for expansion (quoted)

### Pipes to While

- Use process substitution instead of piping to while (pipes create subshells):
  ```bash
  while read -r line; do
    last_line="${line}"
  done < <(your_command)
  ```
- Or use `readarray -t lines < <(command)`

### Arithmetic

- Use `(( ... ))` or `$(( ... ))` — never `let`, `$[ ... ]`, or `expr`
- Variables don't need `${}` inside `$(( ))`:
  ```bash
  (( i += 3 ))
  echo "$(( hr * 3600 + min * 60 + sec ))"
  ```
- **WARNING:** With `set -e`, standalone `(( ))` exits if expression evaluates to zero:
  ```bash
  set -e
  i=0
  (( i++ ))  # EXITS THE SCRIPT — i++ returns 0 (old value), which is falsy
  ```
  Workaround: use `(( i += 1 ))` or `i=$(( i + 1 ))` instead of `(( i++ ))` when `i` might be 0.
- Prefer `local -i` and `declare -i` for integer variables

### Aliases

- Never use aliases in scripts — use functions instead

## 7. Naming

| Type               | Convention         | Example                |
| ------------------ | ------------------ | ---------------------- |
| Functions          | `lower_with_under` | `cleanup_shards`       |
| Package functions  | `package::func`    | `storage::upload`      |
| Variables          | `lower_with_under` | `shard_count`          |
| Constants/env vars | `CAPS_WITH_UNDER`  | `R2_BUCKET`            |
| Source filenames   | `lower_with_under` | `docker_entrypoint.sh` |

- Loop variables named for what they iterate: `for zone in "${zones[@]}"`
- Declare function-specific variables with `local`
- Separate `local` declaration from assignment when using command substitution:
  ```bash
  local my_var
  my_var="$(my_func)"
  ```

## 8. Script Structure

- Constants and `set` statements at top
- All functions together below constants
- No executable code between functions
- `main` function required for scripts with multiple functions
- Last non-comment line: `main "$@"`

## 9. Error Handling

- Always check return values
- Use `if ! command; then` or check `$?`
- Use `PIPESTATUS` for pipe components:
  ```bash
  tar -cf - ./* | ( cd "${dir}" && tar -xf - )
  if (( PIPESTATUS[0] != 0 || PIPESTATUS[1] != 0 )); then
    err "tar failed"
  fi
  ```
- Prefer builtins over external commands: `${string/#foo/bar}` over `sed`

______________________________________________________________________

## Review Checklist

| #    | Check                                                                  | Severity |
| ---- | ---------------------------------------------------------------------- | -------- |
| SH1  | `#!/bin/bash` shebang, `set -euo pipefail`                             | BLOCK    |
| SH2  | All variables double-quoted (except integers and `$?`)                 | BLOCK    |
| SH3  | `[[ ]]` not `[ ]` for tests                                            | BLOCK    |
| SH4  | `$(command)` not backticks                                             | BLOCK    |
| SH5  | `(( ))` for arithmetic, never `let` or `expr`                          | WARN     |
| SH6  | Arrays used for argument lists, not strings                            | WARN     |
| SH7  | Error messages to STDERR (`>&2`)                                       | WARN     |
| SH8  | Return values checked for all commands                                 | BLOCK    |
| SH9  | `local` used for function variables                                    | WARN     |
| SH10 | `local` declaration separate from command substitution assignment      | WARN     |
| SH11 | No `eval`                                                              | BLOCK    |
| SH12 | Functions have header comments (Globals/Arguments/Outputs/Returns)     | WARN     |
| SH13 | Constants declared `readonly` at top                                   | WARN     |
| SH14 | `main` function present for multi-function scripts                     | WARN     |
| SH15 | Script \<= ~100 lines, or rewrite in Python                            | WARN     |
| SH16 | No aliases in scripts — use functions                                  | WARN     |
| SH17 | Process substitution used instead of pipe-to-while                     | WARN     |
| SH18 | `./*` for wildcard expansion, not bare `*`                             | WARN     |
| SH19 | No `(( i++ ))` when `i` could be 0 under `set -e` — use `(( i += 1 ))` | BLOCK    |
