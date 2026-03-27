# doc-map.yaml Schema

The structural mapping file that encodes which source files *should* be
documented where. This is the mechanism that catches drift by omission —
when code adds something new and no doc mentions it.

______________________________________________________________________

## Schema

```yaml
# doc-map.yaml
version: 1

docs:
  - doc: docs/reference/docker.md
    description: Docker build, run, debug reference
    sources:
      - pattern: "docker/**"
        covers: "Dockerfile stages, build targets, layer structure"
      - pattern: "scripts/docker_entrypoint.sh"
        covers: "Entrypoint modes, env var dispatch, MODE values"
      - pattern: "scripts/image_config.py"
        covers: "Image config schema, Pydantic model, validation"
      - pattern: "configs/image/**"
        covers: "Image config YAML files, build parameters"
      - pattern: "Makefile"
        covers: "Docker build targets, secret injection, build flags"
        scope: "docker"  # only review docker-related sections of Makefile

  - doc: docs/reference/rclone.md
    description: rclone R2 operations reference
    sources:
      - pattern: "src/data/uploader.py"
        covers: "Upload commands, flags, dry-run behavior"
      - pattern: "scripts/finalize_shards.py"
        covers: "Download commands, resharding workflow"
      - pattern: "scripts/docker_entrypoint.sh"
        covers: "Train-mode download/upload, stats flags"
        scope: "rclone"
      - pattern: "scripts/r2_shard_report.py"
        covers: "Listing commands, report usage"
      - pattern: "tests/scripts/test_r2_shard_report.py"
        covers: "Test commands, fixture upload, cleanup"

  - doc: docs/reference/wandb-integration.md
    description: W&B logging and auth reference
    sources:
      - pattern: "src/training/**"
        covers: "W&B logging calls, metric names, run config"
      - pattern: "scripts/docker_entrypoint.sh"
        covers: "W&B auth setup, netrc configuration"
        scope: "wandb"
```

______________________________________________________________________

## Field reference

### Top-level

| Field | Type | Required | Description |
|-----------|---------|----------|--------------------------------------|
| `version` | integer | Yes | Schema version. Currently `1`. |
| `docs` | list | Yes | List of doc mapping entries. |

### Doc entry

| Field | Type | Required | Description |
|---------------|--------|----------|--------------------------------------------|
| `doc` | string | Yes | Path to the documentation file (from repo root). |
| `description` | string | No | Human-readable summary of what the doc covers. Used in reports. |
| `sources` | list | Yes | Source files/directories this doc should cover. |

### Source entry

| Field | Type | Required | Description |
|-----------|--------|----------|-------------------------------------------------|
| `pattern` | string | Yes | Glob pattern matching source files. Relative to repo root. Supports `*` and `**`. |
| `covers` | string | Yes | What aspects of this source the doc should cover. This is the key field — it tells the drift checker *what to look for* in the doc. Be specific. |
| `scope` | string | No | If the source file is large or multi-purpose, which section/aspect is relevant to this doc. Helps avoid false positives when a file serves multiple docs. |

______________________________________________________________________

## Design principles

### `covers` is the contract

The `covers` field is what makes the mapping useful. Without it, all you know
is "this doc should mention this file somehow." With it, you know *what* the
doc should say about the file. When the drift checker reads the doc and the
source, it compares the doc's content against the `covers` description to
determine whether the doc adequately covers the expected topics.

Good `covers` values are specific and verifiable:

- "Entrypoint modes, env var dispatch, MODE values" — checker can verify the
  doc lists all modes the entrypoint implements.
- "Upload commands, flags, dry-run behavior" — checker can verify the doc
  shows the flags the code actually uses.

Bad `covers` values are vague:

- "General usage" — checker can't verify anything specific.
- "Everything" — not actionable.

### `scope` prevents false positives

Large files like `Makefile` or `docker_entrypoint.sh` serve multiple docs.
Without `scope`, a change to the Makefile's `lint` target would trigger a
drift check against docker.md — a false positive. With `scope: "docker"`,
the checker only reviews Makefile changes related to Docker build targets.

`scope` is a hint, not a filter. The checker uses it to focus attention but
can still flag issues outside the scope if they're clearly relevant.

### Glob patterns

Patterns follow standard glob syntax:

| Pattern | Matches |
|----------------------------|------------------------------------------------|
| `scripts/docker_entrypoint.sh` | Exactly that file |
| `docker/**` | All files under docker/, recursively |
| `configs/image/*.yaml` | YAML files directly in configs/image/ |
| `src/data/*.py` | Python files directly in src/data/ |
| `src/training/**` | All files under src/training/, recursively |

______________________________________________________________________

## Bootstrapping

When generating a doc-map.yaml for the first time, the skill scans each doc
for source file references, then groups them into mapping entries. The
auto-generated `covers` field is derived from the context around each reference
in the doc (the section heading, surrounding sentences).

The user should then:

1. Review and correct the auto-generated `covers` values.
1. Add source files that *should* be covered but aren't yet referenced in
   the doc. **This is the most important step** — it's where you encode
   knowledge that the grep can't discover.
1. Add `scope` fields where a source file serves multiple docs.
1. Remove any entries that are false positives (doc mentions a file in
   passing but isn't responsible for documenting it).

______________________________________________________________________

## Self-maintenance

The doc-map.yaml is itself subject to drift. The skill checks for:

- **Dead patterns**: glob patterns that match zero files in the repo.
  Suggests the source was renamed, moved, or deleted.
- **Uncovered additions**: new files in directories covered by existing glob
  patterns. These are automatically covered — no action needed unless the
  `covers` field should be updated.
- **Orphan docs**: doc files in `docs/` that have no mapping entry. May be
  intentional (standalone docs like a changelog) or may indicate a missing
  mapping.
- **Orphan sources**: source files in frequently-mapped directories that
  aren't covered by any mapping. Suggests a new file was added without
  updating the mapping.
