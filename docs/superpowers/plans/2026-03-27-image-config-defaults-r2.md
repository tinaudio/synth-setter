# Image Config: Remove Defaults + Add R2 Fields — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all `ImageConfig` fields required (no silent defaults) and add `r2_endpoint` + `r2_bucket` as config-driven values instead of secrets/env vars.

**Architecture:** Remove defaults from the Pydantic model so every YAML config must be explicit. Add `r2_endpoint` and `r2_bucket` as required fields. Update the Dockerfile to accept `R2_ENDPOINT` as a build-arg instead of a secret. Update Makefile and GHA workflow to match.

**Tech Stack:** Python/Pydantic, Docker/BuildKit, GitHub Actions, Make

______________________________________________________________________

### Task 1: Update tests — require all fields, add R2 field tests

**Files:**

- Modify: `tests/scripts/test_image_config.py`

- [ ] **Step 1: Add a complete YAML helper for test fixtures**

All tests that create YAML fixtures need to provide every field (no defaults).
Add this helper at the top of the file, after the imports:

```python
VALID_SHA = "a" * 40
VALID_ISSUE = 266

_COMPLETE_YAML = """\
dockerfile: docker/ubuntu22_04/Dockerfile
image: tinaudio/perm
base_image: "ubuntu@sha256:abc123"
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_index_url: "https://download.pytorch.org/whl/cu128"
r2_endpoint: "https://example.r2.cloudflarestorage.com"
r2_bucket: test-bucket
"""


def _write_config(tmp_path: Path, overrides: str = "") -> Path:
    """Write a complete config YAML, optionally appending overrides."""
    config_path = tmp_path / "dev-snapshot.yaml"
    config_path.write_text(_COMPLETE_YAML + overrides)
    return config_path
```

- [ ] **Step 2: Update existing tests to use `_write_config`**

Replace every `config_path.write_text("")` or `config_path.write_text("# minimal config\n")` with `_write_config(tmp_path)`. For tests that need specific overrides (e.g., `build_mode: source`), write only the override fields — they'll be appended after the complete YAML (last key wins in YAML).

For `TestLoadImageConfigValid.test_all_fields_populated`:

```python
def test_all_fields_populated(self, tmp_path: Path) -> None:
    """Valid YAML + runtime inputs produce ImageConfig with all fields set."""
    config_path = _write_config(tmp_path)

    result = load_image_config(
        config_path,
        github_sha=VALID_SHA,
        issue_number=VALID_ISSUE,
    )

    assert result.github_sha == VALID_SHA
    assert result.issue_number == VALID_ISSUE
    assert result.image_config_id == "dev-snapshot"
    assert result.r2_endpoint == "https://example.r2.cloudflarestorage.com"
    assert result.r2_bucket == "test-bucket"
```

For `TestGithubShaValidation` and `TestIssueNumberValidation` — replace
`config_path.write_text("")` with `config_path = _write_config(tmp_path)`.

For `TestImageConfigIdDerivation` — use `_write_config` but with custom
filenames:

```python
def test_dev_snapshot_yaml_gives_dev_snapshot_id(self, tmp_path: Path) -> None:
    config_path = tmp_path / "dev-snapshot.yaml"
    config_path.write_text(_COMPLETE_YAML)

    result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)
    assert result.image_config_id == "dev-snapshot"

def test_custom_name_gives_matching_id(self, tmp_path: Path) -> None:
    config_path = tmp_path / "my-custom-image.yaml"
    config_path.write_text(_COMPLETE_YAML)

    result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)
    assert result.image_config_id == "my-custom-image"
```

For `TestStaticFieldsAndYamlMerge`:

```python
def test_yaml_static_fields_override_defaults(self, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "build_mode: source\n")

    result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)
    assert result.build_mode == "source"

def test_yaml_empty_file_uses_defaults(self, tmp_path: Path) -> None:
    """Empty YAML now raises ValidationError — all fields are required."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("# just a comment\n")

    with pytest.raises(ValidationError):
        load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)
```

- [ ] **Step 3: Add test for missing field rejection**

```python
def test_missing_field_rejected(self, tmp_path: Path) -> None:
    """YAML missing a required field raises ValidationError."""
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text("dockerfile: docker/ubuntu22_04/Dockerfile\n")

    with pytest.raises(ValidationError):
        load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)
```

- [ ] **Step 4: Update drift-detection test**

```python
def test_static_field_values_match_dev_snapshot_yaml(self) -> None:
    """Real dev-snapshot.yaml fields match expected values (catches drift)."""
    config_path = Path("configs/image/dev-snapshot.yaml")

    result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    assert result.dockerfile == "docker/ubuntu22_04/Dockerfile"
    assert result.image == "tinaudio/perm"
    assert result.base_image == (
        "ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751"
    )
    assert result.base_image_tag == "ubuntu22_04"
    assert result.build_mode == "prebuilt"
    assert result.target_platform == "linux/amd64"
    assert result.torch_index_url == "https://download.pytorch.org/whl/cu128"
    assert result.r2_endpoint == (
        "https://efb9275d571811db929e83eb710b74a7.r2.cloudflarestorage.com"
    )
    assert result.r2_bucket == "intermediate-data"
    assert result.image_config_id == "dev-snapshot"
```

- [ ] **Step 5: Run tests to verify they fail (RED)**

Run: `pytest tests/scripts/test_image_config.py -v`

Expected: Multiple failures — `r2_endpoint` and `r2_bucket` don't exist on the model yet, empty YAML test expects `ValidationError` but currently passes with defaults.

- [ ] **Step 6: Commit test changes**

```bash
git add tests/scripts/test_image_config.py
git commit -m "test(docker): update image_config tests for required fields and R2 config

Tests now require all YAML fields (no defaults) and add assertions for
r2_endpoint and r2_bucket. Tests are RED — implementation follows.

Refs #311"
```

______________________________________________________________________

### Task 2: Update ImageConfig model — remove defaults, add R2 fields

**Files:**

- Modify: `scripts/image_config.py`

- [ ] **Step 1: Remove defaults and add R2 fields**

Replace the static fields section (lines 26-35):

```python
# --- Static fields (from YAML config, no defaults — all required) ---
dockerfile: str
image: str
base_image: str
base_image_tag: str
build_mode: Literal["source", "prebuilt"]
target_platform: Literal["linux/amd64", "linux/arm64"]
torch_index_url: str
r2_endpoint: str
r2_bucket: str
```

- [ ] **Step 2: Run tests to verify they pass (GREEN)**

Run: `pytest tests/scripts/test_image_config.py -v`

Expected: All tests pass except `test_static_field_values_match_dev_snapshot_yaml` (YAML doesn't have R2 fields yet).

- [ ] **Step 3: Commit**

```bash
git add scripts/image_config.py
git commit -m "internal-feat(docker): remove ImageConfig defaults, add r2_endpoint and r2_bucket

All static fields are now required — no silent fallbacks from defaults.
r2_endpoint moves from secret to config value (it's a URL, not a credential).

Refs #311"
```

______________________________________________________________________

### Task 3: Update YAML config

**Files:**

- Modify: `configs/image/dev-snapshot.yaml`

- [ ] **Step 1: Add R2 fields to the YAML**

Append to `configs/image/dev-snapshot.yaml`:

```yaml
r2_endpoint: "https://efb9275d571811db929e83eb710b74a7.r2.cloudflarestorage.com"
r2_bucket: "intermediate-data"
```

- [ ] **Step 2: Run tests to verify all pass (GREEN)**

Run: `pytest tests/scripts/test_image_config.py -v`

Expected: All 18+ tests pass, including the drift-detection test.

- [ ] **Step 3: Commit**

```bash
git add configs/image/dev-snapshot.yaml
git commit -m "build(docker): add r2_endpoint and r2_bucket to image config YAML

r2_endpoint is a Cloudflare account URL, not a credential. Adding it to
the committed config makes the image build self-describing.

Refs #311"
```

______________________________________________________________________

### Task 4: Update Dockerfile — r2_endpoint as build-arg

**Files:**

- Modify: `docker/ubuntu22_04/Dockerfile` (lines 365-387, `r2-config-base` stage)

- [ ] **Step 1: Add R2_ENDPOINT ARG and remove secret mount**

Replace the `r2-config-base` stage (lines 365-387):

```dockerfile
FROM builder-install-synth-setter-deps AS r2-config-base
ARG R2_BUCKET
ARG R2_ENDPOINT
ENV R2_BUCKET=${R2_BUCKET}
RUN --mount=type=secret,id=r2_access_key_id \
    --mount=type=secret,id=r2_secret_access_key \
    set -eu; \
    if [ -z "${R2_ENDPOINT}" ] || [ ! -s /run/secrets/r2_access_key_id ] || [ ! -s /run/secrets/r2_secret_access_key ]; then \
        echo "WARNING: R2_ENDPOINT build-arg or R2 secrets (r2_access_key_id, r2_secret_access_key) are missing or empty." >&2; \
        echo "         R2 upload/download will not be available in this image." >&2; \
        mkdir -p /root/.config/rclone; \
        printf '[r2]\ntype = s3\nprovider = Cloudflare\n# credentials not configured\n' \
            > /root/.config/rclone/rclone.conf; \
    else \
        mkdir -p /root/.config/rclone; \
        printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\n' \
            "$(cat /run/secrets/r2_access_key_id)" \
            "$(cat /run/secrets/r2_secret_access_key)" \
            "${R2_ENDPOINT}" \
            > /root/.config/rclone/rclone.conf; \
        echo "R2 rclone config written." >&2; \
    fi; \
    chmod 600 /root/.config/rclone/rclone.conf
```

Key changes:

- Added `ARG R2_ENDPOINT`

- Removed `--mount=type=secret,id=r2_endpoint` from `RUN`

- Changed `"$(cat /run/secrets/r2_endpoint)"` → `"${R2_ENDPOINT}"`

- Updated the warning check from `[ ! -s /run/secrets/r2_endpoint ]` → `[ -z "${R2_ENDPOINT}" ]`

- [ ] **Step 2: Commit**

```bash
git add docker/ubuntu22_04/Dockerfile
git commit -m "build(docker): use R2_ENDPOINT as build-arg instead of secret

r2_endpoint is a URL, not a credential. Passing it as a build-arg
simplifies the build and makes it visible in docker inspect.

Refs #311"
```

______________________________________________________________________

### Task 5: Update Makefile

**Files:**

- Modify: `Makefile` (lines 90-102)

- [ ] **Step 1: Move r2_endpoint from DOCKER_SECRETS to build-args**

Replace lines 97-102:

```makefile
DOCKER_SECRETS = \
	--secret id=r2_access_key_id,env=R2_ACCESS_KEY_ID \
	--secret id=r2_secret_access_key,env=R2_SECRET_ACCESS_KEY \
	--secret id=wandb_api_key,env=WANDB_API_KEY \
	--build-arg R2_BUCKET=$(R2_BUCKET) \
	--build-arg R2_ENDPOINT=$(R2_ENDPOINT)
```

Key change: removed `--secret id=r2_endpoint,env=R2_ENDPOINT`, added `--build-arg R2_ENDPOINT=$(R2_ENDPOINT)`.

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -m "build(docker): pass R2_ENDPOINT as build-arg in Makefile

Matches Dockerfile change — r2_endpoint is now a build-arg, not a secret.

Refs #311"
```

______________________________________________________________________

### Task 6: Update GHA workflow

**Files:**

- Modify: `.github/workflows/docker-build-validation.yml` (on `ci/docker-image-build-push` branch)

- [ ] **Step 1: Export r2_endpoint and r2_bucket from config loader**

Update the field list in the "Load image config" step (line 52-53):

```python
              for field in ['dockerfile', 'image', 'base_image', 'base_image_tag',
                             'build_mode', 'target_platform', 'torch_index_url',
                             'r2_endpoint', 'r2_bucket']:
```

- [ ] **Step 2: Move r2_endpoint from secrets to build-args, add r2_bucket**

Update the "Build and push Docker image" step. In `build-args:` add:

```yaml
          build-args: |
            BUILD_MODE=${{ steps.config.outputs.build_mode }}
            BASE_IMAGE=${{ steps.config.outputs.base_image }}
            TORCH_INDEX_URL=${{ steps.config.outputs.torch_index_url }}
            SYNTH_PERMUTATIONS_GIT_REF=${{ github.sha }}
            R2_ENDPOINT=${{ steps.config.outputs.r2_endpoint }}
            R2_BUCKET=${{ steps.config.outputs.r2_bucket }}
```

In `secrets:` remove the `r2_endpoint` line:

```yaml
          secrets: |
            git_pat=${{ secrets.GIT_PAT }}
            r2_access_key_id=${{ secrets.R2_ACCESS_KEY_ID }}
            r2_secret_access_key=${{ secrets.R2_SECRET_ACCESS_KEY }}
            wandb_api_key=${{ secrets.WANDB_API_KEY }}
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docker-build-validation.yml
git commit -m "build(docker): pass R2_ENDPOINT and R2_BUCKET as build-args in workflow

Config loader now exports r2_endpoint and r2_bucket. r2_endpoint moves
from secrets to build-args, matching the Dockerfile change.

Refs #311"
```

______________________________________________________________________

### Task 7: Final validation

- [ ] **Step 1: Run all tests**

Run: `pytest tests/scripts/test_image_config.py -v`

Expected: All tests pass.

- [ ] **Step 2: Run pre-commit hooks**

Run: `make format`

Expected: All hooks pass (codespell on CHANGELOG.md is a pre-existing issue).

- [ ] **Step 3: Run full test suite**

Run: `make test`

Expected: All tests pass.
