#!/usr/bin/env bash
# Bootstrap SkyPilot credentials before `sky check` / `sky.launch`.
#
# Writes cred files to disk only. Emits NO secrets to stdout — every caller
# can safely run this in a tee'd context without leaking secrets to public
# logs. (Status/notice messages go to stderr; errors go to stderr.)
#
# Always (R2 is shared across compute providers):
#   - ~/.cloudflare/r2.credentials  (mode 600, [r2] AWS-style profile) —
#     consumed by SkyPilot's R2 storage adaptor (sky/adaptors/cloudflare.py)
#     once #749 is unblocked.
#   - ~/.cloudflare/accountid       (mode 600, plain text)
#
# Per-provider (gated on --provider runpod | oci | local):
#   - runpod: ~/.runpod/config.toml
#   - oci:    ~/.oci/config + ~/.oci/oci_api_key.pem + upserts the `oci:`
#             top-level key into ~/.sky/config.yaml.
#   - local:  upserts the `jobs:` top-level key (managed-jobs controller
#             resources, shrunken so the controller pod schedules on the kind
#             cluster `sky local up` provisions) into ~/.sky/config.yaml. No
#             compute provider auth is needed; R2 creds are still required.
#
# ~/.sky/config.yaml is shared between OCI and local: writes are per-key
# upserts (replace the top-level key we manage, preserve everything else),
# so a multi-provider local-dev flow (e.g. running --provider oci then
# --provider local) ends with both keys present in the file.
#
# Idempotency: if a target file already exists with non-empty content the
# bootstrap leaves it alone — local-dev operators who hand-manage cred files
# must not be silently clobbered. Pass --force to overwrite. (Mode is still
# tightened to 0600 on the skip path so a hand-managed loose-perms file
# doesn't stay world-readable.) Exception: ~/.sky/config.yaml is always
# upserted at the top-level-key granularity (see upsert_sky_config_key) — the
# skip-existing / --force semantics don't apply, since the operation only
# touches the keys this script owns and preserves everything else.
#
# R2 env-var resolution: callers must supply the rclone-prefixed names
# (`RCLONE_CONFIG_R2_*`), matching `.env.example` and the GitHub Actions
# secrets table. `R2_ACCOUNT_ID` has no rclone-prefixed alias — it must be
# set by that name (it's written to ~/.cloudflare/accountid for SkyPilot's
# R2 storage adaptor).
#
# Required env:
#   RCLONE_CONFIG_R2_ACCESS_KEY_ID
#   RCLONE_CONFIG_R2_SECRET_ACCESS_KEY
#   RCLONE_CONFIG_R2_ENDPOINT
#   R2_ACCOUNT_ID
# Provider-specific required env: see write_runpod_creds / write_oci_creds.
set -euo pipefail

umask 077

PROVIDER=""
FORCE=0

usage() {
  cat >&2 <<'EOF'
Usage: skypilot_write_provider_creds.sh --provider <runpod|oci|local> [--force]

Writes R2 + per-provider compute creds to disk. No stdout output by design
so callers can run this in a tee'd context without leaking secrets.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --provider)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          echo "::error::--provider requires a value (runpod | oci | local)" >&2
          usage
          exit 1
        fi
        PROVIDER="$2"
        shift 2
        ;;
      --provider=*)
        PROVIDER="${1#*=}"
        shift
        ;;
      --force)
        FORCE=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "::error::unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

# Resolve $1 from env. If empty, fail. Returns the resolved value verbatim on
# stdout (no trailing newline added) for capture by the caller. Most callers
# pass single-line values, but `OCI_API_KEY_PEM` is multi-line — `printf '%s'`
# preserves the original content either way.
resolve_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "::error::${name} is empty" >&2
    exit 1
  fi
  printf '%s' "${value}"
}

# Returns 0 when the file at $1 already exists with non-empty content AND
# --force was not requested. Caller treats that as "skip overwrite".
should_skip_existing() {
  local path="$1"
  [[ "${FORCE}" -eq 0 && -s "${path}" ]]
}

# Emit the standard skip-notice and tighten the existing file's mode to 0600.
# Called from each skip branch so a hand-managed cred file with loose perms
# (e.g. 0644) doesn't stay world-readable just because we kept its content.
notice_skip_existing() {
  local path="$1"
  echo "::notice::skipping existing ${path} (pass --force to overwrite)" >&2
  chmod 600 "${path}"
}

write_r2_creds() {
  local access_key secret_key endpoint account_id
  access_key="$(resolve_var RCLONE_CONFIG_R2_ACCESS_KEY_ID)"
  secret_key="$(resolve_var RCLONE_CONFIG_R2_SECRET_ACCESS_KEY)"
  endpoint="$(resolve_var RCLONE_CONFIG_R2_ENDPOINT)"
  account_id="$(resolve_var R2_ACCOUNT_ID)"
  : "${endpoint?}" # endpoint isn't written to disk by this script — adaptor reads RCLONE_CONFIG_R2_ENDPOINT — but we validate it's set.

  local creds="$HOME/.cloudflare/r2.credentials"
  if should_skip_existing "${creds}"; then
    notice_skip_existing "${creds}"
  else
    mkdir -p "$HOME/.cloudflare"
    printf '[r2]\naws_access_key_id = %s\naws_secret_access_key = %s\n' \
      "${access_key}" "${secret_key}" \
      > "${creds}"
    chmod 600 "${creds}"
  fi

  local accountid="$HOME/.cloudflare/accountid"
  if should_skip_existing "${accountid}"; then
    notice_skip_existing "${accountid}"
  else
    mkdir -p "$HOME/.cloudflare"
    printf '%s\n' "${account_id}" > "${accountid}"
    chmod 600 "${accountid}"
  fi
}

write_runpod_creds() {
  local api_key
  api_key="$(resolve_var RUNPOD_API_KEY)"
  local config="$HOME/.runpod/config.toml"
  if should_skip_existing "${config}"; then
    notice_skip_existing "${config}"
    return 0
  fi
  mkdir -p "$HOME/.runpod"
  printf '[default]\napi_key = "%s"\n' "${api_key}" > "${config}"
  chmod 600 "${config}"
}

write_oci_creds() {
  local user_ocid tenancy_ocid fingerprint region compartment_ocid api_key_pem
  user_ocid="$(resolve_var OCI_USER_OCID)"
  tenancy_ocid="$(resolve_var OCI_TENANCY_OCID)"
  fingerprint="$(resolve_var OCI_FINGERPRINT)"
  region="$(resolve_var OCI_REGION)"
  compartment_ocid="$(resolve_var OCI_COMPARTMENT_OCID)"
  api_key_pem="$(resolve_var OCI_API_KEY_PEM)"

  local oci_config="$HOME/.oci/config"
  local oci_key="$HOME/.oci/oci_api_key.pem"
  local sky_config="$HOME/.sky/config.yaml"

  mkdir -p "$HOME/.oci" "$HOME/.sky"

  if should_skip_existing "${oci_key}"; then
    notice_skip_existing "${oci_key}"
  else
    printf '%s\n' "${api_key_pem}" > "${oci_key}"
    chmod 600 "${oci_key}"
  fi

  if should_skip_existing "${oci_config}"; then
    notice_skip_existing "${oci_config}"
  else
    printf '[DEFAULT]\nuser=%s\nfingerprint=%s\ntenancy=%s\nregion=%s\nkey_file=%s/.oci/oci_api_key.pem\n' \
      "${user_ocid}" "${fingerprint}" "${tenancy_ocid}" "${region}" "$HOME" \
      > "${oci_config}"
    chmod 600 "${oci_config}"
  fi

  upsert_sky_config_key oci "$(printf 'oci:\n  default:\n    compartment_ocid: %s\n    image_tag_general: %s\n' \
    "${compartment_ocid}" "${OCI_IMAGE_TAG:-skypilot:cpu-ubuntu-2204}")"
}

# Per-key upsert into ~/.sky/config.yaml. The file is shared by multiple providers
# (OCI writes `oci:`; local writes `jobs:` for the managed-jobs controller resource
# floor), and an earlier "skip if file exists" guard meant running the script for
# one provider after another would silently leave the second provider without its
# section. Upsert by top-level key instead: replace exactly the key we manage,
# preserve the *data* under any other top-level keys the user (or another
# provider's run) already populated. NB: the file is re-serialized via PyYAML
# `safe_dump`, so comments and original key ordering / formatting are dropped —
# only mapping data round-trips. This is acceptable because `~/.sky/config.yaml`
# is bootstrap-owned in CI; hand-managed local-dev configs that rely on comments
# should be edited outside this script.
#
# Secret-borne fragments (e.g. `oci:` carrying OCI_COMPARTMENT_OCID) are passed
# to python3 via an env var rather than argv — `/proc/<pid>/cmdline` is
# world-readable on Linux but `/proc/<pid>/environ` is owner-readable, so an
# env-borne secret can't be observed by other users on the runner via ps.
upsert_sky_config_key() {
  local key="$1"
  local fragment="$2"
  local sky_config="$HOME/.sky/config.yaml"

  mkdir -p "$HOME/.sky"

  SYNTH_UPSERT_FRAGMENT="${fragment}" python3 - "${key}" "${sky_config}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

key, path_str = sys.argv[1], sys.argv[2]
fragment = os.environ.pop("SYNTH_UPSERT_FRAGMENT")
path = Path(path_str)
existing = {}
if path.is_file() and path.stat().st_size > 0:
    existing = yaml.safe_load(path.read_text()) or {}
if not isinstance(existing, dict):
    sys.exit(
        f"upsert_sky_config_key: {path} is not a YAML mapping at the top level "
        f"(got {type(existing).__name__}); refusing to upsert. Fix or remove the file."
    )
fragment_doc = yaml.safe_load(fragment) or {}
if not isinstance(fragment_doc, dict):
    sys.exit(
        f"upsert_sky_config_key: fragment is not a YAML mapping "
        f"(got {type(fragment_doc).__name__})"
    )
if key not in fragment_doc:
    sys.exit(f"upsert_sky_config_key: fragment missing top-level {key!r}")
existing[key] = fragment_doc[key]
path.write_text(yaml.safe_dump(existing, sort_keys=False))
os.chmod(path, 0o600)
PY
}

write_local_sky_config() {
  # SkyPilot's managed-jobs controller defaults to cpus=4+, mem=4x, disk_size=50.
  # Those defaults don't fit on the kind cluster `sky local up` provisions in CI
  # (no node satisfies the CPU/memory floor; k8s rejects disk_size altogether).
  # Shrink the controller request so the controller pod schedules on kind and
  # the worker job can run. Omit disk_size — k8s "doesn't support" it; the
  # controller uses the pod's ephemeral storage / a PVC instead.
  upsert_sky_config_key jobs "$(cat <<'YAML'
jobs:
  controller:
    resources:
      cpus: 1+
      memory: 1+
YAML
)"
}

main() {
  parse_args "$@"

  if [[ -z "${PROVIDER}" ]]; then
    echo "::error::--provider is required (runpod | oci | local)" >&2
    usage
    exit 1
  fi

  write_r2_creds

  case "${PROVIDER}" in
    runpod) write_runpod_creds ;;
    oci)    write_oci_creds ;;
    local)  write_local_sky_config ;;
    *)
      echo "::error::unknown provider: ${PROVIDER} (expected runpod | oci | local)" >&2
      exit 1
      ;;
  esac
}

main "$@"
