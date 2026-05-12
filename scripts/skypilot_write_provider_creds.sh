#!/bin/bash
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

  # NOTE: the compartment_ocid lives in this bash process's memory (variable
  # expansion into a function-local positional arg) but NOT in any process's
  # argv — `printf` here is a bash builtin (no fork) and the downstream python3
  # heredoc receives the fragment via the SYNTH_UPSERT_FRAGMENT env var, not
  # argv. So `/proc/<pid>/cmdline` exposes nothing for either process; only
  # the owner-readable `/proc/<bash-pid>/environ` and in-process memory carry
  # the OCID. See #876.
  upsert_sky_config_key oci "$(printf 'oci:\n  default:\n    compartment_ocid: %s\n    image_tag_general: %s\n' \
    "${compartment_ocid}" "${OCI_IMAGE_TAG:-skypilot:cpu-ubuntu-2204}")"
}

# Per-key upsert into ~/.sky/config.yaml. Thin bash wrapper around the sibling
# scripts/upsert_sky_config_key.py module — the Python module owns the YAML
# parsing, error messages, and file-write semantics so the parser branches are
# pytest-unit-testable directly (see tests/scripts/test_upsert_sky_config_key.py).
# The wrapper's responsibilities: ensure ~/.sky/ exists, carry the
# (potentially secret-bearing) fragment via the SYNTH_UPSERT_FRAGMENT env var
# so it never reaches /proc/<pid>/cmdline, invoke the module, and propagate
# its exit code via set -e. Full contract docs (per-key upsert semantics,
# wholesale-replace within the managed key, secret-carriage rationale) live
# in the Python module's docstring.
upsert_sky_config_key() {
  local key="$1"
  local fragment="$2"
  local sky_config="$HOME/.sky/config.yaml"

  mkdir -p "$HOME/.sky"

  SYNTH_UPSERT_FRAGMENT="${fragment}" python3 "${BASH_SOURCE%/*}/upsert_sky_config_key.py" \
    "${key}" "${sky_config}"
}

write_local_sky_config() {
  # SkyPilot's managed-jobs controller defaults to cpus=4+, mem=4x, disk_size=50.
  # Those defaults don't fit on the kind cluster `sky local up` provisions in CI
  # (no node satisfies the CPU/memory floor; k8s rejects disk_size altogether).
  # Shrink the controller request so the controller pod schedules on kind and
  # the worker job can run. Omit disk_size — k8s "doesn't support" it; the
  # controller uses the pod's ephemeral storage / a PVC instead.
  local jobs_fragment
  read -r -d '' jobs_fragment <<'YAML' || true
jobs:
  controller:
    resources:
      cpus: 1+
      memory: 1+
YAML
  upsert_sky_config_key jobs "${jobs_fragment}"
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
    runpod)
      write_runpod_creds
      ;;
    oci)
      write_oci_creds
      ;;
    local)
      write_local_sky_config
      ;;
    *)
      echo "::error::unknown provider: ${PROVIDER} (expected runpod | oci | local)" >&2
      exit 1
      ;;
  esac
}

main "$@"
