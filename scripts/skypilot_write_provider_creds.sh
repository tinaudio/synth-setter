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
#   - oci:    ~/.oci/config + ~/.oci/oci_api_key.pem + ~/.sky/config.yaml
#   - local:  no per-provider files — `sky local up` (kind cluster) needs
#             no compute provider auth. R2 creds are still required.
#
# Idempotency: if a target file already exists with non-empty content the
# bootstrap leaves it alone — local-dev operators who hand-manage cred files
# must not be silently clobbered. Pass --force to overwrite.
#
# R2 env-var resolution: each `R2_*` is read from env, falling back to its
# rclone-prefixed equivalent (`RCLONE_CONFIG_R2_*`) so a `.env` carrying only
# the rclone-prefixed names (per `.env.example`) still bootstraps cleanly.
# `R2_ACCOUNT_ID` has no rclone-prefixed alias — it must be set by name.
#
# Required env (fallbacks shown):
#   R2_ACCESS_KEY_ID       (or RCLONE_CONFIG_R2_ACCESS_KEY_ID)
#   R2_SECRET_ACCESS_KEY   (or RCLONE_CONFIG_R2_SECRET_ACCESS_KEY)
#   R2_ENDPOINT            (or RCLONE_CONFIG_R2_ENDPOINT)
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
        PROVIDER="${2:-}"
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

# Resolve $1 from env. If empty, fail. Returns the resolved value on stdout
# (single-line, no trailing newline) for capture by the caller.
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

write_r2_creds() {
  local access_key secret_key endpoint account_id
  access_key="$(resolve_var RCLONE_CONFIG_R2_ACCESS_KEY_ID)"
  secret_key="$(resolve_var RCLONE_CONFIG_R2_SECRET_ACCESS_KEY)"
  endpoint="$(resolve_var RCLONE_CONFIG_R2_ENDPOINT)"
  account_id="$(resolve_var R2_ACCOUNT_ID)"
  : "${endpoint?}" # endpoint isn't written to disk by this script — adaptor reads RCLONE_CONFIG_R2_ENDPOINT — but we validate it's set.

  local creds="$HOME/.cloudflare/r2.credentials"
  if should_skip_existing "${creds}"; then
    echo "::notice::skipping existing ${creds} (pass --force to overwrite)" >&2
  else
    mkdir -p "$HOME/.cloudflare"
    printf '[r2]\naws_access_key_id = %s\naws_secret_access_key = %s\n' \
      "${access_key}" "${secret_key}" \
      > "${creds}"
    chmod 600 "${creds}"
  fi

  local accountid="$HOME/.cloudflare/accountid"
  if should_skip_existing "${accountid}"; then
    echo "::notice::skipping existing ${accountid} (pass --force to overwrite)" >&2
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
    echo "::notice::skipping existing ${config} (pass --force to overwrite)" >&2
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
    echo "::notice::skipping existing ${oci_key} (pass --force to overwrite)" >&2
  else
    printf '%s\n' "${api_key_pem}" > "${oci_key}"
    chmod 600 "${oci_key}"
  fi

  if should_skip_existing "${oci_config}"; then
    echo "::notice::skipping existing ${oci_config} (pass --force to overwrite)" >&2
  else
    printf '[DEFAULT]\nuser=%s\nfingerprint=%s\ntenancy=%s\nregion=%s\nkey_file=%s/.oci/oci_api_key.pem\n' \
      "${user_ocid}" "${fingerprint}" "${tenancy_ocid}" "${region}" "$HOME" \
      > "${oci_config}"
    chmod 600 "${oci_config}"
  fi

  if should_skip_existing "${sky_config}"; then
    echo "::notice::skipping existing ${sky_config} (pass --force to overwrite)" >&2
  else
    printf 'oci:\n  default:\n    compartment_ocid: %s\n    image_tag_general: %s\n' \
      "${compartment_ocid}" "${OCI_IMAGE_TAG:-skypilot:cpu-ubuntu-2204}" \
      > "${sky_config}"
    chmod 600 "${sky_config}"
  fi
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
    local)  : ;; # kind cluster needs no compute provider auth
    *)
      echo "::error::unknown provider: ${PROVIDER} (expected runpod | oci | local)" >&2
      exit 1
      ;;
  esac
}

main "$@"
