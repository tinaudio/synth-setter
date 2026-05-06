#!/usr/bin/env bash
# Bootstrap SkyPilot credentials before `sky check` / `sky.launch`.
#
# Always (R2 is shared across compute providers):
#   - Write ~/.cloudflare/r2.credentials (mode 600, [r2] profile, AWS-style keys)
#     and ~/.cloudflare/accountid (mode 600, plain text) — consumed by SkyPilot's
#     R2 storage adaptor (sky/adaptors/cloudflare.py) once #749 is unblocked.
#   - Print rclone-prefixed env-var lines (KEY=VALUE) to stdout. The launcher
#     parses these and injects them into its own subprocess env so the spec's
#     R2 upload (rclone copyto) sees rclone-style creds without the caller
#     needing to bridge bare R2_* → RCLONE_CONFIG_R2_* by hand.
#
# Per-provider (gated on --provider runpod | oci):
#   - runpod: ~/.runpod/config.toml
#   - oci:    ~/.oci/config + ~/.oci/oci_api_key.pem + ~/.sky/config.yaml
#
# Idempotency: if a target file already exists with non-empty content the
# bootstrap leaves it alone — local-dev operators who hand-manage cred files
# must not be silently clobbered. Pass --force to overwrite.
#
# Required env: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_ACCOUNT_ID
# Provider-specific required env: see write_runpod_creds / write_oci_creds.
set -euo pipefail

umask 077

PROVIDER=""
FORCE=0

usage() {
  cat >&2 <<'EOF'
Usage: skypilot_write_provider_creds.sh --provider <runpod|oci> [--force]

Writes R2 file_mounts creds + per-provider compute creds. Prints rclone-
prefixed env-var lines to stdout for the caller to source/forward.
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

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "::error::$name is empty" >&2
    exit 1
  fi
}

# Returns 0 when the file at $1 already exists with non-empty content AND
# --force was not requested. Caller treats that as "skip overwrite".
should_skip_existing() {
  local path="$1"
  [[ "${FORCE}" -eq 0 && -s "${path}" ]]
}

write_r2_credentials_file() {
  local creds="$HOME/.cloudflare/r2.credentials"
  if should_skip_existing "${creds}"; then
    echo "::notice::skipping existing ${creds} (pass --force to overwrite)" >&2
    return 0
  fi
  mkdir -p "$HOME/.cloudflare"
  printf '[r2]\naws_access_key_id = %s\naws_secret_access_key = %s\n' \
    "$R2_ACCESS_KEY_ID" "$R2_SECRET_ACCESS_KEY" \
    > "${creds}"
  chmod 600 "${creds}"
}

write_r2_accountid_file() {
  local accountid="$HOME/.cloudflare/accountid"
  if should_skip_existing "${accountid}"; then
    echo "::notice::skipping existing ${accountid} (pass --force to overwrite)" >&2
    return 0
  fi
  mkdir -p "$HOME/.cloudflare"
  printf '%s\n' "$R2_ACCOUNT_ID" > "${accountid}"
  chmod 600 "${accountid}"
}

emit_rclone_env_lines() {
  printf 'RCLONE_CONFIG_R2_TYPE=s3\n'
  printf 'RCLONE_CONFIG_R2_PROVIDER=Cloudflare\n'
  printf 'RCLONE_CONFIG_R2_ACCESS_KEY_ID=%s\n' "$R2_ACCESS_KEY_ID"
  printf 'RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=%s\n' "$R2_SECRET_ACCESS_KEY"
  printf 'RCLONE_CONFIG_R2_ENDPOINT=%s\n' "$R2_ENDPOINT"
}

write_r2_creds() {
  require_var R2_ACCESS_KEY_ID
  require_var R2_SECRET_ACCESS_KEY
  require_var R2_ENDPOINT
  require_var R2_ACCOUNT_ID

  write_r2_credentials_file
  write_r2_accountid_file
  emit_rclone_env_lines
}

write_runpod_creds() {
  require_var RUNPOD_API_KEY
  local config="$HOME/.runpod/config.toml"
  if should_skip_existing "${config}"; then
    echo "::notice::skipping existing ${config} (pass --force to overwrite)" >&2
    return 0
  fi
  mkdir -p "$HOME/.runpod"
  printf '[default]\napi_key = "%s"\n' "$RUNPOD_API_KEY" > "${config}"
  chmod 600 "${config}"
}

write_oci_creds() {
  require_var OCI_USER_OCID
  require_var OCI_TENANCY_OCID
  require_var OCI_FINGERPRINT
  require_var OCI_REGION
  require_var OCI_COMPARTMENT_OCID
  require_var OCI_API_KEY_PEM

  local oci_config="$HOME/.oci/config"
  local oci_key="$HOME/.oci/oci_api_key.pem"
  local sky_config="$HOME/.sky/config.yaml"

  mkdir -p "$HOME/.oci" "$HOME/.sky"

  if ! should_skip_existing "${oci_key}"; then
    printf '%s\n' "$OCI_API_KEY_PEM" > "${oci_key}"
    chmod 600 "${oci_key}"
  else
    echo "::notice::skipping existing ${oci_key} (pass --force to overwrite)" >&2
  fi

  if ! should_skip_existing "${oci_config}"; then
    printf '[DEFAULT]\nuser=%s\nfingerprint=%s\ntenancy=%s\nregion=%s\nkey_file=%s/.oci/oci_api_key.pem\n' \
      "$OCI_USER_OCID" "$OCI_FINGERPRINT" "$OCI_TENANCY_OCID" "$OCI_REGION" "$HOME" \
      > "${oci_config}"
    chmod 600 "${oci_config}"
    grep -Eq "^region=.+" "${oci_config}" \
      || { echo "::error::OCI_REGION secret is empty" >&2; exit 1; }
  else
    echo "::notice::skipping existing ${oci_config} (pass --force to overwrite)" >&2
  fi

  if ! should_skip_existing "${sky_config}"; then
    printf 'oci:\n  default:\n    compartment_ocid: %s\n    image_tag_general: %s\n' \
      "$OCI_COMPARTMENT_OCID" "${OCI_IMAGE_TAG:-skypilot:cpu-ubuntu-2204}" \
      > "${sky_config}"
    grep -Eq "compartment_ocid: .+" "${sky_config}" \
      || { echo "::error::OCI_COMPARTMENT_OCID secret is empty" >&2; exit 1; }
  else
    echo "::notice::skipping existing ${sky_config} (pass --force to overwrite)" >&2
  fi
}

main() {
  parse_args "$@"

  if [[ -z "${PROVIDER}" ]]; then
    echo "::error::--provider is required (runpod | oci)" >&2
    usage
    exit 1
  fi

  write_r2_creds

  case "${PROVIDER}" in
    runpod) write_runpod_creds ;;
    oci)    write_oci_creds ;;
    *)
      echo "::error::unknown provider: ${PROVIDER} (expected runpod | oci)" >&2
      exit 1
      ;;
  esac
}

main "$@"
