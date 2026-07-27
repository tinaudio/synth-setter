#!/usr/bin/env bash
# Bootstrap SkyPilot R2 + per-provider creds to disk before `sky check` /
# `sky.jobs.launch`. No stdout output by design (safe for tee'd contexts).
#
# Providers (gated on --provider runpod | vast): the local (kubernetes / kind)
# provider needs no compute auth — the launcher skips this script for that
# case. The kind controller shrink (~/.sky/config.yaml) is written by the
# launcher's `_ensure_ci_sky_config` when SYNTH_SETTER_CI_MODE is truthy.
#
# Required env:
#   RCLONE_CONFIG_R2_ACCESS_KEY_ID
#   RCLONE_CONFIG_R2_SECRET_ACCESS_KEY
#   RCLONE_CONFIG_R2_ENDPOINT
#   R2_ACCOUNT_ID
# Provider-specific required env: see write_runpod_creds / write_vast_creds.
#
# Idempotency + skip semantics: see should_skip_existing / notice_skip_existing.
set -euo pipefail

umask 077

PROVIDER=""
FORCE=0

usage() {
  cat >&2 <<'EOF'
Usage: skypilot_write_provider_creds.sh --provider <runpod|vast> [--force]

Writes R2 + per-provider compute creds to disk. No stdout output by design
so callers can run this in a tee'd context without leaking secrets.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --provider)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          echo "::error::--provider requires a value (runpod | vast)" >&2
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
# stdout (no trailing newline added) for capture by the caller.
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

write_vast_creds() {
  local api_key
  api_key="$(resolve_var VAST_API_KEY)"
  # Path pinned by SkyPilot's Vast adaptor (sky/clouds/vast.py _CREDENTIAL_PATH).
  local key_file="$HOME/.config/vastai/vast_api_key"
  if should_skip_existing "${key_file}"; then
    notice_skip_existing "${key_file}"
    return 0
  fi
  mkdir -p "$HOME/.config/vastai"
  printf '%s\n' "${api_key}" > "${key_file}"
  chmod 600 "${key_file}"
}

main() {
  parse_args "$@"

  if [[ -z "${PROVIDER}" ]]; then
    echo "::error::--provider is required (runpod | vast)" >&2
    usage
    exit 1
  fi

  write_r2_creds

  case "${PROVIDER}" in
    runpod)
      write_runpod_creds
      ;;
    vast)
      write_vast_creds
      ;;
    *)
      echo "::error::unknown provider: ${PROVIDER} (expected runpod | vast)" >&2
      exit 1
      ;;
  esac
}

main "$@"
