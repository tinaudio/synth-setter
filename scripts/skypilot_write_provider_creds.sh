#!/usr/bin/env bash
# Write per-provider SkyPilot credentials before `sky check` / `sky.launch`.
# Reads $PROVIDER and provider-specific env vars set by the caller.
# Writes ~/.runpod/config.toml (RunPod) or ~/.oci/config + ~/.oci/oci_api_key.pem
# + ~/.sky/config.yaml (OCI) under umask 077 so every cred file is mode 600.
set -euo pipefail

umask 077

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "::error::$name is empty" >&2
    exit 1
  fi
}

write_runpod_creds() {
  require_var RUNPOD_API_KEY
  mkdir -p "$HOME/.runpod"
  printf '[default]\napi_key = "%s"\n' "$RUNPOD_API_KEY" > "$HOME/.runpod/config.toml"
}

write_oci_creds() {
  require_var OCI_USER_OCID
  require_var OCI_TENANCY_OCID
  require_var OCI_FINGERPRINT
  require_var OCI_REGION
  require_var OCI_COMPARTMENT_OCID
  require_var OCI_API_KEY_PEM

  mkdir -p "$HOME/.oci" "$HOME/.sky"
  printf '%s\n' "$OCI_API_KEY_PEM" > "$HOME/.oci/oci_api_key.pem"
  chmod 600 "$HOME/.oci/oci_api_key.pem"
  printf '[DEFAULT]\nuser=%s\nfingerprint=%s\ntenancy=%s\nregion=%s\nkey_file=%s/.oci/oci_api_key.pem\n' \
    "$OCI_USER_OCID" "$OCI_FINGERPRINT" "$OCI_TENANCY_OCID" "$OCI_REGION" "$HOME" \
    > "$HOME/.oci/config"
  chmod 600 "$HOME/.oci/config"
  printf 'oci:\n  default:\n    compartment_ocid: %s\n    image_tag_general: %s\n' \
    "$OCI_COMPARTMENT_OCID" "${OCI_IMAGE_TAG:-skypilot:cpu-ubuntu-2204}" \
    > "$HOME/.sky/config.yaml"

  grep -Eq "^region=.+" "$HOME/.oci/config" \
    || { echo "::error::OCI_REGION secret is empty" >&2; exit 1; }
  grep -Eq "compartment_ocid: .+" "$HOME/.sky/config.yaml" \
    || { echo "::error::OCI_COMPARTMENT_OCID secret is empty" >&2; exit 1; }
}

main() {
  require_var PROVIDER
  case "$PROVIDER" in
    runpod)
      write_runpod_creds
      ;;
    oci)
      write_oci_creds
      ;;
    *)
      echo "::error::unknown PROVIDER=$PROVIDER" >&2
      exit 1
      ;;
  esac
}

main "$@"
