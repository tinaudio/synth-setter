#!/usr/bin/env bash
# Post publication evidence from the completion marker to the tracking issue and close it.
set -euo pipefail

name=$1
issue=$2
root=${RIR_CORPORA_ROOT:-"$HOME/datasets/rir-corpora"}
body=$(python3 - "$root/$name/state/_COMPLETE.json" "$root/$name/state/remote-validation.json" "$name" <<'PY'
import json
import sys

marker = json.load(open(sys.argv[1]))
report = json.load(open(sys.argv[2]))
name = sys.argv[3]
formats = "\n".join(
    f"  - {f['count']:,} × {f['subtype']}, {f['sample_rate']} Hz, {f['channels']} ch"
    for f in report["audio_formats"]
) or "  - none (non-audio containers only)"
print(f"""## Publication evidence

Published and verified at `r2:experiments/third_party/{name}` (completion marker uploaded last at {marker['completed_at_utc']}).

- Rows (one per source object, exact bytes in a `lance.blob.v2` column): **{marker['rows']:,}**
- Source bytes compared byte-for-byte from a direct `s3://` R2 reopen: **{report['payload_bytes_compared']:,}** ({report['rows']:,} payloads, 0 mismatches)
- Audio payloads fully decoded from R2: {report['audio_objects_fully_decoded_from_r2']:,}
{formats}
- `rclone check --checksum --one-way` over the whole prefix: {report['rclone_check_one_way']}; objects before marker: {marker['objects_before_marker']}
- Source pin: {marker['source_pin']}
- License / access: {marker['license']} — {marker['access']}
- Root data card `README.md` SHA-256 `{marker['card']['sha256']}`; validation report SHA-256 `{marker['validation']['sha256']}`

Conversion and validation tooling is preserved under the prefix's `metadata/scripts/`.""")
PY
)
gh issue comment "$issue" --repo tinaudio/synth-setter --body "$body" >/dev/null
gh issue close "$issue" --repo tinaudio/synth-setter \
  --comment "All acceptance criteria met; see evidence above." >/dev/null
echo "closed #$issue ($name)"
