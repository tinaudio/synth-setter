#!/usr/bin/env bash
# Wait for a corpus' acquisition record, then run the four publication stages
# and reclaim the extracted tree plus the local Lance copy once R2 is verified.
set -euo pipefail

name=$1
root=${RIR_CORPORA_ROOT:-"$HOME/datasets/rir-corpora"}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
acquisition="$root/$name/source/acquisition.json"
if [[ "$name" == OpenAIR ]]; then
  acquisition="$root/$name/acquisition.json"
fi
while [[ ! -f "$acquisition" ]]; do
  sleep 60
done
cd "$repo"
for stage in extract build upload verify; do
  echo "[$(date -Is)] $name $stage"
  uv run python -m scripts.data_publication.rir_corpora.rirpub "$stage" "$name"
done
rm -rf "$root/$name/extracted" "$root/$name/publication/all.lance"
echo "[$(date -Is)] $name DONE (extracted/ and local all.lance removed after verification)"
