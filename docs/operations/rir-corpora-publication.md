# Publishing a third-party RIR corpus to R2

Runbook for the tooling in
[`scripts/data_publication/rir_corpora/`](../../scripts/data_publication/rir_corpora/README.md),
which produced the ten room-impulse-response prefixes under `r2:experiments/third_party`
tracked by #3178–#3187 (ACE, #3177, is blocked on a registration step).

## Procedure

1. Register the corpus as a `Spec` in `specs.py` and, for a new source family, add it to
   `fetch_sources.py`. File a Task issue under the Storage Integration phase.
2. Acquire: `uv run python -m scripts.data_publication.rir_corpora.fetch_sources <family> [Corpus]`.
   Zenodo downloads need `aria2c` (`RIR_ARIA2C`, `RIR_ARIA2_LIB`); check free disk against the
   fail-closed threshold in `rirpub.py` before extracting large archives.
3. Publish: `scripts/data_publication/rir_corpora/drive.sh <Corpus>` runs `extract`, `build`,
   `upload`, and `verify` in order and reclaims the local extraction and Lance copy once the
   remote verification passes. Run the stages by hand with
   `uv run python -m scripts.data_publication.rir_corpora.rirpub <stage> <Corpus>` when a step
   needs inspection.
4. Evidence: `scripts/data_publication/rir_corpora/post_evidence.sh <Corpus> <issue>` posts the
   completion-marker figures to the tracking issue and closes it. Refresh
   `r2:experiments/third_party/_dataset_cards/README.md` with the new prefix.

## Invariants

- Every rclone operation passes `--checksum`; uploads also pass `--immutable`.
- `_COMPLETE.json` is the only completion signal and is uploaded strictly last, after every
  payload has been re-read from R2 and compared with the source inventory.
- Source bytes are never transcoded; container formats stay opaque rows. Per-RIR unpacking
  belongs in a derived dataset with its own issue.
- The published prefixes are private and carry per-corpus licence restrictions in their cards.
