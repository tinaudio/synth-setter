# RIR corpus publication tooling

Publishes third-party room-impulse-response corpora as immutable, verified Lance releases under
`r2:experiments/third_party/<Corpus>`, following the same contract as the earlier ESC50, NSynth,
and BBC Sound Effects publications: one row per source object with the exact bytes in a
`lance.blob.v2` column, a hash-pinned source inventory, `rclone copy --checksum --immutable`, an
exhaustive direct-R2 verification, a root `README.md` data card, and `_COMPLETE.json` uploaded
strictly last. The batch it was written for is the RIR set used by
[arXiv:2510.23158](https://arxiv.org/abs/2510.23158).

Run everything from the repository root with `uv run`; the local working root defaults to
`~/datasets/rir-corpora` and is overridden with `RIR_CORPORA_ROOT`.

## Stages

| Stage     | Command                                                                  | What it does                                                                                                                                                                                                    |
| --------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| acquire   | `python -m scripts.data_publication.rir_corpora.fetch_sources <family>`  | Downloads sources into `<root>/<Corpus>/source/` and writes `acquisition.json`. Zenodo files are MD5-checked against the record; direct archives record response headers; every object is SHA-256 hashed.       |
| `extract` | `python -m scripts.data_publication.rir_corpora.rirpub extract <Corpus>` | Unpacks the pinned archives, drops archive litter (`.DS_Store`, `__MACOSX/`), and writes the per-object SHA-256 inventory under `state/`.                                                                       |
| `build`   | `python -m scripts.data_publication.rir_corpora.rirpub build <Corpus>`   | Writes `publication/all.lance`, copies documentation objects under `publication/source/`, records inventories, reports, tool versions, and the tool itself under `metadata/`, and renders the card.             |
| `upload`  | `python -m scripts.data_publication.rir_corpora.rirpub upload <Corpus>`  | `rclone copy --checksum --immutable`, excluding the completion marker.                                                                                                                                          |
| `verify`  | `python -m scripts.data_publication.rir_corpora.rirpub verify <Corpus>`  | Reopens `all.lance` directly from R2, re-reads every payload and compares length and SHA-256 with the inventory, decodes every audio object, runs `rclone check --one-way`, then uploads `_COMPLETE.json` last. |

`drive.sh <Corpus>` waits for the acquisition record and runs the four stages in order, then
removes the extracted tree and the local Lance copy once R2 is verified. `post_evidence.sh <Corpus> <issue>` posts the completion evidence to the tracking issue and closes it.

Acquisition families: `zenodo [Corpus ...]`, `direct`, `ashir`, `openair`. Zenodo throttles a
single stream to about 1 MB/s, so downloads go through `aria2c` with eight connections per file;
point `RIR_ARIA2C` / `RIR_ARIA2_LIB` (or `RIR_ARIA2_ROOT`) at an aria2 install.

## Row schema

| Field                                             | Type                                            | Meaning                                           |
| ------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| `source_row_index`                                | `int32`                                         | Zero-based row in sorted source-path order.       |
| `source_path`                                     | `string`                                        | Relative POSIX path inside the extracted release. |
| `media_type`                                      | `string`                                        | MIME type inferred from the file extension.       |
| `source_num_bytes`                                | `int64`                                         | Exact source byte length.                         |
| `source_sha256`                                   | `string`                                        | SHA-256 of the exact bytes.                       |
| `audio_decodable`                                 | `bool`                                          | Whether libsndfile fully decoded the object.      |
| `sample_rate` / `channels` / `frames` / `subtype` | nullable `int32` / `int32` / `int64` / `string` | libsndfile header fields for decodable audio.     |
| `source_bytes`                                    | `lance.blob.v2`                                 | Exact, untranscoded source bytes.                 |

Container formats (SOFA, MAT, NumPy, pickle) are stored opaquely with `audio_decodable = false`.
Unpacking them into per-RIR rows is a derived dataset, not part of the immutable publication.

## Adding a corpus

Register a `Spec` in `specs.py`: name (also the R2 prefix), title, summary, source URLs, the
immutable pin, licence and citation text, the acquisition command lines to record in the card,
and either `archives` (ZIPs under `source/` that `extract` unpacks, with an optional custom
`extractor` for split archives) or `payload_root="source"` for corpora that arrive as loose
files. `exclude_globs` drop objects before inventory; `provenance_files` are copied into the
prefix's `source/_acquisition/`. Add the acquisition entry to `fetch_sources.py`, then file a
Task issue under the Storage Integration phase and run the stages.

## Published corpora

| Corpus (R2 prefix)    | Source pin                                                              | Licence                                     | Issue           |
| --------------------- | ----------------------------------------------------------------------- | ------------------------------------------- | --------------- |
| `MITIRSurvey`         | `Audio.zip` SHA-256 (no upstream version)                               | not stated upstream; internal research only | #3183           |
| `AachenIR`            | `air_database_release_1_4.zip` SHA-256                                  | not stated in the download; RWTH copyright  | #3182           |
| `EchoThief`           | July-2024 archive SHA-256                                               | custom: derivative works permitted          | #3181           |
| `ASHIR`               | Git `af0d3f51e19cf74dc77a603fd3135a14d8aeb29e`                          | CC BY-NC-SA 4.0                             | #3178           |
| `MultiRoomTransition` | Zenodo 10.5281/zenodo.13341566                                          | CC BY 4.0                                   | #3179           |
| `THKoelnSRIR`         | Zenodo 10.5281/zenodo.5031335                                           | CC BY 4.0                                   | #3187           |
| `MPRIR`               | Zenodo 10.5281/zenodo.11148712                                          | CC BY 4.0                                   | #3185           |
| `OpenAIR`             | York file-store crawl, per-object SHA-256                               | per-entry Creative Commons                  | #3184           |
| `TAUSRIR`             | Zenodo 10.5281/zenodo.6408611 (SRIR archive only)                       | custom non-commercial with attribution      | #3186           |
| `Arni`                | Zenodo 10.5281/zenodo.6985104                                           | CC BY 4.0                                   | #3180           |
| `ACE`                 | not acquired: registration host is gone, IEEE DataPort needs an account | CC BY-ND 4.0                                | #3177 (blocked) |

Every prefix also carries a byte-exact copy of this tooling under `metadata/scripts/` and the
exact command lines under `metadata/publication-commands.txt`.
