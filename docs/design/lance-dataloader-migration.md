# Design Doc: Lance-Native Dataloader Migration

> **Status**: Draft
> **Author**: KT (khaled@tinaudio.com)
> **Last Updated**: 2026-06-16
> **Tracking**: #1738 (feat(training): Lance-native dataloader)
> **Related**: [lance-dataset-api-migration.md](lance-dataset-api-migration.md) (storage/write API; this doc is the *read*/dataloader sibling)

______________________________________________________________________

### Index

| §   | Section                  | What it covers                                                            |
| --- | ------------------------ | ------------------------------------------------------------------------- |
| §1  | Context                  | Why the current Lance read path exists and what it costs                  |
| §2  | Current architecture     | The h5-shaped shim, grounded in code                                      |
| §3  | Goals / non-goals        | What we are and aren't changing                                           |
| §4  | Decision                 | Map-style dataset + `__getitems__` projected read + pure collate          |
| §5  | Tradeoffs considered     | map vs iterable; SafeLanceDataset; shuffle buffer; the full option matrix |
| §6  | Network storage decision | Pre-stage R2 → local; why streaming loses for our shape                   |
| §7  | DDP analysis             | DDP correctness risk in the current code and why the new design fixes it  |
| §8  | Phase plan               | The four phases and their dependency graph                                |
| §9  | Testing strategy         | Cross-cutting test philosophy; per-phase detail lives in the phase issues |
| §10 | References               | All external + internal grounding links                                   |

______________________________________________________________________

## §1 Context

The Lance read path was ported from the HDF5 design. Almost every hand-rolled
piece in the current datamodule — the `batch_size=None` dataloader, the
batch-indexed `__getitem__`, and the `ShiftedBatchSampler` /
`WithinChunkShuffledSampler` / `ShuffledSampler` family — exists to work around a
single HDF5 limitation: **slow random access**. The workarounds keep reads
*contiguous* so h5py stays fast, then claw back approximate shuffling on top of
that contiguity.

Lance's headline property is fast random access (its own docs market "100x
faster random access" vs Parquet — see §10). So the premise those workarounds
were built on no longer holds. This migration deletes the reason they exist
rather than porting them to Lance.

Two goals drive the work: (1) **performance** — stop paying the h5-shaped
per-column read tax and the row-dict conversion tax; (2) **DDP correctness +
simplicity** — stop relying on Lightning silently rewriting a custom sampler,
and get clean rank sharding for free.

## §2 Current architecture (grounded)

- `VSTDataset` is **batch-indexed**: one `__getitem__` returns a whole batch, and
  the dataloader uses `batch_size=None`
  (`src/synth_setter/data/surge_datamodule.py`, `__getitem__` ~L227–272,
  `train_dataloader` ~L527–543).
- Each batch reads **one column at a time**. `__getitem__` calls
  `self.dataset_file["mel_spec"]`, then `["param_array"]`, then optionally
  `["music2latent"]` / `["audio"]` — each a *separate* read
  (`surge_datamodule.py` L236–256).
- The Lance adapter (`src/synth_setter/data/lance_datamodule.py`) makes a Lance
  column look like an `h5py.Dataset`. Contiguous slices go through
  `dataset.scanner(columns=[name], offset=…, limit=…)`; stepped/fancy indices go
  through `dataset.take(indices, columns=[name])` (`LanceColumn.__getitem__`
  L48–74). So each logical column is a separate Lance scan.
- `LanceShardFile.live_dataset()` already reopens the Lance dataset when
  `os.getpid()` changes — i.e. **worker-safe lazy reopen on fork is already
  implemented** (L108–122). This matters in §5: it is the one thing
  `SafeLanceDataset` would otherwise give us, and we already have it.
- Training uses `ShiftedBatchSampler`, which yields `(start, stop)` tuples
  (L362–390), so the *training* path already hits the contiguous-scan fast path.
- Shards are written in Lance format `data_storage_version="2.2"` — the
  `LANCE_DATA_STORAGE_VERSION = "2.2"` module constant
  (`src/synth_setter/pipeline/data/lance_shard.py` L21) passed as the
  `data_storage_version` write kwarg (L111/L142) — which has good random
  access + projection. Rows are large: mel ≈ 400 KB, audio ≈ 1.4 MB,
  music2latent ≈ 21 KB (from fake-mode shapes, `surge_datamodule.py`
  `_get_fake_item`).

**The two costs.** (1) N separate Lance scans per batch instead of one projected
read. (2) Arrow → NumPy → Torch conversion per column. Both are artifacts of the
h5-shaped interface, not of Lance.

## §3 Goals / non-goals

**Goals**

- One projected Lance read per batch (all needed columns in a single scan).
- Map-style, sample-indexed dataset with standard `DataLoader(batch_size=B, shuffle=True)` — no custom sampler, no `batch_size=None`.
- DDP via Lightning's default `DistributedSampler` injection, verified by test.
- Pre-staging R2 → local disk as the default for cloud training.
- Eventually drop HDF5 and the adapter indirection entirely.

**Non-goals**

- No new Lance features (versioning, compaction, vector indices) — that's the
  storage-API epic ([lance-dataset-api-migration.md](lance-dataset-api-migration.md)).
- No change to model-facing batch contents/semantics. The keys, shapes, dtypes,
  normalization, rescale, noise, and OT matching must be byte-identical.
- No backward compat with the legacy single-file `.lance` shim once cut over.

## §4 Decision

Move to a **map-style, sample-indexed** `Dataset` whose `__getitems__(indices)`
issues **one projected `take`** for all required columns, and move per-batch
math (mel norm, param rescale, noise, OT/Hungarian match) into a **pure
`collate_fn`**.

The key enabler is PyTorch's `__getitems__` (note the plural). PyTorch's
map-style fetcher calls `__getitems__(list_of_indices)` when the dataset defines
it, instead of calling `__getitem__` once per sample — so we get standard
DataLoader batching/shuffling **and** a single batched Lance read, with no
custom sampler (see §10: PyTorch data docs and `_MapDatasetFetcher`).

End state:

```python
class LanceMapDataset(torch.utils.data.Dataset):
    def __init__(self, uri, columns, storage_options=None):
        self.uri, self.columns = uri, columns
        self._storage_options = storage_options  # persisted so the fork-time reopen keeps R2 creds
        # Cached once; assumes an immutable staged split (no concurrent
        # compaction/append), like LanceShardFile.
        self._len = lance.dataset(uri, storage_options=storage_options).count_rows()
        self._ds = None                         # reopened after fork (getpid guard in _live)
    def __len__(self): return self._len
    def __getitems__(self, indices):            # single Lance take instead of N __getitem__ calls
        ds = self._live()                       # reopen on fork (reuse existing pattern)
        tbl = ds.take(indices, columns=self.columns)
        # to_numpy_ndarray(), not to_numpy() — Lance returns FixedShapeTensorArray for tensor columns.
        return {c: tbl.column(c).combine_chunks().to_numpy_ndarray() for c in self.columns}
    def __getitem__(self, i):                   # rare fallback; DataLoader uses __getitems__ for batches
        # __getitems__ returns a batch dict (leading dim = len(indices)); unwrap the single element.
        return {c: v[0] for c, v in self.__getitems__([i]).items()}
```

The two index paths return different shapes — `__getitems__` a batch dict (rows
stacked along a leading dim), `__getitem__` a single unwrapped sample — so a
`collate_fn` must never re-stack a `__getitems__` result.

`collate_fn` is the relocated, pure version of today's per-batch logic (norm,
rescale, noise, OT). One contract to flag for implementers: because
`__getitems__` already returns a whole batch (a dict of stacked arrays),
PyTorch's `_MapDatasetFetcher` passes that dict straight to `collate_fn` — so
`collate` receives a single batch dict, **not** a `list[dict]` of samples, and
`prepare_batch` is written to accept it as-is (no per-sample aggregation, no
double-stacking). That pass-through is a torch-internal detail
(`torch/utils/data/_utils/fetch.py`), so Phase 2 pins it with a test that fails
loudly if the `uv.lock`-pinned torch ever changes the collation path. The
dataloader becomes boring on purpose:
`DataLoader(ds, batch_size=B, shuffle=True, num_workers=n, collate_fn=collate, pin_memory=True)`.

## §5 Tradeoffs considered

### 5.1 Map-style vs iterable

| Axis               | Map-style (chosen)                                 | Iterable (`lance.torch.data.LanceDataset`)                       |
| ------------------ | -------------------------------------------------- | ---------------------------------------------------------------- |
| DDP sharding       | Free via Lightning `DistributedSampler` injection  | Must own rank × worker sharding (Lance `ShardedFragmentSampler`) |
| Shuffle            | True global permutation (`shuffle=True`)           | Approximate (reservoir buffer, k=256 in 7.0.0)                   |
| `__len__`          | Exact                                              | Fuzzy                                                            |
| Best when          | Finite, indexable, fits-or-stages locally          | Too big to enumerate / pure streaming from object store          |
| Random access cost | Cheap on local; **scatters I/O on object storage** | Sequential fragment scans + readahead (object-store friendly)    |

For a finite, indexable dataset trained off **local** disk (our case after
pre-staging — §6), map-style wins on every axis that matters and removes the DDP
footgun. Iterable is kept in our back pocket for a future "data no longer fits"
scenario. See §10 for the Lance PyTorch integration docs (confirming
`LanceDataset` is an `IterableDataset`) and `sampler.py` (the samplers).

### 5.2 Why **not** `SafeLanceDataset` (explicitly rejected)

`SafeLanceDataset` (`pylance==7.0.0`, confirmed in `uv.lock`) is **map-style**,
not iterable. Its batched fetch is:

```python
# pylance 7.0.0 — lance/torch/data.py, SafeLanceDataset.__getitems__
batch = self._ds.take(indices)
return batch.to_pylist()
```

Three reasons it is the wrong target:

1. **`to_pylist()` returns per-row Python dicts.** For a datamodule that emits
   full batches of large tensors (audio ≈ 1.4 MB/row), funneling through row
   dicts is more allocation and Python overhead than the current Arrow→NumPy
   path — a regression, not a win.
2. **It always uses `take(indices)` with no `columns=` projection** — it reads
   *every* column, never a contiguous `scanner(offset, limit)` and never a
   projected subset. So it would *regress* both the projection win and the
   contiguous-scan fast path our training sampler already exploits;
   `LanceMapDataset.__getitems__` passing `columns=self.columns` is precisely
   that projection.
3. **Its one genuine virtue — worker-safe lazy open — we already have** in
   `LanceShardFile.live_dataset()` (the `getpid` reopen). So it adds nothing.

Our design takes the projected read directly without `SafeLanceDataset`; the
DDP analysis is in §7.

### 5.3 Do we need a shuffle buffer?

No — not in the map-style design, and not for correctness. `shuffle=True` gives a
true global permutation; a shuffle buffer is an *I/O-locality* optimization, not
a shuffling mechanism. It only matters if random `take` makes us I/O-bound (the
object-storage case — avoided by pre-staging in §6). If we ever needed windowed
shuffling, the options are: a ~15-line windowed `BatchSampler` (map-style, but
reintroduces rank-awareness work), or the iterable path's built-in reservoir
shuffle (`ShardedBatchSampler(rank, world_size, randomize=True)`, k=256 — `rank`
and `world_size` are required positionals). **Decision: do not build
one. Gate it behind a benchmark if ever needed.**

### 5.4 Shuffle-quality note

`ShiftedBatchSampler` keeps disk-adjacent rows together within a batch (weak
shuffle). Moving to `shuffle=True` is a real *improvement* in statistical
shuffle quality, not just a simplification — relevant if rows are ordered by
preset/param sweep on disk.

## §6 Network storage decision: pre-stage R2 → local

Training will run off ~TBs on R2. On object storage, map-style `shuffle=True`
issues a *maximally scattered* `take` — for a 1024-row batch that's ~1024 range
GETs spread across the object, each paying request latency. That is the worst
access pattern for object storage; it inverts the "cheap shuffle" property that
holds on local NVMe.

**Decision: pre-stage each split to node-local disk once, then train off local.**
This is already the direction recorded in
[lance-dataset-api-migration.md](lance-dataset-api-migration.md) ("the training
dataloader keeps rclone local-first … random per-batch reads across epochs must
not hit the network"), and the machinery already exists:

- `VSTDataModule.prepare_data` calls `r2_io.download_dir_no_overwrite(...)`
  (`surge_datamodule.py` L462–472).
- `r2_io.r2_storage_options()` already builds the exact dict Lance needs for
  direct `s3://`/R2 reads (`r2_io.py` L188) — used today only by CI's
  stream-validate path.

R2 specifically makes pre-staging cheap: **zero egress fees** (Cloudflare R2),
so re-downloading the full dataset per job costs nothing. For a multi-epoch run,
amortizing one download across all epochs beats streaming every epoch decisively.

**Streaming (iterable + fragment sharding + readahead + reservoir shuffle,
optionally `cache=True`) is the fallback only if a split does not fit on
node-local disk.** Not our case at a couple of TB on modern NVMe nodes — but the
phase plan keeps it documented as the escape hatch.

DDP caveat to honor: Lightning runs `prepare_data` on local-rank-0 of **each
node** (`prepare_data_per_node=True` by default), which is exactly what we want
for node-local staging — provided `dataset_root` is a node-local path, not a
shared network mount. Phase 3 verifies this.

## §7 DDP analysis (load-bearing)

The DDP risk in the *current* code has nothing to do with Lance. The training
dataloader is `DataLoader(train_dataset, batch_size=None, sampler=ShiftedBatchSampler(...))`, the DDP config uses `strategy: ddp` on 4
devices (`configs/trainer/ddp.yaml`), and nothing sets `use_distributed_sampler`
→ Lightning's default `True`.

Under that default, Lightning **rewrites the dataloader's sampler** for DDP. Its
behavior is version-dependent and is exactly the thing we should not rely on
silently:

- Lightning special-cases plain `SequentialSampler` / `RandomSampler` and
  cleanly swaps in a `DistributedSampler`; for *custom* samplers it instead
  wraps/replaces with a warning ("Replacing custom sampler with distributed
  version …"). See §10 (Lightning trainer docs + issue #5271).
- With a custom `ShiftedBatchSampler` + `batch_size=None`, the most likely
  outcome is correct *sharding* (each integer index → one contiguous batch via
  `__getitem__`), but the sampler's **random per-epoch offset is silently
  dropped**, and the interaction is **untested** in the repo. If replacement
  ever fails to engage, every rank sees identical data — a silent correctness
  bug (duplicated gradients).

**Why the new design fixes this structurally:** dropping the custom sampler and
using `shuffle=True` means the dataset presents a plain `RandomSampler`, which
hits Lightning's *clean, special-cased* `DistributedSampler` path — the
well-tested one — instead of the custom-sampler wrapping path. We then add an
explicit DDP test (Phase 2, §9) asserting disjoint per-rank shards.

`SafeLanceDataset` does **not** help here: it is map-style too, so its DDP story
is identical to ours. The only Lance class with built-in rank sharding is the
*iterable* `LanceDataset`, and adopting it would *remove* the `DistributedSampler`
safety net and force manual rank × worker sharding — the opposite of simpler.

## §8 Phase plan

The pure `collate_fn` is the bridge the whole migration walks across: it stays
constant while storage, sampling, indexing, and DDP change underneath it, and it
is what lets us delete the hand-rolled code at the end without a flag day.

```
Phase 1  Extract pure prepare_batch (no-op refactor, pinned by tests)
   │         training · blocks 2,4
   ▼
Phase 2  Map-style LanceMapDataset + __getitems__ + collate (behind config switch)
   │         training · blocked by 1 · blocks 4
   ▼
Phase 4  Delete hand-rolled samplers, batch_size=None, and the h5 path
   ▲         code-health/training · blocked by 2 (and 3 for full h5 drop)
   │
Phase 3  Pre-stage R2 → local as default for cloud training   (parallel track — independent of 1,2)
             training/storage · soft-blocks 4's full h5 removal
```

Phase 4 completes in two steps: deleting the samplers and `batch_size=None`
unblocks once Phase 2 lands, while the full HDF5/adapter removal additionally
waits on Phase 3 so cloud training no longer depends on the h5 path.

Phase issues (filed, self-contained, worker-pointable), tracked under Epic #1738:

- Phase 1 — #1739 (extract pure `prepare_batch`)
- Phase 2 — #1740 (map-style `LanceMapDataset` + `__getitems__` + collate)
- Phase 3 — #1741 (pre-stage R2 → local)
- Phase 4 — #1742 (delete hand-rolled samplers + the h5 path)

## §9 Testing strategy (cross-cutting)

The single most important rule, enforced per phase:

> **Parity is at the function level, not the stream level.** Assert that
> `prepare_batch` produces identical output for identical raw input. Do **not**
> assert the new dataloader yields the same *batches* as the old one — it won't,
> by design, because true row-level shuffling changes batch membership. A
> stream-equality test will fail on a correct migration and send you chasing a
> non-bug.

Test types used across phases (detail in each phase issue):

1. **Unit / pinning** — `prepare_batch` exact-output under a fixed seed
   (Phase 1).
2. **Read-equivalence** — projected `__getitems__` returns arrays equal to the
   old per-column reads for the *same indices* (Phase 2); this one *can* be
   exact.
3. **Property-based** — epoch coverage (every row exactly once, no dup/drop),
   shape/dtype invariants, projection minimality (Phase 2).
4. **Multi-process / DDP** — `ddp_sim` 2-device run asserts disjoint per-rank
   index sets whose union is the full set; multi-worker output == single-worker
   output (Phase 2).
5. **Integration** — training reads from node-local path after `prepare_data`;
   per-node staging; idempotent no-overwrite (Phase 3).
6. **Regression + benchmark** — full suite green after deletions; batches/sec
   and dataloader-wait before/after (Phase 4, harness introduced in Phase 2).

Determinism: thread an explicit `torch.Generator` through noise + OT so pinning
tests are stable and DDP runs are reproducible across ranks. Flaky "exact"
assertions almost always trace to ungoverned RNG here.

## §10 References

**External**

- Lance PyTorch integration (LanceDataset = IterableDataset; SafeLanceDataset; `cache`, `batch_readahead`): https://lance.org/integrations/pytorch/
- Lance torch dataset source (`SafeLanceDataset.__getitems__` → `to_pylist`): https://github.com/lance-format/lance/blob/main/python/python/lance/torch/data.py
- Lance samplers (`ShardedFragmentSampler`, `ShardedBatchSampler` reservoir shuffle): https://github.com/lance-format/lance/blob/main/python/python/lance/sampler.py
- pylance on PyPI (pinned 7.0.0): https://pypi.org/project/pylance/
- PyTorch `torch.utils.data` (map vs iterable; `__getitems__` "for speedup batched samples loading"; `batch_size=None`): https://docs.pytorch.org/docs/2.12/data.html
- PyTorch `_MapDatasetFetcher` (calls `__getitems__` when present): https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/fetch.py
- PyTorch issue documenting `__getitems__`: https://github.com/pytorch/pytorch/issues/107218
- Lightning `Trainer` distributed-sampler behavior (`use_distributed_sampler` / `replace_sampler_ddp`): https://lightning.ai/docs/pytorch/stable/common/trainer.html
- Lightning custom-sampler replacement internals (SequentialSampler/RandomSampler special-case + "Replacing custom sampler" warning): https://github.com/Lightning-AI/pytorch-lightning/issues/5271
- Lightning `prepare_data` / `prepare_data_per_node`: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
- Cloudflare R2 zero egress (pre-staging is free): https://www.cloudflare.com/developer-platform/products/r2/

**Internal (repo, grounded)**

- `src/synth_setter/data/surge_datamodule.py` — `VSTDataset.__getitem__` per-column reads (L236–256); samplers (L275–390); `train_dataloader`/`batch_size=None` (L527–543); `prepare_data` (L462–472).
- `src/synth_setter/data/lance_datamodule.py` — `LanceColumn.__getitem__` (L48–74); `LanceShardFile.live_dataset` getpid reopen (L108–122).
- `src/synth_setter/pipeline/r2_io.py` — `r2_storage_options` (L188); `download_dir_no_overwrite`.
- `src/synth_setter/pipeline/data/lance_shard.py` — `LANCE_DATA_STORAGE_VERSION = "2.2"` constant (L21), applied as the `data_storage_version` write kwarg (L111/L142).
- `docs/design/lance-dataset-api-migration.md` — storage/write-API sibling; "training dataloader keeps rclone local-first" decision.
- `configs/trainer/ddp.yaml` — `strategy: ddp`, `devices: 4`; no `use_distributed_sampler` override.
