"""Behavior tests for the jqueue-backed distributed shard work queue.

NOTE: multi-machine contention safety (no lost or double-claimed jobs under
concurrent claims) is delegated to jqueue's CAS-retrying ``DirectQueue`` and is
deliberately not simulated here — each facade call runs its own event loop, so
in-process interleaving cannot reproduce real cross-machine races. The one
race this module owns (populate vs. a concurrent populate) is covered below
via a conflict-injecting storage wrapper.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from typing import cast

import pytest
from aioboto3 import Session
from botocore.exceptions import ClientError
from jqueue import CASConflictError, DirectQueue, InMemoryStorage, StorageError
from jqueue.adapters.storage.s3 import S3Storage
from pydantic import SecretStr, ValidationError

from synth_setter.pipeline.schemas.object_storage import StorageConfig
from synth_setter.pipeline.schemas.r2_location import R2Location
from synth_setter.pipeline.shard_queue import (
    _POPULATE_CAS_RETRIES,
    SHARD_QUEUE_ENTRYPOINT,
    ClaimedShard,
    ShardQueue,
    _ConditionalCreateS3Storage,
    shard_queue_location,
)


def _memory_queue() -> ShardQueue:
    """Return a ShardQueue over jqueue's in-memory storage (real queue, no network).

    :returns: Queue with real jqueue semantics and no external dependencies.
    """
    return ShardQueue(storage=InMemoryStorage())


def _claim_or_fail(queue: ShardQueue) -> ClaimedShard:
    """Claim one job, failing the test if the queue is unexpectedly empty.

    :param queue: Queue under test.
    :returns: The claimed job.
    """
    claimed = queue.claim(shard_count=10_001)
    assert claimed is not None, "expected a claimable job"
    return claimed


def test_claim_after_populate_returns_shards_in_order_then_none() -> None:
    """Claims come back in enqueue order and the drained queue yields None."""
    queue = _memory_queue()

    queue.populate([3, 1, 2])

    assert [_claim_or_fail(queue).shard_id for _ in range(3)] == [3, 1, 2]
    assert queue.claim(shard_count=10_001) is None


def test_populate_empty_shard_ids_enqueues_nothing() -> None:
    """An empty shard list is a no-op distinct from the non-empty-queue no-op."""
    queue = _memory_queue()

    assert queue.populate([]) == 0
    assert queue.claim(shard_count=10_001) is None


def test_populate_nonempty_queue_enqueues_only_missing_jobs() -> None:
    """A relaunch preserves active jobs and adds only missing logical shard IDs."""
    queue = _memory_queue()
    queue.populate([0, 1])

    assert queue.populate([0, 1, 2]) == 1

    assert _claim_or_fail(queue).shard_id == 0
    assert _claim_or_fail(queue).shard_id == 1
    assert _claim_or_fail(queue).shard_id == 2
    assert queue.claim(shard_count=10_001) is None


def test_populate_queue_with_only_in_progress_jobs_enqueues_nothing() -> None:
    """Claimed-but-unacked jobs still count as populated (no duplicate enqueue)."""
    queue = _memory_queue()
    queue.populate([7])
    queue.claim(shard_count=10_001)

    assert queue.populate([7]) == 0
    assert queue.claim(shard_count=10_001) is None


def test_populate_reconciles_retired_and_completed_ids_without_duplicates() -> None:
    """Missing IDs return while queued and in-progress jobs remain unique."""
    queue = _memory_queue()
    queue.populate([0, 1, 2, 3])
    completed = _claim_or_fail(queue)
    retired = _claim_or_fail(queue)
    queue.ack(completed.job_id)
    queue.ack(retired.job_id)
    in_progress = _claim_or_fail(queue)

    assert queue.populate([0, 1, 2, 3]) == 2

    assert in_progress.shard_id == 2
    assert [_claim_or_fail(queue).shard_id for _ in range(3)] == [3, 0, 1]
    assert queue.claim(shard_count=10_001) is None


def test_populate_requeues_stale_claim_for_another_worker() -> None:
    """A crashed worker's aged claim becomes claimable again on relaunch populate."""
    queue = _memory_queue()
    queue.populate([5])
    queue.claim(shard_count=10_001)

    # timeout=0 ages out the fresh claim, standing in for a crashed worker's
    # hours-old one.
    assert queue.populate([5], stale_claim_timeout=timedelta(0)) == 0

    assert _claim_or_fail(queue).shard_id == 5


def test_populate_losing_cas_race_reconciles_winner_jobs() -> None:
    """A populate that loses a CAS write reconciles against the winner's jobs."""

    class _RaceOnFirstWrite:
        """Storage wrapper that lets a rival populate win the first CAS write."""

        def __init__(self, inner: InMemoryStorage) -> None:
            """Wrap ``inner`` and arm the one-shot rival write.

            :param inner: Real in-memory storage both writers share.
            """
            self._inner = inner
            self._raced = False

        async def read(self) -> tuple[bytes, str | None]:
            """Delegate to the wrapped storage.

            :returns: The wrapped storage's ``(content, etag)``.
            """
            return await self._inner.read()

        async def write(self, content: bytes, if_match: str | None = None) -> str:
            """Inject a rival's committed write before the first delegated write.

            :param content: Encoded queue state to store.
            :param if_match: CAS etag; stale after the injected rival write.
            :returns: The wrapped storage's new etag.
            """
            if not self._raced:
                self._raced = True
                rival = DirectQueue(self._inner)
                await rival.enqueue(SHARD_QUEUE_ENTRYPOINT, b'{"shard_id": 9}')
            return await self._inner.write(content, if_match=if_match)

    queue = ShardQueue(storage=_RaceOnFirstWrite(InMemoryStorage()))

    assert queue.populate([1, 2]) == 2

    assert _claim_or_fail(queue).shard_id == 9
    assert _claim_or_fail(queue).shard_id == 1
    assert _claim_or_fail(queue).shard_id == 2
    assert queue.claim(shard_count=10_001) is None


def test_ack_removes_job_so_it_is_never_reclaimed() -> None:
    """An acked job leaves the queue permanently."""
    queue = _memory_queue()
    queue.populate([5])
    claimed = _claim_or_fail(queue)

    queue.ack(claimed.job_id)

    assert queue.claim(shard_count=10_001) is None
    assert queue.populate([5]) == 1  # queue is truly empty after the ack


def test_ack_of_already_removed_job_is_benign() -> None:
    """A second ack of the same job (stale-claim reclaim race) never fails the run."""
    queue = _memory_queue()
    queue.populate([5])
    claimed = _claim_or_fail(queue)
    queue.ack(claimed.job_id)

    # Slow worker's late ack, after a peer already reclaimed and acked it.
    queue.ack(claimed.job_id)

    assert queue.claim(shard_count=10_001) is None


def test_claim_malformed_payload_retires_poison_job_and_raises() -> None:
    """A corrupt payload fails loudly and is retired so peers cannot re-claim it."""
    storage = InMemoryStorage()
    asyncio.run(DirectQueue(storage).enqueue(SHARD_QUEUE_ENTRYPOINT, b'{"bogus": 1}'))
    queue = ShardQueue(storage=storage)

    with pytest.raises(ValidationError):
        queue.claim(shard_count=10_001)

    assert queue.claim(shard_count=10_001) is None
    assert queue.populate([1]) == 1  # empty queue ⇒ the poison job was retired


def test_populate_and_claim_ignore_foreign_entrypoint_jobs() -> None:
    """Jobs from other queue tenants are neither counted, claimed, nor disturbed."""
    storage = InMemoryStorage()
    asyncio.run(DirectQueue(storage).enqueue("other-entrypoint", b'{"shard_id": 0}'))
    queue = ShardQueue(storage=storage)

    assert queue.populate([0]) == 1  # the foreign shard_id 0 does not count as active
    claimed = _claim_or_fail(queue)
    assert claimed.shard_id == 0
    queue.ack(claimed.job_id)
    assert queue.claim(shard_count=10_001) is None  # foreign job is never dequeued


def test_heartbeat_live_claim_true_then_false_after_peer_ack() -> None:
    """A heartbeat refreshes a live claim and reports removal after a peer's ack."""
    queue = _memory_queue()
    queue.populate([5])
    claimed = _claim_or_fail(queue)

    assert queue.heartbeat(claimed.job_id) is True

    queue.ack(claimed.job_id)
    assert queue.heartbeat(claimed.job_id) is False


def test_maintain_heartbeat_nonpositive_interval_raises_value_error() -> None:
    """A non-positive interval is a caller bug rejected before any thread starts."""
    queue = _memory_queue()

    with pytest.raises(ValueError, match="heartbeat interval must be positive"):
        with queue.maintain_heartbeat("job-1", interval=timedelta(0)):
            pass


def test_maintain_heartbeat_stops_after_peer_removed_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker thread exits once heartbeat reports the job is gone.

    :param monkeypatch: Replaces the network heartbeat with a job-gone probe.
    """
    queue = _memory_queue()
    called = threading.Event()

    def _heartbeat(self: ShardQueue, job_id: str) -> bool:
        called.set()
        return False

    monkeypatch.setattr(ShardQueue, "heartbeat", _heartbeat)

    with queue.maintain_heartbeat("job-1", interval=timedelta(milliseconds=1)):
        assert called.wait(timeout=1)
    # Context exit joins the worker; reaching here proves it terminated.


def test_claim_malformed_payload_retires_only_the_poison_job() -> None:
    """Retiring a poison job must not consume any healthy queued job."""
    storage = InMemoryStorage()
    asyncio.run(DirectQueue(storage).enqueue(SHARD_QUEUE_ENTRYPOINT, b'{"bogus": 1}'))
    queue = ShardQueue(storage=storage)
    queue.populate([7])  # healthy job queued behind the poison one

    with pytest.raises(ValidationError):
        queue.claim(shard_count=10_001)

    healthy = queue.claim(shard_count=10_001)
    assert healthy is not None and healthy.shard_id == 7
    queue.ack(healthy.job_id)
    assert queue.claim(shard_count=10_001) is None


@pytest.mark.parametrize("shard_id", [-1, 3, 10_000])
def test_claim_out_of_range_shard_retires_poison_job(shard_id: int) -> None:
    """A logical ID outside the current spec is rejected and retired before return.

    :param shard_id: Invalid logical shard ID encoded in the queue job.
    """
    storage = InMemoryStorage()
    payload = f'{{"shard_id": {shard_id}}}'.encode()
    asyncio.run(DirectQueue(storage).enqueue(SHARD_QUEUE_ENTRYPOINT, payload))
    queue = ShardQueue(storage=storage)

    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        queue.claim(shard_count=3)

    assert queue.claim(shard_count=3) is None


def test_maintain_heartbeat_refreshes_claim_until_context_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher context starts promptly and stops its heartbeat worker.

    :param monkeypatch: Replaces the network heartbeat with a synchronization probe.
    """
    queue = _memory_queue()
    called = threading.Event()
    calls: list[str] = []

    def _heartbeat(self: ShardQueue, job_id: str) -> bool:
        calls.append(job_id)
        called.set()
        return True

    monkeypatch.setattr(ShardQueue, "heartbeat", _heartbeat)

    with queue.maintain_heartbeat("job-1", interval=timedelta(hours=1)):
        assert called.wait(timeout=1)

    assert calls == ["job-1"]


def test_maintain_heartbeat_transient_failure_retries_at_next_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed heartbeat must not end heartbeating for the rest of a render.

    :param monkeypatch: Replaces the network heartbeat with a fail-then-recover probe.
    """
    from jqueue import StorageError

    queue = _memory_queue()
    recovered = threading.Event()
    calls: list[str] = []

    def _heartbeat(self: ShardQueue, job_id: str) -> bool:
        calls.append(job_id)
        if len(calls) == 1:
            raise StorageError("injected transient failure", RuntimeError("boom"))
        recovered.set()
        return True

    monkeypatch.setattr(ShardQueue, "heartbeat", _heartbeat)

    with queue.maintain_heartbeat("job-1", interval=timedelta(milliseconds=10)):
        assert recovered.wait(timeout=5), "heartbeat worker died after one transient failure"

    assert len(calls) >= 2


def test_claim_poison_retire_failure_still_raises_validation_error() -> None:
    """A failed poison-job retirement never masks the payload's ValidationError."""
    from jqueue import StorageError

    class _FailWritesAfterFirst:
        """Storage wrapper that fails every write after the dequeue's."""

        def __init__(self, inner: InMemoryStorage) -> None:
            """Wrap ``inner``.

            :param inner: Real in-memory storage holding the poison job.
            """
            self._inner = inner
            self._writes = 0

        async def read(self) -> tuple[bytes, str | None]:
            """Delegate to the wrapped storage.

            :returns: The wrapped storage's ``(content, etag)``.
            """
            return await self._inner.read()

        async def write(self, content: bytes, if_match: str | None = None) -> str:
            """Allow the claim's dequeue write, then fail the retiring ack's.

            :param content: Encoded queue state to store.
            :param if_match: CAS etag forwarded to the wrapped storage.
            :returns: The wrapped storage's new etag (first write only).
            :raises StorageError: On every write after the first.
            """
            self._writes += 1
            if self._writes > 1:
                raise StorageError("injected ack failure", RuntimeError("boom"))
            return await self._inner.write(content, if_match=if_match)

    inner = InMemoryStorage()
    asyncio.run(DirectQueue(inner).enqueue(SHARD_QUEUE_ENTRYPOINT, b'{"bogus": 1}'))
    queue = ShardQueue(storage=_FailWritesAfterFirst(inner))

    with pytest.raises(ValidationError):
        queue.claim(shard_count=10_001)


def test_populate_persistent_cas_contention_exhausts_retries_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contention on every CAS write propagates CASConflictError after the budget.

    :param monkeypatch: Zeroes the back-off delay so the exhaustion run is instant.
    """
    from jqueue import CASConflictError

    monkeypatch.setattr("synth_setter.pipeline.shard_queue._POPULATE_CAS_BASE_DELAY_S", 0)

    class _ConflictAfterFirstWrite:
        """Storage wrapper whose every CAS write after the first loses to a rival's.

        The first write (``populate``'s leading ``requeue_stale`` sweep) must
        succeed so the exhaustion exercises populate's own retry loop, not
        jqueue's internal one.
        """

        def __init__(self, inner: InMemoryStorage) -> None:
            """Wrap ``inner``.

            :param inner: Real in-memory storage the rival writes to.
            """
            self._inner = inner
            self._writes = 0

        async def read(self) -> tuple[bytes, str | None]:
            """Delegate to the wrapped storage.

            :returns: The wrapped storage's ``(content, etag)``.
            """
            return await self._inner.read()

        async def write(self, content: bytes, if_match: str | None = None) -> str:
            """Invalidate the caller's etag with a rival write, then delegate.

            Injects a rival write before delegating, so every re-read still sees an empty queue and
            populate keeps retrying.

            :param content: Encoded queue state to store.
            :param if_match: CAS etag; stale after the injected rival write.
            :returns: The wrapped storage's new etag (first write only).
            """
            self._writes += 1
            if self._writes == 1:
                return await self._inner.write(content, if_match=if_match)
            rival = DirectQueue(self._inner)
            job = await rival.enqueue(SHARD_QUEUE_ENTRYPOINT, b'{"shard_id": 0}')
            await rival.ack(job.id)
            return await self._inner.write(content, if_match=if_match)

    storage = _ConflictAfterFirstWrite(InMemoryStorage())
    queue = ShardQueue(storage=storage)

    with pytest.raises(CASConflictError):
        queue.populate([1, 2])

    # requeue_stale's write plus every populate attempt in the retry budget.
    assert storage._writes == 1 + _POPULATE_CAS_RETRIES  # noqa: SLF001


def test_shard_queue_location_derives_key_under_run_metadata_prefix() -> None:
    """The queue state object lives under the run's metadata/ prefix."""
    r2 = R2Location(bucket="intermediate-data", prefix="data/ci-smoke/run-1/")

    location = shard_queue_location(r2)

    assert location.bucket == "intermediate-data"
    assert location.key == "data/ci-smoke/run-1/metadata/shard-queue.json"


def test_for_location_builds_s3_storage_against_configured_endpoint() -> None:
    """The S3 adapter targets the centralized config's endpoint and credentials."""
    config = StorageConfig(
        access_key_id=SecretStr("ak"),
        secret_access_key=SecretStr("sk"),
        endpoint_url="https://accountid.r2.cloudflarestorage.com",
        region="auto",
    )
    r2 = R2Location(bucket="intermediate-data", prefix="data/ci-smoke/run-1/")

    queue = ShardQueue.for_location(config, shard_queue_location(r2))

    storage = queue.storage
    assert isinstance(storage, S3Storage)
    assert storage.bucket == "intermediate-data"
    assert storage.key == "data/ci-smoke/run-1/metadata/shard-queue.json"
    assert storage.endpoint_url == "https://accountid.r2.cloudflarestorage.com"
    assert storage.region_name == "auto"
    assert storage.session is not None
    assert storage.session._session.get_config_variable("retry_mode") == "standard"
    assert storage.session._session.get_config_variable("max_attempts") == 3


def test_s3_storage_create_is_atomic_across_independent_writers() -> None:
    """A second writer cannot overwrite an object both writers observed absent."""

    class _Client:
        """Minimal async S3 client enforcing conditional put semantics."""

        def __init__(self, objects: dict[str, bytes]) -> None:
            """Share the backing object map across independent sessions.

            :param objects: Stored object bodies keyed by S3 key.
            """
            self._objects = objects

        async def __aenter__(self) -> _Client:
            """Return this client from the async context.

            :returns: The entered client.
            """
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            """Leave the client context.

            :param *exc_info: Exception context supplied by ``async with``.
            """

        async def put_object(self, **kwargs: object) -> dict[str, str]:
            """Store a body unless the requested create precondition fails.

            :param **kwargs: S3 ``PutObject`` request fields.
            :returns: Synthetic response etag.
            :raises ClientError: When ``IfNoneMatch`` rejects an existing key.
            """
            key = str(kwargs["Key"])
            if kwargs.get("IfNoneMatch") == "*" and key in self._objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed", "Message": "exists"}},
                    "PutObject",
                )
            body = kwargs["Body"]
            assert isinstance(body, bytes)
            self._objects[key] = body
            return {"ETag": '"etag"'}

    class _Session:
        """Minimal session returning a client over shared state."""

        def __init__(self, objects: dict[str, bytes]) -> None:
            """Retain the shared object map.

            :param objects: Stored object bodies keyed by S3 key.
            """
            self._objects = objects

        def client(
            self,
            service_name: str,
            *,
            region_name: str | None = None,
            endpoint_url: str | None = None,
        ) -> _Client:
            """Return a new async client.

            :param service_name: AWS service name.
            :param region_name: AWS region override.
            :param endpoint_url: S3-compatible endpoint override.
            :returns: Client sharing this session's object map.
            """
            return _Client(self._objects)

    objects: dict[str, bytes] = {}
    session = cast(Session, _Session(objects))
    storage_a = _ConditionalCreateS3Storage(bucket="bucket", key="queue.json", session=session)
    storage_b = _ConditionalCreateS3Storage(bucket="bucket", key="queue.json", session=session)
    queue_a = ShardQueue(storage=storage_a)
    queue_b = ShardQueue(storage=storage_b)

    etag = asyncio.run(queue_a.storage.write(b"winner", if_match=None))
    with pytest.raises(CASConflictError):
        asyncio.run(queue_b.storage.write(b"loser", if_match=None))

    assert objects["queue.json"] == b"winner"
    # A CAS update against the held etag is the non-create write path.
    asyncio.run(queue_a.storage.write(b"updated", if_match=etag))
    assert objects["queue.json"] == b"updated"


def test_conditional_write_bare_412_status_maps_to_cas_conflict() -> None:
    """A 412 rejection with an unrecognized error code still surfaces as a CAS loss.

    R2/S3 conditional-write rejections are only contractually a 412 status; the
    error-code string varies by provider. An unrecognized code must not turn a
    retryable CAS race into an unretried ``StorageError``.
    """

    class _Reject412Client:
        """Client whose every put fails with a 412 and a provider-specific code."""

        async def __aenter__(self) -> _Reject412Client:
            """Return this client from the async context.

            :returns: The entered client.
            """
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            """Leave the client context.

            :param *exc_info: Exception context supplied by ``async with``.
            """

        async def put_object(self, **kwargs: object) -> dict[str, str]:
            """Reject every conditional put with a bare-412 provider error.

            :param **kwargs: S3 ``PutObject`` request fields.
            :returns: Never returns.
            :raises ClientError: Always, with a code outside the known set.
            """
            raise ClientError(
                {
                    "Error": {"Code": "ProviderSpecificConflict", "Message": "exists"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )

    class _Reject412Session:
        """Minimal session returning the always-rejecting client."""

        def client(
            self,
            service_name: str,
            *,
            region_name: str | None = None,
            endpoint_url: str | None = None,
        ) -> _Reject412Client:
            """Return a new async client.

            :param service_name: AWS service name.
            :param region_name: AWS region override, ignored.
            :param endpoint_url: S3-compatible endpoint override, ignored.
            :returns: The rejecting client.
            """
            return _Reject412Client()

    storage = _ConditionalCreateS3Storage(
        bucket="bucket", key="queue.json", session=cast(Session, _Reject412Session())
    )

    with pytest.raises(CASConflictError):
        asyncio.run(storage.write(b"state", if_match=None))


@pytest.mark.parametrize(
    "injected",
    [
        RuntimeError("no S3 response attached"),
        StorageError("adapter-level failure", RuntimeError("boom")),
        ClientError({}, "PutObject"),
    ],
    ids=[
        "non-conditional-error-wraps",
        "storage-error-passes-through",
        "response-without-error-or-metadata-wraps",
    ],
)
def test_conditional_write_non_conditional_failure_surfaces_storage_error(
    injected: Exception,
) -> None:
    """A failure that is not a precondition loss surfaces as StorageError, never CAS.

    :param injected: Exception the fake client raises on every put.
    """

    class _AlwaysRaiseClient:
        """Client whose every put fails with the injected exception."""

        async def __aenter__(self) -> _AlwaysRaiseClient:
            """Return this client from the async context.

            :returns: The entered client.
            """
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            """Leave the client context.

            :param *exc_info: Exception context supplied by ``async with``.
            """

        # DOC503: the raised type is the parametrized ``injected`` exception.
        async def put_object(self, **kwargs: object) -> dict[str, str]:  # noqa: DOC503
            """Raise the injected failure.

            :param **kwargs: S3 ``PutObject`` request fields.
            :returns: Never returns.
            :raises Exception: The injected failure, always.
            """
            raise injected

    class _AlwaysRaiseSession:
        """Minimal session returning the always-raising client."""

        def client(
            self,
            service_name: str,
            *,
            region_name: str | None = None,
            endpoint_url: str | None = None,
        ) -> _AlwaysRaiseClient:
            """Return a new async client.

            :param service_name: AWS service name.
            :param region_name: AWS region override, ignored.
            :param endpoint_url: S3-compatible endpoint override, ignored.
            :returns: The raising client.
            """
            return _AlwaysRaiseClient()

    storage = _ConditionalCreateS3Storage(
        bucket="bucket", key="queue.json", session=cast(Session, _AlwaysRaiseSession())
    )

    with pytest.raises(StorageError):
        asyncio.run(storage.write(b"state", if_match=None))
