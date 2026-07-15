"""Jqueue-backed distributed work queue for dataset shard generation.

Machines claim shard IDs dynamically from a single queue-state JSON object in
R2 (jqueue ``DirectQueue`` — one compare-and-set write per operation) instead
of owning a static rank/world slice. The queue only distributes work; shard
completion truth stays with the per-shard R2 skip-probe. Recovery is
relaunch-driven: ``populate`` sweeps abandoned claims older than
``stale_claim_timeout`` back to queued and restores requested IDs missing from
the active queue. Workers heartbeat while rendering so live claims stay fresh.

Individual S3 requests use botocore's bounded standard retry policy. Errors
that escape that policy fail the queue operation; dequeue itself is never
replayed as a whole because a lost successful response is ambiguous.

Typical usage::

    queue = ShardQueue.for_location(config, shard_queue_location(spec.r2))
    queue.populate(shard.shard_id for shard in spec.shards)  # operator, once
    while (claimed := queue.claim(shard_count=len(spec.shards))) is not None:
        with queue.maintain_heartbeat(claimed.job_id):
            render(claimed.shard_id)
        queue.ack(claimed.job_id)
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import timedelta
from types import TracebackType
from typing import Final, Protocol, cast

import structlog
from jqueue import CASConflictError, DirectQueue, Job, JobNotFoundError, StorageError
from jqueue.adapters.storage.s3 import S3Storage
from jqueue.core import codec
from jqueue.ports.storage import ObjectStoragePort
from pydantic import BaseModel, ConfigDict, ValidationError

from synth_setter.pipeline.r2_io import to_s3_uri
from synth_setter.pipeline.schemas.object_storage import ObjectLocation, StorageConfig
from synth_setter.pipeline.schemas.r2_location import R2Location

_logger = structlog.get_logger(__name__)

SHARD_QUEUE_ENTRYPOINT: Final = "render-shard"

STALE_CLAIM_TIMEOUT: Final = timedelta(hours=2)
HEARTBEAT_INTERVAL: Final = timedelta(minutes=5)

# Keep bulk-population CAS contention handling aligned with DirectQueue.
_POPULATE_CAS_RETRIES: Final = 10
_POPULATE_CAS_BASE_DELAY_S: Final = 0.01
_S3_MAX_ATTEMPTS: Final = 3


class _PutObjectClient(Protocol):
    async def put_object(self, **kwargs: str | bytes) -> Mapping[str, object]: ...


class _ClientContext(Protocol):
    async def __aenter__(self) -> _PutObjectClient: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _S3Session(Protocol):
    def client(self, service_name: str, **kwargs: str | None) -> _ClientContext: ...


def _s3_error_code(exc: Exception) -> str | None:
    """Return the botocore-style error code carried by an exception.

    :param exc: Exception raised by an S3 request.
    :returns: Error code, or ``None`` when the exception has no S3 response.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return code if isinstance(code, str) else None


def _s3_status_code(exc: Exception) -> int | None:
    """Return the HTTP status code carried by an exception's S3 response.

    :param exc: Exception raised by an S3 request.
    :returns: HTTP status code, or ``None`` when the exception has no S3 response.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    metadata = response.get("ResponseMetadata")
    if not isinstance(metadata, dict):
        return None
    status = metadata.get("HTTPStatusCode")
    return status if isinstance(status, int) else None


@dataclasses.dataclass
class _ConditionalCreateS3Storage(S3Storage):
    """S3 adapter that makes first-write creation part of the CAS contract."""

    async def write(self, content: bytes, if_match: str | None = None) -> str:
        """Write queue state only if its expected presence and etag still hold.

        :param content: Encoded queue state.
        :param if_match: Expected etag, or ``None`` when the object must not exist.
        :returns: Etag of the stored object.
        :raises CASConflictError: When a conditional S3 write loses a race.
        :raises StorageError: When the S3 request otherwise fails.
        """
        session = cast(_S3Session, self._get_session())
        try:
            async with session.client(
                "s3",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            ) as s3:
                put_kwargs: dict[str, str | bytes] = {
                    "Bucket": self.bucket,
                    "Key": self.key,
                    "Body": content,
                    "ContentType": "application/json",
                }
                if if_match is None:
                    put_kwargs["IfNoneMatch"] = "*"
                else:
                    put_kwargs["IfMatch"] = if_match
                response = await s3.put_object(**put_kwargs)
                return str(response["ETag"])
        except (CASConflictError, StorageError):
            raise
        except Exception as exc:
            # Error-code strings vary by provider; a 412/409 status on a
            # conditional put is the contractual precondition-failure signal.
            if _s3_error_code(exc) in {
                "PreconditionFailed",
                "412",
                "ConditionalRequestConflict",
            } or _s3_status_code(exc) in {409, 412}:
                raise CASConflictError("S3 conditional write failed") from exc
            raise StorageError("S3 write failed", exc) from exc


def shard_queue_location(r2: R2Location) -> ObjectLocation:
    """Return the queue-state object location for a dataset run.

    The state lives under the run prefix so queue and shards share one home
    in R2 (the pipeline's single source of truth).

    :param r2: Dataset run location (``prefix`` ends with ``/`` by contract).
    :returns: Bucket/key of the run's queue-state JSON object.
    """
    return ObjectLocation.from_uri(to_s3_uri(r2.shard_queue_uri()))


class _ShardJobPayload(BaseModel):
    """Queue-job payload — trust boundary for job JSON read back from R2.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: shard_id

        Logical shard ID to render (index into ``DatasetSpec.shards``).
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int


@dataclasses.dataclass(frozen=True)
class ClaimedShard:
    """One claimed queue job: the shard to render plus its queue job id.

    .. attribute :: job_id

        Queue job id, passed back to ``ack``.

    .. attribute :: shard_id

        Logical shard ID to render (index into ``DatasetSpec.shards``).
    """

    job_id: str
    shard_id: int


@dataclasses.dataclass(frozen=True)
class ShardQueue:
    """Synchronous facade over a jqueue ``DirectQueue`` of shard-render jobs.

    Each method drives one queue operation to completion on a private event
    loop; the ~1 op/sec DirectQueue write rate is negligible next to
    minutes-long shard renders.

    .. attribute :: storage

        jqueue storage adapter holding the queue-state JSON object.
    """

    storage: ObjectStoragePort

    @classmethod
    def for_location(cls, config: StorageConfig, location: ObjectLocation) -> ShardQueue:
        """Build a queue over S3-compatible storage using centralized credentials.

        :param config: Env-free storage config (credentials, endpoint, region).
        :param location: Queue-state object location, from :func:`shard_queue_location`.
        :returns: Queue backed by jqueue's S3 adapter against ``config``'s endpoint.
        """
        # Deferred import — aioboto3 pulls the full botocore stack.
        import aioboto3
        from aiobotocore.session import get_session

        botocore_session = get_session()
        botocore_session.set_config_variable("retry_mode", "standard")
        botocore_session.set_config_variable("max_attempts", _S3_MAX_ATTEMPTS)
        session = aioboto3.Session(
            aws_access_key_id=config.access_key_id.get_secret_value(),
            aws_secret_access_key=config.secret_access_key.get_secret_value(),
            botocore_session=botocore_session,
        )
        storage = _ConditionalCreateS3Storage(
            bucket=location.bucket,
            key=location.key,
            session=session,
            region_name=config.region,
            endpoint_url=config.endpoint_url,
        )
        return cls(storage=storage)

    def _direct(self) -> DirectQueue:
        """Return a stateless jqueue queue over this facade's storage.

        :returns: ``DirectQueue`` with jqueue's built-in CAS-conflict retries.
        """
        return DirectQueue(self.storage)

    def populate(
        self,
        shard_ids: Iterable[int],
        *,
        stale_claim_timeout: timedelta = STALE_CLAIM_TIMEOUT,
    ) -> int:
        """Ensure one active render job exists per requested logical shard ID.

        The relaunch entrypoint: crashed-worker claims older than
        ``stale_claim_timeout`` are first swept back to queued, then the
        queued and claimed IDs are preserved while missing IDs are restored.
        All new jobs land in a single compare-and-set write —
        per-job enqueues would cost one storage round-trip each and re-upload
        the growing state every time (O(N²) bytes over N shards; the in-memory
        per-job tuple rebuild is a one-time launch cost). A racing populate
        (e.g. a relaunched operator) is retried against fresh state and
        reconciles against the winner's state on every retry. First creation is
        conditional too, so independent writers cannot clobber one another.

        Persistent same-key contention past the ``_POPULATE_CAS_RETRIES``
        budget propagates as ``CASConflictError``.

        :param shard_ids: Shard IDs to enqueue, claimed later in this order.
        :param stale_claim_timeout: Age past which an in-progress claim is
            treated as a crashed worker's leftover and re-queued.
        :returns: Number of missing logical shard IDs enqueued.
        """
        ids = list(dict.fromkeys(shard_ids))

        async def _run() -> int:
            requeued = await self._direct().requeue_stale(stale_claim_timeout)
            if requeued:
                _logger.info("requeued stale shard claims", count=requeued)
            attempt = 0
            while True:
                content, etag = await self.storage.read()
                state = codec.decode(content)
                active_ids: set[int] = set()
                for job in state.jobs:
                    if job.entrypoint != SHARD_QUEUE_ENTRYPOINT:
                        continue
                    try:
                        active_ids.add(_ShardJobPayload.model_validate_json(job.payload).shard_id)
                    except ValidationError:
                        continue
                missing_ids = [shard_id for shard_id in ids if shard_id not in active_ids]
                if not missing_ids:
                    return 0
                for shard_id in missing_ids:
                    payload = _ShardJobPayload(shard_id=shard_id)
                    job = Job.new(SHARD_QUEUE_ENTRYPOINT, payload.model_dump_json().encode())
                    state = state.with_job_added(job)
                try:
                    await self.storage.write(codec.encode(state), if_match=etag)
                    return len(missing_ids)
                except CASConflictError:
                    attempt += 1
                    if attempt >= _POPULATE_CAS_RETRIES:
                        raise
                    await asyncio.sleep(_POPULATE_CAS_BASE_DELAY_S * attempt)

        return asyncio.run(_run())

    def claim(self, *, shard_count: int) -> ClaimedShard | None:
        """Claim and validate the next queued shard before returning it.

        :param shard_count: Number of logical shards in the current dataset spec.
        :returns: The claimed job, or ``None`` when the queue is drained.
        :raises ValidationError: The job payload is malformed.
        :raises ValueError: The shard ID is outside the current dataset spec.
            Poison jobs are retired before either exception escapes.
        """
        jobs = asyncio.run(self._direct().dequeue(SHARD_QUEUE_ENTRYPOINT))
        if not jobs:
            return None
        job = jobs[0]
        try:
            payload = _ShardJobPayload.model_validate_json(job.payload)
            if not 0 <= payload.shard_id < shard_count:
                raise ValueError(f"shard_id {payload.shard_id} is outside [0, {shard_count})")
        except (ValidationError, ValueError):
            try:
                self.ack(job.id)
            except Exception:  # noqa: BLE001 — never mask the payload error
                # The stale-claim sweep recovers the unretired poison job later.
                _logger.exception("failed to retire poison job", job_id=job.id)
            raise
        return ClaimedShard(job_id=job.id, shard_id=payload.shard_id)

    def heartbeat(self, job_id: str) -> bool:
        """Refresh an in-progress claim's liveness timestamp.

        :param job_id: Queue job id from :class:`ClaimedShard`.
        :returns: ``False`` when a peer already removed the job; otherwise ``True``.
        """
        try:
            asyncio.run(self._direct().heartbeat(job_id))
        except JobNotFoundError:
            _logger.info("job already acked by a peer", job_id=job_id)
            return False
        return True

    @contextmanager
    def maintain_heartbeat(
        self,
        job_id: str,
        *,
        interval: timedelta = HEARTBEAT_INTERVAL,
    ) -> Iterator[None]:
        """Heartbeat a claim in a background thread for a blocking render.

        :param job_id: Queue job id from :class:`ClaimedShard`.
        :param interval: Delay between heartbeat requests after the initial refresh.
        :yields: Control while the heartbeat worker is active.
        :ytype: None
        :raises ValueError: When ``interval`` is not positive.
        """
        delay = interval.total_seconds()
        if delay <= 0:
            raise ValueError("heartbeat interval must be positive")
        stop = threading.Event()

        def _run() -> None:
            while not stop.is_set():
                try:
                    if not self.heartbeat(job_id):
                        return
                except Exception:  # noqa: BLE001 — renderer can still finish idempotently
                    # A transient storage error must not end heartbeating for a
                    # minutes-long render; retry at the next interval.
                    _logger.exception("shard heartbeat failed", job_id=job_id)
                stop.wait(delay)

        worker = threading.Thread(target=_run, name=f"shard-heartbeat-{job_id}", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join()

    def ack(self, job_id: str) -> None:
        """Remove a claimed job from the queue — completed or retired.

        Idempotent: a job already removed is a benign no-op. After a
        stale-claim sweep, a slow worker and the peer that reclaimed its job
        both ack the same id; whoever lands second must not fail the run.

        :param job_id: Queue job id from :class:`ClaimedShard`.
        """
        try:
            asyncio.run(self._direct().ack(job_id))
        except JobNotFoundError:
            _logger.info("job already acked by a peer", job_id=job_id)
