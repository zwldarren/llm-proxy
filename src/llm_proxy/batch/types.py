"""Batch-mode vocabulary shared by the service, storage, and provider seam.

Batch is the asynchronous, discounted half of the inference APIs: a client
uploads many requests, gets a job id back, and collects the answers later. The
proxy normalizes every upstream to OpenAI's vocabulary (statuses, counts, result
lines) — see ADR-0058 — so these types are that normalized middle.

The provider side (``providers/capabilities/batch.py``) produces
:class:`BatchSnapshot` and :class:`BatchResultItem` values; the batch service
(``llm_proxy.batch.service``) persists them and renders OpenAI wire objects.
Nothing here knows about files, HTTP, or the database.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from llm_proxy.core.exceptions import AuthenticationFailedError, ValidationError

#: Completion window the proxy requests; all three upstreams document ``24h``
#: as the only accepted value (OpenRouter defaults to it, Gemini ignores it).
DEFAULT_COMPLETION_WINDOW = "24h"

#: The only batch endpoint the proxy accepts. OpenRouter and Gemini both need
#: a per-batch model, and a chat batch is where that restriction is coherent;
#: embeddings and responses batches are out of scope (ADR-0058).
SUPPORTED_BATCH_ENDPOINT = "/v1/chat/completions"

#: Prefix the batch storage scope puts on a console user's owner, i.e.
#: ``user:<name>``. An API key's scope is the bare key name, so the prefix is what
#: tells the two apart.
USER_OWNER_PREFIX = "user:"

#: OpenAI's batch status enum, in the order a job passes through them.
BatchStatus = Literal[
    "validating",
    "in_progress",
    "finalizing",
    "completed",
    "failed",
    "expired",
    "cancelling",
    "cancelled",
]

#: Runtime view of :data:`BatchStatus`, used for validation and membership.
BATCH_STATUSES: frozenset[str] = frozenset(get_args(BatchStatus))

#: Statuses that will not change again: the job is done being polled.
TERMINAL_BATCH_STATUSES: frozenset[str] = frozenset({"completed", "failed", "expired", "cancelled"})

#: Timestamp fields OpenAI's batch object carries, all unix seconds. The proxy
#: stores and returns them under these exact names.
BATCH_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "in_progress_at",
    "finalizing_at",
    "completed_at",
    "failed_at",
    "expired_at",
    "cancelling_at",
    "cancelled_at",
)


@dataclass(frozen=True, slots=True)
class BatchTarget:
    """The configured provider and upstream model one batch is submitted to."""

    provider_name: str
    provider_type: str
    upstream_model: str


@dataclass(frozen=True, slots=True)
class BatchOwner:
    """The storage scope a batch file or job belongs to.

    An API key's name is the scope for ``/v1/*`` traffic; a console user gets the
    ``user:`` prefix so the two namespaces cannot collide. The scope is what the
    database stores, while :attr:`api_key_name` / :attr:`user` are what a usage row
    needs, so both encodings live here rather than in the router and accounting.
    """

    api_key_name: str | None = None
    user: str | None = None

    def __post_init__(self) -> None:
        if bool(self.api_key_name) == bool(self.user):
            raise ValueError("A batch owner must name exactly one API key or console user")

    @classmethod
    def for_api_key(cls, name: str) -> BatchOwner:
        """The owner scope for an API-key-authenticated caller."""
        return cls(api_key_name=name)

    @classmethod
    def for_user(cls, name: str) -> BatchOwner:
        """The owner scope for a console-JWT caller."""
        return cls(user=name)

    @classmethod
    def from_scope(cls, scope: str) -> BatchOwner:
        """Parse a stored scope or raise when it names no authenticated owner."""
        if not scope:
            raise AuthenticationFailedError(
                message="Batch files and jobs require an authenticated caller.",
                code="unauthorized",
                status_code=401,
            )
        if scope.startswith(USER_OWNER_PREFIX):
            user = scope[len(USER_OWNER_PREFIX) :]
            if user:
                return cls.for_user(user)
        else:
            return cls.for_api_key(scope)
        raise ValidationError(
            message="Batch owner scope is malformed.",
            code="invalid_batch_owner",
            status_code=400,
        )

    @property
    def scope(self) -> str:
        """The value stored in ``batch_files.owner`` and ``batch_jobs.owner``."""
        if self.api_key_name is not None:
            return self.api_key_name
        assert self.user is not None
        return f"{USER_OWNER_PREFIX}{self.user}"


@dataclass(frozen=True, slots=True)
class BatchRequestItem:
    """One request inside a batch, as the provider needs to receive it.

    ``body`` is provider-shaped: an OpenAI chat-completions body for the
    verbatim upstreams (OpenAI, OpenRouter), the provider serializer's output
    for a dialect that has to be translated (Gemini). ``custom_id`` rides
    alongside because the client's id is a proxy concept some upstreams carry in
    a field (OpenAI, OpenRouter) and others in per-request metadata or not at
    all (Gemini).
    """

    custom_id: str
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BatchResultItem:
    """One finished request from a batch, in OpenAI result-line terms.

    ``body`` is still provider-shaped when the adapter translates results
    (``BATCH_FORWARDS_CHAT_BODY = False``); the service converts it through
    ``adapter.parse_batch_result`` before writing the output file. ``index`` is
    the position in the submitted batch, which is what Gemini results are keyed
    by — its inline responses carry no client id (the documented metadata echo
    is not reliable).
    """

    index: int
    custom_id: str | None = None
    body: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    status_code: int = 200
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class BatchSnapshot:
    """A provider's current view of a batch, normalized.

    ``results`` is set when the upstream returns results with the status
    (OpenRouter, Gemini); ``output_file_id``/``error_file_id`` are set when it
    hands back file ids instead (OpenAI). The service decides where to read from.
    """

    status: str
    request_counts: dict[str, int] = field(default_factory=dict)
    timestamps: dict[str, int | None] = field(default_factory=dict)
    errors: list[dict[str, Any]] | None = None
    results: list[BatchResultItem] | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    expires_at: int | None = None
    usage: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        """Whether this status is final (no further upstream change expected)."""
        return self.status in TERMINAL_BATCH_STATUSES


@dataclass(frozen=True, slots=True)
class BatchSubmission:
    """What a provider returns after accepting a batch.

    ``upstream_file_id`` is the input file the provider stored, when it stores
    one (OpenAI); it is kept for diagnostics and provider-side cleanup, not for
    serving clients — clients only ever see the proxy's own file ids.
    """

    upstream_id: str
    snapshot: BatchSnapshot
    upstream_file_id: str | None = None


__all__ = [
    "BATCH_STATUSES",
    "BATCH_TIMESTAMP_FIELDS",
    "BatchOwner",
    "BatchRequestItem",
    "BatchResultItem",
    "BatchSnapshot",
    "BatchStatus",
    "BatchSubmission",
    "BatchTarget",
    "DEFAULT_COMPLETION_WINDOW",
    "SUPPORTED_BATCH_ENDPOINT",
    "TERMINAL_BATCH_STATUSES",
    "USER_OWNER_PREFIX",
]
