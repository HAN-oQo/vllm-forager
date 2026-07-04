"""Firestore-backed :class:`~src.store.base.Store` (T0.6.1) — the M0.6 KB backend.

Schema (see ``docs/PLAN.md``'s storage-evolution section):

    items   collection, one document per item, keyed ``{repo}#{number}`` with the repo slug's
            own ``/`` replaced by ``__`` first. The ``google-cloud-firestore`` client parses a
            literal ``/`` inside a ``.document(id)`` string as a **path separator** (not an
            opaque character) — passing ``"o/r#1"`` raises ``ValueError: A document must have
            an even number of path elements``, since it's read as three path segments. The
            document body is the same normalized item dict the collector already produces
            (unchanged schema vs. the JSONL backend).
    state   collection, one document per state key (a repo slug), keyed the same way (its
            ``/`` replaced by ``__``) for the same reason — mirrors the JSONL backend's own
            ``{owner}__{repo}.jsonl`` file-naming convention). Body: ``{"value": <str>}``.

Filtering/ordering is done the same way as :mod:`~src.store.jsonl_store` — fetch the matching
docs, then filter/sort client-side in Python — rather than composing Firestore ``where`` +
``order_by`` queries. This keeps the two backends' semantics identical by construction (no risk
of Firestore-only edge cases in ordering/index requirements) and avoids requiring a composite
index for every filter combination in production Firestore (the local emulator doesn't enforce
these, but real Firestore does). **This is not free the way it is on JSONL**, though: a
``query()`` call with no ``repo`` — or any ``label``/``state``/``type``-only filter — reads the
entire ``items`` collection (billed per document, network-bound), where the JSONL backend's
"scan every file" is a local, unbilled disk read. Narrowing by ``repo`` keeps a query to that
one repo's documents; a KB that outgrows this should add server-side indexes for the other
filters rather than assume this backend is a drop-in performance match for JsonlStore.

Connects to a live Firestore project by default; set ``FIRESTORE_EMULATOR_HOST`` (e.g.
``localhost:8081``) to point the client at a local emulator instead — the client library reads
that env var itself. See `tests/test_store_contract.py`'s firestore parametrization
(``@pytest.mark.integration``) for how the emulator is used in tests.
"""

from __future__ import annotations

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from .base import Store

# Firestore rejects a batch with more than this many writes.
_BATCH_LIMIT = 500


def _item_doc_id(repo: str, number: int) -> str:
    """The `items` collection's document ID for (repo, number): ``{repo}#{number}``.

    `repo`'s own ``/`` is replaced with ``__`` first — see the module docstring for why a
    literal ``/`` can't survive being passed to ``.document(id)``.
    """
    return f"{repo.replace('/', '__')}#{number}"


def _state_doc_id(key: str) -> str:
    """The `state` collection's document ID for a state key (Firestore doc IDs disallow ``/``)."""
    return key.replace("/", "__")


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    """Split `items` into consecutive slices of at most `size` (last slice may be shorter)."""
    return [items[i : i + size] for i in range(0, len(items), size)]


class FirestoreStore(Store):
    """Store items + state in Google Cloud Firestore, against the same :class:`Store` contract."""

    def __init__(self, project: str | None = None) -> None:
        self._client = firestore.Client(project=project) if project else firestore.Client()
        self._items = self._client.collection("items")
        self._state = self._client.collection("state")

    # -- items ------------------------------------------------------------------
    def upsert_items(self, items: list[dict]) -> dict[str, int]:
        """Upsert `items` (batched, ``set`` overwrites) and return post-upsert totals per repo.

        Deduplicated by ``(repo, number)`` first, last occurrence wins — matching
        :class:`JsonlStore`'s dict-merge semantics (``existing[rec["number"]] = rec``).
        Without this, two records for the same item in one call could land in the same
        Firestore batch, and Firestore's batch API rejects more than one write to the same
        document in a single commit (``InvalidArgument``), unlike the JSONL backend's silent
        last-write-wins. Batched in groups of :data:`_BATCH_LIMIT` (the per-batch write cap).
        Returns ``{repo: post_upsert_total}`` for each repo touched, via a ``count()``
        aggregation query per repo (one read regardless of that repo's size).
        """
        if not items:
            return {}
        deduped: dict[tuple[str, int], dict] = {}
        for it in items:
            deduped[(it["repo"], it["number"])] = it
        deduped_items = list(deduped.values())
        repos = {repo for repo, _ in deduped}
        for batch_items in _chunks(deduped_items, _BATCH_LIMIT):
            batch = self._client.batch()
            for it in batch_items:
                batch.set(self._items.document(_item_doc_id(it["repo"], it["number"])), it)
            batch.commit()
        return {repo: self._count_for_repo(repo) for repo in repos}

    def _count_for_repo(self, repo: str) -> int:
        # google-cloud-firestore's type stubs mistype AggregationQuery.count() as returning
        # the *class* rather than an instance, so mypy sees a missing `self` on `.get()` and
        # an unindexable result. Verified correct at runtime against a live emulator — this
        # is a stub bug, not a real type error.
        query = self._items.where(filter=FieldFilter("repo", "==", repo)).count()
        agg = query.get()  # type: ignore[call-arg]
        return int(agg[0][0].value)  # type: ignore[index]

    def get_item(self, repo: str, number: int) -> dict | None:
        doc = self._items.document(_item_doc_id(repo, number)).get()
        return doc.to_dict() if doc.exists else None

    def query(
        self,
        *,
        repo: str | None = None,
        label: str | None = None,
        state: str | None = None,
        type: str | None = None,
    ) -> list[dict]:
        # One repo → a single equality filter narrows the read; otherwise scan the collection.
        # (Multiple simple equality `where`s compose fine without a composite index; `label`
        # and the rest are still applied client-side below to keep both backends' semantics
        # identical rather than depending on Firestore's array-contains + order_by rules.)
        docs = (
            self._items.where(filter=FieldFilter("repo", "==", repo)).stream()
            if repo is not None
            else self._items.stream()
        )
        out: list[dict] = []
        for snap in docs:
            # snap came from a query result, so it always has data; `to_dict()` is typed
            # Optional only because DocumentSnapshot is reused for `.get()` on a possibly
            # absent doc elsewhere in the SDK.
            rec = snap.to_dict() or {}
            if label is not None and label not in (rec.get("labels") or []):
                continue
            if state is not None and rec.get("state") != state:
                continue
            if type is not None and rec.get("type") != type:
                continue
            out.append(rec)
        out.sort(key=lambda r: r.get("updated_at") or "")
        return out

    # -- state ------------------------------------------------------------------
    def get_state(self, key: str) -> str | None:
        doc = self._state.document(_state_doc_id(key)).get()
        return (doc.to_dict() or {}).get("value") if doc.exists else None

    def set_state(self, key: str, value: str) -> None:
        self._state.document(_state_doc_id(key)).set({"value": value})
