"""Client-side BM25 keyword search: the exact-match leg of hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.models.payload import ChunkPayload, Domain
from app.rag.vector_store import SearchResult

# Okapi BM25 parameters: k1 controls term-frequency saturation, b controls length
# normalisation. 1.5 / 0.75 are the standard, well-tested defaults.
_K1 = 1.5
_B = 0.75

# A token is a run of lowercase letters/digits, so 'GRI 305-1' splits into ['gri','305','1'].
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens, no stemming, no stopword removal."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Doc:
    """One indexed chunk: its point id, its payload, and its tokenised text."""

    id: str
    payload: ChunkPayload
    tokens: list[str]


class _DomainIndex:
    """A BM25 index over the chunks of a single domain."""

    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs
        self._n = len(docs)
        self._tf: list[Counter[str]] = [Counter(d.tokens) for d in docs]
        self._doc_len: list[int] = [len(d.tokens) for d in docs]
        self._avgdl: float = (sum(self._doc_len) / self._n) if self._n else 0.0

        # Document frequency: in how many docs each term appears.
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        # Okapi IDF with +1 smoothing; rare terms (small df) get a large IDF.
        self._idf: dict[str, float] = {
            term: math.log(1 + (self._n - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def search(
        self, query_tokens: list[str], limit: int, *, source_name: str | None = None
    ) -> list[SearchResult]:
        """Score every doc against the query terms; return the top `limit` (score>0)."""
        if self._n == 0 or not query_tokens:
            return []
        scored: list[tuple[float, int]] = []
        for i in range(self._n):
            if source_name is not None and self._docs[i].payload.source_name != source_name:
                continue
            tf = self._tf[i]
            dl = self._doc_len[i]
            norm = _K1 * (1 - _B + _B * dl / self._avgdl) if self._avgdl else _K1
            score = 0.0
            for term in query_tokens:
                f = tf.get(term, 0)
                if not f:
                    continue
                score += self._idf.get(term, 0.0) * (f * (_K1 + 1)) / (f + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchResult(id=self._docs[i].id, score=score, payload=self._docs[i].payload)
            for score, i in scored[:limit]
        ]


class KeywordIndex:
    """In-process BM25 index over the whole corpus, partitioned by domain."""

    def __init__(self, documents: Iterable[tuple[str, ChunkPayload]]) -> None:
        by_domain: dict[Domain, list[_Doc]] = defaultdict(list)
        for point_id, payload in documents:
            by_domain[payload.domain].append(
                _Doc(id=point_id, payload=payload, tokens=tokenize(payload.chunk_text))
            )
        self._by_domain: dict[Domain, _DomainIndex] = {
            domain: _DomainIndex(docs) for domain, docs in by_domain.items()
        }

    def search(
        self,
        query: str,
        domain: Domain,
        *,
        source_name: str | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Return the `limit` best BM25 matches for `query` within one domain, optionally scoped."""
        index = self._by_domain.get(domain)
        if index is None:
            return []
        return index.search(tokenize(query), limit, source_name=source_name)


def load_corpus_documents(
    client: QdrantClient,
    collection_name: str,
    *,
    batch_size: int = 256,
) -> list[tuple[str, ChunkPayload]]:
    """Scroll the collection (read-only) and return `(point_id, ChunkPayload)` for every chunk."""
    documents: list[tuple[str, ChunkPayload]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            documents.append((str(point.id), ChunkPayload.from_payload(point.payload or {})))
        if offset is None:
            break
    return documents
