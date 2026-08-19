"""Chat pipeline composition root: wires classifier, retrieval, rerank, generation, loop."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.models.citation import RetrievedChunk
from app.models.conversation import ConversationTurn
from app.models.payload import Domain
from app.observability.tracing import openai_class
from app.rag.chat_anchor_pinning import pin_anchored_chunks
from app.rag.compliance.resolver import source_name_for_entity
from app.rag.document_catalog import DocumentSummary, build_document_catalog
from app.rag.entity_index import Entity, EntityKind
from app.rag.generation.generator import generate_answer
from app.rag.generation.loop import Check, Generate, Retrieve
from app.rag.generation.self_check import GroundednessDecision, check_groundedness
from app.rag.hybrid_search import FusedResult, hybrid_search_with_legs
from app.rag.keyword_search import KeywordIndex, load_corpus_documents
from app.rag.query_classifier import QueryClassification, QueryType, classify_query
from app.rag.reranker import Scorer, build_scorer, rerank
from app.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)

# Widened retrieval limit for a cross-domain query's scoped leg, so a scoped bank's own
# chunk survives fusion instead of being truncated by the unscoped defaults.
_SCOPED_CROSS_DOMAIN_LIMIT = 100


@dataclass
class ChatEngine:
    """Holds the shared retrieval/generation resources and builds the loop's callables."""

    settings: Settings
    qdrant_client: QdrantClient
    keyword_index: KeywordIndex
    scorer: Scorer
    openai_client: OpenAI | None = None
    # Ingested-document catalog derived from the corpus at build time (see
    # build_chat_engine). Served read-only by GET /documents.
    document_catalog: list[DocumentSummary] = field(default_factory=list)

    def classify(self, query: str, *, use_llm_fallback: bool = True) -> QueryClassification:
        """Route the query: query type + domain filter (rule-based, with LLM fallback)."""
        return classify_query(
            query,
            client=self.openai_client,
            settings=self.settings,
            use_llm_fallback=use_llm_fallback,
        )

    def make_retrieve(
        self,
        domains: Sequence[Domain],
        *,
        source_names: Mapping[Domain, str | None] | None = None,
    ) -> Retrieve:
        """Build the loop's `retrieve`: hybrid search per domain, pooled, then one joint re-rank."""
        domains = tuple(domains)
        source_names = source_names or {}

        def retrieve(query: str) -> Sequence[RetrievedChunk]:
            candidates: list[FusedResult] = []
            raw_dense_by_domain: dict[Domain, list[SearchResult]] = {}
            raw_keyword_by_domain: dict[Domain, list[SearchResult]] = {}
            for domain in domains:
                # Widen only a scoped leg that's part of a genuinely cross-domain search.
                widen = len(domains) > 1 and source_names.get(domain) is not None
                wide_kwargs = (
                    {
                        "dense_limit": _SCOPED_CROSS_DOMAIN_LIMIT,
                        "keyword_limit": _SCOPED_CROSS_DOMAIN_LIMIT,
                        "limit": _SCOPED_CROSS_DOMAIN_LIMIT,
                    }
                    if widen
                    else {}
                )
                legs = hybrid_search_with_legs(
                    query,
                    domain=domain,
                    qdrant_client=self.qdrant_client,
                    keyword_index=self.keyword_index,
                    source_name=source_names.get(domain),
                    embed_client=self.openai_client,
                    settings=self.settings,
                    **wide_kwargs,
                )
                candidates.extend(legs.fused)
                raw_dense_by_domain[domain] = legs.dense
                raw_keyword_by_domain[domain] = legs.keyword

            reranked = rerank(
                query, candidates, scorer=self.scorer, top_n=self.settings.reranker_top_n
            )

            for domain in domains:
                if source_names.get(domain) is None:
                    continue
                reranked = pin_anchored_chunks(
                    query,
                    reranked,
                    raw_dense=raw_dense_by_domain[domain],
                    raw_keyword=raw_keyword_by_domain[domain],
                )
            return reranked

        return retrieve

    def make_generate(
        self,
        query_type: QueryType,
        high_accuracy: bool,
        history: Sequence[ConversationTurn] = (),
    ) -> Generate:
        """Build the loop's `generate`: generate an answer and return it as full text."""

        def generate(question: str, chunks: Sequence[RetrievedChunk]) -> str:
            answer = generate_answer(
                question,
                chunks,
                query_type=query_type,
                high_accuracy=high_accuracy,
                history=history,
                client=self.openai_client,
                settings=self.settings,
            )
            return "".join(answer.tokens)

        return generate

    def make_check(self) -> Check:
        """Build the loop's `check`: the groundedness judge bound to this engine's client."""

        def check(
            question: str, answer: str, chunks: Sequence[RetrievedChunk]
        ) -> GroundednessDecision:
            return check_groundedness(
                question, answer, chunks, client=self.openai_client, settings=self.settings
            )

        return check


def scoped_source_names(
    classification: QueryClassification, domains: Sequence[Domain]
) -> dict[Domain, str | None]:
    """Which single document (if any) each domain's retrieval leg should be confined to."""
    domains = tuple(domains)
    result: dict[Domain, str | None] = dict.fromkeys(domains, None)
    if not classification.confident:
        return result

    entities_by_kind: dict[EntityKind, list[Entity]] = {
        EntityKind.BANK: [],
        EntityKind.STANDARD: [],
    }
    for match in classification.entities:
        entities_by_kind.setdefault(match.entity.kind, []).append(match.entity)

    if Domain.COMPANY_DOCUMENTS in result:
        banks = entities_by_kind[EntityKind.BANK]
        if len(banks) == 1:
            result[Domain.COMPANY_DOCUMENTS] = source_name_for_entity(banks[0])

    if Domain.STANDARDS in result:
        standards = entities_by_kind[EntityKind.STANDARD]
        if len(standards) == 1:
            result[Domain.STANDARDS] = standards[0].canonical_name

    return result


def build_chat_engine(settings: Settings | None = None) -> ChatEngine:
    """Construct the engine and its shared resources; call once at app startup."""
    settings = settings or get_settings()
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=60,
    )
    documents = load_corpus_documents(qdrant_client, settings.qdrant_collection)
    keyword_index = KeywordIndex(documents)
    # Same scroll feeds the document catalog, no extra Qdrant round-trip.
    document_catalog = build_document_catalog(documents)
    logger.info(
        "chat engine: BM25 index built from %d chunks across %d documents",
        len(documents),
        len(document_catalog),
    )

    # Concrete Scorer (onnx vs. local cross-encoder) chosen by settings.reranker_provider.
    scorer = build_scorer(settings)
    # Instrumented client shared by generation, the self-check, the classifier, embeddings,
    # and metric extraction; falls back to the plain SDK when tracing isn't configured.
    client_class = openai_class(settings)
    openai_client = (
        client_class(api_key=settings.openai_api_key) if settings.openai_api_key else None
    )

    return ChatEngine(
        settings=settings,
        qdrant_client=qdrant_client,
        keyword_index=keyword_index,
        scorer=scorer,
        openai_client=openai_client,
        document_catalog=document_catalog,
    )
