"""End-to-end ingestion for one document: the entrypoint that ties the separable stages together."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.models.payload import ContentType, Domain
from app.rag.ingestion.chunker import Chunk, chunk_document
from app.rag.ingestion.figure_extractor import render_document
from app.rag.ingestion.image_store import page_object_path, upload_page_image
from app.rag.ingestion.indexer import index_chunks
from app.rag.ingestion.parser import extract_document
from app.rag.ingestion.payload_builder import build_source_name
from app.rag.ingestion.table_extractor import extract_tables
from app.rag.ingestion.vision_captioner import caption_page_image

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Summary of ingesting one document."""

    file_name: str
    source_name: str  # the citation label
    domain: Domain
    text_chunks: int
    table_chunks: int
    chart_caption_chunks: int
    points_upserted: int
    chunks: list[Chunk]  # all chunks, for inspection/sampling


def _caption_chunks(
    pdf_path: Path,
    document_type: str | None,
    source_name: str,
    *,
    settings: Settings,
    vision_client: OpenAI | None,
    upload_client=None,
) -> list[Chunk]:
    """Render every page and emit a chart_caption chunk for pages holding a chart/diagram."""
    chunks: list[Chunk] = []
    for rendered in render_document(pdf_path):
        try:
            caption = caption_page_image(
                rendered.image_bytes,
                image_format=rendered.image_format,
                client=vision_client,
                settings=settings,
            )
        except Exception:
            logger.exception("captioning failed for %s p%d", pdf_path.name, rendered.page_number)
            continue
        if not caption.has_chart or not caption.caption:
            continue
        try:
            object_path = page_object_path(source_name, rendered.page_number, rendered.image_format)
            image_url = upload_page_image(
                rendered.image_bytes,
                object_path,
                image_format=rendered.image_format,
                client=upload_client,
                settings=settings,
            )
        except Exception:
            logger.exception("image upload failed for %s p%d", pdf_path.name, rendered.page_number)
            continue
        chunks.append(
            Chunk(
                chunk_text=caption.caption,
                page_number=rendered.page_number,
                chunk_index=len(chunks),
                document_type=document_type,
                strategy_name="vision_caption",
                content_type=ContentType.CHART_CAPTION,
                image_url=image_url,
            )
        )
    return chunks


def ingest_document(
    pdf_path: str | Path,
    *,
    qdrant_client: QdrantClient,
    embed_client: OpenAI | None = None,
    vision_client: OpenAI | None = None,
    upload_client=None,
    settings: Settings | None = None,
    caption_pages: bool = True,
) -> IngestionResult:
    """Ingest one PDF end to end and upsert its chunks into Qdrant."""
    settings = settings or get_settings()
    pdf_path = Path(pdf_path)

    doc = extract_document(pdf_path)
    document_type = doc.metadata.get("document_type")
    source_name = build_source_name(doc.metadata)

    text_chunks = chunk_document(doc)
    table_chunks = extract_tables(pdf_path, document_type=document_type)
    caption_chunks = (
        _caption_chunks(
            pdf_path,
            document_type,
            source_name,
            settings=settings,
            vision_client=vision_client,
            upload_client=upload_client,
        )
        if caption_pages
        else []
    )

    all_chunks = [*text_chunks, *table_chunks, *caption_chunks]
    points = index_chunks(
        all_chunks,
        domain=doc.domain,
        metadata=doc.metadata,
        qdrant_client=qdrant_client,
        embed_client=embed_client,
        settings=settings,
    )

    return IngestionResult(
        file_name=pdf_path.name,
        source_name=source_name,
        domain=doc.domain,
        text_chunks=len(text_chunks),
        table_chunks=len(table_chunks),
        chart_caption_chunks=len(caption_chunks),
        points_upserted=points,
        chunks=all_chunks,
    )
