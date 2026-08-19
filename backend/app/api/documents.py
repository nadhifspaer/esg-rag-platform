"""GET /documents: list the ingested documents, optionally filtered by domain."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.chat import get_chat_engine
from app.models.payload import Domain
from app.rag.chat_engine import ChatEngine
from app.rag.document_catalog import DocumentSummary

router = APIRouter(tags=["documents"])


class DocumentInfo(BaseModel):
    """One ingested document's metadata, aggregated from its chunks."""

    source_name: str
    domain: str
    document_type: str
    bank: str | None
    year: int | None
    chunk_count: int
    page_count: int

    @classmethod
    def from_summary(cls, summary: DocumentSummary) -> DocumentInfo:
        return cls(
            source_name=summary.source_name,
            domain=summary.domain.value,
            document_type=summary.document_type,
            bank=summary.bank,
            year=summary.year,
            chunk_count=summary.chunk_count,
            page_count=summary.page_count,
        )


class DocumentsResponse(BaseModel):
    """The document listing: a count and the documents (sorted by domain, source_name)."""

    count: int
    documents: list[DocumentInfo]


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    engine: Annotated[ChatEngine, Depends(get_chat_engine)],
    domain: Annotated[
        Domain | None,
        Query(description="Filter to one domain (company_documents/standards)."),
    ] = None,
) -> DocumentsResponse:
    """List ingested documents, optionally filtered to one domain."""
    catalog = engine.document_catalog
    if domain is not None:
        catalog = [summary for summary in catalog if summary.domain is domain]
    documents = [DocumentInfo.from_summary(summary) for summary in catalog]
    return DocumentsResponse(count=len(documents), documents=documents)
