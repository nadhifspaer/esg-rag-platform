"""Tests for the `ingest_document` orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.models.payload import Domain
from app.rag.embedder import EMBEDDING_DIM
from app.rag.ingestion.pipeline import ingest_document

# In-memory Qdrant warns that payload indexes are no-ops locally; expected here.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

COLLECTION = "test_esg_documents"


# --- fixture corpus & fakes -------------------------------------------------


def _make_corpus(tmp_path: Path) -> Path:
    """Write a 1-page PDF under company_documents/ plus a manifest entry for it."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    domain_dir = tmp_path / "company_documents"
    domain_dir.mkdir()
    pdf_path = domain_dir / "test-bank.pdf"

    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    text = pdf.beginText(72, 780)
    for line in (
        "PT Test Bank Tbk Sustainability Report 2025.",
        "The bank reduced its Scope 1 greenhouse gas emissions during the reporting",
        "period through energy-efficiency measures and renewable-energy sourcing.",
    ):
        text.textLine(line)
    pdf.drawText(text)
    pdf.showPage()
    pdf.save()

    manifest = {
        "domains": {
            "company_documents": [
                {
                    "file": "test-bank.pdf",
                    "bank": "PT Test Bank Tbk",
                    "aliases": ["TEST"],
                    "year": 2025,
                    "document_type": "sustainability_report",
                }
            ]
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pdf_path


class _FakeEmbeddings:
    def create(self, model, input):
        data = [
            SimpleNamespace(index=i, embedding=[0.1] * EMBEDDING_DIM) for i, _ in enumerate(input)
        ]
        return SimpleNamespace(data=data)


class FakeEmbedClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


class _FakeVisionCompletions:
    def __init__(self, body: str) -> None:
        self.body = body

    def create(self, **kwargs):
        message = SimpleNamespace(content=self.body)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeVisionClient:
    """Returns a fixed caption reply for every page (has_chart configurable)."""

    def __init__(self, *, has_chart: bool = True) -> None:
        caption = "A segmented category wheel diagram of ESG components." if has_chart else ""
        body = json.dumps({"has_chart": has_chart, "caption": caption})
        self.chat = SimpleNamespace(completions=_FakeVisionCompletions(body))


def _fake_upload_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))


def _settings() -> Settings:
    return Settings(
        openai_api_key="test",
        openai_embedding_model="text-embedding-3-small",
        openai_vision_model="gpt-4.1-mini",
        supabase_url="https://test.supabase.co",
        supabase_service_role_key="test-key",
        supabase_storage_bucket="test-bucket",
        qdrant_collection=COLLECTION,
    )


# --- tests ------------------------------------------------------------------


def test_ingest_document_wires_all_stages(tmp_path: Path) -> None:
    pdf = _make_corpus(tmp_path)
    client = QdrantClient(":memory:")

    result = ingest_document(
        pdf,
        qdrant_client=client,
        embed_client=FakeEmbedClient(),
        vision_client=FakeVisionClient(has_chart=True),
        upload_client=_fake_upload_client(),
        settings=_settings(),
    )

    # Metadata -> citation label + domain.
    assert result.source_name == "PT Test Bank Tbk — 2025 Sustainability Report"
    assert result.domain is Domain.COMPANY_DOCUMENTS
    # Stages produced chunks and the counts add up to what was upserted.
    assert result.text_chunks >= 1
    assert result.chart_caption_chunks == 1  # 1 page, fake vision flags a chart
    assert (
        result.points_upserted
        == result.text_chunks + result.table_chunks + result.chart_caption_chunks
    )
    # Everything actually landed in Qdrant.
    assert client.count(COLLECTION).count == result.points_upserted

    # The chart_caption chunk carries an uploaded image_url.
    points, _ = client.scroll(COLLECTION, limit=100, with_payload=True)
    captions = [p for p in points if p.payload["content_type"] == "chart_caption"]
    assert len(captions) == 1
    assert (
        captions[0]
        .payload["image_url"]
        .startswith("https://test.supabase.co/storage/v1/object/public/test-bucket/")
    )


def test_ingest_document_without_captioning(tmp_path: Path) -> None:
    pdf = _make_corpus(tmp_path)
    client = QdrantClient(":memory:")

    result = ingest_document(
        pdf,
        qdrant_client=client,
        embed_client=FakeEmbedClient(),
        settings=_settings(),
        caption_pages=False,  # skip vision/upload entirely
    )

    assert result.chart_caption_chunks == 0
    assert result.points_upserted == result.text_chunks + result.table_chunks
    assert result.points_upserted >= 1
    assert client.count(COLLECTION).count == result.points_upserted


def test_ingest_document_is_idempotent(tmp_path: Path) -> None:
    pdf = _make_corpus(tmp_path)
    client = QdrantClient(":memory:")
    kwargs = dict(
        qdrant_client=client,
        embed_client=FakeEmbedClient(),
        vision_client=FakeVisionClient(has_chart=True),
        upload_client=_fake_upload_client(),
        settings=_settings(),
    )

    first = ingest_document(pdf, **kwargs)
    second = ingest_document(pdf, **kwargs)  # re-ingest: delete-by-source then upsert

    assert first.points_upserted == second.points_upserted
    # No duplication: the collection holds exactly one document's worth of points.
    assert client.count(COLLECTION).count == second.points_upserted
