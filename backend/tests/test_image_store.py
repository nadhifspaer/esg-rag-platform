"""Tests for the Supabase Storage page-image uploader."""

from __future__ import annotations

import base64
import os

import httpx
import pytest

from app.core.config import Settings
from app.rag.ingestion.image_store import (
    ImageUploadError,
    page_object_path,
    upload_page_image,
)

# --- fakes & helpers -------------------------------------------------------


class _Recorder:
    """An httpx MockTransport handler that records requests and returns a canned
    response, so a test can both drive and inspect the upload without a network."""

    def __init__(self, status: int = 200, body: str = "{}") -> None:
        self.status = status
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, text=self.body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def _settings(
    url: str = "https://proj.supabase.co",
    key: str = "svc-key",
    bucket: str = "esg-page-images",
) -> Settings:
    """Explicit settings so tests never read the real .env or touch Supabase."""
    return Settings(
        supabase_url=url,
        supabase_service_role_key=key,
        supabase_storage_bucket=bucket,
    )


IMAGE_BYTES = b"\x89PNG\r\n\x1a\n-fake-image-bytes"
OBJECT_PATH = "bank-bca/page-0001.png"  # == page_object_path("BANK-BCA", 1)
BASE = "https://proj.supabase.co"
BUCKET = "esg-page-images"


# --- unit: path building ----------------------------------------------------


def test_page_object_path_slugifies_source_and_pads_page() -> None:
    assert (
        page_object_path("BCA Sustainability Report 2023", 7)
        == "bca-sustainability-report-2023/page-0007.png"
    )


def test_page_object_path_falls_back_when_source_has_no_alnum() -> None:
    assert page_object_path("!!!", 1) == "document/page-0001.png"


def test_page_object_path_honours_image_format() -> None:
    assert page_object_path("Doc", 3, image_format="jpg") == "doc/page-0003.jpg"


# --- unit: request construction ---------------------------------------------


def test_upload_posts_to_the_object_endpoint() -> None:
    rec = _Recorder()
    upload_page_image(IMAGE_BYTES, OBJECT_PATH, client=rec.client(), settings=_settings())
    assert rec.last.method == "POST"
    assert str(rec.last.url) == f"{BASE}/storage/v1/object/{BUCKET}/{OBJECT_PATH}"


def test_upload_sends_expected_headers() -> None:
    rec = _Recorder()
    upload_page_image(IMAGE_BYTES, OBJECT_PATH, client=rec.client(), settings=_settings())
    headers = rec.last.headers
    assert headers["authorization"] == "Bearer svc-key"
    assert headers["apikey"] == "svc-key"
    assert headers["content-type"] == "image/png"
    assert headers["x-upsert"] == "true"


def test_upload_sends_image_bytes_as_body() -> None:
    rec = _Recorder()
    upload_page_image(IMAGE_BYTES, OBJECT_PATH, client=rec.client(), settings=_settings())
    assert rec.last.content == IMAGE_BYTES


def test_upload_returns_public_url() -> None:
    rec = _Recorder()
    url = upload_page_image(IMAGE_BYTES, OBJECT_PATH, client=rec.client(), settings=_settings())
    assert url == f"{BASE}/storage/v1/object/public/{BUCKET}/{OBJECT_PATH}"


def test_upsert_false_sets_header_false() -> None:
    rec = _Recorder()
    upload_page_image(
        IMAGE_BYTES, OBJECT_PATH, upsert=False, client=rec.client(), settings=_settings()
    )
    assert rec.last.headers["x-upsert"] == "false"


def test_jpg_format_uses_jpeg_mime() -> None:
    rec = _Recorder()
    upload_page_image(
        IMAGE_BYTES,
        "doc/page-0001.jpg",
        image_format="jpg",
        client=rec.client(),
        settings=_settings(),
    )
    assert rec.last.headers["content-type"] == "image/jpeg"


def test_trailing_slash_in_base_url_is_normalized() -> None:
    rec = _Recorder()
    url = upload_page_image(
        IMAGE_BYTES, OBJECT_PATH, client=rec.client(), settings=_settings(url=f"{BASE}/")
    )
    # No doubled slash in either the request URL or the returned public URL.
    assert str(rec.last.url) == f"{BASE}/storage/v1/object/{BUCKET}/{OBJECT_PATH}"
    assert url == f"{BASE}/storage/v1/object/public/{BUCKET}/{OBJECT_PATH}"


# --- unit: error paths ------------------------------------------------------


def test_http_error_raises_image_upload_error() -> None:
    rec = _Recorder(status=403, body='{"error":"forbidden: bucket not found"}')
    with pytest.raises(ImageUploadError, match="403"):
        upload_page_image(IMAGE_BYTES, OBJECT_PATH, client=rec.client(), settings=_settings())


def test_unconfigured_raises_before_any_request() -> None:
    rec = _Recorder()
    with pytest.raises(ImageUploadError):
        upload_page_image(
            IMAGE_BYTES, OBJECT_PATH, client=rec.client(), settings=_settings(url="", key="")
        )
    # It must fail on the config check, before touching the network.
    assert rec.requests == []


# --- live test (opt-in: explicit flag) --------------------------------------

# A minimal valid 1x1 PNG, small, real image bytes to round-trip through Storage.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_RUN_LIVE = os.environ.get("RUN_LIVE_SUPABASE_TESTS") == "1"
live = pytest.mark.skipif(
    not _RUN_LIVE,
    reason="RUN_LIVE_SUPABASE_TESTS!=1; skipping live Supabase tests (opt-in only)",
)


@live
def test_live_upload_round_trips_through_real_bucket() -> None:
    # Real settings/client (from .env); requires the public bucket to exist.
    path = "pytest/image-store-roundtrip.png"
    url = upload_page_image(_TINY_PNG, path)
    got = httpx.get(url, timeout=30.0)
    assert got.status_code == 200
    assert got.headers.get("content-type") == "image/png"
    assert got.content == _TINY_PNG
