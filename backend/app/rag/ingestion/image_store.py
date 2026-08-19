"""Upload a rendered page image to Supabase Storage and return its URL."""

from __future__ import annotations

import re

import httpx

from app.core.config import Settings, get_settings

_UPLOAD_TIMEOUT = 30.0  # seconds; page PNGs are small, but Storage can be slow-ish

# Collapse any run of non-url-safe characters into a single hyphen, for slugs.
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ImageUploadError(RuntimeError):
    """Raised when the page image cannot be uploaded to Supabase Storage."""


def _mime_for(image_format: str) -> str:
    """Map an image format to its MIME type (jpg/jpeg both -> image/jpeg)."""
    fmt = image_format.lower()
    return "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"


def _short(text: str, limit: int = 200) -> str:
    """Trim a response body to a single readable line for error messages."""
    collapsed = " ".join(text.split())
    return collapsed[:limit] + "..." if len(collapsed) > limit else collapsed


def page_object_path(source_name: str, page_number: int, image_format: str = "png") -> str:
    """Build a stable, URL-safe storage key for one page image; re-ingest overwrites it."""
    slug = _SLUG_RE.sub("-", source_name.lower()).strip("-") or "document"
    return f"{slug}/page-{page_number:04d}.{image_format.lower()}"


def upload_page_image(
    image_bytes: bytes,
    object_path: str,
    *,
    image_format: str = "png",
    upsert: bool = True,
    client: httpx.Client | None = None,
    settings: Settings | None = None,
) -> str:
    """Upload one page image to Supabase Storage; return its public URL."""
    settings = settings or get_settings()
    base = settings.supabase_url.rstrip("/")
    key = settings.supabase_service_role_key
    bucket = settings.supabase_storage_bucket

    if not base or not key:
        raise ImageUploadError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set to upload page "
            "images. Add them to your .env (see .env.example)."
        )

    object_path = object_path.lstrip("/")
    upload_url = f"{base}/storage/v1/object/{bucket}/{object_path}"
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": _mime_for(image_format),
        # Overwrite an existing object instead of returning 409 on re-ingestion.
        "x-upsert": "true" if upsert else "false",
    }

    owns_client = client is None
    client = client or httpx.Client(timeout=_UPLOAD_TIMEOUT)
    try:
        response = client.post(upload_url, content=image_bytes, headers=headers)
    except httpx.HTTPError as exc:
        raise ImageUploadError(f"upload request to Supabase Storage failed: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    if not response.is_success:
        raise ImageUploadError(
            f"Supabase Storage upload failed (HTTP {response.status_code}): {_short(response.text)}"
        )

    # Public object URL; resolves as long as the bucket is public.
    return f"{base}/storage/v1/object/public/{bucket}/{object_path}"
