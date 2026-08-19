"""Full-page rasterization: a PDF page -> a page-sized PNG image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

# 150 DPI (vs. PyMuPDF's default 72) keeps small axis labels and legend text legible
# without producing needlessly huge images.
DEFAULT_DPI = 150

# PNG (lossless) keeps thin chart lines and small text crisp; JPEG's block compression
# would smear exactly those high-contrast edges.
DEFAULT_IMAGE_FORMAT = "png"


@dataclass(frozen=True)
class RenderedPage:
    """One PDF page rendered to an in-memory image."""

    page_number: int  # 1-based, matches parser.PageText and ChunkPayload
    image_bytes: bytes
    image_format: str  # e.g. "png"
    width: int  # rendered image width in pixels
    height: int  # rendered image height in pixels
    dpi: int  # resolution the page was rasterized at

    def save(self, directory: Path, stem: str) -> Path:
        """Write the image to `directory/{stem}_p{page:04d}.{format}` and return it."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}_p{self.page_number:04d}.{self.image_format}"
        path.write_bytes(self.image_bytes)
        return path


def _render_page(page: pymupdf.Page, page_number: int, dpi: int, image_format: str) -> RenderedPage:
    """Rasterize one already-open PyMuPDF page into a `RenderedPage` (plain RGB, no alpha)."""
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return RenderedPage(
        page_number=page_number,
        image_bytes=pixmap.tobytes(output=image_format),
        image_format=image_format,
        width=pixmap.width,
        height=pixmap.height,
        dpi=dpi,
    )


def render_page(
    pdf_path: Path,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> RenderedPage:
    """Render a single page (1-based) of a PDF to a page-sized image."""
    with pymupdf.open(str(pdf_path)) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise IndexError(
                f"page {page_number} out of range for {pdf_path.name} (1..{doc.page_count})"
            )
        return _render_page(doc[page_number - 1], page_number, dpi, image_format)


def render_document(
    pdf_path: Path,
    dpi: int = DEFAULT_DPI,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> list[RenderedPage]:
    """Render every page of a PDF, in order, to page-sized images (document opened once)."""
    with pymupdf.open(str(pdf_path)) as doc:
        return [
            _render_page(page, page_number, dpi, image_format)
            for page_number, page in enumerate(doc, start=1)
        ]
