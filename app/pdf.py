"""PDF extraction for grub-crawl.

Text first: PyMuPDF pulls the text layer of every page. Pages with no usable
text (scanned or image-only) are rendered to PNG and handed to the configured
vision provider, the same one Ghost Protocol uses, with an OCR prompt adapted
from the gnosis-ocr pipeline. Output is one markdown section per page.

Pages carry a ``source`` so consumers can tell verbatim text from a
transcription: ``text_layer`` came from the PDF itself, ``ocr`` came from a
vision model (``ocr_model`` names it), ``empty`` had nothing usable, ``error``
failed. OCR'd sections are also marked in the markdown with an HTML comment
under the page heading.

Nothing here touches a browser. The crawler decides when a URL is a PDF and
hands the bytes to :func:`extract`; the ``/api/pdf/pages`` route uses
:func:`render_pages` for page images.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
PDF_CONTENT_TYPES = ("application/pdf", "application/x-pdf")

SOURCE_TEXT = "text_layer"
SOURCE_OCR = "ocr"
SOURCE_EMPTY = "empty"
SOURCE_ERROR = "error"
SOURCE_RENDER = "render"

# Adapted from gnosis-ocr's Nanonets prompt: markdown out, tables as HTML,
# equations as LaTeX, no image data, tagged watermarks and page numbers.
PDF_OCR_PROMPT = """You are reading one page of a PDF document. Extract all of the text on the page as if you were reading it naturally, and return it as Markdown.

Rules:
- Preserve headings, paragraphs, lists and reading order.
- Tables: return as HTML <table> markup.
- Equations: return as LaTeX wrapped in $$ ... $$.
- Images and figures: an empty markdown image with the caption, e.g. ![Figure 2: caption](). If there is no caption, give a one-line description instead. Never include image data.
- Watermarks: wrap in <watermark>...</watermark>. Page numbers: wrap in <page_number>...</page_number>.
- Check boxes: use ☐ and ☑.
- Do not repeat text. Do not add commentary. If the page is blank, return exactly: [blank page]"""

BLANK_PAGE_MARKER = "[blank page]"
OCR_ATTEMPTS = 2  # one retry when the model returns nothing


@dataclass
class PdfPage:
    """One page of a PDF. ``number`` is 1-based."""

    number: int
    text: str = ""
    char_count: int = 0
    source: str = SOURCE_EMPTY  # text_layer | ocr | empty | error | render
    ocr_model: Optional[str] = None
    image_png: Optional[bytes] = None
    width: int = 0
    height: int = 0
    error: Optional[str] = None
    repetition_trimmed: bool = False

    def describe(self) -> dict:
        out = {"number": self.number, "source": self.source, "char_count": self.char_count}
        if self.ocr_model:
            out["ocr_model"] = self.ocr_model
        if self.error:
            out["error"] = self.error
        if self.repetition_trimmed:
            out["repetition_trimmed"] = True
        return out


@dataclass
class PdfExtraction:
    """Result of extracting a whole PDF."""

    page_count: int = 0
    pages: List[PdfPage] = field(default_factory=list)
    title: str = ""
    author: str = ""
    truncated: bool = False
    ocr_model: Optional[str] = None
    timings_ms: dict = field(default_factory=dict)

    @property
    def text_pages(self) -> int:
        return sum(1 for p in self.pages if p.source == SOURCE_TEXT)

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.source == SOURCE_OCR)

    # Backwards-compatible alias (v0.15.0 called OCR pages "vision").
    vision_pages = ocr_pages

    @property
    def empty_pages(self) -> int:
        return sum(1 for p in self.pages if p.source in (SOURCE_EMPTY, SOURCE_ERROR))

    @property
    def render_mode(self) -> str:
        """Which path produced the text: pdf_text, pdf_vision, pdf_mixed or pdf_empty."""
        has_text, has_ocr = self.text_pages > 0, self.ocr_pages > 0
        if has_text and has_ocr:
            return "pdf_mixed"
        if has_text:
            return "pdf_text"
        if has_ocr:
            return "pdf_vision"
        return "pdf_empty"

    def to_markdown(self) -> str:
        parts: List[str] = []
        if self.title:
            parts.append(f"# {self.title}")
        for page in self.pages:
            body = page.text.strip()
            if not body:
                body = "_(no extractable text on this page)_"
            heading = f"## Page {page.number}"
            if page.source == SOURCE_OCR:
                heading += f"\n\n<!-- ocr: {page.ocr_model or 'vision'} -->"
            parts.append(f"{heading}\n\n{body}")
        if self.truncated:
            parts.append(f"_(stopped after {len(self.pages)} of {self.page_count} pages)_")
        return "\n\n".join(parts).strip() + ("\n" if parts else "")

    def to_text(self) -> str:
        return "\n\n".join(p.text.strip() for p in self.pages if p.text.strip()).strip()

    def summary(self) -> dict:
        return {
            "page_count": self.page_count,
            "pages_extracted": len(self.pages),
            "text_pages": self.text_pages,
            "ocr_pages": self.ocr_pages,
            "empty_pages": self.empty_pages,
            "truncated": self.truncated,
            "render_mode": self.render_mode,
            "ocr_model": self.ocr_model,
            "title": self.title,
            "author": self.author,
            "timings_ms": dict(self.timings_ms),
            "pages": [p.describe() for p in self.pages],
        }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def looks_like_pdf(
    content_type: Optional[str] = None,
    url: Optional[str] = None,
    head: Optional[bytes] = None,
) -> bool:
    """True when any available signal says PDF.

    ``head`` (the first bytes of the body) is authoritative when given.
    ``content_type`` wins over the URL. A ``.pdf`` path is a hint used when
    nothing else is known.
    """
    if head:
        return head.lstrip()[: len(PDF_MAGIC)] == PDF_MAGIC
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base in PDF_CONTENT_TYPES:
            return True
        if base and base not in ("application/octet-stream", "binary/octet-stream"):
            return False
    if url:
        path = urlparse(url).path.lower()
        return path.endswith(".pdf")
    return False


def url_hints_pdf(url: str) -> bool:
    """Cheap pre-browser check: does the URL path end in .pdf?"""
    try:
        return urlparse(url).path.lower().endswith(".pdf")
    except Exception:
        return False


def html_is_pdf_viewer(html: str) -> bool:
    """Detect Firefox's built-in pdf.js viewer in captured HTML.

    When Camoufox is pointed at a PDF it renders the viewer, and a naive crawl
    scrapes the viewer's UI. The ``mozdisallowselectionprint`` attribute and the
    ``pdfViewer`` container only occur in that viewer.
    """
    if not html:
        return False
    head = html[:2000]
    return "mozdisallowselectionprint" in head or 'class="pdfViewer"' in html[:200_000]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trim_repetition(text: str, *, window: int = 160, step: int = 1, max_repeats: int = 3) -> tuple[str, bool]:
    """Cut runaway OCR output where the model starts looping.

    Vision OCR models loop on tables and boilerplate. Two checks: a line
    repeated more than ``max_repeats`` times in a row is collapsed, and once a
    ``window``-character slice of the text reappears later (non-overlapping),
    everything from the first reappearance on is dropped. ``step`` is 1 so a
    loop of any period is caught; windows are hashed, so a page of text costs
    a few hundred KB at most. Returns ``(text, trimmed)``.
    """
    if not text:
        return text, False
    trimmed = False

    # 1. Consecutive duplicate lines.
    lines = text.split("\n")
    kept: List[str] = []
    run = 0
    for line in lines:
        if kept and line.strip() and line == kept[-1]:
            run += 1
            if run >= max_repeats:
                trimmed = True
                continue
        else:
            run = 0
        kept.append(line)
    text = "\n".join(kept)

    # 2. Re-appearing windows (the classic "same paragraph forever" loop).
    if len(text) > window * 2:
        seen: dict = {}
        cut_at: Optional[int] = None
        for start in range(0, len(text) - window + 1, max(1, step)):
            chunk = text[start:start + window]
            if not chunk.strip():
                continue
            key = hash(chunk)
            first = seen.setdefault(key, start)
            if first != start and start - first >= window and text[first:first + window] == chunk:
                cut_at = start
                break
        if cut_at is not None:
            text = text[:cut_at].rstrip()
            trimmed = True

    return text, trimmed


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _open_document(pdf_bytes: bytes):
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("PyMuPDF is not installed; add 'pymupdf' to requirements") from exc
    if not pdf_bytes or not looks_like_pdf(head=pdf_bytes[:16]):
        raise ValueError("Data is not a PDF (missing %PDF header)")
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise ValueError("PDF is password protected")
    return doc


def _metadata_value(doc, key: str) -> str:
    try:
        value = (doc.metadata or {}).get(key) or ""
    except Exception:
        return ""
    return str(value).strip()


def extract_text(
    pdf_bytes: bytes,
    *,
    max_pages: Optional[int] = None,
    min_text_chars: int = 40,
) -> PdfExtraction:
    """Extract the text layer of every page with PyMuPDF (synchronous).

    Pages with fewer than ``min_text_chars`` characters are marked ``empty`` so a
    caller can decide to OCR them.
    """
    started = time.monotonic()
    doc = _open_document(pdf_bytes)
    try:
        result = PdfExtraction(
            page_count=doc.page_count,
            title=_metadata_value(doc, "title"),
            author=_metadata_value(doc, "author"),
        )
        limit = doc.page_count if not max_pages else min(max_pages, doc.page_count)
        result.truncated = limit < doc.page_count
        for index in range(limit):
            page = PdfPage(number=index + 1)
            try:
                raw = doc[index].get_text("text") or ""
                page.text = _clean_text(raw)
                page.char_count = len(page.text)
                page.source = SOURCE_TEXT if page.char_count >= min_text_chars else SOURCE_EMPTY
            except Exception as exc:
                page.source = SOURCE_ERROR
                page.error = str(exc)
                logger.warning("PDF page %d text extraction failed: %s", page.number, exc)
            result.pages.append(page)
    finally:
        doc.close()
    result.timings_ms["text_ms"] = int((time.monotonic() - started) * 1000)
    return result


def render_pages(
    pdf_bytes: bytes,
    *,
    pages: Optional[Sequence[int]] = None,
    dpi: int = 110,
    max_pages: int = 50,
    max_side: Optional[int] = 1280,
) -> List[PdfPage]:
    """Render pages to PNG (synchronous).

    ``pages`` are 1-based; None means the first ``max_pages``. ``max_side`` caps
    the longer edge in pixels (vision models charge by image area), lowering the
    effective DPI for large pages.
    """
    doc = _open_document(pdf_bytes)
    try:
        if pages:
            wanted = [n for n in pages if 1 <= n <= doc.page_count][:max_pages]
        else:
            wanted = list(range(1, min(doc.page_count, max_pages) + 1))
        base_zoom = max(dpi, 36) / 72.0
        out: List[PdfPage] = []
        for number in wanted:
            page = PdfPage(number=number)
            try:
                import pymupdf  # type: ignore

                src = doc[number - 1]
                zoom = base_zoom
                longest = max(src.rect.width, src.rect.height) * zoom
                if max_side and longest > max_side:
                    zoom = max_side / max(src.rect.width, src.rect.height)
                pix = src.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                page.image_png = pix.tobytes("png")
                page.width, page.height = pix.width, pix.height
                page.source = SOURCE_RENDER
            except Exception as exc:
                page.source = SOURCE_ERROR
                page.error = str(exc)
                logger.warning("PDF page %d render failed: %s", number, exc)
            out.append(page)
        return out
    finally:
        doc.close()


def _provider_model_name(provider: Any) -> str:
    for attr in ("vision_model", "model"):
        value = getattr(provider, attr, None)
        if isinstance(value, str) and value:
            return value
    return provider.__class__.__name__


async def extract(
    pdf_bytes: bytes,
    *,
    vision_provider: Any = None,
    max_pages: Optional[int] = None,
    min_text_chars: int = 40,
    dpi: int = 110,
    max_side: Optional[int] = 1280,
    max_vision_pages: int = 20,
    vision_concurrency: int = 1,
) -> PdfExtraction:
    """Text layer first, then vision OCR for pages that came back empty.

    ``vision_provider`` must expose ``async vision(image_bytes, prompt, *, detail)``
    (the agent provider protocol). When it is None, or it raises
    NotImplementedError, empty pages stay empty and the result still succeeds.
    A provider may also expose ``async unload_vision()``; it is called once
    after the last OCR page so a shared GPU gets its memory back.
    """
    started = time.monotonic()
    extraction = await asyncio.to_thread(
        extract_text, pdf_bytes, max_pages=max_pages, min_text_chars=min_text_chars
    )

    candidates = [p for p in extraction.pages if p.source == SOURCE_EMPTY]
    if vision_provider is None or not candidates:
        extraction.timings_ms["total_ms"] = int((time.monotonic() - started) * 1000)
        return extraction

    targets = candidates[:max_vision_pages]
    if len(candidates) > len(targets):
        logger.info(
            "PDF has %d image-only pages; OCRing the first %d (pdf_vision_max_pages)",
            len(candidates), len(targets),
        )

    model_name = _provider_model_name(vision_provider)
    vision_started = time.monotonic()
    rendered = await asyncio.to_thread(
        render_pages, pdf_bytes,
        pages=[p.number for p in targets], dpi=dpi, max_pages=len(targets), max_side=max_side,
    )
    by_number = {p.number: p for p in rendered}
    semaphore = asyncio.Semaphore(max(1, vision_concurrency))
    unsupported = False
    attempted = False

    async def ocr(page: PdfPage) -> None:
        nonlocal unsupported, attempted
        image = by_number.get(page.number)
        if image is None or not image.image_png:
            page.source = SOURCE_ERROR
            page.error = (image.error if image else None) or "render failed"
            return
        async with semaphore:
            if unsupported:
                return
            last_error: Optional[str] = None
            for attempt in range(OCR_ATTEMPTS):
                try:
                    attempted = True
                    text = await vision_provider.vision(image.image_png, PDF_OCR_PROMPT, detail="high")
                except NotImplementedError:
                    unsupported = True
                    return
                except Exception as exc:
                    last_error = f"vision: {exc}"
                    logger.warning("PDF page %d OCR attempt %d failed: %s", page.number, attempt + 1, exc)
                    continue
                cleaned, trimmed = trim_repetition(_clean_text(text or ""))
                if not cleaned or cleaned == BLANK_PAGE_MARKER:
                    last_error = None
                    if attempt + 1 < OCR_ATTEMPTS:
                        logger.info("PDF page %d OCR returned nothing; retrying once", page.number)
                        continue
                    page.text, page.char_count, page.source = "", 0, SOURCE_EMPTY
                    return
                page.text = cleaned
                page.char_count = len(cleaned)
                page.source = SOURCE_OCR
                page.ocr_model = model_name
                page.repetition_trimmed = trimmed
                return
            page.source = SOURCE_ERROR
            page.error = last_error or "vision: no output"

    try:
        await asyncio.gather(*(ocr(p) for p in targets))
    finally:
        unload = getattr(vision_provider, "unload_vision", None)
        if attempted and callable(unload):
            try:
                await unload()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("vision unload failed: %s", exc)

    if unsupported:
        logger.info("Vision provider does not support images; PDF pages left as text-only")
    if extraction.ocr_pages:
        extraction.ocr_model = model_name

    extraction.timings_ms["vision_ms"] = int((time.monotonic() - vision_started) * 1000)
    extraction.timings_ms["total_ms"] = int((time.monotonic() - started) * 1000)
    return extraction


def title_from_url(url: str) -> str:
    """Fallback title: the file name without extension."""
    name = urlparse(url).path.rsplit("/", 1)[-1]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.replace("_", " ").replace("-", " ").strip() or url
