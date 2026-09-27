"""Tests for app.pdf: detection, text extraction, page rendering, vision fallback.

PDFs are generated in-test with PyMuPDF, so these need the ``pymupdf`` package
(present in the service image) but no network and no browser.
"""

import pytest

pymupdf = pytest.importorskip("pymupdf")

from app import pdf as pdfx  # noqa: E402


def _make_pdf(pages):
    """pages: list of strings; an empty string makes a blank (image-only style) page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        if text:
            page.insert_text((72, 72), text, fontsize=12)
    doc.set_metadata({"title": "Test Document", "author": "grub"})
    data = doc.tobytes()
    doc.close()
    return data


LONG = "The quick brown fox jumps over the lazy dog. " * 4


class TestDetection:
    def test_magic_bytes_win(self):
        assert pdfx.looks_like_pdf(head=b"%PDF-1.7\n%...")
        assert not pdfx.looks_like_pdf(head=b"<!doctype html>", content_type="application/pdf")

    def test_content_type(self):
        assert pdfx.looks_like_pdf(content_type="application/pdf; charset=binary")
        assert pdfx.looks_like_pdf(content_type="application/x-pdf")
        assert not pdfx.looks_like_pdf(content_type="text/html", url="https://x/doc.pdf")

    def test_url_hint_only_when_type_unknown(self):
        assert pdfx.looks_like_pdf(url="https://x/files/report.PDF")
        assert pdfx.looks_like_pdf(content_type="application/octet-stream", url="https://x/a.pdf")
        assert not pdfx.looks_like_pdf(url="https://x/page.html")
        assert pdfx.url_hints_pdf("https://x/a.pdf?download=1")
        assert not pdfx.url_hints_pdf("https://x/a.pdf.html")

    def test_pdfjs_viewer_html(self):
        assert pdfx.html_is_pdf_viewer('<!DOCTYPE html><html dir="ltr" mozdisallowselectionprint="" lang="en-US">')
        assert not pdfx.html_is_pdf_viewer("<html><body><h1>Hello</h1></body></html>")
        assert not pdfx.html_is_pdf_viewer("")


class TestExtractText:
    def test_text_and_empty_pages(self):
        data = _make_pdf([LONG, "", "Short"])
        result = pdfx.extract_text(data, min_text_chars=40)
        assert result.page_count == 3
        assert result.title == "Test Document"
        assert result.author == "grub"
        assert [p.source for p in result.pages] == ["text", "empty", "empty"]
        assert "quick brown fox" in result.pages[0].text
        assert result.pages[2].text == "Short"  # kept, just below the threshold
        assert result.render_mode == "pdf_text"
        assert not result.truncated

    def test_max_pages_truncates(self):
        data = _make_pdf([LONG, LONG, LONG])
        result = pdfx.extract_text(data, max_pages=2)
        assert result.page_count == 3
        assert len(result.pages) == 2
        assert result.truncated

    def test_markdown_shape(self):
        data = _make_pdf([LONG, ""])
        md = pdfx.extract_text(data).to_markdown()
        assert md.startswith("# Test Document\n\n## Page 1\n\n")
        assert "## Page 2\n\n_(no extractable text on this page)_" in md

    def test_rejects_non_pdf(self):
        with pytest.raises(ValueError):
            pdfx.extract_text(b"<html>not a pdf</html>")
        with pytest.raises(ValueError):
            pdfx.extract_text(b"")


class TestRender:
    def test_render_returns_png(self):
        data = _make_pdf([LONG, LONG])
        pages = pdfx.render_pages(data, pages=[2], dpi=72)
        assert len(pages) == 1
        assert pages[0].number == 2
        assert pages[0].image_png.startswith(b"\x89PNG")
        assert pages[0].width == 595 and pages[0].height == 842

    def test_render_ignores_out_of_range_and_caps(self):
        data = _make_pdf([LONG, LONG, LONG])
        pages = pdfx.render_pages(data, pages=[0, 3, 9], dpi=50)
        assert [p.number for p in pages] == [3]
        pages = pdfx.render_pages(data, dpi=50, max_pages=2)
        assert [p.number for p in pages] == [1, 2]


class _FakeVision:
    def __init__(self, reply="Scanned page text recovered by OCR.", fail=None):
        self.reply, self.fail, self.calls = reply, fail, []

    async def vision(self, image_bytes, prompt, *, detail="low"):
        self.calls.append((len(image_bytes), detail))
        if self.fail:
            raise self.fail
        return self.reply


class _NoVision:
    async def vision(self, image_bytes, prompt, *, detail="low"):
        raise NotImplementedError("no vision")


class TestExtractWithVision:
    @pytest.mark.asyncio
    async def test_vision_fills_empty_pages_only(self):
        data = _make_pdf([LONG, "", ""])
        fake = _FakeVision()
        result = await pdfx.extract(data, vision_provider=fake, dpi=50)
        assert [p.source for p in result.pages] == ["text", "vision", "vision"]
        assert len(fake.calls) == 2
        assert all(detail == "high" for _, detail in fake.calls)
        assert result.render_mode == "pdf_mixed"
        assert "recovered by OCR" in result.to_markdown()
        assert "vision_ms" in result.timings_ms

    @pytest.mark.asyncio
    async def test_blank_marker_keeps_page_empty(self):
        data = _make_pdf([""])
        result = await pdfx.extract(data, vision_provider=_FakeVision(reply="[blank page]"), dpi=50)
        assert result.pages[0].source == "empty"
        assert result.render_mode == "pdf_empty"

    @pytest.mark.asyncio
    async def test_no_provider_and_unsupported_provider(self):
        data = _make_pdf([LONG, ""])
        result = await pdfx.extract(data, vision_provider=None)
        assert [p.source for p in result.pages] == ["text", "empty"]
        result = await pdfx.extract(data, vision_provider=_NoVision(), dpi=50)
        assert [p.source for p in result.pages] == ["text", "empty"]

    @pytest.mark.asyncio
    async def test_vision_error_is_per_page(self):
        data = _make_pdf(["", LONG])
        result = await pdfx.extract(data, vision_provider=_FakeVision(fail=RuntimeError("boom")), dpi=50)
        assert result.pages[0].source == "error"
        assert "boom" in result.pages[0].error
        assert result.pages[1].source == "text"

    @pytest.mark.asyncio
    async def test_max_vision_pages_cap(self):
        data = _make_pdf(["", "", ""])
        fake = _FakeVision()
        result = await pdfx.extract(data, vision_provider=fake, dpi=50, max_vision_pages=1)
        assert len(fake.calls) == 1
        assert [p.source for p in result.pages] == ["vision", "empty", "empty"]


def test_title_from_url():
    assert pdfx.title_from_url("https://x/papers/attention_is-all.pdf") == "attention is all"
    assert pdfx.title_from_url("https://x/") == "https://x/"
