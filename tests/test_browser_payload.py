"""
Unit/integration tests for browser javascript_payload execution in app.browser.
"""
import asyncio
import pytest
from app.config import settings
from app.browser import BrowserEngine


@pytest.mark.asyncio
async def test_async_iife_javascript_payload(monkeypatch):
    """
    Test that an async IIFE payload (returning a Promise) is fully awaited
    before page content is captured.
    """
    monkeypatch.setattr(settings, "browser_engine", "playwright")
    engine = BrowserEngine()
    await engine.start_browser()
    try:
        payload = """
        (async () => {
            await new Promise(resolve => setTimeout(resolve, 300));
            document.title = "GRUB_PASS";
            const el = document.createElement("pre");
            el.id = "grub-pass-report";
            el.innerText = "payload_completed";
            document.body.appendChild(el);
            return "DONE";
        })()
        """
        html_url = "data:text/html,<html><head><title>Initial</title></head><body><h1>Hello</h1></body></html>"
        content, page_info, _ = await engine.crawl_with_context(
            html_url,
            javascript_enabled=True,
            javascript_payload=payload
        )
        assert page_info["title"] == "GRUB_PASS"
        assert 'id="grub-pass-report"' in content
        assert 'payload_completed' in content
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_sync_statement_javascript_payload(monkeypatch):
    """
    Test that a synchronous statement payload still works as expected.
    """
    monkeypatch.setattr(settings, "browser_engine", "playwright")
    engine = BrowserEngine()
    await engine.start_browser()
    try:
        payload = "document.title = 'SYNC_OK';"
        html_url = "data:text/html,<html><head><title>Initial</title></head><body><h1>Hello</h1></body></html>"
        content, page_info, _ = await engine.crawl_with_context(
            html_url,
            javascript_enabled=True,
            javascript_payload=payload
        )
        assert page_info["title"] == "SYNC_OK"
    finally:
        await engine.close()


if __name__ == "__main__":
    class MockMonkeypatch:
        def setattr(self, obj, attr, value):
            setattr(obj, attr, value)

    async def run_all():
        mp = MockMonkeypatch()
        print("Running test_async_iife_javascript_payload...")
        await test_async_iife_javascript_payload(mp)
        print("PASS: test_async_iife_javascript_payload")

        print("Running test_sync_statement_javascript_payload...")
        await test_sync_statement_javascript_payload(mp)
        print("PASS: test_sync_statement_javascript_payload")

    asyncio.run(run_all())
