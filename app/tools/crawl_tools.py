"""Crawling tools for Grub Crawler agent."""

import uuid
import logging
from typing import List, Dict, Any, Optional
from app.tools.base import BaseTool, ToolResult, tool
from app.crawler import get_crawler_engine

logger = logging.getLogger(__name__)


@tool(description="Crawl a single URL and return HTML content and markdown")
async def crawl(
    url: str,
    javascript: bool = True,
    screenshot: bool = False,
    dedupe_tables: bool = True
) -> Dict[str, Any]:
    """Crawl a single URL with configurable options.
    
    Args:
        url: The URL to crawl
        javascript: Whether to enable JavaScript rendering (default: True)
        screenshot: Whether to take a screenshot (default: False)
        
    Returns:
        Dict containing HTML, markdown, and metadata
    """
    try:
        # Get crawler engine (user context would come from auth in real implementation)
        crawler = await get_crawler_engine()
        
        # Perform crawl
        result = await crawler.crawl_url(
            url=url,
            javascript=javascript,
            screenshot=screenshot,
            dedupe_tables=dedupe_tables,
            session_id=str(uuid.uuid4())  # Generate session ID
        )
        
        if result.success:
            return {
                "url": result.url,
                "title": result.title,
                "html": result.html,
                "markdown": result.markdown,
                "metadata": {
                    "javascript_enabled": javascript,
                    "screenshot_taken": bool(result.screenshot_path),
                    "screenshot_path": result.screenshot_path,
                    "processing_time": result.processing_time,
                    "browser_time": result.browser_time,
                    "markdown_time": result.markdown_time,
                    "page_info": result.page_info,
                    "status": "success"
                }
            }
        else:
            return {
                "url": result.url,
                "error": result.error_message,
                "metadata": {
                    "javascript_enabled": javascript,
                    "processing_time": result.processing_time,
                    "status": "failed"
                }
            }
            
    except Exception as e:
        logger.error(f"Error in crawl tool: {e}", exc_info=True)
        return {
            "url": url,
            "error": str(e),
            "metadata": {
                "status": "error"
            }
        }


@tool(description="Crawl a URL and return only markdown content")
async def markdown(url: str, javascript: bool = True, dedupe_tables: bool = True) -> str:
    """Crawl a URL and return only the markdown content.
    
    Args:
        url: The URL to crawl
        javascript: Whether to enable JavaScript rendering (default: True)
        
    Returns:
        Markdown content as string
    """
    try:
        # Get crawler engine
        crawler = await get_crawler_engine()
        
        # Perform markdown-only crawl
        markdown_content = await crawler.crawl_for_markdown_only(
            url=url,
            javascript=javascript,
            dedupe_tables=dedupe_tables
        )
        
        return markdown_content
        
    except Exception as e:
        logger.error(f"Error in markdown tool: {e}", exc_info=True)
        return f"Error crawling {url}: {str(e)}"


@tool(description="Crawl multiple URLs in batch and return results immediately")
async def batch(
    urls: List[str],
    javascript: bool = True,
    max_concurrent: int = 3,
    dedupe_tables: bool = True
) -> Dict[str, Any]:
    """Crawl multiple URLs in batch and return results.
    
    Args:
        urls: List of URLs to crawl
        javascript: Whether to enable JavaScript rendering (default: True)
        max_concurrent: Maximum concurrent crawls (default: 3)
        
    Returns:
        Dict containing batch results
    """
    try:
        # Get crawler engine
        crawler = await get_crawler_engine()
        
        # Generate session ID for this batch
        session_id = str(uuid.uuid4())
        
        # Perform batch crawl
        batch_result = await crawler.batch_crawl(
            urls=urls,
            javascript=javascript,
            screenshot=False,  # Disable screenshots for batch to save time
            max_concurrent=max_concurrent,
            session_id=session_id,
            dedupe_tables=dedupe_tables
        )
        
        return {
            "session_id": session_id,
            "urls": batch_result["urls"],
            "successful_results": batch_result["results"],
            "failed_results": batch_result["failed"],
            "summary": batch_result["summary"],
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error in batch tool: {e}", exc_info=True)
        return {
            "urls": urls,
            "error": str(e),
            "summary": {
                "total": len(urls),
                "success": 0,
                "failed": len(urls)
            },
            "status": "error"
        }


@tool(description="Ghost Protocol: screenshot a URL and extract content via vision AI, bypassing anti-bot blocks")
async def ghost_extract(
    url: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Ghost Protocol: take a screenshot of a URL and extract content using vision AI.

    Useful when standard crawling is blocked by anti-bot measures (Cloudflare,
    CAPTCHAs, challenge pages). Captures the page visually and reads content
    from the rendered pixels using Claude or GPT-4o vision.

    Args:
        url: The URL to ghost-extract
        timeout: Navigation timeout in seconds (default: 30)

    Returns:
        Dict with extracted content, render_mode, and timing metadata
    """
    from app.config import settings
    from app.agent.ghost import run_ghost_protocol, create_ghost_provider

    if not settings.agent_ghost_enabled:
        return {
            "url": url,
            "error": "Ghost Protocol is disabled. Set AGENT_GHOST_ENABLED=true to enable.",
            "render_mode": "ghost",
            "success": False,
        }

    try:
        provider = create_ghost_provider()
        result = await run_ghost_protocol(
            url,
            provider=provider,
            max_width=settings.agent_ghost_max_image_width,
            timeout=timeout,
        )

        return {
            "url": result.url,
            "success": result.success,
            "content": result.content,
            "render_mode": result.render_mode,
            "block_signal": result.block_signal,
            "block_reason": result.block_reason,
            "capture_ms": result.capture_ms,
            "extraction_ms": result.extraction_ms,
            "total_ms": result.total_ms,
            "provider": result.provider,
            "blocked_content": result.blocked_content,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"Ghost extract failed for {url}: {e}", exc_info=True)
        return {
            "url": url,
            "error": str(e),
            "render_mode": "ghost",
            "success": False,
        }


@tool(description="Get information about the crawler service")
async def crawler_info() -> Dict[str, Any]:
    """Get information about the crawler service capabilities.
    
    Returns:
        Dict containing crawler service information
    """
    from app.config import settings
    
    return {
        "service": "grub-crawl",
        "version": "1.0.0",
        "capabilities": {
            "javascript_execution": True,
            "screenshots": True,
            "batch_crawling": True,
            "markdown_generation": True,
            "content_filtering": True
        },
        "configuration": {
            "max_concurrent_crawls": settings.max_concurrent_crawls,
            "default_timeout": settings.browser_timeout,
            "headless_browser": settings.browser_headless,
            "javascript_enabled": settings.enable_javascript,
            "screenshots_enabled": settings.enable_screenshots
        },
        "supported_formats": ["html", "markdown", "json"],
        "supported_screenshot_modes": ["full", "top", "off"]
    }


@tool(description="Download a file (PDF, image, archive, ...) through the crawler and store it in a session")
async def download(url: str, session_id: str = "", filename: str = "", use_browser: bool = False) -> Dict[str, Any]:
    """Fetch a binary file and save it under downloads/ in the session storage.

    Args:
        url: File URL to download
        session_id: Session to store the file in (generated when empty)
        filename: Override the stored file name (derived from the URL/headers when empty)
        use_browser: Fetch through the stealth browser instead of plain HTTP (default: False)

    Returns:
        Dict with filename, content_type, size_bytes, saved_path and session_id
    """
    try:
        crawler = await get_crawler_engine()
        sid = session_id or str(uuid.uuid4())
        result = await crawler.fetch_binary(
            url=url, use_browser=use_browser, session_id=sid, filename=filename or None
        )
        if not result.get("success"):
            return {"url": url, "error": result.get("error") or "download failed", "metadata": {"status": "failed"}}
        return {
            "url": url,
            "session_id": sid,
            "filename": result.get("filename"),
            "content_type": result.get("content_type"),
            "size_bytes": len(result.get("content") or b""),
            "saved_path": result.get("saved_path"),
            "status_code": result.get("status_code"),
            "metadata": {"status": "success"},
        }
    except Exception as e:
        logger.error(f"Error in download tool: {e}", exc_info=True)
        return {"url": url, "error": str(e), "metadata": {"status": "error"}}


@tool(description="Extract a PDF at a URL as markdown, one section per page; image-only pages go through vision OCR")
async def pdf_extract(url: str, max_pages: int = 50, vision: bool = True) -> Dict[str, Any]:
    """Fetch a PDF and return its text as markdown with a ## Page N section per page.

    Args:
        url: PDF URL
        max_pages: Stop after this many pages (default: 50)
        vision: OCR image-only pages with the configured vision provider (default: True)

    Returns:
        Dict with title, markdown, per-page sources and extraction metadata
    """
    from app import pdf as pdfx
    from app.config import settings

    try:
        crawler = await get_crawler_engine()
        fetched = await crawler._fetch_pdf_bytes(url)
        if fetched is None:
            return {"url": url, "error": "URL did not return a PDF", "metadata": {"status": "failed"}}
        extraction = await pdfx.extract(
            fetched["content"],
            vision_provider=crawler._pdf_vision_provider() if vision else None,
            max_pages=max_pages,
            min_text_chars=settings.pdf_min_text_chars,
            dpi=settings.pdf_render_dpi,
            max_vision_pages=settings.pdf_vision_max_pages,
            vision_concurrency=settings.pdf_vision_concurrency,
        )
        if not extraction.title:
            extraction.title = pdfx.title_from_url(url)
        return {
            "url": url,
            "title": extraction.title,
            "markdown": extraction.to_markdown(),
            "pages": [{"number": p.number, "source": p.source, "char_count": p.char_count} for p in extraction.pages],
            "metadata": {**extraction.summary(), "size_bytes": len(fetched["content"]), "status": "success"},
        }
    except Exception as e:
        logger.error(f"Error in pdf_extract tool: {e}", exc_info=True)
        return {"url": url, "error": str(e), "metadata": {"status": "error"}}
