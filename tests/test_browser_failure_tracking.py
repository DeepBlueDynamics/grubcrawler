"""
Tests for BrowserEngine consecutive failure tracking, proxy restart logic,
and exit IP verification.

Verifies that:
- Consecutive timeout failures are tracked
- After N consecutive failures, browser restarts with a fresh proxy session
- Successful navigation resets the failure counter
- Proxy errors (NS_ERROR_PROXY_BAD_GATEWAY) trigger immediate restart
- _check_exit_ip logs the proxy exit IP on success
- _check_exit_ip is non-fatal on failure
- _check_exit_ip always cleans up the disposable page

Created: 2026-02-28
Updated: 2026-03-01 — proxy error detection + immediate restart
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.browser import BrowserEngine
from app.config import Settings


class TestConsecutiveFailureTracking:
    """Test _consecutive_failures counter initialization and behavior."""

    def test_initial_failure_count_is_zero(self):
        engine = BrowserEngine()
        assert engine._consecutive_failures == 0

    def test_max_failures_from_settings(self):
        engine = BrowserEngine()
        assert engine._max_failures_before_restart == Settings(_env_file=None).proxy_restart_after_failures

    def test_default_max_failures_is_3(self):
        engine = BrowserEngine()
        assert engine._max_failures_before_restart == 3


class TestRestartWithFreshProxy:
    """Test _restart_with_fresh_proxy resets counter and calls close + start."""

    @pytest.mark.asyncio
    async def test_restart_calls_close_and_start(self):
        engine = BrowserEngine()
        engine._consecutive_failures = 5

        engine.close = AsyncMock()
        engine.start_browser = AsyncMock()

        await engine._restart_with_fresh_proxy()

        engine.close.assert_awaited_once()
        engine.start_browser.assert_awaited_once()
        assert engine._consecutive_failures == 0


class TestFailureCounterInCrawl:
    """Test that crawl_with_context increments/resets _consecutive_failures."""

    def _make_engine_with_mocks(self):
        """Create a BrowserEngine with mocked browser internals."""
        engine = BrowserEngine()
        engine._consecutive_failures = 0
        engine._max_failures_before_restart = 3

        # Mock browser as started
        engine.browser = MagicMock()
        engine.browser.is_connected.return_value = True

        return engine

    @pytest.mark.asyncio
    async def test_success_resets_counter(self):
        """A successful navigation should reset _consecutive_failures to 0."""
        engine = self._make_engine_with_mocks()
        engine._consecutive_failures = 2

        # Mock the entire crawl path
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200, headers={}))
        mock_page.content = AsyncMock(return_value="<html>real content</html>")
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.url = "https://example.com"
        mock_page.evaluate = AsyncMock(return_value={"text": "real", "char_count": 4, "word_count": 1})
        mock_page.viewport_size = {"width": 1280, "height": 800}
        mock_page.wait_for_selector = AsyncMock()
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))

        # Mock challenge solver to return no challenge
        with patch('app.challenge_solver.resolve_challenge', new_callable=AsyncMock) as mock_challenge:
            mock_challenge.return_value = MagicMock(
                resolved=True, method="none", wait_time_ms=0,
                challenge_type=MagicMock(value="none")
            )
            # Mock proxy pool
            with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
                mock_pool.return_value = MagicMock()

                content, page_info, screenshot = await engine.crawl_with_context(
                    "https://example.com", timeout=5000
                )

        assert engine._consecutive_failures == 0
        assert content == "<html>real content</html>"

    @pytest.mark.asyncio
    async def test_timeout_increments_counter(self):
        """A timeout failure should increment _consecutive_failures."""
        engine = self._make_engine_with_mocks()
        engine._consecutive_failures = 0
        engine._max_failures_before_restart = 5  # Won't trigger restart

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=TimeoutError("Navigation timeout"))
        mock_page.title = AsyncMock(return_value="")
        mock_page.content = AsyncMock(return_value="")
        mock_page.url = "https://example.com"
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))

        # Mock challenge solver to fail
        with patch('app.challenge_solver.resolve_challenge', new_callable=AsyncMock) as mock_challenge:
            mock_challenge.return_value = MagicMock(resolved=False, method="none")
            with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
                mock_pool.return_value = MagicMock()

                with pytest.raises(TimeoutError):
                    await engine.crawl_with_context("https://example.com", timeout=1000)

        # Should have been incremented (2 retries * 1 timeout each)
        assert engine._consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_restart_triggered_after_max_failures(self):
        """After max consecutive failures, _restart_with_fresh_proxy is called."""
        engine = self._make_engine_with_mocks()
        engine._consecutive_failures = 2  # Already at threshold - 1
        engine._max_failures_before_restart = 3

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=TimeoutError("Navigation timeout"))
        mock_page.title = AsyncMock(return_value="")
        mock_page.content = AsyncMock(return_value="")
        mock_page.url = "https://example.com"
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))
        engine._restart_with_fresh_proxy = AsyncMock()

        with patch('app.challenge_solver.resolve_challenge', new_callable=AsyncMock) as mock_challenge:
            mock_challenge.return_value = MagicMock(resolved=False, method="none")
            with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
                mock_pool.return_value = MagicMock()

                with pytest.raises(TimeoutError):
                    await engine.crawl_with_context("https://example.com", timeout=1000)

        engine._restart_with_fresh_proxy.assert_awaited()


class TestProxyErrorHandling:
    """Test that proxy errors (NS_ERROR_PROXY_BAD_GATEWAY, etc.) trigger immediate restart."""

    def _make_engine_with_mocks(self):
        """Create a BrowserEngine with mocked browser internals."""
        engine = BrowserEngine()
        engine._consecutive_failures = 0
        engine._max_failures_before_restart = 3
        engine.browser = MagicMock()
        engine.browser.is_connected.return_value = True
        return engine

    @pytest.mark.asyncio
    async def test_proxy_bad_gateway_triggers_restart(self):
        """NS_ERROR_PROXY_BAD_GATEWAY should immediately restart browser."""
        engine = self._make_engine_with_mocks()
        engine._consecutive_failures = 0  # Even at 0, proxy error = immediate restart

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=Exception("Page.goto: NS_ERROR_PROXY_BAD_GATEWAY\n  - navigating to \"https://capterra.com\"")
        )
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))
        engine._restart_with_fresh_proxy = AsyncMock()

        with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
            mock_pool.return_value = MagicMock()

            with pytest.raises(Exception, match="NS_ERROR_PROXY_BAD_GATEWAY"):
                await engine.crawl_with_context("https://capterra.com", timeout=5000)

        engine._restart_with_fresh_proxy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_proxy_connection_refused_triggers_restart(self):
        """NS_ERROR_PROXY_CONNECTION_REFUSED should immediately restart browser."""
        engine = self._make_engine_with_mocks()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=Exception("Page.goto: NS_ERROR_PROXY_CONNECTION_REFUSED")
        )
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))
        engine._restart_with_fresh_proxy = AsyncMock()

        with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
            mock_pool.return_value = MagicMock()

            with pytest.raises(Exception, match="NS_ERROR_PROXY_CONNECTION_REFUSED"):
                await engine.crawl_with_context("https://capterra.com", timeout=5000)

        engine._restart_with_fresh_proxy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_proxy_error_increments_failure_counter(self):
        """Proxy errors should increment _consecutive_failures."""
        engine = self._make_engine_with_mocks()
        engine._consecutive_failures = 0

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=Exception("Page.goto: NS_ERROR_PROXY_BAD_GATEWAY")
        )
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))
        engine._restart_with_fresh_proxy = AsyncMock()

        with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
            mock_pool.return_value = MagicMock()

            with pytest.raises(Exception):
                await engine.crawl_with_context("https://capterra.com", timeout=5000)

        assert engine._consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_proxy_error_skips_further_retries(self):
        """Proxy errors should not retry with the same dead proxy — restart and raise immediately."""
        engine = self._make_engine_with_mocks()

        call_count = 0

        async def counting_goto(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Page.goto: NS_ERROR_PROXY_BAD_GATEWAY")

        mock_page = AsyncMock()
        mock_page.goto = counting_goto
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))
        engine._restart_with_fresh_proxy = AsyncMock()

        with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
            mock_pool.return_value = MagicMock()

            with pytest.raises(Exception, match="NS_ERROR_PROXY_BAD_GATEWAY"):
                await engine.crawl_with_context("https://capterra.com", timeout=5000)

        # Should only attempt navigation ONCE — no retry with dead proxy
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_non_proxy_error_does_not_trigger_restart(self):
        """Regular errors (not proxy, not timeout) should not trigger restart."""
        engine = self._make_engine_with_mocks()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=Exception("Page.goto: net::ERR_NAME_NOT_RESOLVED")
        )
        mock_page.is_closed = MagicMock(return_value=False)

        mock_context = AsyncMock()
        mock_context.close = AsyncMock()

        engine.create_isolated_context = AsyncMock(return_value=(mock_context, mock_page))
        engine._restart_with_fresh_proxy = AsyncMock()

        with patch('app.proxy_pool.get_proxy_pool') as mock_pool:
            mock_pool.return_value = MagicMock()

            with pytest.raises(Exception, match="ERR_NAME_NOT_RESOLVED"):
                await engine.crawl_with_context("https://capterra.com", timeout=5000)

        engine._restart_with_fresh_proxy.assert_not_awaited()


class TestCheckExitIp:
    """_check_exit_ip runs out-of-band via httpx and never touches the browser.

    Opening and tearing down a browser context as the first operation after
    launch wedged Camoufox networking (first crawl loaded only the document),
    so the check must not create contexts or pages.
    """

    @staticmethod
    def _mock_httpx(response=None, side_effect=None):
        client = MagicMock()
        client.get = AsyncMock(return_value=response, side_effect=side_effect)
        client_cls = MagicMock()
        client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        return client_cls, client

    @pytest.mark.asyncio
    async def test_logs_exit_ip_on_success(self, caplog):
        """Successful IP check logs the exit IP and creates no browser context."""
        engine = BrowserEngine()
        engine.browser = AsyncMock()
        engine.browser.new_context = AsyncMock()

        resp = MagicMock(status_code=200, text='{"origin": "185.123.45.67"}')
        resp.json = MagicMock(return_value={"origin": "185.123.45.67"})
        client_cls, client = self._mock_httpx(response=resp)

        with caplog.at_level("INFO"), patch("httpx.AsyncClient", client_cls):
            await engine._check_exit_ip()

        client.get.assert_awaited_once_with("https://httpbin.org/ip")
        client_cls.assert_called_once_with(proxy=None, timeout=10.0)
        engine.browser.new_context.assert_not_awaited()
        assert "Proxy exit IP: 185.123.45.67" in caplog.text

    @pytest.mark.asyncio
    async def test_uses_browser_proxy(self):
        """The check goes through the same proxy the browser was launched with."""
        engine = BrowserEngine()
        engine.browser = AsyncMock()
        engine._proxy_config = {
            "server": "http://gate.example.com:7000",
            "username": "user-abc",
            "password": "p@ss word",
        }
        resp = MagicMock(status_code=200, text='{"origin": "1.2.3.4"}')
        resp.json = MagicMock(return_value={"origin": "1.2.3.4"})
        client_cls, client = self._mock_httpx(response=resp)

        with patch("httpx.AsyncClient", client_cls):
            await engine._check_exit_ip()

        client_cls.assert_called_once_with(
            proxy="http://user-abc:p%40ss%20word@gate.example.com:7000", timeout=10.0
        )

    @pytest.mark.asyncio
    async def test_skips_socks_proxy(self, caplog):
        """SOCKS proxies are skipped (httpx needs socksio) rather than failing."""
        engine = BrowserEngine()
        engine.browser = AsyncMock()
        engine._proxy_config = {"server": "socks5://gate.example.com:1080"}
        client_cls, client = self._mock_httpx()

        with caplog.at_level("INFO"), patch("httpx.AsyncClient", client_cls):
            await engine._check_exit_ip()

        client_cls.assert_not_called()
        assert "IP check skipped" in caplog.text

    @pytest.mark.asyncio
    async def test_nonfatal_on_timeout(self, caplog):
        """IP check failure doesn't raise and never touches the browser."""
        engine = BrowserEngine()
        engine.browser = AsyncMock()
        engine.browser.new_context = AsyncMock()
        client_cls, client = self._mock_httpx(side_effect=TimeoutError("IP check timed out"))

        with caplog.at_level("WARNING"), patch("httpx.AsyncClient", client_cls):
            await engine._check_exit_ip()  # Should NOT raise

        engine.browser.new_context.assert_not_awaited()
        assert "IP check failed (non-fatal)" in caplog.text

    @pytest.mark.asyncio
    async def test_noop_when_no_browser(self):
        """IP check is a no-op when browser is None."""
        engine = BrowserEngine()
        engine.browser = None
        client_cls, client = self._mock_httpx()

        with patch("httpx.AsyncClient", client_cls):
            await engine._check_exit_ip()

        client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonfatal_on_malformed_proxy_config(self, caplog):
        """A proxy config that cannot be turned into a URL must not abort startup."""
        engine = BrowserEngine()
        engine.browser = AsyncMock()
        engine._proxy_config = {"server": "http://gate.example.com:7000", "username": object()}
        client_cls, client = self._mock_httpx()

        with caplog.at_level("WARNING"), patch("httpx.AsyncClient", client_cls):
            await engine._check_exit_ip()  # Should NOT raise

        client_cls.assert_not_called()
        assert "IP check failed (non-fatal)" in caplog.text
