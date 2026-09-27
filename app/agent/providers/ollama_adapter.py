"""Ollama provider adapter.

Communicates with a local Ollama server over HTTP. Maps Ollama's
tool_calls response format to the agent's AssistantAction primitives.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.agent.providers.base import LLMAdapter
from app.agent.types import AssistantAction, Respond, ToolCall, ToolCalls

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1:8b-instruct"
DEFAULT_BASE_URL = "http://localhost:11434"
TIMEOUT_S = 120
DEFAULT_VISION_MODEL = "benhaotang/Nanonets-OCR-s:latest"
VISION_TIMEOUT_S = 180


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama local inference with tool calling."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        vision_model: Optional[str] = None,
        keep_alive: Optional[str] = None,
        num_ctx: Optional[int] = None,
        vision_timeout_s: Optional[float] = None,
        **kwargs,
    ):
        self.model = model or DEFAULT_MODEL
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or None  # only a hosted Ollama needs one
        self._vision_model = vision_model
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._vision_timeout_s = vision_timeout_s

    def _setting(self, name: str, default):
        """Read an app setting lazily so the adapter stays usable without app.config."""
        try:
            from app.config import settings
            value = getattr(settings, name, None)
        except Exception:
            value = None
        return default if value in (None, "") else value

    @property
    def vision_model(self) -> str:
        return self._vision_model or self._setting("ollama_vision_model", DEFAULT_VISION_MODEL)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # ------------------------------------------------------------------
    # complete
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AssistantAction:
        """Call Ollama /api/chat and normalize the response."""
        ollama_messages = _convert_messages(messages)
        ollama_tools = _convert_tools(tools) if tools else None

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        async with httpx.AsyncClient(timeout=TIMEOUT_S, headers=self._headers()) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})

        # Tool calls path
        tool_calls_raw = msg.get("tool_calls", [])
        if tool_calls_raw:
            calls = []
            for i, tc in enumerate(tool_calls_raw):
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append(ToolCall(
                    id=f"ollama_{i}",
                    name=fn.get("name", ""),
                    args=args,
                ))
            return ToolCalls(calls=calls)

        # Text response path
        text = msg.get("content", "")
        return Respond(text=text)

    # ------------------------------------------------------------------
    # vision — multimodal models (Nanonets-OCR-s, qwen2.5-vl, llava, ...)
    # ------------------------------------------------------------------

    async def vision(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        detail: str = "low",
    ) -> str:
        """Send an image to the Ollama vision model through the native /api/chat.

        The native API (not /v1) is used because it accepts ``keep_alive`` and
        ``options``. ``detail`` is accepted for protocol compatibility only;
        Ollama has no equivalent, image size is controlled by the caller.
        Settings: OLLAMA_VISION_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_NUM_CTX,
        OLLAMA_VISION_TIMEOUT_S, OLLAMA_API_KEY (hosted Ollama only).
        """
        import base64

        b64 = base64.b64encode(image_bytes).decode("ascii")
        keep_alive = self._keep_alive or self._setting("ollama_keep_alive", "5m")
        num_ctx = int(self._num_ctx or self._setting("ollama_num_ctx", 8192))
        timeout_s = float(self._vision_timeout_s or self._setting("ollama_vision_timeout_s", VISION_TIMEOUT_S))
        payload = {
            "model": self.vision_model,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0.01, "num_predict": 4096, "num_ctx": num_ctx},
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        }
        async with httpx.AsyncClient(timeout=timeout_s, headers=self._headers()) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return (data.get("message") or {}).get("content", "") or ""

    async def unload_vision(self) -> None:
        """Ask Ollama to drop the vision model now (keep_alive 0) so a shared GPU gets its memory back."""
        payload = {"model": self.vision_model, "keep_alive": 0}
        try:
            async with httpx.AsyncClient(timeout=15, headers=self._headers()) as client:
                await client.post(f"{self.base_url}/api/generate", json=payload)
        except Exception as exc:
            logger.debug("Ollama unload of %s failed: %s", self.vision_model, exc)


# ---------------------------------------------------------------------------
# Message / tool schema conversion
# ---------------------------------------------------------------------------

def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert internal message format to Ollama format."""
    ollama_msgs = []
    for msg in messages:
        role = msg.get("role", "user")

        # Tool result → Ollama tool message
        if role == "tool":
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            ollama_msgs.append({
                "role": "tool",
                "content": content,
            })
            continue

        # Assistant with tool calls
        if role == "assistant" and "tool_calls" in msg:
            tc_list = []
            for tc in msg["tool_calls"]:
                tc_list.append({
                    "function": {
                        "name": tc["name"],
                        "arguments": tc.get("args", {}),
                    }
                })
            entry: Dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                entry["content"] = msg["content"]
            entry["tool_calls"] = tc_list
            ollama_msgs.append(entry)
            continue

        # Standard messages
        ollama_msgs.append({
            "role": role,
            "content": msg.get("content", ""),
        })

    return ollama_msgs


def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert internal tool schemas to Ollama tool format."""
    ollama_tools = []
    for t in tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return ollama_tools
