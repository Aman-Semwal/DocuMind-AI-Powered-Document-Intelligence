"""LLM interface and Ollama implementation (async streaming, retries, concurrency cap)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.core.exceptions import LLMTimeout, OllamaUnavailable


class _ModelNotInstalled(OllamaUnavailable):
    """Internal signal used to retry Deep mode with the configured default model."""


class LLMClient(Protocol):
    model_name: str

    async def chat(
        self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None
    ) -> str: ...

    def stream(
        self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None
    ) -> AsyncIterator[str]: ...

    def chat_sync(
        self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None
    ) -> str: ...

    async def warm_up(self) -> None: ...

    async def ping(self) -> bool: ...

    async def aclose(self) -> None: ...


class OllamaLLM:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.model_name = settings.llm_model
        timeout = httpx.Timeout(settings.llm_timeout_s, connect=5.0)
        limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
        self._client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout, limits=limits)
        self._sync = httpx.Client(base_url=settings.ollama_base_url, timeout=timeout, limits=limits)
        self._sem = asyncio.Semaphore(max(1, settings.llm_concurrency))

    # ------------------------------------------------------------------ helpers
    def _payload(
        self, messages: list[dict[str, str]], deep: bool, stream: bool, max_tokens: int | None
    ) -> dict[str, Any]:
        return {
            "model": self.s.deep_llm_model if deep else self.s.llm_model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.s.ollama_keep_alive,
            "options": {
                "temperature": self.s.llm_temperature,
                "seed": self.s.llm_seed,
                "num_ctx": self.s.num_ctx,
                "num_predict": max_tokens or self.s.num_predict,
            },
        }

    @staticmethod
    def _check_status(resp: httpx.Response, model: str) -> None:
        if resp.status_code == 404:
            raise _ModelNotInstalled(f"Model '{model}' is not installed. Run: ollama pull {model}")
        if resp.status_code >= 500:
            raise OllamaUnavailable()
        resp.raise_for_status()

    # --------------------------------------------------------------------- chat
    async def chat(self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None) -> str:
        payload = self._payload(messages, deep, False, max_tokens)
        try:
            async with self._sem:
                last: Exception | None = None
                for attempt in range(self.s.llm_retries + 1):
                    try:
                        resp = await self._client.post("/api/chat", json=payload)
                        self._check_status(resp, payload["model"])
                        return str(resp.json()["message"]["content"]).strip()
                    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                        last = exc
                        await asyncio.sleep(0.5 * (2**attempt))
                    except httpx.ReadTimeout as exc:
                        raise LLMTimeout() from exc
                    except httpx.HTTPError as exc:
                        raise OllamaUnavailable() from exc
                raise OllamaUnavailable() from last
        except _ModelNotInstalled:
            if deep and self.s.deep_llm_model != self.s.llm_model:
                return await self.chat(messages, deep=False, max_tokens=max_tokens)
            raise

    async def stream(
        self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, deep, True, max_tokens)
        streamed_any = False
        try:
            async with self._sem:
                last: Exception | None = None
                for attempt in range(self.s.llm_retries + 1):
                    try:
                        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                            if resp.status_code >= 400:
                                await resp.aread()
                                self._check_status(resp, payload["model"])
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                data = json.loads(line)
                                if "error" in data:
                                    error = str(data["error"])
                                    if "model" in error.lower() and any(
                                        term in error.lower() for term in ("not found", "not installed")
                                    ):
                                        raise _ModelNotInstalled(error)
                                    raise OllamaUnavailable()
                                piece = data.get("message", {}).get("content", "")
                                if piece:
                                    streamed_any = True
                                    yield piece
                                if data.get("done"):
                                    return
                            return
                    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                        last = exc  # safe to retry: nothing was streamed yet
                        await asyncio.sleep(0.5 * (2**attempt))
                    except httpx.ReadTimeout as exc:
                        raise LLMTimeout() from exc
                    except httpx.HTTPError as exc:
                        raise OllamaUnavailable() from exc
                raise OllamaUnavailable() from last
        except _ModelNotInstalled:
            if deep and not streamed_any and self.s.deep_llm_model != self.s.llm_model:
                async for piece in self.stream(messages, deep=False, max_tokens=max_tokens):
                    yield piece
                return
            raise

    def chat_sync(self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None) -> str:
        """Blocking variant for the ingestion worker thread (summaries)."""
        payload = self._payload(messages, deep, False, max_tokens)
        last: Exception | None = None
        try:
            for attempt in range(self.s.llm_retries + 1):
                try:
                    resp = self._sync.post("/api/chat", json=payload)
                    self._check_status(resp, payload["model"])
                    return str(resp.json()["message"]["content"]).strip()
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last = exc
                    time.sleep(0.5 * (2**attempt))
                except httpx.ReadTimeout as exc:
                    raise LLMTimeout() from exc
                except httpx.HTTPError as exc:
                    raise OllamaUnavailable() from exc
            raise OllamaUnavailable() from last
        except _ModelNotInstalled:
            if deep and self.s.deep_llm_model != self.s.llm_model:
                return self.chat_sync(messages, deep=False, max_tokens=max_tokens)
            raise

    # ------------------------------------------------------------------- admin
    async def warm_up(self) -> None:
        """Load the model into VRAM with a tiny prompt so the first real request is fast."""
        with contextlib.suppress(httpx.HTTPError):  # Ollama may not be running yet; /ready reports it
            await self._client.post(
                "/api/chat",
                json={
                    "model": self.s.llm_model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream": False,
                    "keep_alive": self.s.ollama_keep_alive,
                    "options": {"num_predict": 1, "num_ctx": self.s.num_ctx},
                },
                timeout=httpx.Timeout(180.0, connect=5.0),
            )

    async def ping(self) -> bool:
        try:
            return (await self._client.get("/api/tags", timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
        self._sync.close()
