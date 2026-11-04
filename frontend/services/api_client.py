"""HTTP client for the backend REST API (the UI never touches the database or the models).

* one pooled httpx client, short timeouts, retries only for idempotent GET requests
* friendly `ApiError`s - never a raw stack trace, never the API key
* a small Server-Sent-Events parser for /ask/stream
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx


class ApiError(Exception):
    """An error the UI can show to the user as is."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class ApiOffline(ApiError):
    def __init__(self, message: str = "The backend API is not reachable. Is it running?") -> None:
        super().__init__(0, "offline", message)


@dataclass(frozen=True)
class StreamEvent:
    event: str
    data: dict[str, Any]


def parse_sse(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Parse text/event-stream lines into events. Malformed JSON raises ApiError."""
    event, data_lines = "message", []

    def build() -> StreamEvent:
        try:
            payload = json.loads("\n".join(data_lines))
        except ValueError as exc:
            raise ApiError(502, "malformed_stream", "The server sent an unreadable response.") from exc
        if not isinstance(payload, dict):
            raise ApiError(502, "malformed_stream", "The server sent an unreadable response.")
        return StreamEvent(event, payload)

    for line in lines:
        if line == "":
            if data_lines:
                yield build()
            event, data_lines = "message", []
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "event":
            event = value
        elif name == "data":
            data_lines.append(value)
    if data_lines:
        yield build()


def _error_from(response: httpx.Response) -> ApiError:
    try:
        err = response.json()["error"]
        return ApiError(response.status_code, str(err.get("code", "error")), str(err.get("message", "Request failed.")))
    except (ValueError, KeyError, TypeError, AttributeError):
        fallback = {
            401: "The API key was rejected. Check API_KEY in your settings.",
            413: "The file is too large.",
            429: "Too many requests. Please wait a moment.",
            503: "The language model service is unavailable. Is Ollama running?",
            504: "The language model took too long to answer.",
        }
        return ApiError(response.status_code, "http_error", fallback.get(response.status_code, "The request failed."))


def _payload(question: str, options: dict[str, Any]) -> dict[str, Any]:
    """Build the /ask body. The UI says `deep`, the API says `deep_mode`; empty values are dropped."""
    opts = dict(options)
    if "deep" in opts:
        opts["deep_mode"] = bool(opts.pop("deep"))
    return {"question": question, **{k: v for k, v in opts.items() if v not in (None, [], "")}}


class ApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        stream_timeout: float = 180.0,
        retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.stream_timeout = stream_timeout
        self.retries = retries
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=3.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._headers = {"X-API-Key": api_key}

    # ------------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str, *, retry: bool = False, **kwargs: Any) -> httpx.Response:
        attempts = self.retries + 1 if retry else 1
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.request(method, path, headers=self._headers, **kwargs)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(0.2 * 2**attempt)
                continue
            except httpx.ReadTimeout as exc:
                raise ApiError(504, "timeout", "The server took too long to respond.") from exc
            except httpx.HTTPError as exc:
                raise ApiError(502, "network_error", "A network error occurred.") from exc
            if resp.status_code >= 400:
                raise _error_from(resp)
            return resp
        raise ApiOffline() from last

    # ------------------------------------------------------------------ endpoints
    def ready(self) -> dict[str, Any]:
        resp = self._request("GET", "/ready", retry=True)
        return resp.json()  # type: ignore[no-any-return]

    def list_documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents", retry=True).json()["documents"]  # type: ignore[no-any-return]

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        resp = self._request("POST", "/documents/upload", files={"file": (filename, data, "application/pdf")})
        return resp.json()  # type: ignore[no-any-return]

    def delete_document(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}")

    def resume_document(self, doc_id: str) -> dict[str, Any]:
        return self._request("POST", f"/documents/{doc_id}/resume").json()  # type: ignore[no-any-return]

    def preview(self, doc_id: str, page: int) -> bytes:
        return self._request("GET", f"/documents/{doc_id}/pages/{page}/preview", retry=True).content

    def ask(self, question: str, **options: Any) -> dict[str, Any]:
        payload = _payload(question, options)
        return self._request("POST", "/ask", json=payload, timeout=self.stream_timeout).json()  # type: ignore[no-any-return]

    def ask_stream(self, question: str, **options: Any) -> Iterator[StreamEvent]:
        """Yield `token` events and finally a `final` event (the full AskResponse)."""
        payload = _payload(question, options)
        try:
            with self._client.stream(
                "POST",
                "/ask/stream",
                json=payload,
                headers=self._headers,
                timeout=httpx.Timeout(self.stream_timeout, connect=3.0),
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise _error_from(resp)
                for event in parse_sse(resp.iter_lines()):
                    if event.event == "error":
                        d = event.data
                        raise ApiError(
                            int(d.get("status", 500)),
                            str(d.get("code", "error")),
                            str(d.get("message", "Something went wrong.")),
                        )
                    yield event
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ApiOffline() from exc
        except httpx.ReadTimeout as exc:
            raise ApiError(504, "timeout", "The answer took too long. Try again or turn off deep mode.") from exc
        except httpx.HTTPError as exc:
            raise ApiError(502, "network_error", "The connection was interrupted.") from exc

    def feedback(self, question_id: str, rating: int, comment: str | None = None) -> None:
        body: dict[str, Any] = {"question_id": question_id, "rating": "up" if rating > 0 else "down"}
        if comment:
            body["comment"] = comment
        self._request("POST", "/feedback", json=body)

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics/recent", retry=True).json()  # type: ignore[no-any-return]

    def close(self) -> None:
        self._client.close()
