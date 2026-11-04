import httpx
import pytest
import respx

from frontend.services.api_client import ApiClient, ApiError, ApiOffline, _payload, parse_sse

BASE = "http://api.test"


@pytest.fixture
def client() -> ApiClient:
    return ApiClient(BASE, "k", retries=1)


def sse(*events: tuple[str, str]) -> str:
    return "".join(f"event: {e}\ndata: {d}\n\n" for e, d in events)


@respx.mock
def test_list_documents_and_headers(client):
    route = respx.get(f"{BASE}/documents").mock(
        return_value=httpx.Response(200, json={"documents": [{"a": 1}], "count": 1})
    )
    assert client.list_documents() == [{"a": 1}]
    assert route.calls[0].request.headers["x-api-key"] == "k"


@respx.mock
def test_upload_returns_document(client):
    respx.post(f"{BASE}/documents/upload").mock(return_value=httpx.Response(202, json={"document_id": "DOC001"}))
    assert client.upload("a.pdf", b"%PDF-1")["document_id"] == "DOC001"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": {"code": "unauthorized", "message": "Missing or invalid API key."}}, "invalid API key"),
        (413, {"error": {"code": "file_too_large", "message": "The file exceeds the 100 MB limit."}}, "100 MB"),
        (409, {"error": {"code": "duplicate_document", "message": "already uploaded as DOC001."}}, "DOC001"),
        (503, None, "Ollama"),
        (504, None, "too long"),
        (429, None, "Too many"),
        (500, None, "failed"),
    ],
)
@respx.mock
def test_error_statuses_become_friendly_errors(client, status, body, expected):
    respx.post(f"{BASE}/ask").mock(
        return_value=httpx.Response(status, json=body) if body else httpx.Response(status, text="<html>boom</html>")
    )
    with pytest.raises(ApiError) as exc:
        client.ask("q")
    assert exc.value.status == status
    assert expected in exc.value.message
    assert "Traceback" not in exc.value.message and "<html>" not in exc.value.message


@respx.mock
def test_timeout_is_reported(client):
    respx.post(f"{BASE}/ask").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ApiError) as exc:
        client.ask("q")
    assert exc.value.status == 504


@respx.mock
def test_offline_after_retries_for_get(client):
    route = respx.get(f"{BASE}/ready").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(ApiOffline):
        client.ready()
    assert route.call_count == 2  # one retry for idempotent GET


@respx.mock
def test_post_is_not_retried(client):
    route = respx.post(f"{BASE}/ask").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(ApiOffline):
        client.ask("q")
    assert route.call_count == 1


@respx.mock
def test_stream_events_in_order(client):
    body = sse(("token", '{"t": "Hel"}'), ("token", '{"t": "lo"}'), ("final", '{"answer": "Hello"}'))
    respx.post(f"{BASE}/ask/stream").mock(return_value=httpx.Response(200, text=body))
    events = list(client.ask_stream("q", deep=True))
    assert [e.event for e in events] == ["token", "token", "final"]
    assert events[-1].data["answer"] == "Hello"


@respx.mock
def test_stream_error_event_raises(client):
    body = sse(
        ("token", '{"t": "x"}'), ("error", '{"code": "llm_timeout", "message": "The language model took too long."}')
    )
    respx.post(f"{BASE}/ask/stream").mock(return_value=httpx.Response(200, text=body))
    with pytest.raises(ApiError) as exc:
        list(client.ask_stream("q"))
    assert exc.value.code == "llm_timeout"


@respx.mock
def test_stream_http_error_before_first_event(client):
    respx.post(f"{BASE}/ask/stream").mock(
        return_value=httpx.Response(503, json={"error": {"code": "llm_unavailable", "message": "Ollama is down."}})
    )
    with pytest.raises(ApiError) as exc:
        list(client.ask_stream("q"))
    assert exc.value.status == 503


@respx.mock
def test_stream_malformed_json_event(client):
    respx.post(f"{BASE}/ask/stream").mock(return_value=httpx.Response(200, text="event: token\ndata: {oops\n\n"))
    with pytest.raises(ApiError) as exc:
        list(client.ask_stream("q"))
    assert exc.value.code == "malformed_stream"


@respx.mock
def test_stream_connection_failure(client):
    respx.post(f"{BASE}/ask/stream").mock(side_effect=httpx.ConnectError("x"))
    with pytest.raises(ApiOffline):
        list(client.ask_stream("q"))


def test_parse_sse_handles_comments_multiline_and_trailing_event():
    lines = [": keep-alive", "event: token", 'data: {"t":', 'data: "a"}', "", "event: final", 'data: {"x": 1}']
    events = list(parse_sse(lines))
    assert events[0].data == {"t": "a"} and events[1].event == "final"


def test_parse_sse_rejects_non_object():
    with pytest.raises(ApiError):
        list(parse_sse(["data: [1]", ""]))


def test_payload_maps_deep_and_drops_empty():
    p = _payload("q", {"deep": True, "document_ids": [], "top_k": 5, "session_id": None})
    assert p == {"question": "q", "deep_mode": True, "top_k": 5}


@respx.mock
def test_feedback_maps_rating(client):
    route = respx.post(f"{BASE}/feedback").mock(return_value=httpx.Response(200, json={}))
    client.feedback("abc", -1, "meh")
    assert route.calls[0].request.content == b'{"question_id":"abc","rating":"down","comment":"meh"}'


@respx.mock
def test_misc_endpoints(client):
    respx.get(f"{BASE}/documents/DOC001/pages/2/preview").mock(return_value=httpx.Response(200, content=b"PNG"))
    respx.delete(f"{BASE}/documents/DOC001").mock(return_value=httpx.Response(200, json={}))
    respx.post(f"{BASE}/documents/DOC001/resume").mock(return_value=httpx.Response(202, json={"ok": 1}))
    respx.get(f"{BASE}/metrics/recent").mock(return_value=httpx.Response(200, json={"recent": []}))
    assert client.preview("DOC001", 2) == b"PNG"
    client.delete_document("DOC001")
    assert client.resume_document("DOC001") == {"ok": 1}
    assert client.metrics() == {"recent": []}
    client.close()
