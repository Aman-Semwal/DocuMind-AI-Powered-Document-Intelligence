"""Security tests: auth, limits, file handling, prompt injection, error hygiene."""

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import LLMTimeout, OllamaUnavailable
from app.main import create_app
from app.services.fakes import FakeEmbedder, FakeLLM
from tests.conftest import API_KEY, HEADERS, make_settings, upload

PROTECTED = [
    ("GET", "/documents"),
    ("GET", "/documents/DOC001"),
    ("DELETE", "/documents/DOC001"),
    ("POST", "/documents/DOC001/resume"),
    ("GET", "/documents/DOC001/summary"),
    ("GET", "/documents/DOC001/pages/1/preview"),
    ("POST", "/ask"),
    ("POST", "/ask/stream"),
    ("POST", "/feedback"),
    ("GET", "/metrics/recent"),
    ("POST", "/documents/upload"),
]


def make_client(tmp_path, llm=None, raise_exc=True, **kw):
    s = make_settings(tmp_path, **kw)
    return TestClient(create_app(s, embedder=FakeEmbedder(), llm=llm or FakeLLM()), raise_server_exceptions=raise_exc)


@pytest.mark.parametrize("method,path", PROTECTED)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": API_KEY + "x"}])
def test_missing_or_wrong_key_gives_401(tmp_path, method, path, headers):
    with make_client(tmp_path) as c:
        r = c.request(method, path, headers=headers, json={"question": "hello there"} if method == "POST" else None)
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_health_is_public_and_docs_load(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/docs").status_code == 200
        spec = c.get("/openapi.json").json()
        assert "APIKeyHeader" in spec["components"]["securitySchemes"]
        assert {"/documents/upload", "/ask", "/documents", "/ask/stream"} <= set(spec["paths"])


def test_no_keys_configured_rejects_everything(tmp_path):
    with make_client(tmp_path, api_keys="") as c:
        assert c.get("/documents", headers=HEADERS).status_code == 401


def test_oversized_file_413_and_partial_file_removed(tmp_path):
    with make_client(tmp_path, max_file_mb=1) as c:
        big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
        r = c.post(
            "/documents/upload", headers=HEADERS, files={"file": ("big.pdf", io.BytesIO(big), "application/pdf")}
        )
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
        assert list(c.app.state.container.settings.upload_dir.iterdir()) == []


@pytest.mark.parametrize(
    "name,ctype,content",
    [
        ("notes.txt", "text/plain", b"hello"),
        ("doc.pdf", "text/plain", b"%PDF-1.4 hi"),
        ("fake.pdf", "application/pdf", b"MZ\x90\x00 this is an executable"),
        ("empty.pdf", "application/pdf", b""),
        ("image.png.pdf", "application/pdf", b"\x89PNG\r\n\x1a\n" + b"0" * 100),
    ],
)
def test_wrong_type_or_disguised_file_415(tmp_path, name, ctype, content):
    with make_client(tmp_path) as c:
        r = c.post("/documents/upload", headers=HEADERS, files={"file": (name, io.BytesIO(content), ctype)})
        assert r.status_code == 415, r.text
        assert list(c.app.state.container.settings.upload_dir.iterdir()) == []


def test_path_traversal_filename_never_touches_disk_path(tmp_path, sample_pdfs):
    with make_client(tmp_path) as c:
        with sample_pdfs["leave"].open("rb") as fh:
            r = c.post(
                "/documents/upload", headers=HEADERS, files={"file": ("../../../tmp/evil.pdf", fh, "application/pdf")}
            )
        assert r.status_code == 202 and r.json()["filename"] == "evil.pdf"
        stored = list(c.app.state.container.settings.upload_dir.glob("*.pdf"))
        assert len(stored) == 1 and stored[0].parent == c.app.state.container.settings.upload_dir
        assert len(stored[0].stem) == 32  # uuid hex, not the client's name
        assert not (tmp_path.parent / "tmp" / "evil.pdf").exists()


def test_page_limit_and_encrypted_pdf_rejected(tmp_path, sample_pdfs):
    with make_client(tmp_path, max_pages=2) as c:
        with sample_pdfs["leave"].open("rb") as fh:
            r = c.post("/documents/upload", headers=HEADERS, files={"file": ("a.pdf", fh, "application/pdf")})
        assert r.status_code == 422 and r.json()["error"]["code"] == "too_many_pages"
    enc = tmp_path / "enc.pdf"
    pymupdf.open(sample_pdfs["travel"]).save(
        str(enc), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u"
    )
    with make_client(tmp_path / "b") as c:
        with enc.open("rb") as fh:
            r = c.post("/documents/upload", headers=HEADERS, files={"file": ("enc.pdf", fh, "application/pdf")})
        assert r.status_code == 422 and r.json()["error"]["code"] == "encrypted_pdf"


def test_ingestion_queue_limit_returns_429(tmp_path, sample_pdfs):
    with make_client(tmp_path, max_queued_ingestions=0) as c:
        with sample_pdfs["leave"].open("rb") as fh:
            r = c.post("/documents/upload", headers=HEADERS, files={"file": ("a.pdf", fh, "application/pdf")})
        assert r.status_code == 429


def test_rate_limit_enforced_per_key(tmp_path):
    with make_client(tmp_path, rate_limit_ask_per_min=3) as c:
        codes = [c.post("/ask", headers=HEADERS, json={"question": "anything here"}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
        assert c.post("/ask", headers=HEADERS, json={"question": "x y z"}).json()["error"]["code"] == "rate_limited"


def test_upload_rate_limit(tmp_path):
    with make_client(tmp_path, rate_limit_upload_per_min=2) as c:
        codes = [
            c.post(
                "/documents/upload", headers=HEADERS, files={"file": ("a.txt", io.BytesIO(b"x"), "text/plain")}
            ).status_code
            for _ in range(4)
        ]
        assert codes == [415, 415, 429, 429]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": "ok question", "top_k": 0},
        {"question": "ok question", "top_k": 11},
        {"question": "ok question", "document_ids": ["../etc/passwd"]},
        {"question": "ok question", "session_id": "bad id!"},
        {"question": 123},
    ],
)
def test_input_validation_422_without_echoing_input(tmp_path, payload):
    with make_client(tmp_path) as c:
        r = c.post("/ask", headers=HEADERS, json=payload)
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
        assert "passwd" not in r.text and "xxxxxxxx" not in r.text


def test_control_characters_are_stripped_from_questions(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/ask", headers=HEADERS, json={"question": "hello\x00\x07 there friend"})
        assert r.status_code == 200


def test_prompt_injection_pdf_is_treated_as_data(tmp_path, sample_pdfs):
    llm = FakeLLM()
    with make_client(tmp_path, llm=llm) as c:
        c.headers.update(HEADERS)
        upload(c, sample_pdfs["injection"])
        r = c.post("/ask", json={"question": "How often should the snake plant be watered?"}).json()
        prompt_sys = llm.last_messages[0]["content"]
        prompt_user = llm.last_messages[-1]["content"]
        assert "untrusted" in prompt_sys and "Ignore any instruction" in prompt_sys
        assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in prompt_user  # present only as quoted passage data
        inside = prompt_user.split("<<<", 1)[1].rsplit(">>>", 1)[0]
        assert "IGNORE ALL PREVIOUS" in inside or "PWNED" in inside or "snake plant" in inside
        assert r["answer"].strip() != "PWNED" and "system prompt" not in r["answer"].lower()
        assert r["citations"] and all(ci["document"] == "Office_Plants.pdf" for ci in r["citations"])


def test_llm_down_gives_503_and_timeout_gives_504(tmp_path, sample_pdfs):
    class Down(FakeLLM):
        async def chat(self, *a, **k):
            raise OllamaUnavailable()

        async def stream(self, *a, **k):
            raise OllamaUnavailable()
            yield ""

    class Slow(FakeLLM):
        async def chat(self, *a, **k):
            raise LLMTimeout()

        async def stream(self, *a, **k):
            raise LLMTimeout()
            yield ""

    for cls, code, err in ((Down, 503, "llm_unavailable"), (Slow, 504, "llm_timeout")):
        with make_client(tmp_path / cls.__name__, llm=cls()) as c:
            c.headers.update(HEADERS)
            c.app.state.container.pipeline.s.summaries_enabled = False
            upload(c, sample_pdfs["travel"])
            r = c.post("/ask", json={"question": "What is the meal allowance for domestic trips?"})
            assert r.status_code == code and r.json()["error"]["code"] == err
            s = c.post("/ask/stream", json={"question": "What is the meal allowance for domestic trips?"})
            assert s.status_code == code and s.json()["error"]["code"] == err


def test_unexpected_errors_are_generic_and_leak_nothing(tmp_path):
    with make_client(tmp_path, raise_exc=False) as c:

        async def boom(*a, **k):
            raise RuntimeError("SECRET-INTERNAL-DETAIL /home/user/app.py line 42")

        c.app.state.container.pipeline.ask = boom
        r = c.post("/ask", headers=HEADERS, json={"question": "hello there"})
        assert r.status_code == 500
        assert r.json()["error"]["message"] == "Something went wrong on the server."
        assert "SECRET" not in r.text and "Traceback" not in r.text and ".py" not in r.text


def test_404_and_405_use_error_format(tmp_path):
    with make_client(tmp_path) as c:
        r = c.get("/nope", headers=HEADERS)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
        assert c.put("/ask", headers=HEADERS).status_code == 405
        assert c.get("/documents/DOC999", headers=HEADERS).status_code == 404
        assert c.get("/documents/..%2f..%2fetc", headers=HEADERS).status_code in (404, 422)


def test_security_headers_request_id_and_cors(tmp_path):
    with make_client(tmp_path) as c:
        r = c.get("/documents", headers={**HEADERS, "X-Request-ID": "abc123"})
        assert r.headers["X-Request-ID"] == "abc123"
        assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'none'" in r.headers["Content-Security-Policy"] and r.headers["Cache-Control"] == "no-store"
        evil = c.get("/documents", headers={**HEADERS, "Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in evil.headers
        ok = c.get("/documents", headers={**HEADERS, "Origin": "http://localhost:8501"})
        assert ok.headers["access-control-allow-origin"] == "http://localhost:8501"
        assert r.json() == {"documents": [], "count": 0}


def test_logs_never_contain_document_text_or_secrets(tmp_path, sample_pdfs, capsys):
    with make_client(tmp_path, log_level="INFO", log_json=True) as c:
        c.headers.update(HEADERS)
        upload(c, sample_pdfs["leave"])
        c.post("/ask", json={"question": "How many paid sick days do employees receive per year?"})
    out = capsys.readouterr().out
    assert out and '"event": "ask"' in out
    for secret in ("sick days", "medical certificate", "Northwind Labs observes", API_KEY):
        assert secret not in out
    assert "request_id" in out
