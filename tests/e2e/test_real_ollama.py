"""End-to-end test against a real Ollama model. Run with:  pytest -m e2e

Needs: `ollama serve`, `ollama pull qwen2.5:3b-instruct`, `ollama pull nomic-embed-text`.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import HEADERS, make_settings, upload

pytestmark = pytest.mark.e2e


def _ollama_ready() -> bool:
    try:
        names = {m["name"] for m in httpx.get("http://localhost:11434/api/tags", timeout=2).json()["models"]}
    except Exception:
        return False
    return any(n.startswith("qwen2.5:3b") for n in names) and any(n.startswith("nomic-embed-text") for n in names)


@pytest.fixture
def real_client(tmp_path):
    if not _ollama_ready():
        pytest.skip("Ollama with qwen2.5:3b-instruct and nomic-embed-text is not available")
    s = make_settings(
        tmp_path, llm_backend="ollama", embedder_backend="ollama", min_similarity=0.45, rerank_backend="lexical"
    )
    with TestClient(create_app(s)) as c:
        c.headers.update(HEADERS)
        yield c


def test_grounded_answers_citations_and_not_available(real_client, sample_pdfs):
    upload(real_client, sample_pdfs["leave"], timeout=300)
    upload(real_client, sample_pdfs["travel"], timeout=300)

    r = real_client.post("/ask", json={"question": "How many paid sick days do employees get per year?"}).json()
    assert r["answered"] and "12" in r["answer"]
    assert {"document": "Northwind_Leave_Policy.pdf", "page": 2} in [
        {"document": c["document"], "page": c["page"]} for c in r["citations"]
    ]

    r = real_client.post("/ask", json={"question": "What is the hotel nightly limit in London?"}).json()
    assert r["answered"] and "250" in r["answer"]

    r = real_client.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert r["answered"] is False and r["citations"] == []
