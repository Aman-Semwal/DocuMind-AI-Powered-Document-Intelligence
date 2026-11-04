"""Shared fixtures: temp data dirs, fake LLM/embedder, sample PDFs, a ready TestClient."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.fakes import FakeEmbedder, FakeLLM
from eval.sample_docs import build_injection_pdf, build_sample_pdfs

API_KEY = "test-key-123"
HEADERS = {"X-API-Key": API_KEY}


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "app_env": "test",
        "data_dir": tmp_path / "data",
        "api_keys": API_KEY,
        "llm_backend": "fake",
        "embedder_backend": "fake",
        "rerank_backend": "lexical",
        "min_similarity": 0.2,
        "log_level": "WARNING",
        "log_json": False,
        "rate_limit_ask_per_min": 1000,
        "rate_limit_upload_per_min": 1000,
        "rate_limit_default_per_min": 10000,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("pdfs")
    pdfs = build_sample_pdfs(d)
    pdfs["injection"] = build_injection_pdf(d / "Office_Plants.pdf")
    return pdfs


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM) -> Iterator[TestClient]:
    app = create_app(settings, embedder=FakeEmbedder(), llm=fake_llm)
    with TestClient(app) as c:
        c.headers.update(HEADERS)
        yield c


def upload(client: TestClient, path: Path, *, wait: bool = True, timeout: float = 60.0) -> dict[str, Any]:
    with path.open("rb") as fh:
        resp = client.post("/documents/upload", files={"file": (path.name, fh, "application/pdf")})
    assert resp.status_code == 202, resp.text
    doc = resp.json()
    if wait:
        doc = wait_for(client, doc["document_id"], timeout=timeout)
    return doc


def wait_for(client: TestClient, doc_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = client.get(f"/documents/{doc_id}").json()
        if doc["status"] in ("completed", "failed"):
            return doc  # type: ignore[no-any-return]
        time.sleep(0.05)
    raise AssertionError(f"document {doc_id} did not finish in {timeout}s")


@pytest.fixture
def loaded_client(client: TestClient, sample_pdfs: dict[str, Path]) -> TestClient:
    """A client with both sample HR PDFs already ingested."""
    upload(client, sample_pdfs["leave"])
    upload(client, sample_pdfs["travel"])
    return client
