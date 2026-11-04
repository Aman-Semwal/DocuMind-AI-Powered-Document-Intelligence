"""Frontend test fixtures: the real backend (fake LLM/embedder) served in-process to the UI client."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.fakes import FakeEmbedder, FakeLLM
from frontend.services.api_client import ApiClient
from tests.conftest import API_KEY, make_settings


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(make_settings(tmp_path), embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(app, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def api(backend: TestClient) -> ApiClient:
    return ApiClient("http://testserver", API_KEY, client=backend)


@pytest.fixture
def ui_client(monkeypatch: pytest.MonkeyPatch, api: ApiClient) -> Iterator[ApiClient]:
    """Make `frontend.app` use the in-process backend."""
    import frontend.services.state as state

    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setattr(state, "build_client", lambda: api)
    yield api
    st.cache_resource.clear()
    st.cache_data.clear()
