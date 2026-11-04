"""The UI's expectations must match the backend's OpenAPI schema (so they never drift apart)."""

from fastapi.testclient import TestClient

from frontend.services.contract import REQUIRED_ENDPOINTS, REQUIRED_SCHEMA_FIELDS


def test_backend_schema_has_every_field_the_ui_reads(backend: TestClient):
    schemas = backend.app.openapi()["components"]["schemas"]
    for name, fields in REQUIRED_SCHEMA_FIELDS.items():
        assert name in schemas, f"schema {name} missing"
        missing = fields - set(schemas[name]["properties"])
        assert not missing, f"{name} lacks {missing}"


def test_backend_has_every_endpoint_the_ui_calls(backend: TestClient):
    paths = backend.app.openapi()["paths"]
    for method, path in REQUIRED_ENDPOINTS:
        assert path in paths and method in paths[path], f"{method.upper()} {path} missing"
