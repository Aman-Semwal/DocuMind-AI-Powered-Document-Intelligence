"""Auth and rate-limit dependencies."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from app.container import Container
from app.core.exceptions import RateLimited, Unauthorized
from app.core.security import verify_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key")


def get_container(request: Request) -> Container:
    c: Container = request.app.state.container
    return c


def require_api_key(request: Request, api_key: str | None = Depends(api_key_header)) -> str:
    c = get_container(request)
    if not verify_api_key(api_key, c.settings.api_key_list):
        raise Unauthorized()
    # identify callers by a hash of the key, never the key itself
    return hashlib.sha256((api_key or "").encode()).hexdigest()[:16]


def rate_limit(kind: str) -> Callable[..., None]:
    """Dependency factory: per-key and per-IP limits. `kind` is 'ask', 'upload' or 'default'."""

    def dep(request: Request, key_id: str = Depends(require_api_key)) -> None:
        c = get_container(request)
        s = c.settings
        limit = {
            "ask": s.rate_limit_ask_per_min,
            "upload": s.rate_limit_upload_per_min,
        }.get(kind, s.rate_limit_default_per_min)
        ip = request.client.host if request.client else "unknown"
        ok_key = c.limiter.allow(f"{kind}:key:{key_id}", limit)
        ok_ip = c.limiter.allow(f"{kind}:ip:{ip}", limit * 2)
        if not (ok_key and ok_ip):
            raise RateLimited()

    return dep
