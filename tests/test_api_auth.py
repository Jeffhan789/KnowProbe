"""Authentication policy tests for API dependencies."""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from knowprobe.api.dependencies import optional_api_key
from knowprobe.core.config import Settings


@pytest.mark.asyncio
async def test_development_allows_open_access_without_configured_key() -> None:
    settings = Settings()

    assert await optional_api_key(settings, None) is True


@pytest.mark.asyncio
async def test_locked_production_rejects_missing_configuration() -> None:
    settings = Settings(
        app={"environment": "production"},
        api={"api_key": "", "allow_unauthenticated": False},
    )

    with pytest.raises(HTTPException) as exc_info:
        await optional_api_key(settings, None)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_configured_key_requires_matching_bearer_token() -> None:
    settings = Settings(api={"api_key": "test-secret"})

    with pytest.raises(HTTPException) as missing:
        await optional_api_key(settings, None)
    assert missing.value.status_code == 401

    invalid = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong")
    with pytest.raises(HTTPException) as rejected:
        await optional_api_key(settings, invalid)
    assert rejected.value.status_code == 403

    valid = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-secret")
    assert await optional_api_key(settings, valid) is True
