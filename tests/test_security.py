import pytest
from fastapi.testclient import TestClient

import lazyrouter.server as server_mod
from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)

INVALID_API_KEY_DETAIL = "Invalid API Key"
MISSING_API_KEY_DETAIL = "Missing API Key"


def _config_with_auth(api_key: str | None = "secret-key") -> Config:
    return Config(
        serve=ServeConfig(api_key=api_key),
        router=RouterConfig(provider="p1", model="m_fast"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "m_fast": ModelConfig(
                provider="p1", model="provider-fast", description="fast"
            ),
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=100),
    )

def setup_mocks(monkeypatch):
    # Mock HealthChecker
    monkeypatch.setattr(server_mod.HealthChecker, "start", lambda _: None)
    monkeypatch.setattr(server_mod.HealthChecker, "stop", lambda _: None)

    # Mock pipeline functions to avoid logic execution
    async def _fake_select_model(*_args, **_kwargs):
        pass

    monkeypatch.setattr(server_mod, "select_model", _fake_select_model)

    monkeypatch.setattr(server_mod, "compress_context", lambda _: None)
    monkeypatch.setattr(server_mod, "prepare_provider", lambda _: None)

    async def _fake_call_with_fallback(*_args, **_kwargs):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "provider-fast",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(server_mod, "call_with_fallback", _fake_call_with_fallback)


def test_chat_completion_no_auth_fails(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(preloaded_config=_config_with_auth())

    with TestClient(app) as client:
        # Request WITHOUT Authorization header
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    # Note: with HTTPBearer(auto_error=False), missing Authorization headers result in
    # credentials=None; our verify_api_key function then raises the 401 when API key
    # authentication is configured.
    assert response.status_code == 401
    assert response.json()["detail"] == MISSING_API_KEY_DETAIL


def test_chat_completion_valid_auth_succeeds(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(preloaded_config=_config_with_auth())

    with TestClient(app) as client:
        # Request WITH valid Authorization header
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer secret-key"},
        )

    assert response.status_code == 200


def test_chat_completion_invalid_auth_fails(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(preloaded_config=_config_with_auth())

    with TestClient(app) as client:
        # Request WITH INVALID Authorization header
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer wrong-key"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == INVALID_API_KEY_DETAIL


def test_chat_completion_no_api_key_config_allows_unauthenticated(monkeypatch):
    setup_mocks(monkeypatch)
    # Configure the server with no API key (None) so verify_api_key should allow all requests.
    app = server_mod.create_app(preloaded_config=_config_with_auth(api_key=None))

    with TestClient(app) as client:
        # Request WITHOUT Authorization header should succeed when no API key is configured
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        response_with_auth = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer some-token"},
        )

    assert response.status_code == 200
    assert response_with_auth.status_code == 200


def test_chat_completion_empty_api_key_config_raises_validation_error():
    # An empty string API key must be rejected at configuration time to prevent
    # secrets.compare_digest("", "") from accidentally authenticating empty Bearer tokens.
    with pytest.raises(ValueError, match="must not be an empty string"):
        _config_with_auth(api_key="")


def test_chat_completion_whitespace_only_api_key_config_raises_validation_error():
    with pytest.raises(ValueError, match="must not be an empty string"):
        _config_with_auth(api_key="   ")


def test_chat_completion_strips_configured_api_key_whitespace(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(
        preloaded_config=_config_with_auth(api_key="  secret-key  ")
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer secret-key"},
        )

    assert response.status_code == 200


def test_health_endpoints_require_auth_when_api_key_is_configured(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(preloaded_config=_config_with_auth())

    with TestClient(app) as client:
        health_status = client.get("/v1/health-status")
        health_check = client.get("/v1/health-check")
        authed_health_status = client.get(
            "/v1/health-status",
            headers={"Authorization": "Bearer secret-key"},
        )

    assert health_status.status_code == 401
    assert health_status.json()["detail"] == MISSING_API_KEY_DETAIL
    assert health_check.status_code == 401
    assert health_check.json()["detail"] == MISSING_API_KEY_DETAIL
    assert authed_health_status.status_code == 200
