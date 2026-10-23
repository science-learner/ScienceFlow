from __future__ import annotations

import httpx
import pytest

from scienceflow.foundation.config.llm import llm_http
from scienceflow.foundation.config.schema.settings import StageConfig


def test_proxy_credentials_stay_on_transport(monkeypatch):
    captured = {}

    def transport(**kwargs):
        captured["transport"] = kwargs
        return object()

    def client(**kwargs):
        captured["client"] = kwargs
        return object()

    monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("USER_NAME", "proxy-user")
    monkeypatch.setenv("USER_PWD", "proxy-password")
    monkeypatch.setattr(llm_http.httpx, "AsyncHTTPTransport", transport)
    monkeypatch.setattr(llm_http.httpx, "AsyncClient", client)

    llm_http.build_llm_async_client(
        StageConfig(api_key="key", use_proxy=True), pooled=False
    )

    assert isinstance(captured["transport"]["proxy"], httpx.Proxy)
    assert str(captured["transport"]["proxy"].url) == "http://proxy.example:8080"
    assert captured["transport"]["proxy"].raw_auth == (
        b"proxy-user",
        b"proxy-password",
    )
    assert captured["transport"]["verify"] is True
    assert "headers" not in captured["client"]


def test_proxy_disabled_uses_direct_transport(monkeypatch):
    captured = {}
    monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
    monkeypatch.setattr(
        llm_http.httpx,
        "AsyncHTTPTransport",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(llm_http.httpx, "AsyncClient", lambda **kwargs: object())

    llm_http.build_llm_async_client(
        StageConfig(api_key="key", use_proxy=False), pooled=False
    )

    assert captured["proxy"] is None


def test_enabled_proxy_without_environment_fails_closed(monkeypatch):
    for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="use_proxy=true requires"):
        llm_http.build_llm_async_client(
            StageConfig(api_key="key", use_proxy=True), pooled=False
        )
