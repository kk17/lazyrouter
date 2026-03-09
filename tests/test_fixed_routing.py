"""Tests for fixed-priority routing mode and explicit fallback_models ordering."""

import pytest

from lazyrouter.config import Config, ModelConfig, RouterConfig
from lazyrouter.retry_handler import select_fallback_models


def _make_model(provider="anthropic", model="claude", elo=1400, fallback_models=None):
    return ModelConfig(
        provider=provider,
        model=model,
        description="test",
        coding_elo=elo,
        writing_elo=elo,
        fallback_models=fallback_models,
    )


class TestFallbackModelsConfig:
    def test_fallback_models_uses_explicit_order(self):
        models = {
            "claude-sonnet": _make_model(elo=1400, fallback_models=["copilot-claude", "gemini-flash"]),
            "copilot-claude": _make_model(provider="copilot", elo=1400),
            "gemini-flash": _make_model(provider="gemini", elo=1350),
            "gpt-4o": _make_model(provider="openai", elo=1380),
        }
        result = select_fallback_models("claude-sonnet", models)
        assert result == ["copilot-claude", "gemini-flash"]

    def test_fallback_models_skips_already_tried(self):
        models = {
            "claude-sonnet": _make_model(elo=1400, fallback_models=["copilot-claude", "gemini-flash"]),
            "copilot-claude": _make_model(provider="copilot", elo=1400),
            "gemini-flash": _make_model(provider="gemini", elo=1350),
        }
        result = select_fallback_models(
            "claude-sonnet", models, already_tried={"copilot-claude"}
        )
        assert result == ["gemini-flash"]

    def test_fallback_models_skips_missing_models(self):
        models = {
            "claude-sonnet": _make_model(elo=1400, fallback_models=["nonexistent", "gemini-flash"]),
            "gemini-flash": _make_model(provider="gemini", elo=1350),
        }
        result = select_fallback_models("claude-sonnet", models)
        assert result == ["gemini-flash"]

    def test_fallback_models_none_falls_back_to_elo(self):
        models = {
            "claude-sonnet": _make_model(elo=1400),
            "copilot-claude": _make_model(provider="copilot", elo=1390),
            "gemini-flash": _make_model(provider="gemini", elo=1200),
        }
        result = select_fallback_models("claude-sonnet", models)
        assert result[0] == "copilot-claude"

    def test_fallback_models_excludes_self(self):
        models = {
            "claude-sonnet": _make_model(
                elo=1400,
                fallback_models=["claude-sonnet", "gemini-flash"],
            ),
            "gemini-flash": _make_model(provider="gemini", elo=1350),
        }
        result = select_fallback_models("claude-sonnet", models)
        assert result == ["gemini-flash"]


class TestFixedRouterConfig:
    def test_fixed_mode_no_provider_required(self):
        cfg = RouterConfig(mode="fixed")
        assert cfg.mode == "fixed"
        assert cfg.provider is None
        assert cfg.model is None

    def test_auto_mode_requires_provider_and_model(self):
        with pytest.raises(ValueError, match="provider and router.model are required"):
            RouterConfig(mode="auto")

    def test_auto_mode_with_provider_and_model(self):
        cfg = RouterConfig(mode="auto", provider="gemini", model="gemini-2.5-flash")
        assert cfg.mode == "auto"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="'auto' or 'fixed'"):
            RouterConfig(mode="llm", provider="gemini", model="flash")


class TestFixedModeModelSelection:
    @pytest.mark.asyncio
    async def test_fixed_mode_selects_first_healthy_model(self):
        from unittest.mock import AsyncMock, MagicMock

        from lazyrouter.pipeline import RequestContext, select_model
        from lazyrouter.models import ChatCompletionRequest, Message

        config = MagicMock()
        config.router.mode = "fixed"
        config.llms = {
            "claude-sonnet": _make_model(elo=1400),
            "gemini-flash": _make_model(provider="gemini", elo=1350),
        }
        config.context_compression.skip_router_on_tool_results = True

        request = ChatCompletionRequest(
            model="auto",
            messages=[Message(role="user", content="hello")],
        )
        ctx = RequestContext(request=request, config=config)
        ctx.resolved_model = "auto"
        ctx.messages = [{"role": "user", "content": "hello"}]
        ctx.last_user_text = "hello"

        health_checker = MagicMock()
        health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()
        health_checker.unhealthy_models = set()
        health_checker.healthy_models = {"claude-sonnet", "gemini-flash"}

        router = MagicMock()

        await select_model(ctx, health_checker, router)
        assert ctx.selected_model == "claude-sonnet"
        assert ctx.router_skipped_reason == "fixed mode"

    @pytest.mark.asyncio
    async def test_fixed_mode_skips_unhealthy_model(self):
        from unittest.mock import AsyncMock, MagicMock

        from lazyrouter.pipeline import RequestContext, select_model
        from lazyrouter.models import ChatCompletionRequest, Message

        config = MagicMock()
        config.router.mode = "fixed"
        config.llms = {
            "claude-sonnet": _make_model(elo=1400),
            "gemini-flash": _make_model(provider="gemini", elo=1350),
        }
        config.context_compression.skip_router_on_tool_results = True

        request = ChatCompletionRequest(
            model="auto",
            messages=[Message(role="user", content="hello")],
        )
        ctx = RequestContext(request=request, config=config)
        ctx.resolved_model = "auto"
        ctx.messages = [{"role": "user", "content": "hello"}]
        ctx.last_user_text = "hello"

        health_checker = MagicMock()
        health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()
        health_checker.unhealthy_models = {"claude-sonnet"}
        health_checker.healthy_models = {"gemini-flash"}

        router = MagicMock()

        await select_model(ctx, health_checker, router)
        assert ctx.selected_model == "gemini-flash"
        assert ctx.router_skipped_reason == "fixed mode"
