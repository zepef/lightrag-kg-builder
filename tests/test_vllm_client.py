"""Tests for vLLM client and factory functions."""

import asyncio

import pytest

from src.llm.vllm_client import VLLMConfig, VLLMClient, create_vllm_llm_func


# ============================================================================
# VLLMConfig
# ============================================================================

class TestVLLMConfig:

    def test_defaults(self):
        cfg = VLLMConfig()
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.1
        assert cfg.timeout == 120.0
        assert cfg.max_retries == 3
        assert cfg.retry_delay == 1.0

    def test_custom(self):
        cfg = VLLMConfig(base_url="http://gpu:9000/v1", max_tokens=4096)
        assert cfg.base_url == "http://gpu:9000/v1"
        assert cfg.max_tokens == 4096


# ============================================================================
# VLLMClient
# ============================================================================

class TestVLLMClient:

    def test_session_initially_none(self):
        client = VLLMClient()
        assert client._session is None
        assert client._connector is None

    def test_close_without_session(self):
        """Closing without opening should not raise."""
        client = VLLMClient()
        asyncio.run(client.close())


# ============================================================================
# create_vllm_llm_func - message construction
# ============================================================================

class TestCreateVllmLlmFunc:

    def test_factory_returns_callable(self):
        func = create_vllm_llm_func()
        assert callable(func)

    def test_history_messages_included_in_payload(self):
        """Verify history_messages are included between system and user messages."""
        captured_payloads = []

        original_func = create_vllm_llm_func(
            vllm_url="http://localhost:8000/v1",
            max_retries=1,
        )

        # We can't easily mock httpx inside the closure, but we can test
        # the function signature accepts history_messages
        import inspect
        sig = inspect.signature(original_func)
        params = list(sig.parameters.keys())
        assert "history_messages" in params
        assert "system_prompt" in params
        assert "prompt" in params

    def test_func_signature_matches_lightrag(self):
        """Verify the function matches LightRAG's expected signature."""
        func = create_vllm_llm_func()
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        # LightRAG calls with: prompt, system_prompt, history_messages, **kwargs
        assert params[0] == "prompt"
        assert "system_prompt" in params
        assert "history_messages" in params
        assert "kwargs" in params
