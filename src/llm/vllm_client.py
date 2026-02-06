"""
vLLM Client - OpenAI-compatible async client for vLLM server integration.

Provides a unified interface for LightRAG to use vLLM instead of Ollama for LLM calls,
enabling faster inference through continuous batching and optimized GPU utilization.
"""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM client."""
    base_url: str = "http://localhost:8000/v1"
    model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0


class VLLMClient:
    """
    OpenAI-compatible async client for vLLM server.

    Designed for integration with LightRAG's LLM interface, providing
    connection pooling, retries, and batch completion support.
    """

    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or VLLMConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=50,
                keepalive_timeout=30
            )
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector:
            await self._connector.close()

    async def health_check(self) -> bool:
        """Check if vLLM server is healthy and responding."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("data", [])
                    logger.info(f"vLLM server healthy. Available models: {[m['id'] for m in models]}")
                    return True
                return False
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return False

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Send completion request to vLLM server.

        Args:
            prompt: The user prompt to complete
            system_prompt: Optional system prompt for context
            **kwargs: Override default config (max_tokens, temperature, etc.)

        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        for attempt in range(self.config.max_retries):
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await resp.text()
                        logger.warning(f"vLLM request failed (attempt {attempt+1}): {resp.status} - {error_text}")
            except asyncio.TimeoutError:
                logger.warning(f"vLLM request timed out (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"vLLM request error (attempt {attempt+1}): {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"vLLM request failed after {self.config.max_retries} attempts")

    async def batch_complete(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        concurrency: int = 10,
        **kwargs
    ) -> List[str]:
        """
        Batch completion for multiple prompts with controlled concurrency.

        Args:
            prompts: List of prompts to complete
            system_prompt: Optional shared system prompt
            concurrency: Max concurrent requests
            **kwargs: Override default config

        Returns:
            List of generated responses in same order as prompts
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_complete(idx: int, prompt: str) -> tuple[int, str]:
            async with semaphore:
                result = await self.complete(prompt, system_prompt, **kwargs)
                return idx, result

        tasks = [bounded_complete(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ordered_results = [""] * len(prompts)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch completion error: {result}")
                continue
            idx, text = result
            ordered_results[idx] = text

        return ordered_results


def create_lightrag_llm_func(vllm_client: VLLMClient):
    """
    Create a LightRAG-compatible LLM function using vLLM client.

    Usage:
        client = VLLMClient(VLLMConfig(base_url="http://localhost:8000/v1"))
        llm_func = create_lightrag_llm_func(client)
        rag = LightRAG(llm_model_func=llm_func, ...)
    """
    async def llm_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        return await vllm_client.complete(prompt, system_prompt, **kwargs)

    return llm_func


async def test_vllm_connection(base_url: str = "http://localhost:8000/v1") -> bool:
    """
    Test vLLM server connection and run a simple completion.

    Args:
        base_url: vLLM server URL

    Returns:
        True if connection successful and completion works
    """
    config = VLLMConfig(base_url=base_url)
    client = VLLMClient(config)

    try:
        if not await client.health_check():
            return False

        response = await client.complete(
            prompt="What is 2+2? Reply with just the number.",
            max_tokens=10
        )
        logger.info(f"Test completion response: {response}")
        return True
    except Exception as e:
        logger.error(f"vLLM connection test failed: {e}")
        return False
    finally:
        await client.close()
