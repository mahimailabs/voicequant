"""OpenAI-compatible client helpers for VoiceQuant server.

Provides convenience functions for connecting to a VoiceQuant server
using the standard OpenAI Python client.
"""

from __future__ import annotations

from typing import Any


def create_client(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "voicequant",
) -> Any:
    """Create an OpenAI client pointing at a VoiceQuant server.

    Args:
        base_url: VoiceQuant server URL.
        api_key: API key (VoiceQuant accepts any non-empty string).

    Returns:
        openai.OpenAI client instance.
    """
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def create_async_client(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "voicequant",
) -> Any:
    """Create an async OpenAI client pointing at a VoiceQuant server.

    Args:
        base_url: VoiceQuant server URL.
        api_key: API key (VoiceQuant accepts any non-empty string).

    Returns:
        openai.AsyncOpenAI client instance.
    """
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=base_url, api_key=api_key)
