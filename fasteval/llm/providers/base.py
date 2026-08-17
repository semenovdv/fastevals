from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class LLMProcessor(Protocol):
    """Protocol for LLM provider processors."""

    def send_message(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        file_paths: Optional[List[Union[str, Path]]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None,
        thinking_budget: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Send a message to the LLM and get a response."""
        ...

    async def send_message_async(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        file_paths: Optional[List[Union[str, Path]]] = None,
        image_bytes_list: Optional[List[bytes]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None,
        thinking_budget: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Send a message to the LLM asynchronously."""
        ...
