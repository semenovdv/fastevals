import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langfuse.decorators import observe
from loguru import logger
from openai import AsyncOpenAI, OpenAI

from fasteval.llm import LLMServiceError
from fasteval.llm.providers.base_processor import BaseLLMProcessor, GenerationResult


class OpenRouterProcessor(BaseLLMProcessor):
    """OpenRouter provider using the OpenAI SDK against OpenRouter's base_url.

    Extends BaseLLMProcessor to leverage shared Langfuse logging and utilities.
    """

    provider_name: str = "OpenRouter"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        self._async_client: Optional[AsyncOpenAI] = None

    def _get_async_client(self) -> AsyncOpenAI:
        """Lazily initialize async client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        return self._async_client

    @observe(as_type="generation", name="OpenRouter.chat.completions.create")
    def send_message(  # noqa: C901
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        # Kept for signature parity with LLMClient, but intentionally unused here:
        file_paths: Optional[
            List[str]
        ] = None,  # placeholder: OpenRouter provider does not support local file upload in this class
        history: Optional[List[Dict[str, str]]] = None,  # placeholder: not supported for now
        response_schema: Optional[Any] = None,
        thinking_budget: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a response via OpenRouter's Chat Completions endpoint (OpenAI-compatible).

        Returns:
            str when no schema is requested; dict when response_schema is provided (best effort JSON parse).
        """

        # Build minimal messages array (system + user)
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # --- Request payload
        request_args: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if temperature is not None:
            request_args["temperature"] = temperature

        if seed is not None:
            # OpenRouter supports seed (int)
            request_args["seed"] = seed

        if max_output_tokens is not None:
            request_args["max_tokens"] = max_output_tokens

        # If caller passes an OpenAI-style `user` identifier via **kwargs, forward it.
        # Docs mention 'user' being supported (stable ID for end-users).
        # https://openrouter.ai/docs/api-reference/overview
        user_identifier = kwargs.pop("user", None)
        if user_identifier is not None:
            request_args["user"] = user_identifier

        # Structured outputs: map response_schema to OpenRouter's response_format=json_schema
        # https://openrouter.ai/docs/features/structured-outputs
        rf = self._build_response_format(response_schema)
        if rf is not None:
            request_args["response_format"] = rf

        # Reasoning / thinking budget passthrough. OpenRouter normalizes reasoning params:
        # https://openrouter.ai/docs/use-cases/reasoning-tokens
        reasoning_cfg = self._build_reasoning(thinking_budget)
        if reasoning_cfg is not None:
            request_args["reasoning"] = reasoning_cfg

        # Allow OpenRouter/OpenAI SDK extras to pass through (e.g., extra_headers, extra_body, tools)
        # See: Quickstart 'extra_headers' and Model Routing 'models' via extra_body
        # https://openrouter.ai/docs/quickstart
        # https://openrouter.ai/docs/features/model-routing
        if "extra_headers" in kwargs:
            request_args["extra_headers"] = kwargs.pop("extra_headers")
        if "extra_body" in kwargs:
            request_args["extra_body"] = kwargs.pop("extra_body")

        # Any other supported chat-completions args (e.g., max_tokens, top_p, tools, tool_choice, stop...)
        request_args.update(kwargs)

        # --- Call API with simple retry
        response = self._call_llm(request_args)

        # --- Extract output text (or JSON for structured outputs)
        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            logger.opt(exception=True).error("Failed to extract content from OpenRouter response")
            content = None

        # Best-effort JSON parse when schema requested
        if response_schema is not None and isinstance(content, str):
            parsed = self._maybe_parse_json(content)
            self._log_usage_openrouter(
                response=response,
                output=parsed if parsed is not None else content,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                seed=seed,
                response_schema=response_schema,
                user_identifier=user_identifier,
            )
            return parsed if parsed is not None else content

        # Plain text path
        text_out = content or ""
        self._log_usage_openrouter(
            response=response,
            output=text_out,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            seed=seed,
            response_schema=response_schema,
            user_identifier=user_identifier,
        )
        return text_out

    def _build_request_args(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        seed: Optional[int],
        max_output_tokens: Optional[int],
        response_schema: Optional[Any],
        thinking_budget: Optional[Any],
        **kwargs: Any,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Build request arguments for OpenRouter API call.

        Returns:
            Tuple of (request_args dict, user_identifier).
        """
        request_args: Dict[str, Any] = {"model": model, "messages": messages}

        if temperature is not None:
            request_args["temperature"] = temperature
        if seed is not None:
            request_args["seed"] = seed
        if max_output_tokens is not None:
            request_args["max_tokens"] = max_output_tokens

        user_identifier = kwargs.pop("user", None)
        if user_identifier is not None:
            request_args["user"] = user_identifier

        rf = self._build_response_format(response_schema)
        if rf is not None:
            request_args["response_format"] = rf

        reasoning_cfg = self._build_reasoning(thinking_budget)
        if reasoning_cfg is not None:
            request_args["reasoning"] = reasoning_cfg

        if "extra_headers" in kwargs:
            request_args["extra_headers"] = kwargs.pop("extra_headers")
        if "extra_body" in kwargs:
            request_args["extra_body"] = kwargs.pop("extra_body")

        request_args.update(kwargs)
        return request_args, user_identifier

    @observe(as_type="generation", name="OpenRouter.chat.completions.create_async")
    async def send_message_async(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        file_paths: Optional[List[Union[str, Path]]] = None,
        image_bytes_list: Optional[List[bytes]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        response_schema: Optional[Any] = None,
        thinking_budget: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Generate a response asynchronously via OpenRouter's Chat Completions endpoint."""
        # Build minimal messages array (system + user)
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        request_args, user_identifier = self._build_request_args(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_output_tokens=max_output_tokens,
            response_schema=response_schema,
            thinking_budget=thinking_budget,
            **kwargs,
        )

        # --- Call API asynchronously with simple retry
        response = await self._call_llm_async(request_args)

        # --- Extract output text
        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            logger.opt(exception=True).error("Failed to extract content from OpenRouter response")
            content = None

        # Best-effort JSON parse when schema requested
        if response_schema is not None and isinstance(content, str):
            parsed = self._maybe_parse_json(content)
            self._log_usage_openrouter(
                response=response,
                output=parsed if parsed is not None else content,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                seed=seed,
                response_schema=response_schema,
                user_identifier=user_identifier,
            )
            return parsed if parsed is not None else content

        # Plain text path
        text_out = content or ""
        self._log_usage_openrouter(
            response=response,
            output=text_out,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            seed=seed,
            response_schema=response_schema,
            user_identifier=user_identifier,
        )
        return text_out

    # --------------------------
    # Internals
    # --------------------------

    def _call_llm(self, request_args: Dict[str, Any]):
        last_exc: Optional[Exception] = None
        for _ in range(2):
            try:
                # OpenAI SDK (OpenRouter base_url) chat completions
                # https://openrouter.ai/docs/quickstart
                return self.client.chat.completions.create(**request_args)
            except Exception as exc:
                last_exc = exc
        raise LLMServiceError(f"OpenRouter API request failed twice: {last_exc}")

    async def _call_llm_async(self, request_args: Dict[str, Any]):
        """Async version of _call_llm with simple retry."""
        last_exc: Optional[Exception] = None
        client = self._get_async_client()
        for _ in range(2):
            try:
                return await client.chat.completions.create(**request_args)
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(1)  # Brief delay before retry
        raise LLMServiceError(f"OpenRouter API request failed twice: {last_exc}")

    @staticmethod
    def _maybe_parse_json(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        try:
            return json.loads(text)
        except Exception:
            logger.opt(exception=True).error("Failed to parse JSON from OpenRouter response")
            return None

    @staticmethod
    def _build_response_format(response_schema: Optional[Any]) -> Optional[Dict[str, Any]]:
        """
        Map a provided schema to OpenRouter's response_format=json_schema:
        {
          "type": "json_schema",
          "json_schema": {
            "name": "response",
            "strict": true,
            "schema": <JSON Schema object>
          }
        }

        Accepts:
        - Pydantic BaseModel subclass or instance (uses `.model_json_schema()`)
        - A raw JSON Schema dict (with typical 'type'/'properties')
        - Already-formed { "name": ..., "schema": ... } dicts (we'll pass through)
        """
        if not response_schema:
            return None

        # Pydantic model (class or instance)
        schema_obj: Optional[Dict[str, Any]] = None
        name = "response"

        try:
            # Pydantic v2 style
            if hasattr(response_schema, "model_json_schema"):
                schema_obj = response_schema.model_json_schema()  # type: ignore[attr-defined]
                name = getattr(response_schema, "__name__", "response")
            # Pydantic instance with model_dump? (placeholder branch—will try .__class__.__name__)
            elif hasattr(response_schema, "__class__") and hasattr(response_schema.__class__, "model_json_schema"):
                schema_obj = response_schema.__class__.model_json_schema()  # type: ignore[attr-defined]
                name = getattr(response_schema.__class__, "__name__", "response")
        except Exception:
            logger.opt(exception=True).error("Failed to extract Pydantic schema")
            schema_obj = None  # fall through

        # Raw dict schema
        if schema_obj is None and isinstance(response_schema, dict):
            # If caller passed the final "json_schema" block already, pass through.
            if "schema" in response_schema and ("name" in response_schema or "title" in response_schema):
                json_schema_block = {
                    "name": response_schema.get("name") or response_schema.get("title") or "response",
                    "strict": response_schema.get("strict", True),
                    "schema": response_schema.get("schema", {}),
                }
            else:
                # Heuristic: assume it's a plain JSON Schema object
                json_schema_block = {
                    "name": "response",
                    "strict": True,
                    "schema": response_schema,
                }
            return {"type": "json_schema", "json_schema": json_schema_block}

        if schema_obj is not None:
            return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema_obj}}

        # Unsupported schema type: return placeholder (explicitly opt out)
        # (Caller will still receive text; logging will note schema was provided but not applied.)
        return None

    @staticmethod
    def _build_reasoning(thinking_budget: Optional[Any]) -> Optional[Dict[str, Any]]:
        """
        Map a generic 'thinking_budget' to OpenRouter 'reasoning' param:
          - int  -> {"max_tokens": int}
          - str in {"low","medium","high"} -> {"effort": <str>}
          - dict -> passed as-is (assumed valid OpenRouter reasoning config)
        Docs: https://openrouter.ai/docs/use-cases/reasoning-tokens
        """
        if thinking_budget is None:
            return None
        if isinstance(thinking_budget, int):
            return {"max_tokens": thinking_budget}
        if isinstance(thinking_budget, str):
            return {"effort": thinking_budget}
        if isinstance(thinking_budget, dict):
            return thinking_budget
        # Unknown shape → ignore (placeholder behavior)
        return None

    def _log_usage_openrouter(
        self,
        *,
        response: Any,
        output: Union[str, Dict[str, Any], List[Any]],
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float],
        seed: Optional[int],
        response_schema: Optional[Any],
        user_identifier: Optional[str],
    ) -> None:
        """Log usage metrics to Langfuse using the base class method.

        OpenRouter returns usage.prompt_tokens, usage.completion_tokens, usage.total_tokens (non-streaming).
        https://openrouter.ai/docs/api-reference/overview
        """
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # Best-effort extraction of reasoning text (if present on message)
        reasoning_text = None
        try:
            msg = response.choices[0].message
            reasoning_text = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_details", None)
        except Exception:
            logger.opt(exception=True).error("Failed to extract reasoning text from response")
            reasoning_text = None

        # Prepare input dict
        input_dict = {"system": system_prompt, "user": user_prompt}

        # Create GenerationResult for base class logging
        gen_result = GenerationResult(
            content=output,
            raw_response=response,
            input_tokens=prompt_tokens or 0,
            output_tokens=completion_tokens or 0,
            thoughts=reasoning_text if isinstance(reasoning_text, str) else None,
            provider=self.provider_name,
        )

        # Extra metadata specific to OpenRouter
        extra_metadata = {"user_identifier": user_identifier}

        # Use base class logging
        BaseLLMProcessor._log_usage(
            self,
            model=model,
            input_dict=input_dict,
            output=output,
            result=gen_result,
            temperature=temperature,
            seed=seed,
            response_schema=response_schema,
            extra_metadata=extra_metadata,
        )
