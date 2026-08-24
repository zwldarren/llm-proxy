"""Token counting utilities using tiktoken."""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken

from llm_proxy.core.constants import TOKEN_ENCODING_CACHE_SIZE
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

_MESSAGE_CACHE_SIZE = 256
_messages_token_cache: dict[str, int] = {}


def _compute_messages_hash(messages: list[dict[str, Any]]) -> str:
    """Compute a stable hash for message list caching."""
    import orjson

    try:
        data = orjson.dumps(messages, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(data).hexdigest()[:16]
    except TypeError, ValueError:
        return hashlib.sha256(str(messages).encode("utf-8")).hexdigest()[:16]


def _get_cached_messages_tokens(messages_hash: str) -> int | None:
    """Get cached token count for messages."""
    return _messages_token_cache.get(messages_hash)


def _set_cached_messages_tokens(messages_hash: str, count: int) -> None:
    """Cache token count for messages."""
    if len(_messages_token_cache) < _MESSAGE_CACHE_SIZE:
        _messages_token_cache[messages_hash] = count


@dataclass
class TokenUsage:
    """Token usage data extracted from provider responses."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Cache token fields
    cache_creation_input_tokens: int = 0  # Anthropic: tokens used to create cache
    cache_read_input_tokens: int = 0  # Anthropic/Gemini: tokens read from cache
    cached_prompt_tokens: int = 0  # OpenAI: cached_tokens from prompt_tokens_details
    # Audio token fields
    audio_input_tokens: int = 0  # OpenAI: audio tokens in prompt
    audio_output_tokens: int = 0  # OpenAI: audio tokens in completion
    # Image token fields
    image_input_tokens: int = 0  # gpt-image: image tokens in prompt
    # Non-token billable dimensions
    images_generated: int = 0  # number of generated images
    audio_duration_seconds: float = 0.0  # STT: duration of transcribed audio
    tts_characters: int = 0  # TTS: characters of synthesized input text
    web_search_requests: int = 0  # number of web search requests


def extract_tokens_from_usage(
    usage: dict[str, Any] | None,
) -> TokenUsage:
    """Extract token counts from usage dict, handling OpenAI and Anthropic formats.

    OpenAI format uses: prompt_tokens, completion_tokens, total_tokens
    Anthropic format uses: input_tokens, output_tokens

    Cache tokens:
    - Anthropic: cache_creation_input_tokens, cache_read_input_tokens
    - OpenAI: prompt_tokens_details.cached_tokens

    Audio tokens:
    - OpenAI: prompt_tokens_details.audio_tokens, completion_tokens_details.audio_tokens

    Image tokens:
    - gpt-image: input_tokens_details.image_tokens (or prompt_tokens_details.image_tokens)

    Non-token dimensions (top-level keys, usually proxy-computed):
    - images_generated, audio_duration_seconds, tts_characters, web_search_requests

    Args:
        usage: Usage dictionary, may be None

    Returns:
        TokenUsage dataclass with all token counts.
    """
    result = TokenUsage()

    if not usage:
        return result

    # Basic token extraction (OpenAI and Anthropic formats)
    raw_input = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    result.completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    result.total_tokens = usage.get("total_tokens") or 0

    # Extract cache tokens from Anthropic format
    result.cache_creation_input_tokens = usage.get("cache_creation_input_tokens") or 0
    result.cache_read_input_tokens = usage.get("cache_read_input_tokens") or 0

    if "prompt_tokens" not in usage and (
        result.cache_read_input_tokens or result.cache_creation_input_tokens
    ):
        raw_input += result.cache_read_input_tokens + result.cache_creation_input_tokens

    result.prompt_tokens = raw_input

    # Extract cache and audio tokens from OpenAI format
    prompt_tokens_details = usage.get("prompt_tokens_details", {})
    if isinstance(prompt_tokens_details, dict):
        result.cached_prompt_tokens = prompt_tokens_details.get("cached_tokens") or 0
        result.audio_input_tokens = prompt_tokens_details.get("audio_tokens") or 0

    # Cache tokens are one billable fact per kind with two dialect
    # expressions (flat Anthropic fields vs nested OpenAI
    # prompt_tokens_details). The flat field wins; the nested expression is
    # only a fallback — expressing the same tokens twice would apply the
    # cache-rate adjustment twice.
    if result.cache_read_input_tokens and result.cached_prompt_tokens:
        result.cached_prompt_tokens = 0
    if not result.cache_creation_input_tokens and isinstance(prompt_tokens_details, dict):
        result.cache_creation_input_tokens = prompt_tokens_details.get("cache_write_tokens") or 0

    completion_tokens_details = usage.get("completion_tokens_details", {})
    if isinstance(completion_tokens_details, dict):
        result.audio_output_tokens = completion_tokens_details.get("audio_tokens") or 0

    # Image tokens: gpt-image uses input_tokens_details, chat uses prompt_tokens_details
    input_tokens_details = usage.get("input_tokens_details", {})
    if isinstance(input_tokens_details, dict):
        result.image_input_tokens = input_tokens_details.get("image_tokens") or 0
    if not result.image_input_tokens and isinstance(prompt_tokens_details, dict):
        result.image_input_tokens = prompt_tokens_details.get("image_tokens") or 0

    # Realtime API dialect (response.done usage): input_token_details /
    # output_token_details (singular "token"). Same facts as the chat and
    # Responses dialects, different nesting — used only as a fallback so the
    # same tokens are never counted twice.
    input_token_details = usage.get("input_token_details", {})
    if isinstance(input_token_details, dict):
        if not result.cached_prompt_tokens:
            result.cached_prompt_tokens = input_token_details.get("cached_tokens") or 0
        if not result.audio_input_tokens:
            result.audio_input_tokens = input_token_details.get("audio_tokens") or 0
        if not result.image_input_tokens:
            result.image_input_tokens = input_token_details.get("image_tokens") or 0

    output_token_details = usage.get("output_token_details", {})
    if isinstance(output_token_details, dict) and not result.audio_output_tokens:
        result.audio_output_tokens = output_token_details.get("audio_tokens") or 0

    # Non-token billable dimensions (flat keys)
    result.images_generated = usage.get("images_generated") or 0
    result.audio_duration_seconds = usage.get("audio_duration_seconds") or 0.0
    result.tts_characters = usage.get("tts_characters") or 0
    result.web_search_requests = usage.get("web_search_requests") or 0

    # Calculate total if not provided
    if not result.total_tokens and (result.prompt_tokens or result.completion_tokens):
        result.total_tokens = result.prompt_tokens + result.completion_tokens

    # Validate total_tokens against sum (warning only, don't modify)
    # Some providers report total_tokens that includes cache tokens, others don't
    if result.total_tokens and result.prompt_tokens and result.completion_tokens:
        calculated = result.prompt_tokens + result.completion_tokens
        if calculated != result.total_tokens and not result.cache_read_input_tokens:
            logger.debug(
                f"Token usage mismatch: reported total={result.total_tokens}, "
                f"calculated={calculated}. Provider may include cache in total."
            )

    return result


@lru_cache(maxsize=TOKEN_ENCODING_CACHE_SIZE)
def _text_hash(text: str) -> str:
    """Compute a stable hash for text caching.

    Using hash instead of full string as key saves memory for large texts.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_encoding = None
_token_cache: dict[str, int] = {}


def _get_encoding() -> tiktoken.Encoding:
    """Get and cache the tiktoken encoding (o200k_base)."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("o200k_base")
    return _encoding


def _get_cached_token_count(text_hash: str) -> int | None:
    """Get cached token count by hash."""
    return _token_cache.get(text_hash)


def _set_cached_token_count(text_hash: str, count: int) -> None:
    """Cache token count by hash."""
    if len(_token_cache) < TOKEN_ENCODING_CACHE_SIZE:
        _token_cache[text_hash] = count


def _batch_encode_texts(texts: list[str]) -> dict[str, int]:
    """Batch encode multiple texts with caching.

    Uses cache-first approach to avoid redundant encoding.
    Only encodes texts not already cached.

    Args:
        texts: List of texts to encode

    Returns:
        Dictionary mapping text to token count
    """
    result: dict[str, int] = {}
    uncached_texts: list[tuple[str, str]] = []

    for text in texts:
        if not text:
            continue
        text_hash = _text_hash(text)
        cached = _get_cached_token_count(text_hash)
        if cached is not None:
            result[text] = cached
        else:
            uncached_texts.append((text, text_hash))

    if uncached_texts:
        encoding = _get_encoding()
        for text, text_hash in uncached_texts:
            count = len(encoding.encode(text))
            _set_cached_token_count(text_hash, count)
            result[text] = count

    return result


def count_tokens(text: str | None) -> int:
    """Count tokens in a text string using o200k_base encoding.

    Args:
        text: The text to count tokens for

    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0

    text_hash = _text_hash(text)
    cached = _get_cached_token_count(text_hash)
    if cached is not None:
        return cached

    encoding = _get_encoding()
    count = len(encoding.encode(text))
    _set_cached_token_count(text_hash, count)
    return count


def _extract_text_fields(item: dict[str, Any], *keys: str) -> list[str]:
    """Extract text fields from a dictionary for token encoding."""
    texts = []
    for key in keys:
        value = item.get(key)
        if value and isinstance(value, str):
            texts.append(value)
        elif value and isinstance(value, dict):
            texts.append(str(value))
    return texts


def _extract_text_list(content: Any) -> list[str]:
    """Extract text items from a list content."""
    texts = []
    if not isinstance(content, list):
        return texts

    for item in content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            part_type = item.get("type")
            if isinstance(part_type, str):
                handler = _CONTENT_TYPE_HANDLERS.get(part_type)
                if handler:
                    texts.extend(handler(item))
    return texts


def _handle_text_content(item: dict[str, Any]) -> list[str]:
    """Handle text content type."""
    text = item.get("text", "")
    return [text] if text else []


def _handle_image_url(item: dict[str, Any]) -> list[str]:
    """Handle image_url content type - adds fixed token count."""
    # Image URLs add approximately 85 tokens
    return []  # Token count handled by caller


def _handle_thinking(item: dict[str, Any]) -> list[str]:
    """Handle thinking content type."""
    texts = []
    thinking = item.get("thinking", "")
    if thinking:
        texts.append(thinking)
    signature = item.get("signature")
    if signature:
        texts.append(signature)
    return texts


def _handle_tool_use(item: dict[str, Any]) -> list[str]:
    """Handle tool_use content type."""
    return _extract_text_fields(item, "id", "name", "input")


def _handle_tool_result(item: dict[str, Any]) -> list[str]:
    """Handle tool_result content type."""
    texts = []
    tool_use_id = item.get("tool_use_id", "")
    if tool_use_id:
        texts.append(tool_use_id)
    result_content = item.get("content")
    if isinstance(result_content, str):
        texts.append(result_content)
    elif isinstance(result_content, list):
        for rc in result_content:
            if isinstance(rc, dict):
                if rc.get("type") == "text":
                    text = rc.get("text", "")
                    if text:
                        texts.append(text)
                elif isinstance(rc, str):
                    texts.append(rc)
    return texts


def _handle_server_tool_use(item: dict[str, Any]) -> list[str]:
    """Handle server_tool_use content type."""
    return _extract_text_fields(item, "id", "name", "input")


def _handle_web_search_tool_result(item: dict[str, Any]) -> list[str]:
    """Handle web_search_tool_result content type."""
    texts = []
    tool_use_id = item.get("tool_use_id", "")
    if tool_use_id:
        texts.append(tool_use_id)
    web_content = item.get("content")
    if isinstance(web_content, list):
        for result in web_content:
            if isinstance(result, dict):
                title = result.get("title", "")
                url = result.get("url", "")
                if title:
                    texts.append(title)
                if url:
                    texts.append(url)
    return texts


def _handle_web_fetch_tool_result(item: dict[str, Any]) -> list[str]:
    """Handle web_fetch_tool_result content type."""
    texts = []
    tool_use_id = item.get("tool_use_id", "")
    if tool_use_id:
        texts.append(tool_use_id)
    content = item.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts.extend(_extract_text_list(content))
    return texts


def _handle_code_execution_tool_result(item: dict[str, Any]) -> list[str]:
    """Handle code_execution_tool_result content type."""
    texts = []
    tool_use_id = item.get("tool_use_id", "")
    if tool_use_id:
        texts.append(tool_use_id)
    content = item.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts.extend(_extract_text_list(content))
    return texts


def _handle_bash_code_execution_tool_result(item: dict[str, Any]) -> list[str]:
    """Handle bash_code_execution_tool_result content type."""
    texts = []
    tool_use_id = item.get("tool_use_id", "")
    if tool_use_id:
        texts.append(tool_use_id)
    content = item.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts.extend(_extract_text_list(content))
    return texts


def _handle_text_editor_tool_result(item: dict[str, Any]) -> list[str]:
    """Handle text_editor_code_execution_tool_result content type."""
    texts = []
    tool_use_id = item.get("tool_use_id", "")
    if tool_use_id:
        texts.append(tool_use_id)
    content = item.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts.extend(_extract_text_list(content))
    return texts


def _handle_tool_search_tool_result(item: dict[str, Any]) -> list[str]:
    """Handle tool_search_tool_result content type."""
    texts = []
    tool_use_id = item.get("tool_use_id", "")
    if tool_use_id:
        texts.append(tool_use_id)
    content = item.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts.extend(_extract_text_list(content))
    return texts


def _handle_search_result(item: dict[str, Any]) -> list[str]:
    """Handle search_result content type."""
    texts = []
    if item.get("file_id"):
        texts.append(item["file_id"])
    if item.get("title"):
        texts.append(item["title"])
    content = item.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts.extend(_extract_text_list(content))
    return texts


def _handle_container_upload(item: dict[str, Any]) -> list[str]:
    """Handle container_upload content type."""
    texts = []
    if item.get("file_id"):
        texts.append(item["file_id"])
    if item.get("filename"):
        texts.append(item["filename"])
    if item.get("content"):
        texts.append(item["content"])
    return texts


def _handle_tool_reference(item: dict[str, Any]) -> list[str]:
    """Handle tool_reference content type."""
    texts = []
    if item.get("tool_id"):
        texts.append(item["tool_id"])
    if item.get("tool_name"):
        texts.append(item["tool_name"])
    if item.get("tool_type"):
        texts.append(item["tool_type"])
    return texts


# Content type handlers mapping - strategy pattern for extensibility
_CONTENT_TYPE_HANDLERS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "text": _handle_text_content,
    # OpenResponses content block types (input_text/output_text/summary_text
    # all carry a plain ``text`` key, same as ``text``).
    "input_text": _handle_text_content,
    "output_text": _handle_text_content,
    "summary_text": _handle_text_content,
    "image_url": _handle_image_url,
    "thinking": _handle_thinking,
    "tool_use": _handle_tool_use,
    "tool_result": _handle_tool_result,
    "server_tool_use": _handle_server_tool_use,
    "web_search_tool_result": _handle_web_search_tool_result,
    "web_fetch_tool_result": _handle_web_fetch_tool_result,
    "code_execution_tool_result": _handle_code_execution_tool_result,
    "bash_code_execution_tool_result": _handle_bash_code_execution_tool_result,
    "text_editor_code_execution_tool_result": _handle_text_editor_tool_result,
    "tool_search_tool_result": _handle_tool_search_tool_result,
    "search_result": _handle_search_result,
    "container_upload": _handle_container_upload,
    "tool_reference": _handle_tool_reference,
}

# Token constants for special content types
_IMAGE_URL_TOKENS = 85


def _count_text_content(content: Any) -> int:
    """Count tokens from text content in various formats.

    Args:
        content: Content in various formats (str, list, dict)

    Returns:
        Token count
    """
    tokens = 0
    texts_to_encode = []

    if isinstance(content, str):
        if content:
            texts_to_encode.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                texts_to_encode.append(item)
            elif isinstance(item, dict):
                part_type = item.get("type")
                if isinstance(part_type, str):
                    handler = _CONTENT_TYPE_HANDLERS.get(part_type)
                    if handler:
                        texts_to_encode.extend(handler(item))
                    # Handle image_url token count
                    if part_type == "image_url":
                        tokens += _IMAGE_URL_TOKENS

    token_counts = _batch_encode_texts(texts_to_encode)
    tokens += sum(token_counts.values())

    return tokens


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a list of messages using o200k_base encoding.

    This function supports multiple message formats:
    - OpenAI format: messages with role/content, and optional reasoning_content
    - Anthropic format: messages where content is a list of content blocks
    - Gemini format: messages with parts containing text, function calls, etc.

    Uses message-level and text-level caching for optimal performance.

    Args:
        messages: List of message dictionaries

    Returns:
        Total number of tokens in all messages
    """
    if not messages:
        return 0

    messages_hash = _compute_messages_hash(messages)
    cached = _get_cached_messages_tokens(messages_hash)
    if cached is not None:
        return cached

    total_tokens = 0
    texts_to_encode = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        content = message.get("content")

        tokens_per_message = 3
        total_tokens += tokens_per_message

        if role:
            texts_to_encode.append(role)

        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            if isinstance(reasoning_content, str):
                texts_to_encode.append(reasoning_content)
            elif isinstance(reasoning_content, list):
                for item in reasoning_content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("reasoning_content", "")
                        if text:
                            texts_to_encode.append(text)
            elif isinstance(reasoning_content, dict):
                text = reasoning_content.get("text", "") or reasoning_content.get(
                    "reasoning_content", ""
                )
                if text:
                    texts_to_encode.append(text)
                signature = reasoning_content.get("signature")
                if signature:
                    texts_to_encode.append(signature)

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                total_tokens += 6
                tc_id = tc.get("id", "")
                if tc_id:
                    texts_to_encode.append(tc_id)
                function = tc.get("function", {})
                if function:
                    func_name = function.get("name", "")
                    func_args = function.get("arguments", "")
                    if func_name:
                        texts_to_encode.append(func_name)
                    if func_args:
                        texts_to_encode.append(func_args)

        if content:
            total_tokens += _count_text_content(content)

        parts = message.get("parts")
        if parts:
            for part in parts:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        texts_to_encode.append(text)

                    if "functionCall" in part:
                        fc = part["functionCall"]
                        func_name = fc.get("name", "")
                        args = fc.get("args", {})
                        if func_name:
                            texts_to_encode.append(func_name)
                        if args:
                            texts_to_encode.append(str(args))

                    if "functionResponse" in part:
                        fr = part["functionResponse"]
                        func_name = fr.get("name", "")
                        response = fr.get("response", {})
                        if func_name:
                            texts_to_encode.append(func_name)
                        if isinstance(response, dict):
                            content_val = response.get("content")
                            if isinstance(content_val, str):
                                texts_to_encode.append(content_val)
                            elif isinstance(content_val, list):
                                for item in content_val:
                                    if isinstance(item, dict):
                                        text = item.get("text", "")
                                        if text:
                                            texts_to_encode.append(text)

    token_counts = _batch_encode_texts(texts_to_encode)
    total_tokens += sum(token_counts.values())
    total_tokens += 3

    _set_cached_messages_tokens(messages_hash, total_tokens)
    return total_tokens


def estimate_usage_from_request(
    messages: list[dict[str, Any]] | None,
    completion_text: str | None = None,
) -> dict[str, int]:
    """Estimate token usage from request messages and completion.

    Args:
        messages: List of message dictionaries
        completion_text: Optional completion text to count

    Returns:
        Dictionary with prompt_tokens, completion_tokens, and total_tokens
    """
    prompt_tokens = 0
    if messages:
        prompt_tokens = count_messages_tokens(messages)

    completion_tokens = 0
    if completion_text:
        completion_tokens = count_tokens(completion_text)

    total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def count_tools_tokens(tools: list[Any] | None) -> int:
    """Count tokens in a list of tool definitions.

    Handles both dict format (Anthropic/OpenAI) and object format (unified).

    Args:
        tools: List of tool definitions (dicts or objects with name/description/parameters)

    Returns:
        Total token count for all tools
    """
    if not tools:
        return 0

    total_tokens = 0
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            # Anthropic uses input_schema, OpenAI uses parameters
            schema = tool.get("input_schema") or tool.get("parameters", {})
        else:
            name = getattr(tool, "name", "") or ""
            description = getattr(tool, "description", "") or ""
            schema = getattr(tool, "parameters", {}) or {}

        if name:
            total_tokens += count_tokens(name)
        if description:
            total_tokens += count_tokens(description)
        if schema:
            total_tokens += count_tokens(str(schema))

    return total_tokens


def count_embedding_input_tokens(input_data: str | list[str] | None) -> int:
    """Count tokens in embedding input (string or list of strings).

    Args:
        input_data: The embedding input - either a single string or a list of strings

    Returns:
        Total number of tokens in the input
    """
    if not input_data:
        return 0

    if isinstance(input_data, str):
        return count_tokens(input_data)

    # List of strings - count tokens for each and sum
    total = 0
    for text in input_data:
        if text and isinstance(text, str):
            total += count_tokens(text)
    return total


def estimate_embedding_usage(
    input_data: str | list[str] | None,
) -> dict[str, int]:
    """Estimate token usage for embedding requests.

    Embeddings only have input tokens (prompt_tokens), no output/completion tokens.

    Args:
        input_data: The embedding input - either a single string or a list of strings

    Returns:
        Dictionary with prompt_tokens and total_tokens (no completion_tokens for embeddings)
    """
    prompt_tokens = count_embedding_input_tokens(input_data)

    return {
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens,
    }


async def count_messages_tokens_async(messages: list[dict[str, Any]]) -> int:
    """Async wrapper for count_messages_tokens.

    Runs token counting in a thread pool to avoid blocking the event loop.
    This is important for long conversations where token counting may take
    significant CPU time.
    """
    return await asyncio.to_thread(count_messages_tokens, messages)


async def count_tools_tokens_async(tools: list[Any] | None) -> int:
    """Async wrapper for count_tools_tokens.

    Runs token counting in a thread pool to avoid blocking the event loop.
    """
    if not tools:
        return 0
    return await asyncio.to_thread(count_tools_tokens, tools)
