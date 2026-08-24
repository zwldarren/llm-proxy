"""API utility functions."""

import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def create_traced_handler(
    name: str,
    handler_func: Callable[[Any, Request], Awaitable[Any]],
    request_model: type[BaseModel] | None,
) -> Callable[..., Awaitable[Any]]:
    """Create a handler function.

    Args:
        name: The name for the handler (e.g., "openai", "image_generation")
        handler_func: The async handler function
        request_model: The Pydantic model for request validation, or None for no body parsing

    Returns:
        An async handler function
    """
    if request_model is None:

        async def traced_handler(
            fastapi_request: Request,
        ) -> Any:
            return await handler_func(None, fastapi_request)

        return traced_handler

    async def traced_handler(
        request: BaseModel,
        fastapi_request: Request,
    ) -> Any:
        # Stash parsed body for early-failure logging (before processor.process() runs).
        with contextlib.suppress(Exception):
            fastapi_request.state.parsed_request_body = request.model_dump()
        return await handler_func(request, fastapi_request)

    traced_handler.__annotations__["request"] = request_model
    return traced_handler


def create_standard_router(
    request_name: str,
    request_model: type[BaseModel] | None,
    path: str,
    handler: Callable[[Any, Request], Awaitable[Any]],
    response_model: type[BaseModel] | None = None,
    tags: list[str] | None = None,
    prefix: str = "",
    dependencies: list[Any] | None = None,
) -> APIRouter:
    """Create a standard FastAPI router with a single endpoint.

    Args:
        request_name: The name for tracing (e.g., "openai", "image_generation")
        request_model: The Pydantic model for request validation, or None for no body parsing
        path: The endpoint path
        handler: The async handler function
        response_model: Optional response model
        tags: Optional list of tags for documentation
        prefix: Optional router prefix
        dependencies: Optional list of dependencies (e.g., authentication)

    Returns:
        Configured APIRouter instance with a single endpoint
    """
    if tags is None:
        tags = [request_name]

    router = APIRouter(prefix=prefix, tags=tags, dependencies=dependencies or [])  # ty: ignore[invalid-argument-type]
    trace_name = f"http_request:{request_name}"
    traced_handler = create_traced_handler(trace_name, handler, request_model)

    endpoint_kwargs: dict[str, Any] = {
        "path": path,
        "name": f"{request_name}_endpoint",
    }

    if response_model is not None:
        endpoint_kwargs["response_model"] = response_model

    router.post(**endpoint_kwargs)(traced_handler)

    logger.debug(
        f"Created router for '{request_name}' with path '{path}' "
        f"and request model '{request_model.__name__ if request_model else 'None'}'"
    )

    return router


def add_info_endpoint(
    router: APIRouter,
    prefix: str,
    info_name: str,
    info_data: dict[str, Any],
    info_tags: list[str] | None = None,
) -> None:
    """Add a generic info endpoint to a router.

    Args:
        router: The router to add the endpoint to
        prefix: The path prefix (e.g., "/info/")
        info_name: The name for the endpoint path
        info_data: Dictionary of info data to return
        info_tags: Optional list of tags for documentation
    """

    async def info() -> dict[str, Any]:
        return info_data

    router.get(
        f"{prefix}{info_name}",
        name=f"{info_name}_info",
        tags=info_tags,  # ty: ignore[invalid-argument-type]
    )(info)
