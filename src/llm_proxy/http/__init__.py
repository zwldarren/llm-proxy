"""HTTP client package with connection pool management and request utilities."""

__all__ = [
    "DEFAULT_IMAGE_DOWNLOAD_TIMEOUT",
    "HTTPClient",
    "ProviderHTTPClientManager",
    "download_image_as_base64",
    "fetch_json",
]


def __getattr__(name: str):
    _MODULE = "llm_proxy.http.client"
    _SYMBOLS = {
        "HTTPClient": "HTTPClient",
        "ProviderHTTPClientManager": "ProviderHTTPClientManager",
        "fetch_json": "fetch_json",
        "DEFAULT_IMAGE_DOWNLOAD_TIMEOUT": "DEFAULT_IMAGE_DOWNLOAD_TIMEOUT",
        "download_image_as_base64": "download_image_as_base64",
    }
    if name in _SYMBOLS:
        import importlib

        module = importlib.import_module(_MODULE)
        value = getattr(module, _SYMBOLS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
