"""URL rendering for log lines and operator-visible messages.

A URL the operator configured can carry a credential: userinfo on a Redis or
provider URL, an API key in a query string. The whole URL is useless in a log
line anyway, so every rendering drops credentials, query and fragment and keeps
only scheme, host, port and path.
"""

from urllib.parse import urlsplit, urlunsplit


def redact_url(url: str, *, empty_host: str = "<url>") -> str:
    """Render ``url`` without userinfo, query or fragment.

    ``empty_host`` names the placeholder returned when the value carries no
    hostname at all — a relative path, or something that is not a URL.

    Raises nothing: a value ``urlsplit`` cannot parse, or whose port is not a
    number, renders as the placeholder rather than as a traceback in the caller.
    """
    try:
        parsed = urlsplit(url)
        if not parsed.hostname:
            return empty_host
        netloc = parsed.hostname
        if ":" in netloc:
            netloc = f"[{netloc}]"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return empty_host


__all__ = ["redact_url"]
