"""Plain-text log formatting must not allow forged lines."""

import logging
import sys

from llm_proxy.observability.logger import DetailedFormatter


def test_exception_argument_and_traceback_newlines_are_escaped() -> None:
    """An exception object's text and traceback cannot inject a log line."""
    formatter = DetailedFormatter()
    exc = ValueError("upstream\nforged: admin login succeeded")
    try:
        raise exc
    except ValueError:
        record = logging.LogRecord(
            "test",
            logging.ERROR,
            __file__,
            10,
            "request failed: %s",
            (exc,),
            sys.exc_info(),
        )

    output = formatter.format(record)

    assert "\n" not in output
    assert "\r" not in output
    assert "upstream\\nforged" in output
    # The record is restored for other handlers.
    assert record.msg == "request failed: %s"
    assert record.args == (exc,)
