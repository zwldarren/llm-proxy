"""The structured logger's one non-field keyword.

``exc_info`` must reach the stdlib record instead of becoming a structured field:
the field would hold an exception object, which the JSON formatter cannot
serialize, and the traceback would be lost.
"""

import logging

from llm_proxy.observability.logger import JSONFormatter, get_logger


def test_exc_info_reaches_the_record_instead_of_becoming_a_field(caplog):
    logger = get_logger("tests.structured")

    try:
        raise RuntimeError("backend down")
    except RuntimeError as exc:
        with caplog.at_level(logging.WARNING):
            logger.warning("Observation sink failed", exc_info=exc, sink="database_log")

    record = caplog.records[-1]
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    # The payload still carries the structured fields, and the traceback renders
    # from the record rather than from an unserializable ``exc_info`` field.
    assert "sink=database_log" in caplog.text
    assert "RuntimeError: backend down" in caplog.text


def test_the_json_formatter_renders_the_traceback():
    logger = get_logger("tests.structured")
    try:
        raise RuntimeError("backend down")
    except RuntimeError as exc:
        record = logger.logger.makeRecord(
            logger.logger.name,
            logging.WARNING,
            __file__,
            1,
            {"timestamp": 0.0, "message": "failed", "sink": "database_log"},
            None,
            (type(exc), exc, exc.__traceback__),
        )

    rendered = JSONFormatter().format(record)

    assert "RuntimeError: backend down" in rendered
