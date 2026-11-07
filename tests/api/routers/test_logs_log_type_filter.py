"""The usage views span every row that carries spend (ADR-0058).

``log_type`` on the usage endpoints accepts a comma-separated list so the
dashboard can ask for ``endpoint,batch`` in one query: a batch's cost and tokens
show up without counting the batch as a request (request counts come from
``request_logs``).
"""

from llm_proxy.api.routers.logs import _parse_log_type_filter


def test_one_name_stays_a_string() -> None:
    assert _parse_log_type_filter("endpoint") == "endpoint"


def test_a_comma_separated_list_becomes_a_tuple() -> None:
    assert _parse_log_type_filter("endpoint,batch") == ("endpoint", "batch")


def test_surrounding_space_and_empty_entries_are_dropped() -> None:
    assert _parse_log_type_filter(" endpoint , , batch ") == ("endpoint", "batch")


def test_an_empty_value_means_no_filter() -> None:
    assert _parse_log_type_filter(None) is None
    assert _parse_log_type_filter("") is None
    assert _parse_log_type_filter(" , ") is None
