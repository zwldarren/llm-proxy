"""Closed-form ask extraction and its effect on the v2 floors.

A closed-form ask (yes/no directive, lettered multiple-choice run, "which of
the following") constrains the response space by construction: the difficulty
lives in the selection, not in the volume of the material it cites. The
structural signal classifies the ask, and the substance/structural floors —
which exist for implementation/design asks — stand down for it.
"""

from llm_proxy.routing.api import classify
from llm_proxy.routing.signal_tuning import extract_closed_form_ask, is_closed_form_ask
from llm_proxy.routing.signals.structural import StructuralSignal


def _long_reference_text(ask: str, *, filler_chars: int = 2000) -> str:
    """Reference material terminated by *ask* — the single-message shape."""
    filler = (
        "The military history of the United States covers operations against "
        "Germany, Italy and Japan, beginning with the December 1941 attack on "
        "Pearl Harbor and the subsequent declaration of war by Congress. "
    )
    body = filler * (filler_chars // len(filler) + 1)
    return f"{body}\n\n{ask}"


def test_extract_mc_ask_from_long_inline_options():
    text = (
        '{"meta": {"name_exp": "bar_game_v1", "player_num": 10},\n  "log": ["turn 1", "turn 2"]}'
        * 20
    )
    text += (
        "Which player got the least utility in the game?  "
        "A. player_1 B. player_3 C. player_5 D. player_7"
    )
    ask = extract_closed_form_ask(text)
    assert ask is not None
    assert "A. player_1" in ask
    assert "least utility" in ask
    assert len(ask) < len(text) * 0.5


def test_extract_yes_no_ask_from_long_passage():
    ask = "Answer only yes or no: was the united states in world war 2?"
    extracted = extract_closed_form_ask(_long_reference_text(ask))
    assert extracted is not None
    assert "yes or no" in extracted


def test_extract_which_following_ask():
    text = _long_reference_text("Which of the following statements is correct?")
    assert extract_closed_form_ask(text) is not None


def test_short_message_is_classified_whole():
    assert extract_closed_form_ask("Answer only yes or no: is X true?") is None


def test_long_open_form_message_yields_no_ask():
    text = _long_reference_text("Please fix it.")
    assert extract_closed_form_ask(text) is None


def test_instruction_dominant_message_yields_no_ask():
    """When the ask is half the message, the volume is the instruction set."""
    body = (
        "Implement a rate limiter. Requirements: token bucket, Redis-backed, "
        "per-key dimensions, Retry-After headers, atomic Lua script. "
    )
    ask = "Start with the Lua script."
    text = body + ask
    assert extract_closed_form_ask(text) is None


def test_is_closed_form_ask_recognizes_formats():
    assert is_closed_form_ask("Which player won?  A. one B. two C. three D. four")
    assert is_closed_form_ask("Answer only yes or no: is X true?")
    assert is_closed_form_ask("Which of the following is correct?")
    assert not is_closed_form_ask("Please fix the failing checkout handler.")
    assert not is_closed_form_ask("")


def test_structural_signal_classifies_the_ask_not_the_corpus():
    vote = StructuralSignal().predict(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _long_reference_text(
                        "Answer only yes or no: was the united states in world war 2?"
                    ),
                }
            ]
        }
    )
    assert vote.tier_id == 0


def test_structural_signal_keeps_full_text_without_closed_form():
    vote = StructuralSignal().predict(
        {"messages": [{"role": "user", "content": _long_reference_text("Please fix it.")}]}
    )
    # No closed-form ask: the full text is classified, as before the change.
    assert vote.tier_id >= 1


def test_closed_form_ask_stands_down_the_structural_floor():
    result = classify(
        [
            {
                "role": "user",
                "content": _long_reference_text(
                    "Which player got the least utility in the game?  "
                    "A. player_1 B. player_3 C. player_5 D. player_7"
                ),
            }
        ]
    )
    assert "v2:structural-floor" not in result.v2.signals_text
    assert "v2:structural-medium-floor" not in result.v2.signals_text


def test_open_form_long_text_keeps_the_structural_floor():
    result = classify([{"role": "user", "content": _long_reference_text("Please fix it.")}])
    assert "v2:structural-floor" in result.v2.signals_text
