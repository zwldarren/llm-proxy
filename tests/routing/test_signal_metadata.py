from llm_proxy.routing.signals.metadata import MetadataSignal


def _msg(role, content="hi"):
    return {"role": role, "content": content}


def test_short_no_tools_is_low_tier():
    sig = MetadataSignal()
    vote = sig.predict({"messages": [_msg("user"), _msg("assistant"), _msg("user")]})
    assert vote.tier_id == 0
    assert vote.confidence >= 0.5


def test_many_tool_messages_escalates():
    msgs = [_msg("user")] + [_msg("tool", '{"ok": true}') for _ in range(8)]
    msgs += [_msg("assistant")] * 6
    vote = MetadataSignal().predict({"messages": msgs})
    assert vote.tier_id >= 2


def test_empty_messages_default_low():
    vote = MetadataSignal().predict({"messages": []})
    assert vote.tier_id == 0
