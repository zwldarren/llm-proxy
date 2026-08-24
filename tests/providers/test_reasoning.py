from llm_proxy.providers.reasoning import normalize_reasoning_in_stream_chunk


class TestNormalizeReasoningInStreamChunk:
    def test_no_choices_returns_unchanged(self):
        chunk = {"id": "test", "choices": []}
        result = normalize_reasoning_in_stream_chunk(chunk)
        assert result == chunk

    def test_no_delta_returns_unchanged(self):
        chunk = {"choices": [{"message": {"content": "hello"}}]}
        result = normalize_reasoning_in_stream_chunk(chunk)
        assert result == chunk

    def test_converts_reasoning_to_reasoning_content(self):
        chunk = {"choices": [{"delta": {"reasoning": "thinking..."}}]}
        result = normalize_reasoning_in_stream_chunk(chunk)
        assert "reasoning_content" in result["choices"][0]["delta"]
        assert "reasoning" not in result["choices"][0]["delta"]
        assert result["choices"][0]["delta"]["reasoning_content"] == "thinking..."

    def test_keeps_existing_reasoning_content(self):
        chunk = {"choices": [{"delta": {"reasoning": "new", "reasoning_content": "existing"}}]}
        result = normalize_reasoning_in_stream_chunk(chunk)
        assert result["choices"][0]["delta"]["reasoning_content"] == "existing"
        assert "reasoning" in result["choices"][0]["delta"]

    def test_no_reasoning_field_returns_unchanged(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        result = normalize_reasoning_in_stream_chunk(chunk)
        assert result == chunk

    def test_mutates_input_in_place(self):
        chunk = {"choices": [{"delta": {"reasoning": "thinking..."}}]}
        original_id = id(chunk)
        result = normalize_reasoning_in_stream_chunk(chunk)
        assert id(result) == original_id
        assert result is chunk

    def test_only_processes_first_choice(self):
        chunk = {
            "choices": [
                {"delta": {"reasoning": "first"}},
                {"delta": {"reasoning": "second"}},
            ]
        }
        result = normalize_reasoning_in_stream_chunk(chunk)
        assert "reasoning_content" in result["choices"][0]["delta"]
        assert "reasoning" in result["choices"][1]["delta"]
