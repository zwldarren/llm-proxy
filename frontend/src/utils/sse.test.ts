import { describe, it, expect } from "vitest";
import { parseSSEEvent, parseStreamResponse, parseAnthropicStreamResponse } from "./sse";

describe("sse utils", () => {
  describe("parseSSEEvent", () => {
    it("returns null for non-data lines", () => {
      expect(parseSSEEvent("event: ping")).toBeNull();
      expect(parseSSEEvent(": comment")).toBeNull();
    });

    it("returns null for [DONE]", () => {
      expect(parseSSEEvent("data: [DONE]")).toBeNull();
    });

    it("parses valid SSE data", () => {
      const result = parseSSEEvent('data: {"choices":[{"delta":{"content":"Hello"}}]}');
      expect(result).not.toBeNull();
      expect(result?.data?.choices?.[0]?.delta?.content).toBe("Hello");
    });

    it("detects error in SSE data", () => {
      const result = parseSSEEvent('data: {"error":{"message":"Rate limit","code":429}}');
      expect(result).not.toBeNull();
      expect(result?.error?.message).toBe("Rate limit");
      expect(result?.error?.code).toBe(429);
    });

    it("returns null for invalid JSON", () => {
      const result = parseSSEEvent("data: not-json");
      expect(result).toBeNull();
    });
  });

  describe("parseStreamResponse", () => {
    it("reconstructs content from multiple SSE lines", () => {
      const body = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" "}}]}',
        'data: {"choices":[{"delta":{"content":"World"}}]}',
        "data: [DONE]",
      ].join("\n");

      const result = parseStreamResponse(body);
      expect(result.reconstructedContent).toBe("Hello World");
      expect(result.chunks).toHaveLength(3);
      expect(result.meta.eventCount).toBe(3);
    });

    it("reconstructs reasoning content", () => {
      const body = [
        'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":" step by step"}}]}',
      ].join("\n");

      const result = parseStreamResponse(body);
      expect(result.reasoningContent).toBe("thinking step by step");
    });

    it("collects error chunks", () => {
      const body = 'data: {"error":{"message":"Error"}}';
      const result = parseStreamResponse(body);
      expect(result.chunks).toHaveLength(1);
      expect(result.chunks[0]?.error?.message).toBe("Error");
    });
  });

  describe("parseAnthropicStreamResponse", () => {
    it("extracts text content from content_block_delta events", () => {
      const body = [
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" World"}}',
      ].join("\n");

      const result = parseAnthropicStreamResponse(body);
      expect(result.reconstructedContent).toBe("Hello World");
      expect(result.meta.eventCount).toBe(3);
    });

    it("extracts tool calls from tool_use blocks", () => {
      const body = [
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"loc\\":\\"NY\\"}"}}',
      ].join("\n");

      const result = parseAnthropicStreamResponse(body);
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0]?.id).toBe("tu_1");
      expect(result.toolCalls[0]?.function.name).toBe("get_weather");
    });
  });
});
