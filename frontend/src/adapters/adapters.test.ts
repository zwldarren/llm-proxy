import { describe, it, expect } from "vitest";
import { openaiAdapter } from "@/adapters/openaiAdapter";
import { anthropicAdapter } from "@/adapters/anthropicAdapter";
import { openResponsesAdapter } from "@/adapters/openResponsesAdapter";
import { getAdapterForEndpoint } from "@/adapters";

describe("Protocol Adapters", () => {
  describe("getAdapterForEndpoint", () => {
    it("returns openai adapter for /chat/completions", () => {
      const adapter = getAdapterForEndpoint("/v1/chat/completions");
      expect(adapter.id).toBe("chat/completions");
    });

    it("returns anthropic adapter for /messages", () => {
      const adapter = getAdapterForEndpoint("/v1/messages");
      expect(adapter.id).toBe("messages");
    });

    it("returns openResponses adapter for /responses", () => {
      const adapter = getAdapterForEndpoint("/v1/responses");
      expect(adapter.id).toBe("responses");
    });

    it("falls back to openai for unknown endpoints", () => {
      const adapter = getAdapterForEndpoint("/v1/unknown");
      expect(adapter.id).toBe("chat/completions");
    });
  });

  describe("openaiAdapter.formatMessages", () => {
    it("formats basic messages with system prompt", () => {
      const result = openaiAdapter.formatMessages(
        [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there" },
        ],
        "You are a helpful assistant"
      );

      expect(result).toHaveLength(3);
      expect(result[0]).toEqual({ role: "system", content: "You are a helpful assistant" });
      expect(result[1]).toEqual({ role: "user", content: "Hello", tool_calls: undefined });
      expect(result[2]).toEqual({ role: "assistant", content: "Hi there", tool_calls: undefined });
    });

    it("formats tool messages", () => {
      const result = openaiAdapter.formatMessages([
        { role: "user", content: "Check weather" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "get_weather", arguments: "{}" } },
          ],
        },
        { role: "tool", content: '{"temp": 72}', tool_call_id: "call_1", name: "get_weather" },
      ]);

      expect(result).toHaveLength(3);
      expect(result[2]).toEqual({
        role: "tool",
        tool_call_id: "call_1",
        name: "get_weather",
        content: '{"temp": 72}',
      });
    });
  });

  describe("openaiAdapter.formatTools", () => {
    it("returns undefined for empty tools array", () => {
      expect(openaiAdapter.formatTools([])).toBeUndefined();
    });

    it("formats tools in OpenAI format", () => {
      const result = openaiAdapter.formatTools([
        { name: "get_weather", description: "Get weather", parameters: "{}", enabled: true },
      ]);

      expect(result).toHaveLength(1);
      expect(result![0]).toEqual({
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather",
          parameters: {},
        },
      });
    });
  });

  describe("openaiAdapter.parseStreamChunk", () => {
    it("extracts content delta from OpenAI chunk", () => {
      const chunks: string[] = [];
      openaiAdapter.parseStreamChunk(
        { choices: [{ delta: { content: "Hello" }, finish_reason: null }] },
        "",
        { onChunk: (c) => chunks.push(c) }
      );
      expect(chunks).toEqual(["Hello"]);
    });

    it("extracts reasoning content", () => {
      const reasoningChunks: string[] = [];
      openaiAdapter.parseStreamChunk(
        { choices: [{ delta: { content: "", reasoning_content: "thinking..." } }] },
        "",
        { onChunk: () => {}, onReasoningChunk: (c) => reasoningChunks.push(c) }
      );
      expect(reasoningChunks).toEqual(["thinking..."]);
    });

    it("extracts tool calls", () => {
      const toolCalls: Array<[number, string, string, string]> = [];
      openaiAdapter.parseStreamChunk(
        {
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_1",
                    function: { name: "get_weather", arguments: '{"loc":"NY"}' },
                  },
                ],
              },
            },
          ],
        },
        "",
        {
          onChunk: () => {},
          onToolCall: (i, id, name, args) => toolCalls.push([i, id, name, args]),
        }
      );
      expect(toolCalls).toHaveLength(1);
      expect(toolCalls[0]).toEqual([0, "call_1", "get_weather", '{"loc":"NY"}']);
    });

    it("handles error in chunk", () => {
      const errors: string[] = [];
      openaiAdapter.parseStreamChunk({ error: { message: "Rate limit exceeded" } }, "", {
        onChunk: () => {},
        onError: (e) => errors.push(e),
      });
      expect(errors[0]).toContain("Rate limit exceeded");
    });
  });

  describe("anthropicAdapter.formatMessages", () => {
    it("formats basic messages", () => {
      const result = anthropicAdapter.formatMessages([
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" },
      ]);

      expect(result).toHaveLength(2);
      expect(result[0]).toEqual({ role: "user", content: "Hello" });
    });

    it("merges consecutive messages with same role", () => {
      const result = anthropicAdapter.formatMessages([
        { role: "user", content: "Hello" },
        { role: "user", content: "World" },
      ]) as Array<{ role: string; content: string | unknown[] }>;

      expect(result).toHaveLength(1);
      expect(Array.isArray(result[0]?.content)).toBe(true);
      expect(result[0]?.content as unknown[]).toHaveLength(2);
    });
  });

  describe("openResponsesAdapter.parseStreamChunk", () => {
    it("extracts text delta from responses format", () => {
      const chunks: string[] = [];
      openResponsesAdapter.parseStreamChunk(
        { type: "response.output_text.delta", delta: "Hello" },
        "",
        { onChunk: (c) => chunks.push(c) }
      );
      expect(chunks).toEqual(["Hello"]);
    });

    it("extracts reasoning delta", () => {
      const reasoning: string[] = [];
      openResponsesAdapter.parseStreamChunk(
        { type: "response.reasoning_text.delta", delta: "thinking..." },
        "",
        { onChunk: () => {}, onReasoningChunk: (c) => reasoning.push(c) }
      );
      expect(reasoning).toEqual(["thinking..."]);
    });

    it("extracts function call arguments delta", () => {
      const toolCalls: Array<[number, string, string, string]> = [];
      openResponsesAdapter.parseStreamChunk(
        { type: "response.function_call_arguments.delta", output_index: 0, delta: '{"loc":"NY"}' },
        "",
        {
          onChunk: () => {},
          onToolCall: (i, id, name, args) => toolCalls.push([i, id, name, args]),
        }
      );
      expect(toolCalls).toEqual([[0, "", "", '{"loc":"NY"}']]);
    });
  });
});
