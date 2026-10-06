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

  describe("anthropicAdapter.parseStreamChunk web search", () => {
    /** Replay the exact frame sequence the proxy emits for an intercepted search. */
    const replayInterceptedSearch = (callbacks: Record<string, unknown>) => {
      const frames = [
        {
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "server_tool_use",
            id: "call_ws_1",
            name: "web_search",
            input: {},
          },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"query": "latest news"}' },
        },
        { type: "content_block_stop", index: 0 },
        {
          type: "content_block_start",
          index: 1,
          content_block: {
            type: "web_search_tool_result",
            tool_use_id: "call_ws_1",
            content: [{ type: "web_search_result", url: "https://example.com/1", title: "One" }],
          },
        },
        { type: "content_block_stop", index: 1 },
        {
          type: "content_block_start",
          index: 2,
          content_block: { type: "text", text: "" },
        },
        {
          type: "content_block_delta",
          index: 2,
          delta: { type: "text_delta", text: "The answer is 42." },
        },
        { type: "content_block_stop", index: 2 },
      ];
      for (const frame of frames) {
        anthropicAdapter.parseStreamChunk(frame, "", callbacks as never);
      }
    };

    it("surfaces server_tool_use as a web-search trace, not a phantom tool call", () => {
      const text: string[] = [];
      const toolCalls: unknown[] = [];
      const wsCalls: { id: string; q: string; s: string }[] = [];
      // One callbacks object per stream — mirrors chat.ts production usage.
      const callbacks = {
        onChunk: (c: string) => text.push(c),
        onToolCall: (...args: unknown[]) => toolCalls.push(args),
        onWebSearchCall: (_i: number, id: string, q: string, s: string) =>
          wsCalls.push({ id, q, s }),
      };

      replayInterceptedSearch(callbacks);

      expect(text.join("")).toBe("The answer is 42.");
      expect(toolCalls).toEqual([]);
      expect(wsCalls).toEqual([
        { id: "call_ws_1", q: "", s: "in_progress" },
        { id: "call_ws_1", q: "latest news", s: "completed" },
      ]);
    });

    it("marks the trace failed when the result block carries an error payload", () => {
      const wsCalls: { id: string; q: string; s: string }[] = [];
      const callbacks = {
        onChunk: () => {},
        onWebSearchCall: (_i: number, id: string, q: string, s: string) =>
          wsCalls.push({ id, q, s }),
      };
      anthropicAdapter.parseStreamChunk(
        {
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "server_tool_use",
            id: "call_ws_1",
            name: "web_search",
            input: {},
          },
        },
        "",
        callbacks as never
      );
      anthropicAdapter.parseStreamChunk(
        {
          type: "content_block_start",
          index: 1,
          content_block: {
            type: "web_search_tool_result",
            tool_use_id: "call_ws_1",
            content: '{"type": "web_search_tool_result_error", "error_code": "unavailable"}',
          },
        },
        "",
        callbacks as never
      );

      expect(wsCalls).toEqual([
        { id: "call_ws_1", q: "", s: "in_progress" },
        { id: "call_ws_1", q: "", s: "failed" },
      ]);
    });

    it("still forwards regular tool_use blocks as executable tool calls", () => {
      const toolCalls: unknown[] = [];
      const callbacks = {
        onChunk: () => {},
        onToolCall: (...args: unknown[]) => toolCalls.push(args),
      };
      anthropicAdapter.parseStreamChunk(
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "tool_use", id: "call_1", name: "get_weather", input: {} },
        },
        "",
        callbacks as never
      );
      anthropicAdapter.parseStreamChunk(
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"city":' },
        },
        "",
        callbacks as never
      );

      expect(toolCalls).toEqual([
        [0, "call_1", "get_weather", ""],
        [0, "", "", '{"city":'],
      ]);
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

    it("extracts reasoning summary deltas emitted by the proxy", () => {
      const reasoning: string[] = [];
      openResponsesAdapter.parseStreamChunk(
        {
          type: "response.reasoning_summary_text.delta",
          output_index: 0,
          summary_index: 0,
          delta: "weighing options",
        },
        "",
        { onChunk: () => {}, onReasoningChunk: (c) => reasoning.push(c) }
      );
      expect(reasoning).toEqual(["weighing options"]);
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

  describe("web search replay across turns", () => {
    const anthropicResult = [
      {
        type: "web_search_result",
        url: "https://reuters.com/world/iran/",
        title: "Iran War",
        encrypted_content: "cmVsZXZhbnQ=",
      },
    ];

    it("anthropicAdapter captures the native result payload", () => {
      const calls: Array<{
        id: string;
        status: string;
        result: unknown;
        sources: unknown;
      }> = [];
      anthropicAdapter.parseStreamChunk(
        {
          type: "content_block_start",
          index: 1,
          content_block: {
            type: "web_search_tool_result",
            tool_use_id: "ws_1",
            content: anthropicResult,
          },
        },
        "",
        {
          onChunk: () => {},
          onWebSearchCall: (_i, id, _q, status, payload) =>
            calls.push({
              id,
              status,
              result: payload?.result,
              sources: payload?.sources,
            }),
        } as never
      );

      expect(calls).toEqual([
        {
          id: "ws_1",
          status: "completed",
          result: anthropicResult,
          sources: [{ url: "https://reuters.com/world/iran/", title: "Iran War" }],
        },
      ]);
    });

    it("anthropicAdapter replays the search as a server-tool exchange", () => {
      const result = anthropicAdapter.formatMessages([
        { role: "user", content: "news?" },
        {
          role: "assistant",
          content: "Here you go.",
          web_search_calls: [
            { id: "ws_1", query: "news", status: "completed", result: anthropicResult },
          ],
        },
        { role: "user", content: "thanks" },
      ]) as Array<{ role: string; content: unknown }>;

      expect(result[1]).toEqual({
        role: "assistant",
        content: [
          { type: "text", text: "Here you go." },
          { type: "server_tool_use", id: "ws_1", name: "web_search", input: { query: "news" } },
        ],
      });
      // The result message merges with the following user turn, as the API expects.
      expect(result[2]).toEqual({
        role: "user",
        content: [
          { type: "web_search_tool_result", tool_use_id: "ws_1", content: anthropicResult },
          { type: "text", text: "thanks" },
        ],
      });
    });

    it("anthropicAdapter ignores searches with no replayable result", () => {
      const result = anthropicAdapter.formatMessages([
        {
          role: "assistant",
          content: "answer",
          web_search_calls: [{ id: "ws_1", query: "news", status: "in_progress" }],
        },
      ]);
      expect(result).toEqual([{ role: "assistant", content: "answer" }]);
    });

    it("openResponsesAdapter captures action and sources", () => {
      const calls: Array<{ id: string; result: unknown; sources: unknown }> = [];
      openResponsesAdapter.parseStreamChunk(
        {
          type: "response.output_item.done",
          output_index: 2,
          item: {
            type: "web_search_call",
            id: "ws_1",
            status: "completed",
            action: {
              type: "search",
              query: "news",
              queries: ["news"],
              sources: [{ url: "https://reuters.com/world/iran/", title: "Iran War" }],
            },
          },
        },
        "",
        {
          onChunk: () => {},
          onWebSearchCall: (_i, id, _q, _s, payload) =>
            calls.push({ id, result: payload?.result, sources: payload?.sources }),
        } as never
      );

      expect(calls).toEqual([
        {
          id: "ws_1",
          result: {
            type: "search",
            query: "news",
            queries: ["news"],
            sources: [{ url: "https://reuters.com/world/iran/", title: "Iran War" }],
          },
          sources: [{ url: "https://reuters.com/world/iran/", title: "Iran War" }],
        },
      ]);
    });

    it("openResponsesAdapter replays the search as a web_search_call item", () => {
      const result = openResponsesAdapter.formatMessages([
        { role: "user", content: "news?" },
        {
          role: "assistant",
          content: "Here you go.",
          web_search_calls: [
            {
              id: "ws_1",
              query: "news",
              status: "completed",
              result: { type: "search", query: "news", queries: ["news"] },
            },
          ],
        },
      ]);

      expect(result[1]).toEqual({
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "Here you go." }],
      });
      expect(result[2]).toEqual({
        type: "web_search_call",
        id: "ws_1",
        status: "completed",
        action: { type: "search", query: "news", queries: ["news"] },
      });
    });

    it("openResponsesAdapter synthesizes an action for a result-less search", () => {
      const result = openResponsesAdapter.formatMessages([
        {
          role: "assistant",
          content: "answer",
          web_search_calls: [{ id: "ws_1", query: "news", status: "failed" }],
        },
      ]);

      expect(result[1]).toEqual({
        type: "web_search_call",
        id: "ws_1",
        status: "failed",
        action: { type: "search", query: "news", queries: ["news"] },
      });
    });
  });
});
