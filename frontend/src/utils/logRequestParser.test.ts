import { describe, expect, it } from "vitest";
import { parseLogRequest } from "./logRequestParser";

/**
 * Verifies request-body normalization across all three inbound chat
 * protocols: every parameter surfaced, tools parsed, system prompt and
 * messages with typed content blocks (including tool history).
 */

// --- /v1/chat/completions (OpenAI Chat) --------------------------------

const openaiChatBody = {
  model: "gpt-4o",
  temperature: 0.7,
  top_p: 0.9,
  max_completion_tokens: 1024,
  stream: true,
  seed: 42,
  stop: ["###", "END"],
  presence_penalty: 0.1,
  tool_choice: "auto",
  response_format: { type: "json_object" },
  messages: [
    { role: "system", content: "You are helpful." },
    { role: "user", content: "What's the weather in Paris?" },
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_abc",
          type: "function",
          function: { name: "get_weather", arguments: '{"location":"Paris"}' },
        },
      ],
    },
    { role: "tool", tool_call_id: "call_abc", content: '{"temp":20}' },
    {
      role: "user",
      content: [
        { type: "text", text: "And in this picture?" },
        { type: "image_url", image_url: { url: "data:image/png;base64," + "a".repeat(1000) } },
      ],
    },
  ],
  tools: [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get current weather",
        parameters: { type: "object", properties: { location: { type: "string" } } },
      },
    },
  ],
};

// --- /v1/messages (Anthropic) -------------------------------------------

const anthropicBody = {
  model: "claude-sonnet-4",
  max_tokens: 2048,
  temperature: 1,
  top_k: 40,
  stop_sequences: ["\n\nHuman:"],
  stream: true,
  system: [
    { type: "text", text: "System part one.", cache_control: { type: "ephemeral" } },
    { type: "text", text: "System part two." },
  ],
  thinking: { type: "enabled", budget_tokens: 1024 },
  tool_choice: { type: "tool", name: "get_weather" },
  messages: [
    { role: "user", content: "Weather?" },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "Need the weather tool.", signature: "sig" },
        { type: "text", text: "Let me check." },
        { type: "tool_use", id: "toolu_1", name: "get_weather", input: { location: "Paris" } },
      ],
    },
    {
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "toolu_1", content: "sunny, 20°C" },
        {
          type: "image",
          source: { type: "base64", media_type: "image/jpeg", data: "b".repeat(2000) },
        },
      ],
    },
  ],
  tools: [
    {
      name: "get_weather",
      description: "Get current weather",
      input_schema: { type: "object", properties: { location: { type: "string" } } },
    },
    { type: "web_search_20250305", name: "web_search", max_uses: 3 },
  ],
};

// --- /v1/responses (OpenAI Responses) -----------------------------------

const responsesBody = {
  model: "gpt-5",
  instructions: "Be terse.",
  max_output_tokens: 512,
  stream: true,
  reasoning: { effort: "medium" },
  text: { verbosity: "low" },
  parallel_tool_calls: false,
  store: false,
  input: [
    {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "hi" }],
    },
    {
      type: "function_call",
      id: "fc_1",
      call_id: "call_1",
      name: "get_weather",
      arguments: '{"location":"Paris"}',
    },
    { type: "function_call_output", call_id: "call_1", output: '{"temp":20}' },
    {
      type: "reasoning",
      id: "rs_1",
      content: [{ type: "reasoning_text", text: "thinking about weather" }],
    },
  ],
  tools: [
    {
      type: "function",
      name: "get_weather",
      description: "Weather",
      parameters: { type: "object" },
    },
    { type: "web_search_preview", search_context_size: "medium" },
    { type: "mcp", server_label: "deepwiki", server_url: "https://mcp.deepwiki.com/mcp" },
  ],
  tool_choice: { type: "function", name: "get_weather" },
};

describe("parseLogRequest", () => {
  describe("OpenAI Chat", () => {
    it("surfaces all scalar params, not just a hardcoded few", () => {
      const r = parseLogRequest(openaiChatBody)!;
      expect(r.protocol).toBe("openai-chat");
      const keys = r.scalarParams.map((p) => p.key);
      expect(keys).toContain("model");
      expect(keys).toContain("temperature");
      expect(keys).toContain("top_p");
      expect(keys).toContain("max_completion_tokens");
      expect(keys).toContain("stream");
      expect(keys).toContain("seed");
      expect(keys).toContain("presence_penalty");
      // Scalar arrays are joined
      expect(r.scalarParams.find((p) => p.key === "stop")?.value).toBe("###, END");
      // Complex params land in objectParams
      expect(r.objectParams.map((p) => p.key)).toContain("response_format");
      // Section keys never appear as generic params
      expect(keys).not.toContain("messages");
      expect(keys).not.toContain("tools");
    });

    it("extracts the leading system message into systemPrompt", () => {
      const r = parseLogRequest(openaiChatBody)!;
      expect(r.systemPrompt).toBe("You are helpful.");
      expect(r.messages.some((m) => m.role === "system")).toBe(false);
    });

    it("parses tool definitions with schemas", () => {
      const r = parseLogRequest(openaiChatBody)!;
      expect(r.tools).toHaveLength(1);
      expect(r.tools[0]).toMatchObject({
        name: "get_weather",
        kind: "function",
        description: "Get current weather",
      });
      expect(r.tools[0]?.schema).toMatchObject({ type: "object" });
      expect(r.toolChoice).toBe("auto");
    });

    it("normalizes messages with typed blocks incl. tool history", () => {
      const r = parseLogRequest(openaiChatBody)!;
      // user, assistant(tool_call), tool(result), user(multimodal)
      expect(r.messages).toHaveLength(4);

      const assistant = r.messages[1]!;
      const toolCall = assistant.blocks.find((b) => b.kind === "tool_call");
      expect(toolCall).toMatchObject({ kind: "tool_call", id: "call_abc", name: "get_weather" });
      expect(toolCall && toolCall.kind === "tool_call" ? toolCall.parsedArguments : {}).toEqual({
        location: "Paris",
      });

      const toolMsg = r.messages[2]!;
      expect(toolMsg.role).toBe("tool");
      expect(toolMsg.toolCallId).toBe("call_abc");
      expect(toolMsg.blocks[0]).toMatchObject({ kind: "tool_result", output: '{"temp":20}' });

      const multimodal = r.messages[3]!;
      const image = multimodal.blocks.find((b) => b.kind === "image");
      expect(image).toBeDefined();
      // base64 is reduced to a byte estimate, never inlined
      expect(image && image.kind === "image" ? image.bytes : 0).toBe(750);
      expect(JSON.stringify(multimodal)).not.toContain("a".repeat(100));
    });
  });

  describe("Anthropic", () => {
    it("flattens a system block array into the system prompt", () => {
      const r = parseLogRequest(anthropicBody)!;
      expect(r.protocol).toBe("anthropic");
      expect(r.systemPrompt).toBe("System part one.\n\nSystem part two.");
    });

    it("parses thinking/tool_use/tool_result blocks in order", () => {
      const r = parseLogRequest(anthropicBody)!;
      const assistant = r.messages[1]!;
      expect(assistant.blocks.map((b) => b.kind)).toEqual(["thinking", "text", "tool_call"]);

      const userResult = r.messages[2]!;
      expect(userResult.blocks[0]).toMatchObject({
        kind: "tool_result",
        id: "toolu_1",
        output: "sunny, 20°C",
      });
      const img = userResult.blocks[1]!;
      expect(img.kind).toBe("image");
      expect(img.kind === "image" ? img.mediaType : "").toBe("image/jpeg");
    });

    it("parses custom and server tool definitions", () => {
      const r = parseLogRequest(anthropicBody)!;
      expect(r.tools).toHaveLength(2);
      expect(r.tools[0]).toMatchObject({ name: "get_weather", kind: "function" });
      expect(r.tools[1]).toMatchObject({ name: "web_search", kind: "web_search_20250305" });
      expect(r.tools[1]?.summary).toContain("max_uses");
      expect(r.toolChoice).toBe("tool: get_weather");
    });

    it("keeps complex params (thinking) out of the scalar grid", () => {
      const r = parseLogRequest(anthropicBody)!;
      expect(r.scalarParams.map((p) => p.key)).not.toContain("thinking");
      expect(r.objectParams.map((p) => p.key)).toContain("thinking");
      expect(r.scalarParams.map((p) => p.key)).toContain("top_k");
      expect(r.scalarParams.find((p) => p.key === "stop_sequences")?.value).toBe("\n\nHuman:");
    });
  });

  describe("OpenAI Responses", () => {
    it("parses input items, instructions and tools", () => {
      const r = parseLogRequest(responsesBody)!;
      expect(r.protocol).toBe("responses");
      expect(r.systemPrompt).toBe("Be terse.");
      expect(r.messages).toHaveLength(4);

      expect(r.messages[0]?.role).toBe("user");
      expect(r.messages[0]?.plainText).toBe("hi");

      const fc = r.messages[1]!;
      expect(fc.role).toBe("assistant");
      expect(fc.blocks[0]).toMatchObject({ kind: "tool_call", name: "get_weather" });

      const out = r.messages[2]!;
      expect(out.role).toBe("tool");
      expect(out.blocks[0]).toMatchObject({ kind: "tool_result", id: "call_1" });

      const reasoning = r.messages[3]!;
      expect(reasoning.blocks[0]).toMatchObject({
        kind: "thinking",
        text: "thinking about weather",
      });
    });

    it("handles built-in tools with summaries", () => {
      const r = parseLogRequest(responsesBody)!;
      expect(r.tools.map((t) => t.kind)).toEqual(["function", "web_search_preview", "mcp"]);
      expect(r.tools[1]?.summary).toContain("search_context_size");
      expect(r.tools[2]?.summary).toContain("deepwiki");
      expect(r.toolChoice).toBe("function: get_weather");
      expect(r.objectParams.map((p) => p.key)).toEqual(
        expect.arrayContaining(["reasoning", "text"])
      );
      expect(r.scalarParams.find((p) => p.key === "parallel_tool_calls")?.value).toBe("false");
    });

    it("accepts a plain-string input", () => {
      const r = parseLogRequest({ model: "gpt-5", input: "hello" })!;
      expect(r.protocol).toBe("responses");
      expect(r.messages).toHaveLength(1);
      expect(r.messages[0]?.plainText).toBe("hello");
    });
  });

  describe("edge cases", () => {
    it("returns null for non-object bodies", () => {
      expect(parseLogRequest(null)).toBeNull();
      expect(parseLogRequest("not json")).toBeNull();
      expect(parseLogRequest(42)).toBeNull();
    });

    it("parses JSON string bodies", () => {
      const r = parseLogRequest(JSON.stringify({ model: "m", messages: [] }))!;
      expect(r.protocol).toBe("openai-chat");
    });

    it("marks non-chat shapes as non-chat-like", () => {
      const r = parseLogRequest({ prompt: "draw a cat", n: 1, size: "1024x1024" })!;
      expect(r.isChatLike).toBe(false);
      expect(r.scalarParams.map((p) => p.key)).toEqual(
        expect.arrayContaining(["prompt", "n", "size"])
      );
    });
  });
});
