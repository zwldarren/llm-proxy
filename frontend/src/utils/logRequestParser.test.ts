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

describe("parseLogRequest — System One", () => {
  const body = {
    model: "jev-latest",
    state: "Help! My payouts fail.",
    questions: {
      is_urgent: { type: "noul", instructions: "Urgent?" },
      department: {
        type: "choice",
        instructions: "Which team?",
        criteria: { billing: "Payments", technical: "Bugs" },
      },
      frustration: { type: "score", instructions: "How angry?", criteria: ["Calm", "Angry"] },
    },
  };

  it("detects the systemone protocol and extracts state/questions", () => {
    const r = parseLogRequest(body)!;
    expect(r.protocol).toBe("systemone");
    expect(r.systemOne?.state).toBe("Help! My payouts fail.");
    expect(r.systemOne?.questions.map((q) => q.id)).toEqual([
      "is_urgent",
      "department",
      "frustration",
    ]);
    expect(r.systemOne?.questions[1]?.criteria).toEqual({ billing: "Payments", technical: "Bugs" });
  });

  it("keeps state/questions out of the generic parameter grid", () => {
    const r = parseLogRequest(body)!;
    const keys = [...r.scalarParams.map((p) => p.key), ...r.objectParams.map((p) => p.key)];
    expect(keys).not.toContain("state");
    expect(keys).not.toContain("questions");
    expect(r.scalarParams.map((p) => p.key)).toContain("model");
  });

  it("keeps state/questions in the parameter grid for a non-System-One body", () => {
    // Only a body rendered as System One treats these as dedicated sections;
    // any other shape must keep them visible as ordinary parameters.
    const r = parseLogRequest({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
      state: "legacy",
      questions: { a: 1 },
    })!;
    expect(r.protocol).toBe("openai-chat");
    expect(r.systemOne).toBeUndefined();
    expect(r.scalarParams.map((p) => p.key)).toContain("state");
    expect(r.objectParams.map((p) => p.key)).toContain("questions");
  });

  it("keeps unmodeled question fields and structured instructions in extra", () => {
    const r = parseLogRequest({
      state: "s",
      questions: {
        q1: {
          type: "choice",
          instructions: { prompt: "Pick one", locale: "en" },
          criteria: { a: "A" },
          weight: 3,
        },
      },
    })!;
    const question = r.systemOne!.questions[0]!;
    expect(question.instructions).toBeUndefined();
    expect(question.extra).toEqual({
      weight: 3,
      instructions: { prompt: "Pick one", locale: "en" },
    });
  });
});

describe("parseLogRequest — Decisions", () => {
  const body = {
    model: "gpt-6-luna",
    input: "I was charged twice for my order.",
    questions: [
      { type: "predicate", name: "urgent", instructions: "Is it urgent?" },
      {
        type: "choice",
        name: "department",
        instructions: "Which department?",
        choices: [
          { value: "billing", description: "Payments." },
          { value: true, description: "Flagged." },
          { value: "other" },
        ],
      },
      {
        type: "score",
        name: "severity",
        instructions: "How severe?",
        levels: [{ label: "Cosmetic", description: "Appearance only." }, { label: "Blocked" }],
      },
    ],
  };

  it("detects decisions rather than responses", () => {
    // Both shapes carry `input`; only decisions has an ordered `questions`
    // array, so the Responses branch must not win.
    const r = parseLogRequest(body)!;
    expect(r.protocol).toBe("openai-decisions");
    expect(r.systemOne?.state).toBe("I was charged twice for my order.");
    expect(r.systemOne?.questions.map((q) => q.id)).toEqual(["urgent", "department", "severity"]);
  });

  it("folds choices onto the option map the card renders", () => {
    const r = parseLogRequest(body)!;
    // Boolean option values render as their word, matching what the answer echoes.
    expect(r.systemOne?.questions[1]?.criteria).toEqual({
      billing: "Payments.",
      true: "Flagged.",
      other: "",
    });
  });

  it("folds level labels and descriptions onto the ordered rubric", () => {
    const r = parseLogRequest(body)!;
    expect(r.systemOne?.questions[2]?.criteria).toEqual(["Cosmetic — Appearance only.", "Blocked"]);
  });

  it("identifies an unnamed question by its position", () => {
    const r = parseLogRequest({
      model: "m",
      input: "x",
      questions: [{ type: "predicate", instructions: "Is it urgent?" }],
    })!;
    expect(r.systemOne?.questions[0]?.id).toBe("#0");
  });

  it("keeps input/questions out of the generic parameter grid", () => {
    const r = parseLogRequest(body)!;
    const keys = [...r.scalarParams.map((p) => p.key), ...r.objectParams.map((p) => p.key)];
    expect(keys).not.toContain("input");
    expect(keys).not.toContain("questions");
    expect(r.scalarParams.map((p) => p.key)).toContain("model");
  });

  it("keeps unmodeled question fields and structured instructions in extra", () => {
    const r = parseLogRequest({
      input: "x",
      questions: [
        {
          type: "predicate",
          name: "q1",
          instructions: { prompt: "Is it urgent?" },
          weight: 3,
        },
      ],
    })!;
    const question = r.systemOne!.questions[0]!;
    expect(question.instructions).toBeUndefined();
    expect(question.extra).toEqual({ weight: 3, instructions: { prompt: "Is it urgent?" } });
  });

  it("renders a message-array input without flattening it", () => {
    const r = parseLogRequest({
      input: [{ role: "user", content: [{ type: "input_text", text: "Inspect this photo." }] }],
      questions: [{ type: "predicate", instructions: "Damaged?" }],
    })!;
    expect(r.protocol).toBe("openai-decisions");
    expect(Array.isArray(r.systemOne?.state)).toBe(true);
  });

  it("trusts request_type when the shape alone would not qualify", () => {
    // A question whose type is not a Decisions kind only classifies as
    // decisions because the log records the request type.
    const r = parseLogRequest(
      { input: "x", questions: [{ type: "refusal", reason: "nope" }] },
      "decisions"
    )!;
    expect(r.protocol).toBe("openai-decisions");
    expect(r.systemOne?.questions).toHaveLength(1);
  });

  it("leaves an unrelated questions array on a Responses body alone", () => {
    const r = parseLogRequest({ input: "hello", questions: [{ question: "why" }] })!;
    expect(r.protocol).toBe("responses");
    expect(r.systemOne).toBeUndefined();
    expect(r.objectParams.map((p) => p.key)).toContain("questions");
  });

  it("skips a choice without a value instead of keying it 'undefined'", () => {
    const r = parseLogRequest({
      input: "x",
      questions: [
        {
          type: "choice",
          name: "q",
          choices: [{ value: "a", description: "A" }, { description: "no value" }],
        },
      ],
    })!;
    expect(r.systemOne?.questions[0]?.criteria).toEqual({ a: "A" });
  });

  it("keeps a malformed choices/levels field visible", () => {
    const r = parseLogRequest({
      input: "x",
      questions: [{ type: "choice", name: "q", choices: "not-an-array" }],
    })!;
    expect(r.systemOne?.questions[0]?.criteria).toBe("not-an-array");
  });

  it("omits the separator when a level has no label", () => {
    const r = parseLogRequest({
      input: "x",
      questions: [{ type: "score", name: "q", levels: [{ description: "Only a description." }] }],
    })!;
    expect(r.systemOne?.questions[0]?.criteria).toEqual(["Only a description."]);
  });

  it("reduces inline base64 images in input to a size placeholder", () => {
    const r = parseLogRequest({
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "Inspect this photo." },
            { type: "input_image", image_url: "data:image/png;base64," + "a".repeat(1000) },
          ],
        },
      ],
      questions: [{ type: "predicate", instructions: "Damaged?" }],
    })!;
    const state = r.systemOne?.state as Array<{ content: unknown[] }>;
    expect(JSON.stringify(state)).not.toContain("a".repeat(100));
    expect(state[0]?.content[1]).toMatchObject({
      type: "input_image",
      image_url: "[base64 image/png, 750 bytes]",
    });
  });
});
