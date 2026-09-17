import { describe, it, expect } from "vitest";
import {
  collectSettingEffects,
  getEndpointProfile,
  planChatRequest,
  unsupportedSettings,
  type ChatRequestPlanInput,
} from "@/adapters/endpointProfiles";
import type { ChatMessage } from "@/types/schemas";

const conversation: ChatMessage[] = [
  { role: "user", content: "Hello" },
  { role: "assistant", content: "Hi" },
  { role: "user", content: "Explain streaming" },
];

const plan = (overrides: Partial<ChatRequestPlanInput>) =>
  planChatRequest({
    endpoint: "/v1/chat/completions",
    model: "test-model",
    messages: conversation,
    settings: {},
    ...overrides,
  });

describe("endpoint profiles", () => {
  it("lists the three selectable protocols", () => {
    expect(getEndpointProfile("/v1/messages").labelKey).toBe("chat.endpoints.messages");
    expect(getEndpointProfile("/v1/responses").path).toBe("/v1/responses");
    // Unknown paths fall back to Chat Completions rather than throwing.
    expect(getEndpointProfile("/v1/unknown").path).toBe("/v1/chat/completions");
  });

  it("reports repetition penalties as unsupported on Anthropic Messages", () => {
    expect(unsupportedSettings("/v1/messages")).toEqual(["frequencyPenalty", "presencePenalty"]);
    expect(unsupportedSettings("/v1/chat/completions")).toEqual([]);
    expect(unsupportedSettings("/v1/responses")).toEqual([]);
  });
});

describe("planChatRequest: conversation shape", () => {
  it("inlines messages plus a system message for Chat Completions", () => {
    const { payload } = plan({ settings: { systemPrompt: "Be terse" } });
    expect(payload["instructions"]).toBeUndefined();
    expect(payload["system"]).toBeUndefined();
    expect(payload["input"]).toBeUndefined();
    expect((payload["messages"] as ChatMessage[])[0]).toEqual({
      role: "system",
      content: "Be terse",
    });
  });

  it("uses a top-level system field for Messages", () => {
    const { payload } = plan({ endpoint: "/v1/messages", settings: { systemPrompt: "Be terse" } });
    expect(payload["system"]).toBe("Be terse");
    expect(payload["messages"]).toHaveLength(3);
  });

  it("uses input items plus instructions for Responses", () => {
    const { payload } = plan({
      endpoint: "/v1/responses",
      settings: { systemPrompt: "Be terse" },
    });
    expect(payload["instructions"]).toBe("Be terse");
    expect(payload["messages"]).toBeUndefined();
    // The system prompt must not be duplicated as an input item.
    expect(payload["input"]).toHaveLength(3);
  });
});

describe("planChatRequest: generation parameters", () => {
  it("renames max tokens per endpoint", () => {
    expect(plan({ settings: { maxTokens: 1024 } }).payload["max_tokens"]).toBe(1024);
    expect(
      plan({ endpoint: "/v1/messages", settings: { maxTokens: 1024 } }).payload["max_tokens"]
    ).toBe(1024);
    const responses = plan({ endpoint: "/v1/responses", settings: { maxTokens: 1024 } }).payload;
    expect(responses["max_output_tokens"]).toBe(1024);
    expect(responses["max_tokens"]).toBeUndefined();
  });

  it("clamps temperature to the protocol range", () => {
    expect(plan({ settings: { temperature: 1.5 } }).payload["temperature"]).toBe(1.5);
    expect(
      plan({ endpoint: "/v1/messages", settings: { temperature: 1.5 } }).payload["temperature"]
    ).toBe(1);
  });

  it("drops repetition penalties the Anthropic protocol cannot carry", () => {
    const { payload, effects } = plan({
      endpoint: "/v1/messages",
      settings: { frequencyPenalty: 0.5, presencePenalty: 0.5 },
    });
    expect(payload["frequency_penalty"]).toBeUndefined();
    expect(payload["presence_penalty"]).toBeUndefined();
    const ignored = effects.filter((effect) => effect.status === "ignored");
    expect(ignored.map((effect) => effect.id)).toEqual(["frequencyPenalty", "presencePenalty"]);
    expect(ignored[0]?.noteKey).toBe("chat.settingsEffects.notSentByEndpoint");
  });

  it("sends repetition penalties on the OpenAI protocols", () => {
    const { payload } = plan({ settings: { frequencyPenalty: 0.5, presencePenalty: -0.5 } });
    expect(payload["frequency_penalty"]).toBe(0.5);
    expect(payload["presence_penalty"]).toBe(-0.5);
  });
});

describe("planChatRequest: reasoning", () => {
  it("uses reasoning_effort for Chat Completions and Messages", () => {
    const chat = plan({ settings: { reasoningEffort: "high" } }).payload;
    expect(chat["reasoning_effort"]).toBe("high");
    expect(chat["reasoning"]).toBeUndefined();

    const messages = plan({
      endpoint: "/v1/messages",
      settings: { reasoningEffort: "low" },
    }).payload;
    expect(messages["reasoning_effort"]).toBe("low");
    expect(messages["reasoning"]).toBeUndefined();
  });

  it("uses a reasoning object with an auto summary for Responses", () => {
    const { payload } = plan({ endpoint: "/v1/responses", settings: { reasoningEffort: "high" } });
    expect(payload["reasoning"]).toEqual({ effort: "high", summary: "auto" });
    expect(payload["reasoning_effort"]).toBeUndefined();
  });
});

describe("planChatRequest: web search", () => {
  const webSearch = {
    enabled: true,
    maxUses: 4,
    searchContextSize: "high" as const,
    includeSources: true,
  };

  it("uses web_search_options for Chat Completions", () => {
    const { payload } = plan({ webSearch });
    expect(payload["web_search_options"]).toEqual({ search_context_size: "high" });
    expect(payload["tools"]).toBeUndefined();
  });

  it("adds the Anthropic web_search tool with max_uses", () => {
    const { payload } = plan({ endpoint: "/v1/messages", webSearch });
    expect(payload["tools"]).toEqual([
      { type: "web_search_20260318", name: "web_search", max_uses: 4 },
    ]);
    expect(payload["web_search_options"]).toBeUndefined();
  });

  it("adds the hosted web_search tool and include list for Responses", () => {
    const { payload } = plan({ endpoint: "/v1/responses", webSearch });
    expect(payload["tools"]).toEqual([{ type: "web_search", search_context_size: "high" }]);
    expect(payload["include"]).toEqual(["web_search_call.action.sources"]);
  });

  it("keeps user tools next to the web search tool", () => {
    const { payload } = plan({
      endpoint: "/v1/responses",
      webSearch,
      tools: [{ name: "get_weather", description: "", parameters: "{}", enabled: true }],
    });
    expect(payload["tools"]).toHaveLength(2);
  });
});

describe("planChatRequest: custom parameters", () => {
  it("merges free-form parameters and rejects reserved body keys", () => {
    const { payload, effects } = plan({
      customParams: { seed: 42, top_k: 5, model: "hijack", stream: true },
    });
    expect(payload["seed"]).toBe(42);
    expect(payload["top_k"]).toBe(5);
    expect(payload["model"]).toBe("test-model");
    expect(payload["stream"]).toBeUndefined();

    const rejected = effects.filter((effect) => effect.status === "rejected");
    expect(rejected.map((effect) => effect.key)).toEqual(["model", "stream"]);
  });
});

describe("collectSettingEffects", () => {
  it("marks panel defaults so the indicator ignores them", () => {
    const effects = collectSettingEffects({
      endpoint: "/v1/chat/completions",
      settings: { temperature: 0.7, topP: 0.5 },
    });
    expect(effects.find((effect) => effect.id === "temperature")?.isDefault).toBe(true);
    expect(effects.find((effect) => effect.id === "topP")?.isDefault).toBe(false);
  });

  it("reports the wire field for the configured values only", () => {
    const effects = collectSettingEffects({
      endpoint: "/v1/responses",
      settings: { maxTokens: 256, reasoningEffort: "medium" },
    });
    expect(effects).toHaveLength(2);
    expect(effects[0]?.fields).toEqual(["max_output_tokens"]);
    expect(effects[1]?.noteKey).toBe("chat.settingsEffects.reasoningResponses");
  });

  it("flags a clamped temperature with the value it is sent as", () => {
    const effects = collectSettingEffects({
      endpoint: "/v1/messages",
      settings: { temperature: 1.8 },
    });
    expect(effects[0]?.noteKey).toBe("chat.settingsEffects.temperatureClamped");
    expect(effects[0]?.noteParams).toEqual({ value: 1 });
  });
});
