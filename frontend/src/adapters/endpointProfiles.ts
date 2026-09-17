/**
 * Endpoint profiles — the single source of truth for how the Chat console turns
 * its protocol-neutral settings into a request body.
 *
 * The console can target three protocols and they do not share a parameter
 * vocabulary. The mapping below mirrors the backend protocol schemas in
 * `src/llm_proxy/protocols`: a parameter an endpoint cannot carry is reported
 * as ignored instead of being sent and silently dropped upstream.
 *
 * | Panel setting    | /v1/chat/completions | /v1/messages        | /v1/responses           |
 * | ---------------- | -------------------- | ------------------- | ----------------------- |
 * | system prompt    | system message       | `system`            | `instructions`          |
 * | max tokens       | `max_tokens`         | `max_tokens`        | `max_output_tokens`     |
 * | reasoning effort | `reasoning_effort`   | `reasoning_effort`  | `reasoning.effort`      |
 * | web search       | `web_search_options` | `web_search_*` tool | `web_search` hosted tool |
 * | frequency/presence | supported          | ignored by Anthropic | supported               |
 */

import { getAdapterForEndpoint } from "@/adapters";
import type { ChatMessage } from "@/types/schemas";
import type { ToolDefinition } from "./types";

export type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

/** Identifiers for the controls shown in the advanced settings panel. */
export type AdvancedSettingId =
  | "systemPrompt"
  | "temperature"
  | "maxTokens"
  | "topP"
  | "frequencyPenalty"
  | "presencePenalty"
  | "reasoningEffort"
  | "webSearch"
  | "tools"
  | "customParams";

/** Panel defaults, used to tell "configured" from "left alone". */
export const DEFAULT_GENERATION_VALUES = {
  temperature: 0.7,
  topP: 1,
  frequencyPenalty: 0,
  presencePenalty: 0,
} as const;

/** Panel labels, kept as i18n keys so this module stays free of translations. */
export const SETTING_LABEL_KEYS: Record<AdvancedSettingId, string> = {
  systemPrompt: "chat.systemPrompt",
  temperature: "chat.temperature",
  maxTokens: "chat.maxTokens",
  topP: "chat.topP",
  frequencyPenalty: "chat.frequencyPenalty",
  presencePenalty: "chat.presencePenalty",
  reasoningEffort: "chat.reasoningEffort",
  webSearch: "chat.webSearch",
  tools: "chat.toolsDefinition",
  customParams: "chat.customVariables",
};

/** Protocol-neutral generation settings collected from the panel. */
export interface GenerationSettings {
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  reasoningEffort?: ReasoningEffort;
}

/** Web search options as shaped by the panel (structurally `WebSearchConfig`). */
export interface WebSearchRequestOptions {
  enabled: boolean;
  maxUses: number | null;
  searchContextSize: "low" | "medium" | "high";
  includeSources: boolean;
}

/** `sent`: in the body. `ignored`: endpoint has no equivalent. `rejected`: would corrupt the body. */
export type SettingStatus = "sent" | "ignored" | "rejected";

/** What the selected endpoint does with one configured panel value. */
export interface SettingEffect {
  id: AdvancedSettingId;
  /** Custom parameter name, only set when `id` is `customParams`. */
  key?: string;
  status: SettingStatus;
  /** Wire field(s) carrying the value; empty when ignored or rejected. */
  fields: string[];
  /** True when the value still equals the panel default. */
  isDefault: boolean;
  /** Optional i18n key with an endpoint-specific explanation. */
  noteKey?: string;
  noteParams?: Record<string, string | number>;
}

export interface ChatEndpointProfile {
  /** Selector value and request path. */
  path: string;
  /** Protocol name shown in the UI (i18n key). */
  labelKey: string;
  /** One-line description of the wire dialect (i18n key). */
  hintKey: string;
  /** Wire field per panel setting; `null` when the endpoint cannot carry it. */
  fields: Record<AdvancedSettingId, string | null>;
  /** Highest sampling temperature the protocol accepts. */
  temperatureMax: number;
}

export const CHAT_ENDPOINTS: ChatEndpointProfile[] = [
  {
    path: "/v1/chat/completions",
    labelKey: "chat.endpoints.chatCompletions",
    hintKey: "chat.endpoints.chatCompletionsHint",
    temperatureMax: 2,
    fields: {
      systemPrompt: "messages[system]",
      temperature: "temperature",
      maxTokens: "max_tokens",
      topP: "top_p",
      frequencyPenalty: "frequency_penalty",
      presencePenalty: "presence_penalty",
      reasoningEffort: "reasoning_effort",
      webSearch: "web_search_options",
      tools: "tools",
      customParams: "custom",
    },
  },
  {
    path: "/v1/messages",
    labelKey: "chat.endpoints.messages",
    hintKey: "chat.endpoints.messagesHint",
    temperatureMax: 1,
    fields: {
      systemPrompt: "system",
      temperature: "temperature",
      maxTokens: "max_tokens",
      topP: "top_p",
      // Anthropic Messages has no repetition penalties; the proxy drops them.
      frequencyPenalty: null,
      presencePenalty: null,
      reasoningEffort: "reasoning_effort",
      webSearch: "web_search",
      tools: "tools",
      customParams: "custom",
    },
  },
  {
    path: "/v1/responses",
    labelKey: "chat.endpoints.responses",
    hintKey: "chat.endpoints.responsesHint",
    temperatureMax: 2,
    fields: {
      systemPrompt: "instructions",
      temperature: "temperature",
      maxTokens: "max_output_tokens",
      topP: "top_p",
      frequencyPenalty: "frequency_penalty",
      presencePenalty: "presence_penalty",
      reasoningEffort: "reasoning.effort",
      webSearch: "web_search",
      tools: "tools",
      customParams: "custom",
    },
  },
];

export const DEFAULT_CHAT_ENDPOINT = CHAT_ENDPOINTS[0]!.path;

/** Profile for an endpoint, falling back to Chat Completions for unknown paths. */
export function getEndpointProfile(endpoint: string): ChatEndpointProfile {
  return CHAT_ENDPOINTS.find((profile) => profile.path === endpoint) ?? CHAT_ENDPOINTS[0]!;
}

/** Panel values the selected endpoint cannot carry, e.g. penalties on /v1/messages. */
export function unsupportedSettings(endpoint: string): AdvancedSettingId[] {
  const profile = getEndpointProfile(endpoint);
  return (Object.keys(profile.fields) as AdvancedSettingId[]).filter(
    (id) => profile.fields[id] === null
  );
}

/** Whether an endpoint path is one the console can select. */
export function isChatEndpoint(endpoint: string): boolean {
  return CHAT_ENDPOINTS.some((profile) => profile.path === endpoint);
}

/** Body keys the builder owns; custom parameters may not shadow them. */
const RESERVED_BODY_KEYS = new Set([
  "model",
  "messages",
  "input",
  "instructions",
  "system",
  "stream",
  "tools",
]);

export interface ChatRequestPlanInput {
  endpoint: string;
  model: string;
  /** Conversation as stored in the chat store; adapters format it per protocol. */
  messages: ChatMessage[];
  settings: GenerationSettings;
  tools?: ToolDefinition[];
  webSearch?: WebSearchRequestOptions;
  /** Free-form parameters from the panel. */
  customParams?: Record<string, unknown>;
}

export interface ChatRequestPlan {
  payload: Record<string, unknown>;
  /** One entry per configured panel value, in panel order. */
  effects: SettingEffect[];
}

type EffectInput = Omit<ChatRequestPlanInput, "messages" | "model">;

function baseEffect(id: AdvancedSettingId, fields: string[], isDefault = false): SettingEffect {
  return { id, status: "sent", fields, isDefault };
}

/**
 * Report what each configured panel value becomes on the selected endpoint.
 *
 * Shared by {@link planChatRequest} (so the payload cannot drift from the
 * report) and by the header indicator, which lights up exactly when this list
 * is non-empty.
 */
export function collectSettingEffects(input: EffectInput): SettingEffect[] {
  const profile = getEndpointProfile(input.endpoint);
  const { settings } = input;
  const effects: SettingEffect[] = [];
  const ignored = (id: AdvancedSettingId): SettingEffect => ({
    id,
    status: "ignored",
    fields: [],
    isDefault: false,
    noteKey: "chat.settingsEffects.notSentByEndpoint",
    noteParams: { endpoint: profile.path },
  });

  if (settings.systemPrompt?.trim()) {
    effects.push(baseEffect("systemPrompt", [profile.fields.systemPrompt ?? "system"]));
  }
  if (settings.temperature !== undefined) {
    const effect = baseEffect(
      "temperature",
      ["temperature"],
      settings.temperature === DEFAULT_GENERATION_VALUES.temperature
    );
    if (settings.temperature > profile.temperatureMax) {
      effect.noteKey = "chat.settingsEffects.temperatureClamped";
      effect.noteParams = { value: profile.temperatureMax };
    }
    effects.push(effect);
  }
  if (settings.maxTokens !== undefined) {
    effects.push(baseEffect("maxTokens", [profile.fields.maxTokens ?? "max_tokens"]));
  }
  if (settings.topP !== undefined) {
    effects.push(baseEffect("topP", ["top_p"], settings.topP === DEFAULT_GENERATION_VALUES.topP));
  }
  for (const id of ["frequencyPenalty", "presencePenalty"] as const) {
    const value = settings[id];
    if (value === undefined) continue;
    const field = profile.fields[id];
    effects.push(
      field ? baseEffect(id, [field], value === DEFAULT_GENERATION_VALUES[id]) : ignored(id)
    );
  }
  if (settings.reasoningEffort) {
    const field = profile.fields.reasoningEffort ?? "reasoning_effort";
    const effect = baseEffect("reasoningEffort", [field]);
    if (profile.path === "/v1/responses") {
      effect.noteKey = "chat.settingsEffects.reasoningResponses";
    } else if (profile.path === "/v1/messages") {
      effect.noteKey = "chat.settingsEffects.reasoningAnthropic";
    }
    effects.push(effect);
  }
  if (input.webSearch?.enabled) {
    effects.push(baseEffect("webSearch", [profile.fields.webSearch ?? "web_search"], false));
  }
  if (input.tools?.length) {
    effects.push(baseEffect("tools", ["tools"]));
  }
  for (const [key, value] of Object.entries(input.customParams ?? {})) {
    if (value === undefined || value === null || value === "") continue;
    effects.push(
      RESERVED_BODY_KEYS.has(key)
        ? {
            id: "customParams" as const,
            key,
            status: "rejected" as const,
            fields: [],
            isDefault: false,
            noteKey: "chat.settingsEffects.customParamRejected",
            noteParams: { key },
          }
        : {
            id: "customParams" as const,
            key,
            status: "sent" as const,
            fields: [key],
            isDefault: false,
          }
    );
  }
  return effects;
}

/** Build the wire payload for the selected endpoint. */
export function planChatRequest(input: ChatRequestPlanInput): ChatRequestPlan {
  const profile = getEndpointProfile(input.endpoint);
  const adapter = getAdapterForEndpoint(input.endpoint);
  const systemPrompt = input.settings.systemPrompt?.trim() || undefined;
  const payload: Record<string, unknown> = { model: input.model };

  // The Responses API carries the system prompt in `instructions`; the other
  // protocols inline it (Chat Completions) or put it next to `messages`.
  const adapterMessages =
    profile.path === "/v1/responses"
      ? adapter.formatMessages(input.messages)
      : adapter.formatMessages(input.messages, systemPrompt);

  if (profile.path === "/v1/responses") {
    payload["input"] = adapterMessages;
    if (systemPrompt) payload["instructions"] = systemPrompt;
  } else {
    payload["messages"] = adapterMessages;
    if (profile.path === "/v1/messages" && systemPrompt) payload["system"] = systemPrompt;
  }

  const { settings } = input;
  if (settings.temperature !== undefined) {
    // Anthropic rejects temperatures above 1.
    payload["temperature"] = Math.min(settings.temperature, profile.temperatureMax);
  }
  if (settings.maxTokens !== undefined) {
    payload[profile.fields.maxTokens ?? "max_tokens"] = settings.maxTokens;
  }
  if (settings.topP !== undefined) payload["top_p"] = settings.topP;
  if (settings.frequencyPenalty !== undefined && profile.fields.frequencyPenalty) {
    payload["frequency_penalty"] = settings.frequencyPenalty;
  }
  if (settings.presencePenalty !== undefined && profile.fields.presencePenalty) {
    payload["presence_penalty"] = settings.presencePenalty;
  }
  if (settings.reasoningEffort) {
    if (profile.path === "/v1/responses") {
      // Native Responses shape. `summary` is what lets the console render the
      // thinking panel — the API streams reasoning as summary deltas.
      payload["reasoning"] = { effort: settings.reasoningEffort, summary: "auto" };
    } else {
      payload["reasoning_effort"] = settings.reasoningEffort;
    }
  }

  const webSearch = input.webSearch;
  if (webSearch?.enabled) {
    const existingTools = Array.isArray(payload["tools"]) ? (payload["tools"] as unknown[]) : [];
    if (profile.path === "/v1/messages") {
      const tool: Record<string, unknown> = { type: "web_search_20260318", name: "web_search" };
      if (webSearch.maxUses !== null) tool["max_uses"] = webSearch.maxUses;
      payload["tools"] = [...existingTools, tool];
    } else if (profile.path === "/v1/responses") {
      payload["tools"] = [
        ...existingTools,
        { type: "web_search", search_context_size: webSearch.searchContextSize },
      ];
      if (webSearch.includeSources) {
        const include = Array.isArray(payload["include"]) ? (payload["include"] as string[]) : [];
        payload["include"] = [...include, "web_search_call.action.sources"];
      }
    } else {
      payload["web_search_options"] = { search_context_size: webSearch.searchContextSize };
    }
  }

  if (input.tools?.length) {
    const formatted = adapter.formatTools(input.tools);
    if (formatted) {
      const existingTools = Array.isArray(payload["tools"]) ? (payload["tools"] as unknown[]) : [];
      payload["tools"] = [...existingTools, ...formatted];
    }
  }

  // Free-form parameters land last so they can refine anything not reserved.
  const effects = collectSettingEffects(input);
  const rejected = new Set(
    effects
      .filter((effect) => effect.id === "customParams" && effect.status === "rejected")
      .map((effect) => effect.key)
  );
  for (const [key, value] of Object.entries(input.customParams ?? {})) {
    if (rejected.has(key)) continue;
    if (value === undefined || value === null || value === "") continue;
    payload[key] = value;
  }

  return { payload, effects };
}
