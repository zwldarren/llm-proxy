import { reactive, ref } from "vue";
import { useI18n } from "vue-i18n";
import { storeToRefs } from "pinia";
import { chatApi } from "@/services/api/chat";
import { useChatStore } from "@/stores/chat";
import { getAdapterForEndpoint } from "@/adapters";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { ChatMessage, ContentPart } from "@/types/schemas";
import type { ChatRun, ChatRunStatus } from "@/types/runs";
import type { ToolDefinition } from "@/adapters/types";
import { makeRunId } from "@/utils/runs";

export type WebSearchContextSize = "low" | "medium" | "high";

export interface WebSearchConfig {
  enabled: boolean;
  maxUses: number | null;
  searchContextSize: WebSearchContextSize;
  includeSources: boolean;
}

export function getDefaultWebSearchConfig(): WebSearchConfig {
  return {
    enabled: false,
    maxUses: null,
    searchContextSize: "medium",
    includeSources: false,
  };
}

export function loadWebSearchConfig(): WebSearchConfig {
  const defaults = getDefaultWebSearchConfig();
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.CHAT_WEB_SEARCH);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<WebSearchConfig>;
      return {
        enabled: typeof parsed.enabled === "boolean" ? parsed.enabled : defaults.enabled,
        maxUses:
          parsed.maxUses === null || typeof parsed.maxUses === "number"
            ? parsed.maxUses
            : defaults.maxUses,
        searchContextSize: ["low", "medium", "high"].includes(parsed.searchContextSize as string)
          ? (parsed.searchContextSize as WebSearchContextSize)
          : defaults.searchContextSize,
        includeSources:
          typeof parsed.includeSources === "boolean"
            ? parsed.includeSources
            : defaults.includeSources,
      };
    }
  } catch {
    // ignore parse errors and fall back to defaults
  }
  return defaults;
}

/**
 * Advanced chat options for model parameters
 */
export interface ChatOptions {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  system_prompt?: string;
  reasoning_effort?: string;
  tools?: ToolDefinition[];
  isToolResponse?: boolean;
  webSearch?: WebSearchConfig;
  // Allow arbitrary custom parameters
  [key: string]: unknown;
}

export function useChat() {
  const { t } = useI18n();
  const store = useChatStore();
  const { messages, isLoading, error } = storeToRefs(store);

  // AbortController for cancelling in-flight streaming requests
  const abortController = ref<AbortController | null>(null);

  // Create a new AbortController for the current request
  const createAbortController = () => {
    // Abort any existing request
    if (abortController.value) {
      abortController.value.abort();
    }
    abortController.value = new AbortController();
    return abortController.value;
  };

  /**
   * Stop the current streaming generation.
   * This will abort the in-flight request and mark the message as stopped.
   */
  const stopGeneration = () => {
    if (abortController.value) {
      abortController.value.abort();
      abortController.value = null;
      store.setLoading(false);
    }
  };

  const sendMessage = async (
    content: string | ContentPart[],
    model: string,
    apiKey: string,
    endpoint: string,
    onChunk?: (chunk: string) => void,
    options?: ChatOptions
  ) => {
    const isToolResponse = options?.isToolResponse === true;
    const hasContent =
      typeof content === "string"
        ? content.trim() !== ""
        : Array.isArray(content) && content.length > 0;

    if (!isToolResponse && (!hasContent || !model || !apiKey.trim())) return;

    // Push User message if not tool call resume
    if (!isToolResponse) {
      const userMessage: ChatMessage = {
        role: "user",
        content: typeof content === "string" ? content.trim() : content,
      };
      store.pushMessage(userMessage);
    }

    // Get the protocol adapter for this endpoint
    const adapter = getAdapterForEndpoint(endpoint);
    const systemPrompt = options?.system_prompt;
    const requestMessages = adapter.formatMessages(store.messages, systemPrompt);

    // Build endpoint request payload
    const requestPayload: Record<string, unknown> = {
      model,
    };

    if (endpoint.includes("chat/completions") || endpoint.includes("messages")) {
      requestPayload["messages"] = requestMessages;
      if (endpoint.includes("messages") && systemPrompt?.trim()) {
        requestPayload["system"] = systemPrompt.trim();
      }
    } else if (endpoint.includes("responses")) {
      requestPayload["input"] = requestMessages;
    }

    // Pass additional options
    if (options) {
      const skipOptions = new Set([
        "system_prompt",
        "tools",
        "isToolResponse",
        "temperatureEnabled",
        "maxTokensEnabled",
        "topPEnabled",
        "frequencyPenaltyEnabled",
        "presencePenaltyEnabled",
        "reasoningEffortEnabled",
        "webSearch",
      ]);
      for (const [k, v] of Object.entries(options)) {
        if (!skipOptions.has(k) && v !== undefined && v !== null && v !== "") {
          requestPayload[k] = v;
        }
      }

      // Format native web search tools for Anthropic/OpenAI endpoints
      const wsConfig = options.webSearch;
      if (wsConfig?.enabled) {
        if (endpoint.includes("messages")) {
          const anthropicTool: Record<string, unknown> = {
            type: "web_search_20260318",
            name: "web_search",
          };
          if (wsConfig.maxUses !== null) {
            anthropicTool["max_uses"] = wsConfig.maxUses;
          }
          const existingTools = Array.isArray(requestPayload["tools"])
            ? requestPayload["tools"]
            : [];
          requestPayload["tools"] = [...existingTools, anthropicTool];
        } else if (endpoint.includes("responses")) {
          const openaiTool: Record<string, unknown> = {
            type: "web_search",
            search_context_size: wsConfig.searchContextSize,
          };
          if (wsConfig.includeSources) {
            const include = Array.isArray(requestPayload["include"])
              ? requestPayload["include"]
              : [];
            requestPayload["include"] = [...include, "web_search_call.action.sources"];
          }
          const existingTools = Array.isArray(requestPayload["tools"])
            ? requestPayload["tools"]
            : [];
          requestPayload["tools"] = [...existingTools, openaiTool];
        } else if (endpoint.includes("chat/completions")) {
          requestPayload["web_search_options"] = {
            search_context_size: wsConfig.searchContextSize,
          };
        }
      }

      // Format user-defined tools via the adapter
      if (options.tools && options.tools.length > 0) {
        const formatted = adapter.formatTools(options.tools);
        if (formatted) {
          const currentTools = Array.isArray(requestPayload["tools"])
            ? requestPayload["tools"]
            : [];
          requestPayload["tools"] = [...currentTools, ...formatted];
        }
      }
    }

    // Create abort controller for this request
    const controller = createAbortController();
    const { signal } = controller;

    store.setLoading(true);
    store.setError(null);

    // Run telemetry for the specimen tray: one record per API request, emitted
    // when the request starts and again when it settles. Stored in the chat
    // store so the tray survives route navigation (session-scoped, never
    // persisted — payloads can be large).
    const emitRun = (run: ChatRun) => store.upsertRun(run);
    const runStart = performance.now();
    const runBase: ChatRun = {
      id: makeRunId(),
      endpoint,
      model,
      // The exact wire payload: JSON round-trip mirrors what JSON.stringify
      // sends (drops undefined values, e.g. adapter-set tool_calls: undefined).
      payload: JSON.parse(JSON.stringify({ ...requestPayload, stream: true })),
      startedAt: Date.now(),
      status: "streaming",
    };
    let runError: string | undefined;
    let runStopped = false;
    const settleRun = (
      status: ChatRunStatus,
      overrides?: Partial<Pick<ChatRun, "payload" | "responseChars">>
    ) => {
      emitRun({
        ...runBase,
        ...overrides,
        status,
        latencyMs: Math.round(performance.now() - runStart),
        errorMessage: runError,
      });
    };

    const handleAudioSpeech = async () => {
      const assistantMessage = reactive<ChatMessage>({
        role: "assistant",
        content: t("chat.generatingAudio") + "...",
      });
      store.pushMessage(assistantMessage);

      const voice = String(options?.voice ?? "alloy");
      const speed = Number(options?.speed ?? 1.0);
      const response_format = String(options?.response_format ?? "mp3");

      let rawText = "";
      if (typeof content === "string") {
        rawText = content;
      } else if (Array.isArray(content)) {
        rawText = content
          .filter((p): p is { type: "text"; text: string } => p.type === "text")
          .map((p) => p.text)
          .join("\n");
      }

      const audioPayload = { model, input: rawText.trim(), voice, speed, response_format };
      emitRun({ ...runBase, payload: audioPayload });

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify(audioPayload),
        });

        if (!response.ok) {
          let errData;
          try {
            errData = await response.json();
          } catch {
            errData = await response.text();
          }
          const errMsg = errData?.error?.message || errData?.message || `HTTP ${response.status}`;
          throw new Error(errMsg);
        }

        const blob = await response.blob();

        // Revoke previous blob URL to prevent memory leak
        const prevAudioUrl = assistantMessage.audioUrl;
        if (prevAudioUrl && prevAudioUrl.startsWith("blob:")) {
          URL.revokeObjectURL(prevAudioUrl);
        }

        const audioUrl = URL.createObjectURL(blob);
        assistantMessage.content = `${t("chat.audioGeneratedSuccess")}\n\n*Voice: ${voice}, Speed: ${speed}x*`;
        assistantMessage.audioUrl = audioUrl;
        assistantMessage.explicitAudio = true;
        settleRun("ok", { payload: audioPayload });
      } catch (e: unknown) {
        console.error(e);
        const errorObj = e as Error;
        const errMsg = errorObj.message || t("dialogs.errorGenerating");
        store.setError(errMsg);
        assistantMessage.content = `**Error:** ${errMsg}`;
        runError = errMsg;
        settleRun("error", { payload: audioPayload });
      } finally {
        store.setLoading(false);
      }
    };

    if (endpoint === "/v1/audio/speech") {
      await handleAudioSpeech();
      return;
    }

    emitRun(runBase);

    // Add placeholder assistant message
    const assistantMessage = reactive<ChatMessage>({
      role: "assistant",
      content: "",
      reasoning_content: "",
      tool_calls: [],
    });
    store.pushMessage(assistantMessage);

    try {
      await chatApi.streamChatCompletion(
        endpoint,
        requestPayload,
        apiKey,
        (chunk) => {
          assistantMessage.content += chunk;
          if (onChunk) onChunk(chunk);
        },
        (streamError) => {
          const errorMsg = `Stream error: ${streamError}`;
          store.setError(errorMsg);
          assistantMessage.content += `**Error:** ${errorMsg}`;
          runError = errorMsg;
        },
        (reasoningChunk) => {
          assistantMessage.reasoning_content =
            (assistantMessage.reasoning_content || "") + reasoningChunk;
          if (onChunk) onChunk("");
        },
        (index, id, name, args) => {
          if (!assistantMessage.tool_calls) {
            assistantMessage.tool_calls = [];
          }
          if (!assistantMessage.tool_calls[index]) {
            assistantMessage.tool_calls[index] = {
              id: "",
              type: "function",
              function: { name: "", arguments: "" },
            };
          }
          const tc = assistantMessage.tool_calls[index];
          if (id) tc.id = id;
          if (name) tc.function.name = name;
          if (args) tc.function.arguments += args;
          if (onChunk) onChunk("");
        },
        (_index, id, query, status) => {
          if (!assistantMessage.web_search_calls) {
            assistantMessage.web_search_calls = [];
          }
          // Use id as the key to avoid duplicates from mismatched output_index values
          const existingIdx = assistantMessage.web_search_calls.findIndex((c) => c && c.id === id);
          if (existingIdx >= 0) {
            assistantMessage.web_search_calls[existingIdx] = { id, query, status };
          } else {
            assistantMessage.web_search_calls.push({ id, query, status });
          }
          if (onChunk) onChunk("");
        },
        signal
      );
    } catch (e) {
      // Check if this was an abort error
      if (e instanceof Error && e.name === "AbortError") {
        assistantMessage.content += "\n\n*[Generation stopped]*";
        runStopped = true;
      } else {
        console.error(e);
        const errorMsg = t("dialogs.errorGenerating");
        store.setError(errorMsg);
        assistantMessage.content += `\n\n**Error:** ${errorMsg}`;
        runError = e instanceof Error ? e.message : errorMsg;
      }
    } finally {
      abortController.value = null;
      store.setLoading(false);
      settleRun(runStopped ? "stopped" : runError ? "error" : "ok", {
        responseChars: assistantMessage.content.length,
      });
    }
  };

  const clearChat = () => {
    store.clearMessages();
  };

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearChat,
    stopGeneration,
  };
}
