import { reactive, ref } from "vue";
import { useI18n } from "vue-i18n";
import { storeToRefs } from "pinia";
import { chatApi } from "@/services/api/chat";
import { useChatStore } from "@/stores/chat";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { ChatMessage, ContentPart } from "@/types/schemas";
import type { ToolDefinition } from "@/adapters/types";
import type { GenerationSettings } from "@/adapters/endpointProfiles";
import { planChatRequest } from "@/adapters/endpointProfiles";

/**
 * Coalesce streamed deltas into one reactive write per animation frame.
 *
 * At token rate, each delta re-renders the whole message and re-runs the chat
 * store's deep watchers; batching bounds that work by frame instead of by
 * chunk. The buffer is flushed whenever the stream settles, so the finished
 * message is never missing text, and a backgrounded tab simply defers the
 * (invisible) render until it is visible again.
 */
function createDeltaBuffer(apply: (delta: string) => void) {
  let pending = "";
  let frame: number | null = null;

  const flush = () => {
    if (frame !== null) {
      cancelAnimationFrame(frame);
      frame = null;
    }
    if (!pending) return;
    const delta = pending;
    pending = "";
    apply(delta);
  };

  const push = (delta: string) => {
    if (!delta) return;
    pending += delta;
    if (frame !== null) return;
    if (typeof requestAnimationFrame !== "function") {
      flush();
      return;
    }
    frame = requestAnimationFrame(() => {
      frame = null;
      flush();
    });
  };

  return { push, flush };
}

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
 * Advanced chat options for a single request.
 *
 * `settings` carries protocol-neutral panel values; `adapters/endpointProfiles`
 * maps them onto the wire format of the selected endpoint.
 */
export interface ChatOptions {
  settings?: GenerationSettings;
  tools?: ToolDefinition[];
  /** Free-form parameters from the panel; merged last, reserved keys rejected. */
  customParams?: Record<string, unknown>;
  isToolResponse?: boolean;
  webSearch?: WebSearchConfig;
  // Speech-only options for /v1/audio/speech
  voice?: string;
  speed?: number;
  response_format?: string;
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

    // Build the endpoint request body. Every protocol-specific mapping (max
    // tokens, system prompt placement, reasoning, web search) lives in
    // `adapters/endpointProfiles` so the payload, the settings indicator, and
    // the panel hints can never disagree.
    const { payload: requestPayload } = planChatRequest({
      endpoint,
      model,
      messages: store.messages,
      settings: options?.settings ?? {},
      tools: options?.tools,
      webSearch: options?.webSearch,
      customParams: options?.customParams,
    });

    // Create abort controller for this request
    const controller = createAbortController();
    const { signal } = controller;

    store.setLoading(true);
    store.setError(null);

    let runFailed = false;
    let runStopped = false;

    // Every in-place mutation of the transcript must be paired with a dirty
    // mark so the debounced writer persists it (see stores/chat.ts). This
    // helper keeps the pairing atomic at each call site instead of relying on
    // every mutation remembering to call touchChat() itself.
    const mutateChat = (mutate: () => void) => {
      mutate();
      store.touchChat();
    };

    /** Record a failed generation: error banner plus a marked message. */
    const failGeneration = (message: ChatMessage, errorMsg: string) => {
      runFailed = true;
      store.setError(errorMsg);
      mutateChat(() => {
        message.failed = true;
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
        mutateChat(() => {
          assistantMessage.content = `${t("chat.audioGeneratedSuccess")}\n\n*Voice: ${voice}, Speed: ${speed}x*`;
          assistantMessage.audioUrl = audioUrl;
          assistantMessage.explicitAudio = true;
        });
      } catch (e: unknown) {
        console.error(e);
        const errorObj = e as Error;
        const errMsg = errorObj.message || t("dialogs.errorGenerating");
        store.setError(errMsg);
        mutateChat(() => {
          assistantMessage.content = `**Error:** ${errMsg}`;
          assistantMessage.failed = true;
        });
      } finally {
        store.setLoading(false);
      }
    };

    if (endpoint === "/v1/audio/speech") {
      await handleAudioSpeech();
      return;
    }

    // Add placeholder assistant message
    const assistantMessage = reactive<ChatMessage>({
      role: "assistant",
      content: "",
      reasoning_content: "",
      tool_calls: [],
    });
    store.pushMessage(assistantMessage);

    const contentBuffer = createDeltaBuffer((delta) => {
      mutateChat(() => {
        assistantMessage.content += delta;
      });
    });
    const reasoningBuffer = createDeltaBuffer((delta) => {
      mutateChat(() => {
        assistantMessage.reasoning_content = (assistantMessage.reasoning_content || "") + delta;
      });
    });

    try {
      await chatApi.streamChatCompletion(
        endpoint,
        requestPayload,
        apiKey,
        (chunk) => {
          contentBuffer.push(chunk);
          if (onChunk) onChunk(chunk);
        },
        (streamError) => {
          const errorMsg = `Stream error: ${streamError}`;
          contentBuffer.push(`**Error:** ${errorMsg}`);
          failGeneration(assistantMessage, errorMsg);
        },
        (reasoningChunk) => {
          reasoningBuffer.push(reasoningChunk);
          if (onChunk) onChunk("");
        },
        (index, id, name, args) => {
          mutateChat(() => {
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
          });
          if (onChunk) onChunk("");
        },
        (_index, id, query, status, payload) => {
          mutateChat(() => {
            const calls = (assistantMessage.web_search_calls ??= []);
            // Use id as the key to avoid duplicates from mismatched output_index values
            const existingIdx = calls.findIndex((c) => c && c.id === id);
            const previous = existingIdx >= 0 ? calls[existingIdx] : undefined;
            // The completion event settles the trace and carries the native
            // result payload that later turns replay to the model; keep the
            // query the in-progress event may not have had yet.
            const call = {
              id,
              query: query || previous?.query || "",
              status,
              sources: payload?.sources ?? previous?.sources,
              result: payload?.result ?? previous?.result,
            };
            if (existingIdx >= 0) {
              calls[existingIdx] = call;
            } else {
              calls.push(call);
            }
          });
          if (onChunk) onChunk("");
        },
        signal
      );
    } catch (e) {
      // Check if this was an abort error
      if (e instanceof Error && e.name === "AbortError") {
        contentBuffer.push("\n\n*[Generation stopped]*");
        runStopped = true;
      } else if (e instanceof Error && e.name === "TimeoutError") {
        // The request never started responding within the first-byte deadline
        // (see services/api/chat.ts). Surface the real reason instead of the
        // generic failure copy.
        console.error(e);
        const errorMsg = e.message || t("dialogs.errorGenerating");
        contentBuffer.push(`**Error:** ${errorMsg}`);
        failGeneration(assistantMessage, errorMsg);
      } else {
        console.error(e);
        const errorMsg = t("dialogs.errorGenerating");
        contentBuffer.push(`\n\n**Error:** ${errorMsg}`);
        failGeneration(assistantMessage, errorMsg);
      }
    } finally {
      contentBuffer.flush();
      reasoningBuffer.flush();
      abortController.value = null;
      store.setLoading(false);

      // Dead-end guard: the stream settled without an error but the message
      // has no text and no client-executable tool calls. This happens when a
      // hosted-tool request ends after the tool call without a final answer
      // (e.g. web search enabled while the proxy has no server-side search
      // configured). Surface it as a failure so the user gets an error and a
      // retry instead of a silently empty bubble.
      const hasExecutableToolCalls = (assistantMessage.tool_calls ?? []).some((tc) => tc && tc.id);
      if (
        !runStopped &&
        !runFailed &&
        !hasExecutableToolCalls &&
        typeof assistantMessage.content === "string" &&
        assistantMessage.content.trim() === ""
      ) {
        failGeneration(assistantMessage, t("chat.emptyResponse"));
      }
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
