import type { ChatMessage, ContentPart, WebSearchCall } from "@/types/schemas";
import type { ToolDefinition } from "./types";
import type { ProtocolAdapter, StreamChunkCallbacks } from "./types";
import {
  replayableWebSearches,
  safeJsonParse,
  numberOrDefault,
  stringOrEmpty,
  toWebSearchSources,
} from "./utils";

interface AnthropicMessage {
  role: string;
  content: string | unknown[];
}

/**
 * Extract media type and base64 data from a data URL.
 * Returns null if the URL is not a valid base64 data URL.
 */
function parseDataUrl(dataUrl: string): { mediaType: string; data: string } | null {
  const matches = dataUrl.match(/^data:(.+?);base64,(.+)$/);
  if (matches) {
    return { mediaType: matches[1], data: matches[2] };
  }
  return null;
}

/**
 * Transform content parts from canonical (OpenAI-style) format to Anthropic format.
 * - image_url → image (base64 source)
 * - file → document (base64 source)
 * - text → text (passthrough)
 */
function transformContent(content: string | ContentPart[]): string | unknown[] {
  if (typeof content === "string") return content;

  return content.map((part) => {
    if (part.type === "text") {
      return { type: "text", text: part.text };
    }
    if (part.type === "image_url") {
      const parsed = parseDataUrl(part.image_url.url);
      if (parsed) {
        return {
          type: "image",
          source: {
            type: "base64",
            media_type: parsed.mediaType,
            data: parsed.data,
          },
        };
      }
      // For URL-based images (not data URL), fall back to text
      return { type: "text", text: `[Image: ${part.image_url.url}]` };
    }
    if (part.type === "file") {
      const parsed = parseDataUrl(part.file.file_data);
      if (parsed) {
        return {
          type: "document",
          source: {
            type: "base64",
            media_type: parsed.mediaType,
            data: parsed.data,
          },
          ...(part.file.filename ? { title: part.file.filename } : {}),
        };
      }
      return { type: "text", text: `[File: ${part.file.filename}]` };
    }
    // Pass through unknown part types (tool_use, etc.)
    return part;
  });
}

function formatMessages(storeMessages: ChatMessage[], _systemPrompt?: string): AnthropicMessage[] {
  const messages: AnthropicMessage[] = [];
  for (const msg of storeMessages) {
    if (msg.role === "tool") {
      messages.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: msg.tool_call_id || "",
            content: msg.content || "",
          },
        ],
      });
    } else if (msg.role === "assistant") {
      // Proxy-intercepted web searches are replayed as a completed server-tool
      // exchange so the model keeps the results across later turns. Only a
      // search that recorded a native result can be paired with a
      // `server_tool_use` block; one stopped mid-flight is dropped.
      const webSearches = replayableWebSearches(
        msg.web_search_calls,
        (call) => call.result !== undefined
      );
      if ((msg.tool_calls && msg.tool_calls.length > 0) || webSearches.length > 0) {
        const contentParts: Record<string, unknown>[] = [];
        if (msg.content) {
          contentParts.push({ type: "text", text: msg.content });
        }
        // `server_tool_use` blocks come first so the matching
        // `web_search_tool_result` blocks below keep the API's pairing order.
        for (const call of webSearches) {
          contentParts.push({
            type: "server_tool_use",
            id: call.id,
            name: WEB_SEARCH_TOOL_NAME,
            input: { query: call.query },
          });
        }
        for (const tc of msg.tool_calls || []) {
          if (!tc || !tc.id || !tc.function) continue;
          const inputObj = safeJsonParse(tc.function.arguments);
          contentParts.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: inputObj,
          });
        }
        messages.push({
          role: "assistant",
          content: contentParts,
        });
        if (webSearches.length > 0) {
          messages.push({
            role: "user",
            content: webSearches.map(anthropicWebSearchResult),
          });
        }
      } else {
        messages.push({
          role: "assistant",
          content: msg.content || "",
        });
      }
    } else if (msg.role === "user") {
      messages.push({
        role: "user",
        content: transformContent(msg.content),
      });
    }
  }

  // Merge consecutive messages with the same role
  const groupedMessages: AnthropicMessage[] = [];
  for (const msg of messages) {
    const last = groupedMessages[groupedMessages.length - 1];
    if (last && last.role === msg.role) {
      const currentContent = Array.isArray(last.content)
        ? last.content
        : [{ type: "text", text: last.content }];
      const newContent = Array.isArray(msg.content)
        ? msg.content
        : [{ type: "text", text: msg.content }];
      last.content = [...currentContent, ...newContent];
    } else {
      groupedMessages.push(msg);
    }
  }
  return groupedMessages;
}

function formatTools(rawTools: ToolDefinition[]) {
  if (!rawTools || rawTools.length === 0) return undefined;
  return rawTools.map((t) => ({
    name: t.name,
    description: t.description || undefined,
    input_schema: safeJsonParse(t.parameters),
  }));
}

/** Per-stream content-block tracking, keyed by the per-request callbacks object. */
interface AnthropicBlockState {
  /** Block type per content index (tool_use, server_tool_use, text, ...). */
  blockTypes: Map<number, string>;
  /** server_tool_use id per content index, for the web-search trace. */
  serverToolIds: Map<number, string>;
  /** Accumulated input JSON per server_tool_use index (holds the query). */
  serverToolArgs: Map<number, string>;
}

const streamStates = new WeakMap<StreamChunkCallbacks, AnthropicBlockState>();

const getStreamState = (callbacks: StreamChunkCallbacks): AnthropicBlockState => {
  let state = streamStates.get(callbacks);
  if (!state) {
    state = { blockTypes: new Map(), serverToolIds: new Map(), serverToolArgs: new Map() };
    streamStates.set(callbacks, state);
  }
  return state;
};

/** Models emit web-search tool names in several casings/separators. */
const isWebSearchToolName = (name: string): boolean =>
  name.toLowerCase().replace(/[_-]/g, "") === "websearch";

const WEB_SEARCH_TOOL_NAME = "web_search";

/** Rebuild the `web_search_tool_result` block paired with a replayed call. */
function anthropicWebSearchResult(call: WebSearchCall): Record<string, unknown> {
  return {
    type: "web_search_tool_result",
    tool_use_id: call.id,
    content: call.result,
  };
}

function parseStreamChunk(
  parsedData: Record<string, unknown>,
  currentEvent: string,
  callbacks: StreamChunkCallbacks
): void {
  const { onChunk, onReasoningChunk, onToolCall, onError, onWebSearchCall } = callbacks;
  const type = (parsedData.type as string) || currentEvent;

  // Handle API-level errors delivered in stream chunks
  if (parsedData.error) {
    const errorMsg =
      typeof parsedData.error === "object"
        ? ((parsedData.error as Record<string, unknown>).message as string) ||
          JSON.stringify(parsedData.error)
        : String(parsedData.error);
    onError?.(`Error: ${errorMsg}`);
    return;
  }

  const state = getStreamState(callbacks);

  if (type === "content_block_start") {
    const index = numberOrDefault(parsedData.index, 0);
    const block = parsedData.content_block as Record<string, unknown> | undefined;
    if (!block) return;
    const blockType = stringOrEmpty(block.type);
    state.blockTypes.set(index, blockType);
    if (blockType === "tool_use" && onToolCall) {
      onToolCall(index, stringOrEmpty(block.id), stringOrEmpty(block.name), "");
    } else if (
      blockType === "server_tool_use" &&
      isWebSearchToolName(stringOrEmpty(block.name)) &&
      onWebSearchCall
    ) {
      // Proxy-intercepted web search: surface it as a search trace instead of
      // a client-executable tool call (which would wait for output forever).
      state.serverToolIds.set(index, stringOrEmpty(block.id));
      onWebSearchCall(index, stringOrEmpty(block.id), "", "in_progress");
    } else if (blockType === "web_search_tool_result" && onWebSearchCall) {
      // The result block settles the trace: the wire carries errors as a
      // string payload, successes as a list of web_search_result objects.
      const toolUseId = stringOrEmpty(block.tool_use_id);
      const callIndex =
        [...state.serverToolIds.entries()].find(([, id]) => id === toolUseId)?.[0] ?? index;
      const rawArgs = state.serverToolArgs.get(callIndex) ?? "";
      let query = "";
      try {
        query = stringOrEmpty((JSON.parse(rawArgs) as Record<string, unknown>).query);
      } catch {
        // Partial/invalid JSON — the trace simply shows no query.
      }
      const failed =
        typeof block.content === "string" && block.content.includes("web_search_tool_result_error");
      onWebSearchCall(callIndex, toolUseId, query, failed ? "failed" : "completed", {
        result: block.content,
        sources: toWebSearchSources(block.content),
      });
      state.blockTypes.delete(callIndex);
      state.serverToolIds.delete(callIndex);
      state.serverToolArgs.delete(callIndex);
    }
  } else if (type === "content_block_delta") {
    const index = numberOrDefault(parsedData.index, 0);
    const delta = parsedData.delta as Record<string, unknown> | undefined;
    if (delta) {
      if (delta.type === "text_delta") {
        onChunk(stringOrEmpty(delta.text));
      } else if (delta.type === "thinking_delta" && onReasoningChunk) {
        onReasoningChunk(stringOrEmpty(delta.thinking));
      } else if (delta.type === "input_json_delta") {
        const blockType = state.blockTypes.get(index);
        if (blockType === "server_tool_use") {
          const partial = stringOrEmpty(delta.partial_json);
          state.serverToolArgs.set(index, (state.serverToolArgs.get(index) ?? "") + partial);
        } else if (onToolCall) {
          onToolCall(index, "", "", stringOrEmpty(delta.partial_json));
        }
      }
    }
  }
}

export const anthropicAdapter: ProtocolAdapter = {
  id: "messages",
  formatMessages,
  formatTools,
  parseStreamChunk,
};
