import type { ChatMessage, ContentPart } from "@/types/schemas";
import type { ToolDefinition } from "./types";
import type { ProtocolAdapter, StreamChunkCallbacks } from "./types";
import { safeJsonParse, numberOrDefault, stringOrEmpty } from "./utils";

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
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        const contentParts: Record<string, unknown>[] = [];
        if (msg.content) {
          contentParts.push({ type: "text", text: msg.content });
        }
        for (const tc of msg.tool_calls) {
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

function parseStreamChunk(
  parsedData: Record<string, unknown>,
  currentEvent: string,
  callbacks: StreamChunkCallbacks
): void {
  const { onChunk, onReasoningChunk, onToolCall, onError } = callbacks;
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

  if (type === "content_block_start") {
    const index = numberOrDefault(parsedData.index, 0);
    const block = parsedData.content_block as Record<string, unknown> | undefined;
    if (block?.type === "tool_use" && onToolCall) {
      onToolCall(index, stringOrEmpty(block.id), stringOrEmpty(block.name), "");
    }
  } else if (type === "content_block_delta") {
    const index = numberOrDefault(parsedData.index, 0);
    const delta = parsedData.delta as Record<string, unknown> | undefined;
    if (delta) {
      if (delta.type === "text_delta") {
        onChunk(stringOrEmpty(delta.text));
      } else if (delta.type === "thinking_delta" && onReasoningChunk) {
        onReasoningChunk(stringOrEmpty(delta.thinking));
      } else if (delta.type === "input_json_delta" && onToolCall) {
        onToolCall(index, "", "", stringOrEmpty(delta.partial_json));
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
