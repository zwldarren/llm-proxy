import type { ChatMessage, ContentPart } from "@/types/schemas";
import type { ToolDefinition } from "./types";
import type { ProtocolAdapter, StreamChunkCallbacks } from "./types";
import { safeJsonParse, numberOrDefault, stringOrEmpty } from "./utils";

/**
 * Transform content parts from canonical (OpenAI-style) format to Responses API format.
 * - image_url → input_image
 * - file → input_file
 * - text → input_text
 */
function transformContent(content: string | ContentPart[]): string | unknown[] {
  if (typeof content === "string") return content;

  return content.map((part) => {
    if (part.type === "text") {
      return { type: "input_text", text: part.text };
    }
    if (part.type === "image_url") {
      return {
        type: "input_image",
        image_url: part.image_url.url,
        ...(part.image_url.detail ? { detail: part.image_url.detail } : {}),
      };
    }
    if (part.type === "file") {
      return {
        type: "input_file",
        file_data: part.file.file_data,
        filename: part.file.filename,
      };
    }
    // Pass through unknown part types
    return part;
  });
}

function formatMessages(
  storeMessages: ChatMessage[],
  systemPrompt?: string
): Record<string, unknown>[] {
  const items: Record<string, unknown>[] = [];
  if (systemPrompt?.trim()) {
    items.push({
      type: "message",
      role: "system",
      content: systemPrompt.trim(),
    });
  }
  for (const msg of storeMessages) {
    if (msg.role === "tool") {
      items.push({
        type: "function_call_output",
        call_id: msg.tool_call_id,
        output: msg.content,
      });
    } else if (msg.role === "assistant") {
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        if (msg.content) {
          items.push({
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: msg.content }],
          });
        }
        for (const tc of msg.tool_calls) {
          if (!tc || !tc.id) continue;
          items.push({
            type: "function_call",
            call_id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
          });
        }
      } else {
        items.push({
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: msg.content || "" }],
        });
      }
    } else if (msg.role === "system") {
      items.push({
        type: "message",
        role: "system",
        content: msg.content || "",
      });
    } else if (msg.role === "user") {
      items.push({
        type: "message",
        role: "user",
        content: transformContent(msg.content),
      });
    }
  }
  return items;
}

function formatTools(rawTools: ToolDefinition[]) {
  if (!rawTools || rawTools.length === 0) return undefined;
  return rawTools.map((t) => ({
    type: "function",
    name: t.name,
    description: t.description || undefined,
    parameters: safeJsonParse(t.parameters),
  }));
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

  if (type === "response.output_item.added") {
    const index = numberOrDefault(parsedData.output_index, 0);
    const item = parsedData.item as Record<string, unknown> | undefined;
    if (item?.type === "function_call" && onToolCall) {
      onToolCall(
        index,
        stringOrEmpty(item.call_id) || stringOrEmpty(item.id),
        stringOrEmpty(item.name),
        ""
      );
    } else if (item?.type === "web_search_call" && onWebSearchCall) {
      const action = item.action as Record<string, unknown> | undefined;
      onWebSearchCall(index, stringOrEmpty(item.id), stringOrEmpty(action?.query), "in_progress");
    }
  } else if (type === "response.output_text.delta") {
    const content = stringOrEmpty(parsedData.delta);
    if (content) onChunk(content);
  } else if (type === "response.reasoning_text.delta") {
    const reasoning = stringOrEmpty(parsedData.delta);
    if (reasoning && onReasoningChunk) onReasoningChunk(reasoning);
  } else if (type === "response.function_call_arguments.delta") {
    const index = numberOrDefault(parsedData.output_index, 0);
    const delta = stringOrEmpty(parsedData.delta);
    if (delta && onToolCall) {
      onToolCall(index, "", "", delta);
    }
  } else if (type === "response.output_item.done") {
    const item = parsedData.item as Record<string, unknown> | undefined;
    if (item?.type === "web_search_call" && onWebSearchCall) {
      const index = numberOrDefault(parsedData.output_index, 0);
      const action = item.action as Record<string, unknown> | undefined;
      onWebSearchCall(
        index,
        stringOrEmpty(item.id),
        stringOrEmpty(action?.query),
        (stringOrEmpty(item.status) as "in_progress" | "completed" | "failed") || "completed"
      );
    }
  }
}

export const openResponsesAdapter: ProtocolAdapter = {
  id: "responses",
  formatMessages,
  formatTools,
  parseStreamChunk,
};
