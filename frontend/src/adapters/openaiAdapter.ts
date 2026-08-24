import type { ChatMessage } from "@/types/schemas";
import type { ToolDefinition } from "./types";
import type { ProtocolAdapter, StreamChunkCallbacks } from "./types";

import { safeJsonParse, numberOrDefault } from "./utils";

function formatMessages(
  storeMessages: ChatMessage[],
  systemPrompt?: string
): Record<string, unknown>[] {
  const messages: Record<string, unknown>[] = [];
  if (systemPrompt?.trim()) {
    messages.push({ role: "system", content: systemPrompt.trim() });
  }
  for (const msg of storeMessages) {
    if (msg.role === "tool") {
      messages.push({
        role: "tool",
        tool_call_id: msg.tool_call_id,
        name: msg.name,
        content: msg.content,
      });
    } else {
      const filteredCalls = msg.tool_calls ? msg.tool_calls.filter((tc) => tc && tc.id) : undefined;
      messages.push({
        role: msg.role,
        content: msg.content || "",
        tool_calls: filteredCalls && filteredCalls.length > 0 ? filteredCalls : undefined,
      });
    }
  }
  return messages;
}

function formatTools(rawTools: ToolDefinition[]) {
  if (!rawTools || rawTools.length === 0) return undefined;
  return rawTools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description || undefined,
      parameters: safeJsonParse(t.parameters),
    },
  }));
}

function parseStreamChunk(
  parsedData: Record<string, unknown>,
  _currentEvent: string,
  callbacks: StreamChunkCallbacks
): void {
  const { onChunk, onReasoningChunk, onToolCall } = callbacks;

  if (parsedData.error) {
    const fullError = `Error: ${
      typeof parsedData.error === "object"
        ? (parsedData.error as Record<string, unknown>).message || JSON.stringify(parsedData.error)
        : parsedData.error
    }`;
    callbacks.onError?.(fullError);
    return;
  }

  const choices = parsedData.choices as Array<Record<string, unknown>> | undefined;
  const choice = choices?.[0];
  if (!choice) return;

  const delta = choice.delta as Record<string, unknown> | undefined;
  if (!delta) return;

  const content = (delta.content as string) || "";
  if (content) onChunk(content);

  const reasoning = (delta.reasoning_content as string) || "";
  if (reasoning && onReasoningChunk) onReasoningChunk(reasoning);

  const toolCalls = delta.tool_calls as Array<Record<string, unknown>> | undefined;
  if (toolCalls && onToolCall) {
    for (const tc of toolCalls) {
      onToolCall(
        numberOrDefault(tc.index, 0),
        (tc.id as string) || "",
        ((tc.function as Record<string, unknown>)?.name as string) || "",
        ((tc.function as Record<string, unknown>)?.arguments as string) || ""
      );
    }
  }
}

export const openaiAdapter: ProtocolAdapter = {
  id: "chat/completions",
  formatMessages,
  formatTools,
  parseStreamChunk,
};
