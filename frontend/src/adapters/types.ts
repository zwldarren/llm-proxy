export interface CustomVariable {
  key: string;
  value: string;
  type: "string" | "number" | "boolean";
  enabled: boolean;
}

/**
 * A tool definition as used in the UI and adapters.
 */
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: string; // JSON schema string
  enabled: boolean;
}

import type { ChatMessage } from "@/types/schemas";

/**
 * Protocol adapter interface.
 * Each adapter handles request formatting and stream response parsing
 * for a specific API protocol (OpenAI, Anthropic, OpenResponses, etc.).
 */
export interface ProtocolAdapter {
  /** Unique identifier for this protocol (matches the endpoint path segment) */
  id: string;

  /** Format messages array for this protocol's request body */
  formatMessages(messages: ChatMessage[], systemPrompt?: string): unknown[];

  /** Format tool definitions for this protocol */
  formatTools(tools: ToolDefinition[]): unknown[] | undefined;

  /** Parse a stream chunk and dispatch to callbacks */
  parseStreamChunk(
    parsedData: Record<string, unknown>,
    currentEvent: string,
    callbacks: StreamChunkCallbacks
  ): void;
}

export interface StreamChunkCallbacks {
  onChunk: (chunk: string) => void;
  onReasoningChunk?: (chunk: string) => void;
  onToolCall?: (index: number, id: string, name: string, args: string) => void;
  onError?: (error: string) => void;
  /** Called when a web_search_call output item is received */
  onWebSearchCall?: (
    index: number,
    id: string,
    query: string,
    status: "in_progress" | "completed" | "failed"
  ) => void;
}
