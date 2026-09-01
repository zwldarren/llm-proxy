/**
 * Response-level metadata extracted from a stream, shared by both SSE
 * dialects. Returned as one object so consumers can pass it straight
 * through instead of re-bundling flat fields.
 */
export interface StreamMeta {
  /** Completion / message id (first event that carries one) */
  id?: string;
  /** Model (first event that carries one) */
  model?: string;
  /** finish_reason (OpenAI-style streams) */
  finishReason?: string;
  /** stop_reason (Anthropic streams) */
  stopReason?: string;
  /** stop_sequence (Anthropic streams) */
  stopSequence?: string;
  /** Number of successfully parsed SSE data events */
  eventCount: number;
}

interface StreamChunk {
  id?: string;
  model?: string;
  choices?: Array<{
    delta?: {
      content?: string;
      reasoning_content?: string;
      // OpenAI Chat streaming tool calls: split across chunks, indexed.
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string;
  }>;
  error?: {
    type?: string;
    message?: string;
    code?: number | string;
    error_id?: string;
    hint?: string;
  };
}

interface ParsedStreamResponse {
  reconstructedContent: string;
  reasoningContent: string;
  toolCalls: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
  chunks: StreamChunk[];
  /** Stream-level metadata (id, model, finish reason, event count) */
  meta: StreamMeta;
}

interface ParsedSSEEvent {
  data: StreamChunk | null;
  error?: {
    type: string;
    message: string;
    code?: number | string;
    error_id?: string;
    hint?: string;
  };
}

/**
 * Parse a single SSE event line starting with "data: "
 * Returns null if the line is not a data event or is "[DONE]"
 */
export const parseSSEEvent = (line: string): ParsedSSEEvent | null => {
  const trimmed = line.trim();
  if (!trimmed.startsWith("data: ")) return null;

  const dataStr = trimmed.slice(6);
  if (dataStr === "[DONE]") return null;

  try {
    const data = JSON.parse(dataStr) as StreamChunk;
    // Check for error in the response
    if (data.error) {
      return {
        data: null,
        error: {
          type: data.error.type || "unknown_error",
          message: data.error.message || "Unknown error occurred",
          code: data.error.code,
          error_id: data.error.error_id,
          hint: data.error.hint,
        },
      };
    }
    return { data };
  } catch {
    if (import.meta.env.DEV) {
      console.warn("Failed to parse SSE event:", dataStr);
    }
    return null;
  }
};

export const isStreamResponse = (body: unknown): boolean => {
  if (typeof body !== "string") return false;
  // OpenAI streaming format
  if (
    body.includes("data: ") &&
    (body.includes("chat.completion.chunk") || body.includes("[DONE]"))
  ) {
    return true;
  }
  // Anthropic streaming format
  return isAnthropicStreamResponse(body);
};

export const isAnthropicStreamResponse = (body: string): boolean => {
  return body.includes("event:") && body.includes("content_block_delta");
};

/**
 * Parse an entire SSE response body into reconstructed content and chunks.
 * This is used for replay / log viewing where the full body has already been captured.
 */
export const parseStreamResponse = (body: string): ParsedStreamResponse => {
  const lines = body.split("\n");
  let reconstructedContent = "";
  let reasoningContent = "";
  const meta: StreamMeta = { eventCount: 0 };
  const chunks: StreamChunk[] = [];
  // Reconstruct streaming tool calls incrementally, keyed by delta index.
  const toolBuffers = new Map<
    number,
    { id: string; type: string; name: string; arguments: string }
  >();

  for (const line of lines) {
    const parsed = parseSSEEvent(line);
    if (!parsed) continue;
    meta.eventCount++;
    if (parsed.error) {
      chunks.push({ error: parsed.error });
      continue;
    }
    if (parsed.data) {
      chunks.push(parsed.data);
      meta.id = meta.id ?? parsed.data.id;
      meta.model = meta.model ?? parsed.data.model;
      const choice = parsed.data.choices?.[0];
      const delta = choice?.delta;
      if (choice?.finish_reason) {
        meta.finishReason = choice.finish_reason;
      }
      if (!delta) continue;
      if (delta.content) {
        reconstructedContent += delta.content;
      }
      if (delta.reasoning_content) {
        reasoningContent += delta.reasoning_content;
      }
      if (Array.isArray(delta.tool_calls)) {
        for (const tc of delta.tool_calls) {
          if (!tc || typeof tc.index !== "number") continue;
          const buf = toolBuffers.get(tc.index) ?? {
            id: "",
            type: "function",
            name: "",
            arguments: "",
          };
          if (tc.id) buf.id = tc.id;
          if (tc.type) buf.type = tc.type;
          if (tc.function?.name) buf.name = tc.function.name;
          if (tc.function?.arguments) buf.arguments += tc.function.arguments;
          toolBuffers.set(tc.index, buf);
        }
      }
    }
  }

  // Emit tool calls ordered by their stream index.
  const toolCalls: ParsedStreamResponse["toolCalls"] = [...toolBuffers.keys()]
    .sort((a, b) => a - b)
    .map((idx) => {
      const buf = toolBuffers.get(idx)!;
      return {
        id: buf.id,
        type: buf.type || "function",
        function: { name: buf.name, arguments: buf.arguments },
      };
    });

  return {
    reconstructedContent,
    reasoningContent,
    toolCalls,
    chunks,
    meta,
  };
};

/** One ordered content block reconstructed from an Anthropic stream. */
export type AnthropicStreamBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string }
  | { type: "redacted_thinking" }
  | { type: "tool_use"; id: string; name: string; arguments: string };

interface ParsedAnthropicStreamResponse {
  reconstructedContent: string;
  reasoningContent: string;
  toolCalls: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
  /** Content blocks in their original (content-block index) order */
  blocks: AnthropicStreamBlock[];
  /** Stream-level metadata (message id, model, stop reason, event count) */
  meta: StreamMeta;
}

/**
 * Parse an Anthropic SSE response body for replay / log viewing.
 */
export const parseAnthropicStreamResponse = (body: string): ParsedAnthropicStreamResponse => {
  const lines = body.split("\n");
  let reconstructedContent = "";
  let reasoningContent = "";
  const toolCalls: ParsedAnthropicStreamResponse["toolCalls"] = [];
  const meta: StreamMeta = { eventCount: 0 };

  // Ordered content blocks keyed by their content-block index.
  const blockSlots = new Map<number, AnthropicStreamBlock>();

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]?.trim();
    if (!line) continue;

    let dataStr = "";

    if (line.startsWith("event:")) {
      const nextLine = lines[i + 1]?.trim();
      if (nextLine?.startsWith("data:")) {
        dataStr = nextLine.slice(5).trim();
        i++;
      }
    } else if (line.startsWith("data:")) {
      dataStr = line.slice(5).trim();
    }

    if (!dataStr) continue;

    let data: Record<string, unknown>;
    try {
      data = JSON.parse(dataStr) as Record<string, unknown>;
    } catch {
      continue;
    }
    meta.eventCount++;

    const type = data.type as string | undefined;

    if (type === "message_start") {
      const message = data.message as Record<string, unknown> | undefined;
      meta.id = (message?.id as string) || meta.id;
      meta.model = (message?.model as string) || meta.model;
    } else if (type === "message_delta") {
      const delta = data.delta as Record<string, unknown> | undefined;
      meta.stopReason = (delta?.stop_reason as string) || meta.stopReason;
      meta.stopSequence = (delta?.stop_sequence as string) || meta.stopSequence;
    } else if (type === "content_block_start") {
      const index = data.index as number;
      const contentBlock = data.content_block as Record<string, unknown> | undefined;
      if (contentBlock?.type === "text") {
        blockSlots.set(index, { type: "text", text: (contentBlock.text as string) || "" });
      } else if (contentBlock?.type === "thinking") {
        blockSlots.set(index, { type: "thinking", thinking: "" });
      } else if (contentBlock?.type === "redacted_thinking") {
        blockSlots.set(index, { type: "redacted_thinking" });
      } else if (contentBlock?.type === "tool_use" || contentBlock?.type === "server_tool_use") {
        blockSlots.set(index, {
          type: "tool_use",
          id: (contentBlock.id as string) || "",
          name: (contentBlock.name as string) || "",
          arguments: "",
        });
      }
    } else if (type === "content_block_delta") {
      const index = data.index as number;
      const delta = data.delta as Record<string, unknown> | undefined;
      const slot = blockSlots.get(index);
      if (delta?.type === "text_delta" && slot?.type === "text") {
        const text = (delta.text as string) || "";
        slot.text += text;
        reconstructedContent += text;
      } else if (delta?.type === "thinking_delta" && slot?.type === "thinking") {
        const thinking = (delta.thinking as string) || "";
        slot.thinking += thinking;
        reasoningContent += thinking;
      } else if (delta?.type === "input_json_delta" && slot?.type === "tool_use") {
        slot.arguments += (delta.partial_json as string) || "";
      }
    }
  }

  const blocks = [...blockSlots.entries()].sort((a, b) => a[0] - b[0]).map(([, block]) => block);

  for (const block of blocks) {
    if (block.type === "tool_use") {
      toolCalls.push({
        id: block.id,
        type: "function",
        function: { name: block.name, arguments: block.arguments },
      });
    }
  }

  return {
    reconstructedContent,
    reasoningContent,
    toolCalls,
    blocks,
    meta,
  };
};
