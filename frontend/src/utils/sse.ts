interface StreamChunk {
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
  const chunks: StreamChunk[] = [];
  // Reconstruct streaming tool calls incrementally, keyed by delta index.
  const toolBuffers = new Map<
    number,
    { id: string; type: string; name: string; arguments: string }
  >();

  for (const line of lines) {
    const parsed = parseSSEEvent(line);
    if (!parsed) continue;
    if (parsed.error) {
      chunks.push({ error: parsed.error });
      continue;
    }
    if (parsed.data) {
      chunks.push(parsed.data);
      const delta = parsed.data.choices?.[0]?.delta;
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

  return { reconstructedContent, reasoningContent, toolCalls, chunks };
};

interface ParsedAnthropicStreamResponse {
  reconstructedContent: string;
  reasoningContent: string;
  toolCalls: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
}

/**
 * Parse an Anthropic SSE response body for replay / log viewing.
 */
export const parseAnthropicStreamResponse = (body: string): ParsedAnthropicStreamResponse => {
  const lines = body.split("\n");
  let reconstructedContent = "";
  let reasoningContent = "";
  const toolCalls: ParsedAnthropicStreamResponse["toolCalls"] = [];
  const toolBuffers: Map<string, { name: string; arguments: string }> = new Map();
  let currentToolId = "";
  let currentToolArgs = "";
  let currentContentIndex = -1;
  let currentThinkingIndex = -1;

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

    const type = data.type as string | undefined;

    if (type === "content_block_start") {
      const index = data.index as number;
      const contentBlock = data.content_block as Record<string, unknown> | undefined;
      if (contentBlock?.type === "text") {
        currentContentIndex = index;
      } else if (contentBlock?.type === "thinking") {
        currentThinkingIndex = index;
      } else if (contentBlock?.type === "tool_use") {
        currentToolId = (contentBlock.id as string) || "";
        currentToolArgs = "";
        toolBuffers.set(currentToolId, {
          name: (contentBlock.name as string) || "",
          arguments: "",
        });
      }
    } else if (type === "content_block_delta") {
      const index = data.index as number;
      const delta = data.delta as Record<string, unknown> | undefined;
      if (delta?.type === "text_delta" && index === currentContentIndex) {
        reconstructedContent += (delta.text as string) || "";
      } else if (delta?.type === "thinking_delta" && index === currentThinkingIndex) {
        reasoningContent += (delta.thinking as string) || "";
      } else if (delta?.type === "input_json_delta") {
        currentToolArgs += (delta.partial_json as string) || "";
        if (currentToolId && toolBuffers.has(currentToolId)) {
          const tool = toolBuffers.get(currentToolId)!;
          tool.arguments = currentToolArgs;
        }
      }
    } else if (type === "content_block_stop") {
      const index = data.index as number;
      if (index === currentContentIndex) {
        currentContentIndex = -1;
      } else if (index === currentThinkingIndex) {
        currentThinkingIndex = -1;
      }
    }
  }

  for (const [id, tool] of toolBuffers) {
    toolCalls.push({
      id,
      type: "function",
      function: { name: tool.name, arguments: tool.arguments },
    });
  }

  return { reconstructedContent, reasoningContent, toolCalls };
};
