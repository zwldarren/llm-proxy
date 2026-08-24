/**
 * Unified log response parser.
 *
 * Normalizes every inbound-protocol `response_body` shape stored by the proxy
 * into a single intermediate structure so the Logs UI can render content,
 * reasoning, tool calls, tool results, images, embeddings, and audio
 * consistently — without having to special-case each protocol in the view
 * layer, and without feeding megabyte-sized base64 image strings into a JSON
 * tree viewer (which freezes the page).
 *
 * Supported inbound protocols (the proxy only stores client-facing shapes):
 *  - OpenAI Chat        (`object:"chat.completion"`, `choices[].message`)
 *  - OpenAI Responses   (`object:"response"`, `output[]` items)
 *  - Anthropic          (`type:"message"`, `content[]` blocks)
 *  - Streaming SSE      (raw string; OpenAI / Anthropic / Responses streams)
 *  - Images             (`created` + `data[]` with `b64_json`|`url`)
 *  - Embeddings         (`object:"list"`, `data[].embedding`)
 *  - Audio              (`{raw:true,size}` or `{text}`)
 */

import {
  isAnthropicStreamResponse,
  isStreamResponse,
  parseAnthropicStreamResponse,
  parseStreamResponse,
} from "@/utils/sse";
import { parseToolArgs } from "@/utils/logFormat";

// --- Types -------------------------------------------------------------

type ResponseProtocol =
  | "openai-chat"
  | "openai-responses"
  | "anthropic"
  | "stream-openai"
  | "stream-anthropic"
  | "stream-responses"
  | "image"
  | "embedding"
  | "audio-raw"
  | "audio-text"
  | "sampled-out"
  | "unknown";

export interface ToolCallInfo {
  /** Tool call id (OpenAI tool_calls[].id / Anthropic tool_use.id / Responses item id) */
  id?: string;
  /** Function/tool display name */
  name: string;
  /** Raw arguments as a JSON string (OpenAI/Responses) — may be empty for Anthropic */
  arguments: string;
  /** Parsed arguments object (best-effort) */
  parsedArguments?: Record<string, unknown>;
  /** Semantic kind for visual treatment */
  kind: "function" | "custom" | "server_tool_use" | "web_search" | "tool_search";
  /** Item status (Responses items carry status) */
  status?: string;
}

export interface ToolResultInfo {
  /** Id of the tool call this result belongs to */
  callId?: string;
  /** Anthropic tool_use_id */
  toolUseId?: string;
  /** Result output as a string */
  output: string;
  isError?: boolean;
  /** Source block/item type */
  kind?: string;
}

export interface ImageInfo {
  /** Direct image URL (when provider returned url instead of b64) */
  url?: string;
  /** Raw base64 image string — kept by reference, NOT copied into JSON viewers */
  b64Json?: string;
  /** Revised prompt from the model */
  revisedPrompt?: string;
  /** Output format hint (png/jpeg/...) for constructing data URLs */
  outputFormat?: string;
}

interface EmbeddingInfo {
  index: number;
  /** Truncated preview of the vector for display; full vector available in raw */
  vectorPreview: number[];
  fullLength: number;
}

export interface ParsedResponse {
  protocol: ResponseProtocol;
  /** Reconstructed assistant text content */
  content: string;
  /** Thinking / reasoning text */
  reasoning: string;
  /** Tool/function calls the assistant made */
  toolCalls: ToolCallInfo[];
  /** Tool results returned to the model (Anthropic tool_result / Responses function_call_output) */
  toolResults: ToolResultInfo[];
  /** Generated images (image_generation / image_edit) */
  images: ImageInfo[];
  /** Embedding vectors (embedding requests) */
  embeddings?: EmbeddingInfo[];
  /** Transcription/translation text (audio-text requests) */
  audioText?: string;
  /** Raw-byte audio marker (speech / text-format audio) */
  audioRaw?: { size: number };
  /** Backend sampled out the full body (only a sentinel was stored) */
  isSampledOut?: boolean;
  /** Whether any parseable data was found */
  hasData: boolean;
}

const empty = (overrides: Partial<ParsedResponse> = {}): ParsedResponse => ({
  protocol: "unknown",
  content: "",
  reasoning: "",
  toolCalls: [],
  toolResults: [],
  images: [],
  hasData: false,
  ...overrides,
});

// --- Helpers -----------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function safeJsonStringify(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  try {
    return typeof value === "string" ? value : JSON.stringify(value);
  } catch {
    return undefined;
  }
}

/** Inferred MIME type for a base64 image data URL. */
function imageMimeType(format?: string): string {
  switch (format) {
    case "jpeg":
    case "jpg":
      return "image/jpeg";
    case "webp":
      return "image/webp";
    case "gif":
      return "image/gif";
    default:
      return "image/png";
  }
}

/** Build a data: URL for a base64 image. Returns undefined when no b64 payload. */
export function buildImageDataUrl(img: ImageInfo): string | undefined {
  if (img.url) return img.url;
  if (!img.b64Json) return undefined;
  return `data:${imageMimeType(img.outputFormat)};base64,${img.b64Json}`;
}

// --- Protocol detection ------------------------------------------------

/** Cheap detection for an OpenAI Responses SSE stream. */
export function isResponsesStreamResponse(body: string): boolean {
  if (typeof body !== "string") return false;
  // Responses stream emits typed events like `response.output_text.delta`.
  return (
    (body.includes("event:") && body.includes("response.")) ||
    body.includes('"type":"response.') ||
    body.includes('"type": "response.')
  );
}

// --- Per-protocol parsers ----------------------------------------------

function parseOpenAIChat(body: Record<string, unknown>): ParsedResponse {
  const choices = body.choices;
  if (!Array.isArray(choices) || choices.length === 0) return empty({ protocol: "openai-chat" });

  const choice = choices[0] as Record<string, unknown> | undefined;
  const message = (choice?.message as Record<string, unknown>) || {};
  const content = asString(message.content) ?? "";
  const reasoning = asString(message.reasoning_content) ?? "";

  const toolCalls: ToolCallInfo[] = [];
  const rawToolCalls = message.tool_calls;
  if (Array.isArray(rawToolCalls)) {
    for (const tc of rawToolCalls) {
      if (!isRecord(tc)) continue;
      const fn = (tc.function as Record<string, unknown>) || {};
      const args = asString(fn.arguments) ?? "";
      const type = asString(tc.type) ?? "function";
      toolCalls.push({
        id: asString(tc.id),
        name: asString(fn.name) ?? "",
        arguments: args,
        parsedArguments: parseToolArgs(args),
        kind: type === "custom" ? "custom" : "function",
      });
    }
  }

  const hasData = Boolean(content || reasoning || toolCalls.length);
  return {
    protocol: "openai-chat",
    content,
    reasoning,
    toolCalls,
    toolResults: [],
    images: [],
    hasData,
  };
}

function parseOpenAIResponses(body: Record<string, unknown>): ParsedResponse {
  const output = body.output;
  if (!Array.isArray(output)) return empty({ protocol: "openai-responses" });

  let content = "";
  let reasoning = "";
  const toolCalls: ToolCallInfo[] = [];
  const toolResults: ToolResultInfo[] = [];

  for (const item of output) {
    if (!isRecord(item)) continue;
    const type = asString(item.type);
    const status = asString(item.status);

    if (type === "message") {
      const parts = item.content;
      if (Array.isArray(parts)) {
        for (const part of parts) {
          if (!isRecord(part)) continue;
          if (part.type === "output_text" || part.type === "text") {
            content += asString(part.text) ?? "";
            content += "\n";
          } else if (part.type === "refusal") {
            content += asString(part.refusal) ?? "";
            content += "\n";
          }
        }
      }
    } else if (type === "reasoning") {
      const parts = item.content;
      if (Array.isArray(parts)) {
        for (const part of parts) {
          if (isRecord(part) && (part.type === "reasoning_text" || part.type === "text")) {
            reasoning += asString(part.text) ?? "";
            reasoning += "\n";
          }
        }
      }
      // Encrypted reasoning content (when redacted)
      const encrypted = asString(item.encrypted_content);
      if (!reasoning && encrypted) reasoning = encrypted;
    } else if (type === "function_call" || type === "custom_tool_call") {
      const args = asString(item.arguments) ?? "";
      const input = item.input;
      const argStr = args || safeJsonStringify(input) || "";
      toolCalls.push({
        id: asString(item.id) || asString(item.call_id),
        name: asString(item.name) ?? "",
        arguments: argStr,
        parsedArguments: parseToolArgs(argStr),
        kind: type === "custom_tool_call" ? "custom" : "function",
        status,
      });
    } else if (type === "web_search_call") {
      const action = isRecord(item.action) ? item.action : undefined;
      const query =
        asString(action?.query) ||
        (Array.isArray(action?.queries)
          ? (action!.queries as unknown[]).map((q) => asString(q) ?? "").join(", ")
          : "");
      toolCalls.push({
        id: asString(item.id),
        name: "web_search",
        arguments: query ? JSON.stringify({ query }) : "",
        parsedArguments: query ? { query } : undefined,
        kind: "web_search",
        status,
      });
    } else if (type === "tool_search_call") {
      const args = safeJsonStringify(item.arguments) ?? "";
      toolCalls.push({
        id: asString(item.id) || asString(item.call_id),
        name: "tool_search",
        arguments: args,
        parsedArguments: parseToolArgs(args),
        kind: "tool_search",
        status,
      });
    } else if (type === "function_call_output" || type === "tool_search_output") {
      toolResults.push({
        callId: asString(item.call_id),
        output: asString(item.output) ?? safeJsonStringify(item.output) ?? "",
        kind: type,
      });
    }
  }

  const hasData = Boolean(content || reasoning || toolCalls.length || toolResults.length);
  return {
    protocol: "openai-responses",
    content: content.trim(),
    reasoning: reasoning.trim(),
    toolCalls,
    toolResults,
    images: [],
    hasData,
  };
}

function parseAnthropic(body: Record<string, unknown>): ParsedResponse {
  const contentBlocks = body.content;
  if (!Array.isArray(contentBlocks)) return empty({ protocol: "anthropic" });

  let content = "";
  let reasoning = "";
  const toolCalls: ToolCallInfo[] = [];
  const toolResults: ToolResultInfo[] = [];
  const images: ImageInfo[] = [];

  for (const block of contentBlocks) {
    if (!isRecord(block)) continue;
    const type = asString(block.type);

    switch (type) {
      case "text":
        content += asString(block.text) ?? "";
        content += "\n";
        break;
      case "thinking":
        reasoning += asString(block.thinking) ?? "";
        reasoning += "\n";
        break;
      case "redacted_thinking":
        reasoning += "[redacted]\n";
        break;
      case "tool_use":
      case "server_tool_use": {
        const input = block.input;
        const argStr = safeJsonStringify(input) ?? "";
        toolCalls.push({
          id: asString(block.id),
          name: asString(block.name) ?? "",
          arguments: argStr,
          parsedArguments: isRecord(input) ? input : undefined,
          kind: type === "server_tool_use" ? "server_tool_use" : "function",
        });
        break;
      }
      case "tool_result": {
        const c = block.content;
        const out =
          typeof c === "string"
            ? c
            : Array.isArray(c)
              ? c
                  .map((p) => {
                    if (isRecord(p) && p.type === "text") return asString(p.text) ?? "";
                    return safeJsonStringify(p) ?? "";
                  })
                  .join("\n")
              : (safeJsonStringify(c) ?? "");
        toolResults.push({
          toolUseId: asString(block.tool_use_id),
          output: out,
          isError: block.is_error === true,
          kind: "tool_result",
        });
        break;
      }
      case "image": {
        const source = isRecord(block.source) ? block.source : undefined;
        const mediaType = asString(source?.media_type) ?? "image/png";
        const data = asString(source?.data);
        const url = asString(source?.url);
        if (data) {
          images.push({ b64Json: data, outputFormat: mediaType.replace("image/", "") });
        } else if (url) {
          images.push({ url, outputFormat: mediaType.replace("image/", "") });
        }
        break;
      }
      default:
        break;
    }
  }

  const hasData = Boolean(
    content || reasoning || toolCalls.length || toolResults.length || images.length
  );
  return {
    protocol: "anthropic",
    content: content.trim(),
    reasoning: reasoning.trim(),
    toolCalls,
    toolResults,
    images,
    hasData,
  };
}

function parseImage(body: Record<string, unknown>): ParsedResponse {
  const data = body.data;
  if (!Array.isArray(data)) return empty({ protocol: "image" });

  const images: ImageInfo[] = [];
  for (const entry of data) {
    if (!isRecord(entry)) continue;
    images.push({
      url: asString(entry.url),
      b64Json: asString(entry.b64_json),
      revisedPrompt: asString(entry.revised_prompt),
      outputFormat: asString(body.output_format),
    });
  }

  return {
    protocol: "image",
    content: "",
    reasoning: "",
    toolCalls: [],
    toolResults: [],
    images,
    hasData: images.length > 0,
  };
}

const EMBEDDING_PREVIEW_LEN = 16;

function parseEmbedding(body: Record<string, unknown>): ParsedResponse {
  const data = body.data;
  if (!Array.isArray(data)) return empty({ protocol: "embedding" });

  const embeddings: EmbeddingInfo[] = [];
  for (const entry of data) {
    if (!isRecord(entry)) continue;
    const vector = entry.embedding;
    if (!Array.isArray(vector)) continue;
    embeddings.push({
      index: typeof entry.index === "number" ? entry.index : embeddings.length,
      vectorPreview: vector.slice(0, EMBEDDING_PREVIEW_LEN).map((v) => Number(v)),
      fullLength: vector.length,
    });
  }

  return {
    protocol: "embedding",
    content: "",
    reasoning: "",
    toolCalls: [],
    toolResults: [],
    images: [],
    embeddings,
    hasData: embeddings.length > 0,
  };
}

function parseAudioText(body: Record<string, unknown>): ParsedResponse {
  const text = asString(body.text) ?? "";
  return {
    protocol: "audio-text",
    content: text,
    reasoning: "",
    toolCalls: [],
    toolResults: [],
    images: [],
    audioText: text,
    hasData: Boolean(text),
  };
}

function parseResponsesStream(body: string): ParsedResponse {
  // Reconstruct content/reasoning/tool calls from Responses SSE events.
  // A function call arrives as:
  //   response.output_item.added   { item: { type:"function_call", id, call_id, name, arguments:"" } }
  //   response.function_call_arguments.delta  { item_id, delta }
  //   response.function_call_arguments.done    { item_id, arguments }
  //   response.output_item.done    { item: { type:"function_call", ... arguments } }
  const lines = body.split("\n");
  let content = "";
  let reasoning = "";
  const toolCalls: ToolCallInfo[] = [];
  // Keyed by item id (data.item_id / item.id).
  const toolBuffers = new Map<string, { id: string; name: string; args: string }>();

  const keyOf = (data: Record<string, unknown>): string | undefined =>
    asString(data.item_id) ??
    (typeof data.output_index === "number" ? String(data.output_index) : undefined);

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]?.trim();
    if (!line) continue;

    let dataStr = "";
    if (line.startsWith("event:")) {
      const next = lines[i + 1]?.trim();
      if (next?.startsWith("data:")) {
        dataStr = next.slice(5).trim();
        i++;
      }
    } else if (line.startsWith("data:")) {
      dataStr = line.slice(5).trim();
    }
    if (!dataStr || dataStr === "[DONE]") continue;

    let data: Record<string, unknown>;
    try {
      data = JSON.parse(dataStr) as Record<string, unknown>;
    } catch {
      continue;
    }

    const type = asString(data.type) ?? "";

    if (type === "response.output_text.delta") {
      content += asString(data.delta) ?? "";
    } else if (type === "response.reasoning_text.delta") {
      reasoning += asString(data.delta) ?? "";
    } else if (type === "response.output_item.added" || type === "response.output_item.done") {
      // A function_call item is announced (added) and finalized (done). Capture
      // the name (and, on done, the final assembled arguments).
      const item = isRecord(data.item) ? data.item : undefined;
      if (item && asString(item.type) === "function_call") {
        const id = asString(item.id) ?? keyOf(data);
        if (id) {
          const buf = toolBuffers.get(id) ?? { id, name: "", args: "" };
          const itemName = asString(item.name);
          if (itemName) buf.name = itemName;
          const finalArgs = asString(item.arguments);
          if (finalArgs && type === "response.output_item.done") buf.args = finalArgs;
          toolBuffers.set(id, buf);
        }
      }
    } else if (type === "response.function_call_arguments.delta") {
      const id = keyOf(data);
      if (id) {
        const buf = toolBuffers.get(id) ?? { id, name: "", args: "" };
        buf.args += asString(data.delta) ?? "";
        toolBuffers.set(id, buf);
      }
    } else if (type === "response.function_call_arguments.done") {
      const id = keyOf(data);
      const doneArgs = asString(data.arguments);
      if (id && doneArgs) {
        const buf = toolBuffers.get(id) ?? { id, name: "", args: "" };
        buf.args = doneArgs;
        toolBuffers.set(id, buf);
      }
    }
  }

  for (const [, buf] of toolBuffers) {
    if (buf.name) {
      toolCalls.push({
        id: buf.id,
        name: buf.name,
        arguments: buf.args,
        parsedArguments: parseToolArgs(buf.args),
        kind: "function",
      });
    }
  }

  return {
    protocol: "stream-responses",
    content,
    reasoning,
    toolCalls,
    toolResults: [],
    images: [],
    hasData: Boolean(content || reasoning || toolCalls.length),
  };
}

function parseStream(body: string): ParsedResponse {
  if (isAnthropicStreamResponse(body)) {
    const parsed = parseAnthropicStreamResponse(body);
    const toolCalls: ToolCallInfo[] = parsed.toolCalls.map((tc) => ({
      id: tc.id,
      name: tc.function.name,
      arguments: tc.function.arguments,
      parsedArguments: parseToolArgs(tc.function.arguments),
      kind: "function",
    }));
    return {
      protocol: "stream-anthropic",
      content: parsed.reconstructedContent,
      reasoning: parsed.reasoningContent,
      toolCalls,
      toolResults: [],
      images: [],
      hasData: Boolean(parsed.reconstructedContent || parsed.reasoningContent || toolCalls.length),
    };
  }
  if (isResponsesStreamResponse(body)) {
    return parseResponsesStream(body);
  }
  // Default OpenAI-style stream
  const parsed = parseStreamResponse(body);
  const toolCalls: ToolCallInfo[] = parsed.toolCalls.map((tc) => ({
    id: tc.id,
    name: tc.function.name,
    arguments: tc.function.arguments,
    parsedArguments: parseToolArgs(tc.function.arguments),
    kind: "function",
  }));
  return {
    protocol: "stream-openai",
    content: parsed.reconstructedContent,
    reasoning: parsed.reasoningContent,
    toolCalls,
    toolResults: [],
    images: [],
    hasData: Boolean(parsed.reconstructedContent || parsed.reasoningContent || toolCalls.length),
  };
}

// --- Public entry point ------------------------------------------------

/**
 * Parse a stored `response_body` into a protocol-agnostic structure.
 *
 * @param body        The raw `response_body` value (object, string, or other).
 * @param requestType Optional `log_metadata.request_type` ("chat" | "image_generation"
 *                    | "image_edit" | "embedding" | "speech" | "transcription"
 *                    | "translation" | ...) — used to disambiguate shapes that
 *                    share fields (e.g. audio vs chat both can have a `text`).
 */
export function parseLogResponse(body: unknown, requestType?: string): ParsedResponse {
  // Streaming responses are always stored as a raw SSE string.
  if (typeof body === "string") {
    if (
      isStreamResponse(body) ||
      isAnthropicStreamResponse(body) ||
      isResponsesStreamResponse(body)
    ) {
      return parseStream(body);
    }
    // A plain string body (rare) — surface as content.
    return {
      protocol: "unknown",
      content: body,
      reasoning: "",
      toolCalls: [],
      toolResults: [],
      images: [],
      hasData: body.trim().length > 0,
    };
  }

  if (!isRecord(body)) return empty();

  // Backend sentinel: full body was sampled out and not stored.
  if (body._sampled_out === true) {
    return empty({ protocol: "sampled-out", isSampledOut: true });
  }

  // Raw-byte audio (speech, or text-format transcription/translation).
  if (body.raw === true && typeof body.size === "number") {
    return empty({ protocol: "audio-raw", audioRaw: { size: body.size }, hasData: true });
  }

  // Images: identified by request_type OR by the {created, data[]} shape.
  if (requestType === "image_generation" || requestType === "image_edit") {
    return parseImage(body);
  }

  // Embeddings: request_type OR {object:"list", data[].embedding}.
  if (requestType === "embedding") {
    return parseEmbedding(body);
  }

  // Audio text responses (transcription/translation in json family).
  if (requestType === "transcription" || requestType === "translation") {
    if ("text" in body) return parseAudioText(body);
  }

  // Protocol detection by shape.
  const object = asString(body.object);

  // OpenAI Responses API.
  if (object === "response" || Array.isArray(body.output)) {
    return parseOpenAIResponses(body);
  }

  // OpenAI Chat.
  if (object === "chat.completion" || Array.isArray(body.choices)) {
    return parseOpenAIChat(body);
  }

  // Embeddings (shape-based fallback).
  if (object === "list" && Array.isArray(body.data) && body.data.length > 0) {
    const first = body.data[0];
    if (isRecord(first) && Array.isArray(first.embedding)) {
      return parseEmbedding(body);
    }
  }

  // Images (shape-based fallback when request_type is absent).
  if (Array.isArray(body.data) && body.data.length > 0) {
    const first = body.data[0] as Record<string, unknown> | undefined;
    if (first && ("b64_json" in first || "url" in first)) {
      return parseImage(body);
    }
  }

  // Anthropic.
  if (body.type === "message" || Array.isArray(body.content)) {
    return parseAnthropic(body);
  }

  // Audio text fallback (no request_type but has only a text field).
  if ("text" in body && !("choices" in body) && !("output" in body) && !("content" in body)) {
    return parseAudioText(body);
  }

  return empty();
}
