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
import { asString, isRecord, parseToolArgs, safeStringify } from "@/utils/logFormat";

/**
 * Reduce a Responses `web_search_call` item to its query — either the single
 * `action.query` or the `action.queries` list joined with commas.
 */
function webSearchQuery(action: Record<string, unknown> | undefined): string {
  return (
    asString(action?.query) ||
    (Array.isArray(action?.queries)
      ? (action!.queries as unknown[]).map((q) => asString(q) ?? "").join(", ")
      : "")
  );
}
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

/**
 * One output item in the order it appeared in the response — this is what
 * makes the parsed view as faithful as reading the raw stream: reasoning,
 * text and tool calls stay interleaved exactly as the model emitted them.
 */
export type ResponseItem =
  | { kind: "reasoning"; text: string; redacted?: boolean }
  | { kind: "text"; text: string }
  | { kind: "tool_call"; call: ToolCallInfo }
  | { kind: "tool_result"; result: ToolResultInfo }
  | { kind: "image"; image: ImageInfo };

/** Response-level metadata (finish reason, ids, stream event counts). */
export interface ResponseMeta {
  id?: string;
  model?: string;
  /** OpenAI finish_reason */
  finishReason?: string;
  /** Anthropic stop_reason / stop_sequence */
  stopReason?: string;
  stopSequence?: string;
  /** Responses API status (completed / incomplete / ...) */
  status?: string;
  serviceTier?: string;
  /** Number of parsed SSE events (streams only) */
  eventCount?: number;
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
  /** Ordered output items (reasoning/text/tool_call/tool_result/image) */
  items: ResponseItem[];
  /** Response-level metadata */
  meta: ResponseMeta;
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
  items: [],
  meta: {},
  images: [],
  hasData: false,
  ...overrides,
});

// --- Helpers -----------------------------------------------------------

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

  // Approximate emission order: reasoning -> text -> tool calls.
  const items: ResponseItem[] = [];
  if (reasoning) items.push({ kind: "reasoning", text: reasoning });
  if (content) items.push({ kind: "text", text: content });
  for (const call of toolCalls) items.push({ kind: "tool_call", call });

  const meta: ResponseMeta = {
    id: asString(body.id),
    model: asString(body.model),
    finishReason: asString(choice?.finish_reason),
    serviceTier: asString(body.service_tier),
  };

  const hasData = Boolean(content || reasoning || toolCalls.length);
  return {
    protocol: "openai-chat",
    content,
    reasoning,
    toolCalls,
    toolResults: [],
    items,
    meta,
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
  const items: ResponseItem[] = [];

  for (const item of output) {
    if (!isRecord(item)) continue;
    const type = asString(item.type);
    const status = asString(item.status);

    if (type === "message") {
      const parts = item.content;
      let messageText = "";
      if (Array.isArray(parts)) {
        for (const part of parts) {
          if (!isRecord(part)) continue;
          if (part.type === "output_text" || part.type === "text") {
            messageText += asString(part.text) ?? "";
          } else if (part.type === "refusal") {
            messageText += asString(part.refusal) ?? "";
          }
        }
      }
      if (messageText) {
        content += messageText;
        content += "\n";
        items.push({ kind: "text", text: messageText });
      }
    } else if (type === "reasoning") {
      const parts = item.content;
      let reasoningText = "";
      if (Array.isArray(parts)) {
        for (const part of parts) {
          if (isRecord(part) && (part.type === "reasoning_text" || part.type === "text")) {
            reasoningText += asString(part.text) ?? "";
          }
        }
      }
      // Encrypted reasoning content (when redacted)
      const encrypted = asString(item.encrypted_content);
      if (reasoningText) {
        reasoning += reasoningText;
        reasoning += "\n";
        items.push({ kind: "reasoning", text: reasoningText });
      } else if (encrypted) {
        items.push({ kind: "reasoning", text: "", redacted: true });
        if (!reasoning) reasoning = encrypted;
      }
    } else if (type === "function_call" || type === "custom_tool_call") {
      const args = asString(item.arguments) ?? "";
      const input = item.input;
      const argStr = args || safeStringify(input) || "";
      const call: ToolCallInfo = {
        id: asString(item.id) || asString(item.call_id),
        name: asString(item.name) ?? "",
        arguments: argStr,
        parsedArguments: parseToolArgs(argStr),
        kind: type === "custom_tool_call" ? "custom" : "function",
        status,
      };
      toolCalls.push(call);
      items.push({ kind: "tool_call", call });
    } else if (type === "web_search_call") {
      const query = webSearchQuery(isRecord(item.action) ? item.action : undefined);
      const call: ToolCallInfo = {
        id: asString(item.id),
        name: "web_search",
        arguments: query ? JSON.stringify({ query }) : "",
        parsedArguments: query ? { query } : undefined,
        kind: "web_search",
        status,
      };
      toolCalls.push(call);
      items.push({ kind: "tool_call", call });
    } else if (type === "tool_search_call") {
      const args = safeStringify(item.arguments) ?? "";
      const call: ToolCallInfo = {
        id: asString(item.id) || asString(item.call_id),
        name: "tool_search",
        arguments: args,
        parsedArguments: parseToolArgs(args),
        kind: "tool_search",
        status,
      };
      toolCalls.push(call);
      items.push({ kind: "tool_call", call });
    } else if (type === "function_call_output" || type === "tool_search_output") {
      const result: ToolResultInfo = {
        callId: asString(item.call_id),
        output: asString(item.output) ?? safeStringify(item.output) ?? "",
        kind: type,
      };
      toolResults.push(result);
      items.push({ kind: "tool_result", result });
    } else if (type === "image_generation_call") {
      const b64 = asString(item.result);
      if (b64) {
        const image: ImageInfo = { b64Json: b64, outputFormat: asString(item.output_format) };
        items.push({ kind: "image", image });
      }
    }
  }

  const incomplete = isRecord(body.incomplete_details)
    ? asString(body.incomplete_details.reason)
    : undefined;
  const meta: ResponseMeta = {
    id: asString(body.id),
    model: asString(body.model),
    status: asString(body.status),
    finishReason: incomplete,
    serviceTier: asString(body.service_tier),
  };

  const images = items.filter((i) => i.kind === "image").map((i) => i.image);
  const hasData = Boolean(
    content || reasoning || toolCalls.length || toolResults.length || images.length
  );
  return {
    protocol: "openai-responses",
    content: content.trim(),
    reasoning: reasoning.trim(),
    toolCalls,
    toolResults,
    items,
    meta,
    images,
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
  const items: ResponseItem[] = [];

  for (const block of contentBlocks) {
    if (!isRecord(block)) continue;
    const type = asString(block.type);

    switch (type) {
      case "text": {
        const text = asString(block.text) ?? "";
        if (text) {
          content += text;
          content += "\n";
          items.push({ kind: "text", text });
        }
        break;
      }
      case "thinking": {
        const thinking = asString(block.thinking) ?? "";
        if (thinking) {
          reasoning += thinking;
          reasoning += "\n";
          items.push({ kind: "reasoning", text: thinking });
        }
        break;
      }
      case "redacted_thinking":
        reasoning += "[redacted]\n";
        items.push({ kind: "reasoning", text: "", redacted: true });
        break;
      case "tool_use":
      case "server_tool_use": {
        const input = block.input;
        const argStr = safeStringify(input) ?? "";
        const call: ToolCallInfo = {
          id: asString(block.id),
          name: asString(block.name) ?? "",
          arguments: argStr,
          parsedArguments: isRecord(input) ? input : undefined,
          kind: type === "server_tool_use" ? "server_tool_use" : "function",
        };
        toolCalls.push(call);
        items.push({ kind: "tool_call", call });
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
                    return safeStringify(p) ?? "";
                  })
                  .join("\n")
              : (safeStringify(c) ?? "");
        const result: ToolResultInfo = {
          toolUseId: asString(block.tool_use_id),
          output: out,
          isError: block.is_error === true,
          kind: "tool_result",
        };
        toolResults.push(result);
        items.push({ kind: "tool_result", result });
        break;
      }
      case "image": {
        const source = isRecord(block.source) ? block.source : undefined;
        const mediaType = asString(source?.media_type) ?? "image/png";
        const data = asString(source?.data);
        const url = asString(source?.url);
        let image: ImageInfo | undefined;
        if (data) {
          image = { b64Json: data, outputFormat: mediaType.replace("image/", "") };
        } else if (url) {
          image = { url, outputFormat: mediaType.replace("image/", "") };
        }
        if (image) {
          images.push(image);
          items.push({ kind: "image", image });
        }
        break;
      }
      default:
        break;
    }
  }

  const meta: ResponseMeta = {
    id: asString(body.id),
    model: asString(body.model),
    stopReason: asString(body.stop_reason),
    stopSequence: asString(body.stop_sequence),
    serviceTier: asString(body.service_tier),
  };

  const hasData = Boolean(
    content || reasoning || toolCalls.length || toolResults.length || images.length
  );
  return {
    protocol: "anthropic",
    content: content.trim(),
    reasoning: reasoning.trim(),
    toolCalls,
    toolResults,
    items,
    meta,
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
    items: [],
    meta: {},
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
    items: [],
    meta: { model: asString(body.model) },
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
    items: [],
    meta: {},
    images: [],
    audioText: text,
    hasData: Boolean(text),
  };
}

function parseResponsesStream(body: string): ParsedResponse {
  // Reconstruct ordered output items from Responses SSE events.
  // Items are announced with response.output_item.added (carrying output_index
  // and the item skeleton), filled by *.delta events, and finalized by
  // response.output_item.done. Slots are keyed by output_index so the final
  // item order matches the model's emission order exactly.
  const lines = body.split("\n");

  interface ItemSlot {
    type: string;
    id?: string;
    callId?: string;
    name?: string;
    args: string;
    text: string;
    reasoning: string;
    status?: string;
    query?: string;
    b64?: string;
    outputFormat?: string;
  }

  const slots = new Map<number, ItemSlot>();
  const indexByItemId = new Map<string, number>();
  const meta: ResponseMeta = {};
  let eventCount = 0;

  const slotOf = (data: Record<string, unknown>): ItemSlot | undefined => {
    const idx =
      typeof data.output_index === "number"
        ? data.output_index
        : indexByItemId.get(asString(data.item_id) ?? "");
    return idx !== undefined ? slots.get(idx) : undefined;
  };

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
    eventCount++;

    const type = asString(data.type) ?? "";

    if (type === "response.output_item.added") {
      const item = isRecord(data.item) ? data.item : undefined;
      const idx = typeof data.output_index === "number" ? data.output_index : slots.size;
      if (item) {
        const slot: ItemSlot = {
          type: asString(item.type) ?? "",
          id: asString(item.id),
          callId: asString(item.call_id),
          name: asString(item.name),
          args: "",
          text: "",
          reasoning: "",
          status: asString(item.status),
        };
        slots.set(idx, slot);
        if (slot.id) indexByItemId.set(slot.id, idx);
      }
    } else if (type === "response.output_text.delta") {
      const slot = slotOf(data);
      if (slot) slot.text += asString(data.delta) ?? "";
    } else if (
      type === "response.reasoning_text.delta" ||
      type === "response.reasoning_summary_text.delta"
    ) {
      const slot = slotOf(data);
      if (slot) slot.reasoning += asString(data.delta) ?? "";
    } else if (type === "response.function_call_arguments.delta") {
      const slot = slotOf(data);
      if (slot) slot.args += asString(data.delta) ?? "";
    } else if (type === "response.function_call_arguments.done") {
      const slot = slotOf(data);
      const doneArgs = asString(data.arguments);
      if (slot && doneArgs) slot.args = doneArgs;
    } else if (type === "response.output_item.done") {
      const item = isRecord(data.item) ? data.item : undefined;
      const slot = slotOf(data);
      if (item && slot) {
        slot.status = asString(item.status) ?? slot.status;
        slot.name = asString(item.name) ?? slot.name;
        slot.callId = slot.callId ?? asString(item.call_id);
        const finalArgs = asString(item.arguments);
        if (finalArgs) slot.args = finalArgs;
        // web_search_call carries the query in its action.
        const query = webSearchQuery(isRecord(item.action) ? item.action : undefined);
        if (query) slot.query = query;
        // image_generation_call carries the finished base64 image in `result`.
        if (slot.type === "image_generation_call") {
          const b64 = asString(item.result);
          if (b64) {
            slot.b64 = b64;
            slot.outputFormat = asString(item.output_format);
          }
        }
      }
    } else if (type === "response.completed" || type === "response.created") {
      const response = isRecord(data.response) ? data.response : undefined;
      if (response) {
        meta.id = asString(response.id) ?? meta.id;
        meta.model = asString(response.model) ?? meta.model;
        meta.serviceTier = asString(response.service_tier) ?? meta.serviceTier;
        if (type === "response.completed") {
          meta.status = asString(response.status);
          const incomplete = isRecord(response.incomplete_details)
            ? asString(response.incomplete_details.reason)
            : undefined;
          meta.finishReason = incomplete;
        }
      }
    } else if (type === "response.incomplete" || type === "response.failed") {
      meta.status = type.replace("response.", "");
    }
  }

  let content = "";
  let reasoning = "";
  const toolCalls: ToolCallInfo[] = [];
  const items: ResponseItem[] = [];
  const images: ImageInfo[] = [];

  for (const [, slot] of [...slots.entries()].sort((a, b) => a[0] - b[0])) {
    if (slot.type === "message") {
      if (slot.text) {
        content += slot.text;
        items.push({ kind: "text", text: slot.text });
      }
    } else if (slot.type === "reasoning") {
      if (slot.reasoning) {
        reasoning += slot.reasoning;
        items.push({ kind: "reasoning", text: slot.reasoning });
      } else {
        items.push({ kind: "reasoning", text: "", redacted: true });
      }
    } else if (slot.type === "function_call" || slot.type === "custom_tool_call") {
      const call: ToolCallInfo = {
        id: slot.id || slot.callId,
        name: slot.name ?? "",
        arguments: slot.args,
        parsedArguments: parseToolArgs(slot.args),
        kind: slot.type === "custom_tool_call" ? "custom" : "function",
        status: slot.status,
      };
      toolCalls.push(call);
      items.push({ kind: "tool_call", call });
    } else if (slot.type === "web_search_call") {
      const call: ToolCallInfo = {
        id: slot.id,
        name: "web_search",
        arguments: slot.query ? JSON.stringify({ query: slot.query }) : "",
        parsedArguments: slot.query ? { query: slot.query } : undefined,
        kind: "web_search",
        status: slot.status,
      };
      toolCalls.push(call);
      items.push({ kind: "tool_call", call });
    } else if (slot.type === "tool_search_call") {
      const call: ToolCallInfo = {
        id: slot.id || slot.callId,
        name: "tool_search",
        arguments: slot.args,
        parsedArguments: parseToolArgs(slot.args),
        kind: "tool_search",
        status: slot.status,
      };
      toolCalls.push(call);
      items.push({ kind: "tool_call", call });
    } else if (slot.type === "image_generation_call") {
      // The finished image only arrives on response.output_item.done.
      if (slot.b64) {
        const image: ImageInfo = { b64Json: slot.b64, outputFormat: slot.outputFormat };
        images.push(image);
        items.push({ kind: "image", image });
      }
    }
  }

  meta.eventCount = eventCount;

  return {
    protocol: "stream-responses",
    content,
    reasoning,
    toolCalls,
    toolResults: [],
    items,
    meta,
    images,
    hasData: Boolean(content || reasoning || toolCalls.length || images.length),
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
    // Faithful per-block ordering from content-block indices.
    const items: ResponseItem[] = [];
    for (const block of parsed.blocks) {
      if (block.type === "text") {
        if (block.text) items.push({ kind: "text", text: block.text });
      } else if (block.type === "thinking") {
        if (block.thinking) items.push({ kind: "reasoning", text: block.thinking });
      } else if (block.type === "redacted_thinking") {
        items.push({ kind: "reasoning", text: "", redacted: true });
      } else if (block.type === "tool_use") {
        const call = toolCalls.find((tc) => tc.id === block.id);
        if (call) items.push({ kind: "tool_call", call });
      }
    }
    return {
      protocol: "stream-anthropic",
      content: parsed.reconstructedContent,
      reasoning: parsed.reasoningContent,
      toolCalls,
      toolResults: [],
      items,
      meta: {
        ...parsed.meta,
      },
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
  // Approximate emission order: reasoning -> text -> tool calls.
  const items: ResponseItem[] = [];
  if (parsed.reasoningContent) items.push({ kind: "reasoning", text: parsed.reasoningContent });
  if (parsed.reconstructedContent) items.push({ kind: "text", text: parsed.reconstructedContent });
  for (const call of toolCalls) items.push({ kind: "tool_call", call });
  return {
    protocol: "stream-openai",
    content: parsed.reconstructedContent,
    reasoning: parsed.reasoningContent,
    toolCalls,
    toolResults: [],
    items,
    meta: {
      ...parsed.meta,
    },
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
      items: body.trim() ? [{ kind: "text", text: body }] : [],
      meta: {},
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
