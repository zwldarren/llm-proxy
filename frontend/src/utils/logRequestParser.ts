/**
 * Log request parser.
 *
 * Normalizes every inbound-protocol `request_body` shape stored by the proxy
 * into a single structure so the Logs UI can render the request faithfully:
 * ALL parameters (not just a hardcoded few), the offered tool definitions,
 * the system prompt, and the conversation messages with typed content blocks
 * (text / thinking / tool_call / tool_result / image / audio / file).
 *
 * Supported inbound protocols (the proxy only stores client-facing shapes):
 *  - OpenAI Chat        (`messages[]`, `tools[].function`, top-level scalars)
 *  - Anthropic          (`messages[]`, top-level `system`, `tools[].input_schema`)
 *  - OpenAI Responses   (`input` string|items, `instructions`, `tools[]`)
 *
 * Design goals:
 *  - Nothing in the body is hidden: unrecognized top-level keys land in
 *    `objectParams`, unrecognized content blocks land in `other` blocks.
 *  - Never inline base64 blobs: image/audio/file blocks become small chips
 *    carrying a byte-size estimate, so multi-MB vision requests don't freeze
 *    the details sheet.
 *  - Cheap: strings are referenced, not copied; charCount is O(1) `.length`.
 */

import { asString, isRecord, parseToolArgs, safeStringify } from "@/utils/logFormat";

// --- Types -------------------------------------------------------------

export type RequestProtocol = "openai-chat" | "anthropic" | "responses" | "unknown";

export type MessageBlock =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string; redacted?: boolean }
  | {
      kind: "tool_call";
      id?: string;
      name: string;
      arguments: string;
      parsedArguments: Record<string, unknown>;
    }
  | { kind: "tool_result"; id?: string; output: string; isError?: boolean }
  | { kind: "image"; url?: string; bytes?: number; mediaType?: string; detail?: string }
  | { kind: "audio"; bytes?: number; format?: string }
  | { kind: "file"; name?: string; bytes?: number }
  | { kind: "other"; label: string; raw: unknown };

export interface ParsedLogMessage {
  /** Display role: system / developer / user / assistant / tool / reasoning */
  role: string;
  /** OpenAI message.name (participant name) */
  name?: string;
  /** OpenAI role=tool messages: the call this message answers */
  toolCallId?: string;
  blocks: MessageBlock[];
  /** Concatenated text for one-line previews */
  plainText: string;
  /** Total character count across text-ish blocks (for collapse hints) */
  charCount: number;
}

export interface RequestToolInfo {
  name: string;
  /** function | custom | web_search | code_interpreter | mcp | file_search | ... */
  kind: string;
  description?: string;
  /** JSON schema of the parameters (functions only) */
  schema?: unknown;
  /** One-line detail for non-function kinds (e.g. mcp server label) */
  summary?: string;
  /** Original definition, shown when there is no schema */
  raw?: unknown;
}

export interface RequestScalarParam {
  key: string;
  value: string;
}

export interface RequestObjectParam {
  key: string;
  value: unknown;
}

export interface ParsedLogRequest {
  protocol: RequestProtocol;
  /** All scalar top-level params (model, temperature, seed, ...) in body order */
  scalarParams: RequestScalarParam[];
  /** Complex top-level params (response_format, thinking, metadata, ...) */
  objectParams: RequestObjectParam[];
  /** Formatted tool_choice value, if present */
  toolChoice?: string;
  /** Offered tool definitions */
  tools: RequestToolInfo[];
  /** System prompt (top-level system/instructions, or leading system message) */
  systemPrompt: string;
  /** Conversation messages (leading system message extracted into systemPrompt) */
  messages: ParsedLogMessage[];
  /** False when the body matched no known chat shape (non-standard payload) */
  isChatLike: boolean;
}

// --- Helpers -----------------------------------------------------------

function isScalar(value: unknown): value is string | number | boolean | null {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

/** Approximate decoded byte size of a base64 payload. */
function base64Bytes(b64: string): number {
  return Math.floor((b64.length * 3) / 4);
}

/** Parse a data: URL into media type + decoded byte estimate. */
function parseDataUrl(url: string): { mediaType?: string; bytes?: number } | null {
  if (!url.startsWith("data:")) return null;
  const comma = url.indexOf(",");
  if (comma === -1) return null;
  const meta = url.slice(5, comma);
  const mediaType = meta.split(";")[0] || undefined;
  const isBase64 = meta.includes(";base64");
  const data = url.slice(comma + 1);
  return { mediaType, bytes: isBase64 ? base64Bytes(data) : data.length };
}

function formatScalar(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

/** Compact one-line summary of an object, for odd tool kinds. */
function compactSummary(obj: Record<string, unknown>, skipKeys: string[]): string | undefined {
  const rest: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (!skipKeys.includes(k) && v !== undefined) rest[k] = v;
  }
  if (Object.keys(rest).length === 0) return undefined;
  try {
    const json = JSON.stringify(rest);
    return json.length > 96 ? json.slice(0, 93) + "…" : json;
  } catch {
    return undefined;
  }
}

/** Top-level keys rendered as dedicated sections, never as generic params. */
const SECTION_KEYS = new Set(["messages", "input", "tools", "functions", "system", "instructions"]);

// --- Block-level normalization ------------------------------------------

function textBlock(text: unknown): MessageBlock | null {
  const s = asString(text);
  return s ? { kind: "text", text: s } : null;
}

function imageBlockFromUrl(url: string, detail?: string): MessageBlock {
  const dataUrl = parseDataUrl(url);
  if (dataUrl) {
    return { kind: "image", bytes: dataUrl.bytes, mediaType: dataUrl.mediaType, detail };
  }
  return { kind: "image", url, detail };
}

/** OpenAI Chat content-part blocks (multimodal messages). */
function openaiContentBlock(block: Record<string, unknown>): MessageBlock {
  const type = asString(block.type) ?? "";
  switch (type) {
    case "text": {
      return textBlock(block.text) ?? { kind: "other", label: type, raw: block };
    }
    case "image_url": {
      const img = isRecord(block.image_url) ? block.image_url : {};
      const url = asString(img.url) ?? "";
      return imageBlockFromUrl(url, asString(img.detail));
    }
    case "input_audio": {
      const audio = isRecord(block.input_audio) ? block.input_audio : {};
      const data = asString(audio.data);
      return {
        kind: "audio",
        bytes: data ? base64Bytes(data) : undefined,
        format: asString(audio.format),
      };
    }
    case "file": {
      const file = isRecord(block.file) ? block.file : {};
      const data = asString(file.file_data);
      return {
        kind: "file",
        name: asString(file.filename) ?? asString(file.file_id),
        bytes: data ? base64Bytes(data) : undefined,
      };
    }
    case "refusal": {
      const t = textBlock(block.refusal);
      return t ?? { kind: "other", label: type, raw: block };
    }
    default:
      return { kind: "other", label: type || "block", raw: block };
  }
}

/** Anthropic content blocks (multimodal + tool history). */
function anthropicContentBlock(block: Record<string, unknown>): MessageBlock {
  const type = asString(block.type) ?? "";
  switch (type) {
    case "text":
      return textBlock(block.text) ?? { kind: "other", label: type, raw: block };
    case "thinking": {
      const t = asString(block.thinking);
      return t ? { kind: "thinking", text: t } : { kind: "other", label: type, raw: block };
    }
    case "redacted_thinking":
      return { kind: "thinking", text: "", redacted: true };
    case "image": {
      const source = isRecord(block.source) ? block.source : {};
      if (asString(source.type) === "base64") {
        const data = asString(source.data);
        return {
          kind: "image",
          bytes: data ? base64Bytes(data) : undefined,
          mediaType: asString(source.media_type),
        };
      }
      const url = asString(source.url);
      if (url) return { kind: "image", url };
      return { kind: "image", mediaType: asString(source.media_type) };
    }
    case "tool_use":
    case "server_tool_use": {
      const args = JSON.stringify(block.input ?? {});
      return {
        kind: "tool_call",
        id: asString(block.id),
        name: asString(block.name) ?? "",
        arguments: args,
        parsedArguments: parseToolArgs(args),
      };
    }
    case "tool_result":
    case "web_search_tool_result": {
      return {
        kind: "tool_result",
        id: asString(block.tool_use_id),
        output: flattenAnthropicResultContent(block.content),
        isError: block.is_error === true,
      };
    }
    case "document": {
      const source = isRecord(block.source) ? block.source : {};
      const data = asString(source.data);
      return {
        kind: "file",
        name: asString(block.title) ?? asString(source.media_type),
        bytes: data ? base64Bytes(data) : undefined,
      };
    }
    default:
      return { kind: "other", label: type || "block", raw: block };
  }
}

function flattenAnthropicResultContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => {
        if (typeof p === "string") return p;
        if (isRecord(p) && p.type === "text") return asString(p.text) ?? "";
        try {
          return JSON.stringify(p);
        } catch {
          return "";
        }
      })
      .filter(Boolean)
      .join("\n");
  }
  if (content === null || content === undefined) return "";
  try {
    return JSON.stringify(content);
  } catch {
    return "";
  }
}

/** Responses API input-item content parts (input_text / input_image / ...). */
function responsesContentBlock(block: Record<string, unknown>): MessageBlock {
  const type = asString(block.type) ?? "";
  switch (type) {
    case "input_text":
    case "output_text":
    case "text":
      return textBlock(block.text) ?? { kind: "other", label: type, raw: block };
    case "input_image": {
      const url = asString(block.image_url);
      if (url) return imageBlockFromUrl(url, asString(block.detail));
      const fileId = asString(block.file_id);
      if (fileId) return { kind: "image", url: `file:${fileId}` };
      return { kind: "image", detail: asString(block.detail) };
    }
    case "input_audio": {
      const data = asString(block.data);
      return {
        kind: "audio",
        bytes: data ? base64Bytes(data) : undefined,
        format: asString(block.format),
      };
    }
    case "input_file": {
      const data = asString(block.file_data);
      return {
        kind: "file",
        name: asString(block.filename) ?? asString(block.file_id),
        bytes: data ? base64Bytes(data) : undefined,
      };
    }
    case "refusal":
      return textBlock(block.refusal) ?? { kind: "other", label: type, raw: block };
    default:
      return { kind: "other", label: type || "block", raw: block };
  }
}

// --- Message assembly ----------------------------------------------------

function makeMessage(
  role: string,
  blocks: MessageBlock[],
  extra: { name?: string; toolCallId?: string } = {}
): ParsedLogMessage {
  const visible = blocks.filter(
    (b) =>
      !(b.kind === "text" && b.text.length === 0) &&
      !(b.kind === "thinking" && b.text.length === 0 && !b.redacted)
  );
  const plainParts: string[] = [];
  let charCount = 0;
  for (const b of visible) {
    if (b.kind === "text" || b.kind === "thinking") {
      plainParts.push(b.text);
      charCount += b.text.length;
    } else if (b.kind === "tool_call") {
      plainParts.push(`[tool: ${b.name}]`);
      charCount += b.arguments.length;
    } else if (b.kind === "tool_result") {
      charCount += b.output.length;
    } else if (b.kind === "image") {
      plainParts.push("[image]");
    } else if (b.kind === "audio") {
      plainParts.push("[audio]");
    } else if (b.kind === "file") {
      plainParts.push("[file]");
    }
  }
  return {
    role,
    name: extra.name,
    toolCallId: extra.toolCallId,
    blocks: visible,
    plainText: plainParts.join("\n"),
    charCount,
  };
}

function parseOpenAIMessage(msg: Record<string, unknown>): ParsedLogMessage {
  const role = asString(msg.role) ?? "unknown";
  const blocks: MessageBlock[] = [];
  const content = msg.content;

  if (role === "tool") {
    // A tool message IS a tool result — render it as such.
    let output = "";
    if (typeof content === "string") output = content;
    else if (Array.isArray(content)) {
      output = content
        .map((p) => {
          if (typeof p === "string") return p;
          if (isRecord(p)) {
            const b = openaiContentBlock(p);
            if (b.kind === "text") return b.text;
          }
          try {
            return JSON.stringify(p);
          } catch {
            return "";
          }
        })
        .filter(Boolean)
        .join("\n");
    }
    blocks.push({ kind: "tool_result", id: asString(msg.tool_call_id), output });
  } else {
    if (typeof content === "string") {
      const b = textBlock(content);
      if (b) blocks.push(b);
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === "string") {
          const b = textBlock(part);
          if (b) blocks.push(b);
        } else if (isRecord(part)) {
          blocks.push(openaiContentBlock(part));
        }
      }
    }

    const toolCalls = msg.tool_calls;
    if (Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        if (!isRecord(tc)) continue;
        const fn = isRecord(tc.function) ? tc.function : {};
        const args = asString(fn.arguments) ?? "";
        blocks.push({
          kind: "tool_call",
          id: asString(tc.id),
          name: asString(fn.name) ?? "",
          arguments: args,
          parsedArguments: parseToolArgs(args),
        });
      }
    }

    const refusal = asString(msg.refusal);
    if (refusal && blocks.length === 0) blocks.push({ kind: "text", text: refusal });
  }

  return makeMessage(role, blocks, {
    name: asString(msg.name),
    toolCallId: asString(msg.tool_call_id),
  });
}

function parseAnthropicMessage(msg: Record<string, unknown>): ParsedLogMessage {
  const role = asString(msg.role) ?? "unknown";
  const blocks: MessageBlock[] = [];
  const content = msg.content;

  if (typeof content === "string") {
    const b = textBlock(content);
    if (b) blocks.push(b);
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (typeof part === "string") {
        const b = textBlock(part);
        if (b) blocks.push(b);
      } else if (isRecord(part)) {
        blocks.push(anthropicContentBlock(part));
      }
    }
  }

  return makeMessage(role, blocks, { name: asString(msg.name) });
}

function parseResponsesInputItem(item: unknown): ParsedLogMessage | null {
  if (typeof item === "string") return makeMessage("user", [{ kind: "text", text: item }]);
  if (!isRecord(item)) return null;

  const type = asString(item.type);

  // Plain message item (type may be omitted on hand-rolled payloads).
  if (type === "message" || (!type && typeof item.role === "string")) {
    const role = asString(item.role) ?? "user";
    const blocks: MessageBlock[] = [];
    const content = item.content;
    if (typeof content === "string") {
      const b = textBlock(content);
      if (b) blocks.push(b);
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === "string") {
          const b = textBlock(part);
          if (b) blocks.push(b);
        } else if (isRecord(part)) {
          blocks.push(responsesContentBlock(part));
        }
      }
    }
    return makeMessage(role, blocks, { name: asString(item.name) });
  }

  if (type === "function_call" || type === "custom_tool_call") {
    const args =
      asString(item.arguments) ?? (item.input !== undefined ? safeStringify(item.input) : "") ?? "";
    return makeMessage("assistant", [
      {
        kind: "tool_call",
        id: asString(item.call_id) ?? asString(item.id),
        name: asString(item.name) ?? "",
        arguments: args,
        parsedArguments: parseToolArgs(args),
      },
    ]);
  }

  if (type === "function_call_output") {
    const output = asString(item.output) ?? safeStringify(item.output) ?? "";
    return makeMessage("tool", [{ kind: "tool_result", id: asString(item.call_id), output }], {
      toolCallId: asString(item.call_id),
    });
  }

  if (type === "reasoning") {
    let text = "";
    const content = item.content;
    if (Array.isArray(content)) {
      for (const part of content) {
        if (isRecord(part)) text += asString(part.text) ?? "";
      }
    }
    if (!text) {
      const summary = item.summary;
      if (Array.isArray(summary)) {
        for (const part of summary) {
          if (isRecord(part)) text += asString(part.text) ?? "";
        }
      }
    }
    if (!text && asString(item.encrypted_content)) {
      return makeMessage("reasoning", [{ kind: "thinking", text: "", redacted: true }]);
    }
    return makeMessage("reasoning", [{ kind: "thinking", text }]);
  }

  if (type === "item_reference") {
    return makeMessage("reference", [
      { kind: "other", label: "item_reference", raw: asString(item.id) ?? item },
    ]);
  }

  // Generic "*_call" items with a name render as tool calls.
  if (type && type.endsWith("_call") && typeof item.name === "string") {
    const args = safeStringify(item.action ?? item.arguments ?? item.input) ?? "";
    return makeMessage("assistant", [
      {
        kind: "tool_call",
        id: asString(item.call_id) ?? asString(item.id),
        name: item.name,
        arguments: args,
        parsedArguments: parseToolArgs(args),
      },
    ]);
  }

  return makeMessage(asString(item.role) ?? type ?? "unknown", [
    { kind: "other", label: type ?? "item", raw: item },
  ]);
}

// --- Tool definitions ----------------------------------------------------

function normalizeToolDefs(body: Record<string, unknown>): RequestToolInfo[] {
  const tools: RequestToolInfo[] = [];

  const rawTools = Array.isArray(body.tools) ? body.tools : [];
  for (const raw of rawTools) {
    if (!isRecord(raw)) continue;
    const type = asString(raw.type);

    // OpenAI Chat: {type:"function"|"custom", function:{...}} — or Anthropic
    // custom tool {name, description, input_schema} with no wrapper.
    const fn = isRecord(raw.function) ? raw.function : undefined;
    if (fn) {
      tools.push({
        name: asString(fn.name) ?? "",
        kind: type ?? "function",
        description: asString(fn.description),
        schema: fn.parameters,
      });
      continue;
    }
    if (type === "custom" && isRecord(raw.custom)) {
      tools.push({
        name: asString(raw.custom.name) ?? "",
        kind: "custom",
        description: asString(raw.custom.description),
        schema: raw.custom.format,
      });
      continue;
    }

    // Anthropic tool: {name, description, input_schema}
    if (isRecord(raw.input_schema)) {
      tools.push({
        name: asString(raw.name) ?? "",
        kind: type ?? "function",
        description: asString(raw.description),
        schema: raw.input_schema,
      });
      continue;
    }

    // Responses function tool: {type:"function", name, description, parameters}
    if (type === "function" && typeof raw.name === "string") {
      tools.push({
        name: raw.name,
        kind: "function",
        description: asString(raw.description),
        schema: raw.parameters,
      });
      continue;
    }

    // Server / built-in tools (web_search, code_interpreter, mcp, file_search,
    // image_generation, anthropic web_search_20250305, ...).
    const name = asString(raw.name) ?? type ?? "tool";
    tools.push({
      name,
      kind: type ?? "builtin",
      description: asString(raw.description),
      summary: compactSummary(raw, ["type", "name", "description"]),
      raw,
    });
  }

  // Legacy OpenAI functions[].
  const rawFunctions = Array.isArray(body.functions) ? body.functions : [];
  for (const raw of rawFunctions) {
    if (!isRecord(raw)) continue;
    tools.push({
      name: asString(raw.name) ?? "",
      kind: "function",
      description: asString(raw.description),
      schema: raw.parameters,
    });
  }

  return tools;
}

function formatToolChoice(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") return value;
  if (isRecord(value)) {
    const type = asString(value.type);
    const name =
      asString(value.name) ??
      (isRecord(value.function) ? asString(value.function.name) : undefined);
    if (type && name) return `${type}: ${name}`;
    if (name) return name;
    if (type) return type;
    return compactSummary(value, []) ?? undefined;
  }
  return String(value);
}

// --- System prompt --------------------------------------------------------

function flattenSystemPrompt(system: unknown): string {
  if (typeof system === "string") return system;
  if (Array.isArray(system)) {
    // Anthropic: system can be an array of text blocks (with cache_control).
    return system
      .map((part) => {
        if (typeof part === "string") return part;
        if (isRecord(part) && part.type === "text") return asString(part.text) ?? "";
        return "";
      })
      .filter(Boolean)
      .join("\n\n");
  }
  return "";
}

function flattenMessageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => {
        if (typeof p === "string") return p;
        if (isRecord(p)) return asString(p.text) ?? "";
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

// --- Public entry point ----------------------------------------------------

/**
 * Parse a stored `request_body` into a protocol-agnostic structure.
 * Returns null when the body isn't a non-empty object.
 */
export function parseLogRequest(body: unknown): ParsedLogRequest | null {
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return null;
    }
  }
  if (!isRecord(body)) return null;

  // --- Params: every top-level scalar except section keys ----------------
  const scalarParams: RequestScalarParam[] = [];
  const objectParams: RequestObjectParam[] = [];
  for (const [key, value] of Object.entries(body)) {
    if (SECTION_KEYS.has(key) || value === undefined) continue;
    if (isScalar(value)) {
      scalarParams.push({ key, value: formatScalar(value) });
    } else if (Array.isArray(value) && value.every(isScalar)) {
      // stop sequences, include[], modalities... render as a joined scalar.
      scalarParams.push({ key, value: value.map(formatScalar).join(", ") });
    } else {
      objectParams.push({ key, value });
    }
  }

  const tools = normalizeToolDefs(body);
  const toolChoice = formatToolChoice(body.tool_choice);

  // --- Protocol-specific conversation extraction --------------------------
  let protocol: RequestProtocol = "unknown";
  let systemPrompt = "";
  let messages: ParsedLogMessage[] = [];
  let isChatLike = false;

  if (Array.isArray(body.messages)) {
    isChatLike = true;
    const rawMessages = body.messages.filter(isRecord);

    // Anthropic top-level system (string or block array).
    if (body.system !== undefined) {
      protocol = "anthropic";
      systemPrompt = flattenSystemPrompt(body.system);
      messages = rawMessages.map(parseAnthropicMessage);
    } else {
      protocol = "openai-chat";
      const parsed = rawMessages.map(parseOpenAIMessage);
      // Extract a leading system/developer message into the system section.
      const first = parsed[0];
      if (first && (first.role === "system" || first.role === "developer")) {
        systemPrompt = first.plainText;
        messages = parsed.slice(1);
      } else {
        messages = parsed;
      }
    }
  } else if (body.input !== undefined || typeof body.instructions === "string") {
    // OpenAI Responses API.
    protocol = "responses";
    isChatLike = true;
    systemPrompt = asString(body.instructions) ?? "";
    const input = body.input;
    if (typeof input === "string") {
      messages = [makeMessage("user", [{ kind: "text", text: input }])];
    } else if (Array.isArray(input)) {
      messages = input
        .map(parseResponsesInputItem)
        .filter((m): m is ParsedLogMessage => m !== null);
    }
  }

  // Anthropic system arrays may hold non-text entries; fall back to raw text.
  if (!systemPrompt && protocol === "anthropic" && body.system !== undefined) {
    systemPrompt = flattenMessageText(body.system);
  }

  return {
    protocol,
    scalarParams,
    objectParams,
    toolChoice,
    tools,
    systemPrompt,
    messages,
    isChatLike,
  };
}
