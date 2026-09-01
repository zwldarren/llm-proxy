import { describe, expect, it } from "vitest";
import { parseLogResponse } from "./logResponseParser";

/**
 * Verifies tool-call parsing across all three inbound chat protocols, both
 * non-streaming and streaming, since that was the reported blind spot.
 */

// --- /v1/chat/completions (OpenAI Chat) --------------------------------

const openaiChatToolCallBody = {
  id: "chatcmpl-1",
  object: "chat.completion",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_abc",
            type: "function",
            function: { name: "get_weather", arguments: '{"location":"Paris"}' },
          },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
};

// OpenAI Chat STREAM with an incremental tool call (arguments split across
// chunks, indexed). This is the shape the old stream parser dropped.
const openaiChatToolCallStream = [
  'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}',
  "",
  'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc"}}]},"finish_reason":null}]}',
  "",
  'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\":\\"Paris\\"}"}}]},"finish_reason":null}]}',
  "",
  'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
  "",
  "data: [DONE]",
  "",
].join("\n");

// --- /v1/messages (Anthropic) -----------------------------------------

const anthropicToolUseBody = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  content: [
    { type: "text", text: "I'll check the weather." },
    { type: "tool_use", id: "toolu_1", name: "get_weather", input: { location: "Paris" } },
  ],
};

const anthropicToolUseStream = [
  'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude"}}',
  "",
  'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
  "",
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I\'ll check."}}',
  "",
  'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
  "",
  'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}',
  "",
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"locat"}}',
  "",
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"ion\\":\\"Paris\\"}"}}',
  "",
  'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}',
  "",
  'event: message_stop\ndata: {"type":"message_stop"}',
  "",
].join("\n");

// --- /v1/responses (OpenAI Responses) ---------------------------------

const responsesToolCallBody = {
  id: "resp_1",
  object: "response",
  status: "completed",
  output: [
    {
      type: "reasoning",
      id: "item_r",
      content: [{ type: "reasoning_text", text: "deciding to call a tool" }],
    },
    {
      type: "function_call",
      id: "item_f",
      call_id: "call_1",
      name: "get_weather",
      status: "completed",
      arguments: '{"location":"Paris"}',
    },
    {
      type: "function_call_output",
      id: "item_o",
      call_id: "call_1",
      output: '{"temp":20}',
      status: "completed",
    },
  ],
  usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
};

const responsesToolCallStream = [
  'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}',
  "",
  'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"item_f","call_id":"call_1","name":"get_weather","arguments":""}}',
  "",
  'event: response.function_call_arguments.delta\ndata: {"type":"response.function_call_arguments.delta","item_id":"item_f","output_index":0,"delta":"{\\"loc"}',
  "",
  'event: response.function_call_arguments.delta\ndata: {"type":"response.function_call_arguments.delta","item_id":"item_f","output_index":0,"delta":"ation\\":\\"Paris\\"}"}',
  "",
  'event: response.function_call_arguments.done\ndata: {"type":"response.function_call_arguments.done","item_id":"item_f","arguments":"{\\"location\\":\\"Paris\\"}"}',
  "",
  'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"item_f","call_id":"call_1","name":"get_weather","arguments":"{\\"location\\":\\"Paris\\"}"}}',
  "",
  'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1","object":"response","output":[],"status":"completed"}}',
  "",
  "data: [DONE]",
  "",
].join("\n");

// Responses STREAM carrying an image_generation_call — the finished base64
// image only arrives on response.output_item.done; the stream parser used to
// drop the announced slot entirely.
const responsesImageStream = [
  'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_2"}}',
  "",
  'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"image_generation_call","id":"item_i","status":"in_progress"}}',
  "",
  'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","id":"item_i","status":"completed","result":"QUJD","output_format":"png"}}',
  "",
  'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_2","object":"response","output":[],"status":"completed"}}',
  "",
  "data: [DONE]",
  "",
].join("\n");

describe("parseLogResponse — tool calls", () => {
  describe("ordered items + meta", () => {
    it("keeps anthropic blocks in emission order (thinking, text, tool_use)", () => {
      const r = parseLogResponse(
        {
          id: "msg_9",
          type: "message",
          model: "claude-sonnet-4",
          stop_reason: "tool_use",
          content: [
            { type: "thinking", thinking: "hmm" },
            { type: "text", text: "first" },
            { type: "tool_use", id: "toolu_1", name: "a", input: {} },
            { type: "text", text: "second" },
            { type: "tool_use", id: "toolu_2", name: "b", input: {} },
          ],
        },
        "chat"
      );
      expect(r.items.map((i) => i.kind)).toEqual([
        "reasoning",
        "text",
        "tool_call",
        "text",
        "tool_call",
      ]);
      expect(r.meta).toMatchObject({
        id: "msg_9",
        model: "claude-sonnet-4",
        stopReason: "tool_use",
      });
    });

    it("keeps responses output order and meta", () => {
      const r = parseLogResponse(responsesToolCallBody, "chat");
      expect(r.items.map((i) => i.kind)).toEqual(["reasoning", "tool_call", "tool_result"]);
      expect(r.meta).toMatchObject({ id: "resp_1", status: "completed" });
    });

    it("keeps anthropic stream block order and stop reason", () => {
      const r = parseLogResponse(anthropicToolUseStream, "chat");
      expect(r.items.map((i) => i.kind)).toEqual(["text", "tool_call"]);
      expect(r.meta.id).toBe("msg_1");
      expect(r.meta.eventCount).toBeGreaterThan(0);
    });

    it("captures openai stream finish reason and event count", () => {
      const r = parseLogResponse(openaiChatToolCallStream, "chat");
      expect(r.meta.finishReason).toBe("tool_calls");
      expect(r.meta.id).toBe("c");
      expect(r.meta.eventCount).toBeGreaterThan(0);
      expect(r.items.map((i) => i.kind)).toEqual(["tool_call"]);
    });

    it("captures responses stream item order", () => {
      const r = parseLogResponse(responsesToolCallStream, "chat");
      expect(r.items.map((i) => i.kind)).toEqual(["tool_call"]);
      expect(r.meta.id).toBe("resp_1");
    });

    it("renders streamed image_generation_call items", () => {
      const r = parseLogResponse(responsesImageStream, "chat");
      expect(r.protocol).toBe("stream-responses");
      expect(r.hasData).toBe(true);
      expect(r.images).toEqual([{ b64Json: "QUJD", outputFormat: "png" }]);
      expect(r.items.map((i) => i.kind)).toEqual(["image"]);
    });
  });

  describe("/v1/chat/completions (OpenAI Chat)", () => {
    it("parses non-streaming tool calls", () => {
      const r = parseLogResponse(openaiChatToolCallBody, "chat");
      expect(r.protocol).toBe("openai-chat");
      expect(r.toolCalls).toHaveLength(1);
      expect(r.toolCalls[0]).toMatchObject({
        id: "call_abc",
        name: "get_weather",
        arguments: '{"location":"Paris"}',
      });
      expect(r.toolCalls[0]?.parsedArguments).toEqual({ location: "Paris" });
      // content is null -> empty string, but hasData stays true via tool calls
      expect(r.content).toBe("");
      expect(r.hasData).toBe(true);
    });

    it("reconstructs streaming tool calls (split across chunks)", () => {
      const r = parseLogResponse(openaiChatToolCallStream, "chat");
      expect(r.protocol).toBe("stream-openai");
      expect(r.toolCalls).toHaveLength(1);
      expect(r.toolCalls[0]?.name).toBe("get_weather");
      expect(r.toolCalls[0]?.id).toBe("call_abc");
      expect(r.toolCalls[0]?.arguments).toBe('{"location":"Paris"}');
      expect(r.toolCalls[0]?.parsedArguments).toEqual({ location: "Paris" });
    });
  });

  describe("/v1/messages (Anthropic)", () => {
    it("parses non-streaming tool_use blocks", () => {
      const r = parseLogResponse(anthropicToolUseBody, "chat");
      expect(r.protocol).toBe("anthropic");
      expect(r.content).toBe("I'll check the weather.");
      expect(r.toolCalls).toHaveLength(1);
      expect(r.toolCalls[0]).toMatchObject({
        id: "toolu_1",
        name: "get_weather",
        kind: "function",
      });
      expect(r.toolCalls[0]?.parsedArguments).toEqual({ location: "Paris" });
    });

    it("reconstructs streaming tool_use", () => {
      const r = parseLogResponse(anthropicToolUseStream, "chat");
      expect(r.protocol).toBe("stream-anthropic");
      expect(r.content).toBe("I'll check.");
      expect(r.toolCalls).toHaveLength(1);
      expect(r.toolCalls[0]?.name).toBe("get_weather");
      expect(r.toolCalls[0]?.id).toBe("toolu_1");
      expect(r.toolCalls[0]?.arguments).toBe('{"location":"Paris"}');
    });
  });

  describe("/v1/responses (OpenAI Responses)", () => {
    it("parses non-streaming function_call + function_call_output", () => {
      const r = parseLogResponse(responsesToolCallBody, "chat");
      expect(r.protocol).toBe("openai-responses");
      expect(r.reasoning).toBe("deciding to call a tool");
      expect(r.toolCalls).toHaveLength(1);
      expect(r.toolCalls[0]).toMatchObject({
        name: "get_weather",
        id: "item_f",
      });
      expect(r.toolCalls[0]?.arguments).toBe('{"location":"Paris"}');
      expect(r.toolResults).toHaveLength(1);
      expect(r.toolResults[0]?.callId).toBe("call_1");
      expect(r.toolResults[0]?.output).toBe('{"temp":20}');
    });

    it("reconstructs streaming function_call", () => {
      const r = parseLogResponse(responsesToolCallStream, "chat");
      expect(r.protocol).toBe("stream-responses");
      expect(r.toolCalls).toHaveLength(1);
      expect(r.toolCalls[0]?.name).toBe("get_weather");
      expect(r.toolCalls[0]?.arguments).toBe('{"location":"Paris"}');
    });
  });
});
