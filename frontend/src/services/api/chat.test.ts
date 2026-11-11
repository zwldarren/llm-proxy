import { afterEach, describe, expect, it, vi } from "vitest";
import { chatApi } from "./chat";

/**
 * Regression tests for the streaming request deadline.
 *
 * The deadline must bound only the wait for the server to start responding.
 * It previously used `AbortSignal.timeout`, which stays attached to the
 * response body and therefore aborts the stream itself once the deadline
 * passes — cutting every long generation (multi-round web search, long
 * reasoning) at the 30s mark.
 */
describe("chatApi.streamChatCompletion deadlines", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("keeps streaming past the first-byte deadline once the response has arrived", async () => {
    vi.useFakeTimers();

    const encoder = new TextEncoder();
    let streamController!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller;
      },
    });

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(body, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        })
      )
    );

    const chunks: string[] = [];
    const done = chatApi.streamChatCompletion(
      "/v1/chat/completions",
      { model: "m", messages: [] },
      (chunk) => chunks.push(chunk)
    );

    // Let `fetch` resolve and the body reader attach.
    await vi.advanceTimersByTimeAsync(0);

    streamController.enqueue(
      encoder.encode('data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
    );

    // Far beyond the 30s deadline: the body must still be readable.
    await vi.advanceTimersByTimeAsync(120_000);

    streamController.enqueue(
      encoder.encode('data: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n\n')
    );
    streamController.close();
    await done;

    expect(chunks.join("")).toBe("hello world");
  });

  it("parses comments, fragmented CRLF, and multiline data events", async () => {
    const encoder = new TextEncoder();
    let streamController!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller;
      },
    });

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(body, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        })
      )
    );

    const chunks: string[] = [];
    const done = chatApi.streamChatCompletion(
      "/v1/chat/completions",
      { model: "m", messages: [] },
      (chunk) => chunks.push(chunk)
    );

    for (const wireChunk of [
      ": ping\r\n\r\n",
      'data: {"choices":[{"delta":{"content":"hel',
      'lo"}}]}\r\n\r\n',
      'data: {"choices":[\r\n',
      'data: {"delta":{"content":" world"}}]}\r\n\r\ndata: [DONE]\r\n\r\n',
    ]) {
      streamController.enqueue(encoder.encode(wireChunk));
    }
    streamController.close();
    await done;

    expect(chunks.join("")).toBe("hello world");
  });

  it("throws TimeoutError when the server never starts responding", async () => {
    vi.useFakeTimers();

    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            const signal = init.signal as AbortSignal;
            signal.addEventListener("abort", () => reject(signal.reason));
          })
      )
    );

    const done = chatApi.streamChatCompletion(
      "/v1/chat/completions",
      { model: "m", messages: [] },
      () => {}
    );
    const rejected = expect(done).rejects.toMatchObject({ name: "TimeoutError" });

    await vi.advanceTimersByTimeAsync(30_000);

    await rejected;
  });
});
