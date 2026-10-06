import { http, handleUnauthorized, TimeoutError } from "@/services/http";
import { getAdapterForEndpoint } from "@/adapters";
import type { WebSearchResultPayload } from "@/adapters/types";

const BASE_URL = "/v1";

/**
 * How long the server may take to START responding (time to first byte).
 *
 * Only the connection is bounded, never the stream: a web-search turn performs
 * several upstream calls before it produces text, so an overall deadline would
 * cut long (but healthy) generations. See `streamChatCompletion`.
 */
const FIRST_BYTE_TIMEOUT_MS = 30_000;

export const chatApi = {
  /**
   * Streams chat completion from the API.
   * Uses protocol adapters to handle different response formats
   * (OpenAI, Anthropic, OpenResponses, etc.).
   * @param signal - Optional AbortSignal to cancel the request
   */
  streamChatCompletion: async (
    endpoint: string,
    data: Record<string, unknown>,
    apiKey: string,
    onChunk: (chunk: string) => void,
    onError?: (error: string) => void,
    onReasoningChunk?: (chunk: string) => void,
    onToolCall?: (index: number, id: string, name: string, args: string) => void,
    onWebSearchCall?: (
      index: number,
      id: string,
      query: string,
      status: "in_progress" | "completed" | "failed",
      payload?: WebSearchResultPayload
    ) => void,
    signal?: AbortSignal
  ) => {
    // Strip leading /v1 if present in endpoint
    const relativePath = endpoint.startsWith("/v1") ? endpoint.slice(3) : endpoint;

    // Bound only the wait for the first byte. `AbortSignal.timeout` must NOT
    // be used here: it stays attached to the response body for the lifetime of
    // the request, so it aborts the stream itself once the deadline passes
    // (verified: a reader throws `TimeoutError` mid-stream). Long web-search
    // turns routinely outlive a fixed deadline while streaming perfectly well.
    const firstByteController = new AbortController();
    const firstByteTimer = setTimeout(() => {
      firstByteController.abort(
        new DOMException("Timed out waiting for the response to start", "TimeoutError")
      );
    }, FIRST_BYTE_TIMEOUT_MS);

    // Combine the first-byte deadline with any caller-provided signal.
    const combinedSignal = signal
      ? AbortSignal.any([signal, firstByteController.signal])
      : firstByteController.signal;

    let response: Response;
    try {
      response = await fetch(`${BASE_URL}${relativePath}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({ ...data, stream: true }),
        signal: combinedSignal,
      });
    } catch (e) {
      if (e instanceof DOMException && e.name === "TimeoutError") {
        throw new TimeoutError(
          "Chat request timed out. The server took too long to respond. Please try again."
        );
      }
      throw e;
    } finally {
      // The response headers arrived (or the attempt failed): disarm the
      // first-byte deadline so it can never interrupt the streaming body.
      clearTimeout(firstByteTimer);
    }

    if (!response.ok) {
      if (response.status === 401) {
        handleUnauthorized();
      }
      let errorText = "";
      try {
        errorText = await response.text();
      } catch {
        // ignore
      }
      throw new Error(
        `HTTP error! status: ${response.status} ${errorText ? `detail: ${errorText}` : ""}`
      );
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) return;

    let buffer = "";
    let currentEvent = "";

    // Get the protocol adapter for this endpoint
    const adapter = getAdapterForEndpoint(endpoint);
    const callbacks = { onChunk, onReasoningChunk, onToolCall, onError, onWebSearchCall };

    // Read the stream
    while (true) {
      // Check if the request has been aborted
      if (signal?.aborted) {
        reader.cancel();
        throw new Error("Request aborted");
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep the last line in the buffer because it might be incomplete
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        if (trimmed.startsWith("event: ")) {
          currentEvent = trimmed.slice(7).trim();
          continue;
        }

        if (trimmed.startsWith("data: ")) {
          const dataStr = trimmed.slice(6).trim();
          if (dataStr === "[DONE]") {
            currentEvent = "";
            continue;
          }

          try {
            const parsedData = JSON.parse(dataStr) as Record<string, unknown>;

            // Dispatch to the protocol adapter for parsing
            adapter.parseStreamChunk(parsedData, currentEvent, callbacks);
          } catch (e) {
            console.warn("Failed to parse SSE line:", dataStr, e);
          }

          currentEvent = "";
        }
      }
    }
  },

  getModels: (apiKey: string) =>
    http.get<{ data: { id: string; provider: string }[] }>(`${BASE_URL}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    }),
};
