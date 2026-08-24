import { http, handleUnauthorized, TimeoutError } from "@/services/http";
import { getAdapterForEndpoint } from "@/adapters";

const BASE_URL = "/v1";

/** Default timeout for streaming chat requests (30 seconds to first byte). */
const STREAM_TIMEOUT_MS = 30_000;

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
      status: "in_progress" | "completed" | "failed"
    ) => void,
    signal?: AbortSignal
  ) => {
    // Strip leading /v1 if present in endpoint
    const relativePath = endpoint.startsWith("/v1") ? endpoint.slice(3) : endpoint;

    // Timeout signal ensures the connection attempt cannot hang forever.
    // Once the stream is established, individual reads may take longer.
    const timeoutSignal = AbortSignal.timeout(STREAM_TIMEOUT_MS);

    // Combine timeout signal with any caller-provided signal
    const combinedSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;

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
