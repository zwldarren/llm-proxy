import { openaiAdapter } from "./openaiAdapter";
import { anthropicAdapter } from "./anthropicAdapter";
import { openResponsesAdapter } from "./openResponsesAdapter";
import type { ProtocolAdapter } from "./types";

/**
 * Registry of all available protocol adapters.
 * To add support for a new protocol, create an adapter file
 * and register it here.
 */
const adapters: ProtocolAdapter[] = [openaiAdapter, anthropicAdapter, openResponsesAdapter];

/**
 * Get the protocol adapter for a given endpoint path.
 * Falls back to OpenAI adapter for unknown endpoints.
 */
export function getAdapterForEndpoint(endpoint: string): ProtocolAdapter {
  // Match adapter ID against path segments to avoid substring false positives.
  // For multi-segment IDs (e.g., "chat/completions"), check segment-boundary match.
  const segments = endpoint.split("/").filter(Boolean);
  for (const adapter of adapters) {
    const idParts = adapter.id.split("/");
    if (idParts.length > 1) {
      // Multi-segment ID: check consecutive segment match
      if (
        idParts.length <= segments.length &&
        segments.slice(segments.length - idParts.length).join("/") === adapter.id
      ) {
        return adapter;
      }
    } else {
      // Single-segment ID: exact segment match
      if (segments.includes(adapter.id)) {
        return adapter;
      }
    }
  }
  // Default to OpenAI format
  return openaiAdapter;
}

export type { ProtocolAdapter } from "./types";
