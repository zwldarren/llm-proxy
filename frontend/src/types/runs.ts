import type { ImageData } from "@/types/schemas";

/**
 * A single API run through a playground — the unit shown in the run tray.
 * Emitted when a request starts (status "streaming") and again when it settles
 * (ok / error / stopped). Session-scoped: payloads can be large, so runs live
 * in the pinia store, never in localStorage.
 *
 * Chat and Image runs share a telemetry base (RunBase) and diverge only in the
 * fields that describe the request shape (endpoint vs. mode + params) and the
 * status set they can reach.
 */

export type ChatRunStatus = "streaming" | "ok" | "error" | "stopped";
type ImageRunStatus = "streaming" | "ok" | "error";

/** Common run telemetry — shared by ChatRun and ImageRun. */
interface RunBase<S extends string> {
  id: string;
  model: string;
  /** The exact request payload sent to the proxy. Null on runs restored from
   * a previous session — stubs persist telemetry only, never payloads. */
  payload: Record<string, unknown> | null;
  /** Wall-clock start (Date.now()) for display. */
  startedAt: number;
  status: S;
  /** Set when the run settles. */
  latencyMs?: number;
  errorMessage?: string;
}

/** A chat-completions run. */
export interface ChatRun extends RunBase<ChatRunStatus> {
  endpoint: string;
  /** Streamed response size in characters, for orientation. */
  responseChars?: number;
}

/** An image generations/edits run. Payloads are always retained in-session. */
export interface ImageRun extends RunBase<ImageRunStatus> {
  mode: "generations" | "edits";
  prompt: string;
  n: number;
  size: string;
  quality: string;
  images: ImageData[];
}
