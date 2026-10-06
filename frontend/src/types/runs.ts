import type { ImageData } from "@/types/schemas";

/**
 * A single API run through a playground page — the unit the Images canvas and
 * run inspector render. Emitted when a request starts (status "streaming")
 * and again when it settles (ok / error). Session-scoped, in-memory only:
 * payloads can be large.
 */

export type RunStatus = "streaming" | "ok" | "error" | "stopped";

/** An image generations/edits run. Payloads are always retained in-session. */
export interface ImageRun {
  id: string;
  model: string;
  /** The exact request payload sent to the proxy. */
  payload: Record<string, unknown> | null;
  /** Wall-clock start (Date.now()) for display. */
  startedAt: number;
  status: RunStatus;
  /** Set when the run settles. */
  latencyMs?: number;
  errorMessage?: string;
  mode: "generations" | "edits";
  prompt: string;
  n: number;
  size: string;
  quality: string;
  images: ImageData[];
}
