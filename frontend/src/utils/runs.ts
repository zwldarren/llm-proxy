/** Shared formatting helpers for playground run specimens. */

/** Unique run id: `run_<timestamp>_<rand7>`. Shared by Chat and Images runs. */
export function makeRunId(): string {
  return `run_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/** Append a run keeping the tray bounded to `max` (newest last). */
export function appendRun<T extends { id: string }>(runs: T[], run: T, max: number): T[] {
  return [...runs, run].slice(-max);
}

/** 324 ms under a second, 1.24 s above. */
export function formatLatency(ms: number | undefined): string {
  if (ms === undefined) return "…";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/** Wall-clock time as HH:MM:SS (24h) for run specimens. */
export function formatClock(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString(undefined, { hour12: false });
}

/** Short endpoint label for tight chips: "/v1/chat/completions" → "chat/completions". */
export function endpointName(endpoint: string): string {
  return endpoint.replace(/^\/v1\//, "");
}
