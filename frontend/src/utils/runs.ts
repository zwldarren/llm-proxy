/** Shared formatting helpers for playground run telemetry. */

/** Unique run id: `run_<timestamp>_<rand7>`. */
export function makeRunId(): string {
  return `run_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
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
