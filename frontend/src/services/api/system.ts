import { http } from "../http";

const BASE_URL = "/api/system";

/** Instance version and update-check state, as reported by the backend. */
export interface SystemInfo {
  version: string;
  update_check_enabled: boolean;
  latest_version: string | null;
  update_available: boolean;
  /** ISO datetime of the last upstream check, or null when never checked. */
  checked_at: string | null;
  /** True when the last upstream check could not be performed. */
  check_failed: boolean;
}

export const systemApi = {
  // Pass force=true to request a fresh upstream check; the backend enforces a
  // cooldown between forced checks, so only call this from the manual button.
  getSystemInfo: (force = false) =>
    http.get<SystemInfo>(`${BASE_URL}/info${force ? "?force=true" : ""}`),
};
