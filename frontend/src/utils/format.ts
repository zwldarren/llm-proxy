// Intl.NumberFormat instance cache to avoid repeated instantiation in tight loops
const formatterCache = new Map<string, Intl.NumberFormat>();
function getFormatter(locale: string, options: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = `${locale}:${JSON.stringify(options)}`;
  let formatter = formatterCache.get(key);
  if (!formatter) {
    formatter = new Intl.NumberFormat(locale, options);
    formatterCache.set(key, formatter);
  }
  return formatter;
}

export const formatDate = (timestamp: number | string | null | undefined, locale?: string) => {
  if (!timestamp) return "-";

  let date: Date;
  if (typeof timestamp === "string") {
    // Try parsing as ISO 8601 date string first
    date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
      // If that fails, try as numeric string (Unix timestamp)
      const ts = Number(timestamp);
      if (Number.isNaN(ts)) return "-";
      date = new Date(ts * 1000);
    }
  } else {
    // Numeric timestamp (assumed to be Unix seconds)
    date = new Date(timestamp * 1000);
  }

  if (Number.isNaN(date.getTime())) return "-";
  try {
    return date.toLocaleString(locale);
  } catch {
    return date.toLocaleString();
  }
};

/**
 * Get status type for StatusBadge component.
 * Returns 'success' | 'warning' | 'error' | 'unknown'
 */
export const getStatusType = (
  status?: number | null
): "success" | "warning" | "error" | "unknown" => {
  if (!status) return "unknown";
  if (status >= 200 && status < 300) return "success";
  if (status >= 400 && status < 500) return "warning";
  if (status >= 500) return "error";
  return "unknown";
};

export const formatCost = (cost?: number | null, locale?: string) => {
  if (cost === undefined || cost === null) return "-";
  if (!locale) return `$${cost.toFixed(6)}`;
  try {
    return getFormatter(locale, {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(cost);
  } catch {
    return `$${cost.toFixed(6)}`;
  }
};

/**
 * Format latency in milliseconds.
 * Returns the raw value with 'ms' suffix or '-' if null/undefined.
 * Use this for displaying raw latency values (e.g., in audit logs).
 */

/**
 * Format duration in ms or seconds with appropriate precision.
 *
 * @example
 * formatDuration(500) -> "500ms"
 * formatDuration(1500) -> "1.5s"
 * formatDuration(null) -> "-"
 */
export const formatDuration = (ms: number | null | undefined): string => {
  if (ms === null || ms === undefined) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
};

export const formatTokens = (tokens?: number | null, locale?: string) => {
  if (tokens === undefined || tokens === null) return "-";
  if (!locale) return tokens.toLocaleString();
  try {
    return getFormatter(locale, {}).format(tokens);
  } catch {
    return tokens.toLocaleString();
  }
};

import type { LogListItem, LogRead } from "@/types/schemas";

export type LogItem = LogRead | LogListItem;

// Extract action from endpoint for audit logs (e.g., POST /admin/providers -> "Create Provider")
export const getActionFromEndpoint = (method: string, endpoint: string): string => {
  const parts = endpoint.split("/").filter(Boolean);
  const resource = parts[parts.length - 1] || parts[parts.length - 2] || "unknown";

  // Map HTTP methods to action verbs
  const actionMap: Record<string, string> = {
    GET: "View",
    POST: "Create",
    PUT: "Update",
    PATCH: "Update",
    DELETE: "Delete",
  };

  const action = actionMap[method] || method;
  // Capitalize resource name
  const formattedResource = resource.charAt(0).toUpperCase() + resource.slice(1).replace(/-/g, " ");

  return `${action} ${formattedResource}`;
};

/**
 * Format API key or session name to be more compact.
 */
export const formatApiKeyName = (name?: string | null): string => {
  if (!name) return "-";
  if (name.startsWith("session:")) {
    const keyPart = name.substring(8);
    if (keyPart.length > 12) {
      return `session:${keyPart.substring(0, 6)}...${keyPart.slice(-4)}`;
    }
  } else if (name.length > 24) {
    return `${name.substring(0, 12)}...${name.slice(-4)}`;
  }
  return name;
};

export const getActor = (log: LogItem): string => {
  const userId =
    log.user_identity || log.api_key_name || log.client_ip || (log.log_metadata?.user as string);
  return userId ? formatApiKeyName(userId) : "-";
};

/**
 * Format number with K, M, B suffixes for large values.
 */
export const formatNumberWithSuffix = (num: number, locale?: string): string => {
  // Use locale-aware compact notation when a locale is provided
  if (locale && num >= 1000) {
    try {
      return getFormatter(locale, {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(num);
    } catch {
      // fall through to default
    }
  }
  if (num >= 1000000000) return `${(num / 1000000000).toFixed(1)}B`;
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  if (!locale) return num.toLocaleString();
  try {
    return getFormatter(locale, {}).format(num);
  } catch {
    return num.toLocaleString();
  }
};

/**
 * Format a model context length compactly (e.g. 200000 -> "200K", 1000000 -> "1M").
 */
export const formatContextLength = (length?: number | null): string => {
  if (length === null || length === undefined || length <= 0) return "-";
  if (length >= 1_000_000) {
    const m = length / 1_000_000;
    return `${Number.isInteger(m) ? m : m.toFixed(1)}M`;
  }
  if (length >= 1_000) return `${Math.round(length / 1000)}K`;
  return String(length);
};

/**
 * Format percentage value.
 */
export const formatPercentage = (value: number, locale?: string): string => {
  if (!locale) return `${value.toFixed(1)}%`;
  try {
    return getFormatter(locale, {
      style: "percent",
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }).format(value / 100);
  } catch {
    return `${value.toFixed(1)}%`;
  }
};

/**
 * Format cost in USD with configurable precision.
 */
export const formatCostWithPrecision = (
  cost: number,
  precision: number = 3,
  locale?: string
): string => {
  if (!locale) return `$${cost.toFixed(precision)}`;
  try {
    return getFormatter(locale, {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: precision,
      maximumFractionDigits: precision,
    }).format(cost);
  } catch {
    return `$${cost.toFixed(precision)}`;
  }
};
