import { HttpError, NetworkError } from "@/services/http";

/**
 * Extended error detail format from the backend
 */
interface ErrorDetail {
  message?: string;
  error_id?: string;
  type?: string;
  hint?: string;
  param?: string | null;
  code?: string | null;
}

/**
 * Pydantic validation error format
 */
interface ValidationErrorDetail {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

/**
 * Extract a user-friendly error message from various error formats.
 * Handles:
 * - NetworkError (offline, timeout)
 * - HttpError with structured error details (including error_id)
 * - Pydantic validation errors (array format)
 * - Simple string details
 * - Generic Error objects
 */
export const getErrorMessage = (e: unknown): string => {
  // Handle network/offline errors first
  if (e instanceof NetworkError) {
    return e.message;
  }

  if (e instanceof HttpError) {
    const data = e.data as {
      detail?: string | ErrorDetail | ValidationErrorDetail[];
      error?: ErrorDetail;
    };

    // Handle OpenAI-style error format with error object
    if (data?.error && typeof data.error === "object") {
      return formatErrorDetail(data.error);
    }

    if (data?.detail) {
      // Handle Pydantic validation errors (array format)
      if (Array.isArray(data.detail)) {
        return formatValidationErrors(data.detail);
      }

      // Handle structured error detail object
      if (typeof data.detail === "object") {
        return formatErrorDetail(data.detail as ErrorDetail);
      }

      // Handle simple string detail
      return String(data.detail);
    }
    return `${e.status}: ${e.statusText}`;
  }
  return e instanceof Error ? e.message : String(e);
};

/**
 * Format a structured error detail into a readable message.
 */
function formatErrorDetail(detail: ErrorDetail): string {
  const parts: string[] = [];

  // Main message
  if (detail.message) {
    parts.push(detail.message);
  }

  // Add error type if available and different from message
  if (detail.type && !detail.message?.includes(detail.type)) {
    parts.push(`[Type: ${detail.type}]`);
  }

  // Add error ID for debugging
  if (detail.error_id) {
    parts.push(`[Error ID: ${detail.error_id}]`);
  }

  // Add hint if available
  if (detail.hint) {
    parts.push(`\nHint: ${detail.hint}`);
  }

  // Add parameter info if available
  if (detail.param) {
    parts.push(`[Param: ${detail.param}]`);
  }

  return parts.join(" ") || "Unknown error";
}

/**
 * Format Pydantic validation errors into a readable message.
 */
function formatValidationErrors(errors: ValidationErrorDetail[]): string {
  return errors
    .map((err) => {
      const loc = err.loc || [];
      // Remove 'body' prefix if present to make error messages cleaner
      const parts = loc[0] === "body" ? loc.slice(1) : loc;
      const field = parts.join(".") || (err.loc ? "error" : "validation");
      return `${field}: ${err.msg || "validation error"}`;
    })
    .join("; ");
}
