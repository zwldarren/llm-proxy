/**
 * Validation utility functions for forms
 */

interface ValidationResult {
  valid: boolean;
  error?: string;
}

/**
 * Internal helper to parse JSON and return result or error
 */
function tryParseJson(
  jsonString: string
): { success: true; value: unknown } | { success: false; error: string } {
  if (!jsonString?.trim()) {
    return { success: true, value: null }; // Empty string is considered valid null
  }

  try {
    const parsed = JSON.parse(jsonString);
    return { success: true, value: parsed };
  } catch {
    return { success: false, error: "Invalid JSON format" };
  }
}

/**
 * Parses a JSON string safely, returning null on failure
 * @param jsonString - The JSON string to parse
 * @param defaultValue - Default value to return on parse failure
 * @returns Parsed object or default value
 */
export function safeParseJson<T = Record<string, unknown>>(
  jsonString: string,
  defaultValue: T | null = null
): T | null {
  const result = tryParseJson(jsonString);
  if (!result.success) {
    return defaultValue;
  }
  // If empty string resulted in null, return defaultValue
  if (result.value === null && !jsonString?.trim()) {
    return defaultValue;
  }
  return result.value as T;
}

/**
 * Validates that a JSON string is either empty or a valid object
 * @param jsonString - The JSON string to validate
 * @returns ValidationResult with valid status and optional error message
 */
export function validateJsonObject(jsonString: string): ValidationResult {
  if (!jsonString?.trim() || jsonString.trim() === "{}") {
    return { valid: true };
  }

  const result = tryParseJson(jsonString);
  if (!result.success) {
    return { valid: false, error: result.error };
  }

  const parsed = result.value;
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return {
      valid: false,
      error: "Must be a valid JSON object",
    };
  }
  return { valid: true };
}
