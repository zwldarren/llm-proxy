/** Read the machine-readable code from any API error payload shape.
 *
 * Exception handlers wrap console errors as ``{error: {code}}``. A few
 * middleware paths return ``{code}`` directly, and legacy detail envelopes
 * carry ``{detail: {code}}``. Keeping this parser independent of HttpError
 * lets both the HTTP layer and user-facing error helpers share one contract.
 */
export function errorCodeFromPayload(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;

  if ("error" in data) {
    const error = data.error;
    if (error && typeof error === "object" && "code" in error && typeof error.code === "string") {
      return error.code;
    }
  }
  if ("code" in data && typeof data.code === "string") return data.code;
  if ("detail" in data && data.detail && typeof data.detail === "object") {
    const detail = data.detail;
    if (!Array.isArray(detail) && "code" in detail && typeof detail.code === "string") {
      return detail.code;
    }
  }
  return null;
}
