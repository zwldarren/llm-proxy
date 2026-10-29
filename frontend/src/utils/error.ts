import i18n from "@/i18n";
import { HttpError, NetworkError } from "@/services/http";
import { errorCodeFromPayload } from "@/utils/apiError";

/**
 * Backend error codes that deserve an actionable, localized message.
 *
 * The API returns English ``message`` text built for logs and API consumers;
 * a console user reading a toast is better served by a sentence that says what
 * it means for them. Codes without an entry keep the backend message.
 */
const CODE_MESSAGES: Record<string, string> = {
  account_locked: "errors.codes.accountLocked",
  email_delivery_failed: "errors.codes.emailDeliveryFailed",
  email_verification_invalid: "errors.codes.emailVerificationInvalid",
  email_verification_superseded: "errors.codes.emailVerificationSuperseded",
  email_verification_unavailable: "errors.codes.emailVerificationUnavailable",
  invalid_challenge: "errors.codes.invalidChallenge",
  invalid_mfa_code: "errors.codes.invalidMfaCode",
  invitation_invalid: "errors.codes.invitationInvalid",
  last_factor: "errors.codes.lastFactor",
  mfa_challenge_expired: "errors.codes.mfaChallengeExpired",
  mfa_enrollment_required: "errors.codes.mfaEnrollmentRequired",
  mfa_not_enabled: "errors.codes.mfaNotEnabled",
  mfa_unavailable: "errors.codes.mfaUnavailable",
  invalid_current_password: "profile.wrongPassword",
  no_passkey: "errors.codes.noPasskey",
  oauth_complete_expired: "errors.codes.oauthCompleteExpired",
  oauth_link_expired: "errors.codes.oauthLinkExpired",
  oauth_link_mismatch: "errors.codes.oauthLinkFailed",
  oauth_link_session_changed: "errors.codes.oauthLinkFailed",
  oauth_onboarding_only: "errors.codes.oauthOnboardingOnly",
  oauth_unavailable: "errors.codes.oauthUnavailable",
  oidc_discovery_failed: "errors.codes.oidcUnreachable",
  oidc_email_missing: "errors.codes.oidcEmailUnverified",
  oidc_email_unverified: "errors.codes.oidcEmailUnverified",
  oidc_id_token_invalid: "errors.codes.oidcRejected",
  oidc_insecure_issuer: "errors.codes.oidcInsecureIssuer",
  oidc_issuer_mismatch: "errors.codes.oidcIssuerMismatch",
  oidc_token_exchange_failed: "errors.codes.oidcRejected",
  oidc_userinfo_failed: "errors.codes.oidcUnreachable",
  passkey_challenge_expired: "errors.codes.passkeyChallengeExpired",
  passkey_not_recognized: "errors.codes.passkeyNotRecognized",
  passkey_unavailable: "errors.codes.passkeyUnavailable",
  passkey_verification_failed: "errors.codes.passkeyVerificationFailed",
  password_reset_invalid: "errors.codes.passwordResetInvalid",
  password_reset_superseded: "errors.codes.passwordResetSuperseded",
  setup_complete: "errors.codes.setupComplete",
  step_up_required: "errors.codes.stepUpRequired",
  too_many_auth_failures: "errors.codes.tooManyAuthFailures",
  username_required: "errors.codes.usernameRequired",
};

/** Read the machine-readable code from an HttpError. */
export function apiErrorCode(e: unknown): string | null {
  if (!(e instanceof HttpError)) return null;
  return errorCodeFromPayload(e.data);
}

/** Localized message for a mapped code, or null when the code is unknown. */
function localizedCodeMessage(e: unknown): string | null {
  const code = apiErrorCode(e);
  if (!code) return null;
  const key = CODE_MESSAGES[code];
  return key ? i18n.global.t(key) : null;
}

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

  // A mapped backend code gives the console a message in the user's language.
  const localized = localizedCodeMessage(e);
  if (localized) return localized;

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
 *
 * Toasts carry the message and, when the backend supplied one, its hint. Error
 * type and request id are support metadata, not something to read: they are
 * logged to the console instead of appended to every notification.
 */
function formatErrorDetail(detail: ErrorDetail): string {
  const parts: string[] = [];

  // Main message
  if (detail.message) {
    parts.push(detail.message);
  }

  if (detail.error_id) {
    console.debug("API error", { errorId: detail.error_id, type: detail.type });
  }

  // Add hint if available
  if (detail.hint) {
    parts.push(detail.hint);
  }

  return parts.join("\n") || "Unknown error";
}

/**
 * Format Pydantic validation errors into a readable message.
 */
function formatValidationErrors(errors: ValidationErrorDetail[]): string {
  const messages = errors.map((err) => {
    // Pydantic prefixes ValueError validators with ``Value error,``. The
    // field path is useful in API logs, but in the console it adds noise to
    // sentences that already say what the user must change.
    return (err.msg || "validation error").replace(/^Value error,\s*/, "");
  });
  return [...new Set(messages)].join("; ");
}
