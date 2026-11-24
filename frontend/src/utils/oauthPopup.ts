/**
 * Contract for the `postMessage` the OAuth callback page sends to the dashboard.
 *
 * Kept out of the dialog so the filtering rules — which messages may act on
 * which provider — are testable without mounting the sheet.
 */
import { OAUTH_CALLBACK_MESSAGE_TYPE } from "@/constants/providers";

export interface OAuthPopupMessage {
  type?: string;
  status?: string;
  provider?: string | null;
  message?: string;
}

/** Parse a message payload, or `null` when it is not one of ours. */
export function parseOAuthPopupMessage(data: unknown): OAuthPopupMessage | null {
  if (typeof data !== "object" || data === null) return null;
  const message = data as OAuthPopupMessage;
  return message.type === OAUTH_CALLBACK_MESSAGE_TYPE ? message : null;
}

/**
 * True when `message` is about `providerName`.
 *
 * The callback page posts `provider: null` for setup failures (missing or
 * expired state), and those cannot be attributed to a provider: acting on them
 * would surface one sign-in's error next to another provider's row, because the
 * dialog is a single reused instance.
 */
export function isMessageForProvider(
  message: OAuthPopupMessage | null,
  providerName: string | null | undefined
): boolean {
  if (!message || !providerName) return false;
  return message.provider === providerName;
}

/** True when the message reports a completed connection. */
export function isSuccessMessage(message: OAuthPopupMessage): boolean {
  return message.status === "success";
}
