import type { ApiKeyRead } from "@/services/api/apiKeys";

type ApiKeyStatus = "active" | "disabled" | "expired";

export function isApiKeyExpired(expiresAt: string | null | undefined): boolean {
  return (
    expiresAt !== null && expiresAt !== undefined && new Date(expiresAt).getTime() <= Date.now()
  );
}

/** An explicitly disabled key reports "disabled" even when it is also past its expiry. */
export function getApiKeyStatus(key: Pick<ApiKeyRead, "is_active" | "expires_at">): ApiKeyStatus {
  if (!key.is_active) return "disabled";
  if (isApiKeyExpired(key.expires_at)) return "expired";
  return "active";
}
