import { describe, expect, it } from "vitest";
import type { ApiKeyResponse } from "@/services/api/apiKeys";
import { useApiKeyReveal } from "./useApiKeyReveal";

const created = {
  name: "new key",
  key: "sk-plaintext",
  allowed_models: null,
  allowed_mcp_servers: null,
  created_at: "2026-11-12T00:00:00Z",
  expires_at: null,
  budget_usd: null,
  budget_period: null,
  budget_reset_day: null,
  rate_limit_rpm: null,
  message: "created",
} satisfies ApiKeyResponse;

describe("useApiKeyReveal", () => {
  it("discards the plaintext key when the dialog closes through either path", () => {
    const reveal = useApiKeyReveal();
    reveal.reveal(created);
    expect(reveal.createdKey.value).toEqual(created);

    reveal.open.value = false;

    expect(reveal.createdKey.value).toBeNull();
    expect(reveal.open.value).toBe(false);
  });

  it("discards the plaintext key through the explicit close action", () => {
    const reveal = useApiKeyReveal();
    reveal.reveal(created);

    reveal.close();

    expect(reveal.createdKey.value).toBeNull();
  });
});
