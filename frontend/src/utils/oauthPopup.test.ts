import { describe, expect, it } from "vitest";
import { isMessageForProvider, isSuccessMessage, parseOAuthPopupMessage } from "./oauthPopup";

describe("parseOAuthPopupMessage", () => {
  it("accepts a payload carrying our message type", () => {
    const message = parseOAuthPopupMessage({
      type: "llm-proxy-provider-oauth",
      status: "success",
      provider: "chatgpt",
      message: "Connected",
    });
    expect(message).not.toBeNull();
    expect(message?.status).toBe("success");
    expect(message?.provider).toBe("chatgpt");
  });

  it("rejects foreign messages and non-objects", () => {
    expect(parseOAuthPopupMessage({ type: "something-else" })).toBeNull();
    expect(parseOAuthPopupMessage({ status: "success" })).toBeNull();
    expect(parseOAuthPopupMessage(null)).toBeNull();
    expect(parseOAuthPopupMessage(undefined)).toBeNull();
    expect(parseOAuthPopupMessage("llm-proxy-provider-oauth")).toBeNull();
    expect(parseOAuthPopupMessage(42)).toBeNull();
  });
});

describe("isMessageForProvider", () => {
  const message = (provider?: string | null) => ({
    type: "llm-proxy-provider-oauth",
    status: "success",
    provider,
  });

  it("matches only the provider the dialog is showing", () => {
    expect(isMessageForProvider(message("chatgpt"), "chatgpt")).toBe(true);
    expect(isMessageForProvider(message("chatgpt"), "other")).toBe(false);
  });

  it("ignores messages that name no provider", () => {
    // The callback page posts `provider: null` for setup failures; acting on
    // those would surface another provider's error in this dialog.
    expect(isMessageForProvider(message(null), "chatgpt")).toBe(false);
    expect(isMessageForProvider(message(undefined), "chatgpt")).toBe(false);
  });

  it("ignores messages when no provider is open", () => {
    expect(isMessageForProvider(message("chatgpt"), null)).toBe(false);
    expect(isMessageForProvider(message("chatgpt"), undefined)).toBe(false);
    expect(isMessageForProvider(null, "chatgpt")).toBe(false);
  });
});

describe("isSuccessMessage", () => {
  it("treats only `success` as completion", () => {
    expect(isSuccessMessage({ status: "success" })).toBe(true);
    expect(isSuccessMessage({ status: "error" })).toBe(false);
    expect(isSuccessMessage({})).toBe(false);
  });
});
