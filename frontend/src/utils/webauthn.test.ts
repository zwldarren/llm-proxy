import { afterEach, describe, expect, it, vi } from "vitest";
import i18n from "@/i18n";
import { createPasskey, getPasskey } from "@/utils/webauthn";

const BASE_OPTIONS = {
  challenge: "AAAA",
  user: { id: "AAAA", name: "operator", displayName: "operator" },
};

/** Install a fake WebAuthn container whose calls reject with ``error``. */
function stubWebauthn(error: unknown) {
  vi.stubGlobal("PublicKeyCredential", class {});
  vi.stubGlobal("navigator", {
    credentials: {
      create: vi.fn().mockRejectedValue(error),
      get: vi.fn().mockRejectedValue(error),
    },
  });
}

describe("passkey browser bridge", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("turns a dismissed registration prompt into a readable message", async () => {
    stubWebauthn(new DOMException("The operation either timed out", "NotAllowedError"));

    await expect(createPasskey(BASE_OPTIONS)).rejects.toThrow(
      i18n.global.t("auth.passkeyRegistrationCancelled")
    );
  });

  it("turns a dismissed sign-in prompt into a readable message", async () => {
    stubWebauthn(new DOMException("The operation either timed out", "NotAllowedError"));

    await expect(getPasskey(BASE_OPTIONS)).rejects.toThrow(
      i18n.global.t("auth.passkeySignInCancelled")
    );
  });

  it("translates the insecure-context rejection", async () => {
    stubWebauthn(new DOMException("Not allowed in this context", "SecurityError"));

    await expect(createPasskey(BASE_OPTIONS)).rejects.toThrow(
      i18n.global.t("auth.passkeyUnavailable")
    );
  });

  it("explains that a registered device cannot be registered twice", async () => {
    stubWebauthn(new DOMException("Credential already exists", "InvalidStateError"));

    await expect(createPasskey(BASE_OPTIONS)).rejects.toThrow(
      i18n.global.t("auth.passkeyAlreadyRegistered")
    );
  });

  it("reports unsupported browsers before prompting", async () => {
    vi.stubGlobal("PublicKeyCredential", undefined);

    await expect(createPasskey(BASE_OPTIONS)).rejects.toThrow(
      i18n.global.t("auth.passkeyUnavailable")
    );
  });
});
