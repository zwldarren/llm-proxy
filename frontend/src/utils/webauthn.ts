/** Browser WebAuthn helpers: JSON options -> ArrayBuffer, credential -> JSON. */

import i18n from "@/i18n";
import { base64UrlToBuffer, bytesToBase64Url } from "@/utils/base64url";

function descriptor(value: { id: string; transports?: string[] | null }) {
  return {
    id: base64UrlToBuffer(value.id),
    type: "public-key" as const,
    ...(value.transports ? { transports: value.transports as AuthenticatorTransport[] } : {}),
  };
}

/**
 * The browser rejects WebAuthn calls with DOMExceptions whose text is written
 * for developers ("The operation either timed out or was not allowed…").
 * Translate the ones a user can actually cause; every caller renders whatever
 * this throws, so translating here covers all of them.
 */
function credentialError(e: unknown, cancelledKey: string): unknown {
  if (!(e instanceof DOMException)) return e;
  if (e.name === "NotAllowedError" || e.name === "AbortError") {
    return new Error(i18n.global.t(cancelledKey), { cause: e });
  }
  if (e.name === "SecurityError") {
    return new Error(i18n.global.t("auth.passkeyUnavailable"), { cause: e });
  }
  return e;
}

function credentialsContainer(): CredentialsContainer {
  if (typeof window === "undefined" || !window.PublicKeyCredential || !navigator.credentials) {
    throw new Error(i18n.global.t("auth.passkeyUnavailable"));
  }
  return navigator.credentials;
}

function creationOptionsFromJson(
  options: Record<string, unknown>
): PublicKeyCredentialCreationOptions {
  const user = options.user as { id: string; name: string; displayName: string };
  const exclude =
    (options.excludeCredentials as { id: string; transports?: string[] | null }[] | undefined) ||
    [];
  return {
    ...(options as unknown as PublicKeyCredentialCreationOptions),
    challenge: base64UrlToBuffer(String(options.challenge)),
    user: { ...user, id: base64UrlToBuffer(user.id) },
    excludeCredentials: exclude.map(descriptor),
  };
}

function requestOptionsFromJson(
  options: Record<string, unknown>
): PublicKeyCredentialRequestOptions {
  const allow =
    (options.allowCredentials as { id: string; transports?: string[] | null }[] | undefined) || [];
  return {
    ...(options as unknown as PublicKeyCredentialRequestOptions),
    challenge: base64UrlToBuffer(String(options.challenge)),
    allowCredentials: allow.map(descriptor),
  };
}

export async function createPasskey(
  options: Record<string, unknown>
): Promise<Record<string, unknown>> {
  let credential: PublicKeyCredential | null;
  try {
    credential = (await credentialsContainer().create({
      publicKey: creationOptionsFromJson(options),
    })) as PublicKeyCredential | null;
  } catch (e) {
    // The backend sends ``excludeCredentials``, so a device that is already
    // registered rejects registration instead of silently duplicating it.
    if (e instanceof DOMException && e.name === "InvalidStateError") {
      throw new Error(i18n.global.t("auth.passkeyAlreadyRegistered"), { cause: e });
    }
    throw credentialError(e, "auth.passkeyRegistrationCancelled");
  }
  if (!credential) {
    throw new Error(i18n.global.t("auth.passkeyRegistrationCancelled"));
  }
  const response = credential.response as AuthenticatorAttestationResponse;
  return {
    id: credential.id,
    rawId: bytesToBase64Url(credential.rawId),
    type: credential.type,
    response: {
      clientDataJSON: bytesToBase64Url(response.clientDataJSON),
      attestationObject: bytesToBase64Url(response.attestationObject),
      transports: response.getTransports?.() ?? [],
    },
    clientExtensionResults: credential.getClientExtensionResults(),
  };
}

export async function getPasskey(
  options: Record<string, unknown>
): Promise<Record<string, unknown>> {
  let credential: PublicKeyCredential | null;
  try {
    credential = (await credentialsContainer().get({
      publicKey: requestOptionsFromJson(options),
    })) as PublicKeyCredential | null;
  } catch (e) {
    throw credentialError(e, "auth.passkeySignInCancelled");
  }
  if (!credential) {
    throw new Error(i18n.global.t("auth.passkeySignInCancelled"));
  }
  const response = credential.response as AuthenticatorAssertionResponse;
  return {
    id: credential.id,
    rawId: bytesToBase64Url(credential.rawId),
    type: credential.type,
    response: {
      clientDataJSON: bytesToBase64Url(response.clientDataJSON),
      authenticatorData: bytesToBase64Url(response.authenticatorData),
      signature: bytesToBase64Url(response.signature),
      userHandle: response.userHandle ? bytesToBase64Url(response.userHandle) : null,
    },
    clientExtensionResults: credential.getClientExtensionResults(),
  };
}
