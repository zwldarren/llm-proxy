import { describe, expect, it } from "vitest";
import {
  OAUTH_CALLBACK_PATH,
  OAUTH_LOOPBACK_REDIRECT_URI,
  defaultOAuthRedirectUri,
  isLoopbackHost,
} from "./providers";

describe("isLoopbackHost", () => {
  it("recognizes every loopback spelling the browser reports", () => {
    expect(isLoopbackHost("localhost")).toBe(true);
    expect(isLoopbackHost("LOCALHOST")).toBe(true);
    expect(isLoopbackHost("127.0.0.1")).toBe(true);
    expect(isLoopbackHost("::1")).toBe(true);
    expect(isLoopbackHost("[::1]")).toBe(true);
    expect(isLoopbackHost(" localhost ")).toBe(true);
  });

  it("rejects everything else", () => {
    expect(isLoopbackHost("proxy.example.com")).toBe(false);
    expect(isLoopbackHost("10.0.0.5")).toBe(false);
    expect(isLoopbackHost("")).toBe(false);
  });
});

describe("defaultOAuthRedirectUri", () => {
  it("uses the console's own callback route on a loopback host", () => {
    expect(
      defaultOAuthRedirectUri({ origin: "http://localhost:9911", hostname: "localhost" })
    ).toBe(`http://localhost:9911${OAUTH_CALLBACK_PATH}`);
    expect(
      defaultOAuthRedirectUri({ origin: "http://127.0.0.1:9911", hostname: "127.0.0.1" })
    ).toBe(`http://127.0.0.1:9911${OAUTH_CALLBACK_PATH}`);
  });

  it("falls back to the loopback URI OpenAI accepts on a remote host", () => {
    expect(
      defaultOAuthRedirectUri({
        origin: "https://proxy.example.com",
        hostname: "proxy.example.com",
      })
    ).toBe(OAUTH_LOOPBACK_REDIRECT_URI);
  });

  it("reads the console's own location when called without arguments", () => {
    expect(defaultOAuthRedirectUri()).toBe(`${window.location.origin}${OAUTH_CALLBACK_PATH}`);
  });
});
