import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import { HttpError } from "@/services/http";
import { apiErrorCode, getErrorMessage } from "@/utils/error";

/** Build the console error envelope the backend actually sends. */
function httpError(code: string | null, message: string, status = 400): HttpError {
  return new HttpError(status, "Bad Request", {
    error: {
      message,
      type: "invalid_request_error",
      ...(code === null ? {} : { code }),
      error_id: "req-1",
    },
  });
}

describe("getErrorMessage", () => {
  it("translates a mapped backend code instead of echoing the English message", () => {
    const error = httpError("last_factor", "This is the last sign-in factor");

    expect(getErrorMessage(error)).toBe(i18n.global.t("errors.codes.lastFactor"));
  });

  it("reads the code from a bare detail envelope too", () => {
    const error = new HttpError(400, "Bad Request", {
      code: "email_verification_unavailable",
      message: "Email verification requires HTTPS and SMTP",
    });

    expect(apiErrorCode(error)).toBe("email_verification_unavailable");
    expect(getErrorMessage(error)).toBe(i18n.global.t("errors.codes.emailVerificationUnavailable"));
  });

  it("falls back to the backend message for an unknown code", () => {
    const error = httpError("some_unmapped_code", "Something specific went wrong");

    expect(getErrorMessage(error)).toBe("Something specific went wrong");
  });

  it("keeps the message readable: no error type or request id in the toast", () => {
    const message = getErrorMessage(httpError(null, "Provider not found"));

    expect(message).toBe("Provider not found");
    expect(message).not.toContain("[Type:");
    expect(message).not.toContain("[Error ID:");
  });

  it("appends a hint on its own line when the backend supplies one", () => {
    const error = new HttpError(400, "Bad Request", {
      error: { message: "Invalid value", hint: "Use one of the listed models" },
    });

    expect(getErrorMessage(error)).toBe("Invalid value\nUse one of the listed models");
  });

  it("shows Pydantic validation text without field paths or Value error prefixes", () => {
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: [
        {
          loc: ["body", "email"],
          msg: "Value error, Enter a valid email address",
          type: "value_error",
        },
      ],
    });

    expect(getErrorMessage(error)).toBe("Enter a valid email address");
  });
});
