import i18n from "@/i18n";

/**
 * Password strength policy — mirrors the backend
 * `llm_proxy.security.passwords.validate_password_strength`.
 *
 * 8-72 characters with at least one uppercase letter, one lowercase letter,
 * one digit, and one special character.
 */

const PASSWORD_MIN_LENGTH = 8;
const PASSWORD_MAX_LENGTH = 72;
const RE_UPPERCASE = /[A-Z]/;
const RE_LOWERCASE = /[a-z]/;
const RE_DIGIT = /[0-9]/;
const RE_SPECIAL = /[!@#$%^&*(),.?":{}|<>_~`\-+=;'\\[\]/]/;

/** A concise, human-readable summary of the policy (i18n). */
export function passwordRequirementsText(): string {
  return i18n.global.t("auth.passwordRequirements");
}

/**
 * Validate *password* against the strength policy.
 *
 * Returns a localized error message describing the first failing rule, or
 * an empty string when the password satisfies all requirements. Empty input
 * is treated as valid (presence is enforced by `required` on the input).
 */
export function validatePasswordStrength(password: string): string {
  if (!password) return "";
  if (password.length < PASSWORD_MIN_LENGTH) {
    return i18n.global.t("auth.passwordTooShort");
  }
  if (password.length > PASSWORD_MAX_LENGTH) {
    return i18n.global.t("auth.passwordWeak");
  }
  if (!RE_UPPERCASE.test(password)) {
    return i18n.global.t("auth.passwordWeak");
  }
  if (!RE_LOWERCASE.test(password)) {
    return i18n.global.t("auth.passwordWeak");
  }
  if (!RE_DIGIT.test(password)) {
    return i18n.global.t("auth.passwordWeak");
  }
  if (!RE_SPECIAL.test(password)) {
    return i18n.global.t("auth.passwordWeak");
  }
  return "";
}
