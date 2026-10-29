import { ref } from "vue";
import { useI18n } from "vue-i18n";
import { meApi } from "@/services/api/me";
import { getErrorMessage } from "@/utils/error";
import { passwordRequirementsText, validatePasswordStrength } from "@/utils/password";

interface ChangePasswordOptions {
  /**
   * Whether the account has a password to prove before setting a new one.
   * Passwordless (passkey/OIDC) accounts cannot: the backend accepts a null
   * current password and asks for a recent step-up instead. Defaults to true.
   */
  requireCurrentPassword?: () => boolean;
}

/**
 * Shared state and submit flow for the change-password form.
 *
 * Used by the account-security password panel and the forced-password-change
 * screen so validation, error mapping, and the API call are defined once. The
 * caller decides what happens on success (end the session and redirect); the
 * form only reports whether the change landed.
 */
export function useChangePassword(options: ChangePasswordOptions = {}) {
  const { t } = useI18n();

  const currentPassword = ref("");
  const newPassword = ref("");
  const confirmPassword = ref("");
  const error = ref("");
  const loading = ref(false);
  const passwordHint = passwordRequirementsText();

  const requiresCurrentPassword = () => options.requireCurrentPassword?.() ?? true;

  function reset() {
    currentPassword.value = "";
    newPassword.value = "";
    confirmPassword.value = "";
    error.value = "";
    loading.value = false;
  }

  async function submit(): Promise<boolean> {
    if (loading.value) return false;

    // Resolve once: a passwordless account can gain a password while the form
    // is open, and validation and submission must agree on which mode ran.
    const needsCurrentPassword = requiresCurrentPassword();

    if (needsCurrentPassword && !currentPassword.value) {
      error.value = t("profile.passwordRequired");
      return false;
    }
    if (newPassword.value.length < 8) {
      error.value = t("auth.passwordTooShort");
      return false;
    }
    if (newPassword.value !== confirmPassword.value) {
      error.value = t("profile.passwordMismatch");
      return false;
    }
    const strengthError = validatePasswordStrength(newPassword.value);
    if (strengthError) {
      error.value = strengthError;
      return false;
    }

    error.value = "";
    loading.value = true;
    try {
      await meApi.changePassword(
        needsCurrentPassword ? currentPassword.value : null,
        newPassword.value
      );
      return true;
    } catch (e: unknown) {
      error.value = getErrorMessage(e) || t("errors.saveFailed");
      return false;
    } finally {
      loading.value = false;
    }
  }

  return {
    currentPassword,
    newPassword,
    confirmPassword,
    error,
    loading,
    passwordHint,
    reset,
    submit,
  };
}
