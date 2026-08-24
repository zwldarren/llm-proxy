import { ref } from "vue";
import { useI18n } from "vue-i18n";
import { meApi } from "@/services/api/me";
import { HttpError } from "@/services/http";
import { getErrorMessage } from "@/utils/error";
import { passwordRequirementsText, validatePasswordStrength } from "@/utils/password";

/**
 * Shared state and submit flow for the change-password form.
 *
 * Used by the sidebar change-password dialog and the forced-password-change
 * page so validation, error mapping, and the API call are defined once. The
 * caller decides what happens on success (close the dialog / end the
 * session and redirect); the form only reports whether the change landed.
 */
export function useChangePassword() {
  const { t } = useI18n();

  const currentPassword = ref("");
  const newPassword = ref("");
  const confirmPassword = ref("");
  const error = ref("");
  const loading = ref(false);
  const passwordHint = passwordRequirementsText();

  function reset() {
    currentPassword.value = "";
    newPassword.value = "";
    confirmPassword.value = "";
    error.value = "";
    loading.value = false;
  }

  async function submit(): Promise<boolean> {
    if (loading.value) return false;

    if (!currentPassword.value) {
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
      await meApi.changePassword(currentPassword.value, newPassword.value);
      return true;
    } catch (e: unknown) {
      // The backend rejects a wrong current password with 401.
      if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
        error.value = t("profile.wrongPassword");
      } else {
        error.value = getErrorMessage(e) || t("errors.saveFailed");
      }
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
