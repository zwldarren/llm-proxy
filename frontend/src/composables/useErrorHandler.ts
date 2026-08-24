import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { getErrorMessage } from "@/utils/error";

export function useErrorHandler() {
  const { t } = useI18n();

  const handleError = (
    e: unknown,
    options?: {
      title?: string;
      description?: string;
    }
  ) => {
    console.error("Error occurred:", e);
    const title = options?.title || t("errors.fetchFailed") || "An error occurred";
    const description = options?.description || getErrorMessage(e);
    toast.error(title, { description });
  };

  const handleSaveError = (e: unknown) => {
    handleError(e, { title: t("errors.saveFailed") });
  };

  /**
   * Handle a delete operation error with standard toast.
   */
  const handleDeleteError = (e: unknown) => {
    handleError(e, { title: t("errors.deleteFailed") });
  };

  return {
    handleError,
    handleSaveError,
    handleDeleteError,
  };
}
