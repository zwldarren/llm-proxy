import { beforeEach, describe, expect, it, vi } from "vitest";
import i18n from "@/i18n";
import { HttpError } from "@/services/http";
import { useChangePassword } from "./useChangePassword";
import { meApi } from "@/services/api/me";

vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return {
    ...actual,
    useI18n: () => ({ t: (key: string) => key }),
  };
});

vi.mock("@/services/api/me", () => ({
  meApi: { changePassword: vi.fn() },
}));

const mockedChangePassword = vi.mocked(meApi.changePassword);

function fillStrongPassword(form: ReturnType<typeof useChangePassword>) {
  form.newPassword.value = "NewPassw0rd!";
  form.confirmPassword.value = "NewPassw0rd!";
}

describe("useChangePassword", () => {
  beforeEach(() => {
    mockedChangePassword.mockReset();
    mockedChangePassword.mockResolvedValue({ message: "ok" });
  });

  it("requires the current password for accounts that have one", async () => {
    const form = useChangePassword();
    fillStrongPassword(form);

    await expect(form.submit()).resolves.toBe(false);
    expect(form.error.value).toBe("profile.passwordRequired");
    expect(mockedChangePassword).not.toHaveBeenCalled();
  });

  it("submits null for passwordless accounts, which prove a step-up instead", async () => {
    const form = useChangePassword({ requireCurrentPassword: () => false });
    fillStrongPassword(form);

    await expect(form.submit()).resolves.toBe(true);
    expect(mockedChangePassword).toHaveBeenCalledWith(null, "NewPassw0rd!");
  });

  it("sends the current password when the account has one", async () => {
    const form = useChangePassword();
    fillStrongPassword(form);
    form.currentPassword.value = "0ldPassw0rd!";

    await expect(form.submit()).resolves.toBe(true);
    expect(mockedChangePassword).toHaveBeenCalledWith("0ldPassw0rd!", "NewPassw0rd!");
  });

  it("keeps the form open and explains a wrong current password", async () => {
    mockedChangePassword.mockRejectedValue(
      new HttpError(400, "Bad Request", {
        error: { code: "invalid_current_password", message: "Current password is incorrect" },
      })
    );
    const form = useChangePassword();
    fillStrongPassword(form);
    form.currentPassword.value = "wrong";

    await expect(form.submit()).resolves.toBe(false);
    expect(form.error.value).toBe(i18n.global.t("profile.wrongPassword"));
  });
});
