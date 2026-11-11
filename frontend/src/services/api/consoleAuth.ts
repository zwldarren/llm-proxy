import { http } from "../http";

interface SmtpSettings {
  enabled: boolean;
  host: string;
  port: number;
  username: string;
  password: string;
  from_address: string;
  starttls: boolean;
  use_ssl: boolean;
}

interface ResendSettings {
  enabled: boolean;
  api_key: string;
  from_address: string;
}

export interface TestEmailResult {
  sent: boolean;
  transport: "smtp" | "resend";
}

export interface OidcProviderSettings {
  id: string;
  name: string;
  issuer: string;
  client_id: string;
  client_secret: string;
  scopes: string;
  enabled: boolean;
}

export interface ConsoleAuthSettings {
  public_base_url: string;
  require_2fa_for_admins: boolean;
  require_2fa_after_passkey: boolean;
  smtp: SmtpSettings;
  resend: ResendSettings;
  oidc_providers: OidcProviderSettings[];
}

export const consoleAuthApi = {
  get: () => http.get<ConsoleAuthSettings>("/api/config/server/console-auth"),
  update: (data: ConsoleAuthSettings) =>
    http.put<ConsoleAuthSettings>("/api/config/server/console-auth", data),
  testEmail: (to: string) =>
    http.post<TestEmailResult>("/api/config/server/console-auth/test-email", { to }),
};
