import { http } from "../http";
import type { LoginRequest, Token } from "@/types/schemas";

const BASE_URL = "/api/auth";

interface SetupRequest {
  username: string;
  password: string;
}

interface SetupStatus {
  needs_setup: boolean;
}

export const authApi = {
  getSetupStatus: () => http.get<SetupStatus>(`${BASE_URL}/setup-status`),
  setup: (data: SetupRequest) => http.post<Token>(`${BASE_URL}/setup`, data),
  login: (data: LoginRequest) => http.post<Token>(`${BASE_URL}/login`, data),
  logout: () => http.post<{ success: boolean }>(`${BASE_URL}/logout`, undefined),
};
