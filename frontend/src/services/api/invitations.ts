import { http } from "../http";
import type { Invitation, InvitationCreated, PublicInvitation, Token } from "@/types/schemas";

interface InvitationCreateRequest {
  email: string;
  role: "admin" | "viewer";
  allowed_models?: string[] | null;
  budget_usd?: number | null;
  budget_period?: "daily" | "weekly" | "monthly" | null;
  budget_reset_day?: number | null;
}

export const invitationsApi = {
  create: (data: InvitationCreateRequest) =>
    http.post<InvitationCreated>("/api/team/invitations", data),
  list: () => http.get<{ items: Invitation[]; total: number }>("/api/team/invitations"),
  revoke: (id: string) => http.delete<Invitation>(`/api/team/invitations/${id}`),
  resend: (id: string) => http.post<InvitationCreated>(`/api/team/invitations/${id}/resend`, {}),
  getPublic: (token: string) =>
    http.get<PublicInvitation>(`/api/auth/invitations/${encodeURIComponent(token)}`),
  accept: (token: string, username: string, password: string) =>
    http.post<Token>(`/api/auth/invitations/${encodeURIComponent(token)}/accept`, {
      username,
      password,
    }),
};
