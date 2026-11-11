# 0060 — Resend as a second email transport

## Status

Accepted (2026-11).

## Context

Console mail (invitations, email-address verification, password resets) had one
delivery path: a plain SMTP relay the operator configures. That is a problem for
the deployments this proxy is often installed on. A self-hosted host usually has
no local mail relay, and its IP reputation makes direct-to-MX delivery
unreliable, so invitations and password resets silently degrade to "copy the
link" and "ask an administrator".

Resend is a transactional email API: one HTTPS `POST` with a bearer key. It needs
no inbound port, no relay, and no DNS work beyond a verified sender domain.

The delivery contract already existed and had to stay: one entry point
(`notifications.email.send_email`), a boolean result, delivery failures logged
and reported as `False` rather than raised, so an invitation row is never rolled
back and the copyable link remains the fallback.

## Decision

- **Configuration.** A second block, `console_auth.resend`
  (`enabled`, `api_key`, `from_address`), sits beside the existing `smtp` block.
  Neither block replaces the other; both are stored, so an operator can switch
  transports without retyping credentials.
- **Selection.** `ConsoleAuthSettings.email_transport()` returns the block to
  use: Resend when it is enabled and carries both a key and a sender, otherwise
  nothing, and SMTP when Resend is not enabled and SMTP is complete, otherwise
  nothing. An enabled block is the operator's choice, so an incomplete one
  selects no transport rather than the other block; each block answers for
  itself with `is_deliverable()`, and each carries the name it goes by in
  `kind`. Callers ask `email_transport()` (or `email_enabled`) instead of
  reading `smtp.enabled`, so "can this instance send mail?" has one answer.
- **Transport.** `POST https://api.resend.com/emails` through the existing
  `httpx2` dependency with a 20-second timeout, a bearer API key, and a
  plain-text body (`from`, `to`, `subject`, `text`). `raise_for_status` turns a
  rejected message into the same `False` result SMTP failures produce. No SDK
  dependency is added.
- **Secrets.** The Resend key is encrypted at rest and masked as `****` on read,
  exactly like the SMTP password and OIDC client secrets. Echoing the mask keeps
  the stored key; editing the mask is rejected with 422.
- **Verification.** `POST /api/config/server/console-auth/test-email`
  (admin-only, step-up within the last 15 minutes like the `PUT` that stores the
  credentials, rate-limited by the `config.test_email` bucket) sends one message
  through the selected transport and names it in the response. It reads the
  saved configuration, so the UI disables it while the form is dirty.

## Alternatives considered

- **A single `email` block with `provider: smtp | resend`.** Cleaner on paper,
  but it rewrites every stored `console_auth` row and every contract assertion
  about the SMTP shape, and it buys nothing: the two blocks share only the
  sender address.
- **The official Resend SDK.** Four lines of HTTP do not justify a new runtime
  dependency with its own lazy-import boundary and release cadence.
- **Automatic SMTP fallback when Resend fails.** A timeout after Resend already
  accepted the message would deliver it twice, and a bad API key would look like
  working mail. The operator chooses the transport with the enable switches; the
  test endpoint is how a misconfiguration is found.

## Consequences

An instance with Resend enabled and misconfigured does not fall back to SMTP: it
reports delivery failures as `False` until the key or sender is fixed, which is
visible in the invitation response (`email_sent: false`) and in the test button.
Verified addresses written from now on carry `email_verified_by` = `email`, the
transport-neutral name for an emailed link; rows written before Resend existed
carry `smtp`, which means the same thing, so no migration is needed. The column
comment says so.
