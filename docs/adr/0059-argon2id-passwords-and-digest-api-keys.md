# Argon2id for passwords, digests for API keys

## Status

Accepted (2026-11).

## Context

Two different secrets were both stored with bcrypt, through two helpers in
`security/passwords.py` (`hash_password` / `verify_admin_password` and
`hash_api_key` / `verify_api_key`). bcrypt fits one of them.

**Passwords.** bcrypt at the library default cost (12 rounds) is not broken,
but it is the older choice: it is not memory-hard, so a GPU or ASIC farms it
far more cheaply than a CPU, and it truncates at 72 bytes — the reason the
strength policy has a byte cap at all. The alternative costs nothing in
compatibility: the hash format is self-describing, so old and new hashes can
coexist in the same column.

**API keys.** A key is `sk-` plus 32 random bytes — 256 bits of entropy, handed
to the client and never chosen by a human. bcrypt's cost factor buys protection
against exactly one attack it cannot suffer: an offline search of a
low-entropy secret. What it did cost was real:

- `key_hash` was a salted bcrypt hash, so it could not be indexed. Every cache
  miss resolved the key by comparing the presented string against **every**
  stored key (`gate._find_matching_key`), one bcrypt verify per key, on a
  thread, behind a process-wide lock (`api_key_cache.get_key_verification_lock`)
  that existed only to bound that burst.
- The verified-key cache had a special revalidation path
  (`VerifiedKeyInfo.key_hash`) whose only purpose was to avoid re-running the
  scan after a TTL rollover.

On the development machine, one bcrypt cost-12 verify measured ~250 ms; a
SHA-256 digest of the same string is microseconds, and an index lookup is one
dictionary hit.

## Decision

- **Passwords move to Argon2id.** `hash_password` writes Argon2id at the OWASP
  baseline (m = 19 MiB, t = 2, p = 1), measured at ~22 ms per hash/verify here.
  The one-lane configuration is deliberate: the memory cost per concurrent
  login is bounded, which matters because the login endpoint is reachable
  unauthenticated. Parameters are recorded inside each hash, so raising them
  later is a rehash, not a migration.
- **Legacy bcrypt password hashes stay verifiable.** `verify_password`
  dispatches on the stored prefix (`$argon2id$` vs `$2a$/$2b$/$2y$`), and
  `password_needs_rehash` reports a legacy hash. The login endpoint rehashes on
  success — the only moment the plaintext is in hand — and commits before it
  returns, so the MFA-challenge path (which writes nothing) still persists it.
  A bcrypt verify of a password longer than 72 bytes fails closed rather than
  raising, because no bcrypt credential can be that long.
- **API keys are stored as a bare SHA-256 hex digest.** `hash_api_key` is the
  stored value *and* the lookup key: `CachedApiKey.key_hash` is the digest, so
  the gate builds `dict[str, CachedApiKey]` from the snapshot and resolves a
  request in one hit. No pepper and no salt: with 256 bits of entropy there is
  nothing to brute-force, and a server-side secret would add rotation cost
  without adding strength. `api_key_matches_hash` (imported by the gate as
  `key_matches_hash`, the name ADR-0032 gave it) keeps a bcrypt branch for
  legacy rows.
- **Legacy API-key rows are upgraded lazily, in place.** The digest can only be
  derived while the plaintext is presented, so after a successful bcrypt match
  the gate calls `ApiKeyRepository.set_key_hash(name, digest)`, guarded by the
  current value, and updates the in-memory snapshot record. The write is best
  effort: a failure is logged and the key simply stays on the bcrypt path until
  its next verification. No migration script, no forced key rotation, and the
  gate no longer needs a bcrypt scan once a key has been used.
- **`has_password_hash` replaces `is_bcrypt_hash` at the call sites that were
  asking a different question.** `is_bcrypt_hash` answers "is this a bcrypt
  hash"; five call sites used it as "does this account have a password"
  (password-reset eligibility, step-up, last-passkey removal, and the
  `users.has_password` property). They now ask the format-agnostic question, so
  the next format change touches one place.
- **The 72-byte policy cap stays for now.** It is obsolete for Argon2id but
  still protects the bcrypt verify path (nothing larger can match a legacy
  hash) and bounds request size. Removing it is a separate policy decision with
  its own console strings.

## Alternatives rejected

- **Argon2id defaults (64 MiB, t = 3, p = 4).** ~50 ms here, and 64 MiB per
  concurrent login attempt on an unauthenticated endpoint. The OWASP baseline
  is the better trade for a self-hosted proxy.
- **Keep bcrypt for API keys and only index them by a separate column.** It
  would still require a bcrypt verify per successful authentication, add a
  column, and leave the process-wide verification lock in place for no
  security gain.
- **Force rotation of every existing API key.** Guarantees a digest-only table
  immediately, at the cost of breaking every deployment's integrations on
  upgrade. Lazy upgrade reaches the same end state for keys that are actually
  used.
- **A peppered/HMAC digest for API keys.** Useful when the stored value could
  be brute-forced offline; pointless at 256 bits, and it turns `ENCRYPTION_KEY`
  rotation into a key-hash rotation.
- **Keeping `verify_admin_password` as the name.** It is no longer
  admin-specific (team members authenticate through it) and no longer bcrypt;
  it is now `verify_password`.

## Consequences

- The gate's hot path is one indexed lookup. The negative cache, the
  single-flight verification lock, and the shielded-task wrapper remain, but
  they now cover only un-migrated rows and the session-key fallback.
- `hash_api_key_for_cache` disappeared: one digest function serves both the
  stored value and the cache key, so they cannot drift.
- Outstanding password-reset links are pinned to a digest of the stored hash
  (`_password_fingerprint`), so a rehash on sign-in invalidates a link that was
  minted before it. That is the same fail-safe direction as a used link: the
  holder signs in with the password they just proved and requests a new one.
- A database backup still contains legacy bcrypt password hashes and legacy
  API-key hashes until the corresponding credentials are used once. Both are
  verified unchanged; nothing needs re-saving.
- Console docs and the security guide describe the new storage, and the
  login-timing note in `docs/admin/users.md` now says "password-hash cost"
  rather than "bcrypt cost".
