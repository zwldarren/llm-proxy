"""Offline recovery for a console account whose password is unknown.

The operator runs this on the host that owns the database while the proxy is
stopped (or at least not writing to the same account). It forces a password
change at the next sign-in and revokes every existing JWT and session API key,
so the new password is the only surviving credential.
"""

import argparse
import asyncio
import getpass
import os
import sys

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def _read_new_password() -> str:
    from llm_proxy.security.passwords import validate_password_strength

    password = getpass.getpass("New password: ")
    confirmation = getpass.getpass("Confirm new password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")
    try:
        validate_password_strength(password)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return password


async def _reset(username: str, password: str) -> int:
    from llm_proxy.database import UserRepository, UserSessionRepository, close_db, init_db
    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.security.passwords import hash_password

    await init_db()
    try:
        async with get_async_session_context() as session:
            repo = UserRepository(session)
            user = await repo.get_by_username(username)
            if user is None:
                print(f"No account named {username!r}.", file=sys.stderr)
                return 1
            await repo.update_password(user.id, hash_password(password))
            # An offline reset is a recovery action: the operator chose this
            # password, so the account owner must replace it on next sign-in.
            await repo.set_must_change_password(user.id, True)
            await repo.increment_token_version(user.id)
            await UserSessionRepository(session).deactivate_user_sessions(user.id)
            await session.commit()
            print(
                f"Password reset for {user.username} ({user.role}). "
                "All existing sessions were revoked; a password change is required at next sign-in."
            )
            return 0
    finally:
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset a console account password while the proxy is offline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="SQLite database path (overrides LLM_PROXY_DB_PATH/DATABASE_URL)",
    )
    parser.add_argument("--username", required=True, help="Account to reset")
    args = parser.parse_args()

    if args.config:
        # get_db_url() prefers DATABASE_URL, so a stale Postgres URL in the
        # environment would silently send the reset to the wrong store.
        os.environ.pop("DATABASE_URL", None)
        os.environ["LLM_PROXY_DB_PATH"] = args.config
    password = _read_new_password()
    raise SystemExit(asyncio.run(_reset(args.username, password)))


if __name__ == "__main__":
    main()
