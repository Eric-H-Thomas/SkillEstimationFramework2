"""Configuration helpers for connecting to the Blackhawks Snowflake warehouse."""

from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class SnowflakeConfig:
    """Connection details for Snowflake.

    The values are read from environment variables prefixed with ``BLACKHAWKS_SNOWFLAKE_``:

    - ``BLACKHAWKS_SNOWFLAKE_USER``
    - ``BLACKHAWKS_SNOWFLAKE_PASSWORD``
    - ``BLACKHAWKS_SNOWFLAKE_ACCOUNT``
    - ``BLACKHAWKS_SNOWFLAKE_DATABASE``
    - ``BLACKHAWKS_SNOWFLAKE_ROLE`` (optional)
    - ``BLACKHAWKS_SNOWFLAKE_WAREHOUSE`` (optional)
    """

    user: str
    password: str
    account: str
    database: str
    role: Optional[str] = None
    warehouse: Optional[str] = None


def _get_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value and value.strip() else None


def load_snowflake_config() -> SnowflakeConfig:
    """Load Snowflake configuration from environment variables.

    Returns
    -------
    SnowflakeConfig
        Populated configuration object.

    Raises
    ------
    ValueError
        If a required environment variable is missing.
    """

    missing = []
    user = _get_env("BLACKHAWKS_SNOWFLAKE_USER")
    if not user:
        missing.append("BLACKHAWKS_SNOWFLAKE_USER")

    password = _get_env("BLACKHAWKS_SNOWFLAKE_PASSWORD")
    if not password:
        missing.append("BLACKHAWKS_SNOWFLAKE_PASSWORD")

    account = _get_env("BLACKHAWKS_SNOWFLAKE_ACCOUNT")
    if not account:
        missing.append("BLACKHAWKS_SNOWFLAKE_ACCOUNT")

    database = _get_env("BLACKHAWKS_SNOWFLAKE_DATABASE")
    if not database:
        missing.append("BLACKHAWKS_SNOWFLAKE_DATABASE")

    if missing:
        missing_vars = ", ".join(missing)
        raise ValueError(
            "Missing required Snowflake environment variables: " f"{missing_vars}"
        )

    return SnowflakeConfig(
        user=user,
        password=password,
        account=account,
        database=database,
        role=_get_env("BLACKHAWKS_SNOWFLAKE_ROLE"),
        warehouse=_get_env("BLACKHAWKS_SNOWFLAKE_WAREHOUSE"),
    )
