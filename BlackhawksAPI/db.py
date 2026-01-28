"""Snowflake database helpers for the Blackhawks API layer."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import pandas as pd
import snowflake.connector

from .config import load_snowflake_config

LOG_TOPIC = "blackhawks_snowflake"


def get_connection() -> snowflake.connector.SnowflakeConnection:
    """Create and return a new Snowflake connection."""

    snow_config = load_snowflake_config()
    return snowflake.connector.connect(
        user=snow_config.user,
        account=snow_config.account,
        database=snow_config.database,
        authenticator="SNOWFLAKE_JWT",
        private_key_file=snow_config.private_key_file,
        private_key_file_pwd=snow_config.private_key_file_pwd,
        role=snow_config.role,
    )


def execute(
    query: str,
    query_params: Optional[dict | Sequence] = None,
    warehouse: Optional[str] = None,
    timeout: Optional[int] = None,
):
    """Run a query and return the rows as dictionaries."""

    with get_connection() as conn:
        with conn.cursor(snowflake.connector.DictCursor) as cursor:
            if warehouse:
                logging.getLogger(LOG_TOPIC).info("Using specified warehouse %s", warehouse)
                cursor.execute(f"USE WAREHOUSE {warehouse};")
            cursor.execute(query, query_params, timeout=timeout)
            return cursor.fetchall()


def get_df(query: str, query_params: Optional[dict | Sequence] = None) -> pd.DataFrame:
    """Run the provided query and return a DataFrame of results."""

    return pd.DataFrame(execute(query, query_params=query_params))


def get_last_updated(table: str, column: str = "last_updated"):
    """Retrieve the latest timestamp in ``column`` from ``table``."""

    logging.getLogger(LOG_TOPIC).info("Getting latest %s timestamp from %s", column, table)
    query = f"""
    SELECT COALESCE(MAX({column})::DATETIME, '1800-01-01'::DATETIME) AS last_updated
    FROM {table}
    """
    with get_connection() as conn:
        with conn.cursor(snowflake.connector.DictCursor) as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
            return row.get("LAST_UPDATED") if row else None


def run_file(filepath: str):
    """Execute a SQL file relative to the repository root."""

    import os
    import re

    full_filepath = os.path.normpath(f"{os.path.dirname(__file__)}/../{filepath}")
    logger = logging.getLogger(LOG_TOPIC)
    logger.info("Running SQL found in %s", full_filepath)
    if not os.path.isfile(full_filepath):
        logger.error("Unable to continue. File does not exist %s", full_filepath)
        return

    with open(full_filepath, mode="r", encoding="utf-8") as filehandle:
        sql = filehandle.read()
        with get_connection() as conn:
            with conn.cursor(snowflake.connector.DictCursor) as cursor:
                try:
                    cursor.execute("BEGIN;")
                    statements = [x for x in sql.split(";") if x.strip()]
                    for statement in statements:
                        logger.debug(
                            "Running statement: %s", statement.strip().partition("\n")[0]
                        )
                        cursor.execute(statement)
                        if re.search(r"^CREATE|DROP|ALTER", statement, flags=re.MULTILINE):
                            logger.debug("Beginning new transaction")
                            cursor.execute("BEGIN;")
                        res = cursor.fetchall()
                        if res:
                            logger.debug("Statement result: %s", res[0])
                    cursor.execute("COMMIT;")
                except snowflake.connector.errors.ProgrammingError as error:
                    cursor.execute("ROLLBACK;")
                    logger.error("Failed running queries in: %s", filepath)
                    logger.error(error)
                    raise
