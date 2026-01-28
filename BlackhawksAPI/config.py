"""Configuration helpers for connecting to the Blackhawks Snowflake warehouse."""

from dataclasses import dataclass
import os
from configparser import ConfigParser
from typing import Optional


@dataclass
class SnowflakeConfig:
    """Connection details for Snowflake.

    The values are read from ~/.hawks.ini:

    - ``account``
    - ``user``
    - ``dbname``
    - ``private_key_path``
    - ``private_key_passphrase``
    - ``role`` (optional)
    - ``warehouse`` (optional)
    """

    user: str
    account: str
    database: str
    private_key_file: str
    private_key_file_pwd: str
    role: Optional[str] = None
    warehouse: Optional[str] = None


def load_snowflake_config() -> SnowflakeConfig:
    """Load Snowflake configuration from ~/.hawks.ini.

    Returns
    -------
    SnowflakeConfig
        Populated configuration object.

    Raises
    ------
    ValueError
        If the config file is missing or required values are not present.
    FileNotFoundError
        If ~/.hawks.ini does not exist.
    """

    config_path = os.path.expanduser("~/.hawks.ini")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = ConfigParser()
    config.read(config_path)
    
    if "snowflake" not in config:
        raise ValueError("Missing [snowflake] section in ~/.hawks.ini")
    
    sf_section = config["snowflake"]
    
    missing = []
    user = sf_section.get("user", "").strip() if "user" in sf_section else None
    if not user:
        missing.append("user")
    
    account = sf_section.get("account", "").strip() if "account" in sf_section else None
    if not account:
        missing.append("account")
    
    database = sf_section.get("dbname", "").strip() if "dbname" in sf_section else None
    if not database:
        missing.append("dbname")
    
    private_key_file = sf_section.get("private_key_path", "").strip() if "private_key_path" in sf_section else None
    if not private_key_file:
        missing.append("private_key_path")
    else:
        private_key_file = os.path.expanduser(private_key_file)
    
    private_key_file_pwd = sf_section.get("private_key_passphrase", "").strip() if "private_key_passphrase" in sf_section else None
    if not private_key_file_pwd:
        missing.append("private_key_passphrase")
    
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(
            f"Missing required keys in [snowflake] section of ~/.hawks.ini: {missing_keys}"
        )
    
    return SnowflakeConfig(
        user=user,
        account=account,
        database=database,
        private_key_file=private_key_file,
        private_key_file_pwd=private_key_file_pwd,
        role=sf_section.get("role", "").strip() if "role" in sf_section and sf_section.get("role", "").strip() else None,
        warehouse=sf_section.get("warehouse", "").strip() if "warehouse" in sf_section and sf_section.get("warehouse", "").strip() else None,
    )
