"""Configuration loading (YAML file + .env)."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load .env and configs/config.yaml (override the path with AGENT_CONFIG)."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(os.getenv("AGENT_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
