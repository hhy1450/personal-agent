"""Application configuration loaded from environment variables."""
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# LLM — Text (DeepSeek)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# LLM — Vision (Qwen-VL via DashScope)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")

# Workspace
WORKSPACE_DIR = PROJECT_ROOT / os.getenv("WORKSPACE_DIR", "workspace")

# MySQL
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "personal_agent")

# Log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Max retries for agent tool calls
MAX_RETRIES = 3


def setup_logging(name: str = "personal_agent") -> logging.Logger:
    """Configure structured logging for the application.

    Sets up a consistent format across all modules:
    ``HH:MM:SS [LEVEL] module: message``

    Call once at startup (CLI or Web); individual modules get their
    logger via ``logging.getLogger(__name__)`` and inherit this config.

    Args:
        name: Root logger name (default ``personal_agent``).

    Returns:
        The configured root logger.
    """
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    return logger
