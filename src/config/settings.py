"""Application configuration loaded from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Workspace
WORKSPACE_DIR = PROJECT_ROOT / os.getenv("WORKSPACE_DIR", "workspace")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///personal_agent.db")

# Log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Max retries for agent tool calls
MAX_RETRIES = 3
