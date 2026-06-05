"""Shared test fixtures."""
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def workspace_dir(tmp_path):
    """Temporary workspace directory for tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def sample_task():
    """A typical user task for testing."""
    return "帮我调研一下 DeepSeek V3 的特点"
