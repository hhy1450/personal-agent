"""Tests for web_search tool."""
import importlib
from unittest.mock import patch, MagicMock
import src.tools.web_search
from src.tools.web_search import web_search
from src.tools.registry import _registry, get_tool


def test_web_search_registered():
    """web_search should be registered as a tool after import."""
    # Re-import to ensure web_search is registered (other tests may clear the registry)
    importlib.reload(src.tools.web_search)
    t = get_tool("web_search")
    assert t is not None
    assert t.name == "web_search"


def test_web_search_returns_results():
    """web_search should return list of dicts with title/url/snippet."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "AbstractText": "DeepSeek V3 is a large language model.",
        "AbstractSource": "Wikipedia",
        "AbstractURL": "https://en.wikipedia.org/wiki/DeepSeek",
        "RelatedTopics": [
            {"Text": "DeepSeek V3 features MoE architecture.", "FirstURL": "https://example.com/deepseek-v3"},
            {"Text": "DeepSeek is cost-effective.", "FirstURL": "https://example.com/deepseek"},
        ],
    }

    with patch("src.tools.web_search.httpx.get", return_value=mock_response):
        results = web_search(query="DeepSeek V3", max_results=2)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert "title" in results[0]
    assert "url" in results[0]
    assert "snippet" in results[0]


def test_web_search_empty_results():
    """web_search should return a placeholder when no results found."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "AbstractText": "",
        "RelatedTopics": [],
    }

    with patch("src.tools.web_search.httpx.get", return_value=mock_response):
        results = web_search(query="xyznonexistent12345")

    assert len(results) == 1
    assert "No results" in results[0]["title"]


def test_web_search_error_handling():
    """web_search should return error info instead of raising."""
    with patch("src.tools.web_search.httpx.get", side_effect=Exception("Connection error")):
        results = web_search(query="test")

    assert len(results) == 1
    assert "Search Error" in results[0]["title"]
