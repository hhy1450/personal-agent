"""Web search tool. Tries DuckDuckGo, falls back to model knowledge."""
import httpx

from src.tools.registry import tool


@tool(
    name="web_search",
    description="Search the web for information. Returns a list of relevant results with titles, URLs, and snippets.",
)
def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web. If search API is unavailable, prompts the model to use its own knowledge.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []

        if data.get("AbstractText"):
            results.append({
                "title": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"],
            })

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " ").title(),
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic["Text"],
                })

        if results:
            return results[:max_results]

        # No results found, tell model to use its own knowledge
        return [{
            "title": "Use your knowledge",
            "url": "",
            "snippet": (
                f"No search results found for '{query}'. "
                f"You are a knowledgeable AI — please answer based "
                f"on your training data and provide a thorough response."
            )
        }]

    except Exception:
        # Search API unavailable, tell model to use its own knowledge
        return [{
            "title": "Use your knowledge",
            "url": "",
            "snippet": (
                f"Search is currently unavailable. "
                f"You are a knowledgeable AI trained on vast data — "
                f"please answer the question '{query}' based on your "
                f"training knowledge. Be thorough and specific."
            )
        }]
