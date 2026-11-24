# tools/notion_tool.py
"""
Notion productivity tool.

Requires in .env:
  NOTION_API_KEY               – Notion integration token
  NOTION_DEFAULT_DATABASE_ID   – (optional) default database for new pages
"""
import os
import json
import asyncio
import dotenv
from typing import Optional

dotenv.load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_DEFAULT_DATABASE_ID = os.getenv("NOTION_DEFAULT_DATABASE_ID", "")

# Minimum delay between consecutive Notion API calls to respect rate limits
_RATE_LIMIT_DELAY = 0.3
_last_call_time: float = 0.0


async def _rate_limit():
    """Enforce a minimum gap between Notion API calls."""
    global _last_call_time
    import time
    elapsed = time.monotonic() - _last_call_time
    if elapsed < _RATE_LIMIT_DELAY:
        await asyncio.sleep(_RATE_LIMIT_DELAY - elapsed)
    _last_call_time = time.monotonic()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

async def notion_create_page(self, title: str, content: str, database_id: str = None) -> str:
    try:
        await asyncio.sleep(0.3)
        
        # Use parent page ID instead of database
        parent_page_id = database_id or os.getenv("NOTION_DEFAULT_DATABASE_ID")
        
        response = await asyncio.to_thread(
            self.client.pages.create,
            parent={"type": "page_id", "page_id": parent_page_id},
            properties={
                "title": {
                    "title": [{"type": "text", "text": {"content": title}}]
                }
            },
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content}}]
                    }
                }
            ]
        )
        return f"Created page '{title}': {response['url']}"
    except Exception as e:
        return f"Failed to create Notion page: {str(e)}"


async def notion_search(query: str) -> str:
    """
    Search Notion for pages/databases matching the query.

    Args:
        query: Search term.

    Returns:
        Formatted list of matching titles and URLs, or an error message.
    """
    if not NOTION_API_KEY:
        return "Notion API key not configured."

    await _rate_limit()

    payload = {"query": query, "page_size": 5}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.notion.com/v1/search",
                headers=_headers(),
                json=payload,
            )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if not results:
                return f"No Notion results found for '{query}'."
            lines = []
            for item in results:
                obj_type = item.get("object", "")
                url = item.get("url", "")
                if obj_type == "page":
                    props = item.get("properties", {})
                    # Try common title property names
                    for key in ("title", "Name", "Title"):
                        if key in props:
                            title_arr = props[key].get("title", [])
                            if title_arr:
                                name = title_arr[0].get("text", {}).get("content", "Untitled")
                                lines.append(f"- {name}: {url}")
                                break
                    else:
                        lines.append(f"- (Untitled page): {url}")
                elif obj_type == "database":
                    db_title = item.get("title", [])
                    name = db_title[0].get("text", {}).get("content", "Untitled DB") if db_title else "Untitled DB"
                    lines.append(f"- [Database] {name}: {url}")
            return "Notion search results:\n" + "\n".join(lines)
        else:
            err = resp.json().get("message", resp.text)
            return f"Notion search failed: {err}"
    except Exception as e:
        return f"Notion API error: {e}"


async def notion_append_to_page(page_id: str, content: str) -> str:
    """
    Append a paragraph block to an existing Notion page.

    Args:
        page_id: The target Notion page ID.
        content: Text to append.

    Returns:
        Confirmation string or error message.
    """
    if not NOTION_API_KEY:
        return "Notion API key not configured."

    await _rate_limit()

    payload = {
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": content}}]
                },
            }
        ]
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.patch(
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                headers=_headers(),
                json=payload,
            )
        if resp.status_code == 200:
            return f"Successfully appended content to Notion page {page_id}."
        else:
            err = resp.json().get("message", resp.text)
            return f"Failed to append to Notion page: {err}"
    except Exception as e:
        return f"Notion API error: {e}"
