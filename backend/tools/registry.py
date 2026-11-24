# tools/registry.py
"""
Tool Registry — single source of truth for all Gemini function declarations.

Adds Notion and Google Calendar tools to Gemini's CONFIG and dispatches
function_call responses received from the live session back to the correct
async handler.
"""
import json
from google.genai import types

from tools.notion_tool import (
    notion_create_page,
    notion_search,
    notion_append_to_page,
)
from tools.calendar_tool import (
    calendar_get_events,
    calendar_create_event,
    calendar_set_reminder,
)

# ---------------------------------------------------------------------------
# Function declarations (schema exposed to Gemini)
# ---------------------------------------------------------------------------

_NOTION_CREATE_PAGE = types.FunctionDeclaration(
    name="notion_create_page",
    description=(
        "Create a new page in the user's Notion workspace or a specific database."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "title": types.Schema(
                type=types.Type.STRING,
                description="Title of the new Notion page.",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="Main text content for the page body.",
            ),
            "database_id": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Optional. Notion database ID to create the page in. "
                    "If omitted, the default database from .env is used."
                ),
            ),
        },
        required=["title", "content"],
    ),
)

_NOTION_SEARCH = types.FunctionDeclaration(
    name="notion_search",
    description="Search the user's Notion workspace for pages or databases.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="Search term to look for in Notion.",
            ),
        },
        required=["query"],
    ),
)

_NOTION_APPEND = types.FunctionDeclaration(
    name="notion_append_to_page",
    description="Append a block of text to an existing Notion page.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "page_id": types.Schema(
                type=types.Type.STRING,
                description="The Notion page ID to append content to.",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="Text content to append as a new paragraph block.",
            ),
        },
        required=["page_id", "content"],
    ),
)

_CALENDAR_GET_EVENTS = types.FunctionDeclaration(
    name="calendar_get_events",
    description="Retrieve events from Google Calendar for a specific date.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "date": types.Schema(
                type=types.Type.STRING,
                description="ISO date string, e.g. '2025-03-15'.",
            ),
            "calendar_id": types.Schema(
                type=types.Type.STRING,
                description="Optional. Calendar ID (default 'primary').",
            ),
        },
        required=["date"],
    ),
)

_CALENDAR_CREATE_EVENT = types.FunctionDeclaration(
    name="calendar_create_event",
    description="Create a new event in Google Calendar.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "title": types.Schema(
                type=types.Type.STRING,
                description="Title of the calendar event.",
            ),
            "start_datetime": types.Schema(
                type=types.Type.STRING,
                description="Event start as ISO 8601 datetime, e.g. '2025-03-15T14:00:00'.",
            ),
            "end_datetime": types.Schema(
                type=types.Type.STRING,
                description="Event end as ISO 8601 datetime, e.g. '2025-03-15T15:00:00'.",
            ),
            "description": types.Schema(
                type=types.Type.STRING,
                description="Optional event description.",
            ),
        },
        required=["title", "start_datetime", "end_datetime"],
    ),
)

_CALENDAR_SET_REMINDER = types.FunctionDeclaration(
    name="calendar_set_reminder",
    description="Create a timed reminder as a short 15-minute calendar event.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "title": types.Schema(
                type=types.Type.STRING,
                description="What to be reminded about.",
            ),
            "remind_at": types.Schema(
                type=types.Type.STRING,
                description="ISO 8601 datetime when the reminder should fire, e.g. '2025-03-15T17:00:00'.",
            ),
        },
        required=["title", "remind_at"],
    ),
)

# ---------------------------------------------------------------------------
# Dispatch table  name → async callable
# ---------------------------------------------------------------------------

_DISPATCH: dict = {
    "notion_create_page": notion_create_page,
    "notion_search": notion_search,
    "notion_append_to_page": notion_append_to_page,
    "calendar_get_events": calendar_get_events,
    "calendar_create_event": calendar_create_event,
    "calendar_set_reminder": calendar_set_reminder,
}


class ToolRegistry:
    """
    Central registry that exposes Gemini FunctionDeclarations and
    dispatches function_call results to the correct async handler.
    """

    def get_gemini_tool_declarations(self) -> list:
        """
        Return a list of types.Tool objects to be merged into the Gemini CONFIG.
        Each types.Tool can hold multiple FunctionDeclarations.
        """
        notion_tool = types.Tool(
            function_declarations=[
                _NOTION_CREATE_PAGE,
                _NOTION_SEARCH,
                _NOTION_APPEND,
            ]
        )
        calendar_tool = types.Tool(
            function_declarations=[
                _CALENDAR_GET_EVENTS,
                _CALENDAR_CREATE_EVENT,
                _CALENDAR_SET_REMINDER,
            ]
        )
        return [notion_tool, calendar_tool]

    async def execute(self, tool_name: str, args: dict) -> str:
        """
        Dispatch a function call from Gemini to the correct tool implementation.

        Args:
            tool_name: Name matching a FunctionDeclaration.
            args: Keyword arguments forwarded to the handler.

        Returns:
            A JSON-serializable string result. Never raises — returns an
            error string so Gemini can inform the user gracefully.
        """
        handler = _DISPATCH.get(tool_name)
        if handler is None:
            return f"Unknown tool '{tool_name}'. No handler registered."

        try:
            result = await handler(**args)
            return result if isinstance(result, str) else json.dumps(result)
        except TypeError as e:
            return f"Invalid arguments for tool '{tool_name}': {e}"
        except Exception as e:
            return f"Tool '{tool_name}' raised an unexpected error: {e}"
