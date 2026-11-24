# tools/calendar_tool.py
"""
Google Calendar productivity tool.

Requires in .env:
  GOOGLE_CALENDAR_CREDENTIALS_FILE  – path to the OAuth client secrets JSON
                                       (e.g. credentials/google_calendar_credentials.json)

On first run the OAuth consent flow will open a browser window.
The resulting token is stored at credentials/token.json and refreshed automatically.
"""
import os
import json
import asyncio
import dotenv
from pathlib import Path
from typing import Optional

dotenv.load_dotenv()

CREDENTIALS_FILE = os.getenv(
    "GOOGLE_CALENDAR_CREDENTIALS_FILE",
    "credentials/google_calendar_credentials.json",
)
TOKEN_FILE = str(Path(CREDENTIALS_FILE).parent / "token.json")

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_credentials():
    """Load or refresh OAuth2 credentials, running the consent flow if needed."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                return None  # credentials file missing; caller will handle
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Persist token for future runs
        Path(TOKEN_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return creds


def _build_service():
    """Build and return a Google Calendar API service object."""
    from googleapiclient.discovery import build

    creds = _get_credentials()
    if creds is None:
        return None
    return build("calendar", "v3", credentials=creds)


async def calendar_get_events(
    date: str,
    calendar_id: str = "primary",
) -> str:
    """
    Retrieve events for a given date from Google Calendar.

    Args:
        date: ISO date string, e.g. "2025-03-15".
        calendar_id: Calendar ID (default "primary").

    Returns:
        Formatted list of events or an informative message.
    """
    try:
        service = await asyncio.to_thread(_build_service)
        if service is None:
            return "Google Calendar credentials not found. Please add the credentials file."

        from datetime import datetime, timezone
        # Build start/end as RFC3339
        day_start = f"{date}T00:00:00Z"
        day_end = f"{date}T23:59:59Z"

        events_result = await asyncio.to_thread(
            lambda: service.events().list(
                calendarId=calendar_id,
                timeMin=day_start,
                timeMax=day_end,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        )
        events = events_result.get("items", [])
        if not events:
            return f"No events found on {date}."

        lines = [f"Events on {date}:"]
        for ev in events:
            start = ev["start"].get("dateTime", ev["start"].get("date", ""))
            summary = ev.get("summary", "(No title)")
            # Shorten to HH:MM if datetime
            if "T" in start:
                start = start[11:16]
            lines.append(f"  • {start} – {summary}")
        return "\n".join(lines)
    except Exception as e:
        return f"Google Calendar error: {e}"


async def calendar_create_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    calendar_id: str = "primary",
) -> str:
    """
    Create a new event in Google Calendar.

    Args:
        title: Event title.
        start_datetime: ISO 8601 datetime, e.g. "2025-03-15T14:00:00".
        end_datetime: ISO 8601 datetime, e.g. "2025-03-15T15:00:00".
        description: Optional event description.
        calendar_id: Calendar ID (default "primary").

    Returns:
        Confirmation with event link, or an error message.
    """
    try:
        service = await asyncio.to_thread(_build_service)
        if service is None:
            return "Google Calendar credentials not found."

        event_body = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_datetime, "timeZone": "UTC"},
            "end": {"dateTime": end_datetime, "timeZone": "UTC"},
        }

        created = await asyncio.to_thread(
            lambda: service.events().insert(
                calendarId=calendar_id,
                body=event_body,
            ).execute()
        )
        link = created.get("htmlLink", "")
        return f"Created calendar event '{title}'. Link: {link}"
    except Exception as e:
        return f"Google Calendar error: {e}"


async def calendar_set_reminder(title: str, remind_at: str) -> str:
    """
    Create a short reminder event (15-minute block) in Google Calendar.

    Args:
        title: Reminder title.
        remind_at: ISO 8601 datetime for the reminder, e.g. "2025-03-15T17:00:00".

    Returns:
        Confirmation string or error message.
    """
    from datetime import datetime, timedelta

    try:
        # Parse start time and set end = start + 15 min
        dt = datetime.fromisoformat(remind_at)
        end_dt = dt + timedelta(minutes=15)
        return await calendar_create_event(
            title=f"⏰ Reminder: {title}",
            start_datetime=dt.isoformat(),
            end_datetime=end_dt.isoformat(),
            description=f"Reminder set by Jarvis: {title}",
        )
    except Exception as e:
        return f"Failed to set reminder: {e}"
