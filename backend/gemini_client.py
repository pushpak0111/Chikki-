# gemini_client.py
from google import genai
from google.genai import types
import os
import dotenv

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")

client = genai.Client(api_key=GOOGLE_API_KEY)

system_prompt = """You are Jarvis, a personal AI assistant and productivity companion. You have access to the following tools:
- Notion: create pages, search notes, append content
- Google Calendar: check schedule, create events, set reminders
- Google Search: look up real-time information on the web

When the user asks you to do something that requires one of these tools, call the appropriate function.
Always confirm the action verbally after completing it. Keep responses short and conversational.
Maintain a polite, professional tone with a personal touch when appropriate.
"""


def build_config(extra_tools=None):
    """Build LiveConnectConfig, merging Google Search with any extra tool declarations."""
    tools = [types.Tool(google_search=types.GoogleSearch())]
    if extra_tools:
        tools.extend(extra_tools)
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_LOW",
        system_instruction=system_prompt,
        speech_config=types.SpeechConfig(
            language_code="en-US",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
        tools=tools,
    )


class GeminiClient:
    def __init__(self, extra_tools=None):
        self.client = client
        self.config = build_config(extra_tools)

    def connect(self):
        """Return async context manager for Gemini Live session."""
        return self.client.aio.live.connect(model=MODEL, config=self.config)
