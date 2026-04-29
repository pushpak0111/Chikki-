import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import (
    google,
    ai_coustics,
    silero
)
from gemini_client import MODEL, API_KEY, system_prompt

load_dotenv(".env")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=system_prompt)

server = AgentServer()

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):
    print(f"--- New Session for Room: {ctx.room.name} ---")
    
    # Initialize Gemini Realtime Model with explicit API Key and Model ID
    model = google.realtime.RealtimeModel(
        model=MODEL, # Using model from gemini_client.py
        api_key=API_KEY, # Using API Key from gemini_client.py
        voice="Aoede",
        instructions=system_prompt
    )

    # Initialize AgentSession with VAD
    # VAD is CRITICAL for it to work with the frontend audio stream
    session = AgentSession(
        llm=model,
        vad=silero.VAD.load() 
    )

    print("🚀 Starting session with audio enhancement...")
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Keeping your ai_coustics enhancement
                noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_L),
            ),
        ),
    )

    # Log some events to help debugging
    @session.on("user_speech_started")
    def on_user_speech_started(ev):
        print("👤 User started speaking...")

    @session.on("agent_speech_started")
    def on_agent_speech_started(ev):
        print("🤖 Chikki is responding...")

    # Initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly as Chikki."
    )

    # Keep alive loop
    try:
        while not session._closing:
            await agents.asyncio.sleep(1)
    finally:
        await session.aclose()
        print("--- Session Finished ---")

if __name__ == "__main__":
    agents.cli.run_app(server)
