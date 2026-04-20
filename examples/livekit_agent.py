"""LiveKit voice agent powered by VoiceQuant.

This example creates a real-time voice agent using the LiveKit Agents SDK.
The LLM backend points to a VoiceQuant server (OpenAI-compatible API) which
uses TurboQuant KV cache compression to serve ~5x more concurrent voice
sessions on the same GPU.

Architecture:
    User (WebRTC) -> LiveKit Room -> Agent
        Agent pipeline:
            1. Deepgram STT  — speech-to-text (real-time transcription)
            2. VoiceQuant LLM — compressed KV cache inference via OpenAI plugin
            3. Cartesia TTS  — text-to-speech (low-latency voice synthesis)

Setup:
    1. Start a VoiceQuant server:
        voicequant serve --model Qwen/Qwen2.5-7B-Instruct-AWQ --tq-bits 4

    2. Install agent dependencies:
        pip install "livekit-agents[openai,deepgram,cartesia]>=1.4"

    3. Set environment variables:
        export LIVEKIT_URL=wss://your-project.livekit.cloud
        export LIVEKIT_API_KEY=your-api-key
        export LIVEKIT_API_SECRET=your-api-secret
        export DEEPGRAM_API_KEY=your-deepgram-key
        export CARTESIA_API_KEY=your-cartesia-key
        export VOICEQUANT_BASE_URL=http://localhost:8000/v1  # VoiceQuant server

    4. Run the agent:
        python examples/livekit_agent.py

Persona:
    This example implements "Sofia", a restaurant receptionist for
    "The Golden Fork" who handles reservations, hours, and menu inquiries.
    Swap the system prompt to change the persona entirely.
"""

from __future__ import annotations

import logging
import os

from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("voicequant-agent")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# VoiceQuant LLM configuration
# ---------------------------------------------------------------------------
# The OpenAI plugin connects to any OpenAI-compatible server.
# Point it at your VoiceQuant instance for compressed KV cache inference.
VOICEQUANT_BASE_URL = os.environ.get("VOICEQUANT_BASE_URL", "http://localhost:8000/v1")

# ---------------------------------------------------------------------------
# Restaurant receptionist system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Sofia, the friendly receptionist at The Golden Fork restaurant.

About The Golden Fork:
- Fine dining Italian restaurant in downtown San Francisco
- Open Tuesday-Sunday, 5:00 PM - 10:00 PM (closed Mondays)
- Reservations recommended, walk-ins welcome based on availability
- Private dining room available for parties of 8-20
- Chef's tasting menu: 7 courses, $120 per person
- Full bar with extensive Italian wine list

Your responsibilities:
- Help callers make, modify, or cancel reservations
- Answer questions about hours, menu, dietary accommodations
- Provide directions (we're at 425 Market Street)
- Transfer to the manager for complaints or special requests

Style guidelines:
- Be warm, professional, and concise — this is a phone call
- Keep responses to 1-2 sentences when possible
- Confirm details by repeating them back
- If you don't know something, offer to check with the kitchen/manager
"""


class RestaurantReceptionist(Agent):
    """LiveKit Agent implementing the restaurant receptionist persona.

    Uses VoiceQuant as the LLM backend via the OpenAI-compatible plugin.
    The compressed KV cache means the server can handle many concurrent
    calls simultaneously on a single GPU.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )

    async def on_enter(self) -> None:
        """Called when the agent joins the room. Greet the caller."""
        self.session.generate_reply(
            instructions="Greet the caller warmly as Sofia from The Golden Fork."
        )


async def entrypoint(session: AgentSession) -> None:
    """Agent entrypoint — configure the voice pipeline and start the session.

    This is called by the LiveKit Agents framework when a new participant
    joins the room. It sets up:
        - Deepgram for real-time speech-to-text
        - VoiceQuant (via OpenAI plugin) for LLM inference
        - Cartesia for low-latency text-to-speech
    """

    # --- STT: Deepgram Nova-3 for real-time transcription ---
    stt = deepgram.STT(
        model="nova-3",
        language="en",
    )

    # --- LLM: VoiceQuant server via OpenAI-compatible client ---
    # The key insight: by pointing the OpenAI plugin at VoiceQuant,
    # we get ~5x more concurrent sessions from TurboQuant KV compression
    # with negligible quality loss.
    llm = openai.LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        base_url=VOICEQUANT_BASE_URL,
        api_key="not-needed",  # VoiceQuant doesn't require auth by default
        temperature=0.7,
    )

    # --- TTS: Cartesia Sonic for fast voice synthesis ---
    tts = cartesia.TTS(
        model="sonic",
        voice="sofia",  # Choose a voice that matches the persona
        language="en",
    )

    # Log voice pipeline metrics for monitoring
    @session.on("metrics_collected")
    def on_metrics(event: MetricsCollectedEvent) -> None:
        """Log pipeline latency metrics for performance monitoring."""
        metrics = event.metrics
        logger.info(
            "Pipeline metrics: ttfb=%.0fms, stt=%.0fms, llm=%.0fms, tts=%.0fms",
            metrics.ttfb_ms,
            metrics.stt_duration_ms,
            metrics.llm_ttfb_ms,
            metrics.tts_ttfb_ms,
        )

    # Start the voice agent session
    await session.start(
        agent=RestaurantReceptionist(),
        room_input_options=RoomInputOptions(),
        stt=stt,
        llm=llm,
        tts=tts,
    )


# ---------------------------------------------------------------------------
# Main: register and run the agent worker
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Agent name shown in LiveKit dashboard
            agent_name="voicequant-restaurant",
        ),
    )
