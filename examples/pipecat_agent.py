"""PipeCat voice agent powered by VoiceQuant.

This example creates a real-time voice agent using the PipeCat SDK.
The LLM service points to a VoiceQuant server (OpenAI-compatible API)
which uses TurboQuant KV cache compression to serve ~5x more concurrent
voice sessions on the same GPU.

Architecture:
    User (WebRTC) -> Daily room -> PipeCat pipeline
        Pipeline frames flow:
            Audio In -> Deepgram STT -> User Aggregator
            -> VoiceQuant LLM -> Cartesia TTS -> Audio Out

Setup:
    1. Start a VoiceQuant server:
        voicequant serve --model Qwen/Qwen2.5-7B-Instruct-AWQ --tq-bits 4

    2. Install PipeCat dependencies:
        pip install "pipecat-ai[daily,deepgram,cartesia,openai]"

    3. Set environment variables:
        export DAILY_API_KEY=your-daily-key
        export DEEPGRAM_API_KEY=your-deepgram-key
        export CARTESIA_API_KEY=your-cartesia-key
        export VOICEQUANT_BASE_URL=http://localhost:8000/v1

    4. Run the agent:
        python examples/pipecat_agent.py

Persona:
    Same restaurant receptionist as the LiveKit example — "Sofia" at
    The Golden Fork. Demonstrates the same use case with PipeCat patterns.
"""

from __future__ import annotations

import asyncio
import os
import sys

from loguru import logger
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOICEQUANT_BASE_URL = os.environ.get("VOICEQUANT_BASE_URL", "http://localhost:8000/v1")
DAILY_API_KEY = os.environ.get("DAILY_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")

# Restaurant receptionist system prompt
SYSTEM_PROMPT = """\
You are Sofia, the friendly receptionist at The Golden Fork restaurant.

About The Golden Fork:
- Fine dining Italian restaurant in downtown San Francisco
- Open Tuesday-Sunday, 5:00 PM - 10:00 PM (closed Mondays)
- Reservations recommended, walk-ins welcome based on availability
- Private dining room available for parties of 8-20
- Chef's tasting menu: 7 courses, $120 per person

Your responsibilities:
- Help callers make, modify, or cancel reservations
- Answer questions about hours, menu, dietary accommodations
- Provide directions (425 Market Street, San Francisco)

Style: warm, professional, concise (1-2 sentences per response).
"""


async def main() -> None:
    """Set up and run the PipeCat voice agent pipeline.

    The pipeline connects:
        Transport (audio) -> STT -> LLM -> TTS -> Transport (audio)

    The LLM service uses OpenAILLMService pointed at the VoiceQuant
    server, which provides OpenAI-compatible chat completions with
    TurboQuant compressed KV cache.
    """

    # --- Transport: Daily WebRTC room ---
    # Creates a room automatically; in production you'd use a pre-created room URL
    transport = DailyTransport(
        room_url=os.environ.get("DAILY_ROOM_URL", ""),
        token=os.environ.get("DAILY_ROOM_TOKEN", ""),
        bot_name="Sofia",
        params=DailyParams(
            api_key=DAILY_API_KEY,
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer_params=DailyParams.VADAnalyzerParams(
                stop_secs=0.3,      # Faster end-of-speech detection for voice
            ),
            transcription_enabled=False,  # We use Deepgram directly
        ),
    )

    # --- STT: Deepgram Nova-3 ---
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model="nova-3",
        language="en",
    )

    # --- LLM: VoiceQuant via OpenAI-compatible API ---
    # This is where VoiceQuant plugs in. The OpenAILLMService sends
    # standard chat completion requests to the VoiceQuant server,
    # which handles inference with compressed KV cache internally.
    llm = OpenAILLMService(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        base_url=VOICEQUANT_BASE_URL,
        api_key="not-needed",  # VoiceQuant doesn't require auth by default
    )

    # --- TTS: Cartesia Sonic ---
    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        model_id="sonic-english",
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Warm female voice
    )

    # --- Conversation context ---
    # PipeCat manages the message history through an OpenAI-compatible context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Pipeline: wire everything together ---
    # Frames flow left to right through the pipeline:
    #   transport.input -> stt -> user_aggregator -> llm -> tts -> transport.output
    pipeline = Pipeline(
        [
            transport.input(),                  # Audio frames from WebRTC
            stt,                                # Speech -> text frames
            context_aggregator.user(),          # Collect user text into context
            llm,                                # Generate response via VoiceQuant
            tts,                                # Text -> speech frames
            transport.output(),                 # Audio frames back to WebRTC
            context_aggregator.assistant(),     # Track assistant responses in context
        ]
    )

    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,       # Let callers interrupt (barge-in)
            enable_metrics=True,            # Track pipeline latency
            enable_usage_metrics=True,      # Track token usage
        ),
    )

    # --- Event handlers ---

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport_inst, participant) -> None:
        """When a caller joins, send the initial greeting."""
        await task.queue_frames(
            [
                LLMMessagesFrame(
                    [
                        {
                            "role": "system",
                            "content": "Greet the caller warmly as Sofia from The Golden Fork. "
                            "Ask how you can help them today.",
                        }
                    ]
                )
            ]
        )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport_inst, participant, reason) -> None:
        """Clean up when the caller leaves."""
        await task.queue_frames([EndFrame()])

    # --- Run the pipeline ---
    runner = PipelineRunner()

    logger.info("Starting PipeCat voice agent with VoiceQuant LLM backend")
    logger.info(f"VoiceQuant server: {VOICEQUANT_BASE_URL}")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
