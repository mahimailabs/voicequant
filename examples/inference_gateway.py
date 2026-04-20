"""VoiceQuant as a LiveKit Inference Gateway provider.

LiveKit's Inference Gateway allows you to register custom LLM providers
that agents can use transparently. This example shows how to configure
VoiceQuant as a gateway provider so any LiveKit agent can use it without
changing their code.

Architecture:
    LiveKit Agent -> Inference Gateway -> VoiceQuant Server
                                            (TurboQuant KV compression)

The gateway acts as a proxy: agents request "openai/gpt-4o" or any model
name, and the gateway routes it to VoiceQuant's OpenAI-compatible API.

Setup:
    1. Start a VoiceQuant server:
        voicequant serve --model Qwen/Qwen2.5-7B-Instruct-AWQ --tq-bits 4

    2. Create the gateway configuration (gateway.yaml):
        See the GATEWAY_YAML constant below, or create a file with that content.

    3. Start the LiveKit Inference Gateway:
        lk-inference-gateway --config gateway.yaml

    4. Run this example agent:
        python examples/inference_gateway.py

Gateway Configuration (gateway.yaml):
    The YAML below registers VoiceQuant as an OpenAI-compatible provider.
    Agents that request any model routed through this gateway will
    automatically use VoiceQuant's compressed inference.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Gateway YAML configuration
# ---------------------------------------------------------------------------
# Save this as gateway.yaml alongside your LiveKit Inference Gateway binary.
# Adjust the base_url to point to your VoiceQuant deployment.

GATEWAY_YAML = """\
# gateway.yaml — LiveKit Inference Gateway configuration for VoiceQuant
#
# This routes LLM requests through VoiceQuant's OpenAI-compatible API,
# giving agents access to TurboQuant KV cache compression transparently.
#
# Start: lk-inference-gateway --config gateway.yaml

log_level: info

# Listen address for the gateway
listen_address: 0.0.0.0:8089

providers:
  # VoiceQuant provider — OpenAI-compatible endpoint
  - name: voicequant
    type: openai
    base_url: http://localhost:8000/v1
    api_key: "not-needed"  # VoiceQuant doesn't require auth by default

    # Models served by this provider. Agents request these model names
    # and the gateway routes to VoiceQuant automatically.
    models:
      - name: "voicequant/qwen-7b"
        upstream_model: "Qwen/Qwen2.5-7B-Instruct-AWQ"
        max_tokens: 150
        # Voice-optimized defaults
        default_params:
          temperature: 0.7
          max_tokens: 150

    # Rate limiting (optional)
    rate_limit:
      requests_per_minute: 600
      tokens_per_minute: 100000

    # Health check
    health_check:
      endpoint: /v1/health
      interval: 30s

  # You can add a fallback provider for overflow or A/B testing
  # - name: openai-fallback
  #   type: openai
  #   base_url: https://api.openai.com/v1
  #   api_key: ${OPENAI_API_KEY}
  #   models:
  #     - name: "gpt-4o-mini"

# Routing rules
routing:
  # Default: route all requests to VoiceQuant
  default_provider: voicequant

  # Route specific models to specific providers
  rules:
    - model: "voicequant/*"
      provider: voicequant
    # - model: "gpt-*"
    #   provider: openai-fallback
"""


def print_gateway_config() -> None:
    """Print the gateway YAML configuration to stdout.

    Pipe this to a file:
        python examples/inference_gateway.py --print-config > gateway.yaml
    """
    print(GATEWAY_YAML)


# ---------------------------------------------------------------------------
# Example: Agent using VoiceQuant through the Inference Gateway
# ---------------------------------------------------------------------------

async def entrypoint(session) -> None:
    """LiveKit Agent entrypoint using the Inference Gateway.

    When the Inference Gateway is running, agents use it as their LLM
    backend. The gateway transparently routes requests to VoiceQuant.
    No special client configuration is needed beyond pointing to the
    gateway URL.
    """
    from livekit.agents import Agent, RoomInputOptions
    from livekit.plugins import cartesia, deepgram, openai

    # The gateway URL — agents connect here instead of directly to VoiceQuant.
    # The gateway handles routing, load balancing, and health checks.
    gateway_url = os.environ.get("LK_INFERENCE_GATEWAY_URL", "http://localhost:8089/v1")

    # --- LLM: Connect through the Inference Gateway ---
    # Request "voicequant/qwen-7b" — the gateway maps this to the actual
    # model on the VoiceQuant server via the routing rules in gateway.yaml.
    llm = openai.LLM(
        model="voicequant/qwen-7b",
        base_url=gateway_url,
        api_key="not-needed",
        temperature=0.7,
    )

    # --- STT and TTS (same as direct integration) ---
    stt = deepgram.STT(model="nova-3", language="en")
    tts = cartesia.TTS(model="sonic", voice="sofia", language="en")

    # --- Agent ---
    agent = Agent(
        instructions=(
            "You are a helpful voice assistant. Keep responses brief and "
            "conversational, ideally 1-2 sentences. You are powered by "
            "VoiceQuant compressed inference through the LiveKit Inference Gateway."
        ),
    )

    await session.start(
        agent=agent,
        room_input_options=RoomInputOptions(),
        stt=stt,
        llm=llm,
        tts=tts,
    )


# ---------------------------------------------------------------------------
# Example: Direct Python client using the gateway
# ---------------------------------------------------------------------------

def example_gateway_client() -> None:
    """Demonstrate calling VoiceQuant through the gateway using httpx.

    This shows how any OpenAI-compatible client can use the gateway
    without knowing about VoiceQuant at all.
    """
    import httpx

    gateway_url = os.environ.get("LK_INFERENCE_GATEWAY_URL", "http://localhost:8089/v1")

    # Standard OpenAI-compatible request — the gateway routes it to VoiceQuant
    response = httpx.post(
        f"{gateway_url}/chat/completions",
        json={
            "model": "voicequant/qwen-7b",
            "messages": [
                {"role": "system", "content": "You are a helpful voice assistant."},
                {"role": "user", "content": "What are your hours today?"},
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "stream": False,
        },
        timeout=30.0,
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    print(f"Response: {content}")
    print(f"Usage: {result.get('usage', {})}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--print-config" in sys.argv:
        # Print the gateway YAML config
        print_gateway_config()
    elif "--client" in sys.argv:
        # Run the direct client example
        example_gateway_client()
    else:
        # Run as a LiveKit agent
        from livekit.agents import WorkerOptions, cli

        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name="voicequant-gateway-agent",
            ),
        )
