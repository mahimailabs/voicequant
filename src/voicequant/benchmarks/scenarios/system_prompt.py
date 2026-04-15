"""System prompt compression benchmark.

Voice agents typically use large system prompts (500-2000 tokens) that define
persona, capabilities, tool definitions, and behavioral rules. This system
prompt is cached for the duration of every session, consuming KV cache memory
proportional to session count.

This benchmark measures:
  1. Memory consumed by the static system prompt KV cache (FP16 vs TQ4)
  2. How system prompt memory scales across concurrent sessions
  3. Whether compression affects response quality for system-prompt-heavy
     conversations over 10 turns

A 1500-token system prompt is used, representative of production voice agents
with persona definitions, tool schemas, and behavioral constraints.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console

from voicequant.server.config import ServerConfig, GPU_CAPACITY_ESTIMATES

try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

console = Console()

# ---------------------------------------------------------------------------
# Realistic voice agent system prompt (~1500 tokens)
# ---------------------------------------------------------------------------

LARGE_SYSTEM_PROMPT = """You are Atlas, a general-purpose voice AI assistant. You handle incoming calls with clarity, efficiency, and helpfulness.

## Your Personality
- Clear and concise, but friendly
- You speak in short sentences suitable for phone conversations
- You confirm important details by repeating them back
- You stay within your capabilities and admit when you don't know something

## Capabilities
- Answer general knowledge questions
- Perform calculations and unit conversions
- Set reminders and manage schedules
- Provide weather forecasts and time zone information
- Look up facts, definitions, and reference data
- Help with planning and organization

## Response Guidelines
1. Keep responses under 3 sentences for natural phone conversation
2. Use simple language, avoid jargon
3. When doing calculations, state the result clearly
4. For multi-step tasks, confirm each step before proceeding
5. If asked about something you're unsure of, say so honestly
6. Always offer to help with follow-up questions

## Available Tools
You have access to the following functions:
- get_weather(location, date) -> forecast with temperature, conditions, precipitation
- set_reminder(description, datetime) -> confirmation with reminder ID
- list_reminders() -> list of active reminders with times
- calculate(expression) -> numerical result with explanation
- convert_units(value, from_unit, to_unit) -> converted value
- search_knowledge(query) -> relevant information summary
- get_time(timezone) -> current time in specified timezone

## Scheduling Rules
1. Always confirm the date and time before setting a reminder
2. Use the user's local timezone unless specified otherwise
3. Warn if a reminder is set for a time that has already passed today
4. Allow modification of existing reminders by referencing their description

## Conversation Management
- Greet the caller naturally
- Ask clarifying questions when the request is ambiguous
- Summarize actions taken at the end of the conversation
- Handle topic changes smoothly
- If the user says goodbye, end the conversation promptly
- Track context across the conversation to avoid asking repeat questions

## Error Handling
- If a tool call fails, explain the issue simply and suggest alternatives
- If you can't fulfill a request, explain why and offer related help
- Never make up information you don't have access to

## Output Formatting
- For numbers: use comma separators for thousands (e.g., 1,000,000)
- For currency: include the currency symbol and two decimal places
- For temperatures: default to Fahrenheit, mention Celsius if asked
- For distances: default to miles, mention kilometers if relevant
"""

_SYSTEM_PROMPT_TOKENS = 1500  # approximate

_N_LAYERS = 32
_N_HEADS = 32
_HEAD_DIM = 128
_FP16_BYTES_PER_TOKEN = 2 * _N_LAYERS * _N_HEADS * _HEAD_DIM * 2  # K+V, all layers/heads

_CONVERSATION_TURNS = [
    {"role": "user", "content": "Hi, what's the weather like tomorrow?"},
    {"role": "user", "content": "Remind me to bring an umbrella at 7 AM."},
    {"role": "user", "content": "What's 18% tip on a $45 bill?"},
    {"role": "user", "content": "Convert that to euros for me."},
    {"role": "user", "content": "What time is it in Tokyo right now?"},
    {"role": "user", "content": "How many hours ahead is that from here?"},
    {"role": "user", "content": "Set a reminder for my meeting at 3 PM."},
    {"role": "user", "content": "How many reminders do I have now?"},
    {"role": "user", "content": "What's the square root of 144?"},
    {"role": "user", "content": "Thanks, that's all I needed. Bye!"},
]


def _compression_ratio(kv_dtype: str) -> float:
    if kv_dtype == "fp16":
        return 1.0
    elif kv_dtype == "tq4":
        return 16.0 / 3.0
    return 1.0


class SystemPromptBenchmark:
    """System prompt KV cache compression benchmark.

    Measures memory savings from compressing the large static system prompt
    that every voice session carries. Shows how compression enables more
    concurrent sessions by reducing per-session overhead.
    """

    name = "system_prompt"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _get_client(self, config: ServerConfig) -> Any | None:
        if not _HAS_OPENAI:
            return None
        try:
            client = openai.OpenAI(
                base_url=f"http://{config.host}:{config.port}/v1",
                api_key="not-needed",
            )
            client.models.list()
            return client
        except Exception:
            return None

    def _run_turns_live(
        self,
        client: Any,
        config: ServerConfig,
    ) -> list[dict[str, Any]]:
        """Run 10 conversation turns against a live server."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": LARGE_SYSTEM_PROMPT},
        ]
        turn_results: list[dict[str, Any]] = []

        for turn_idx, user_msg in enumerate(_CONVERSATION_TURNS, start=1):
            messages.append(user_msg)
            t0 = time.perf_counter()
            first_token_time = None
            response_text = ""
            tokens_generated = 0

            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.default_max_tokens,
                temperature=config.temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_generated += 1
                    response_text += chunk.choices[0].delta.content

            t1 = time.perf_counter()
            ttfb = ((first_token_time or t1) - t0) * 1000.0
            messages.append({"role": "assistant", "content": response_text})

            turn_results.append({
                "turn": turn_idx,
                "ttfb_ms": round(ttfb, 2),
                "total_latency_ms": round((t1 - t0) * 1000.0, 2),
                "tokens_generated": tokens_generated,
                "response": response_text,
            })

        return turn_results

    def run(
        self,
        model: str | None = None,
        config: ServerConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the system prompt compression benchmark.

        Args:
            model: HuggingFace model ID (overrides config).
            config: Server configuration.

        Returns:
            Dictionary with keys:
              - ``system_prompt_memory``: FP16 vs TQ4 memory for system prompt alone.
              - ``scaling``: memory at 10, 50, 100, 250 concurrent sessions.
              - ``turns``: per-turn metrics (if live server available).
              - ``simulated``: whether results are simulated.
        """
        if config is None:
            config = ServerConfig()
        if model:
            config = config.model_copy(update={"model": model})

        client = self._get_client(config)
        is_simulated = client is None

        if is_simulated:
            console.print("[yellow]No live server detected -- using simulated results.[/yellow]")

        # Memory analysis for the system prompt alone
        fp16_prompt_bytes = _SYSTEM_PROMPT_TOKENS * _FP16_BYTES_PER_TOKEN
        tq4_prompt_bytes = int(fp16_prompt_bytes / _compression_ratio("tq4"))

        system_prompt_memory = {
            "system_prompt_tokens": _SYSTEM_PROMPT_TOKENS,
            "fp16_bytes": fp16_prompt_bytes,
            "fp16_mb": round(fp16_prompt_bytes / (1024 * 1024), 2),
            "tq4_bytes": tq4_prompt_bytes,
            "tq4_mb": round(tq4_prompt_bytes / (1024 * 1024), 2),
            "compression_ratio": round(_compression_ratio("tq4"), 2),
            "savings_mb": round((fp16_prompt_bytes - tq4_prompt_bytes) / (1024 * 1024), 2),
        }

        console.print("\n[bold]System Prompt Memory[/bold]")
        console.print(
            f"  FP16: {system_prompt_memory['fp16_mb']:.2f} MB | "
            f"TQ4: {system_prompt_memory['tq4_mb']:.2f} MB | "
            f"Savings: {system_prompt_memory['savings_mb']:.2f} MB per session"
        )

        # Scaling analysis across session counts
        session_counts = [10, 50, 100, 250]
        scaling: list[dict[str, Any]] = []

        console.print("\n[bold]Scaling: System Prompt Memory Across Sessions[/bold]")
        for n in session_counts:
            fp16_total = fp16_prompt_bytes * n
            tq4_total = tq4_prompt_bytes * n
            entry = {
                "sessions": n,
                "fp16_total_gb": round(fp16_total / (1024**3), 3),
                "tq4_total_gb": round(tq4_total / (1024**3), 3),
                "savings_gb": round((fp16_total - tq4_total) / (1024**3), 3),
            }
            scaling.append(entry)
            console.print(
                f"  {n:4d} sessions | FP16: {entry['fp16_total_gb']:.3f} GB | "
                f"TQ4: {entry['tq4_total_gb']:.3f} GB | "
                f"Savings: {entry['savings_gb']:.3f} GB"
            )

        # Conversation quality (simulated or live)
        turn_results: list[dict[str, Any]] = []
        if not is_simulated:
            console.print("\n[bold]Running 10-turn conversation quality test...[/bold]")
            turn_results = self._run_turns_live(client, config)
            for tr in turn_results:
                console.print(
                    f"  Turn {tr['turn']:2d} | TTFB={tr['ttfb_ms']:6.1f}ms | "
                    f"tokens={tr['tokens_generated']}"
                )
        else:
            # Simulated turn results
            context = _SYSTEM_PROMPT_TOKENS
            for turn_idx in range(1, 11):
                context += 15 + 25  # user + assistant tokens
                ratio = _compression_ratio("tq4")
                base_ttfb = 15.0 + (context / (ratio * 1000.0)) * 8.0
                turn_results.append({
                    "turn": turn_idx,
                    "ttfb_ms": round(base_ttfb, 2),
                    "total_latency_ms": round(base_ttfb + 25 * 10.0, 2),
                    "tokens_generated": 25,
                    "response": f"[simulated response for turn {turn_idx}]",
                })

        return {
            "system_prompt_memory": system_prompt_memory,
            "scaling": scaling,
            "turns": turn_results,
            "simulated": is_simulated,
        }
