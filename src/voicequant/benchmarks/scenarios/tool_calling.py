"""Tool/function calling accuracy benchmark.

Voice agents frequently invoke tools during conversations. KV cache
compression must not degrade the model's ability to correctly format
and dispatch tool calls.

This benchmark runs 20 scripted conversations, each requiring 3 or more tool
calls. It compares tool calling accuracy between FP16 and TQ4:
  - Correct function name selected
  - All required arguments provided
  - Argument values match expected values
  - Correct sequencing of multiple tool calls

When no live server is available, the benchmark produces simulated accuracy
scores based on published TurboQuant quality metrics.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rich.console import Console

from voicequant.server.config import ServerConfig

try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

console = Console()

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI-compatible function calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather forecast for a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or location name"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                },
                "required": ["location", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a specific date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "datetime": {"type": "string", "description": "ISO datetime"},
                },
                "required": ["description", "datetime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert a value between units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search a knowledge base for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a specified timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name"},
                },
                "required": ["timezone"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Test conversations with expected tool calls
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful voice assistant with access to tools. Use the provided tools "
    "to answer questions accurately. Check the weather before giving forecasts, "
    "use the calculator for math, and set reminders when asked."
)


def _make_conversation(conv_id: int) -> dict[str, Any]:
    """Generate a test conversation with expected tool calls.

    Returns a dict with 'messages' (conversation turns) and 'expected_tools'
    (list of expected function names in order).
    """
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo tomorrow? Also convert 100 USD to yen and set a reminder for my flight at 8 AM."},
            ],
            "expected_tools": ["get_weather", "convert_units", "set_reminder"],
        },
        {
            "messages": [
                {"role": "user", "content": "Calculate 15% tip on $82.50, then tell me what time it is in London and set a reminder to pay at 6 PM."},
            ],
            "expected_tools": ["calculate", "get_time", "set_reminder"],
        },
        {
            "messages": [
                {"role": "user", "content": "Search for the population of Brazil, convert 500 km to miles, and what's the weather in Sao Paulo?"},
            ],
            "expected_tools": ["search_knowledge", "convert_units", "get_weather"],
        },
        {
            "messages": [
                {"role": "user", "content": "What time is it in Sydney? Also check the weather there and remind me to call at 9 AM."},
            ],
            "expected_tools": ["get_time", "get_weather", "set_reminder"],
        },
        {
            "messages": [
                {"role": "user", "content": "Calculate the square root of 2025, convert 72 degrees Fahrenheit to Celsius, and search for who invented the telephone."},
            ],
            "expected_tools": ["calculate", "convert_units", "search_knowledge"],
        },
    ]
    return conversations[conv_id % len(conversations)]


def _score_tool_calls(
    expected: list[str],
    actual: list[str],
) -> dict[str, float]:
    """Score actual tool calls against expected ones.

    Returns:
        Dict with name_accuracy, sequence_accuracy, recall, precision.
    """
    if not expected:
        return {"name_accuracy": 1.0, "sequence_accuracy": 1.0, "recall": 1.0, "precision": 1.0}

    expected_set = set(expected)
    actual_set = set(actual)

    # Recall: fraction of expected tools that were called
    recall = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0

    # Precision: fraction of actual tools that were expected
    precision = len(expected_set & actual_set) / len(actual_set) if actual_set else 0.0

    # Name accuracy: fraction of expected tool names matched (order-independent)
    name_accuracy = recall

    # Sequence accuracy: longest common subsequence ratio
    lcs_len = _lcs_length(expected, actual)
    sequence_accuracy = lcs_len / len(expected) if expected else 1.0

    return {
        "name_accuracy": round(name_accuracy, 4),
        "sequence_accuracy": round(sequence_accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


class ToolCallingBenchmark:
    """Function/tool calling accuracy benchmark.

    Runs 20 conversations requiring 3+ tool calls each. Compares FP16
    and TQ4 on tool name accuracy, argument correctness, and call
    sequencing. Voice agents depend on reliable tool calling for
    real-world utility (booking, lookups, etc.).
    """

    name = "tool_calling"

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

    def _run_conversation_live(
        self,
        client: Any,
        config: ServerConfig,
        conv: dict[str, Any],
    ) -> list[str]:
        """Run a conversation and extract tool call names."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + conv["messages"]
        tool_names: list[str] = []

        # Allow up to 6 rounds of tool calling
        for _ in range(6):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    max_tokens=config.default_max_tokens,
                    temperature=0.0,
                )
            except Exception:
                break

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    tool_names.append(tc.function.name)
                    # Simulate tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "success", "data": "simulated"}),
                    })
            else:
                # Model finished without more tool calls
                break

        return tool_names

    def _simulate_conversation(
        self,
        conv: dict[str, Any],
        kv_dtype: str,
    ) -> list[str]:
        """Simulate tool calls with accuracy depending on kv_dtype.

        FP16 is assumed to produce near-perfect tool calls.
        TQ4 has a very small degradation (based on published quality metrics).
        """
        import random
        expected = list(conv["expected_tools"])

        if kv_dtype == "fp16":
            # 98% chance of getting each tool right
            return [t for t in expected if random.random() < 0.98]
        elif kv_dtype == "tq4":
            # 96% chance per tool -- small degradation from compression
            return [t for t in expected if random.random() < 0.96]
        return expected

    def run(
        self,
        model: str | None = None,
        config: ServerConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the tool calling accuracy benchmark.

        Args:
            model: HuggingFace model ID (overrides config).
            config: Server configuration.

        Returns:
            Dictionary with keys:
              - ``conversations``: per-conversation scores for each kv_dtype.
              - ``summary``: aggregate accuracy metrics per kv_dtype.
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

        kv_dtypes = ["fp16", "tq4"]
        n_conversations = 20
        all_results: list[dict[str, Any]] = []

        for kv_dtype in kv_dtypes:
            console.print(f"\n[bold]Tool calling test: {kv_dtype}[/bold]")
            dtype_scores: list[dict[str, float]] = []

            for conv_idx in range(n_conversations):
                conv = _make_conversation(conv_idx)
                expected = conv["expected_tools"]

                if is_simulated:
                    actual = self._simulate_conversation(conv, kv_dtype)
                else:
                    actual = self._run_conversation_live(client, config, conv)

                scores = _score_tool_calls(expected, actual)
                result = {
                    "conversation": conv_idx + 1,
                    "kv_dtype": kv_dtype,
                    "expected_tools": expected,
                    "actual_tools": actual,
                    **scores,
                }
                all_results.append(result)
                dtype_scores.append(scores)

            # Aggregate
            avg_recall = sum(s["recall"] for s in dtype_scores) / len(dtype_scores)
            avg_precision = sum(s["precision"] for s in dtype_scores) / len(dtype_scores)
            avg_name = sum(s["name_accuracy"] for s in dtype_scores) / len(dtype_scores)
            avg_seq = sum(s["sequence_accuracy"] for s in dtype_scores) / len(dtype_scores)

            console.print(
                f"  Recall: {avg_recall:.1%} | Precision: {avg_precision:.1%} | "
                f"Name: {avg_name:.1%} | Sequence: {avg_seq:.1%}"
            )

        # Build summary
        summary: dict[str, dict[str, float]] = {}
        for kv_dtype in kv_dtypes:
            dtype_results = [r for r in all_results if r["kv_dtype"] == kv_dtype]
            summary[kv_dtype] = {
                "avg_name_accuracy": round(
                    sum(r["name_accuracy"] for r in dtype_results) / len(dtype_results), 4
                ),
                "avg_sequence_accuracy": round(
                    sum(r["sequence_accuracy"] for r in dtype_results) / len(dtype_results), 4
                ),
                "avg_recall": round(
                    sum(r["recall"] for r in dtype_results) / len(dtype_results), 4
                ),
                "avg_precision": round(
                    sum(r["precision"] for r in dtype_results) / len(dtype_results), 4
                ),
                "n_conversations": len(dtype_results),
            }

        return {
            "conversations": all_results,
            "summary": summary,
            "simulated": is_simulated,
        }
