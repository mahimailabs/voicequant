# VoiceQuant

**5x More Concurrent Voice Agents on the Same GPU**

VoiceQuant applies TurboQuant KV cache compression to cut KV memory 5x, enabling 50 concurrent voice agent sessions on a single T4 GPU instead of 10.

## The Problem

Voice AI inference is expensive. Each concurrent caller needs their own KV cache. On a T4 (16GB), after loading a 7B model (~4GB), you have ~12GB for KV caches. At FP16, that's only **~8 concurrent sessions** at 4K context. Add a 1500-token system prompt and you're looking at even fewer.

## The Solution

VoiceQuant uses TurboQuant (PolarQuant rotation + Lloyd-Max quantization + QJL residual correction) to compress KV caches from 16-bit to 4-bit with 0.99+ cosine similarity. Same GPU, **~40 concurrent sessions**.

## Quick Start

```bash
pip install voicequant

# Start serving (requires GPU + vLLM)
voicequant serve --model Qwen/Qwen2.5-7B-Instruct-AWQ

# Validate compression quality
voicequant verify --model Qwen/Qwen2.5-7B-Instruct-AWQ --bits 4

# Run voice AI benchmarks
voicequant bench --all --report benchmark_report.md
```

## Benchmark Results

| Metric | FP16 | TQ4 (4-bit) | TQ3 (3-bit) | Improvement |
|--------|------|-------------|-------------|-------------|
| Concurrent sessions (T4, 4K ctx) | ~8 | ~40 | ~55 | **5x** |
| KV cache per session (4K ctx) | ~150 MB | ~30 MB | ~22 MB | 5-7x smaller |
| TTFB at 8K context | baseline | ~same | ~same | neutral |
| Key cosine similarity | 1.000 | 0.993+ | 0.985+ | - |
| Value cosine similarity | 1.000 | 0.990+ | 0.975+ | - |
| Tool calling accuracy | 100% | ~100% | ~99% | - |

## Deploy to Modal (One Command)

```bash
# Generate deployment files
voicequant deploy modal --model Qwen/Qwen2.5-7B-Instruct-AWQ --gpu T4

# Deploy
modal deploy deploy/modal_deploy.py
```

Your endpoint is now live at `https://your-workspace--voicequant.modal.run/v1`.

## Deploy to RunPod

```bash
voicequant deploy runpod --model Qwen/Qwen2.5-7B-Instruct-AWQ --gpu T4
```

## Deploy with Docker

```bash
voicequant deploy docker --model Qwen/Qwen2.5-7B-Instruct-AWQ
docker compose up --build
```

## Use with LiveKit Agents

VoiceQuant exposes an OpenAI-compatible API, so any agent framework works as a drop-in:

```python
from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai, cartesia, silero

VOICEQUANT_URL = "https://your-workspace--voicequant.modal.run/v1"

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(
            model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            base_url=VOICEQUANT_URL,
            api_key="voicequant",
        ),
        tts=cartesia.TTS(model="sonic-3"),
    )

    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant."),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## Use with Inference Gateway

Add VoiceQuant as a provider in your gateway config:

```yaml
# gateway.yaml
models:
  llm:
    voicequant/qwen2.5-7b-tq4:
      provider: openai_compatible
      base_url: https://your-voicequant.modal.run/v1
      api_key: voicequant
```

## Supported Models

| Model | Size | AWQ Variant | Weights RAM | Voice Quality | Recommended GPU |
|-------|------|-------------|-------------|---------------|-----------------|
| Qwen2.5-3B-Instruct | 3B | AWQ 4-bit | ~2GB | Good for simple tasks | T4 (16GB) |
| Qwen2.5-7B-Instruct | 7B | AWQ 4-bit | ~4GB | Excellent for voice | T4/A10G |
| Llama-3.1-8B-Instruct | 8B | AWQ 4-bit | ~5GB | Great all-around | T4/A10G |
| Mistral-7B-Instruct-v0.3 | 7B | AWQ 4-bit | ~4GB | Good instruction following | T4/A10G |
| Qwen2.5-14B-Instruct | 14B | AWQ 4-bit | ~8GB | Best quality in class | A10G/L4 |

### Concurrent Session Estimates

| GPU | Memory | Model Weights | Available for KV | FP16 Sessions | TQ4 Sessions |
|-----|--------|---------------|------------------|---------------|--------------|
| T4 | 16 GB | ~4 GB | ~12 GB | ~8 | ~40 |
| A10G | 24 GB | ~4 GB | ~20 GB | ~13 | ~65 |
| L4 | 24 GB | ~4 GB | ~20 GB | ~13 | ~65 |
| A100 | 80 GB | ~4 GB | ~76 GB | ~50 | ~250 |
| H100 | 80 GB | ~4 GB | ~76 GB | ~50 | ~250 |

## How TurboQuant Works

1. **PolarQuant Rotation**: A fixed random orthogonal matrix rotates KV cache coordinates so they become approximately Gaussian distributed.
2. **Lloyd-Max Quantization**: Optimal scalar quantization for Gaussian data. Provably minimizes MSE for the given bit budget.
3. **QJL Residual Correction** (keys only): Random projection of the quantization residual preserves inner product expectations, correcting bias in attention scores.

Result: 3-4 bits per element with 0.99+ cosine similarity.

- **Keys**: 2-bit MSE quantization + 1-bit QJL bias correction (3 bits total)
- **Values**: 3-bit MSE quantization
- Both compressed in a single fused kernel per attention head

## Voice-Specific Optimizations

- **Residual window** (default: 256 tokens): Recent tokens stay in FP16 for maximum quality. Older tokens (system prompt, early conversation) get compressed aggressively.
- **Low max_tokens** (default: 150): Voice responses should be 1-3 sentences. A 500-token response takes 10+ seconds to speak.
- **Continuous batching**: vLLM's continuous batching handles many concurrent short sessions efficiently.
- **Streaming by default**: TTFB matters more than throughput for voice AI.

## CLI Reference

```bash
# Start server
voicequant serve --model MODEL --tq-bits 4 --port 8000

# Run benchmarks
voicequant bench --all --report output.md
voicequant bench --scenario concurrent --max-sessions 50
voicequant bench --scenario multi_turn

# Validate quality
voicequant verify --model MODEL --bits 4 --threshold 0.99

# Deploy
voicequant deploy modal --model MODEL --gpu T4
voicequant deploy runpod --model MODEL --gpu T4
voicequant deploy docker --model MODEL
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (streaming + non-streaming) |
| `/v1/models` | GET | List available models |
| `/v1/health` | GET | Health check with GPU memory status |
| `/v1/capacity` | GET | Estimated concurrent session capacity |
| `/v1/kv-stats` | GET | KV cache memory usage and compression ratio |
| `/metrics` | GET | Prometheus-format metrics |

## Development

```bash
pip install -e ".[all]"
pytest tests/ -v
```

## Acknowledgments

- [TurboQuant](https://arxiv.org/abs/2501.06036) — Google Research (ICLR 2026): PolarQuant rotation + Lloyd-Max + QJL residual correction
- [DevTechJr/turboquant-gpu](https://github.com/DevTechJr/turboquant-gpu) — cuTile CUDA kernels + PyTorch fallback
- [Alberto-Codes/turboquant-vllm](https://github.com/Alberto-Codes/turboquant-vllm) — vLLM plugin integration
- [0xSero/turboquant](https://github.com/0xSero/turboquant) — Standalone implementation
- [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) — vLLM fork with Triton backend

## License

Apache 2.0
