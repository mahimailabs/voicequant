# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceQuant is a production-ready voice AI inference library built on TurboQuant KV cache compression (~5x compression). It wraps a core compression engine with an OpenAI-compatible server, CLI, benchmark suite, and deployment templates targeting self-hosted LLM serving on NVIDIA GPUs (Modal, RunPod, Docker).

The core math: random orthogonal rotation makes KV cache coordinates ~Gaussian, then Lloyd-Max optimal quantization is applied. Keys get 2-bit MSE + 1-bit QJL bias correction; values get 3-bit MSE quantization.

## Build and Install

```bash
pip install -e .                      # base install (torch + scipy + typer + rich)
pip install -e ".[serve]"             # server: vLLM + FastAPI + uvicorn
pip install -e ".[all]"               # everything including benchmarks + modal
```

## Running Tests

```bash
pytest tests/                          # all tests (CPU-only, no GPU needed)
pytest tests/test_compress.py          # single test file
pytest tests/test_compress.py::test_signs_are_pm1  # single test
```

## CLI

```bash
voicequant serve --model Qwen/Qwen2.5-7B-Instruct-AWQ   # start OpenAI-compatible server
voicequant bench --all --report output.md                 # run all benchmarks
voicequant verify --model MODEL --bits 4                  # validate compression quality
voicequant deploy modal --model MODEL --gpu T4            # generate deployment files
```

## Architecture

The codebase has two layers: a core compression engine and a voice AI application layer built on top.

### Core Compression Engine

Low-level quantization with dual backend (cuTile GPU kernels + PyTorch fallback):

- **`engine.py`** — `TurboQuantEngine`: main entry point for compress/decompress/attention. Every method tries cuTile first, falls back to `_*_pt` methods when unavailable or `_force_pytorch=True`.
- **`codebook.py`** — `LloydMaxCodebook`: solves 1D Lloyd-Max quantization against Gaussian(0, 1/d) at init using scipy.
- **`compress.py`** / **`decompress.py`** / **`attention.py`** — cuTile kernels with `@ct.kernel` decorators, `BLOCK_S=64` tile size.
- **`constants.py`** — Tile sizes (`BLOCK_Q=16`, `BLOCK_KV=64`, `BLOCK_S=64`), `HEAD_DIM=128`.
- **`cache/session.py`** — `CacheSession`: per-session cache lifecycle (compress/truncate/build/clear).

### Voice AI Application Layer

Built on top of the engine with voice-optimized defaults:

- **`cli.py`** — Typer CLI (`voicequant` command) with serve/bench/verify/deploy subcommands. Each subcommand lazily imports its implementation module.
- **`turboquant/wrapper.py`** — `TurboQuantWrapper`: wraps `TurboQuantEngine` with `TurboQuantConfig` (Pydantic model in `turboquant/config.py`). Adds `estimate_capacity()` and `validate_quality()`.
- **`turboquant/validator.py`** — `validate_model()`: per-layer cosine similarity validation with Rich table output.
- **`server/app.py`** — FastAPI app with OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/health`, `/v1/capacity`, `/v1/kv-stats`, `/metrics`). Created via `create_app(ServerConfig)`.
- **`server/engine.py`** — `VoiceQuantEngine`: async vLLM wrapper with TurboQuant config.
- **`server/config.py`** — `ServerConfig` (Pydantic), `VOICE_DEFAULTS` dict, `GPU_CAPACITY_ESTIMATES` lookup table.
- **`benchmarks/`** — 6 scenario classes (multi_turn, concurrent, ttfb, system_prompt, tool_calling, quality) in `scenarios/`, orchestrated by `runner.py`, report via `report.py`. Conversation data in `prompts/conversations/*.json`.
- **`deploy/`** — `modal_deploy.py`, `runpod_handler.py`, `templates/Dockerfile`, `templates/docker-compose.yml`.

### Key Design Patterns

- **Lazy imports**: CLI and `__init__.py` use lazy imports so the base package loads without GPU/vLLM.
- **Dual backend**: Every compression operation checks cuTile first, catches `ImportError`/`RuntimeError`, falls back to PyTorch.
- **Pydantic configs**: `TurboQuantConfig` and `ServerConfig` carry voice-optimized defaults (150 max_tokens, 256 residual window, TQ4, streaming on).
- **Compressed representation**: Keys produce `{indices, k_mse, qjl_signs, vec_norms, residual_norms}`. Values produce `{indices, vec_norms}`.

### Attention Score Formula

`score(q, k) ~ <q, k_mse> + ||r|| * sqrt(pi/2)/m * <S*q, signs>`
