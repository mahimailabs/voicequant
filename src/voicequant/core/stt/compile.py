"""Model download / inventory helpers for STT."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_AVAILABLE_MODELS: list[dict[str, Any]] = [
    {
        "name": "tiny",
        "hf_id": "Systran/faster-whisper-tiny",
        "size_mb": 75,
        "compute_type": "int8",
    },
    {
        "name": "base",
        "hf_id": "Systran/faster-whisper-base",
        "size_mb": 145,
        "compute_type": "int8",
    },
    {
        "name": "small",
        "hf_id": "Systran/faster-whisper-small",
        "size_mb": 475,
        "compute_type": "int8_float16",
    },
    {
        "name": "medium",
        "hf_id": "Systran/faster-whisper-medium",
        "size_mb": 1500,
        "compute_type": "int8_float16",
    },
    {
        "name": "large-v3",
        "hf_id": "Systran/faster-whisper-large-v3",
        "size_mb": 3000,
        "compute_type": "float16",
    },
    {
        "name": "large-v3-turbo",
        "hf_id": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "size_mb": 1620,
        "compute_type": "float16",
    },
    {
        "name": "distil-large-v3",
        "hf_id": "Systran/faster-distil-whisper-large-v3",
        "size_mb": 1500,
        "compute_type": "float16",
    },
]


def list_available_models() -> list[dict[str, Any]]:
    return list(_AVAILABLE_MODELS)


def _hf_id_for(model_name: str) -> str:
    for m in _AVAILABLE_MODELS:
        if m["name"] == model_name or m["hf_id"] == model_name:
            return m["hf_id"]
    return model_name


def download_model(model_name: str, output_dir: str | None = None) -> str:
    """Download a faster-whisper CTranslate2 model and return the local path."""
    hf_id = _hf_id_for(model_name)
    try:
        from faster_whisper import download_model as _dl

        return _dl(hf_id, output_dir=output_dir) if output_dir else _dl(hf_id)
    except ImportError:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=hf_id, local_dir=output_dir)


def get_model_path(model_name: str) -> str | None:
    """Return cached model path if already downloaded, else None."""
    hf_id = _hf_id_for(model_name)
    cache_home = os.environ.get(
        "HUGGINGFACE_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    repo_dir = Path(cache_home) / f"models--{hf_id.replace('/', '--')}"
    return str(repo_dir) if repo_dir.exists() else None
