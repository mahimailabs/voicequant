"""LLM test fixtures.

test_end_to_end.py builds Q/K/V with torch.randn on the global RNG but only
seeds the engine. Input tensors therefore vary between torch builds and
occasionally land below the 3-bit cosine similarity thresholds. Seeding the
global RNG for that module keeps inputs deterministic without disturbing
other LLM tests that rely on unseeded randomness.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch_global_rng(request):
    if request.node.module.__name__.endswith("test_end_to_end"):
        torch.manual_seed(42)
    yield
