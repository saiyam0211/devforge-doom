"""Unit tests for SpatialEgoNet and DoomAgent.

Verifies:
- Circular convolution 360-degree equivariance
- Raycast and Entity Attention tensor shapes & edge cases (all-zero mask)
- Recurrent spatial memory hidden state transitions
- MultiBinary(9) action generation and evaluate_actions()
- Sub-40ms CPU inference latency compliance
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

try:
    import torch
    from submission.model import (
        CircularConv1d,
        EntityAttentionEncoder,
        RaycastEncoder,
        SpatialEgoNet,
        VitalsEncoder,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from submission.agent import DoomAgent


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_circular_conv1d():
    conv = CircularConv1d(in_channels=2, out_channels=4, kernel_size=5)
    x = torch.randn(2, 2, 64)
    out = conv(x)
    assert out.shape == (2, 4, 64)

    # Test circular shift equivariance: roll input by 1, output should be rolled by 1
    x_rolled = torch.roll(x, shifts=1, dims=-1)
    out_rolled = conv(x_rolled)
    expected_rolled = torch.roll(out, shifts=1, dims=-1)
    assert torch.allclose(out_rolled, expected_rolled, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_raycast_encoder():
    encoder = RaycastEncoder(num_hit_types=6, emb_dim=16, out_dim=128)
    dist = torch.rand(4, 64)
    hit_type = torch.randint(0, 6, (4, 64))
    out = encoder(dist, hit_type)
    assert out.shape == (4, 128)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_entity_attention_encoder_edge_cases():
    encoder = EntityAttentionEncoder(num_entity_types=12, type_dim=16, feat_dim=5, out_dim=64)

    # Case 1: Normal entities
    feats = torch.randn(2, 32, 5)
    types = torch.randint(0, 12, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.int8)
    out = encoder(feats, types, mask)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()

    # Case 2: Zero entities visible (all masked)
    mask_zeros = torch.zeros(2, 32, dtype=torch.int8)
    out_empty = encoder(feats, types, mask_zeros)
    assert out_empty.shape == (2, 64)
    assert not torch.isnan(out_empty).any()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_vitals_encoder():
    encoder = VitalsEncoder(in_dim=17, out_dim=64)
    vitals = torch.randn(3, 17)
    out = encoder(vitals)
    assert out.shape == (3, 64)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_spatial_egonet_forward_and_eval():
    model = SpatialEgoNet(hidden_dim=256, num_gru_layers=2)
    b = 2
    obs = {
        "rays": {
            "distance": torch.rand(b, 64),
            "hit_type": torch.randint(0, 6, (b, 64)),
        },
        "visible_entities": {
            "features": torch.randn(b, 32, 5),
            "entity_type": torch.randint(0, 12, (b, 32)),
            "mask": torch.randint(0, 2, (b, 32), dtype=torch.int8),
        },
        "vitals": torch.randn(b, 17),
    }

    hidden = model.initial_hidden_state(batch_size=b)
    actions, log_prob, value, next_hidden = model(obs, hidden, deterministic=False)

    assert actions.shape == (b, 9)
    assert actions.dtype == torch.int8
    assert ((actions == 0) | (actions == 1)).all()
    assert log_prob.shape == (b,)
    assert value.shape == (b,)
    assert next_hidden.shape == (2, b, 256)

    # Test evaluate_actions
    eval_log_prob, eval_val, eval_ent = model.evaluate_actions(obs, hidden, actions)
    assert eval_log_prob.shape == (b,)
    assert eval_val.shape == (b,)
    assert eval_ent.shape == (b,)


def test_doom_agent_interface():
    # Mock observation and action space
    class MockSpace:
        shape = (9,)

    agent = DoomAgent(observation_space=None, action_space=MockSpace(), device="cpu")
    agent.reset()

    # Generate synthetic observation matching spaces.py
    obs = {
        "vitals": np.zeros(17, dtype=np.float32),
        "rays": {
            "distance": np.random.uniform(0.0, 1.0, size=(64,)).astype(np.float32),
            "hit_type": np.random.randint(0, 6, size=(64,), dtype=np.int64),
        },
        "visible_entities": {
            "features": np.random.randn(32, 5).astype(np.float32),
            "entity_type": np.random.randint(0, 12, size=(32,), dtype=np.int64),
            "mask": np.random.randint(0, 2, size=(32,), dtype=np.int8),
        },
    }

    action = agent.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (9,)
    assert action.dtype == np.int8
    assert set(np.unique(action)).issubset({0, 1})


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_agent_latency_budget():
    """Verify inference latency is strictly below the 40ms requirement on single thread."""
    torch.set_num_threads(1)

    class MockSpace:
        shape = (9,)

    agent = DoomAgent(observation_space=None, action_space=MockSpace(), device="cpu")
    agent.reset()

    obs = {
        "vitals": np.zeros(17, dtype=np.float32),
        "rays": {
            "distance": np.random.uniform(0.0, 1.0, size=(64,)).astype(np.float32),
            "hit_type": np.random.randint(0, 6, size=(64,), dtype=np.int64),
        },
        "visible_entities": {
            "features": np.random.randn(32, 5).astype(np.float32),
            "entity_type": np.random.randint(0, 12, size=(32,), dtype=np.int64),
            "mask": np.random.randint(0, 2, size=(32,), dtype=np.int8),
        },
    }

    # Warmup
    for _ in range(5):
        agent.act(obs)

    # Benchmark 50 steps
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        agent.act(obs)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000.0
    p95_ms = np.percentile(times, 95) * 1000.0
    print(f"\n[Benchmark] Avg step: {avg_ms:.2f}ms | 95th percentile: {p95_ms:.2f}ms")
    assert avg_ms < 40.0, f"Average step latency ({avg_ms:.2f}ms) exceeds 40ms limit"
