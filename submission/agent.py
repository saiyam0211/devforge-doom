"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
Implements Spatial-EgoNet inference with single-thread optimization and recurrent state management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"

try:
    import torch
    try:
        from model import SpatialEgoNet
    except ImportError:
        from submission.model import SpatialEgoNet

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DoomAgent:
    """Production DoomAgent executing Spatial-EgoNet neural policy."""

    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space

        # Enforce single CPU thread as specified in EVALUATION.md §Hardware
        if TORCH_AVAILABLE:
            torch.set_num_threads(1)
            if device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
            else:
                self.device = device

            self.model = SpatialEgoNet(hidden_dim=256, num_gru_layers=2).to(self.device)

            # Load checkpoint if available
            weights_path = WEIGHTS_DIR / "agent_weights.pt"
            if weights_path.is_file():
                try:
                    state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state_dict)
                except Exception:
                    pass  # Fall back to initialized parameters if weights file is damaged

            self.model.eval()
            self.hidden = self.model.initial_hidden_state(batch_size=1, device=self.device)
        else:
            self.device = device
            self.model = None
            self.hidden = None

        self.rng = np.random.default_rng(0)
        self.use_cooldown = 0

    def reset(self) -> None:
        """Called before every map. Clear recurrent state and trackers."""
        if TORCH_AVAILABLE and self.model is not None:
            self.hidden = self.model.initial_hidden_state(batch_size=1, device=self.device)
        self.use_cooldown = 0

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int array of shape action_space.shape with values in {0, 1}."""
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback if torch is not available during dry-run inspection
            return self.rng.integers(0, 2, size=self.action_space.shape, dtype=np.int8)

        # Convert observation dictionary to batched PyTorch tensors
        with torch.inference_mode():
            obs_tensor: Dict[str, Any] = {
                "rays": {
                    "distance": torch.from_numpy(observation["rays"]["distance"]).unsqueeze(0).to(self.device),
                    "hit_type": torch.from_numpy(observation["rays"]["hit_type"]).unsqueeze(0).to(self.device),
                },
                "visible_entities": {
                    "features": torch.from_numpy(observation["visible_entities"]["features"]).unsqueeze(0).to(self.device),
                    "entity_type": torch.from_numpy(observation["visible_entities"]["entity_type"]).unsqueeze(0).to(self.device),
                    "mask": torch.from_numpy(observation["visible_entities"]["mask"]).unsqueeze(0).to(self.device),
                },
                "vitals": torch.from_numpy(observation["vitals"]).unsqueeze(0).to(self.device),
            }

            # Forward pass through recurrent spatial policy
            action_tensor, _, _, next_hidden = self.model(obs_tensor, self.hidden, deterministic=True)
            self.hidden = next_hidden

            action = action_tensor[0].cpu().numpy().astype(np.int8)

            # Prevent DOOM USE spamming: pulse USE and enforce a 12-tick cooldown
            if action[7] == 1:
                if self.use_cooldown > 0:
                    action[7] = 0
                    self.use_cooldown -= 1
                else:
                    self.use_cooldown = 12
            elif self.use_cooldown > 0:
                self.use_cooldown -= 1

            return action
