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

    def __init__(self, observation_space, action_space, device: str = "cpu"):
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
                    loaded = torch.load(weights_path, map_location=self.device, weights_only=True)
                    if isinstance(loaded, dict) and "model_state_dict" in loaded:
                        state_dict = loaded["model_state_dict"]
                    else:
                        state_dict = loaded
                    self.model.load_state_dict(state_dict)
                    print(f"[DoomAgent] Successfully loaded trained Spatial-EgoNet weights from {weights_path.name}")
                except Exception as e:
                    print(f"[DoomAgent] Warning: Failed loading weights ({e}), using default initialization")

            self.model.eval()
            self.hidden = self.model.initial_hidden_state(batch_size=1, device=self.device)
        else:
            self.device = device
            self.model = None
            self.hidden = None

        self.rng = np.random.default_rng(0)
        self.door_cooldown = 0
        self.switch_cooldown = 0
        self.stall_count = 0
        self.last_action_fwd = 0

    def reset(self) -> None:
        """Called before every map. Clear recurrent state and trackers."""
        if TORCH_AVAILABLE and self.model is not None:
            self.hidden = self.model.initial_hidden_state(batch_size=1, device=self.device)
        self.door_cooldown = 0
        self.switch_cooldown = 0
        self.stall_count = 0
        self.last_action_fwd = 0

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

            # Observation-Field Action Masking (Rule 1):
            front_d = float(observation["rays"]["distance"][0])
            hits = observation["rays"]["hit_type"]
            is_door = hits[0] in (2, 3) or hits[1] in (2, 3) or hits[63] in (2, 3)

            # 1. Door interaction: press USE once and wait 10 steps (~1.1s) for door to raise
            # Avoids spamming USE which vanilla DOOM treats as reverse-door command
            if is_door and front_d < 0.22:
                if self.door_cooldown == 0:
                    action[7] = 1
                    self.door_cooldown = 10

            # 2. Physical stall detection
            vel_fwd = float(observation["vitals"][14])
            if self.last_action_fwd and vel_fwd < 0.5:
                self.stall_count += 1
            else:
                self.stall_count = 0

            # 3. Wall / Switch interaction: if pressed against a non-door wall (< 120 map units),
            # pulse USE every 2 steps in case of manual switches or exit panels
            if not is_door and self.stall_count >= 1 and front_d < 0.12:
                if self.switch_cooldown == 0:
                    action[7] = 1
                    self.switch_cooldown = 2

            # 4. Disentanglement: only if stalled against non-door for >= 10 consecutive steps (40 tics)
            if self.stall_count >= 10 and not is_door:
                action[0] = 0
                left_open = float(np.mean(observation["rays"]["distance"][2:16]))
                right_open = float(np.mean(observation["rays"]["distance"][48:62]))
                if left_open >= right_open:
                    action[4] = 1  # Turn left toward open space
                    action[5] = 0
                else:
                    action[5] = 1  # Turn right toward open space
                    action[4] = 0

                # Backpedal if wedged deep in a sharp vertex
                if self.stall_count >= 14:
                    action[1] = 1

            self.last_action_fwd = int(action[0])

            # Manage cooldown timers
            if self.door_cooldown > 0:
                self.door_cooldown -= 1
            if self.switch_cooldown > 0:
                self.switch_cooldown -= 1

            return action
