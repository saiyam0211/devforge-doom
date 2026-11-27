"""Standalone inference for spatial or visual-spatial checkpoints."""

from pathlib import Path

import torch

try:
    from model import SpatialEgoNet
except ImportError:
    from submission.model import SpatialEgoNet


class DoomAgent:
    def __init__(self, observation_space, action_space, device="cpu", checkpoint=None):
        torch.set_num_threads(1)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        path = Path(checkpoint) if checkpoint is not None else Path(__file__).parent / "weights/agent_weights.pt"
        weights = torch.load(path, map_location="cpu", weights_only=True)
        state = weights.get("model_state_dict", weights)
        self.model = SpatialEgoNet(use_rgb=weights.get("use_rgb", False)).to(device)
        self.model.load_state_dict(state)
        self.generator = torch.Generator(device=device) if weights.get("sample_actions", False) else None
        self.model.eval()
        self.reset()

    def reset(self):
        if self.generator is not None:
            self.generator.manual_seed(0)
        self.hidden = self.model.initial_hidden_state(device=self.device)

    @torch.inference_mode()
    def act(self, observation):
        def batch(array):
            return torch.from_numpy(array).unsqueeze(0).to(self.device)

        inputs = {
            "rays": {name: batch(array) for name, array in observation["rays"].items()},
            "visible_entities": {name: batch(array) for name, array in observation["visible_entities"].items()},
            "vitals": batch(observation["vitals"]),
        }
        if self.model.use_rgb:
            inputs["rgb"] = batch(observation["rgb"])
        actions, _, _, self.hidden = self.model(inputs, self.hidden, deterministic=True, generator=self.generator)
        return actions[0].cpu().numpy()
