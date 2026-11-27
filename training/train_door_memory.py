"""Train a recurrent auxiliary prediction for a previously observed locked door."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wrapper import DoomEnv
from submission.model import SpatialEgoNet
from training.progression_maps import CELL, make_course
from training.progression_teacher import CourseTeacher
from training.train_recurrent_rl import encode


def record(directory, model, seed):
    course = make_course(directory, seed, "key", long_route=True)
    lock = np.array(next(cell for cell, special in course.cells.items() if special > 1))
    center = (lock + 0.5) * CELL
    observations, targets, visible, remembered = [], [], [], []
    seen = False
    previous_use = False
    teacher = CourseTeacher(course)
    with DoomEnv(course.path, episode_ticks=2100) as env:
        obs, _ = env.reset(seed=seed)
        for _ in range(525):
            state = env.privileged_state()
            rays = obs["rays"]
            in_view = bool(((rays["hit_type"] == 3) & (rays["distance"] < 0.2)).any())
            seen |= in_view and not state["keys"].any()
            dx, dy = center - [state["x"], state["y"]]
            bearing = np.arctan2(dy, dx) - np.deg2rad(state["angle"])
            observations.append(obs)
            visible.append(float(seen))
            remembered.append(seen and state["keys"].any() and not in_view and np.hypot(dx, dy) > CELL)
            targets.append((np.cos(bearing), np.sin(bearing)))
            action = teacher.act(obs, state, previous_use)
            previous_use = bool(action[7])
            obs, _, done, timeout, _ = env.step(action)
            if done or timeout:
                break
    with torch.no_grad():
        features = torch.cat([encode(model, observations[i:i + 32])[0]
                              for i in range(0, len(observations), 32)])[None]
        reference, _ = model.gru(features)
    return features, reference, torch.tensor(visible), torch.tensor(targets, dtype=torch.float32), torch.tensor(remembered)


def predict(model, head, route):
    features, reference, seen, target, mask = route
    latent, _ = model.gru(features)
    prediction = head(latent.squeeze(0))
    cosine = (F.normalize(prediction[:, 1:], dim=-1) * target).sum(-1)
    return latent, reference, prediction[:, 0], seen, cosine, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.manual_seed(81)
    weights = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model = SpatialEgoNet(use_rgb=weights.get("use_rgb", False)).eval()
    model.load_state_dict(weights["model_state_dict"])
    model.requires_grad_(False)
    model.gru.requires_grad_(True)

    train = [record(args.output, model, seed) for seed in range(79601, 79630, 2)]
    test = [record(args.output, model, seed) for seed in (79801, 79803, 79805, 79807, 79809)]
    head = nn.Linear(model.hidden_dim, 3)
    frozen_head = deepcopy(head)
    frozen_optimizer = torch.optim.AdamW(frozen_head.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW([
        {"params": model.gru.parameters(), "lr": 1e-5},
        {"params": head.parameters(), "lr": 1e-3},
    ])
    for epoch in range(6):
        for index in torch.randperm(len(train)):
            latent, reference, logits, seen, cosine, mask = predict(model, head, train[index])
            loss = F.binary_cross_entropy_with_logits(logits, seen)
            loss += ((1 - cosine) * mask).sum() / mask.sum().clamp_min(1)
            loss += 30 * F.mse_loss(latent, reference)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            frozen_prediction = frozen_head(reference.squeeze(0))
            frozen_cosine = (F.normalize(frozen_prediction[:, 1:], dim=-1) * train[index][3]).sum(-1)
            frozen_loss = F.binary_cross_entropy_with_logits(frozen_prediction[:, 0], seen)
            frozen_loss += ((1 - frozen_cosine) * mask).sum() / mask.sum().clamp_min(1)
            frozen_optimizer.zero_grad(set_to_none=True)
            frozen_loss.backward()
            frozen_optimizer.step()
        with torch.no_grad():
            directions = torch.cat([cosine[mask] for *_, cosine, mask in
                                    (predict(model, head, route) for route in test)])
            frozen_directions = torch.cat([
                (F.normalize(frozen_head(reference.squeeze(0))[:, 1:], dim=-1) * target).sum(-1)[mask]
                for _, reference, _, target, mask in test
            ])
            print(epoch + 1, "test bearing samples", len(directions), "within 45 degrees",
                  int((directions >= 2 ** -0.5).sum()), "frozen", int((frozen_directions >= 2 ** -0.5).sum()),
                  flush=True)
    torch.save({**weights, "model_state_dict": model.state_dict(), "refinement": "door_memory"},
               args.output / "memory.pt")


if __name__ == "__main__":
    main()
