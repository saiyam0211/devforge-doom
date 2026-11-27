"""Train on expert corrections after short policy handoffs on known maps."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wrapper import DoomEnv, IWAD
from scripts.route_handoff import episode, make_teacher
from submission.model import SpatialEgoNet
from training.progression_maps import make_course
from training.train_hybrid_bc_ppo import train_behavioral_cloning
from training.train_progression import collect
from training.train_wad_exits import trajectory


def collect_recovery(wad, name, model):
    examples = []
    with DoomEnv(wad, name, episode_ticks=4200) as env:
        for index, seed in enumerate(range(701, 719)):
            route = episode(env, seed)
            target = next((i + 1 for i, (_, state) in enumerate(route) if state["keys"].any()), len(route))
            final = route[-1][1]
            if not (final["exit_reached"] or final["keys"].any()):
                continue
            cut = round(target * (0.85, 0.6, 0.35, 0.0)[index % 4])
            env.reset(seed=seed)
            teacher = make_teacher(env)
            sequence, exited, keys, _, counts = trajectory(
                env, teacher, model, seed, 0.65, policy_window=(cut, cut + 20)
            )
            if exited or keys:
                examples.append({**sequence, "wad_name": f"{Path(wad).name}:{name}"})
                print(name, seed, cut, exited, keys, dict(counts), flush=True)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scythe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1932)
    np.random.seed(1932)
    torch.set_num_threads(3)
    weights = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model = SpatialEgoNet(use_rgb=weights.get("use_rgb", False)).eval()
    model.load_state_dict(weights["model_state_dict"])
    examples = collect_recovery(IWAD, "MAP01", model)
    examples += collect_recovery(args.scythe, "MAP02", model)
    if not examples:
        raise RuntimeError("No successful recovery trajectories")
    print(f"Retained {len(examples)} corrected full-map trajectories", flush=True)
    courses = [make_course(args.output, 67800 + 10 * i + j, phase, long_route=True)
               for i, phase in enumerate(("key", "gauntlet")) for j in range(5)]
    examples.extend(collect(courses, model, seed=1932, teacher_only=True, episode_ticks=2100))
    train_behavioral_cloning(model, examples, epochs=4, batch_size=8, lr=4e-5,
                            diversify=True, sequence_length=48, include_short=True)
    torch.save({**weights, "model_state_dict": model.state_dict(), "sample_actions": False,
                "refinement": "route_recovery"}, args.output / "recovery.pt")


if __name__ == "__main__":
    main()
