"""Rehearse successful full-map routes and correct observations visited by the PPO policy."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wrapper import DoomEnv, IWAD
from submission.model import SpatialEgoNet
from training.progression_maps import make_course
from training.train_progression import collect
from training.train_hybrid_bc_ppo import DoomExpertTeacher, collect_demonstrations, train_behavioral_cloning
from training.train_wad_exits import trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1931)
    np.random.seed(1931)
    torch.set_num_threads(3)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SpatialEgoNet(use_rgb=weights["use_rgb"]).eval()
    model.load_state_dict(weights["model_state_dict"])
    demos = collect_demonstrations([(IWAD, "MAP01", 8)], diversify=True, use_rgb=model.use_rgb)
    if not demos:
        raise RuntimeError("No complete original-spawn Freedoom routes")
    env = DoomEnv(IWAD, "MAP01", episode_ticks=4200)
    try:
        for seed in range(681, 689):
            teacher = DoomExpertTeacher()
            teacher.reset("freedoom2.wad", "MAP01", jitter=bool(seed % 2))
            sequence, exited, _, _, _ = trajectory(env, teacher, model, seed, 0.25)
            if exited:
                demos.append({"wad_name": "freedoom2.wad", **sequence})
    finally:
        env.close()
    print(f"Full successful routes: {len(demos)}", flush=True)
    courses = [make_course(args.output, 67200 + 10 * i + j, phase, long_route=True)
               for i, phase in enumerate(("key", "gauntlet")) for j in range(4)]
    demos.extend(collect(courses, model, seed=1931, teacher_only=True, episode_ticks=2100))
    train_behavioral_cloning(model, demos, epochs=5, batch_size=8, lr=5e-5, diversify=True,
                            sequence_length=48, include_short=True)
    torch.save({**weights, "model_state_dict": model.state_dict(), "sample_actions": False,
                "refinement": "full_route_imitation", "warm_start": args.checkpoint.name}, args.output / "refined.pt")


if __name__ == "__main__":
    main()
