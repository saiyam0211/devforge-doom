"""Fine-tune a visual checkpoint with longer recurrent windows and backtracking."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wrapper import IWAD
from submission.model import SpatialEgoNet
from training.progression_maps import make_course
from training.train_hybrid_bc_ppo import collect_demonstrations, train_behavioral_cloning
from training.train_progression import PHASES, collect
from training.train_wad_exits import collect_wads

TICKS = 1800


def save(model, path, warm_start):
    torch.save({"model_state_dict": model.state_dict(), "use_rgb": True, "sequence_length": 48,
                "warm_start": warm_start.name, "training_seed": 1481}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--scythe", type=Path, required=True)
    parser.add_argument("--requiem", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--correction-epochs", type=int, default=5)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1481)
    np.random.seed(1481)
    checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=True)
    if not checkpoint.get("use_rgb"):
        parser.error("--warm-start must be a visual-spatial checkpoint")
    model = SpatialEgoNet(use_rgb=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    long_courses = [
        make_course(args.work_dir, 20000 + stage * 100 + i, phase, long_route=True)
        for stage, phase in enumerate(("key", "gauntlet")) for i in range(10)
    ]
    review_courses = [
        make_course(args.work_dir, 20500 + stage * 100 + i, phase)
        for stage, phase in enumerate(PHASES) for i in range(3)
    ]
    demos = collect(long_courses, model, seed=311, teacher_only=True, episode_ticks=TICKS)
    assert len(demos) >= 15, "Fewer than 15 longer courses yielded complete demonstrations"
    demos += collect(review_courses, model, seed=481, teacher_only=True)
    demos += collect_demonstrations(
        [(str(ROOT/"maps/two_rooms.wad"), "MAP01", 3), (IWAD, "MAP01", 6)],
        diversify=True, use_rgb=True,
    )
    replay, _ = collect_wads(
        [(args.scythe, ["MAP02", "MAP03"]), (args.requiem, ["MAP06"])],
        args.work_dir, model, mix=0, seeds=2718,
    )
    demos += replay
    train_behavioral_cloning(model, demos, epochs=args.epochs, batch_size=16, lr=9e-5,
                            diversify=True, sequence_length=48, include_short=True)
    save(model, args.work_dir/"long_imitation.pt", args.warm_start)
    corrected = collect(long_courses, model, seed=915, teacher_only=False, episode_ticks=TICKS)
    train_behavioral_cloning(model, demos + corrected, epochs=args.correction_epochs,
                            batch_size=16, lr=5e-5, diversify=True, sequence_length=48, include_short=True)
    save(model, args.work_dir/"long_corrected.pt", args.warm_start)


if __name__ == "__main__":
    main()
