"""Train Spatial-EgoNet on procedural courses with imitation, corrections, and PPO."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wrapper import DoomEnv
from training.progression_maps import Course, make_course
from training.progression_teacher import CourseTeacher
from training.train_hybrid_bc_ppo import collect_demonstrations, train_behavioral_cloning
from training.train_ppo import PPOTrainer, run_training_loop

PHASES = ("navigation", "combat", "key", "gauntlet")
EPISODE_TICKS = 1000


def observe(obs, use_rgb: bool = False) -> dict:
    return {
        key: (
            {name: torch.from_numpy(array).unsqueeze(0) for name, array in obs[key].items()}
            if isinstance(obs[key], dict) else torch.from_numpy(obs[key]).unsqueeze(0)
        )
        for key in (("rays", "visible_entities", "vitals", "rgb") if use_rgb else ("rays", "visible_entities", "vitals"))
    }


def collect(courses: list[Course], model, seed: int, teacher_only: bool, *, episode_ticks: int = EPISODE_TICKS) -> list[dict]:
    rng = np.random.default_rng(seed)
    sequences = []
    counts = Counter()
    model.eval()
    for index, course in enumerate(courses):
        env = DoomEnv(course.path, episode_ticks=episode_ticks, render_buffers=model.use_rgb)
        try:
            obs, _ = env.reset(seed=seed + index)
            hidden = model.initial_hidden_state()
            teacher = CourseTeacher(course)
            cooldown = 0
            samples = []
            rgb_frames = []
            for step in range(episode_ticks // 4):
                state = env.privileged_state()
                expert = teacher.act(obs, state, cooldown > 0)
                action = expert
                if not teacher_only:
                    with torch.inference_mode():
                        prediction, _, _, hidden = model(observe(obs, model.use_rgb), hidden, deterministic=True)
                    proposal = prediction[0].numpy()
                    if step % 8 == 0:
                        use_teacher = rng.random() < 0.55
                    if not use_teacher:
                        action = proposal
                        counts["policy_steps"] += 1
                        counts["corrections"] += int(not np.array_equal(proposal, expert))
                cooldown = 12 if action[7] and not cooldown else max(0, cooldown - 1)
                samples.append((
                    obs["rays"]["distance"], obs["rays"]["hit_type"],
                    obs["visible_entities"]["features"], obs["visible_entities"]["entity_type"],
                    obs["visible_entities"]["mask"], obs["vitals"], expert,
                ))
                if model.use_rgb:
                    rgb_frames.append(obs["rgb"][1::4, 1::4].copy())
                obs, _, end, timeout, info = env.step(action)
                if end or timeout:
                    break
            counts["exits"] += int(info["exit_reached"])
            counts["keys"] += int(env.privileged_state()["keys"].sum())
            if len(samples) >= 24 and (not teacher_only or info["exit_reached"]):
                keys = ("ray_dist", "ray_hit", "ent_feat", "ent_type", "ent_mask", "vitals", "actions")
                sequences.append({"wad_name": course.path.name, **{
                    k: np.stack(values) for k, values in zip(keys, zip(*samples), strict=True)
                }})
                if model.use_rgb:
                    sequences[-1]["rgb"] = np.stack(rgb_frames)
                counts[course.phase] += 1
        finally:
            env.close()
    print(f"Collection ({'teacher' if teacher_only else 'mixed rollouts'}): {dict(counts)}", flush=True)
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--layouts-per-stage", type=int, default=10)
    parser.add_argument("--imitation-epochs", type=int, default=12)
    parser.add_argument("--correction-epochs", type=int, default=8)
    parser.add_argument("--ppo-rounds", type=int, default=2)
    parser.add_argument("--extend-freedoom", action="store_true",
                        help="Continue PPO from --warm-start on full training maps with generated-course rehearsal")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Keep the imitation encoders and GRU fixed during the Freedoom PPO extension")
    args = parser.parse_args()
    if not args.warm_start.is_file():
        parser.error(f"Missing checkpoint: {args.warm_start}")
    if args.freeze_encoder and not args.extend_freedoom:
        parser.error("--freeze-encoder requires --extend-freedoom")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    np.random.seed(73)

    lr = 1e-5 if args.freeze_encoder else 3e-5 if args.extend_freedoom else 5e-5
    trainer = PPOTrainer(device="cpu", lr=lr, ent_coef=0.003, rnd_weight=0.0)
    trainer.load_checkpoint(args.warm_start)
    if args.extend_freedoom:
        if args.freeze_encoder:
            for module in (trainer.model.ray_encoder, trainer.model.entity_encoder, trainer.model.vitals_encoder,
                           trainer.model.fusion, trainer.model.gru):
                module.requires_grad_(False)
        # These levels were part of the earlier training curriculum. No evaluation map is used.
        maps = [("freedoom2.wad", name, 2100) for name in ("MAP01", "MAP02", "MAP04", "MAP06", "MAP11")]
        for index, (layout, phase) in enumerate(((1200, "key"), (1300, "gauntlet"))):
            course = make_course(args.work_dir, layout, phase)
            maps.insert(2 + index * 3, (str(course.path), "MAP01", EPISODE_TICKS))
        for index, (wad, name, ticks) in enumerate(maps):
            run_training_loop(wad, name, total_steps=1024, rollout_len=128, batch_size=64, ppo_epochs=3,
                              trainer=trainer, progression=True, seed=2600 + index, episode_ticks=ticks, export=False)
        filename = "ppo_frozen.pt" if args.freeze_encoder else "ppo_freedoom.pt"
        trainer.save_checkpoint(args.work_dir/filename)
        return
    courses = [
        make_course(args.work_dir, 1000 + 100 * stage + index, phase)
        for stage, phase in enumerate(PHASES) for index in range(args.layouts_per_stage)
    ]
    demos = collect(courses, trainer.model, seed=73, teacher_only=True)
    # Rehearse the starter entry point while learning new behavior.
    demos += collect_demonstrations([(str(ROOT/"maps"/"two_rooms.wad"), "MAP01", 5)])
    train_behavioral_cloning(trainer.model, demos, epochs=args.imitation_epochs,
                            batch_size=24, lr=3e-4, diversify=True)
    trainer.save_checkpoint(args.work_dir/"imitation.pt")
    mixed = collect(courses, trainer.model, seed=811, teacher_only=False)
    train_behavioral_cloning(trainer.model, demos + mixed, epochs=args.correction_epochs,
                            batch_size=24, lr=1.5e-4, diversify=True)
    trainer.save_checkpoint(args.work_dir/"corrections.pt")

    trainer.model.train()
    for round_index in range(args.ppo_rounds):
        for stage, _ in enumerate(PHASES):
            start = stage * args.layouts_per_stage
            for course in courses[start + round_index % args.layouts_per_stage:
                                  start + args.layouts_per_stage:5]:
                run_training_loop(str(course.path), total_steps=512, rollout_len=128, batch_size=64,
                                  ppo_epochs=3, trainer=trainer, progression=True, seed=1800 + round_index,
                                  episode_ticks=EPISODE_TICKS, export=False)
        trainer.save_checkpoint(args.work_dir/"ppo.pt")


if __name__ == "__main__":
    main()
