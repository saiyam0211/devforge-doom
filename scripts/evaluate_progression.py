"""Evaluate neural checkpoints on frozen procedural courses and public WADs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wrapper import DoomEnv, IWAD
from env.doom_env.wad import map_lumps, read_wad
from scripts.test_and_rate import LiveDoomEnv
from submission.model import SpatialEgoNet
from training.progression_maps import make_course
from training.train_progression import EPISODE_TICKS, observe
from training.train_recurrent_rl import Progress

SEEDS = (0, 7, 42)


def geometry(wad: Path) -> tuple:
    vertices = map_lumps(read_wad(wad), "MAP01")["VERTEXES"]
    return tuple(sorted(struct.iter_unpack("<hh", vertices)))


def cases(directory: Path, full: bool, scythe: Path | None, requiem: Path | None,
          fresh: bool = False, fresh_short: bool = False, long_courses: bool = False,
          long_full: bool = False, rl_gate: bool = False, rl_public: bool = False,
          rl_freedoom: bool = False, full_map_audit: bool = False, route_gate: bool = False):
    if route_gate:
        for wad, name in ((Path(IWAD), "MAP01"), (Path(IWAD), "MAP06"), (scythe, "MAP02")):
            for seed in (0, 7, 42, 101, 137):
                yield "route_known", wad, name, seed, 4200, 3, 0
        for wad, name in ((scythe, "MAP19"), (requiem, "MAP20")):
            for seed in SEEDS:
                yield "route_holdout", wad, name, seed, 4200, 3, 0
        return
    if full_map_audit:
        for label, names, skill in (("known", ("MAP01", "MAP06"), 3),
                                   ("unseen", ("MAP23", "MAP29"), 3),
                                   ("hard", ("MAP23", "MAP29"), 4)):
            for name in names:
                for seed in SEEDS:
                    yield label, Path(IWAD), name, seed, 4200, skill, 0
        return
    if rl_gate:
        yield "starter", ROOT / "maps/two_rooms.wad", "MAP01", 0, 2100, 3, 0
        selections = (("navigation", 58001, 3), ("combat", 58101, 3),
                      ("key", 58201, 10), ("gauntlet", 58301, 6))
        for phase, first_seed, count in selections:
            for index in range(count):
                course = make_course(directory, first_seed + index, phase,
                                     long_route=phase in ("key", "gauntlet"))
                yield phase, course.path, "MAP01", SEEDS[index % 3], 2100, 3, course.lock_count
        return
    if rl_public:
        for wad, maps in ((scythe, ("MAP07", "MAP25")), (requiem, ("MAP04", "MAP24"))):
            if wad is not None:
                for name in maps:
                    for seed in SEEDS:
                        yield "public_original_spawn", wad, name, seed, 4200, 3, 0
        if requiem is not None:
            for seed in SEEDS:
                yield "requiem_regression", requiem, "MAP21", seed, 4200, 3, 0
        if scythe is not None:
            for seed in SEEDS:
                yield "ultraviolence_regression", scythe, "MAP29", seed, 4200, 4, 0
        return
    if rl_freedoom:
        for name in ("MAP19", "MAP26", "MAP30"):
            for seed in SEEDS:
                yield "freedoom_holdout", Path(IWAD), name, seed, 4200, 3, 0
        return
    if long_courses:
        for index in range(14):
            phase = "key" if index < 10 else "gauntlet"
            course = make_course(directory, 22550 + index, phase, long_route=True)
            yield ("long_key" if phase == "key" else "long_gauntlet", course.path, "MAP01",
                   SEEDS[index % 3], 1800, 3, course.lock_count)
        return
    if long_full:
        for wad, maps in ((scythe, ("MAP21", "MAP26")), (requiem, ("MAP17", "MAP23"))):
            if wad is not None:
                for name in maps:
                    for seed in SEEDS:
                        yield "fresh_full", wad, name, seed, 4200, 3, 0
        if requiem is not None:
            for seed in SEEDS:
                yield "requiem_regression", requiem, "MAP21", seed, 4200, 3, 0
        return
    if fresh:
        for wad, maps in ((scythe, ("MAP20", "MAP23")), (requiem, ("MAP10", "MAP14")),
                          (Path(IWAD), ("MAP32",))):
            if wad is not None:
                for name in maps:
                    for seed in SEEDS:
                        yield "fresh_full", wad, name, seed, 4200, 3, 0
        return
    if not full:
        yield "starter", ROOT / "maps/two_rooms.wad", "MAP01", 0, 2100, 3, 0
        key_seed = 10750 if fresh_short else 9340
        course_seed = 12500 if fresh_short else 9700
        for index in range(10):
            course = make_course(directory, key_seed + index, "key")
            yield "key_gate", course.path, "MAP01", SEEDS[index % 3], EPISODE_TICKS, 3, course.lock_count
        for phase_index, phase in enumerate(("navigation", "combat", "gauntlet")):
            for index, seed in enumerate(SEEDS):
                course = make_course(directory, course_seed + phase_index * 100 + index, phase)
                yield phase, course.path, "MAP01", seed, EPISODE_TICKS, 3, course.lock_count
        return

    for wad, maps in ((scythe, ("MAP01", "MAP10", "MAP22")),
                      (requiem, ("MAP01", "MAP09", "MAP15")), (Path(IWAD), ("MAP31",))):
        if wad is not None:
            for name in maps:
                for seed in SEEDS:
                    yield "public_wad", wad, name, seed, 4200, 3, 0
    for wad, maps in ((scythe, ("MAP17", "MAP24")), (requiem, ("MAP12", "MAP21"))):
        if wad is not None:
            for name in maps:
                for seed in SEEDS:
                    yield "public_holdout", wad, name, seed, 4200, 3, 0
    if scythe is not None:
        for seed in SEEDS:
            yield "timed_stress", scythe, "MAP28", seed, 4200, 4, 0


def evaluate(model, wad, name, seed, ticks, skill, sample_actions=False):
    generator = torch.Generator().manual_seed(0) if sample_actions else None
    env = (DoomEnv(wad, name, episode_ticks=ticks) if skill == 3
           else LiveDoomEnv(str(wad), name, episode_ticks=ticks, skill=skill))
    try:
        obs, _ = env.reset(seed=seed)
        state = env.privileged_state()
        progress = Progress(state, None)
        initially_open = progress.opened.copy()
        hidden = model.initial_hidden_state()
        steps = 0
        stalled_steps = 0
        longest_stall_steps = 0
        stall_windows = 0
        while True:
            actions, _, _, hidden = model(observe(obs, model.use_rgb), hidden, deterministic=True, generator=generator)
            obs, _, end, timeout, _ = env.step(actions[0].numpy())
            steps += 1
            state = env.privileged_state()
            _, stalled = progress.step(state)
            stalled_steps += int(stalled)
            stall_windows = stall_windows + 1 if stalled else 0
            if stalled:
                longest_stall_steps = max(longest_stall_steps, stall_windows + progress.positions.maxlen - 1)
            if end or timeout:
                break
        return {
            "outcome": "exit" if state["exit_reached"] else "death" if state["dead"] else "timeout",
            "steps": steps, "keys": int(state["keys"].sum()),
            "locks_opened": int(sum(state["door_lock"][door] > 0 for door in progress.opened - initially_open)),
            "doors_opened": len(progress.opened - initially_open), "kills": state["kills"],
            "hp": max(0, int(state["health"])), "visited_cells": len(progress.visited),
            "stalled_steps": stalled_steps, "longest_stall_steps": longest_stall_steps,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-courses", type=Path, help="Reject generated test files that duplicate training WADs")
    suite = parser.add_mutually_exclusive_group()
    suite.add_argument("--full", action="store_true", help="Run the public-map regression suite")
    suite.add_argument("--fresh-full", action="store_true", help="Run the five additional full-map holdouts")
    suite.add_argument("--fresh-short", action="store_true", help="Use geometry-disjoint short courses for the visual training pass")
    suite.add_argument("--long-courses", action="store_true", help="Run held-out long backtracking courses")
    suite.add_argument("--long-full", action="store_true", help="Run four fresh public layouts and Requiem MAP21")
    suite.add_argument("--rl-gate", action="store_true", help="Run the recurrent RL held-out procedural courses")
    suite.add_argument("--rl-public", action="store_true", help="Run the recurrent RL public-map checks")
    suite.add_argument("--rl-freedoom", action="store_true", help="Run the original-spawn Freedoom holdouts")
    parser.add_argument("--disable-rgb", action="store_true", help="Ablate RGB in a visual checkpoint")
    parser.add_argument("--scythe", type=Path)
    parser.add_argument("--requiem", type=Path)
    suite.add_argument("--full-map-audit", action="store_true",
                       help="Compare original-spawn training levels with separate Freedoom holdouts")
    suite.add_argument("--route-gate", action="store_true", help="Run five original-spawn seeds per known map and two holdouts")
    args = parser.parse_args()
    if args.route_gate and not (args.scythe and args.requiem):
        parser.error("--route-gate requires --scythe and --requiem")
    torch.set_num_threads(1)
    fieldnames = ("checkpoint", "sha256", "suite", "wad", "map", "seed", "ticks", "skill",
                  "outcome", "steps", "keys", "locks_opened", "doors_opened",
                  "kills", "hp", "visited_cells", "stalled_steps", "longest_stall_steps", "chain")
    training_hashes = {
        hashlib.sha256(wad.read_bytes()).digest() for wad in args.training_courses.glob("*.wad")
    } if args.training_courses else set()
    if args.training_courses and not training_hashes:
        parser.error("--training-courses must contain generated training WADs")
    training_geometry = {geometry(wad) for wad in args.training_courses.glob("*.wad")} if args.training_courses else set()
    with TemporaryDirectory(prefix="progression-eval-") as tmp, args.output.open("w", newline="") as stream:
        tests = list(cases(Path(tmp), args.full, args.scythe, args.requiem, fresh=args.fresh_full,
                           fresh_short=args.fresh_short, long_courses=args.long_courses, long_full=args.long_full,
                           rl_gate=args.rl_gate, rl_public=args.rl_public, rl_freedoom=args.rl_freedoom, full_map_audit=args.full_map_audit,
                           route_gate=args.route_gate))
        generated = {"key", "key_gate", "navigation", "combat", "gauntlet", "long_key", "long_gauntlet"}
        holdout_hashes = [hashlib.sha256(p.read_bytes()).digest()
                          for group, p, *_ in tests if group in generated]
        assert len(set(holdout_hashes)) == len(holdout_hashes)
        assert not training_hashes.intersection(holdout_hashes)
        assert not training_geometry.intersection(
            geometry(p) for group, p, *_ in tests if group in generated
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for checkpoint in args.checkpoint:
                sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                weights = torch.load(checkpoint, weights_only=True, map_location="cpu")
                model = SpatialEgoNet(use_rgb=weights.get("use_rgb", False)).eval()
                model.load_state_dict(weights["model_state_dict"])
                label = checkpoint.stem
                if args.disable_rgb:
                    if not model.use_rgb:
                        parser.error("--disable-rgb requires a visual checkpoint")
                    model.use_rgb = False
                    label += "_no_rgb"
                counts = Counter()
                for suite, wad, name, seed, ticks, skill, locks in tests:
                    result = evaluate(model, wad, name, seed, ticks, skill, weights.get("sample_actions", False))
                    chain = locks > 0 and result["outcome"] == "exit" and result["keys"] == result["locks_opened"] == locks
                    writer.writerow(dict(checkpoint=label, sha256=sha, suite=suite, wad=wad.name, map=name,
                                         seed=seed, ticks=ticks, skill=skill, chain=chain, **result))
                    stream.flush()
                    counts[suite, "runs"] += 1
                    counts[suite, result["outcome"]] += 1
                    counts[suite, "chain"] += int(chain)
                    print(label, suite, wad.name, name, seed, result, flush=True)
                print(label, dict(counts), flush=True)


if __name__ == "__main__":
    main()
