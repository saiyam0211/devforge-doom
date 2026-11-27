"""Replay verified routes and measure progress after handing control to a checkpoint."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wrapper import DoomEnv, IWAD
from submission.neural_agent import DoomAgent
from training.train_hybrid_bc_ppo import DoomExpertTeacher
from training.train_recurrent_rl import Progress
from training.wad_teacher import WadTeacher


def make_teacher(env, jitter=False):
    if Path(env.wad).name.lower() != "freedoom2.wad" or env.map_name != "MAP01":
        return WadTeacher(env)
    teacher = DoomExpertTeacher()
    teacher.reset("freedoom2.wad", "MAP01", jitter=jitter)
    return teacher


def episode(env, seed, agent=None, prefix=()):
    obs, _ = env.reset(seed=seed)
    state = env.privileged_state()
    teacher = make_teacher(env) if agent is None else None
    if agent is not None:
        agent.reset()
    history = []
    for step in range(env.episode_ticks // 4):
        if agent is None:
            action = teacher.act(obs, state)
        else:
            # Advance recurrent state even while the teacher controls the player.
            proposal = agent.act(obs)
            action = prefix[step] if step < len(prefix) else proposal
        obs, _, done, timeout, _ = env.step(action)
        state = env.privileged_state()
        history.append((action.copy(), state))
        if done or timeout:
            break
    return history


def probe(env, seed, checkpoint):
    env.reset(seed=seed)
    initial = env.privileged_state()
    reference = episode(env, seed)
    final = reference[-1][1]
    n_keys = int(final["keys"].sum())
    if not (final["exit_reached"] or n_keys):
        return []
    agent = DoomAgent(env.observation_space, env.action_space, checkpoint=checkpoint)
    actions = [action for action, _ in reference]
    cuts = {0, *(round(len(actions) * p) for p in (0.25, 0.5, 0.75, 0.9))}
    if n_keys:
        first_key = next(i + 1 for i, (_, state) in enumerate(reference) if state["keys"].any())
        cuts.update((max(0, first_key - 25), first_key))
    rows = []
    for cut in sorted(cuts):
        played = episode(env, seed, agent, actions[:cut])
        states = [state for _, state in played]
        if cut:
            expected = reference[cut - 1][1]
            actual = states[min(cut, len(states)) - 1]
            keys_match = np.array_equal(expected["keys"], actual["keys"])
            if len(states) < cut or any(expected[k] != actual[k] for k in ("x", "y", "angle", "health", "tic")) or not keys_match:
                raise RuntimeError(f"Teacher prefix diverged on {env.map_name}, seed {seed}, cut {cut}")
        visited = set()
        waypoint_offset = 0
        for _, target in reference[cut:]:
            cell = Progress.cell(target)
            if cell in visited:
                continue
            visited.add(cell)
            if any((s["x"] - target["x"]) ** 2 + (s["y"] - target["y"]) ** 2 <= 48 ** 2 for s in states[cut:]):
                waypoint_offset += 1
        end = states[-1]
        start = states[cut - 1] if cut else initial
        new_passable = np.logical_or.reduce([Progress.passable(s) for s in states[cut:]]) & ~Progress.passable(start)
        new_locks = int(np.count_nonzero(new_passable & (start["door_lock"] > 0)))
        key_at_cut = int(states[cut - 1]["keys"].sum()) if cut else 0
        rows.append(dict(wad=Path(env.wad).name, map=env.map_name, seed=seed, cut=cut,
                         reference_steps=len(reference), reference_keys=n_keys, reference_exit=final["exit_reached"],
                         route_cells_reached=waypoint_offset, route_cells_remaining=len(visited),
                         policy_steps=len(states) - cut, new_keys=int(end["keys"].sum()) - key_at_cut,
                         new_locked_doors=new_locks, exit=end["exit_reached"], died=end["dead"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", action="append", help="WAD path and level, delimited by a colon")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 7, 42))
    args = parser.parse_args()
    rows = []
    for wad, name in (item.rsplit(":", 1) for item in (args.map or [f"{IWAD}:MAP01"])):
        env = DoomEnv(IWAD if wad == "freedoom2.wad" else wad, name, episode_ticks=4200)
        try:
            for seed in args.seeds:
                results = probe(env, seed, args.checkpoint)
                print(f"{Path(env.wad).name}:{name} seed {seed}: {len(results)} verified handoffs", flush=True)
                rows.extend(results)
        finally:
            env.close()
    if not rows:
        parser.error("No expert episode reached a key or exit")
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
