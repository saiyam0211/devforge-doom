#!/usr/bin/env python
"""Run a seeded local ViZDoom episode and report outcome and timing."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add system site-packages for PIL if not in local venv
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

import vizdoom as vd
from env.doom_env import limits
import env.doom_env.spaces as S
from env.doom_env.sensors import PLAYER_HEIGHT
from env.doom_env.wrapper import DoomEnv, GAME_VARIABLES, IWAD, SKILL
from submission.agent import DoomAgent


def resolve_wad_path(wad_spec: str) -> str:
    p = Path(wad_spec)
    if p.is_file():
        return str(p.resolve())
    if (ROOT / wad_spec).is_file():
        return str((ROOT / wad_spec).resolve())
    if (ROOT / "maps" / wad_spec).is_file():
        return str((ROOT / "maps" / wad_spec).resolve())
    try:
        candidate = Path(vd.__path__[0]) / wad_spec
        if candidate.is_file():
            return str(candidate.resolve())
        candidate_scen = Path(vd.__path__[0]) / "scenarios" / wad_spec
        if candidate_scen.is_file():
            return str(candidate_scen.resolve())
    except ImportError:
        pass
    raise FileNotFoundError(f"Cannot resolve WAD: {wad_spec}")


SKILL_NAMES = {
    1: "I'm Too Young To Die (Easy)",
    2: "Hey, Not Too Rough (Medium-Low)",
    3: "Hurt Me Plenty (Standard Evaluation)",
    4: "Ultra-Violence (Hard Difficulty)",
    5: "Nightmare (Extreme Difficulty)",
}


class LiveDoomEnv(DoomEnv):
    """Subclass of DoomEnv supporting visible window mode and configurable difficulty."""

    def __init__(self, wad: str, map_name: str = "MAP01", visible: bool = False, skill: int = SKILL, **kwargs):
        super().__init__(wad, map_name, **kwargs)
        self.skill = skill
        if visible or skill != SKILL:
            self._game.close()
            g = vd.DoomGame()
            g.set_doom_game_path(IWAD)
            g.set_doom_scenario_path(self.wad)
            g.set_doom_map(self.map_name)
            g.set_doom_skill(skill)
            g.set_mode(vd.Mode.PLAYER)
            g.set_window_visible(visible)
            g.set_sound_enabled(False)
            # Match the official observation buffers in windowed and headless tests.
            g.set_console_enabled(False)
            g.set_screen_resolution(vd.ScreenResolution.RES_160X120)
            g.set_screen_format(vd.ScreenFormat.RGB24)
            g.set_depth_buffer_enabled(True)
            g.set_render_hud(False)
            g.set_render_crosshair(False)
            g.set_render_weapon(True)
            g.set_render_decals(False)
            g.set_render_particles(False)
            g.set_render_messages(False)
            g.set_render_corpses(True)
            g.set_objects_info_enabled(True)
            g.set_sectors_info_enabled(True)
            g.set_available_buttons([getattr(vd.Button, b) for b in S.BUTTONS])
            g.set_available_game_variables(GAME_VARIABLES)
            g.set_episode_timeout(self.episode_ticks)
            g.init()
            self._game = g


def run_rulebook_test(
    wad_name: str,
    map_name: str = "MAP01",
    episode_ticks: int = 2100,
    visible: bool = False,
    device: str = "cpu",
    seed: int | None = 0,
    skill: int = 3,
    checkpoint: Path | None = None,
) -> Dict[str, Any]:
    wad_path = resolve_wad_path(wad_name)
    skill_str = SKILL_NAMES.get(skill, f"Skill {skill}")
    print(f"\n=======================================================")
    print(f" DevForge Local Test: {Path(wad_path).name}:{map_name} (Seed: {seed})")
    print(f" Difficulty: {skill_str}")
    print(f" Episode Cap: {episode_ticks} tics ({math.ceil(episode_ticks / S.FRAME_SKIP)} steps) | Window: {visible}")
    print(f"=======================================================")

    env = LiveDoomEnv(wad_path, map_name=map_name, visible=visible, episode_ticks=episode_ticks, skill=skill)
    if checkpoint is None:
        agent = DoomAgent(env.observation_space, env.action_space, device=device)
    else:
        from submission.neural_agent import DoomAgent as NeuralAgent

        agent = NeuralAgent(env.observation_space, env.action_space, device=device, checkpoint=checkpoint)

    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    agent.reset()

    start_tic = info["tic"]
    initial_keys = int(obs["vitals"][8:14].sum())
    step_latencies: list[float] = []
    visited_cells = set()
    opened_doors = set()
    start_state = env.privileged_state()
    doors = start_state["door_lock"] >= 0
    initially_passable = start_state["sector_ceiling"] - start_state["sector_floor"] >= PLAYER_HEIGHT
    ep_start = time.perf_counter()
    total_reward = 0.0

    state = start_state
    while True:
        visited_cells.add((math.floor(state["x"] / 128), math.floor(state["y"] / 128)))
        t0 = time.perf_counter()
        action = agent.act(obs)
        step_latencies.append(time.perf_counter() - t0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = env.privileged_state()
        passable = state["sector_ceiling"] - state["sector_floor"] >= PLAYER_HEIGHT
        opened_doors.update(np.flatnonzero(doors & ~initially_passable & passable))

        if visible:
            # Throttle slightly to simulate real-time ~35 FPS playback in window
            time.sleep(0.02)

        if terminated or truncated:
            break

    total_wall_time = time.perf_counter() - ep_start
    final = state
    env.close()

    visited_cells.add((math.floor(final["x"] / 128), math.floor(final["y"] / 128)))
    steps = len(step_latencies)
    # ViZDoom resets its episode timer when leaving some maps.
    elapsed_tics = final["tic"] - start_tic
    estimated_tics = elapsed_tics <= 0
    if estimated_tics:
        elapsed_tics = min(steps * S.FRAME_SKIP, episode_ticks)

    outcome = "Exit reached" if final["exit_reached"] else "Died" if final["dead"] else "Tick cap"
    lat_ms = np.array(step_latencies) * 1000
    agent_time = sum(step_latencies)
    budget = limits.agent_budget_s(episode_ticks)

    return {
        "wad": Path(wad_path).name,
        "map": map_name,
        "steps": steps,
        "ticks": elapsed_tics,
        "estimated_ticks": estimated_tics,
        "wall_time": total_wall_time,
        "agent_time": agent_time,
        "agent_budget": budget,
        "timing_pass": (
            step_latencies[0] <= limits.FIRST_ACT_TIMEOUT_S
            and max(step_latencies[1:], default=0.0) <= limits.ACT_TIMEOUT_S
            and agent_time <= budget
        ),
        "exit_reached": bool(final["exit_reached"]),
        "outcome": outcome,
        "keys_collected": max(0, int(final["keys"].sum()) - initial_keys),
        "doors_opened": len(opened_doors),
        "locked_doors_opened": sum(start_state["door_lock"][door] > 0 for door in opened_doors),
        "visited_cells": len(visited_cells),
        "hp_remaining": max(0, int(final["health"])),
        "kills": final["kills"],
        "damage_taken": final["damage_taken"],
        "env_reward": total_reward,
        "p50_latency_ms": float(np.percentile(lat_ms, 50)),
        "p95_latency_ms": float(np.percentile(lat_ms, 95)),
        "p99_latency_ms": float(np.percentile(lat_ms, 99)),
        "max_latency_ms": float(np.max(lat_ms)),
    }


def local_scorecard(result, episode_ticks):
    """Keep the previous 100-point layout using measured local outcomes."""
    points = (
        35.0 if result["exit_reached"] else 15.0 * min(1.0, result["ticks"] / episode_ticks),
        min(25.0, 10.0 * result["keys_collected"] + 5.0 * result["doors_opened"]),
        min(20.0, result["visited_cells"] * 0.8),
        20.0 if result["timing_pass"] else 0.0,
    )
    total = sum(points)
    grade = next(grade for threshold, grade in ((85, "A+"), (70, "A"), (55, "B+"), (40, "B"), (0, "C"))
                 if total >= threshold)
    return points, total, grade


def main():
    parser = argparse.ArgumentParser(description="Run a local ViZDoom benchmark")
    parser.add_argument("--wad", type=str, default="freedoom2.wad", help="WAD file")
    parser.add_argument("--map", type=str, default="MAP01", help="Map name")
    parser.add_argument("--ticks", type=int, default=2100, help="Episode ticks limit")
    parser.add_argument("--window", action="store_true", help="Open native window on desktop for live visual viewing")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    parser.add_argument("--checkpoint", type=Path, help="Run the specified checkpoint's neural policy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment reset (default: 0)")
    parser.add_argument("--skill", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Doom skill: 1 (easy) to 5 (nightmare), default 3, 4 is Ultra-Violence (Hard)")
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")

    res = run_rulebook_test(
        wad_name=args.wad,
        map_name=args.map,
        episode_ticks=args.ticks,
        visible=args.window,
        device=args.device,
        seed=args.seed,
        skill=args.skill,
        checkpoint=args.checkpoint,
    )

    print("\n" + "=" * 70)
    print(" DEVFORGE TRACK 1: LOCAL BENCHMARK")
    print("=" * 70)
    print(f" Environment         : {res['wad']}:{res['map']} (Skill {args.skill}: {SKILL_NAMES[args.skill]})")
    tics = f"{'~' if res['estimated_ticks'] else ''}{res['ticks']}"
    print(f" Episode Length      : {res['steps']} steps ({tics} tics / {args.ticks} cap)")
    print(f" Outcome             : {res['outcome']}")
    print(f" Health Remaining    : {res['hp_remaining']} HP")
    print(f" Keys Picked Up      : {res['keys_collected']}")
    print(f" Doors Opened        : {res['doors_opened']}")
    print(f" Locked Doors Opened : {res['locked_doors_opened']}")
    print(f" Enemies Killed      : {res['kills']}")
    print(f" Damage Taken        : {res['damage_taken']:.0f} HP")
    print(f" Areas Visited       : {res['visited_cells']} distinct 128-unit grid cells")
    print("-" * 70)
    print(f" Decision Latency    : p50 {res['p50_latency_ms']:.2f} ms, p99 {res['p99_latency_ms']:.2f} ms")
    print(f" Agent Compute Time  : {res['agent_time']:.2f} s / {res['agent_budget']:.2f} s budget")
    print(f" Local Timing Check  : {'PASSED' if res['timing_pass'] else 'FAILED'}")
    points, total, grade = local_scorecard(res, args.ticks)
    print("-" * 70)
    print(" LOCAL SCORECARD (OUT OF 100)")
    for label, value, maximum in zip(
        ("Exit & Survival", "Progression", "Exploration", "Timing"),
        points, (35, 25, 20, 20), strict=True,
    ):
        print(f" {label:<19}: {value:5.1f} / {maximum:.1f} pts")
    print(f" {'Total Score':<19}: {total:5.1f} / 100.0 pts")
    print(f" {'Local Rating':<19}: {grade}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
