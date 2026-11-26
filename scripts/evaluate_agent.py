#!/usr/bin/env python
"""Comprehensive Benchmarking and Evaluation Harness for SpatialEgoNet DoomAgent.

Evaluates trained policy performance across official and community WADs:
- Tracks survival time, exit completion rate, key pickups, and combat telemetry.
- Measures strict per-step inference latency (p50, p95, p99) against the 40ms SLA.
- Generates ASCII benchmark scorecards for competition submission reviews.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from env.doom_env import make_env
from submission.agent import DoomAgent


def resolve_wad_path(wad_spec: str) -> str:
    p = Path(wad_spec)
    if p.is_file():
        return str(p.resolve())
    if (ROOT / wad_spec).is_file():
        return str((ROOT / wad_spec).resolve())
    try:
        import vizdoom
        vizdoom_dir = Path(vizdoom.__path__[0])
        candidate = vizdoom_dir / wad_spec
        if candidate.is_file():
            return str(candidate.resolve())
        candidate_scen = vizdoom_dir / "scenarios" / wad_spec
        if candidate_scen.is_file():
            return str(candidate_scen.resolve())
    except ImportError:
        pass
    raise FileNotFoundError(f"Cannot resolve WAD path for: {wad_spec}")


def evaluate_map(
    wad_name: str,
    map_name: str = "MAP01",
    num_episodes: int = 3,
    ticks: int = 2100,
    device: str = "cpu",
) -> Dict[str, float]:
    wad_path = resolve_wad_path(wad_name)
    env = make_env(wad_path, map_name, episode_ticks=ticks)
    agent = DoomAgent(env.observation_space, env.action_space, device=device)

    returns: List[float] = []
    durations: List[int] = []
    exits: List[bool] = []
    latencies: List[float] = []
    keys_collected: List[int] = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        agent.reset()
        ep_return = 0.0
        step_count = 0
        initial_keys = int(np.sum(obs["vitals"][8:14]))
        max_keys = initial_keys

        while True:
            t0 = time.perf_counter()
            action = agent.act(obs)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            step_count += 1
            curr_keys = int(np.sum(obs["vitals"][8:14]))
            max_keys = max(max_keys, curr_keys)

            if terminated or truncated:
                break

        returns.append(ep_return)
        durations.append(step_count)
        success = bool(info.get("exit_reached", False) or info.get("success", False))
        exits.append(success)
        keys_collected.append(max_keys - initial_keys)

    env.close()

    lat_arr = np.array(latencies)
    return {
        "wad": Path(wad_path).name,
        "map": map_name,
        "episodes": num_episodes,
        "mean_return": float(np.mean(returns)),
        "mean_ticks": float(np.mean(durations)),
        "success_rate": float(np.mean(exits)) * 100.0,
        "keys_picked": float(np.mean(keys_collected)),
        "p50_latency_ms": float(np.percentile(lat_arr, 50)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "p99_latency_ms": float(np.percentile(lat_arr, 99)),
        "max_latency_ms": float(np.max(lat_arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DoomAgent on single or multiple maps")
    parser.add_argument("--maps", nargs="*", default=["maps/two_rooms.wad:MAP01", "freedoom2.wad:MAP01", "freedoom2.wad:MAP02"], help="Maps to evaluate")
    parser.add_argument("--episodes", type=int, default=2, help="Number of evaluation episodes per map")
    parser.add_argument("--ticks", type=int, default=2100, help="Max ticks per episode")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" DevForge 2026 Track 1: SpatialEgoNet DoomAgent Benchmark Evaluation")
    print(f" Maps: {args.maps} | Episodes/Map: {args.episodes} | Device: {args.device}")
    print("=" * 80)

    results = []
    for spec in args.maps:
        if ":" in spec:
            wad, map_name = spec.split(":", 1)
        else:
            wad, map_name = spec, "MAP01"

        print(f"\n[Evaluating] {wad}:{map_name} ...")
        res = evaluate_map(wad, map_name, num_episodes=args.episodes, ticks=args.ticks, device=args.device)
        results.append(res)
        print(f" -> Return: {res['mean_return']:+.2f} | Ticks: {res['mean_ticks']:.1f} | Success: {res['success_rate']:.1f}% | Latency p50: {res['p50_latency_ms']:.2f}ms")

    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK SCORECARD")
    print("=" * 80)
    header = f"{'WAD/Map':<24} | {'Return':<8} | {'Ticks':<7} | {'Success':<8} | {'p50 (ms)':<8} | {'p99 (ms)':<8} | {'Status'}"
    print(header)
    print("-" * len(header))

    for r in results:
        map_label = f"{r['wad']}:{r['map']}"
        status = "COMPLIANT" if r["p99_latency_ms"] < 40.0 else "SLA BREACH"
        print(
            f"{map_label:<24} | {r['mean_return']:>+8.2f} | {r['mean_ticks']:>7.1f} | "
            f"{r['success_rate']:>7.1f}% | {r['p50_latency_ms']:>8.2f} | {r['p99_latency_ms']:>8.2f} | {status}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
