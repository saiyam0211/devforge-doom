"""Train a visual-spatial policy on verified WAD exit routes and procedural rehearsal."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import struct
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.wad import Lump, read_wad, write_wad
from env.doom_env.wrapper import DoomEnv, IWAD
from submission.model import SpatialEgoNet
from training.progression_maps import make_course
from training.train_hybrid_bc_ppo import collect_demonstrations, train_behavioral_cloning
from training.train_progression import PHASES, collect, observe
from training.wad_teacher import STRIDE, RouteGrid, WadTeacher


def valid_position(point, polygons, floors, ceilings):
    x, y = point
    for sector, lines in polygons.items():
        if ceilings[sector] - floors[sector] < 56:
            continue
        x1, y1, x2, y2 = lines.T
        edge = (y1 > y) != (y2 > y)
        intersection = x1[edge] + (y - y1[edge]) * (x2[edge] - x1[edge]) / (y2[edge] - y1[edge])
        if np.count_nonzero(intersection > x) % 2:
            return True
    return False


def spawn_variants(env, seed):
    """Sample reachable positions in progressively longer routes from the exit."""
    teacher = WadTeacher(env)
    grid = RouteGrid(env._geometry, (False,) * 6)
    rng = np.random.default_rng(seed)
    polys = env.mapinfo.sector_lines()
    state = env.privileged_state()
    if not teacher.exits:
        return []
    goal, _, _ = teacher.exits[0]
    if grid.route(goal, goal) is None:
        return []
    target = grid.cells_near(goal)[0]
    parents = grid.routes[target]
    depth = {target: 0}
    for node, previous in parents.items():
        if previous is not None:
            depth[node] = depth[previous] + 1
    candidates = []
    for step_count in (3, 5, 8, 12, 17, 24, 32):
        options = [node for node, steps in depth.items() if steps == step_count]
        rng.shuffle(options)
        for node in options[:40]:
            point = grid.origin + np.array(node) * STRIDE
            if valid_position(point, polys, state["sector_floor"], state["sector_ceiling"]):
                candidates.append((int(point[0]), int(point[1]), int(rng.choice(np.arange(0, 360, 45)))))
                break
    return candidates


def relocate(lumps, name, position, output):
    marker = next(i for i, lump in enumerate(lumps) if lump.name == name)
    things = lumps[marker + 1]
    assert things.name == "THINGS"
    content = bytearray(things.data)
    for offset in range(0, len(content), 10):
        if struct.unpack_from("<h", content, offset + 6)[0] == 1:
            struct.pack_into("<hhh", content, offset, *position)
            write_wad(output, lumps[:marker+1] + [Lump("THINGS", bytes(content))] + lumps[marker+2:])
            return
    raise ValueError(f"{name} has no player start")


def trajectory(env, teacher, model, seed, mix, *, policy_window=None):
    obs, info = env.reset(seed=seed)
    counts = Counter()
    buf = []
    hidden = model.initial_hidden_state()
    rng = np.random.default_rng(seed)
    motion = []
    previous_use = False
    for step in range(env.episode_ticks // 4):
        state = env.privileged_state()
        expert = teacher.act(obs, state)
        action = expert
        if mix:
            with torch.inference_mode():
                outputs, _, _, hidden = model(observe(obs, model.use_rgb), hidden, deterministic=True)
            if step % 5 == 0:
                follow_policy = rng.random() < mix
            in_window = policy_window is None or policy_window[0] <= step < policy_window[1]
            if follow_policy and in_window:
                action = outputs[0].numpy()
                counts["policy_steps"] += 1
                counts["corrections"] += int(not np.array_equal(expert, action))
        # Preserve spacing between actual USE button-down events in collected play.
        if action[7] and previous_use:
            action = action.copy()
            action[7] = 0
        previous_use = bool(action[7])
        buf.append((obs["rays"]["distance"], obs["rays"]["hit_type"],
                    obs["visible_entities"]["features"], obs["visible_entities"]["entity_type"],
                    obs["visible_entities"]["mask"], obs["vitals"], expert,
                    obs["rgb"][1::4, 1::4].copy()))
        motion.append((state["x"], state["y"]))
        obs, _, end, timeout, info = env.step(action)
        if end or timeout:
            break
        if step >= 79 and step % 20 == 19 and np.ptp(motion[-60:], axis=0).max() < 18:
            break
    end = env.privileged_state()
    fields = ("ray_dist", "ray_hit", "ent_feat", "ent_type", "ent_mask", "vitals", "actions", "rgb")
    data = dict(zip(fields, map(np.stack, zip(*buf)), strict=True))
    return data, info["exit_reached"], int(end["keys"].sum()), end["kills"], counts


def collect_wads(wads, output, model, mix, seeds, replay=None):
    collected, successful, counts = [], [], Counter()
    model.eval()
    for wad, names in wads:
        lumps = read_wad(wad)
        for name in names:
            if replay is None:
                source = DoomEnv(wad, name, render_buffers=False)
                try:
                    source.reset(seed=seeds)
                    positions = [None] + spawn_variants(source, seeds)
                finally:
                    source.close()
            else:
                positions = replay.get((wad, name), ())
            map_exits = 0
            map_keys = 0
            starts = []
            for index, position in enumerate(positions):
                relocated = output / "relocated.wad"
                if position is not None:
                    relocate(lumps, name, position, relocated)
                env = DoomEnv(relocated if position is not None else wad, name,
                              episode_ticks=2100, render_buffers=True)
                try:
                    env.reset(seed=seeds + index)
                    data, exited, keys, kills, corrections = trajectory(
                        env, WadTeacher(env, exit_only=position is not None), model, seeds + index, mix
                    )
                    counts.update(corrections)
                    counts["attempts"] += 1
                    counts["exits"] += int(exited)
                    counts["original_start_exits"] += int(exited and position is None)
                    map_exits += int(exited)
                    map_keys += keys
                    if exited:
                        starts.append(position)
                    if len(data["actions"]) >= 12 and (exited or keys or (mix and kills >= 2)):
                        data["wad_name"] = f"{wad.name}:{name}"
                        collected.append(data)
                        counts["kept"] += 1
                finally:
                    env.close()
            successful.append(((wad, name), starts))
            print(f"{wad.name}:{name} {map_exits}/{len(positions)} exits, {map_keys} keys", flush=True)
    print(f"WAD collection mix={mix}: {dict(counts)}", flush=True)
    return collected, dict(successful)


def save(model, path, sources):
    torch.save({"model_state_dict": model.state_dict(), "use_rgb": model.use_rgb,
                "training_wads": [(p.name, maps) for p, maps in sources], "seed": 2718}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--scythe", type=Path, required=True)
    parser.add_argument("--requiem", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--correction-epochs", type=int, default=8)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(2718)
    np.random.seed(2718)
    weights = torch.load(args.warm_start, map_location="cpu", weights_only=True)
    model = SpatialEgoNet(use_rgb=True)
    result = model.load_state_dict(weights["model_state_dict"], strict=False)
    assert not result.unexpected_keys and all(key.startswith("rgb_encoder.") for key in result.missing_keys)
    maps = [
        (args.scythe, [f"MAP{n:02d}" for n in (2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14)]),
        (args.requiem, [f"MAP{n:02d}" for n in (2, 3, 5, 6, 7)]),
        (Path(IWAD), [f"MAP{n:02d}" for n in (2, 3, 4, 5, 6, 8)]),
    ]
    wad_data, starts = collect_wads(maps, args.work_dir, model, mix=0, seeds=2718)
    demos = collect_demonstrations([(str(ROOT/"maps/two_rooms.wad"), "MAP01", 5),
                                    (IWAD, "MAP01", 14)], diversify=True, use_rgb=True)
    courses = [make_course(args.work_dir, 4000 + 100*stage + i, phase)
               for stage, phase in enumerate(PHASES) for i in range(6)]
    demos.extend(collect(courses, model, seed=1370, teacher_only=True))
    demos.extend(wad_data)
    assert sum(bool(variants) for variants in starts.values()) >= 4, "Too few confirmed real-WAD exit routes"
    train_behavioral_cloning(model, demos, epochs=args.epochs, batch_size=24, lr=1.5e-4,
                            diversify=True, sequence_length=16)
    save(model, args.work_dir/"visual_routes.pt", maps)
    corrections, _ = collect_wads(maps, args.work_dir, model, mix=0.35, seeds=3671, replay=starts)
    train_behavioral_cloning(model, demos + corrections, epochs=args.correction_epochs,
                            batch_size=24, lr=8e-5, diversify=True, sequence_length=16)
    save(model, args.work_dir/"visual_corrected.pt", maps)


if __name__ == "__main__":
    main()
