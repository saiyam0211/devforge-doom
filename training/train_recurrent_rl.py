"""Recurrent PPO with a learned-policy anchor and a mixed ViZDoom curriculum."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, kl_divergence
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.doom_env.sensors import PLAYER_HEIGHT
from env.doom_env.wrapper import DoomEnv, IWAD
from submission.model import SpatialEgoNet
from training.progression_maps import CELL, DIRECTIONS, Course, make_course

CHUNK = 24
ROLLOUT = 96
PHASES = ("navigation", "combat", "key", "gauntlet", "public", "public")
PUBLIC_MAPS = ("MAP01", "MAP02", "MAP04", "MAP05", "MAP06", "MAP08", "MAP09", "MAP11")
SCYTHE_MAPS = ("MAP02", "MAP03", "MAP04", "MAP05", "MAP09")
REQUIEM_MAPS = ("MAP03", "MAP05", "MAP06")


def distributions(model, recurrent):
    # These biases match the deterministic ATTACK and USE thresholds used by the checkpoint.
    binary_logits = model.head_binary(recurrent)
    binary_logits = binary_logits + binary_logits.new_tensor([0.8, 1.2, 0.0])
    return (
        Categorical(logits=model.head_move_fb(recurrent)),
        Categorical(logits=model.head_move_lr(recurrent)),
        Categorical(logits=model.head_turn(recurrent)),
        Bernoulli(logits=binary_logits),
    )


def log_probability(policy, discrete, buttons, mask):
    binary = policy[3].log_prob(buttons) * mask
    return sum(head.log_prob(discrete[..., index]) for index, head in enumerate(policy[:3])) + binary.sum(-1)


def policy_kl(reference, policy, mask):
    binary = kl_divergence(reference[3], policy[3]) * mask
    return sum(kl_divergence(old, new) for old, new in zip(reference[:3], policy[:3])) + binary.sum(-1)


def control_buttons(discrete, binaries, mask):
    move, strafe, turn = discrete.unbind(-1)
    flags = torch.stack((move == 1, move == 2, strafe == 1, strafe == 2, turn == 1, turn == 2), -1)
    return torch.cat((flags, binaries * mask), dim=-1).to(torch.int8).numpy()


def encode(model, observations):
    tensors = {
        key: {name: torch.from_numpy(np.stack([obs[key][name] for obs in observations]))
              for name in observations[0][key]}
        for key in ("rays", "visible_entities")
    }
    tensors["vitals"] = torch.from_numpy(np.stack([obs["vitals"] for obs in observations]))
    if model.use_rgb:
        tensors["rgb"] = torch.from_numpy(np.stack([obs["rgb"][1::4, 1::4] for obs in observations]))
    validity = torch.ones(len(observations), 3)
    validity[:, 2] = tensors["vitals"][:, 3] <= 0
    return model.extract_features(tensors), validity


def recurrent_features(gru, features, hidden, reset):
    """Unroll contiguous chunks without sending memory across episode boundaries."""
    boundaries = (reset[:, 1:].any(dim=0).nonzero().flatten() + 1).tolist()
    outputs = []
    start = 0
    for stop in (*boundaries, features.shape[1]):
        hidden = hidden * (~reset[:, start])[None, :, None]
        output, hidden = gru(features[:, start:stop], hidden)
        outputs.append(output)
        start = stop
    return torch.cat(outputs, dim=1)


class Progress:
    def __init__(self, state, course: Course | None):
        self.previous = state
        self.visited = {self.cell(state)}
        self.opened = set(np.flatnonzero(self.passable(state)))
        self.positions = deque(maxlen=24)
        self.route_lengths = {}
        self.targets = []
        if course is not None:
            self.targets = [(goal, index - 8) for goal, index in course.objectives]
            self.targets.append((course.exit_cell, None))
            for goal, _ in self.targets:
                queue = deque([goal])
                distances = {goal: 0}
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in DIRECTIONS:
                        node = (x + dx, y + dy)
                        if node in course.cells and node not in distances:
                            distances[node] = distances[x, y] + 1
                            queue.append(node)
                self.route_lengths[goal] = distances

    @staticmethod
    def cell(state, size=64):
        return int(state["x"] // size), int(state["y"] // size)

    @staticmethod
    def passable(state):
        height = state["sector_ceiling"] - state["sector_floor"]
        return (state["door_lock"] >= 0) & (height >= PLAYER_HEIGHT)

    def step(self, state):
        old = self.previous
        cell = self.cell(state)
        opened = set(np.flatnonzero(self.passable(state))) - self.opened
        locked = sum(state["door_lock"][door] > 0 for door in opened)
        self.positions.append((state["x"], state["y"]))
        stalled = False
        if len(self.positions) == self.positions.maxlen:
            xs, ys = zip(*self.positions)
            stalled = (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 < 22 ** 2
        reward = (
            0.18 * (cell not in self.visited) - 0.006 - 0.055 * stalled
            + 3.0 * np.count_nonzero(state["keys"] > old["keys"])
            + 0.65 * len(opened) + 2.0 * locked
            + 0.5 * max(0, state["kills"] - old["kills"])
            - 0.015 * max(0.0, old["health"] - max(0.0, state["health"]))
            + 14.0 * state["exit_reached"] - 2.0 * state["dead"]
        )
        if self.targets:
            # This supervised distance signal is a training reward. Policy inputs use standard observations.
            goal, _ = next((target for target in self.targets
                            if target[1] is None or not old["keys"][target[1]]))
            distances = self.route_lengths[goal]
            old_distance = distances.get(self.cell(old, CELL))
            new_distance = distances.get(self.cell(state, CELL))
            if old_distance is not None and new_distance is not None:
                reward += 0.12 * (old_distance - new_distance)
        self.visited.add(cell)
        self.opened.update(opened)
        self.previous = state
        return float(reward), stalled


class FullMapDistance:
    """Training-only distance to a reachable key or exit on the current geometry."""

    def __init__(self, env):
        from training.wad_teacher import WadTeacher
        self.env = env
        self.exits = [goal for goal, _, _ in WadTeacher(env).exits]
        self.inventory = None

    def distance(self, state):
        from training.wad_teacher import KEY_NAMES, OFFSETS, RouteGrid, STRIDE
        inventory = tuple(state["keys"])
        if self.inventory != inventory:
            self.inventory = inventory
            grid = RouteGrid(self.env._geometry, inventory)
            self.grid = grid
            objects = self.env._game.get_state().objects
            goals = [np.array([o.position_x, o.position_y]) for o in objects
                     if o.name in KEY_NAMES and not inventory[KEY_NAMES.index(o.name)]]
            self.fields = []
            for targets in (goals, self.exits):
                sources = {cell for goal in targets for cell in grid.cells_near(goal)[:1]}
                field = dict.fromkeys(sources, 0)
                pending = deque(sources)
                while pending:
                    x, y = pending.popleft()
                    for dx, dy in OFFSETS:
                        cell = (x + dx, y + dy)
                        if cell in field or not grid.free[cell]:
                            continue
                        if dx and dy and not (grid.free[x + dx, y] and grid.free[x, y + dy]):
                            continue
                        field[cell] = field[x, y] + 1
                        pending.append(cell)
                self.fields.append(field)
        position = np.array([state["x"], state["y"]])
        cell = tuple(np.rint((position - self.grid.origin) / STRIDE).astype(int))
        return next(((i, field[cell]) for i, field in enumerate(self.fields) if cell in field), None)


@dataclass
class Lane:
    phase: str
    pool: list
    rng: np.random.Generator
    route_reward: bool = False
    env: DoomEnv | None = None
    course: Course | None = None

    def reset(self):
        if self.env is not None:
            self.env.close()
        path, name, ticks, self.course = self.pool[int(self.rng.integers(len(self.pool)))]
        self.map_label = f"{path.name}:{name}"
        self.env = DoomEnv(path, name, episode_ticks=ticks)
        self.obs, _ = self.env.reset(seed=int(self.rng.integers(1, 2**30)))
        state = self.env.privileged_state()
        self.progress = Progress(state, self.course)
        self.route = FullMapDistance(self.env) if self.route_reward and self.phase == "public" else None
        self.distance = self.route.distance(state) if self.route else None


def rollout(actor, reference, lanes, hidden, reference_hidden, reset, counters):
    records = []
    with torch.no_grad():
        for _ in range(ROLLOUT):
            features, mask = encode(actor, [lane.obs for lane in lanes])
            previous_hidden = hidden
            previous_reference = reference_hidden
            memory, hidden = actor.gru(features[:, None], hidden)
            _, reference_hidden = reference.gru(features[:, None], reference_hidden)
            latent = memory[:, 0]
            policy = distributions(actor, latent)
            discrete = torch.stack([head.sample() for head in policy[:3]], -1)
            buttons = policy[3].sample() * mask
            old_logp = log_probability(policy, discrete, buttons, mask)
            values = actor.critic(latent).squeeze(-1)
            rewards, dones = [], []
            for lane, action in zip(lanes, control_buttons(discrete, buttons, mask), strict=True):
                lane.obs, _, terminal, timeout, _ = lane.env.step(action)
                state = lane.env.privileged_state()
                old_keys = lane.progress.previous["keys"]
                shaped, stalled = lane.progress.step(state)
                if lane.route is not None and not (terminal or timeout):
                    distance = lane.route.distance(state)
                    if (distance and lane.distance and distance[0] == lane.distance[0]
                            and np.array_equal(old_keys, state["keys"])):
                        shaped += 0.08 * np.clip(lane.distance[1] - distance[1], -3, 3)
                    lane.distance = distance
                rewards.append(shaped)
                done = terminal or timeout
                dones.append(done)
                counters[lane.phase]["steps"] += 1
                counters[lane.phase]["stalled_steps"] += int(stalled)
                if done:
                    metrics = counters[lane.phase]
                    metrics["episodes"] += 1
                    metrics["exits"] += int(state["exit_reached"])
                    metrics["deaths"] += int(state["dead"])
                    metrics["keys"] += int(state["keys"].sum())
                    metrics["kills"] += state["kills"]
                    metrics["locks"] += int(sum(state["door_lock"][door] > 0 for door in lane.progress.opened))
                    if lane.phase in ("public", "exit_practice") and (state["exit_reached"] or state["keys"].any()):
                        print(f"{lane.phase} {lane.map_label} exit={state['exit_reached']} "
                              f"keys={int(state['keys'].sum())} kills={state['kills']}", flush=True)
                    lane.reset()
            records.append((features, mask, previous_hidden, previous_reference, reset, discrete, buttons,
                            old_logp, values, torch.tensor(rewards), torch.tensor(dones)))
            reset = records[-1][-1]
            hidden = hidden * (~reset)[None, :, None]
            reference_hidden = reference_hidden * (~reset)[None, :, None]
        features, _ = encode(actor, [lane.obs for lane in lanes])
        memory, _ = actor.gru(features[:, None], hidden)
        next_value = actor.critic(memory[:, 0]).squeeze(-1)
    columns = list(map(torch.stack, zip(*records, strict=True)))
    advantages = torch.empty_like(columns[8])
    carry = torch.zeros_like(next_value)
    for index in reversed(range(ROLLOUT)):
        mask = ~columns[10][index]
        delta = columns[9][index] + 0.995 * next_value * mask - columns[8][index]
        carry = delta + 0.995 * 0.96 * carry * mask
        advantages[index] = carry
        next_value = columns[8][index]
    targets = advantages + columns[8]
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    return columns, advantages, targets, hidden, reference_hidden, reset


def update(actor, reference, optimizer, columns, advantages, targets, chunk_length=CHUNK):
    features, mask, memory, ref_memory, reset, actions, buttons, old_logp, _, _, _ = columns
    n_env = features.shape[1]
    n_chunks = ROLLOUT // chunk_length

    def chunks(value):
        return value.transpose(0, 1).reshape(n_env * n_chunks, chunk_length, *value.shape[2:])

    initial = memory[::chunk_length].permute(2, 0, 1, 3).reshape(n_env * n_chunks, actor.num_gru_layers, -1)
    ref_initial = ref_memory[::chunk_length].permute(2, 0, 1, 3).reshape_as(initial)
    features, mask, reset, actions, buttons, old_logp, advantages, targets = map(
        chunks, (features, mask, reset, actions, buttons, old_logp, advantages, targets)
    )
    losses, divergences = [], []
    for _ in range(3):
        for batch in torch.randperm(n_env * n_chunks).split(8):
            latent = recurrent_features(actor.gru, features[batch], initial[batch].transpose(0, 1).contiguous(), reset[batch])
            policy = distributions(actor, latent)
            with torch.no_grad():
                old_latent = recurrent_features(reference.gru, features[batch],
                                                ref_initial[batch].transpose(0, 1).contiguous(), reset[batch])
                old_policy = distributions(reference, old_latent)
            divergence = policy_kl(old_policy, policy, mask[batch]).mean()
            ratio = torch.exp(log_probability(policy, actions[batch], buttons[batch], mask[batch]) - old_logp[batch])
            clipped_reward = torch.minimum(ratio * advantages[batch], ratio.clamp(0.85, 1.15) * advantages[batch])
            value = actor.critic(latent.detach()).squeeze(-1)
            entropy = sum(head.entropy() for head in policy[:3]) + (policy[3].entropy() * mask[batch]).sum(-1)
            loss = (-clipped_reward.mean() + 1.2 * divergence - 0.002 * entropy.mean()
                    + 0.5 * F.smooth_l1_loss(value, targets[batch]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
            divergences.append(float(divergence.detach()))
    return float(np.mean(losses)), float(np.mean(divergences))


def pools(directory, scythe=None, requiem=None):
    training = {phase: [] for phase in PHASES}
    for index, phase in enumerate(dict.fromkeys(PHASES[:-2])):
        for offset in range(24):
            course = make_course(directory, 30000 + 100 * index + offset, phase,
                                 long_route=phase in ("key", "gauntlet") and offset >= 6)
            training[phase].append((course.path, "MAP01", 1800, course))
    training["public"] = [(Path(IWAD), name, 2100, None) for name in PUBLIC_MAPS]
    for wad, names in ((scythe, SCYTHE_MAPS), (requiem, REQUIEM_MAPS)):
        if wad is not None:
            training["public"].extend((wad, name, 2100, None) for name in names)
    return training


def exit_practice(directory, scythe, requiem):
    """Verify training-only spawn changes on known WADs before using them for RL."""
    from env.doom_env.wad import read_wad
    from training.train_wad_exits import relocate, spawn_variants
    from training.wad_teacher import WadTeacher

    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "routes.json"
    if manifest.exists():
        return [(directory / file, name, 1000, None) for file, name in json.loads(manifest.read_text())]

    specifications = ((Path(IWAD), ("MAP05", "MAP06", "MAP08")),
                      (scythe, ("MAP03", "MAP09")), (requiem, ("MAP06",)))
    saved = []
    for wad, names in specifications:
        if wad is None:
            continue
        lumps = read_wad(wad)
        for name in names:
            source_env = DoomEnv(wad, name, render_buffers=False)
            try:
                source_env.reset(seed=2718)
                positions = spawn_variants(source_env, 2718)
            finally:
                source_env.close()
            successes = 0
            for i, start in enumerate(positions):
                path = directory / f"{wad.stem.lower()}-{name}-{i}.wad"
                relocate(lumps, name, start, path)
                env = DoomEnv(path, name, episode_ticks=1000, render_buffers=False)
                try:
                    obs, _ = env.reset(seed=2719 + i)
                    teacher = WadTeacher(env, exit_only=True)
                    for _ in range(250):
                        action = teacher.act(obs, env.privileged_state())
                        obs, _, done, timed_out, info = env.step(action)
                        if done or timed_out:
                            break
                    if info["exit_reached"]:
                        saved.append((path.name, name))
                        successes += 1
                    else:
                        path.unlink()
                    if successes >= 2:
                        break
                finally:
                    env.close()
    if not saved:
        raise RuntimeError("Could not verify any exit practice routes")
    manifest.write_text(json.dumps(saved))
    return [(directory / file, name, 1000, None) for file, name in saved]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=72)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scythe", type=Path)
    parser.add_argument("--requiem", type=Path)
    parser.add_argument("--exit-practice", type=Path,
                        help="Use verified end-of-level starts for one of the two public-WAD lanes")
    parser.add_argument("--full-map-focus", action="store_true",
                        help="Use four original-spawn lanes and retain key and gauntlet practice")
    parser.add_argument("--sequence-length", type=int, choices=(24, 48, 96), default=24)
    parser.add_argument("--route-reward", action=argparse.BooleanOptionalAction, default=None,
                        help="Use training-only key/exit distances (default on for full-map focus)")
    parser.add_argument("--keep-critic", action="store_true", help="Continue the value estimates from a PPO checkpoint")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(8902)
    weights = torch.load(args.warm_start, weights_only=True, map_location="cpu")
    prior_decisions = weights.get("cumulative_rl_decisions", weights.get("training_decisions", 0))
    actor = SpatialEgoNet(use_rgb=weights.get("use_rgb", False))
    actor.load_state_dict(weights["model_state_dict"])
    actor.requires_grad_(False)
    policy_modules = (actor.gru, actor.head_move_fb, actor.head_move_lr, actor.head_turn, actor.head_binary)
    for module in (*policy_modules, actor.critic):
        module.requires_grad_(True)
    if not args.keep_critic:
        nn.init.zeros_(actor.critic[-1].weight)
        nn.init.zeros_(actor.critic[-1].bias)
    reference = deepcopy(actor).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW([
        {"params": [p for module in policy_modules for p in module.parameters()], "lr": 2e-5},
        {"params": actor.critic.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    training = pools(args.work_dir, args.scythe, args.requiem)
    phases = PHASES
    if args.exit_practice is not None:
        training["public"] = [(wad, name, 4200, course) for wad, name, _, course in training["public"]]
        training["exit_practice"] = exit_practice(args.exit_practice, args.scythe, args.requiem)
        phases = (*PHASES[:-1], "exit_practice")
        print(f"Verified exit practice starts: {len(training['exit_practice'])}", flush=True)
    if args.full_map_focus:
        if args.exit_practice is not None:
            parser.error("--full-map-focus and --exit-practice use different lane mixes")
        training["public"] = [(wad, name, 4200, course) for wad, name, _, course in training["public"]]
        phases = ("key", "gauntlet", "public", "public", "public", "public")
    route_reward = args.full_map_focus if args.route_reward is None else args.route_reward
    lanes = [Lane(phase, training[phase], np.random.default_rng(8902 + i), route_reward) for i, phase in enumerate(phases)]
    hidden = actor.initial_hidden_state(len(lanes))
    ref_hidden = hidden.clone()
    reset = torch.ones(len(lanes), dtype=torch.bool)
    counters = {phase: Counter() for phase in training}
    started = time.monotonic()
    try:
        for lane in lanes:
            lane.reset()
        for step in range(1, args.updates + 1):
            data, advantages, targets, hidden, ref_hidden, reset = rollout(
                actor, reference, lanes, hidden, ref_hidden, reset, counters
            )
            loss, drift = update(actor, reference, optimizer, data, advantages, targets, args.sequence_length)
            if not np.isfinite(loss) or not np.isfinite(drift):
                raise RuntimeError("Non-finite RL training loss")
            decisions = step * ROLLOUT * len(lanes)
            print(f"update {step}/{args.updates} decisions={decisions} loss={loss:.3f} "
                  f"anchor_kl={drift:.4f} elapsed={time.monotonic()-started:.1f}s", flush=True)
            if step % 12 == 0 or step == args.updates:
                print(json.dumps(counters), flush=True)
                torch.save({"model_state_dict": actor.state_dict(), "use_rgb": actor.use_rgb,
                            "training_decisions": decisions, "training_seed": 8902,
                            "sample_actions": args.full_map_focus, "sequence_length": args.sequence_length,
                            "cumulative_rl_decisions": prior_decisions + decisions,
                            "warm_start": args.warm_start.name,
                            "training_wads": [(path.name, name) for phase in ("public", "exit_practice")
                                              for path, name, *_ in training.get(phase, [])]},
                           args.work_dir / f"rl-{step:03d}.pt")
    finally:
        for lane in lanes:
            if lane.env is not None:
                lane.env.close()


if __name__ == "__main__":
    main()
