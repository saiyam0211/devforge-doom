#!/usr/bin/env python
"""Train Spatial-EgoNet from demonstrations collected in ViZDoom.

The teacher uses waypoints on the two guided maps and explores other layouts using
spatial observations. The expanded curriculum samples key and locked-door windows
more often and reflects sequences left-to-right to diversify recurrent training.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

import vizdoom as vd
from env.doom_env.wrapper import DoomEnv, IWAD
from submission.model import SpatialEgoNet


# -----------------------------------------------------------------------------
# 1. Multi-Map Algorithmic Expert Teacher
# -----------------------------------------------------------------------------
class DoomExpertTeacher:
    """Algorithmic teacher operating strictly during offline training to generate expert demonstrations."""

    def __init__(self):
        self.wad_name = ""
        self.map_name = ""
        self.waypoints: List[Tuple[float, float]] = []
        self.reset()

    def reset(self, wad_name: str = "", map_name: str = "", jitter: bool = False):
        self.wad_name = wad_name
        self.map_name = map_name
        self.step_idx = 0
        self.stuck_steps = 0
        self.escape_turn_steps = 0
        self.escape_dir = 1
        self.door_timer = 0
        self.use_cooldown = 0
        self.last_act_fwd = 0
        self.wp_idx = 0

        if "two_rooms" in self.wad_name:
            base_wp = [
                (350.0, 350.0),
                (430.0, 352.0),  # Door 1
                (540.0, 350.0),
                (750.0, 350.0),
                (750.0, 96.0),
                (672.0, 96.0),   # Exit Trigger
            ]
        elif "freedoom2" in self.wad_name and self.map_name == "MAP01":
            base_wp = [
                # 1. Start room to Hallway
                (-32.0, -128.0),
                (64.0, -128.0),
                (160.0, -128.0),
                (256.0, -128.0),
                (320.0, -128.0),
                (380.0, -140.0),
                # 2. Shotgun pickup & Door 1
                (448.0, -160.0),   # Shotgun
                (544.0, -160.0),
                (704.0, -160.0),   # Door 1 (Line 167) [Index 8]
                (760.0, -160.0),
                # 3. High North Bridge across Courtyard
                (800.0, 0.0),
                (850.0, 150.0),
                (1100.0, 200.0),
                (1350.0, 150.0),
                (1408.0, -40.0),
                # 4. East Corridor & Stimpack
                (1440.0, -80.0),
                (1472.0, -80.0),
                (1536.0, -80.0),
                (1600.0, -80.0),
                (1664.0, -80.0),
                (1736.0, -88.0),   # Stimpack
                (1792.0, -160.0),
                (1824.0, -160.0),  # Door 2 (Line 332) [Index 22]
                # 5. Long South Corridor
                (1950.0, -200.0),
                (1950.0, -400.0),
                (1950.0, -700.0),
                (1930.0, -830.0),  # Top of curved staircase [Index 26]
                # 6. Curving Downwards Staircase into Lower Corridor
                (1870.0, -860.0),  # [Index 27]
                (1800.0, -860.0),
                (1740.0, -856.0),
                (1680.0, -856.0),
                (1600.0, -856.0),  # Clear corner blocking wall at x=1696 [Index 31]
                # 7. Lower West Corridor
                (1550.0, -950.0),  # [Index 32]
                (1400.0, -1000.0), # [Index 33]
                (1200.0, -1024.0),
                (900.0, -1024.0),
                (672.0, -1024.0),
                (648.0, -1024.0),  # Door 3 (Line 876) [Index 37]
                (608.0, -1024.0),
                # 8. Hallway to Exit Booth (smooth cornering & centered entrance)
                (450.0, -1024.0),
                (320.0, -1024.0),
                (240.0, -980.0),   # Diagonally cut corner into exit corridor [Index 41]
                (216.0, -880.0),
                (216.0, -740.0),
                (216.0, -670.0),
                (216.0, -636.0),   # Exit Switch (Line 774) [Index 45]
            ]
        else:
            base_wp = []

        if jitter and base_wp:
            self.waypoints = []
            for i, (wx, wy) in enumerate(base_wp):
                # Never jitter critical door thresholds, narrow staircase, or the exit switch itself
                if ("two_rooms" in self.wad_name and i in (1, 5)) or ("freedoom2" in self.wad_name and (i in (8, 22, 37, 44, 45) or 26 <= i <= 34)):
                    self.waypoints.append((wx, wy))
                else:
                    jx = float(np.clip(np.random.normal(0, 4.0), -8.0, 8.0))
                    jy = float(np.clip(np.random.normal(0, 4.0), -8.0, 8.0))
                    self.waypoints.append((wx + jx, wy + jy))
        else:
            self.waypoints = base_wp

    def act(self, obs: Dict[str, Any], priv: Dict[str, Any] | None = None) -> np.ndarray:
        self.step_idx += 1
        act = np.zeros(9, dtype=np.int8)

        rays = obs["rays"]["distance"]
        hits = obs["rays"]["hit_type"]
        vitals = obs["vitals"]
        entities = obs["visible_entities"]

        front_d = float(rays[0])
        vel_fwd = float(vitals[14])
        hp = float(vitals[0])

        mask = entities["mask"]
        types = entities["entity_type"]
        feats = entities["features"]

        # =========================================================================
        # Case 1: Guided Map Waypoint Navigation (for two_rooms and freedoom2 MAP01)
        # =========================================================================
        if self.waypoints and priv is not None:
            px, py, pang = priv["x"], priv["y"], priv["angle"]
            tx, ty = self.waypoints[self.wp_idx]

            dist = math.hypot(tx - px, ty - py)
            adv_threshold = 16.0 if self.wp_idx >= len(self.waypoints) - 2 else (40.0 if "two_rooms" in self.wad_name else 48.0)
            if dist < adv_threshold and self.wp_idx < len(self.waypoints) - 1:
                self.wp_idx += 1
                tx, ty = self.waypoints[self.wp_idx]
                dist = math.hypot(tx - px, ty - py)

            target_deg = math.degrees(math.atan2(ty - py, tx - px)) % 360.0
            angle_diff = (target_deg - pang + 180.0) % 360.0 - 180.0

            # Steering
            if angle_diff > 8.0:
                act[4] = 1
            elif angle_diff < -8.0:
                act[5] = 1

            # Throttle
            if abs(angle_diff) < 35.0:
                act[0] = 1

            # Dynamic Combat Engagement (Prioritize close blockers, keep distance)
            enemies = [i for i in range(len(mask)) if mask[i] == 1 and types[i] in (1, 2, 3)]
            if enemies:
                close_enemies = [i for i in enemies if feats[i, 0] * 1024.0 < 280.0 and feats[i, 2] > 0.0]
                if close_enemies:
                    best_e = min(close_enemies, key=lambda i: feats[i, 0])
                    e_dist = feats[best_e, 0] * 1024.0
                    e_sin = feats[best_e, 1]
                    e_cos = feats[best_e, 2]

                    # Fast aim lock on close frontal enemies
                    if e_sin > 0.05:
                        act[4] = 1
                        act[5] = 0
                    elif e_sin < -0.05:
                        act[5] = 1
                        act[4] = 0

                    # Throttle gating: do not run into enemy face, but keep pressure
                    if e_dist < 140.0:
                        act[0] = 0
                        if e_dist < 80.0:
                            act[1] = 1
                    elif abs(angle_diff) < 35.0:
                        act[0] = 1

                    # Safety Check: NEVER fire if an explosive barrel (type 10) is nearby
                    has_barrel = any(
                        mask[j] == 1 and types[j] == 10 and (
                            (feats[j, 0] * 1024.0 < 160.0) or
                            (feats[j, 0] * 1024.0 < 350.0 and feats[j, 2] > 0.4 and abs(feats[j, 1]) < 0.35)
                        )
                        for j in range(len(mask))
                    )

                    # Fire weapon when target is centered in FOV and safe
                    if not has_barrel and e_cos > 0.4 and abs(e_sin) < 0.28:
                        act[6] = 1

            # Door & Switch interaction
            front_d = obs["rays"]["distance"][0]
            hits = obs["rays"]["hit_type"]
            is_door_ahead = hits[0] in (2, 3) or hits[63] in (2, 3) or hits[1] in (2, 3) or hits[2] in (2, 3) or hits[62] in (2, 3)

            # Special door/switch trigger positions
            if "two_rooms" in self.wad_name:
                if px > 405.0 and px < 442.0 and front_d < 0.12:
                    if self.use_cooldown == 0:
                        act[7] = 1
                        self.use_cooldown = 10
            elif "freedoom2" in self.wad_name:
                # Door waypoints: Door 1 (8), Door 2 (22), Door 3 (37)
                if self.wp_idx in (8, 22, 37) and dist < 85.0:
                    if self.use_cooldown == 0:
                        act[7] = 1
                        self.use_cooldown = 10
                # Exit switch: waypoint >= 44 (Line 774)
                elif self.wp_idx >= 44 and dist < 85.0:
                    if self.use_cooldown == 0:
                        act[7] = 1
                        self.use_cooldown = 2
                    act[0] = 1  # drive forward into switch face

            if is_door_ahead and front_d < 0.20 and self.use_cooldown == 0:
                act[7] = 1
                self.use_cooldown = 12

            if self.use_cooldown > 0:
                self.use_cooldown -= 1

            self.last_act_fwd = act[0]
            return act

        # =========================================================================
        # Case 2: Autonomous Generalized Navigation (Unseen & Arbitrary Maps)
        # =========================================================================
        self.use_cooldown = max(0, self.use_cooldown - 1)
        if self.door_timer:
            self.door_timer -= 1
            act[0] = 1
            self.last_act_fwd = 1
            return act

        # Stuck detection
        if self.last_act_fwd and (vel_fwd < 0.5 or front_d < 0.08) and self.step_idx > 5:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        # E. Committed Stuck Recovery (Top Priority when physically stalled against obstacle)
        if self.escape_turn_steps > 0:
            self.escape_turn_steps -= 1
            act[1] = 1
            if self.escape_dir == 1:
                act[4] = 1
            else:
                act[5] = 1
            self.last_act_fwd = 0
            return act

        if self.stuck_steps >= 3:
            self.escape_turn_steps = 6
            best_r = int(np.argmax(rays))
            self.escape_dir = 1 if best_r < 32 else 2
            act[1] = 1
            if self.escape_dir == 1:
                act[4] = 1
            else:
                act[5] = 1
            self.last_act_fwd = 0
            return act

        # A. Keycard Acquisition (Top Priority for Progression)
        key_idx = [i for i in range(len(mask)) if mask[i] == 1 and types[i] == 9]
        if key_idx:
            best_k = min(key_idx, key=lambda i: feats[i, 0])
            sin_k = feats[best_k, 1]
            act[0] = 1
            if abs(sin_k) > 0.05:
                if sin_k > 0:
                    act[4] = 1
                else:
                    act[5] = 1
            self.last_act_fwd = act[0]
            return act

        # B. Urgent Health Collection if low HP
        if hp < 60:
            health_items = [i for i in range(len(mask)) if mask[i] == 1 and types[i] in (5, 6)]
            if health_items:
                best_h = min(health_items, key=lambda i: feats[i, 0])
                sin_h = feats[best_h, 1]
                act[0] = 1
                if sin_h > 0.05:
                    act[4] = 1
                elif sin_h < -0.05:
                    act[5] = 1
                self.last_act_fwd = act[0]
                return act

        # C. Door Interaction & Aperture Passage
        has_key = vitals[8:14].any()
        door_rays = [
            r for r in (*range(58, 64), *range(7))
            if hits[r] == 2 or (hits[r] == 3 and has_key)
        ]
        if door_rays:
            min_door_d = min(rays[r] for r in door_rays)
            best_door_r = min(door_rays, key=lambda r: rays[r])
            act[0] = 1
            if best_door_r not in (63, 0, 1):
                act[4 if best_door_r < 32 else 5] = 1
            elif min_door_d < 0.055 and self.use_cooldown == 0:
                act[7] = 1
                self.use_cooldown = 12
                self.door_timer = 10
            self.last_act_fwd = 1
            return act

        # D. Tactical Combat: Aim, Fire, and Strafe-Dodge Visible Monsters
        enemy_idx = [i for i in range(len(mask)) if mask[i] == 1 and types[i] in (1, 2, 3)]
        if enemy_idx:
            frontal = [i for i in enemy_idx if feats[i, 2] > 0.0]
            targets = frontal if frontal else enemy_idx
            best_e = min(targets, key=lambda i: feats[i, 0])
            dist_e = feats[best_e, 0]
            sin_a = feats[best_e, 1]
            cos_a = feats[best_e, 2]

            if abs(sin_a) < 0.30 and cos_a > 0.4:
                act[6] = 1  # ATTACK!

            if sin_a > 0.05:
                act[4] = 1
            elif sin_a < -0.05:
                act[5] = 1

            if (self.step_idx // 8) % 2 == 0:
                act[2] = 1
            else:
                act[3] = 1

            if dist_e > 0.18 and front_d > 0.12:
                act[0] = 1
            elif dist_e < 0.10:
                act[1] = 1
            self.last_act_fwd = act[0]
            return act

        # F. Spatial Exploration: Seek Maximum Clearance Angle
        front_cone = list(range(58, 64)) + list(range(0, 7))
        best_front = front_cone[int(np.argmax([rays[r] for r in front_cone]))]

        if front_d > 0.12:
            act[0] = 1
            if best_front != 0:
                if best_front < 32:
                    act[4] = 1
                else:
                    act[5] = 1
        else:
            best_360 = int(np.argmax(rays))
            if best_360 < 32:
                act[4] = 1
            else:
                act[5] = 1

        self.last_act_fwd = act[0]
        return act


# -----------------------------------------------------------------------------
# 2. Expert Trajectory Dataset
# -----------------------------------------------------------------------------
class DoomTrajectoryDataset(Dataset):
    def __init__(self, sequences: List[Dict[str, np.ndarray]], seq_len: int = 32, diversify: bool = False,
                 include_short: bool = False):
        self.sequences = sequences
        self.seq_len = seq_len
        self.diversify = diversify
        self.include_short = include_short

        # Keep starter-map practice without letting it dominate the expanded curriculum.
        map_indices: Dict[str, List[Tuple[int, int]]] = {}
        for s_idx, seq in enumerate(sequences):
            m_key = "two_rooms" if "two_rooms" in seq.get("wad_name", "") else "freedoom"
            if m_key not in map_indices:
                map_indices[m_key] = []
            n_steps = len(seq["actions"])
            stride = max(1, seq_len // 2)
            starts = list(range(0, n_steps - seq_len + 1, stride))
            if include_short and 0 < n_steps < seq_len:
                starts = [0]
                if diversify:
                    # Give the terminal actions in short exit clips a second pass.
                    starts.append(max(0, 2 * n_steps - seq_len))
            elif diversify and starts and starts[-1] != n_steps - seq_len:
                starts.append(n_steps - seq_len)
            for t in starts:
                map_indices[m_key].append((s_idx, t))
                if diversify:
                    window = slice(t, t + seq_len)
                    sees_key = (seq["ent_type"][window] == 9).any()
                    at_lock = (seq["ray_hit"][window, 0] == 3).any()
                    if sees_key or (at_lock and seq["vitals"][window, 8:14].any()):
                        map_indices[m_key].extend([(s_idx, t)] * 2)

        max_count = max(len(idxs) for idxs in map_indices.values()) if map_indices else 0
        self.indices: List[Tuple[int, int]] = []
        for m_key, idxs in map_indices.items():
            if not idxs:
                continue
            count = max_count
            if diversify:
                count = max(len(idxs), max_count // 4) if m_key == "two_rooms" else len(idxs)
            reps = (count + len(idxs) - 1) // len(idxs)
            balanced_idxs = (idxs * reps)[:count]
            self.indices.extend(balanced_idxs)
            print(f"  [Dataset] {m_key}: {len(idxs)} original windows -> balanced to {len(balanced_idxs)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s_idx, start = self.indices[idx]
        seq = self.sequences[s_idx]
        end = start + self.seq_len

        sample = {
            "ray_dist": torch.from_numpy(seq["ray_dist"][start:end]).float(),
            "ray_hit": torch.from_numpy(seq["ray_hit"][start:end]).long(),
            "ent_feat": torch.from_numpy(seq["ent_feat"][start:end]).float(),
            "ent_type": torch.from_numpy(seq["ent_type"][start:end]).long(),
            "ent_mask": torch.from_numpy(seq["ent_mask"][start:end]).float(),
            "vitals": torch.from_numpy(seq["vitals"][start:end]).float(),
            "actions": torch.from_numpy(seq["actions"][start:end]).long(),
        }
        if "rgb" in seq:
            sample["rgb"] = torch.from_numpy(seq["rgb"][start:end])
        if self.diversify and torch.rand(()).item() < 0.5:
            # Reflect the entire recurrent window in ego space.
            for name in ("ray_dist", "ray_hit"):
                sample[name] = sample[name].flip(-1).roll(1, dims=-1)
            sample["ent_feat"] = sample["ent_feat"].clone()
            sample["ent_feat"][..., [1, 4]] *= -1
            sample["vitals"] = sample["vitals"].clone()
            sample["vitals"][..., 15] *= -1
            sample["actions"] = sample["actions"][..., [0, 1, 3, 2, 5, 4, 6, 7, 8]]
            if "rgb" in sample:
                sample["rgb"] = sample["rgb"].flip(-2)
        if self.include_short:
            frames = len(sample["actions"])
            if frames < self.seq_len:
                sample = {name: torch.cat((data, data[-1:].expand(self.seq_len - frames, *data.shape[1:])))
                          for name, data in sample.items()}
            sample["valid"] = torch.arange(self.seq_len) < frames
        return sample


def collect_demonstrations(
    maps_config: List[Tuple[str, str, int]],
    diversify: bool = False,
    use_rgb: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Collects multi-map expert demonstrations using the teacher policy."""
    teacher = DoomExpertTeacher()
    demonstrations: List[Dict[str, np.ndarray]] = []
    total_steps = 0
    total_exits = 0

    print(f"\n[1/3] Collecting Expert Demonstrations across {len(maps_config)} curricula...")
    for wad_path, map_name, target_episodes in maps_config:
        wad_name = Path(wad_path).name
        try:
            env = DoomEnv(wad_path, map_name, render_buffers=use_rgb)
        except Exception as e:
            print(f"  Skipping {wad_name}:{map_name} ({e})")
            continue

        is_guided = "two_rooms" in wad_name or map_name == "MAP01"
        ep_completed = 0
        key_episodes = 0
        map_exits = 0
        attempts = 0
        max_attempts = target_episodes * 4

        while ep_completed < target_episodes and attempts < max_attempts:
            attempts += 1
            seed = ep_completed * 3 + attempts
            obs, info = env.reset(seed=seed)
            apply_jitter = (ep_completed % 2 == 1)
            teacher.reset(wad_name, map_name, jitter=apply_jitter)

            ray_dists, ray_hits = [], []
            ent_feats, ent_types, ent_masks = [], [], []
            vitals_list, actions_list = [], []
            rgb_frames = []
            exit_flag = False
            visited = set()

            for step in range(525):
                priv = env.privileged_state()
                visited.add((int(priv["x"] // 128), int(priv["y"] // 128)))
                act = teacher.act(obs, priv)

                ray_dists.append(obs["rays"]["distance"])
                ray_hits.append(obs["rays"]["hit_type"])
                ent_feats.append(obs["visible_entities"]["features"])
                ent_types.append(obs["visible_entities"]["entity_type"])
                ent_masks.append(obs["visible_entities"]["mask"])
                vitals_list.append(obs["vitals"])
                actions_list.append(act)
                if use_rgb:
                    rgb_frames.append(obs["rgb"][1::4, 1::4].copy())

                obs, r, term, trunc, info = env.step(act)
                total_steps += 1

                if info.get("exit_reached", False):
                    exit_flag = True
                    total_exits += 1

                if term or trunc:
                    break

            # Require an exit on guided maps and keep evidence of progress on expanded maps.
            valid = exit_flag if is_guided else (len(actions_list) >= 150)
            final = env.privileged_state()
            if diversify and not is_guided:
                valid = exit_flag or bool(final["keys"].any()) or (len(visited) >= 8 and final["kills"] >= 2)
            if valid and len(actions_list) >= (24 if diversify else 16):
                ep_completed += 1
                map_exits += int(exit_flag)
                key_episodes += int(final["keys"].any())
                seq = {
                    "wad_name": wad_name,
                    "ray_dist": np.array(ray_dists, dtype=np.float32),
                    "ray_hit": np.array(ray_hits, dtype=np.int64),
                    "ent_feat": np.array(ent_feats, dtype=np.float32),
                    "ent_type": np.array(ent_types, dtype=np.int64),
                    "ent_mask": np.array(ent_masks, dtype=np.float32),
                    "vitals": np.array(vitals_list, dtype=np.float32),
                    "actions": np.array(actions_list, dtype=np.int8),
                }
                if use_rgb:
                    seq["rgb"] = np.stack(rgb_frames)
                demonstrations.append(seq)

        env.close()
        print(f"  {wad_name}:{map_name}: {ep_completed}/{attempts} demonstrations, "
              f"{map_exits} exits, {key_episodes} with keys")

    print(f"\nTeacher environment steps: {total_steps} | Exits during collection: {total_exits}\n")
    return demonstrations


# -----------------------------------------------------------------------------
# 3. Supervised Behavioral Cloning Trainer
# -----------------------------------------------------------------------------
def train_behavioral_cloning(
    model: SpatialEgoNet,
    demonstrations: List[Dict[str, np.ndarray]],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 6e-4,
    device: str = "cpu",
    diversify: bool = False,
    sequence_length: int | None = None,
    include_short: bool = False,
) -> SpatialEgoNet:
    print(f"[2/3] Training Spatial-EgoNet with Behavioral Cloning ({epochs} epochs on {device})...")
    dataset = DoomTrajectoryDataset(demonstrations, seq_len=sequence_length or (24 if diversify else 32),
                                    diversify=diversify, include_short=include_short)
    if not len(dataset):
        raise ValueError("No complete training windows were collected")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Class weights for binary actions: ATTACK (pos_weight 3.0), USE (pos_weight 3.0), WEAPON (pos_weight 0.2)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0, 3.0, 0.2]).to(device))
    weight_fb = torch.tensor([1.2, 1.0, 1.2]).to(device)
    weight_turn = torch.tensor([0.8, 1.2, 1.2]).to(device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        acc_fb, acc_lr, acc_turn, acc_use, acc_atk = 0.0, 0.0, 0.0, 0.0, 0.0
        batches = 0

        for b in loader:
            optimizer.zero_grad()
            B, T = b["ray_dist"].shape[:2]

            # Ray data augmentation: small perturbation noise during training
            ray_dist_aug = b["ray_dist"].clone()
            if model.training:
                noise = torch.randn_like(ray_dist_aug) * 0.008
                ray_dist_aug = (ray_dist_aug + noise).clamp(0.0, 1.0)

            obs_b = {
                "rays": {
                    "distance": ray_dist_aug.view(B * T, 64).to(device),
                    "hit_type": b["ray_hit"].view(B * T, 64).to(device),
                },
                "visible_entities": {
                    "features": b["ent_feat"].view(B * T, 32, 5).to(device),
                    "entity_type": b["ent_type"].view(B * T, 32).to(device),
                    "mask": b["ent_mask"].view(B * T, 32).to(device),
                },
                "vitals": b["vitals"].view(B * T, 17).to(device),
            }
            if model.use_rgb:
                obs_b["rgb"] = b["rgb"].reshape(B * T, 30, 40, 3).to(device)

            features = model.extract_features(obs_b)
            features_seq = features.view(B, T, -1)

            # Pass through 2-layer Spatial GRU
            h0 = model.initial_hidden_state(batch_size=B, device=device)
            gru_out, _ = model.gru(features_seq, h0)
            rnn_feats = gru_out.contiguous().view(B * T, -1)
            actions = b["actions"].view(B * T, 9).to(device)
            if include_short:
                valid = b["valid"].flatten().to(device)
                rnn_feats = rnn_feats[valid]
                actions = actions[valid]

            # Action Heads
            logits_fb = model.head_move_fb(rnn_feats)
            logits_lr = model.head_move_lr(rnn_feats)
            logits_turn = model.head_turn(rnn_feats)
            logits_bin = model.head_binary(rnn_feats)

            # Targets
            target_fb = torch.zeros(len(actions), dtype=torch.long, device=device)
            target_fb[actions[:, 0] == 1] = 1
            target_fb[actions[:, 1] == 1] = 2

            target_lr = torch.zeros(len(actions), dtype=torch.long, device=device)
            target_lr[actions[:, 2] == 1] = 1
            target_lr[actions[:, 3] == 1] = 2

            target_turn = torch.zeros(len(actions), dtype=torch.long, device=device)
            target_turn[actions[:, 4] == 1] = 1
            target_turn[actions[:, 5] == 1] = 2

            target_bin = actions[:, 6:9].float()

            l_fb = F.cross_entropy(logits_fb, target_fb, weight=weight_fb)
            l_lr = F.cross_entropy(logits_lr, target_lr)
            l_turn = F.cross_entropy(logits_turn, target_turn, weight=weight_turn)
            l_bin = bce_loss(logits_bin, target_bin)

            loss = l_fb + l_lr + l_turn + 2.0 * l_bin
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            acc_fb += (logits_fb.argmax(dim=-1) == target_fb).float().mean().item()
            acc_lr += (logits_lr.argmax(dim=-1) == target_lr).float().mean().item()
            acc_turn += (logits_turn.argmax(dim=-1) == target_turn).float().mean().item()

            pred_bin = (torch.sigmoid(logits_bin) > 0.5).float()
            acc_atk += (pred_bin[:, 0] == target_bin[:, 0]).float().mean().item()
            acc_use += (pred_bin[:, 1] == target_bin[:, 1]).float().mean().item()
            batches += 1

        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss / batches:.4f} | "
                f"Acc FB: {acc_fb / batches * 100:.1f}% | "
                f"Acc Turn: {acc_turn / batches * 100:.1f}% | "
                f"Acc USE: {acc_use / batches * 100:.1f}% | "
                f"Acc ATK: {acc_atk / batches * 100:.1f}%"
            )

    print("Behavioral Cloning Pretraining Complete!\n")
    return model


# -----------------------------------------------------------------------------
# 4. Main Training Pipeline Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid Behavioral Cloning DOOM Pipeline")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--expanded", action="store_true", help="Expand the curriculum and mirror demonstrations")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "submission/weights/agent_weights.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    two_rooms_path = str((ROOT / "maps" / "two_rooms.wad").resolve())
    freedoom2_path = IWAD

    # Curricula: (WAD path, Map name, Target episodes)
    # Focused high-quality curriculum: 100% exit completion on two_rooms and freedoom2 MAP01 across varied seeds
    training_curricula = [
        (two_rooms_path, "MAP01", 25),
        (freedoom2_path, "MAP01", 50),
    ]
    if args.expanded:
        # MAP19/23/26/29/30 are frozen holdouts. MAP25 was previously demoed.
        training_curricula = [(two_rooms_path, "MAP01", 12), (freedoom2_path, "MAP01", 16)]
        training_curricula += [
            (freedoom2_path, f"MAP{number:02d}", 3)
            for number in (2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15, 17, 18, 20, 22, 24, 27, 28)
        ]
        training_curricula.append((str(ROOT / "tests/fixtures/e7ab54fb6834ffa3.wad"), "MAP01", 6))

    # 1. Collect Demonstrations
    demonstrations = collect_demonstrations(training_curricula, diversify=args.expanded)

    # 2. Instantiate Spatial-EgoNet Architecture
    model = SpatialEgoNet(hidden_dim=256, num_gru_layers=2)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    # 3. Supervised Behavioral Cloning
    model = train_behavioral_cloning(
        model=model,
        demonstrations=demonstrations,
        epochs=args.epochs,
        batch_size=32,
        lr=2e-4 if args.warm_start else 6e-4,
        device=args.device,
        diversify=args.expanded,
    )

    # 4. Save Export Weights directly to submission/weights/agent_weights.pt
    export_path = args.output
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Move to CPU for universal serialization
    model.to("cpu")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": 256,
            "num_gru_layers": 2,
            "timestamp": time.time(),
            "arch": "Spatial-EgoNet-EgoCentric-Spatial-Encoding",
            "training_maps": [(Path(wad).name, name) for wad, name, _ in training_curricula],
            "training_seed": args.seed,
        },
        export_path,
    )
    file_size_mb = export_path.stat().st_size / (1024 * 1024)
    print(f"[3/3] Exported Hardened Production Weights to {export_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
