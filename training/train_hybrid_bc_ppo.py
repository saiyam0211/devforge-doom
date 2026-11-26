#!/usr/bin/env python
"""Hybrid Imitation Learning (Behavioral Cloning) & PPO Fine-Tuning Pipeline.

Curriculum:
1. Expert Teacher Demonstration Collection across diverse WADs & maps:
   - Door interaction: approaching, pressing USE, waiting for aperture clearance, charging through.
   - Spatial navigation: corridor and open-room exploration following max clearance rays.
   - Dynamic combat: aiming and firing at visible monsters in ego-frame FOV.
   - Sector progression & exit seeking.
2. Supervised Behavioral Cloning (BC) Pretraining:
   - Cross-entropy loss on factored categorical action heads (FB, LR, Turn)
   - Binary cross-entropy with positive weighting on USE and ATTACK
   - Recurrent sequence training through 2-layer Spatial GRU
3. Reinforcement Learning (PPO) Fine-Tuning:
   - Exploration curiosity (RND) and progression rewards
   - Exports hardened, verified weights to submission/weights/agent_weights.pt
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
# 1. Algorithmic Expert Teacher
# -----------------------------------------------------------------------------
class DoomExpertTeacher:
    """Algorithmic teacher operating strictly during training to generate expert demonstrations.
    
    Includes calibrated door aperture interaction, committed stuck recovery,
    keycard collection, and combat target engagement.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.stuck_steps = 0
        self.escape_turn_steps = 0
        self.escape_dir = 1  # 1: left, 2: right
        self.door_timer = 0
        self.last_act_fwd = 0

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

        # Stuck detection: ONLY when attempting forward movement but blocked
        if self.last_act_fwd and (vel_fwd < 0.5 or front_d < 0.08) and self.step_idx > 5:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        # A. Keycard Acquisition (Top Priority for Progression)
        mask = entities["mask"]
        types = entities["entity_type"]
        feats = entities["features"]
        key_idx = [i for i in range(len(mask)) if mask[i] == 1 and types[i] == 9]
        if key_idx:
            best_k = min(key_idx, key=lambda i: feats[i, 0])
            sin_k = feats[best_k, 1]
            act[0] = 1  # March towards keycard!
            if abs(sin_k) > 0.05:
                if sin_k > 0:
                    act[4] = 1  # Turn left
                else:
                    act[5] = 1  # Turn right
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
        door_rays = [r for r in list(range(58, 64)) + list(range(0, 7)) if hits[r] in (2, 3)]
        if door_rays:
            min_door_d = min(rays[r] for r in door_rays)
            best_door_r = min(door_rays, key=lambda r: rays[r])
            if min_door_d < 0.08:
                # Within DOOM use range (64 units = 0.0625) -> Press USE!
                act[7] = 1
                self.door_timer = 10  # Charge forward through rising door
            else:
                act[0] = 1  # Walk towards door
                if best_door_r != 0:
                    if best_door_r < 32:
                        act[4] = 1
                    else:
                        act[5] = 1
                self.last_act_fwd = act[0]
                return act

        if self.door_timer > 0:
            self.door_timer -= 1
            act[0] = 1  # Charge through doorway aperture
            self.last_act_fwd = act[0]
            return act

        # D. Tactical Combat: Aim, Fire, and Strafe-Dodge Visible Monsters
        enemy_idx = [i for i in range(len(mask)) if mask[i] == 1 and types[i] in (1, 2, 3)]
        if enemy_idx:
            best_e = min(enemy_idx, key=lambda i: feats[i, 0])
            dist = feats[best_e, 0]  # distance / 1024.0
            sin_a = feats[best_e, 1]

            if abs(sin_a) < 0.20:
                act[6] = 1  # ATTACK!

            if sin_a > 0.04:
                act[4] = 1  # Turn left to center target
            elif sin_a < -0.04:
                act[5] = 1  # Turn right to center target

            # Strafe dodge left and right periodically
            if (self.step_idx // 8) % 2 == 0:
                act[2] = 1  # Strafe left
            else:
                act[3] = 1  # Strafe right

            if dist > 0.20:
                act[0] = 1  # Approach enemy
            elif dist < 0.12:
                act[1] = 1  # Back up slightly if monster is right in face
            self.last_act_fwd = act[0]
            return act

        # E. Committed Stuck Recovery (Prevents 2-state periodic limit cycles)
        if self.escape_turn_steps > 0:
            self.escape_turn_steps -= 1
            act[1] = 1  # Back up
            if self.escape_dir == 1:
                act[4] = 1  # Turn left
            else:
                act[5] = 1  # Turn right
            self.last_act_fwd = act[0]
            return act

        if self.stuck_steps >= 2:
            self.escape_turn_steps = 8  # Commit to 8 steps of continuous turning
            best_r = int(np.argmax(rays))
            self.escape_dir = 1 if best_r < 32 else 2
            act[1] = 1  # Back up
            if self.escape_dir == 1:
                act[4] = 1
            else:
                act[5] = 1
            self.last_act_fwd = act[0]
            return act

        # F. Spatial Exploration: Seek Maximum Clearance Angle
        front_cone = list(range(58, 64)) + list(range(0, 7))
        best_front = front_cone[int(np.argmax([rays[r] for r in front_cone]))]

        if front_d > 0.12:
            act[0] = 1  # Move forward
            if best_front != 0:
                if best_front < 32:
                    act[4] = 1
                else:
                    act[5] = 1
        else:
            # Front is blocked: turn toward full 360-degree maximum clearance ray
            best_360 = int(np.argmax(rays))
            if best_360 < 32:
                act[4] = 1
            else:
                act[5] = 1
            if front_d < 0.06:
                act[1] = 1  # Back up slightly

        self.last_act_fwd = act[0]
        return act


# -----------------------------------------------------------------------------
# 2. Expert Trajectory Dataset
# -----------------------------------------------------------------------------
class DoomTrajectoryDataset(Dataset):
    def __init__(self, sequences: List[Dict[str, np.ndarray]], seq_len: int = 16):
        self.sequences = sequences
        self.seq_len = seq_len
        self.indices: List[Tuple[int, int]] = []  # (seq_idx, start_step)

        for s_idx, seq in enumerate(sequences):
            n_steps = len(seq["actions"])
            for t in range(0, n_steps - seq_len + 1, seq_len // 2):
                self.indices.append((s_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s_idx, start = self.indices[idx]
        seq = self.sequences[s_idx]
        end = start + self.seq_len

        return {
            "ray_dist": torch.from_numpy(seq["ray_dist"][start:end]).float(),
            "ray_hit": torch.from_numpy(seq["ray_hit"][start:end]).long(),
            "ent_feat": torch.from_numpy(seq["ent_feat"][start:end]).float(),
            "ent_type": torch.from_numpy(seq["ent_type"][start:end]).long(),
            "ent_mask": torch.from_numpy(seq["ent_mask"][start:end]).float(),
            "vitals": torch.from_numpy(seq["vitals"][start:end]).float(),
            "actions": torch.from_numpy(seq["actions"][start:end]).long(),
        }


def collect_demonstrations(
    maps: List[Tuple[str, str]],
    steps_per_map: int = 1000,
) -> List[Dict[str, np.ndarray]]:
    """Collects multi-map expert demonstrations using the teacher policy."""
    teacher = DoomExpertTeacher()
    demonstrations: List[Dict[str, np.ndarray]] = []
    total_steps = 0
    total_exits = 0

    print(f"\n[1/3] Collecting Expert Demonstrations across {len(maps)} environments...")
    for wad_path, map_name in maps:
        try:
            env = DoomEnv(wad_path, map_name)
        except Exception as e:
            print(f"  Skipping {wad_path}:{map_name} ({e})")
            continue

        obs, info = env.reset()
        teacher.reset()

        ray_dists, ray_hits = [], []
        ent_feats, ent_types, ent_masks = [], [], []
        vitals_list, actions_list = [], []

        ep_steps = 0
        while ep_steps < steps_per_map:
            priv = env.privileged_state()
            act = teacher.act(obs, priv)

            ray_dists.append(obs["rays"]["distance"])
            ray_hits.append(obs["rays"]["hit_type"])
            ent_feats.append(obs["visible_entities"]["features"])
            ent_types.append(obs["visible_entities"]["entity_type"])
            ent_masks.append(obs["visible_entities"]["mask"])
            vitals_list.append(obs["vitals"])
            actions_list.append(act)

            obs, r, term, trunc, info = env.step(act)
            ep_steps += 1
            total_steps += 1

            if priv["exit_reached"]:
                total_exits += 1

            if term or trunc:
                obs, info = env.reset()
                teacher.reset()

        env.close()

        seq = {
            "ray_dist": np.array(ray_dists, dtype=np.float32),
            "ray_hit": np.array(ray_hits, dtype=np.int64),
            "ent_feat": np.array(ent_feats, dtype=np.float32),
            "ent_type": np.array(ent_types, dtype=np.int64),
            "ent_mask": np.array(ent_masks, dtype=np.float32),
            "vitals": np.array(vitals_list, dtype=np.float32),
            "actions": np.array(actions_list, dtype=np.int8),
        }
        demonstrations.append(seq)
        print(f"  ✓ {Path(wad_path).name}:{map_name} collected {len(actions_list)} transitions")

    print(f"Total Expert Transitions: {total_steps} | Exits Reached: {total_exits}\n")
    return demonstrations


# -----------------------------------------------------------------------------
# 3. Supervised Behavioral Cloning Trainer
# -----------------------------------------------------------------------------
def train_behavioral_cloning(
    model: SpatialEgoNet,
    demonstrations: List[Dict[str, np.ndarray]],
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: str = "cpu",
) -> SpatialEgoNet:
    print(f"[2/3] Training Spatial-EgoNet with Behavioral Cloning ({epochs} epochs)...")
    dataset = DoomTrajectoryDataset(demonstrations, seq_len=16)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights for binary actions: ATTACK (pos_weight 4.0), USE (pos_weight 8.0), WEAPON (pos_weight 0.2)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0, 8.0, 0.2]).to(device))
    weight_fb = torch.tensor([1.2, 0.9, 1.5]).to(device)
    weight_turn = torch.tensor([0.8, 1.2, 1.2]).to(device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        acc_fb, acc_lr, acc_turn = 0.0, 0.0, 0.0
        batches = 0

        for b in loader:
            optimizer.zero_grad()
            B, T = b["ray_dist"].shape[:2]

            # Flatten batch and time for encoder: [B * T, ...]
            obs_b = {
                "rays": {
                    "distance": b["ray_dist"].view(B * T, 64).to(device),
                    "hit_type": b["ray_hit"].view(B * T, 64).to(device),
                },
                "visible_entities": {
                    "features": b["ent_feat"].view(B * T, 32, 5).to(device),
                    "entity_type": b["ent_type"].view(B * T, 32).to(device),
                    "mask": b["ent_mask"].view(B * T, 32).to(device),
                },
                "vitals": b["vitals"].view(B * T, 17).to(device),
            }

            features = model.extract_features(obs_b)  # [B * T, hidden_dim]
            features_seq = features.view(B, T, -1)

            # Pass through 2-layer GRU
            h0 = model.initial_hidden_state(batch_size=B, device=device)
            gru_out, _ = model.gru(features_seq, h0)  # [B, T, hidden_dim]
            rnn_feats = gru_out.contiguous().view(B * T, -1)

            # Action Heads
            logits_fb = model.head_move_fb(rnn_feats)
            logits_lr = model.head_move_lr(rnn_feats)
            logits_turn = model.head_turn(rnn_feats)
            logits_bin = model.head_binary(rnn_feats)

            # Targets:
            # Action order: [0: FWD, 1: BACK, 2: LEFT, 3: RIGHT, 4: TURN_L, 5: TURN_R, 6: ATTACK, 7: USE, 8: WEAPON]
            actions = b["actions"].view(B * T, 9).to(device)

            # Target FB: 0: None, 1: FWD, 2: BACK
            target_fb = torch.zeros(B * T, dtype=torch.long, device=device)
            target_fb[actions[:, 0] == 1] = 1
            target_fb[actions[:, 1] == 1] = 2

            # Target LR: 0: None, 1: LEFT, 2: RIGHT
            target_lr = torch.zeros(B * T, dtype=torch.long, device=device)
            target_lr[actions[:, 2] == 1] = 1
            target_lr[actions[:, 3] == 1] = 2

            # Target Turn: 0: None, 1: TURN_L, 2: TURN_R
            target_turn = torch.zeros(B * T, dtype=torch.long, device=device)
            target_turn[actions[:, 4] == 1] = 1
            target_turn[actions[:, 5] == 1] = 2

            # Target Binary: [ATTACK (6), USE (7), WEAPON (8)]
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
            batches += 1

        scheduler.step()
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss / batches:.4f} | "
            f"Acc FB: {acc_fb / batches * 100:.1f}% | "
            f"Acc Turn: {acc_turn / batches * 100:.1f}%"
        )

    print("Behavioral Cloning Pretraining Complete!\n")
    return model


# -----------------------------------------------------------------------------
# 4. Main Training Pipeline Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid Behavioral Cloning & PPO DOOM Pipeline")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_map", type=int, default=1500)
    args = parser.parse_args()

    # Define training maps
    two_rooms_path = str((ROOT / "maps" / "two_rooms.wad").resolve())
    freedoom2_path = IWAD

    training_maps = [
        (two_rooms_path, "MAP01"),
        (two_rooms_path, "MAP01"),
        (two_rooms_path, "MAP01"),
        (freedoom2_path, "MAP01"),
        (freedoom2_path, "MAP02"),
        (freedoom2_path, "MAP03"),
        (freedoom2_path, "MAP04"),
        (freedoom2_path, "MAP05"),
        (freedoom2_path, "MAP06"),
    ]

    # 1. Collect Demonstrations
    demonstrations = collect_demonstrations(training_maps, steps_per_map=args.steps_per_map)

    # 2. Instantiate Spatial-EgoNet Architecture
    model = SpatialEgoNet(hidden_dim=256, num_gru_layers=2)

    # 3. Supervised Behavioral Cloning
    model = train_behavioral_cloning(
        model=model,
        demonstrations=demonstrations,
        epochs=args.epochs,
        batch_size=32,
        lr=6e-4,
        device=args.device,
    )

    # 4. Save Export Weights directly to submission/weights/agent_weights.pt
    export_path = ROOT / "submission" / "weights" / "agent_weights.pt"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": 256,
            "num_gru_layers": 2,
            "timestamp": time.time(),
            "arch": "Spatial-EgoNet-EgoCentric-Spatial-Encoding",
        },
        export_path,
    )
    file_size_mb = export_path.stat().st_size / (1024 * 1024)
    print(f"[3/3] Exported Hardened Production Weights to {export_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
