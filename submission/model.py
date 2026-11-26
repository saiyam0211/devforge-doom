"""Spatial-EgoNet: Ego-Centric Spatial Memory Transformer & Recurrent Policy for DOOM.

Designed for DevForge Track 1 ("Can It Play DOOM?"):
- Zero texture dependency: processes purely ego-centric raycast geometry, entity telemetry, and vitals.
- Circular 1D-CNN preserves continuous 360-degree spatial topology.
- Masked Entity Attention processes variable dynamic targets (monsters, projectiles, pickups) permutation-invariantly.
- 2-Layer GRU maintains long-horizon spatial memory (keycard possession, visited rooms, door locations).
- Fully compliant with sub-40ms latency and <2GB VRAM limits.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical


class CircularConv1d(nn.Module):
    """1D convolution with circular boundary padding to represent 360-degree rays seamlessly."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, L]
        # Wrap circular ends
        x_padded = torch.cat([x[..., -self.pad :], x, x[..., : self.pad]], dim=-1)
        return self.conv(x_padded)


class RaycastEncoder(nn.Module):
    """Processes 64 360-degree ego-centric ray distances and discrete hit types with directional sector encoding."""

    def __init__(self, num_hit_types: int = 6, emb_dim: int = 16, out_dim: int = 128):
        super().__init__()
        self.hit_type_emb = nn.Embedding(num_hit_types, emb_dim)

        # 64 ray angles in ego frame: 0 is fwd (+x), 16 is left (+y), 32 is back (-x), 48 is right (-y)
        angles = torch.linspace(0, 2 * math.pi, 65)[:-1]
        self.register_buffer("cos_a", torch.cos(angles))  # [64]
        self.register_buffer("sin_a", torch.sin(angles))  # [64]

        # Ray channels: dist(1) + dir(2) + cartesian_hit(2) + hit_emb(16) = 21
        in_channels = 1 + 2 + 2 + emb_dim

        self.pad2 = 2
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5)
        self.norm1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
        self.pad1 = 1

        # 8-Sector Spatial Pooling: pools features across 8 cardinal & intercardinal directions:
        # Sector 0: Front (-22.5 to +22.5 deg)
        # Sector 1: Front-Left (+22.5 to +67.5 deg)
        # Sector 2: Left (+67.5 to +112.5 deg)
        # Sector 3: Back-Left (+112.5 to +157.5 deg)
        # Sector 4: Back (+157.5 to +202.5 deg)
        # Sector 5: Back-Right (+202.5 to +247.5 deg)
        # Sector 6: Right (+247.5 to +292.5 deg)
        # Sector 7: Front-Right (+292.5 to +337.5 deg)
        self.sector_proj = nn.Conv1d(128, 16, kernel_size=1)  # 128 (max+avg of 64 ch) -> 16 per sector
        # 8 sectors * 16 = 128 features

        # Explicit geometric sector stats: 8 sectors * 6 stats (min_d, max_d, mean_d, is_door, is_enemy, is_item) = 48
        self.geo_proj = nn.Linear(48, 32)

        # Global pooling summary: max(64) + avg(64) = 128 -> 32
        self.global_proj = nn.Linear(128, 32)

        # Final ray representation
        self.out_proj = nn.Linear(128 + 32 + 32, out_dim)

    def forward(self, distance: torch.Tensor, hit_type: torch.Tensor) -> torch.Tensor:
        # distance: [B, 64], hit_type: [B, 64]
        b, n = distance.shape

        # Cartesian hit points in ego coordinates: x is forward, y is left
        hit_x = (distance * self.cos_a).unsqueeze(1)  # [B, 1, 64]
        hit_y = (distance * self.sin_a).unsqueeze(1)  # [B, 1, 64]
        dir_x = self.cos_a.view(1, 1, 64).expand(b, -1, -1)
        dir_y = self.sin_a.view(1, 1, 64).expand(b, -1, -1)
        dist_in = distance.unsqueeze(1)  # [B, 1, 64]
        type_emb = self.hit_type_emb(hit_type).transpose(1, 2)  # [B, 16, 64]

        x = torch.cat([dist_in, dir_x, dir_y, hit_x, hit_y, type_emb], dim=1)  # [B, 21, 64]

        # Circular Conv 1
        x = torch.cat([x[..., -self.pad2 :], x, x[..., : self.pad2]], dim=-1)
        x = F.gelu(self.norm1(self.conv1(x)))

        # Circular Conv 2
        x = torch.cat([x[..., -self.pad2 :], x, x[..., : self.pad2]], dim=-1)
        x = F.gelu(self.norm2(self.conv2(x)))

        # Circular Conv 3
        x = torch.cat([x[..., -self.pad1 :], x, x[..., : self.pad1]], dim=-1)
        feat_map = F.gelu(self.conv3(x))  # [B, 64, 64]

        # 8-Sector Spatial Pooling
        # Roll by 4 so sector 0 covers rays [-4..+3] (indices 60..63 and 0..3: straight ahead)
        feat_rolled = torch.roll(feat_map, shifts=4, dims=-1)
        sectors = feat_rolled.view(b, 64, 8, 8)  # [B, 64, 8 sectors, 8 rays/sec]
        sec_max = torch.max(sectors, dim=-1)[0]  # [B, 64, 8]
        sec_avg = torch.mean(sectors, dim=-1)  # [B, 64, 8]
        sec_cat = torch.cat([sec_max, sec_avg], dim=1)  # [B, 128, 8]
        sec_out = self.sector_proj(sec_cat).view(b, -1)  # [B, 128]

        # Explicit Geometric Sector Stats: min distance, max clearance, mean distance, door, enemy, item
        dist_rolled = torch.roll(distance, shifts=4, dims=-1).view(b, 8, 8)
        hit_rolled = torch.roll(hit_type, shifts=4, dims=-1).view(b, 8, 8)

        sec_min_d = torch.min(dist_rolled, dim=-1)[0]  # [B, 8]
        sec_max_d = torch.max(dist_rolled, dim=-1)[0]  # [B, 8]
        sec_mean_d = torch.mean(dist_rolled, dim=-1)  # [B, 8]
        sec_door = ((hit_rolled == 2) | (hit_rolled == 3)).float().max(dim=-1)[0]  # [B, 8]
        sec_enemy = (hit_rolled == 4).float().max(dim=-1)[0]  # [B, 8]
        sec_item = (hit_rolled == 5).float().max(dim=-1)[0]  # [B, 8]

        geo_feats = torch.cat([sec_min_d, sec_max_d, sec_mean_d, sec_door, sec_enemy, sec_item], dim=-1)  # [B, 48]
        geo_out = F.gelu(self.geo_proj(geo_feats))  # [B, 32]

        # Global Pooling Summary
        glob_max = torch.max(feat_map, dim=-1)[0]  # [B, 64]
        glob_avg = torch.mean(feat_map, dim=-1)  # [B, 64]
        glob_out = F.gelu(self.global_proj(torch.cat([glob_max, glob_avg], dim=-1)))  # [B, 32]

        # Fuse all ego-spatial components
        combined = torch.cat([sec_out, geo_out, glob_out], dim=-1)  # [B, 192]
        return self.out_proj(combined)


class EntityAttentionEncoder(nn.Module):
    """Permutation-invariant self-attention over up to 32 visible entities with nearest-target ego tracking."""

    def __init__(self, num_entity_types: int = 12, type_dim: int = 16, feat_dim: int = 5, out_dim: int = 64):
        super().__init__()
        self.type_emb = nn.Embedding(num_entity_types, type_dim)
        in_dim = feat_dim + type_dim
        self.entity_proj = nn.Linear(in_dim, out_dim)

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)
        # Pooled attention (out_dim * 2) + 4 target ego telemetry (4 * 5 = 20) -> out_dim
        self.out_proj = nn.Linear(out_dim * 2 + 20, out_dim)

    def forward(self, features: torch.Tensor, entity_type: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: [B, 32, 5], entity_type: [B, 32], mask: [B, 32]
        b, m, _ = features.shape
        type_feat = self.type_emb(entity_type)  # [B, 32, 16]
        x = torch.cat([features, type_feat], dim=-1)  # [B, 32, 21]
        x = F.gelu(self.entity_proj(x))  # [B, 32, out_dim]

        # Key padding mask: True indicates value SHOULD be ignored in PyTorch MHA
        key_padding_mask = mask == 0  # [B, 32]

        # If an entire batch row has mask == 0, avoid NaN in attention softmax
        all_masked = key_padding_mask.all(dim=-1)
        safe_mask = key_padding_mask.clone()
        safe_mask[all_masked, 0] = False  # allow first dummy slot so softmax doesn't divide by zero

        attn_out, _ = self.mha(x, x, x, key_padding_mask=safe_mask)
        x = self.norm(x + attn_out)  # [B, 32, out_dim]

        # Masked aggregation (pool only valid entities)
        mask_expanded = mask.unsqueeze(-1).float()  # [B, 32, 1]
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
        masked_x = x * mask_expanded

        mean_pool = masked_x.sum(dim=1) / valid_counts  # [B, out_dim]
        # Masked max pool: replace invalid with large negative before max
        neg_inf_x = x.masked_fill(~mask.bool().unsqueeze(-1), -1e4)
        max_pool = torch.max(neg_inf_x, dim=1)[0]
        max_pool = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(max_pool), max_pool)

        # Explicit Ego-Target Telemetry Extraction:
        # 1. Nearest Enemy (types 1, 2, 3)
        is_enemy = (mask.bool()) & ((entity_type == 1) | (entity_type == 2) | (entity_type == 3))
        d_enemy = torch.where(is_enemy, features[:, :, 0], torch.tensor(1e6, device=features.device))
        idx_enemy = torch.argmin(d_enemy, dim=1)
        has_enemy = is_enemy.any(dim=1).unsqueeze(-1).float()
        near_enemy = torch.gather(features, 1, idx_enemy.view(b, 1, 1).expand(-1, -1, 5)).squeeze(1) * has_enemy

        # 2. Nearest Keycard (type 9)
        is_key = (mask.bool()) & (entity_type == 9)
        d_key = torch.where(is_key, features[:, :, 0], torch.tensor(1e6, device=features.device))
        idx_key = torch.argmin(d_key, dim=1)
        has_key = is_key.any(dim=1).unsqueeze(-1).float()
        near_key = torch.gather(features, 1, idx_key.view(b, 1, 1).expand(-1, -1, 5)).squeeze(1) * has_key

        # 3. Nearest Item Pickup (types 5, 6, 7, 8)
        is_item = (mask.bool()) & ((entity_type >= 5) & (entity_type <= 8))
        d_item = torch.where(is_item, features[:, :, 0], torch.tensor(1e6, device=features.device))
        idx_item = torch.argmin(d_item, dim=1)
        has_item = is_item.any(dim=1).unsqueeze(-1).float()
        near_item = torch.gather(features, 1, idx_item.view(b, 1, 1).expand(-1, -1, 5)).squeeze(1) * has_item

        # 4. Nearest Projectile (type 4: incoming projectile to dodge)
        is_proj = (mask.bool()) & (entity_type == 4)
        d_proj = torch.where(is_proj, features[:, :, 0], torch.tensor(1e6, device=features.device))
        idx_proj = torch.argmin(d_proj, dim=1)
        has_proj = is_proj.any(dim=1).unsqueeze(-1).float()
        near_proj = torch.gather(features, 1, idx_proj.view(b, 1, 1).expand(-1, -1, 5)).squeeze(1) * has_proj

        target_telemetry = torch.cat([near_enemy, near_key, near_item, near_proj], dim=-1)  # [B, 20]

        pooled = torch.cat([mean_pool, max_pool, target_telemetry], dim=-1)  # [B, out_dim * 2 + 20]
        return self.out_proj(pooled)  # [B, out_dim]


class VitalsEncoder(nn.Module):
    """Processes 17 continuous and discrete vitals signals into an ego-state embedding."""

    def __init__(self, in_dim: int = 17, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )

    def forward(self, vitals: torch.Tensor) -> torch.Tensor:
        # vitals: [B, 17]
        # Apply normalization: HP (0..200), armor (0..200), ammo (0..100)
        norm_vitals = vitals.clone()
        norm_vitals[:, 0] = norm_vitals[:, 0] / 100.0  # health
        norm_vitals[:, 1] = norm_vitals[:, 1] / 100.0  # armor
        norm_vitals[:, 3:8] = norm_vitals[:, 3:8] / 50.0  # ammo counts
        norm_vitals[:, 14:17] = norm_vitals[:, 14:17] / 20.0  # velocities
        return self.net(norm_vitals)


class SpatialEgoNet(nn.Module):
    """Multi-modal ego-centric architecture with recurrent spatial memory and factored policy."""

    def __init__(self, hidden_dim: int = 256, num_gru_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        # Modality Encoders
        self.ray_encoder = RaycastEncoder(num_hit_types=6, emb_dim=16, out_dim=128)
        self.entity_encoder = EntityAttentionEncoder(num_entity_types=12, type_dim=16, feat_dim=5, out_dim=64)
        self.vitals_encoder = VitalsEncoder(in_dim=17, out_dim=64)

        # Multi-modal fusion
        fused_dim = 128 + 64 + 64  # 256
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2-Layer Recurrent Spatial Memory
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Factored Policy Action Heads:
        # 1. Forward / Backward: 0: None, 1: FWD, 2: BACK
        self.head_move_fb = nn.Linear(hidden_dim, 3)
        # 2. Strafe Left / Right: 0: None, 1: LEFT, 2: RIGHT
        self.head_move_lr = nn.Linear(hidden_dim, 3)
        # 3. Turn Left / Right: 0: None, 1: TURN_L, 2: TURN_R
        self.head_turn = nn.Linear(hidden_dim, 3)
        # 4. Binary actions: Attack, Use, NextWeapon
        self.head_binary = nn.Linear(hidden_dim, 3)

        # State-Value Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def initial_hidden_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(self.num_gru_layers, batch_size, self.hidden_dim, device=device)

    def extract_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract multi-modal spatial features from observation dict."""
        ray_dist = obs["rays"]["distance"]
        ray_hit = obs["rays"]["hit_type"].long()
        ray_feats = self.ray_encoder(ray_dist, ray_hit)

        ent_feats = obs["visible_entities"]["features"]
        ent_types = obs["visible_entities"]["entity_type"].long()
        ent_mask = obs["visible_entities"]["mask"]
        ent_summary = self.entity_encoder(ent_feats, ent_types, ent_mask)

        vitals = obs["vitals"]
        vitals_feats = self.vitals_encoder(vitals)

        fused = torch.cat([ray_feats, ent_summary, vitals_feats], dim=-1)
        return self.fusion(fused)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass emitting MultiBinary(9) actions, log_prob, value, and next hidden state."""
        features = self.extract_features(obs)  # [B, hidden_dim]
        # Add sequence dimension for GRU: [B, 1, hidden_dim]
        gru_out, next_hidden = self.gru(features.unsqueeze(1), hidden)
        rnn_feat = gru_out.squeeze(1)  # [B, hidden_dim]

        # Value estimate
        value = self.critic(rnn_feat).squeeze(-1)  # [B]

        # Action distributions
        logits_fb = self.head_move_fb(rnn_feat)
        logits_lr = self.head_move_lr(rnn_feat)
        logits_turn = self.head_turn(rnn_feat)
        logits_binary = self.head_binary(rnn_feat)

        dist_fb = Categorical(logits=logits_fb)
        dist_lr = Categorical(logits=logits_lr)
        dist_turn = Categorical(logits=logits_turn)
        dist_binary = Bernoulli(logits=logits_binary)

        if deterministic:
            act_fb = torch.argmax(logits_fb, dim=-1)
            act_lr = torch.argmax(logits_lr, dim=-1)
            act_turn = torch.argmax(logits_turn, dim=-1)
            # Calibrated thresholds for class-imbalanced binary actions:
            act_atk = (logits_binary[:, 0] > -0.8).float()  # ATTACK (p > 0.31)
            act_use = (logits_binary[:, 1] > -1.2).float()  # USE (p > 0.23)
            act_wep = (logits_binary[:, 2] > 0.0).float()   # WEAPON
            act_binary = torch.stack([act_atk, act_use, act_wep], dim=-1)
        else:
            act_fb = dist_fb.sample()
            act_lr = dist_lr.sample()
            act_turn = dist_turn.sample()
            act_binary = dist_binary.sample()

        log_prob = (
            dist_fb.log_prob(act_fb)
            + dist_lr.log_prob(act_lr)
            + dist_turn.log_prob(act_turn)
            + dist_binary.log_prob(act_binary).sum(dim=-1)
        )

        # Convert factored choices into MultiBinary(9) buttons:
        # [0: MOVE_FWD, 1: MOVE_BACK, 2: MOVE_L, 3: MOVE_R, 4: TURN_L, 5: TURN_R, 6: ATTACK, 7: USE, 8: SELECT_NEXT]
        b = features.shape[0]
        actions = torch.zeros((b, 9), dtype=torch.int8, device=features.device)

        # FB
        actions[:, 0] = (act_fb == 1).to(torch.int8)  # FWD
        actions[:, 1] = (act_fb == 2).to(torch.int8)  # BACK
        # LR
        actions[:, 2] = (act_lr == 1).to(torch.int8)  # LEFT
        actions[:, 3] = (act_lr == 2).to(torch.int8)  # RIGHT
        # Turn
        actions[:, 4] = (act_turn == 1).to(torch.int8)  # TURN_LEFT
        actions[:, 5] = (act_turn == 2).to(torch.int8)  # TURN_RIGHT
        # Binary
        actions[:, 6] = act_binary[:, 0].to(torch.int8)  # ATTACK
        actions[:, 7] = act_binary[:, 1].to(torch.int8)  # USE
        actions[:, 8] = act_binary[:, 2].to(torch.int8)  # WEAPON

        # Guard: only allow weapon switch if current weapon is empty (ammo == 0)
        current_ammo = obs["vitals"][:, 3]
        has_ammo = (current_ammo > 0).to(torch.int8)
        actions[:, 8] = actions[:, 8] * (1 - has_ammo)

        return actions, log_prob, value, next_hidden

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor,
        actions_multibinary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch log-probabilities and entropy for PPO updates."""
        features = self.extract_features(obs)
        gru_out, _ = self.gru(features.unsqueeze(1), hidden)
        rnn_feat = gru_out.squeeze(1)

        value = self.critic(rnn_feat).squeeze(-1)

        logits_fb = self.head_move_fb(rnn_feat)
        logits_lr = self.head_move_lr(rnn_feat)
        logits_turn = self.head_turn(rnn_feat)
        logits_binary = self.head_binary(rnn_feat)

        dist_fb = Categorical(logits=logits_fb)
        dist_lr = Categorical(logits=logits_lr)
        dist_turn = Categorical(logits=logits_turn)
        dist_binary = Bernoulli(logits=logits_binary)

        # Deconstruct MultiBinary(9) back to categorical actions
        act_fb = torch.zeros(rnn_feat.shape[0], dtype=torch.long, device=rnn_feat.device)
        act_fb[actions_multibinary[:, 0] == 1] = 1
        act_fb[actions_multibinary[:, 1] == 1] = 2

        act_lr = torch.zeros(rnn_feat.shape[0], dtype=torch.long, device=rnn_feat.device)
        act_lr[actions_multibinary[:, 2] == 1] = 1
        act_lr[actions_multibinary[:, 3] == 1] = 2

        act_turn = torch.zeros(rnn_feat.shape[0], dtype=torch.long, device=rnn_feat.device)
        act_turn[actions_multibinary[:, 4] == 1] = 1
        act_turn[actions_multibinary[:, 5] == 1] = 2

        act_binary = actions_multibinary[:, 6:9].float()

        log_prob = (
            dist_fb.log_prob(act_fb)
            + dist_lr.log_prob(act_lr)
            + dist_turn.log_prob(act_turn)
            + dist_binary.log_prob(act_binary).sum(dim=-1)
        )

        entropy = (
            dist_fb.entropy()
            + dist_lr.entropy()
            + dist_turn.entropy()
            + dist_binary.entropy().sum(dim=-1)
        )

        return log_prob, value, entropy
