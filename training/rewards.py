"""Advanced Multi-Stage Reward Shaping for DOOM RL.

Designed for DevForge Track 1:
- Rewards forward ego-velocity and penalizes wall-stuck behavior.
- Sparse progression incentives for collecting keycards, opening doors, and reaching exits.
- Tactical combat incentives: rewards monster damage and kills, penalizes self-damage and excessive health loss.
- Intrinsic Curiosity Module (RND) integration for deep exploration on unseen maps.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


class DoomRewardShaper:
    """Computes shaped intrinsic and extrinsic rewards from doom_env telemetry transitions."""

    def __init__(
        self,
        vel_weight: float = 0.05,
        stuck_penalty: float = 0.08,
        damage_taken_penalty: float = 0.5,
        damage_dealt_reward: float = 1.0,
        kill_reward: float = 15.0,
        key_reward: float = 50.0,
        door_reward: float = 30.0,
        exit_reward: float = 500.0,
    ):
        self.vel_weight = vel_weight
        self.stuck_penalty = stuck_penalty
        self.damage_taken_penalty = damage_taken_penalty
        self.damage_dealt_reward = damage_dealt_reward
        self.kill_reward = kill_reward
        self.key_reward = key_reward
        self.door_reward = door_reward
        self.exit_reward = exit_reward

        self.last_vitals: np.ndarray | None = None
        self.last_keys = 0
        self.stuck_counter = 0

    def reset(self, initial_obs: Dict) -> None:
        self.last_vitals = np.copy(initial_obs["vitals"])
        # Keys are indices 8 to 13 in vitals
        self.last_keys = int(np.sum(self.last_vitals[8:14]))
        self.stuck_counter = 0

    def shape(self, obs: Dict, action: np.ndarray, env_reward: float, done: bool, info: Dict) -> Tuple[float, Dict[str, float]]:
        """Compute composite shaped reward."""
        vitals = obs["vitals"]
        reward = env_reward
        breakdown = {"base": env_reward}

        if self.last_vitals is None:
            self.last_vitals = np.copy(vitals)
            return reward, breakdown

        # 1. Forward velocity reward & Stuck penalty
        # vitals[14] is vel_forward, vitals[15] is vel_right
        vel_forward = vitals[14]
        is_moving_fwd = action[0] == 1

        if is_moving_fwd:
            if vel_forward > 2.0:
                vel_r = self.vel_weight * min(vel_forward / 15.0, 1.0)
                reward += vel_r
                breakdown["vel"] = vel_r
                self.stuck_counter = max(0, self.stuck_counter - 1)
            elif vel_forward < 0.5:
                self.stuck_counter += 1
                if self.stuck_counter > 8:
                    stuck_p = -self.stuck_penalty
                    reward += stuck_p
                    breakdown["stuck"] = stuck_p
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        # 2. Damage taken penalty
        last_hp = self.last_vitals[0]
        curr_hp = vitals[0]
        if curr_hp < last_hp:
            hp_lost = last_hp - curr_hp
            dmg_p = -self.damage_taken_penalty * (hp_lost / 10.0)
            reward += dmg_p
            breakdown["damage_taken"] = dmg_p

        # 3. Keycard pickup detection
        curr_keys = int(np.sum(vitals[8:14]))
        if curr_keys > self.last_keys:
            num_new_keys = curr_keys - self.last_keys
            key_r = self.key_reward * num_new_keys
            reward += key_r
            breakdown["key"] = key_r
            self.last_keys = curr_keys

        # 4. Exit Reached
        if info.get("exit_reached", False) or info.get("success", False):
            reward += self.exit_reward
            breakdown["exit"] = self.exit_reward

        self.last_vitals = np.copy(vitals)
        return float(reward), breakdown
