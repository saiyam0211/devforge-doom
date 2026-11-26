"""Curriculum management for multi-stage DOOM policy learning.

Phases:
1. Phase 1: Navigation & Obstacle Avoidance (Hallways, doors, RND exploration).
2. Phase 2: Tactical Combat & Circle-Strafing (Monsters, projectile dodging, ammo management).
3. Phase 3: Key-Door-Exit Hierarchical Progression (Keycards, locked doors, backtracking).
4. Phase 4: Full Gauntlet Generalization (Unseen procedural combinations).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TrainingPhase(str, Enum):
    NAV_EXPLORATION = "nav_exploration"
    TACTICAL_COMBAT = "tactical_combat"
    HIERARCHICAL_PROGRESSION = "hierarchical_progression"
    FULL_GAUNTLET = "full_gauntlet"


@dataclass
class PhaseConfig:
    name: TrainingPhase
    description: str
    target_wad: str
    max_ticks: int
    vel_weight: float
    stuck_penalty: float
    damage_taken_penalty: float
    kill_reward: float
    key_reward: float
    exit_reward: float
    success_threshold: float  # Mean reward or exit rate to advance


CURRICULUM_STAGES: List[PhaseConfig] = [
    PhaseConfig(
        name=TrainingPhase.NAV_EXPLORATION,
        description="Ego-centric corridor traversal, open doorways, and RND curiosity.",
        target_wad="maps/two_rooms.wad",
        max_ticks=2100,
        vel_weight=0.10,
        stuck_penalty=0.10,
        damage_taken_penalty=0.1,
        kill_reward=5.0,
        key_reward=20.0,
        exit_reward=300.0,
        success_threshold=150.0,
    ),
    PhaseConfig(
        name=TrainingPhase.TACTICAL_COMBAT,
        description="Aiming, strafing, fire evasion, and ammo management.",
        target_wad="maps/two_rooms.wad",
        max_ticks=2100,
        vel_weight=0.03,
        stuck_penalty=0.05,
        damage_taken_penalty=0.8,
        kill_reward=25.0,
        key_reward=20.0,
        exit_reward=300.0,
        success_threshold=250.0,
    ),
    PhaseConfig(
        name=TrainingPhase.HIERARCHICAL_PROGRESSION,
        description="Multi-room locked door layouts, keycard retrieval, and backtracking.",
        target_wad="maps/two_rooms.wad",
        max_ticks=4200,
        vel_weight=0.04,
        stuck_penalty=0.08,
        damage_taken_penalty=0.5,
        kill_reward=15.0,
        key_reward=80.0,
        exit_reward=600.0,
        success_threshold=400.0,
    ),
    PhaseConfig(
        name=TrainingPhase.FULL_GAUNTLET,
        description="Unseen multi-hazard procedural layouts with domain randomization.",
        target_wad="maps/two_rooms.wad",
        max_ticks=4200,
        vel_weight=0.05,
        stuck_penalty=0.08,
        damage_taken_penalty=0.5,
        kill_reward=15.0,
        key_reward=50.0,
        exit_reward=500.0,
        success_threshold=500.0,
    ),
]


class CurriculumManager:
    """Tracks training progress across curriculum phases."""

    def __init__(self, start_phase: TrainingPhase = TrainingPhase.NAV_EXPLORATION):
        self.stage_index = 0
        for i, stage in enumerate(CURRICULUM_STAGES):
            if stage.name == start_phase:
                self.stage_index = i
                break

    @property
    def current_phase(self) -> PhaseConfig:
        return CURRICULUM_STAGES[self.stage_index]

    def check_advance(self, current_mean_metric: float) -> bool:
        """Advance to next phase if threshold reached."""
        if current_mean_metric >= self.current_phase.success_threshold:
            if self.stage_index < len(CURRICULUM_STAGES) - 1:
                self.stage_index += 1
                return True
        return False
