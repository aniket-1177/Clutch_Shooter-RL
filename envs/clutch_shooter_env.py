"""
envs/clutch_shooter_env.py
--------------------------
Stage 1 — A minimal 1D basketball environment for learning tabular Q-Learning.

State  : (distance, shot_clock)  — both integers
Actions: 0 = Move Closer | 1 = Shoot
Reward : +3 three-pointer, +2 mid-range/layup, -2 shot-clock violation, 0 otherwise

This does NOT use the Gymnasium API on purpose — the goal here is to show the
raw MDP loop before introducing the standard interface in Stage 2.
"""

import numpy as np
from utils.config import QTableConfig


class ClutchShooterEnv:
    """
    A 1-D 'last possession' basketball scenario.

    The player starts 5 steps from the basket with 5 seconds on the clock.
    Each timestep the agent must decide: step closer, or pull the trigger.

    Shot probabilities by distance
    --------------------------------
    distance == 5 (three-pointer zone) : 35%  → reward +3
    distance 3–4 (mid-range)           : 45%  → reward +2
    distance 1–2 (paint / layup)       : 75%  → reward +2

    The asymmetry (paint scores the same as mid-range) makes the agent
    weigh whether it's worth using shot-clock time to drive inside.
    """

    SHOT_ZONES = {
        5:        (0.35, 3),   # (success_probability, reward)
        4:        (0.45, 2),
        3:        (0.45, 2),
        2:        (0.75, 2),
        1:        (0.75, 2),
    }

    def __init__(self, config: QTableConfig = None):
        self.cfg = config or QTableConfig()
        self.max_distance = self.cfg.max_distance
        self.max_time = self.cfg.max_time
        self.distance: int = self.max_distance
        self.shot_clock: int = self.max_time

    # ------------------------------------------------------------------
    # Core MDP interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset to initial state and return it."""
        self.distance = self.max_distance
        self.shot_clock = self.max_time
        return (self.distance, self.shot_clock)

    def step(self, action: int) -> tuple:
        """
        Execute one timestep.

        Args:
            action: 0 = move closer, 1 = shoot

        Returns:
            (next_state, reward, done)
        """
        if action == 0:
            return self._move_closer()
        elif action == 1:
            return self._shoot()
        else:
            raise ValueError(f"Invalid action {action}. Expected 0 or 1.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _move_closer(self):
        self.distance = max(1, self.distance - 1)
        self.shot_clock -= 1
        if self.shot_clock <= 0:
            return (self.distance, self.shot_clock), -2, True   # Violation
        return (self.distance, self.shot_clock), 0, False

    def _shoot(self):
        prob, reward_val = self.SHOT_ZONES.get(self.distance, (0.2, 2))
        reward = reward_val if np.random.rand() < prob else 0
        return (self.distance, self.shot_clock), reward, True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def describe_state(self) -> str:
        return f"Distance={self.distance} | Shot Clock={self.shot_clock}s"

    @property
    def state_space_shape(self) -> tuple:
        """Shape needed for the Q-Table: (max_dist+1, max_time+1, n_actions)."""
        return (self.max_distance + 1, self.max_time + 1, 2)
