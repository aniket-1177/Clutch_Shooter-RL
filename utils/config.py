"""
utils/config.py - Centralised hyperparameter configuration.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class QTableConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    episodes: int = 10_000
    max_distance: int = 5
    max_time: int = 5

    def __post_init__(self):
        assert 0 < self.alpha <= 1
        assert 0 < self.gamma <= 1
        assert 0 <= self.epsilon <= 1


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    total_timesteps: int = 300_000
    verbose: int = 1
    seed: Optional[int] = 42
    policy: str = "MlpPolicy"

    def __post_init__(self):
        assert self.learning_rate > 0
        assert self.n_steps > 0
        assert self.total_timesteps > 0


@dataclass
class EnvConfig:
    grid_size: int = 10
    shot_clock: int = 24

    # Basket at top-centre; defender guards it from one step in front
    basket_pos: tuple = (5, 0)
    defender_start: tuple = (5, 1)    # Guards the basket

    contest_radius: float = 1.5       # Within this, shot is contested
    base_shot_penalty: float = -0.3   # Penalty for any missed shot
    basket_reward: float = 3.0        # Reward for scoring
    shot_clock_penalty: float = -1.0  # Shot clock violation

    # Proportional progress reward:  reward = delta_distance * progress_reward
    # Moving 1 cell closer to basket (~1.0 dist unit) gives +0.3
    # Driving 6 cells from y=7 to y=1 gives ~6 * 0.3 = +1.8 accumulated
    progress_reward: float = 0.3

    # NO collision penalty and NO blocked_penalty in the new design.
    # Defender proximity only affects shot success probability (contested mult).
