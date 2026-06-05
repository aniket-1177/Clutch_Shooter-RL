# """
# tests/test_envs.py
# ------------------
# Unit tests for both custom environments.
# Run with:  pytest tests/test_envs.py -v
# """

# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# import numpy as np
# import pytest

# from envs.clutch_shooter_env import ClutchShooterEnv
# from envs.basketball_2d_env import Basketball2DEnv
# from utils.config import QTableConfig, EnvConfig


# # ══════════════════════════════════════════════════════
# # Stage 1: ClutchShooterEnv
# # ══════════════════════════════════════════════════════

# class TestClutchShooterEnv:

#     def setup_method(self):
#         self.env = ClutchShooterEnv()

#     def test_reset_returns_initial_state(self):
#         state = self.env.reset()
#         assert state == (5, 5), f"Expected (5, 5), got {state}"

#     def test_move_closer_decreases_distance(self):
#         self.env.reset()
#         state, reward, done = self.env.step(0)   # Move closer
#         assert state[0] == 4, "Distance should decrease after move"
#         assert state[1] == 4, "Shot clock should decrease after move"
#         assert done is False

#     def test_shoot_terminates_episode(self):
#         self.env.reset()
#         _, _, done = self.env.step(1)   # Shoot
#         assert done is True

#     def test_shot_clock_violation(self):
#         config = QTableConfig(max_time=1)
#         env    = ClutchShooterEnv(config)
#         env.reset()
#         env.shot_clock = 1
#         _, reward, done = env.step(0)   # Move: drains the only second
#         assert done is True
#         assert reward == -2

#     def test_distance_never_below_1(self):
#         config = QTableConfig(max_distance=1)
#         env    = ClutchShooterEnv(config)
#         env.reset()
#         env.distance = 1
#         state, _, _ = env.step(0)
#         assert state[0] >= 1, "Distance must not go below 1"

#     def test_state_space_shape(self):
#         assert self.env.state_space_shape == (6, 6, 2)

#     def test_invalid_action_raises(self):
#         self.env.reset()
#         with pytest.raises(ValueError):
#             self.env.step(99)


# # ══════════════════════════════════════════════════════
# # Stage 2–4: Basketball2DEnv
# # ══════════════════════════════════════════════════════

# class TestBasketball2DEnv:

#     @pytest.mark.parametrize("mode", ["static", "moving", "engineered"])
#     def test_reset_observation_shape(self, mode):
#         env = Basketball2DEnv(mode=mode)
#         obs, info = env.reset(seed=0)
#         assert obs.shape == (5,), f"Expected obs shape (5,), got {obs.shape}"
#         env.close()

#     @pytest.mark.parametrize("mode", ["static", "moving", "engineered"])
#     def test_observation_within_bounds(self, mode):
#         env = Basketball2DEnv(mode=mode)
#         obs, _ = env.reset(seed=0)
#         assert np.all(obs >= env.observation_space.low), "Obs below lower bound"
#         assert np.all(obs <= env.observation_space.high), "Obs above upper bound"
#         env.close()

#     def test_action_space_size(self):
#         env = Basketball2DEnv()
#         assert env.action_space.n == 5
#         env.close()

#     def test_shoot_terminates_episode(self):
#         env = Basketball2DEnv()
#         env.reset(seed=42)
#         _, _, terminated, _, _ = env.step(4)   # Shoot
#         assert terminated is True
#         env.close()

#     def test_defender_moves_in_moving_mode(self):
#         env = Basketball2DEnv(mode="moving")
#         env.reset(seed=0)
#         initial_defender = env.defender_pos.copy()
#         env.step(0)   # Agent moves; defender should chase
#         assert not np.array_equal(env.defender_pos, initial_defender), \
#             "Defender should move toward agent in 'moving' mode"
#         env.close()

#     def test_defender_static_in_static_mode(self):
#         env = Basketball2DEnv(mode="static")
#         env.reset(seed=0)
#         initial_defender = env.defender_pos.copy()
#         env.step(0)   # Agent moves; defender should stay put
#         assert np.array_equal(env.defender_pos, initial_defender), \
#             "Defender should NOT move in 'static' mode"
#         env.close()

#     def test_shot_clock_decrements(self):
#         env = Basketball2DEnv()
#         env.reset(seed=0)
#         initial_clock = env.shot_clock
#         env.step(0)
#         assert env.shot_clock == initial_clock - 1
#         env.close()

#     def test_progress_reward_in_engineered_mode(self):
#         """Agent moving closer to basket should receive a small positive reward."""
#         env = Basketball2DEnv(mode="engineered")
#         env.reset(seed=0)

#         # Force the agent to a position where moving up (toward basket) is closer
#         env.agent_pos = np.array([5, 5], dtype=np.int32)
#         env.defender_pos = np.array([9, 9], dtype=np.int32)  # Far away

#         _, reward, terminated, _, _ = env.step(0)   # Move Up (closer to basket at y=0)
#         if not terminated:
#             assert reward >= 0, "Moving closer should not penalise in engineered mode"
#         env.close()

#     def test_invalid_mode_raises(self):
#         with pytest.raises(AssertionError):
#             Basketball2DEnv(mode="invalid_mode")

#     @pytest.mark.parametrize("mode", ["static", "moving", "engineered"])
#     def test_gymnasium_env_checker(self, mode):
#         """SB3 check_env should pass without errors for all modes."""
#         from stable_baselines3.common.env_checker import check_env
#         env = Basketball2DEnv(mode=mode)
#         check_env(env, warn=True)
#         env.close()


"""
tests/test_envs.py - Unit tests for both environments.
Run with:  pytest tests/test_envs.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from envs.clutch_shooter_env import ClutchShooterEnv
from envs.basketball_2d_env import Basketball2DEnv
from utils.config import QTableConfig, EnvConfig


class TestClutchShooterEnv:

    def setup_method(self):
        self.env = ClutchShooterEnv()

    def test_reset_returns_initial_state(self):
        state = self.env.reset()
        assert state == (5, 5)

    def test_move_closer_decreases_distance(self):
        self.env.reset()
        state, reward, done = self.env.step(0)
        assert state[0] == 4
        assert done is False

    def test_shoot_terminates_episode(self):
        self.env.reset()
        _, _, done = self.env.step(1)
        assert done is True

    def test_shot_clock_violation(self):
        config = QTableConfig(max_time=1)
        env = ClutchShooterEnv(config)
        env.reset()
        env.shot_clock = 1
        _, reward, done = env.step(0)
        assert done is True
        assert reward == -2

    def test_invalid_action_raises(self):
        self.env.reset()
        with pytest.raises(ValueError):
            self.env.step(99)


class TestBasketball2DEnv:

    @pytest.mark.parametrize("mode", ["static", "moving", "engineered"])
    def test_reset_obs_shape(self, mode):
        env = Basketball2DEnv(mode=mode)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (6,)
        env.close()

    @pytest.mark.parametrize("mode", ["static", "moving", "engineered"])
    def test_obs_normalised(self, mode):
        env = Basketball2DEnv(mode=mode)
        obs, _ = env.reset(seed=0)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "All obs must be in [0,1]"
        env.close()

    def test_action_space_size(self):
        env = Basketball2DEnv()
        assert env.action_space.n == 5
        env.close()

    def test_shoot_terminates(self):
        env = Basketball2DEnv()
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(4)
        assert terminated is True
        env.close()

    def test_no_collision_termination(self):
        """
        Critical design check: driving through the defender must NOT terminate.
        Only shooting while contested (or shot clock) ends the episode.
        """
        env = Basketball2DEnv(mode='moving')
        env.reset(seed=0)
        env.agent_pos    = np.array([5, 2], dtype=np.int32)
        env.defender_pos = np.array([5, 2], dtype=np.int32)
        # Move action should NOT terminate even when on same square as defender
        _, _, terminated, _, info = env.step(0)
        assert terminated is False, "Collision should NOT terminate the episode"
        env.close()

    def test_shot_clock_violation_terminates(self):
        env = Basketball2DEnv()
        env.reset(seed=0)
        env.shot_clock = 1
        _, r, terminated, _, info = env.step(0)
        assert terminated is True
        assert info.get("outcome") == "shot_clock_violation"
        env.close()

    def test_defender_static_in_static_mode(self):
        env = Basketball2DEnv(mode='static')
        env.reset(seed=0)
        initial = env.defender_pos.copy()
        env.step(0)
        assert np.array_equal(env.defender_pos, initial)
        env.close()

    def test_defender_moves_in_moving_mode(self):
        """Defender should move every other step."""
        env = Basketball2DEnv(mode='moving')
        env.reset(seed=0)
        initial = env.defender_pos.copy()
        # Step 1 (step_count becomes 1, odd -> no move), Step 2 (even -> move)
        env.step(0)
        env.step(0)
        # After 2 steps, defender should have moved once
        moved = not np.array_equal(env.defender_pos, initial)
        assert moved, "Defender should have moved after 2 steps"
        env.close()

    def test_agent_spawns_in_backcourt(self):
        env = Basketball2DEnv()
        for seed in range(20):
            env.reset(seed=seed)
            assert env.agent_pos[1] >= 7, f"Agent y={env.agent_pos[1]} should be >= 7"
        env.close()

    def test_progress_reward_positive_when_moving_closer(self):
        env = Basketball2DEnv(mode='engineered')
        env.reset(seed=0)
        env.agent_pos    = np.array([5, 5], dtype=np.int32)
        env.defender_pos = np.array([0, 0], dtype=np.int32)
        env.shot_clock   = 24
        _, reward, term, _, _ = env.step(0)  # Move Up (toward basket at y=0)
        if not term:
            assert reward > 0, "Moving closer to basket should give positive reward"
        env.close()

    def test_contested_shot_lower_success(self):
        """Shooting when defender is adjacent should succeed less often."""
        np.random.seed(0)
        env = Basketball2DEnv(mode='engineered')
        
        open_scores, contested_scores = 0, 0
        n = 300
        
        for i in range(n):
            env.reset(seed=i)
            env.agent_pos    = np.array([5, 2], dtype=np.int32)
            env.defender_pos = np.array([9, 9], dtype=np.int32)  # far away
            env.shot_clock   = 24
            _, r, _, _, info = env.step(4)
            if info.get("outcome") == "scored":
                open_scores += 1
        
        for i in range(n):
            env.reset(seed=i)
            env.agent_pos    = np.array([5, 2], dtype=np.int32)
            env.defender_pos = np.array([5, 2], dtype=np.int32)  # right on top
            env.shot_clock   = 24
            _, r, _, _, info = env.step(4)
            if info.get("outcome") == "scored":
                contested_scores += 1
        
        assert open_scores > contested_scores, \
            f"Open ({open_scores}) should score more than contested ({contested_scores})"
        env.close()

    @pytest.mark.parametrize("mode", ["static", "moving", "engineered"])
    def test_gymnasium_check_env(self, mode):
        from stable_baselines3.common.env_checker import check_env
        env = Basketball2DEnv(mode=mode)
        check_env(env, warn=True)
        env.close()
