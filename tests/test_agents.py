"""
tests/test_agents.py
---------------------
Unit tests for the Q-Table agent.
Run with:  pytest tests/test_agents.py -v
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from agents.q_table_agent import QTableAgent
from utils.config import QTableConfig


class TestQTableAgent:

    def setup_method(self):
        self.config = QTableConfig(alpha=0.1, gamma=0.9, epsilon=0.5, episodes=100)
        self.agent  = QTableAgent(self.config, state_shape=(6, 6, 2))

    def test_initial_q_table_is_zero(self):
        assert np.all(self.agent.q_table == 0), "Q-table should initialise to zeros"

    def test_q_table_shape(self):
        assert self.agent.q_table.shape == (6, 6, 2)

    def test_update_changes_q_value(self):
        state      = (5, 5)
        action     = 1
        reward     = 2.0
        next_state = (5, 4)

        self.agent.update(state, action, reward, next_state)
        assert self.agent.q_table[5, 5, 1] != 0.0, "Q-value should change after update"

    def test_bellman_update_correctness(self):
        """Manually verify the Bellman update formula."""
        config = QTableConfig(alpha=0.1, gamma=0.9)
        agent  = QTableAgent(config, state_shape=(6, 6, 2))

        state      = (3, 3)
        action     = 0
        reward     = 1.0
        next_state = (2, 2)

        # Pre-set a known Q-value for next state
        agent.q_table[2, 2, 0] = 4.0
        agent.q_table[2, 2, 1] = 2.0   # max next Q = 4.0

        agent.update(state, action, reward, next_state)

        # Expected: 0 + 0.1 * (1.0 + 0.9 * 4.0 - 0) = 0.1 * 4.6 = 0.46
        expected = 0.1 * (1.0 + 0.9 * 4.0)
        assert abs(agent.q_table[3, 3, 0] - expected) < 1e-6, \
            f"Bellman update incorrect. Expected {expected}, got {agent.q_table[3, 3, 0]}"

    def test_greedy_action_selection(self):
        """With epsilon=0, agent should always pick the greedy action."""
        self.agent.epsilon = 0.0
        self.agent.q_table[5, 5, 1] = 10.0   # Strongly prefer action 1
        action = self.agent.select_action((5, 5), training=True)
        assert action == 1

    def test_epsilon_decay(self):
        initial_eps = self.agent.epsilon
        self.agent.decay_epsilon(decay_rate=0.9)
        assert self.agent.epsilon < initial_eps, "Epsilon should decrease after decay"

    def test_epsilon_does_not_go_below_min(self):
        self.agent.epsilon = 0.001
        self.agent.decay_epsilon(decay_rate=0.5, min_epsilon=0.01)
        assert self.agent.epsilon >= 0.01

    def test_save_and_load(self):
        self.agent.q_table[2, 3, 1] = 99.0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test_q.npy"
            self.agent.save(path)

            new_agent = QTableAgent(self.config, state_shape=(6, 6, 2))
            new_agent.load(path)

            assert new_agent.q_table[2, 3, 1] == 99.0, "Loaded Q-table should match saved"

    def test_policy_table_shape(self):
        policy = self.agent.get_policy_table()
        assert policy.shape == (6, 6), "Policy table should be (distances × times)"

    def test_policy_values_are_valid_actions(self):
        policy = self.agent.get_policy_table()
        assert np.all((policy == 0) | (policy == 1)), "Policy must only contain valid actions"
