"""
agents/q_table_agent.py
-----------------------
Stage 1 — Tabular Q-Learning agent.

Implements the classic off-policy TD update rule (Bellman equation):

    Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max_a' Q(s', a')  −  Q(s, a) ]

where:
  α  = learning rate    (how fast we update our estimates)
  γ  = discount factor  (how much we value future rewards vs. immediate ones)
  r  = reward received
  s' = next state

Exploration strategy: ε-greedy with optional linear decay.
"""

import numpy as np
from pathlib import Path
from utils.config import QTableConfig


class QTableAgent:
    """
    Q-Table agent for discrete state/action spaces.

    The Q-table is a 3-D NumPy array of shape:
        (max_distance + 1,  max_time + 1,  n_actions)

    Each cell Q[d, t, a] stores the estimated return for taking
    action a when at distance d with t seconds on the clock.

    Parameters
    ----------
    config : QTableConfig
        Hyperparameter config.
    state_shape : tuple
        (n_distances, n_times, n_actions)
    """

    def __init__(self, config: QTableConfig = None, state_shape: tuple = (6, 6, 2)):
        self.cfg = config or QTableConfig()
        self.q_table = np.zeros(state_shape)
        self.epsilon = self.cfg.epsilon
        self._episode = 0

    # ────────────────────────────────────────────────────────────────────
    # Action selection
    # ────────────────────────────────────────────────────────────────────

    def select_action(self, state: tuple, training: bool = True) -> int:
        """
        ε-greedy action selection.

        During training: explore (random) with prob ε, exploit (greedy) otherwise.
        During evaluation: always exploit (deterministic).
        """
        dist, clock = state
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[2])   # Random action
        return int(np.argmax(self.q_table[dist, clock]))       # Greedy action

    # ────────────────────────────────────────────────────────────────────
    # Learning
    # ────────────────────────────────────────────────────────────────────

    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Apply the Bellman equation update rule.

        This is the heart of Q-Learning. We're asking:
        "Given what I now know (reward r + best possible future value),
         how wrong was my old estimate? Nudge it by α in the right direction."
        """
        d,  t  = state
        nd, nt = next_state

        old_value  = self.q_table[d, t, action]
        next_max   = np.max(self.q_table[nd, nt])                    # Best future Q

        # Bellman target: r + γ · max Q(s', a')
        target     = reward + self.cfg.gamma * next_max
        td_error   = target - old_value                               # Temporal difference

        self.q_table[d, t, action] = old_value + self.cfg.alpha * td_error

    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """
        Linearly or exponentially decay exploration over training.
        Call once per episode after all updates.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        self._episode += 1

    # ────────────────────────────────────────────────────────────────────
    # Persistence
    # ────────────────────────────────────────────────────────────────────

    def save(self, path: str = "models/q_table.npy"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str = "models/q_table.npy"):
        self.q_table = np.load(path)

    # ────────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ────────────────────────────────────────────────────────────────────

    def get_policy_table(self) -> np.ndarray:
        """
        Returns the greedy policy as a 2D array (distance × time).
        Value 0 = Move Closer, 1 = Shoot.
        """
        return np.argmax(self.q_table, axis=2)

    def print_policy(self):
        """Pretty-print the learned policy."""
        policy = self.get_policy_table()
        action_labels = {0: "MOVE ", 1: "SHOOT"}
        print("\n=== Learned Q-Table Policy ===")
        print(f"{'':8}", end="")
        for t in range(self.q_table.shape[1]):
            print(f"Clock={t}  ", end="")
        print()
        for d in range(self.q_table.shape[0]):
            print(f"Dist={d}  ", end="")
            for t in range(self.q_table.shape[1]):
                print(f"  {action_labels[policy[d, t]]}  ", end="")
            print()
        print()
