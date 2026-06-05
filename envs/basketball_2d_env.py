"""
envs/basketball_2d_env.py
--------------------------
Stages 2-4: 2D basketball court as a proper Gymnasium environment.

Three modes:
  "static"     - Defender stays fixed. Learns basic navigation + shooting.
  "moving"     - Defender pursues agent (half-speed: moves every other step).
  "engineered" - Moving defender + shaped rewards. The full problem.

=== Core Game Design Principles ===

The original design had a fatal flaw: the defender moved at FULL speed (2 cells
per step along both axes simultaneously) while the agent moves 1 cell per step.
The agent was physically incapable of outrunning the defender, so shooting
immediately was the CORRECT response. The RL wasn't broken -- the game was.

Fixes applied:
  1. NO COLLISION TERMINATION. In real basketball, the dribbler can drive through
     the defender's space -- only the shot quality suffers when contested.
     Removing collision removes the perverse incentive to shoot before moving.

  2. HALF-SPEED DEFENDER. Defender moves every OTHER timestep (step_count % 2).
     A faster agent makes driving strategically viable.

  3. DEFENDER GUARDS THE BASKET. Starts at (5,1), one cell in front of the hoop.
     Agent spawns in backcourt (y=7-8). The interesting problem: how do you
     get past a defender guarding the basket?

  4. SHOT QUALITY SCALES WITH DISTANCE AND CONTEST.
     - Open shot at dist 1: ~82% success
     - Open shot at dist 6: ~2% success  (must drive!)
     - Contested penalty: success *= 0.35 when defender within contest_radius

  5. DENSE PROGRESS REWARD. +0.3 * delta_dist each step closer to basket.
     Driving 6 steps to the basket accumulates ~+1.8 in progress reward,
     making the drive clearly better than shooting from distance.

  6. NORMALISED OBSERVATIONS. All values in [0,1] for stable MLP training.
     obs = [agent_x, agent_y, defender_x, defender_y, dist_to_basket, shot_clock]
     (6 features, all normalised)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from utils.config import EnvConfig


class Basketball2DEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    ACTION_NAMES = {
        0: "Move Up",
        1: "Move Down",
        2: "Move Left",
        3: "Move Right",
        4: "SHOOT",
    }

    def __init__(
        self,
        mode: str = "engineered",
        config: EnvConfig = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert mode in ("static", "moving", "engineered"), \
            f"mode must be 'static', 'moving', or 'engineered'. Got: {mode}"

        self.mode        = mode
        self.cfg         = config or EnvConfig()
        self.render_mode = render_mode

        g = self.cfg.grid_size

        # ── Action & Observation spaces ──────────────────────────────────
        self.action_space = spaces.Discrete(5)

        # 6 features, all normalised to [0, 1]:
        # [agent_x, agent_y, def_x, def_y, dist_to_basket, shot_clock_remaining]
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Constants ────────────────────────────────────────────────────
        self.basket_pos  = np.array(self.cfg.basket_pos, dtype=np.int32)
        self._def_start  = np.array(self.cfg.defender_start, dtype=np.int32)
        self._max_dist   = float(np.sqrt(2) * (g - 1))

        # ── Mutable state (set in reset) ─────────────────────────────────
        self.agent_pos:    np.ndarray = np.zeros(2, dtype=np.int32)
        self.defender_pos: np.ndarray = np.zeros(2, dtype=np.int32)
        self.shot_clock:   int = 0
        self._step_count:  int = 0   # for half-speed defender

        # ── Rendering ────────────────────────────────────────────────────
        self.fig = None
        self.ax  = None

    # ────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        g = self.cfg.grid_size

        # Agent always spawns deep in backcourt so it must DRIVE to score
        self.agent_pos = np.array([
            self.np_random.integers(1, g - 1),   # anywhere across court width
            self.np_random.integers(7, g - 1),   # always deep: y = 7 or 8
        ], dtype=np.int32)

        self.defender_pos = self._def_start.copy()
        self.shot_clock   = self.cfg.shot_clock
        self._step_count  = 0

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.shot_clock  -= 1
        self._step_count += 1

        reward     = 0.0
        terminated = False
        truncated  = False
        info: Dict[str, Any] = {}

        old_dist = float(np.linalg.norm(self.agent_pos - self.basket_pos))

        # ── Agent movement ────────────────────────────────────────────────
        g = self.cfg.grid_size
        if   action == 0: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1: self.agent_pos[1] = min(g - 1, self.agent_pos[1] + 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(g - 1, self.agent_pos[0] + 1)

        # ── Dense progress reward (engineered mode) ───────────────────────
        # Proportional to actual distance closed - not a flat per-step bonus.
        # Driving 1 cell = +0.3. Driving 6 cells = +1.8 accumulated.
        # This overwhelms the expected value of shooting from distance.
        if self.mode == "engineered" and action != 4:
            new_dist = float(np.linalg.norm(self.agent_pos - self.basket_pos))
            delta    = old_dist - new_dist
            if delta > 0:
                reward += delta * self.cfg.progress_reward

        # ── Defender movement (half-speed: every other step) ──────────────
        # Half-speed = agent moves 2 steps for every 1 defender step.
        # This makes driving viable. A full-speed defender is physically
        # uncatchable (moves 2 cells/step diagonally vs agent's 1).
        if self.mode != "static" and action != 4:
            if self._step_count % 2 == 0:
                self._move_defender_one_step()

        # ── Shoot ─────────────────────────────────────────────────────────
        if action == 4:
            terminated  = True
            dist_basket   = float(np.linalg.norm(self.agent_pos - self.basket_pos))
            dist_defender = float(np.linalg.norm(self.agent_pos - self.defender_pos))

            # Base success probability - steep curve to reward driving inside
            # dist=0: 100%, dist=1: 82%, dist=2: 64%, dist=3: 46%, dist=6: ~2%
            base_chance = max(0.02, 1.0 - dist_basket * 0.18)

            # Contested shot: heavily penalised (defender within contest radius)
            if dist_defender < self.cfg.contest_radius:
                # Contest cuts success to ~35% of base - mimics blocked/altered shot
                success_chance = base_chance * 0.35
                info["contested"] = True
            else:
                success_chance = base_chance
                info["contested"] = False

            if self.np_random.random() < success_chance:
                reward = self.cfg.basket_reward
                info["outcome"] = "scored"
            else:
                reward = self.cfg.base_shot_penalty
                info["outcome"] = "missed"

        # ── Shot clock violation ──────────────────────────────────────────
        if self.shot_clock <= 0 and not terminated:
            reward     = self.cfg.shot_clock_penalty
            terminated = True
            info["outcome"] = "shot_clock_violation"

        return self._get_obs(), reward, terminated, truncated, info

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Fully normalised 6-feature observation vector (all values in [0, 1]).

        Why normalise?
        Neural networks learn poorly when input features have very different
        scales. Raw coords (0-9) mixed with shot_clock (0-24) cause the first
        layer weights to compensate for scale differences instead of learning
        the actual task. Normalising to [0,1] removes this entirely.
        """
        g    = float(self.cfg.grid_size - 1)
        dist = float(np.linalg.norm(self.agent_pos - self.basket_pos))
        return np.array([
            self.agent_pos[0]    / g,
            self.agent_pos[1]    / g,
            self.defender_pos[0] / g,
            self.defender_pos[1] / g,
            dist                 / self._max_dist,
            self.shot_clock      / float(self.cfg.shot_clock),
        ], dtype=np.float32)

    def _move_defender_one_step(self):
        """
        Defender moves ONE axis at a time toward agent (not both simultaneously).
        Priority: close the axis with the larger gap first.
        This makes the defender's path predictable - agent can learn to juke it.
        """
        diff = self.agent_pos - self.defender_pos
        if abs(diff[1]) >= abs(diff[0]):   # Prioritise vertical (y-axis)
            if diff[1] > 0:
                self.defender_pos[1] += 1
            elif diff[1] < 0:
                self.defender_pos[1] -= 1
        else:
            if diff[0] > 0:
                self.defender_pos[0] += 1
            elif diff[0] < 0:
                self.defender_pos[0] -= 1

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
