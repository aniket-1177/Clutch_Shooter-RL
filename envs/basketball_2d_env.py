# """
# envs/basketball_2d_env.py
# --------------------------
# Stages 2–4 — A 2D basketball court as a proper Gymnasium environment.

# Three progressive difficulty modes controlled by the `mode` argument:

#   "static"     — Defender stays fixed. Agent learns basic navigation + shooting.
#   "moving"     — Defender actively pursues the agent (greedy tracking AI).
#   "engineered" — Moving defender + shaped reward signal to fix local-minima.

# Observation space (5 floats)
#   [agent_x, agent_y, defender_x, defender_y, shot_clock]

# Action space (Discrete 5)
#   0=Up | 1=Down | 2=Left | 3=Right | 4=Shoot

# Coordinate system
#   (0,0) is the basket (top-left on the grid).
#   Y increases downward. Agent starts near y=5-8, basket is at y=0.
# """

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from typing import Optional, Tuple, Dict, Any

# from utils.config import EnvConfig


# class Basketball2DEnv(gym.Env):
#     """
#     Custom Gymnasium environment simulating a one-on-one basketball possession.

#     Parameters
#     ----------
#     mode : str
#         "static" | "moving" | "engineered"
#     config : EnvConfig, optional
#         Hyperparameter config. Defaults to EnvConfig().
#     render_mode : str, optional
#         "human" for matplotlib animation, None for headless training.
#     """

#     metadata = {"render_modes": ["human"], "render_fps": 4}

#     ACTION_NAMES = {0: "Move Up", 1: "Move Down", 2: "Move Left",
#                     3: "Move Right", 4: "SHOOT"}

#     def __init__(
#         self,
#         mode: str = "engineered",
#         config: EnvConfig = None,
#         render_mode: Optional[str] = None,
#     ):
#         super().__init__()
#         assert mode in ("static", "moving", "engineered"), \
#             f"mode must be 'static', 'moving', or 'engineered'. Got: {mode}"

#         self.mode = mode
#         self.cfg = config or EnvConfig()
#         self.render_mode = render_mode

#         # ── Spaces ──────────────────────────────────────────────────────
#         self.action_space = spaces.Discrete(5)

#         low  = np.array([0, 0, 0, 0, 0],                           dtype=np.float32)
#         high = np.array([self.cfg.grid_size - 1,
#                          self.cfg.grid_size - 1,
#                          self.cfg.grid_size - 1,
#                          self.cfg.grid_size - 1,
#                          self.cfg.shot_clock],                      dtype=np.float32)
#         self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

#         # ── Constants ────────────────────────────────────────────────────
#         self.basket_pos    = np.array(self.cfg.basket_pos,   dtype=np.int32)
#         self._def_start    = np.array(self.cfg.defender_start, dtype=np.int32)

#         # ── Runtime state (initialised in reset) ─────────────────────────
#         self.agent_pos:    np.ndarray = np.zeros(2, dtype=np.int32)
#         self.defender_pos: np.ndarray = np.zeros(2, dtype=np.int32)
#         self.shot_clock:   int = 0

#         # ── Render state ─────────────────────────────────────────────────
#         self.fig = None
#         self.ax  = None

#     # ────────────────────────────────────────────────────────────────────
#     # Gymnasium API
#     # ────────────────────────────────────────────────────────────────────

#     def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
#         super().reset(seed=seed)

#         # Randomise agent start along the far side of the court
#         self.agent_pos = np.array([
#             self.np_random.integers(1, self.cfg.grid_size - 1),
#             self.np_random.integers(5, self.cfg.grid_size - 2),
#         ], dtype=np.int32)

#         self.defender_pos = self._def_start.copy()
#         self.shot_clock   = self.cfg.shot_clock

#         return self._get_obs(), {}

#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         self.shot_clock -= 1
#         reward      = 0.0
#         terminated  = False
#         truncated   = False
#         info: Dict[str, Any] = {}

#         # ── Record pre-move distance for reward shaping ─────────────────
#         old_dist_to_basket = np.linalg.norm(self.agent_pos - self.basket_pos)

#         # ── Agent movement ───────────────────────────────────────────────
#         if   action == 0: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
#         elif action == 1: self.agent_pos[1] = min(self.cfg.grid_size - 1, self.agent_pos[1] + 1)
#         elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
#         elif action == 3: self.agent_pos[0] = min(self.cfg.grid_size - 1, self.agent_pos[0] + 1)

#         # ── Progress reward (engineered mode only) ───────────────────────
#         if self.mode == "engineered" and action in (0, 1, 2, 3):
#             new_dist = np.linalg.norm(self.agent_pos - self.basket_pos)
#             if new_dist < old_dist_to_basket:
#                 reward += self.cfg.progress_reward

#         # ── Defender movement (moving & engineered modes) ────────────────
#         if self.mode != "static" and action != 4:
#             self._move_defender_toward_agent()

#         # ── Shoot action ─────────────────────────────────────────────────
#         if action == 4:
#             terminated = True
#             dist_basket   = np.linalg.norm(self.agent_pos - self.basket_pos)
#             dist_defender = np.linalg.norm(self.agent_pos - self.defender_pos)

#             if dist_defender < self.cfg.contest_radius:
#                 reward = self.cfg.blocked_penalty
#                 info["outcome"] = "blocked"
#             else:
#                 success_chance = max(0.1, 1.0 - dist_basket * 0.12)
#                 if self.np_random.random() < success_chance:
#                     reward = self.cfg.basket_reward
#                     info["outcome"] = "scored"
#                 else:
#                     reward = self.cfg.base_shot_penalty if self.mode == "engineered" else 0.0
#                     info["outcome"] = "missed"

#         # ── Terminal conditions ──────────────────────────────────────────
#         if self.shot_clock <= 0 and not terminated:
#             reward     = self.cfg.shot_clock_penalty
#             terminated = True
#             info["outcome"] = "shot_clock_violation"

#         if np.array_equal(self.agent_pos, self.defender_pos) and not terminated:
#             reward     = self.cfg.collision_penalty
#             terminated = True
#             info["outcome"] = "collision"

#         return self._get_obs(), reward, terminated, truncated, info

#     # ────────────────────────────────────────────────────────────────────
#     # Rendering
#     # ────────────────────────────────────────────────────────────────────

#     def render(self, action_taken: str = "None", current_reward: float = 0.0):
#         """
#         Renders the court in a Jupyter-friendly way using matplotlib.
#         Pass render_mode="human" to enable; ignored otherwise.
#         """
#         try:
#             import matplotlib.pyplot as plt
#             from IPython.display import clear_output, display
#         except ImportError:
#             return   # Silently skip if matplotlib / IPython not available

#         if self.fig is None or not plt.fignum_exists(self.fig.number):
#             self.fig, self.ax = plt.subplots(figsize=(6, 6))

#         ax = self.ax
#         ax.clear()

#         # Grid
#         ax.set_xticks(np.arange(-0.5, self.cfg.grid_size + 0.5, 1))
#         ax.set_yticks(np.arange(-0.5, self.cfg.grid_size + 0.5, 1))
#         ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)

#         # Entities
#         ax.plot(*self.basket_pos,   "o", color="orange", markersize=18, label="Basket 🏀")
#         ax.plot(*self.defender_pos, "X", color="red",    markersize=14, label="Defender 🔴")
#         ax.plot(*self.agent_pos,    "o", color="blue",   markersize=14, label="Agent (AI) 🔵")

#         ax.set_xlim(-0.5, self.cfg.grid_size - 0.5)
#         ax.set_ylim(self.cfg.grid_size - 0.5, -0.5)   # Y=0 at top (near basket)
#         ax.set_title(
#             f"Shot Clock: {self.shot_clock}  |  Mode: {self.mode}\n"
#             f"Last Action: {action_taken}  |  Reward: {current_reward:+.2f}"
#         )
#         ax.legend(loc="upper right", fontsize=8)

#         clear_output(wait=True)
#         display(self.fig)
#         plt.close(self.fig)

#     def close(self):
#         if self.fig is not None:
#             import matplotlib.pyplot as plt
#             plt.close(self.fig)
#             self.fig = None

#     # ────────────────────────────────────────────────────────────────────
#     # Private helpers
#     # ────────────────────────────────────────────────────────────────────

#     def _get_obs(self) -> np.ndarray:
#         return np.array([
#             self.agent_pos[0], self.agent_pos[1],
#             self.defender_pos[0], self.defender_pos[1],
#             self.shot_clock,
#         ], dtype=np.float32)

#     def _move_defender_toward_agent(self):
#         """Greedy defender: steps one cell horizontally then vertically toward agent."""
#         for axis in range(2):
#             if self.defender_pos[axis] < self.agent_pos[axis]:
#                 self.defender_pos[axis] += 1
#             elif self.defender_pos[axis] > self.agent_pos[axis]:
#                 self.defender_pos[axis] -= 1



# """
# envs/basketball_2d_env.py
# --------------------------
# Stages 2–4 — A 2D basketball court as a proper Gymnasium environment.

# Three progressive difficulty modes:
#   "static"     — Defender stays fixed. Agent learns basic navigation + shooting.
#   "moving"     — Defender pursues the agent (greedy).
#   "engineered" — Moving defender + normalised obs + strong reward shaping.

# Key design fixes vs. naive version
# ------------------------------------
# 1. **Normalised observations** — all values scaled to [0, 1].
#    Raw pixel coords (0-9) mixed with shot_clock (0-24) confuse the network.

# 2. **Defender starts beside, not between, agent and basket.**
#    Original start (5,2) was directly in the path. Agent was punished for
#    doing the right thing. New start is offset to the side: (3, 3).

# 3. **Stronger, denser progress reward.**
#    +0.1 per step was too weak vs a miss penalty of -0.2.
#    New: proportional shaping = delta_distance * 0.3 (scales with how much
#    closer you got, not a flat +0.1 regardless of distance moved).

# 4. **Shot probability curve steepened.**
#    Original: success = max(0.1, 1.0 - dist*0.12) → barely changes between
#    dist 4 and 5. New: success = max(0.05, 1.0 - dist*0.18) → much stronger
#    incentive to drive inside.

# Observation space (6 floats, all normalised 0..1)
#   [agent_x, agent_y, defender_x, defender_y, dist_to_basket, shot_clock]

# Action space (Discrete 5)
#   0=Up | 1=Down | 2=Left | 3=Right | 4=Shoot
# """

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from typing import Optional, Tuple, Dict, Any

# from utils.config import EnvConfig


# class Basketball2DEnv(gym.Env):

#     metadata = {"render_modes": ["human"], "render_fps": 4}

#     ACTION_NAMES = {0: "Move Up", 1: "Move Down", 2: "Move Left",
#                     3: "Move Right", 4: "SHOOT"}

#     def __init__(
#         self,
#         mode: str = "engineered",
#         config: EnvConfig = None,
#         render_mode: Optional[str] = None,
#     ):
#         super().__init__()
#         assert mode in ("static", "moving", "engineered"), \
#             f"mode must be 'static', 'moving', or 'engineered'. Got: {mode}"

#         self.mode = mode
#         self.cfg  = config or EnvConfig()
#         self.render_mode = render_mode

#         g = self.cfg.grid_size

#         # ── Spaces ──────────────────────────────────────────────────────────
#         self.action_space = spaces.Discrete(5)

#         # All 6 obs values are normalised to [0, 1]
#         self.observation_space = spaces.Box(
#             low=np.zeros(6, dtype=np.float32),
#             high=np.ones(6, dtype=np.float32),
#             dtype=np.float32,
#         )

#         # ── Constants ────────────────────────────────────────────────────────
#         self.basket_pos  = np.array(self.cfg.basket_pos, dtype=np.int32)
#         self._def_start  = np.array(self.cfg.defender_start, dtype=np.int32)
#         self._max_dist   = float(np.sqrt(2) * (g - 1))   # diagonal of grid

#         # ── Runtime state ────────────────────────────────────────────────────
#         self.agent_pos:    np.ndarray = np.zeros(2, dtype=np.int32)
#         self.defender_pos: np.ndarray = np.zeros(2, dtype=np.int32)
#         self.shot_clock:   int = 0

#         self.fig = None
#         self.ax  = None

#     # ────────────────────────────────────────────────────────────────────────
#     # Gymnasium API
#     # ────────────────────────────────────────────────────────────────────────

#     def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
#         super().reset(seed=seed)

#         g = self.cfg.grid_size

#         # Agent spawns in the back-court (y = 5..8), random x
#         self.agent_pos = np.array([
#             self.np_random.integers(1, g - 1),
#             self.np_random.integers(5, g - 2),
#         ], dtype=np.int32)

#         self.defender_pos = self._def_start.copy()
#         self.shot_clock   = self.cfg.shot_clock

#         return self._get_obs(), {}

#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         self.shot_clock -= 1
#         reward     = 0.0
#         terminated = False
#         truncated  = False
#         info: Dict[str, Any] = {}

#         old_dist = float(np.linalg.norm(self.agent_pos - self.basket_pos))

#         # ── Agent movement ────────────────────────────────────────────────
#         g = self.cfg.grid_size
#         if   action == 0: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
#         elif action == 1: self.agent_pos[1] = min(g - 1, self.agent_pos[1] + 1)
#         elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
#         elif action == 3: self.agent_pos[0] = min(g - 1, self.agent_pos[0] + 1)

#         # ── Shaped progress reward (engineered only) ──────────────────────
#         # Proportional: reward scales with HOW MUCH closer we got, not flat +0.1
#         if self.mode == "engineered" and action != 4:
#             new_dist = float(np.linalg.norm(self.agent_pos - self.basket_pos))
#             delta    = old_dist - new_dist       # positive = moved closer
#             if delta > 0:
#                 reward += delta * self.cfg.progress_reward   # e.g. 0.3 per unit closer

#         # ── Defender movement ─────────────────────────────────────────────
#         if self.mode != "static" and action != 4:
#             self._move_defender_toward_agent()

#         # ── Shoot ─────────────────────────────────────────────────────────
#         if action == 4:
#             terminated  = True
#             dist_basket   = float(np.linalg.norm(self.agent_pos - self.basket_pos))
#             dist_defender = float(np.linalg.norm(self.agent_pos - self.defender_pos))

#             if dist_defender < self.cfg.contest_radius:
#                 reward = self.cfg.blocked_penalty
#                 info["outcome"] = "blocked"
#             else:
#                 # Steeper probability curve: much better to drive inside
#                 success_chance = max(0.05, 1.0 - dist_basket * 0.18)
#                 if self.np_random.random() < success_chance:
#                     reward = self.cfg.basket_reward
#                     info["outcome"] = "scored"
#                 else:
#                     pen = self.cfg.base_shot_penalty if self.mode == "engineered" else 0.0
#                     reward = pen
#                     info["outcome"] = "missed"

#         # ── Shot clock violation ──────────────────────────────────────────
#         if self.shot_clock <= 0 and not terminated:
#             reward     = self.cfg.shot_clock_penalty
#             terminated = True
#             info["outcome"] = "shot_clock_violation"

#         # ── Collision ─────────────────────────────────────────────────────
#         if np.array_equal(self.agent_pos, self.defender_pos) and not terminated:
#             reward     = self.cfg.collision_penalty
#             terminated = True
#             info["outcome"] = "collision"

#         return self._get_obs(), reward, terminated, truncated, info

#     # ────────────────────────────────────────────────────────────────────────
#     # Helpers
#     # ────────────────────────────────────────────────────────────────────────

#     def _get_obs(self) -> np.ndarray:
#         """
#         Returns a fully normalised observation vector in [0, 1].

#         Why normalise?
#           The MLP sees numbers. If one feature ranges 0-24 (shot_clock) and
#           another 0-9 (position), the network's first-layer weights become
#           very unequal in scale, making learning slow and unstable.
#           Normalising all features to [0,1] removes this problem.
#         """
#         g = float(self.cfg.grid_size - 1)
#         dist = float(np.linalg.norm(self.agent_pos - self.basket_pos))
#         return np.array([
#             self.agent_pos[0]    / g,
#             self.agent_pos[1]    / g,
#             self.defender_pos[0] / g,
#             self.defender_pos[1] / g,
#             dist                 / self._max_dist,
#             self.shot_clock      / float(self.cfg.shot_clock),
#         ], dtype=np.float32)

#     def _move_defender_toward_agent(self):
#         """Greedy defender: closes one step per axis per timestep."""
#         for axis in range(2):
#             if self.defender_pos[axis] < self.agent_pos[axis]:
#                 self.defender_pos[axis] += 1
#             elif self.defender_pos[axis] > self.agent_pos[axis]:
#                 self.defender_pos[axis] -= 1

#     def close(self):
#         if self.fig is not None:
#             import matplotlib.pyplot as plt
#             plt.close(self.fig)
#             self.fig = None


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
