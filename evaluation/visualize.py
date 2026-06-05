"""
evaluation/visualize.py
------------------------
Visual evaluation tools:
  1. Run the trained PPO agent on one episode with animated court rendering.
  2. Plot training reward curves from Monitor logs.
  3. Optionally export the simulation as an animated GIF.

Usage
-----
    python evaluation/visualize.py --model models/ppo_engineered --mode engineered
    python evaluation/visualize.py --model models/ppo_engineered --save-gif results/demo.gif
    python evaluation/visualize.py --plot-curve results/evaluations.npz
"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Must be set BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from envs.basketball_2d_env import Basketball2DEnv
from utils.logger import get_logger

logger = get_logger("visualize")


# ──────────────────────────────────────────────────────────────────────────────
# Court renderer
# ──────────────────────────────────────────────────────────────────────────────

class CourtRenderer:
    """
    Standalone matplotlib court renderer.
    Captures frames as PIL Images and saves as animated GIF via Pillow directly.
    This avoids all FuncAnimation / canvas.tostring_rgb reliability issues.
    """

    COLORS = {
        "background": "#F5DEB3",   # Wheat / hardwood
        "basket":     "#FF8C00",   # Orange
        "defender":   "#D32F2F",   # Red
        "agent":      "#1565C0",   # Blue
        "grid":       "#C8A96E",   # Wood grain
        "paint":      "#E8D5A3",   # Paint area
    }

    def __init__(self, grid_size: int = 10, figsize: tuple = (7, 7)):
        self.grid_size = grid_size
        self.figsize   = figsize
        # Create ONE persistent figure — never close it between frames
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=100)
        self.fig.patch.set_facecolor(self.COLORS["background"])
        self._pil_frames: list = []   # List[PIL.Image]

    def draw_frame(
        self,
        agent_pos,
        defender_pos,
        basket_pos,
        shot_clock: int,
        action_str: str,
        reward: float,
        mode: str,
        step: int,
    ):
        """Render one frame onto the persistent figure."""
        ax = self.ax
        ax.clear()
        ax.set_facecolor(self.COLORS["background"])

        g = self.grid_size

        # Minor-grid lines to look like court floor tiles
        ax.set_xticks(np.arange(-0.5, g + 0.5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, g + 0.5, 1), minor=True)
        ax.grid(which="minor", color=self.COLORS["grid"], linewidth=0.6, linestyle="-")
        ax.tick_params(which="both", size=0, labelsize=0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Paint area rectangle around the basket
        paint = mpatches.FancyBboxPatch(
            (basket_pos[0] - 2 - 0.5, basket_pos[1] - 0.5),
            4, 4,
            boxstyle="round,pad=0",
            linewidth=1.5,
            edgecolor="#8B6914",
            facecolor=self.COLORS["paint"],
            alpha=0.5,
        )
        ax.add_patch(paint)

        # ── Entities (no emoji — use ASCII labels for font compatibility) ──
        ax.scatter(
            basket_pos[0], basket_pos[1],
            s=600, color=self.COLORS["basket"],
            zorder=5, edgecolors="black", linewidths=1.5,
            label="Basket (O)",
        )
        ax.scatter(
            defender_pos[0], defender_pos[1],
            s=400, marker="X", color=self.COLORS["defender"],
            zorder=5, edgecolors="darkred", linewidths=1.5,
            label="Defender (X)",
        )
        ax.scatter(
            agent_pos[0], agent_pos[1],
            s=400, color=self.COLORS["agent"],
            zorder=5, edgecolors="navy", linewidths=1.5,
            label="AI Agent (o)",
        )

        # ── Coordinate labels on each entity ──
        for pos, label, color in [
            (basket_pos,   "HOOP",   self.COLORS["basket"]),
            (defender_pos, "DEF",    self.COLORS["defender"]),
            (agent_pos,    "AGENT",  self.COLORS["agent"]),
        ]:
            ax.text(
                pos[0], pos[1] + 0.55, label,
                ha="center", va="bottom", fontsize=7,
                color=color, fontweight="bold",
            )

        # Axis limits — Y=0 at top (near basket)
        ax.set_xlim(-0.5, g - 0.5)
        ax.set_ylim(g - 0.5, -0.5)

        # ── Title with colour-coded reward ──
        reward_color = "#2E7D32" if reward > 0 else ("#C62828" if reward < 0 else "#555555")
        title_line1  = f"Step {step}  |  Shot Clock: {shot_clock}  |  Mode: {mode}"
        title_line2  = f"Action: {action_str}"
        ax.set_title(f"{title_line1}\n{title_line2}", fontsize=10, pad=6)

        # Reward badge in top-right corner of the axes
        ax.text(
            0.98, 0.98, f"Reward: {reward:+.2f}",
            transform=ax.transAxes,
            fontsize=10, color=reward_color, fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=reward_color),
        )

        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
        self.fig.tight_layout(pad=1.2)

    def capture_frame(self):
        """
        Render the current figure to a PIL Image and store it.
        Uses io.BytesIO — reliable across all matplotlib backends.
        """
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).copy()   # .copy() detaches from the BytesIO buffer
        buf.close()
        self._pil_frames.append(img.convert("RGB"))

    def save_gif(self, path: str, fps: int = 2):
        """
        Save all captured PIL frames as an animated GIF using Pillow directly.
        Much more reliable than matplotlib's FuncAnimation + PillowWriter.
        """
        if not self._pil_frames:
            logger.warning("No frames to save — call capture_frame() after each draw_frame().")
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert all frames to palette mode for GIF (Pillow requirement)
        frames_p = [f.convert("P", palette=Image.ADAPTIVE, colors=256)
                    for f in self._pil_frames]

        duration_ms = int(1000 / fps)
        frames_p[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=frames_p[1:],
            loop=0,                  # 0 = loop forever
            duration=duration_ms,
            optimize=False,
        )
        logger.info(
            f"GIF saved → {path}  "
            f"({len(self._pil_frames)} frames @ {fps}fps, "
            f"{Path(path).stat().st_size // 1024}KB)"
        )

    def close(self):
        plt.close(self.fig)


# ──────────────────────────────────────────────────────────────────────────────
# Run simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    model_path: str,
    mode: str = "engineered",
    save_gif: str = None,
    fps: int = 2,
    seed: int = None,
    display_inline: bool = False,
):
    """
    Load a trained PPO agent and run one visual episode.

    Parameters
    ----------
    model_path    : Path to saved model (.zip suffix optional).
    mode          : Environment mode ("static" | "moving" | "engineered").
    save_gif      : Path to write the animated GIF. None = no GIF.
    fps           : Frames per second in the output GIF (default 2).
    seed          : Random seed for the environment reset. None = random.
    display_inline: If True, attempt IPython display (Jupyter notebooks).
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    env      = Basketball2DEnv(mode=mode)
    model    = PPO.load(model_path, device="cpu")   # CPU is faster for inference
    renderer = CourtRenderer(grid_size=env.cfg.grid_size)

    reset_kwargs = {"seed": seed} if seed is not None else {}
    obs, _    = env.reset(**reset_kwargs)
    done      = False
    step      = 0
    reward    = 0.0
    total_r   = 0.0

    logger.info(f"Simulating episode | mode={mode} | model={model_path} | seed={seed}")
    print(f"\nStarting Position: Agent at ({obs[0]*9:.0f}, {obs[1]*9:.0f})"
          f" | Defender at ({obs[2]*9:.0f}, {obs[3]*9:.0f})"
          f" | Dist-to-basket: {obs[4]*env._max_dist:.1f}")

    # ── Frame 0: initial state ─────────────────────────────────────────
    renderer.draw_frame(
        env.agent_pos, env.defender_pos, env.basket_pos,
        env.shot_clock, "Game Start", 0.0, mode, step,
    )
    if save_gif:
        renderer.capture_frame()

    # ── Episode loop ───────────────────────────────────────────────────
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_str = Basketball2DEnv.ACTION_NAMES[int(action)]

        obs, reward, terminated, truncated, info = env.step(int(action))
        done    = terminated or truncated
        total_r += reward
        step    += 1

        renderer.draw_frame(
            env.agent_pos, env.defender_pos, env.basket_pos,
            env.shot_clock, action_str, reward, mode, step,
        )
        if save_gif:
            renderer.capture_frame()

        outcome_str = info.get("outcome", "")
        print(
            f"  Step {step:>2}: {action_str:<12} | "
            f"Pos ({obs[0]*9:.0f},{obs[1]*9:.0f}) | "
            f"Clock {obs[5]*env.cfg.shot_clock:.0f} | "
            f"Reward {reward:+.2f}"
            + (f"  ← {outcome_str.upper()}" if done else "")
        )

    print(f"\nEpisode finished in {step} steps. Total reward: {total_r:+.2f}")

    if save_gif:
        renderer.save_gif(save_gif, fps=fps)

    renderer.close()
    env.close()
    return total_r


# ──────────────────────────────────────────────────────────────────────────────
# Training curve plotter
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curve(npz_path: str, save_path: str = None):
    """
    Plot the evaluation reward curve from a Stable-Baselines3 EvalCallback .npz file.

    The file is produced automatically at results/evaluations.npz when you use
    the EvalCallback in train_ppo.py.
    """
    data      = np.load(npz_path)
    timesteps = data["timesteps"]
    results   = data["results"]        # Shape: (n_evals, n_eval_episodes)

    mean_r = results.mean(axis=1)
    std_r  = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, mean_r, color="#1565C0", linewidth=2, label="Mean eval reward")
    ax.fill_between(
        timesteps,
        mean_r - std_r,
        mean_r + std_r,
        alpha=0.2,
        color="#1565C0",
        label="±1 std",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("PPO Training Curve — Basketball2D", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curve saved → {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise trained RL agent")
    parser.add_argument("--model",       type=str, default="models/ppo_engineered")
    parser.add_argument("--mode",        type=str, default="engineered",
                        choices=["static", "moving", "engineered"])
    parser.add_argument("--save-gif",    type=str, default=None,
                        help="Save episode as animated GIF at this path")
    parser.add_argument("--fps",         type=int, default=2,
                        help="Frames per second for the output GIF (default: 2)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for the episode (for reproducibility)")
    parser.add_argument("--plot-curve",  type=str, default=None,
                        help="Path to evaluations.npz — plot training curve instead")
    parser.add_argument("--curve-out",   type=str, default=None,
                        help="Save training curve plot to this file")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.plot_curve:
        plot_training_curve(args.plot_curve, save_path=args.curve_out)
    else:
        run_simulation(
            model_path=args.model,
            mode=args.mode,
            save_gif=args.save_gif,
            fps=args.fps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
