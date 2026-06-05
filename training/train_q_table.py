"""
training/train_q_table.py
--------------------------
Stage 1 training script for the tabular Q-Learning agent.

Usage
-----
    python training/train_q_table.py                          # Default config
    python training/train_q_table.py --episodes 20000 --alpha 0.05
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure project root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs.clutch_shooter_env import ClutchShooterEnv
from agents.q_table_agent import QTableAgent
from utils.config import QTableConfig
from utils.logger import get_logger

logger = get_logger("train_q_table")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Table agent on ClutchShooterEnv")
    parser.add_argument("--episodes", type=int,   default=10_000)
    parser.add_argument("--alpha",    type=float, default=0.1,  help="Learning rate")
    parser.add_argument("--gamma",    type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon",  type=float, default=0.2,  help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--save-path", type=str, default="models/q_table.npy")
    return parser.parse_args()


def train(config: QTableConfig, save_path: str = "models/q_table.npy"):
    """
    Main training loop.

    For each episode:
      1. Reset environment to initial state.
      2. Agent selects actions (ε-greedy) until terminal.
      3. Agent updates Q-table via Bellman equation after each step.
      4. Epsilon decays to reduce exploration over time.
    """
    env   = ClutchShooterEnv(config)
    agent = QTableAgent(config, state_shape=env.state_space_shape)

    episode_rewards = []
    log_interval = max(1, config.episodes // 10)

    logger.info(f"Starting Q-Table training | Episodes={config.episodes} | α={config.alpha} | γ={config.gamma}")

    for episode in range(config.episodes):
        state = env.reset()
        done  = False
        total_reward = 0.0

        while not done:
            action                   = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state        = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % log_interval == 0:
            recent_avg = np.mean(episode_rewards[-log_interval:])
            logger.info(
                f"Episode {episode+1:>6}/{config.episodes} | "
                f"Avg Reward (last {log_interval}): {recent_avg:+.3f} | "
                f"ε={agent.epsilon:.4f}"
            )

    # ── Save model + metrics ──────────────────────────────────────────────
    agent.save(save_path)
    logger.info(f"Q-Table saved → {save_path}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_path = results_dir / "q_table_training_rewards.json"
    with open(metrics_path, "w") as f:
        json.dump({"episode_rewards": episode_rewards}, f)
    logger.info(f"Training rewards saved → {metrics_path}")

    # ── Print learned policy ──────────────────────────────────────────────
    agent.print_policy()

    return agent, episode_rewards


def main():
    args   = parse_args()
    config = QTableConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        episodes=args.episodes,
    )
    train(config, save_path=args.save_path)


if __name__ == "__main__":
    main()
