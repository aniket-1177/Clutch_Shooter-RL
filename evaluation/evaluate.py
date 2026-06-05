"""
evaluation/evaluate.py
-----------------------
Quantitative evaluation of trained agents over N episodes.

Produces a metrics summary (JSON + console) covering:
  - Mean / std / min / max episode reward
  - Shot success rate
  - Shot clock violation rate
  - Collision rate
  - Avg steps per episode

Usage
-----
    # Evaluate Q-Table
    python evaluation/evaluate.py --agent qtable --model models/q_table.npy --episodes 500

    # Evaluate PPO
    python evaluation/evaluate.py --agent ppo --model models/ppo_engineered --mode engineered
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs.clutch_shooter_env import ClutchShooterEnv
from envs.basketball_2d_env import Basketball2DEnv
from agents.q_table_agent import QTableAgent
from utils.config import QTableConfig, PPOConfig, EnvConfig
from utils.logger import get_logger

logger = get_logger("evaluate")


def evaluate_q_table(model_path: str, n_episodes: int = 500) -> dict:
    """Run evaluation loop for Q-Table agent."""
    config = QTableConfig()
    env    = ClutchShooterEnv(config)
    agent  = QTableAgent(config, state_shape=env.state_space_shape)
    agent.load(model_path)

    episode_rewards = []
    outcomes = defaultdict(int)

    for _ in range(n_episodes):
        state = env.reset()
        done  = False
        total_reward = 0.0

        while not done:
            action                   = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)
        if total_reward > 0:
            outcomes["scored"] += 1
        elif total_reward == -2:
            outcomes["shot_clock_violation"] += 1
        else:
            outcomes["missed_or_other"] += 1

    return _compile_metrics(episode_rewards, outcomes, n_episodes)


def evaluate_ppo(model_path: str, mode: str = "engineered", n_episodes: int = 500) -> dict:
    """Run evaluation loop for PPO agent."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
        sys.exit(1)

    env   = Basketball2DEnv(mode=mode)
    model = PPO.load(model_path, device="cpu")

    episode_rewards = []
    episode_lengths = []
    outcomes = defaultdict(int)

    for _ in range(n_episodes):
        obs, _   = env.reset()
        done     = False
        total_r  = 0.0
        steps    = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done    = terminated or truncated
            total_r += reward
            steps   += 1

        episode_rewards.append(total_r)
        episode_lengths.append(steps)
        outcome = info.get("outcome", "unknown")
        outcomes[outcome] += 1

    env.close()
    metrics = _compile_metrics(episode_rewards, outcomes, n_episodes)
    metrics["mean_episode_length"] = float(np.mean(episode_lengths))
    return metrics


def _compile_metrics(episode_rewards: list, outcomes: dict, n_episodes: int) -> dict:
    arr = np.array(episode_rewards)
    metrics = {
        "n_episodes":       n_episodes,
        "mean_reward":      float(np.mean(arr)),
        "std_reward":       float(np.std(arr)),
        "min_reward":       float(np.min(arr)),
        "max_reward":       float(np.max(arr)),
        "median_reward":    float(np.median(arr)),
        "outcomes":         dict(outcomes),
    }

    # Derived rates
    scored   = outcomes.get("scored", 0)
    sc_viol  = outcomes.get("shot_clock_violation", 0)
    collision = outcomes.get("collision", 0)

    metrics["shot_success_rate"]         = scored    / n_episodes
    metrics["shot_clock_violation_rate"] = sc_viol   / n_episodes
    metrics["collision_rate"]            = collision  / n_episodes

    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Episodes evaluated : {metrics['n_episodes']}")
    print(f"  Mean reward        : {metrics['mean_reward']:+.4f} ± {metrics['std_reward']:.4f}")
    print(f"  Median reward      : {metrics['median_reward']:+.4f}")
    print(f"  Min / Max reward   : {metrics['min_reward']:+.4f} / {metrics['max_reward']:+.4f}")
    print(f"  Shot success rate  : {metrics['shot_success_rate']:.1%}")
    print(f"  Shot clock viol.   : {metrics['shot_clock_violation_rate']:.1%}")
    print(f"  Collision rate     : {metrics['collision_rate']:.1%}")
    if "mean_episode_length" in metrics:
        print(f"  Avg steps/episode  : {metrics['mean_episode_length']:.1f}")
    print(f"\n  Outcome breakdown:")
    for k, v in metrics["outcomes"].items():
        print(f"    {k:<28}: {v} ({v/metrics['n_episodes']:.1%})")
    print(sep + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent")
    parser.add_argument("--agent",    type=str, default="ppo", choices=["qtable", "ppo"])
    parser.add_argument("--model",    type=str, default="models/ppo_engineered")
    parser.add_argument("--mode",     type=str, default="engineered",
                        choices=["static", "moving", "engineered"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--save",     type=str, default=None,
                        help="Path to save JSON metrics (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.agent == "qtable":
        metrics = evaluate_q_table(args.model, args.episodes)
        print_metrics(metrics, title="Q-Table Agent Evaluation")
    else:
        metrics = evaluate_ppo(args.model, args.mode, args.episodes)
        print_metrics(metrics, title=f"PPO Agent Evaluation (mode={args.mode})")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved → {args.save}")


if __name__ == "__main__":
    main()
