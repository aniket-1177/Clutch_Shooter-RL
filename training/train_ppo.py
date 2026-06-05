"""
training/train_ppo.py  -  PPO training for Basketball2DEnv (Stages 2-4).

Usage
-----
    python training/train_ppo.py --mode static      --timesteps 200000
    python training/train_ppo.py --mode moving      --timesteps 300000
    python training/train_ppo.py --mode engineered  --timesteps 300000

After training, evaluate with the BEST checkpoint:
    python evaluation/evaluate.py --model models/best_engineered/best_model --mode engineered
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.basketball_2d_env import Basketball2DEnv
from utils.config import PPOConfig, EnvConfig
from utils.logger import get_logger

logger = get_logger("train_ppo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      type=str,   default="engineered",
                        choices=["static", "moving", "engineered"])
    parser.add_argument("--timesteps", type=int,   default=300_000)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--n-steps",   type=int,   default=2048)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--eval-freq", type=int,   default=10_000)
    parser.add_argument("--save-dir",  type=str,   default="models")
    return parser.parse_args()


def train(
    mode: str = "engineered",
    ppo_config: PPOConfig = None,
    env_config: EnvConfig = None,
    save_dir: str = "models",
    eval_freq: int = 10_000,
):
    ppo_config = ppo_config or PPOConfig()
    env_config = env_config or EnvConfig()
    save_path  = f"{save_dir}/ppo_{mode}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    env      = Monitor(Basketball2DEnv(mode=mode, config=env_config))
    eval_env = Monitor(Basketball2DEnv(mode=mode, config=env_config))

    logger.info("Running env check...")
    check_env(env)
    logger.info("check_env passed")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_{mode}",
        log_path="results/",
        eval_freq=eval_freq,
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=f"{save_dir}/checkpoints/{mode}/",
        name_prefix=f"ppo_{mode}",
        verbose=0,
    )

    model = PPO(
        policy        = ppo_config.policy,
        env           = env,
        learning_rate = ppo_config.learning_rate,
        n_steps       = ppo_config.n_steps,
        batch_size    = ppo_config.batch_size,
        n_epochs      = ppo_config.n_epochs,
        gamma         = ppo_config.gamma,
        gae_lambda    = ppo_config.gae_lambda,
        clip_range    = ppo_config.clip_range,
        ent_coef      = 0.01,                    # entropy bonus prevents premature convergence
        verbose       = ppo_config.verbose,
        seed          = ppo_config.seed,
        device        = "cpu",                   # MlpPolicy is faster on CPU
        policy_kwargs = dict(net_arch=[128, 128]),
    )

    logger.info(f"Training | mode={mode} | timesteps={ppo_config.total_timesteps} | device=cpu")

    model.learn(
        total_timesteps=ppo_config.total_timesteps,
        callback=[eval_cb, ckpt_cb],
        progress_bar=True,
    )

    model.save(save_path)
    logger.info(f"Final model  -> {save_path}.zip")
    logger.info(f"Best model   -> {save_dir}/best_{mode}/best_model.zip")

    env.close()
    eval_env.close()
    return model


def main():
    args = parse_args()
    config = PPOConfig(
        learning_rate   = args.lr,
        n_steps         = args.n_steps,
        total_timesteps = args.timesteps,
        seed            = args.seed,
    )
    train(mode=args.mode, ppo_config=config, save_dir=args.save_dir, eval_freq=args.eval_freq)


if __name__ == "__main__":
    main()
