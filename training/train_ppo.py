# """
# training/train_ppo.py
# ----------------------
# Stages 2–4: Train a PPO agent on the Basketball2D environment.

# Usage
# -----
#     python training/train_ppo.py --mode static      --timesteps 100000
#     python training/train_ppo.py --mode moving      --timesteps 150000
#     python training/train_ppo.py --mode engineered  --timesteps 150000  # recommended
# """

# import argparse
# import json
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
# from stable_baselines3.common.monitor import Monitor

# from envs.basketball_2d_env import Basketball2DEnv
# from utils.config import PPOConfig, EnvConfig
# from utils.logger import get_logger

# logger = get_logger("train_ppo")


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train PPO on Basketball2DEnv")
#     parser.add_argument("--mode",       type=str,   default="engineered",
#                         choices=["static", "moving", "engineered"])
#     parser.add_argument("--timesteps",  type=int,   default=150_000)
#     parser.add_argument("--lr",         type=float, default=3e-4,   help="Learning rate")
#     parser.add_argument("--n-steps",    type=int,   default=2048,   help="Rollout steps")
#     parser.add_argument("--seed",       type=int,   default=42)
#     parser.add_argument("--eval-freq",  type=int,   default=10_000, help="Evaluate every N steps")
#     parser.add_argument("--save-dir",   type=str,   default="models")
#     return parser.parse_args()


# def train(
#     mode: str = "engineered",
#     ppo_config: PPOConfig = None,
#     env_config: EnvConfig = None,
#     save_dir: str = "models",
#     eval_freq: int = 10_000,
# ):
#     """
#     Full PPO training pipeline with evaluation and checkpointing.

#     PPO (Proximal Policy Optimisation) is an actor-critic algorithm.
#     It alternates between:
#       1. Collecting rollouts with the current policy.
#       2. Computing advantage estimates (how much better was this action?).
#       3. Updating the policy with a clipped objective to prevent large updates.

#     The "MlpPolicy" uses a fully-connected neural network for both the
#     actor (what action to take) and the critic (how good is this state).
#     """
#     ppo_config = ppo_config or PPOConfig()
#     env_config = env_config or EnvConfig()
#     save_path  = f"{save_dir}/ppo_{mode}"

#     Path(save_dir).mkdir(parents=True, exist_ok=True)
#     Path("results").mkdir(exist_ok=True)

#     # ── Build + validate environment ──────────────────────────────────────
#     env = Monitor(Basketball2DEnv(mode=mode, config=env_config))
#     logger.info("Running environment sanity check...")
#     check_env(env)
#     logger.info("Environment check passed ✓")

#     eval_env = Monitor(Basketball2DEnv(mode=mode, config=env_config))

#     # ── Callbacks ─────────────────────────────────────────────────────────
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=f"{save_dir}/best_{mode}",
#         log_path="results/",
#         eval_freq=eval_freq,
#         n_eval_episodes=20,
#         deterministic=True,
#         verbose=0,
#     )
#     checkpoint_callback = CheckpointCallback(
#         save_freq=eval_freq,
#         save_path=f"{save_dir}/checkpoints/{mode}/",
#         name_prefix=f"ppo_{mode}",
#         verbose=0,
#     )

#     # ── Initialise PPO ────────────────────────────────────────────────────
#     model = PPO(
#         policy        = ppo_config.policy,
#         env           = env,
#         learning_rate = ppo_config.learning_rate,
#         n_steps       = ppo_config.n_steps,
#         batch_size    = ppo_config.batch_size,
#         n_epochs      = ppo_config.n_epochs,
#         gamma         = ppo_config.gamma,
#         gae_lambda    = ppo_config.gae_lambda,
#         clip_range    = ppo_config.clip_range,
#         verbose       = ppo_config.verbose,
#         seed          = ppo_config.seed,
#         tensorboard_log="results/tensorboard/",
#     )

#     logger.info(
#         f"Training PPO | mode={mode} | timesteps={ppo_config.total_timesteps} | "
#         f"lr={ppo_config.learning_rate} | n_steps={ppo_config.n_steps}"
#     )

#     # ── Train ─────────────────────────────────────────────────────────────
#     model.learn(
#         total_timesteps=ppo_config.total_timesteps,
#         callback=[eval_callback, checkpoint_callback],
#         progress_bar=True,
#     )

#     # ── Save final model ──────────────────────────────────────────────────
#     model.save(save_path)
#     logger.info(f"Model saved → {save_path}.zip")

#     env.close()
#     eval_env.close()

#     return model


# def main():
#     args = parse_args()
#     ppo_config = PPOConfig(
#         learning_rate   = args.lr,
#         n_steps         = args.n_steps,
#         total_timesteps = args.timesteps,
#         seed            = args.seed,
#     )
#     train(
#         mode       = args.mode,
#         ppo_config = ppo_config,
#         save_dir   = args.save_dir,
#         eval_freq  = args.eval_freq,
#     )


# if __name__ == "__main__":
#     main()


# """
# training/train_ppo.py
# ----------------------
# Stages 2–4: Train a PPO agent on the Basketball2D environment.

# Usage
# -----
#     python training/train_ppo.py --mode static      --timesteps 200000
#     python training/train_ppo.py --mode moving      --timesteps 300000
#     python training/train_ppo.py --mode engineered  --timesteps 300000  # recommended

# Why device=cpu?
#   MlpPolicy on a small env is faster on CPU — GPU overhead > compute gain.
#   The SB3 warning you may have seen is confirming exactly this.
# """

# import argparse
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
# from stable_baselines3.common.monitor import Monitor

# from envs.basketball_2d_env import Basketball2DEnv
# from utils.config import PPOConfig, EnvConfig
# from utils.logger import get_logger

# logger = get_logger("train_ppo")


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train PPO on Basketball2DEnv")
#     parser.add_argument("--mode",       type=str,   default="engineered",
#                         choices=["static", "moving", "engineered"])
#     parser.add_argument("--timesteps",  type=int,   default=300_000)
#     parser.add_argument("--lr",         type=float, default=3e-4)
#     parser.add_argument("--n-steps",    type=int,   default=2048)
#     parser.add_argument("--seed",       type=int,   default=42)
#     parser.add_argument("--eval-freq",  type=int,   default=10_000)
#     parser.add_argument("--save-dir",   type=str,   default="models")
#     return parser.parse_args()


# def train(
#     mode: str = "engineered",
#     ppo_config: PPOConfig = None,
#     env_config: EnvConfig = None,
#     save_dir: str = "models",
#     eval_freq: int = 10_000,
# ):
#     """
#     Full PPO training pipeline with evaluation and checkpointing.

#     Key decisions:
#     - device="cpu": MlpPolicy is faster on CPU; avoids the SB3 GPU warning.
#     - ent_coef=0.01: small entropy bonus keeps the policy from collapsing
#       to a single action too early (critical for the shooting environment
#       where argmax can get stuck at "shoot immediately").
#     - net_arch=[128,128]: slightly larger than the SB3 default [64,64]
#       to handle the 5-dimensional observation with defender tracking.
#     - n_steps=2048 with batch_size=64: good balance of sample efficiency
#       and update stability for episodes averaging 5-15 steps.
#     """
#     ppo_config = ppo_config or PPOConfig()
#     env_config = env_config or EnvConfig()
#     save_path  = f"{save_dir}/ppo_{mode}"

#     Path(save_dir).mkdir(parents=True, exist_ok=True)
#     Path("results").mkdir(exist_ok=True)

#     # ── Build + validate environment ──────────────────────────────────────
#     env = Monitor(Basketball2DEnv(mode=mode, config=env_config))
#     logger.info("Running environment sanity check...")
#     check_env(env)
#     logger.info("Environment check passed ✓")

#     eval_env = Monitor(Basketball2DEnv(mode=mode, config=env_config))

#     # ── Callbacks ─────────────────────────────────────────────────────────
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=f"{save_dir}/best_{mode}",
#         log_path="results/",
#         eval_freq=eval_freq,
#         n_eval_episodes=20,
#         deterministic=True,
#         verbose=0,
#     )
#     checkpoint_callback = CheckpointCallback(
#         save_freq=eval_freq,
#         save_path=f"{save_dir}/checkpoints/{mode}/",
#         name_prefix=f"ppo_{mode}",
#         verbose=0,
#     )

#     # ── Initialise PPO ────────────────────────────────────────────────────
#     # policy_kwargs lets us customise the neural network architecture.
#     # net_arch=[128,128] = two hidden layers of 128 units each (for both
#     # actor and critic). Larger than default [64,64] for better generalisation
#     # across the 10x10 grid with varying defender positions.
#     policy_kwargs = dict(net_arch=[128, 128])

#     model = PPO(
#         policy        = ppo_config.policy,
#         env           = env,
#         learning_rate = ppo_config.learning_rate,
#         n_steps       = ppo_config.n_steps,
#         batch_size    = ppo_config.batch_size,
#         n_epochs      = ppo_config.n_epochs,
#         gamma         = ppo_config.gamma,
#         gae_lambda    = ppo_config.gae_lambda,
#         clip_range    = ppo_config.clip_range,
#         ent_coef      = 0.01,          # Entropy bonus: discourages premature convergence
#         verbose       = ppo_config.verbose,
#         seed          = ppo_config.seed,
#         device        = "cpu",         # MlpPolicy is faster on CPU
#         policy_kwargs = policy_kwargs,
#         tensorboard_log="results/tensorboard/",
#     )

#     logger.info(
#         f"Training PPO | mode={mode} | timesteps={ppo_config.total_timesteps} | "
#         f"lr={ppo_config.learning_rate} | n_steps={ppo_config.n_steps} | device=cpu"
#     )

#     # ── Train ─────────────────────────────────────────────────────────────
#     model.learn(
#         total_timesteps=ppo_config.total_timesteps,
#         callback=[eval_callback, checkpoint_callback],
#         progress_bar=True,
#     )

#     # ── Save final model ──────────────────────────────────────────────────
#     model.save(save_path)
#     logger.info(f"Final model saved → {save_path}.zip")

#     # The best model during training is already saved by EvalCallback at:
#     # models/best_{mode}/best_model.zip
#     logger.info(f"Best model (during training) → {save_dir}/best_{mode}/best_model.zip")

#     env.close()
#     eval_env.close()

#     return model


# def main():
#     args = parse_args()
#     ppo_config = PPOConfig(
#         learning_rate   = args.lr,
#         n_steps         = args.n_steps,
#         total_timesteps = args.timesteps,
#         seed            = args.seed,
#     )
#     train(
#         mode       = args.mode,
#         ppo_config = ppo_config,
#         save_dir   = args.save_dir,
#         eval_freq  = args.eval_freq,
#     )


# if __name__ == "__main__":
#     main()


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
