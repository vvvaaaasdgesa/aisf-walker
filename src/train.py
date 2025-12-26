import os
import argparse
import shutil
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

ENV_ID = "BipedalWalker-v3"


def make_env():
    return gym.make(ENV_ID)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    p.add_argument("--n_steps", type=int, default=2048, help="Rollout length per update")
    p.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--eval_freq", type=int, default=10_000, help="Evaluation frequency (steps)")
    p.add_argument("--n_eval_episodes", type=int, default=5, help="Number of eval episodes")
    return p.parse_args()


def safe_copy(src, dst):
    """Copy a file, ensuring destination folder exists."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()

    run_name = f"ent{args.ent_coef}_ns{args.n_steps}_lr{args.learning_rate}_seed{args.seed}"
    run_dir = os.path.join("logs", "runs", run_name)

    # Per-run folders
    tb_dir = os.path.join(run_dir, "tb")
    best_dir = os.path.join(run_dir, "best")
    eval_dir = os.path.join(run_dir, "eval")

    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Also keep legacy folders for compatibility
    os.makedirs("logs/tb", exist_ok=True)
    os.makedirs("logs/best", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    # ---- Training env (updates normalization stats) ----
    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---- Eval env (does NOT update normalization stats) ----
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=eval_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tb_dir,  # per-run TB logs
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=args.learning_rate,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_cb,
        tb_log_name="PPO",
    )

    # Save per-run artifacts
    final_model_path = os.path.join(run_dir, "final_model")
    vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")

    model.save(final_model_path)
    train_env.save(vecnorm_path)

    # Record which run was last trained
    with open("logs/LATEST_RUN.txt", "w") as f:
        f.write(run_dir + "\n")

    print(f"\n✅ Run saved to: {run_dir}")
    print(f"   - Best model: {os.path.join(best_dir, 'best_model.zip')}")
    print(f"   - Final model: {final_model_path}.zip")
    print(f"   - VecNormalize: {vecnorm_path}")
    print("   - TensorBoard: ", tb_dir)

    per_run_best = os.path.join(best_dir, "best_model.zip")
    per_run_final = final_model_path + ".zip"

    if os.path.exists(per_run_best):
        safe_copy(per_run_best, os.path.join("logs", "best", "best_model.zip"))
    if os.path.exists(per_run_final):
        safe_copy(per_run_final, os.path.join("logs", "final_model.zip"))
    if os.path.exists(vecnorm_path):
        safe_copy(vecnorm_path, os.path.join("logs", "vecnormalize.pkl"))

    print("\n📌 Also copied latest artifacts to legacy paths:")
    print("   - logs/best/best_model.zip")
    print("   - logs/final_model.zip")
    print("   - logs/vecnormalize.pkl")


if __name__ == "__main__":
    main()
