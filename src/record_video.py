import os
import imageio
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


MODEL_PATH = "logs/final_model.zip"
VECNORM_PATH = "logs/vecnormalize.pkl"
OUT_GIF = "videos/best.gif"
ENV_ID = "BipedalWalker-v3"
MAX_STEPS = 1600


def make_env():
    return gym.make(ENV_ID, render_mode="rgb_array")


def main():
    os.makedirs("videos", exist_ok=True)

    # build env + load normalization
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(VECNORM_PATH, venv)

    # IMPORTANT: evaluation mode
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(MODEL_PATH)

    obs = venv.reset()
    frames = []
    total_reward = 0.0

    for t in range(MAX_STEPS):
        frame = venv.render()
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        total_reward += float(reward)

        if done:
            break

    imageio.mimsave(OUT_GIF, frames, fps=30)
    print(f"Saved GIF to {OUT_GIF}")
    print(f"Steps: {len(frames)}  Total normalized reward (vec env): {total_reward:.2f}")

    venv.close()


if __name__ == "__main__":
    main()

