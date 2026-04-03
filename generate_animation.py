"""
单独生成动态效果图（GIF），使用快速训练后的 Q-table
"""

from q_learning_frozenlake import (
    QLearningAgent, train, create_agent_animation,
    CUSTOM_12x12_MAP, OUTPUT_DIR,
)
import gymnasium as gym
import numpy as np


def quick_train_and_animate(map_name, desc, n_episodes, epsilon_decay,
                            title, gif_name, q_init=1.0,
                            max_episode_steps=None):
    kwargs = {"is_slippery": True}
    if desc is not None:
        kwargs["desc"] = desc
    else:
        kwargs["map_name"] = map_name
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps

    env = gym.make("FrozenLake-v1", **kwargs)
    agent = QLearningAgent(
        env.observation_space.n, env.action_space.n,
        alpha=0.15, gamma=0.99,
        epsilon=1.0, epsilon_min=0.02, epsilon_decay=epsilon_decay,
        q_init=q_init,
    )
    print(f"快速训练 {map_name} ({n_episodes} 回合)...")
    train(env, agent, n_episodes, log_interval=n_episodes)

    unwrapped = env.unwrapped
    nrow, ncol = unwrapped.nrow, unwrapped.ncol
    map_desc = unwrapped.desc.astype(str).tolist()
    env.close()

    create_agent_animation(agent, map_desc, nrow, ncol,
                           title, gif_name, env_kwargs=kwargs)


if __name__ == "__main__":
    quick_train_and_animate(
        "8x8", None, 50000, 0.99995,
        "8×8 智能体行走动画", "animation_8x8.gif",
    )
    quick_train_and_animate(
        "12x12", CUSTOM_12x12_MAP, 100000, 0.99998,
        "12×12 智能体行走动画", "animation_12x12.gif",
        max_episode_steps=500,
    )
    print("动态效果图生成完毕！")
