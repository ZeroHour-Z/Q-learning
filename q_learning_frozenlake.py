import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import os

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUSTOM_12x12_MAP = [
    "SFFFFFFFHFFF",
    "FFFFFFFFFFFF",
    "FFFHFFFFHFFF",
    "FFFFFFFFFFFF",
    "FFFFFFFFFFFH",
    "FFHFFFFFFFFF",
    "FFFFFFFHFFFF",
    "FFFFFFFFHFFF",
    "FHFFFFFFFFFF",
    "FFFFFFHFFFFF",
    "FFFHFFFFFFHF",
    "FFFFFFFFFFFG",
]


class QLearningAgent:
    """表格型 Q-learning 智能体"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 q_init=0.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.full((n_states, n_actions), q_init, dtype=float)

    def choose_action(self, state):
        """epsilon-greedy 动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Q-learning 更新: Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]"""
        td_target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_greedy_policy(self):
        return np.argmax(self.q_table, axis=1)


def train(env, agent, n_episodes, log_interval=2000):
    """训练过程，记录每个 episode 的奖励和步数"""
    rewards_history = []
    steps_history = []
    success_history = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, float(terminated))
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        steps_history.append(steps)
        success_history.append(1.0 if total_reward > 0 else 0.0)

        if (ep + 1) % log_interval == 0:
            recent = slice(-log_interval, None)
            avg_r = np.mean(rewards_history[recent])
            avg_s = np.mean(success_history[recent])
            print(f"  Episode {ep+1:>6d} | "
                  f"平均奖励: {avg_r:.3f} | "
                  f"成功率: {avg_s:.2%} | "
                  f"epsilon: {agent.epsilon:.4f}")

    return rewards_history, steps_history, success_history


def test(env, agent, n_episodes=1000):
    """测试过程：使用贪婪策略评估"""
    total_rewards = []
    total_steps = []
    successes = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = int(np.argmax(agent.q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
            steps += 1

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        if ep_reward > 0:
            successes += 1

    avg_reward = np.mean(total_rewards)
    success_rate = successes / n_episodes
    avg_steps = np.mean(total_steps)
    return avg_reward, success_rate, avg_steps


def moving_average(data, window=200):
    """计算滑动平均"""
    cumsum = np.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    result = np.empty_like(data, dtype=float)
    result[:window] = cumsum[:window] / np.arange(1, window + 1)
    result[window:] = cumsum[window:] / window
    return result

# 可视化
def plot_training_curves(rewards_8x8, rewards_12x12,
                         success_8x8, success_12x12, window=200):
    """绘制训练奖励曲线和成功率曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ma_r8 = moving_average(np.array(rewards_8x8), window)
    ma_r12 = moving_average(np.array(rewards_12x12), window)
    axes[0].plot(ma_r8, label="8×8 地图", linewidth=1.2)
    axes[0].plot(ma_r12, label="12×12 地图", linewidth=1.2)
    axes[0].set_xlabel("训练回合")
    axes[0].set_ylabel("平均奖励（滑动窗口）")
    axes[0].set_title("训练奖励曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ma_s8 = moving_average(np.array(success_8x8), window)
    ma_s12 = moving_average(np.array(success_12x12), window)
    axes[1].plot(ma_s8, label="8×8 地图", linewidth=1.2)
    axes[1].plot(ma_s12, label="12×12 地图", linewidth=1.2)
    axes[1].set_xlabel("训练回合")
    axes[1].set_ylabel("成功率（滑动窗口）")
    axes[1].set_title("训练成功率曲线")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[已保存] {OUTPUT_DIR}/training_curves.png")


def print_policy(policy, nrow, ncol, map_desc):
    """以箭头形式打印策略"""
    action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    print(f"\n{'='*40}")
    print(f"  最终贪婪策略 ({nrow}×{ncol})")
    print(f"{'='*40}")
    for r in range(nrow):
        row_str = ""
        for c in range(ncol):
            idx = r * ncol + c
            cell = map_desc[r][c]
            if cell == "S":
                row_str += f" S{action_symbols[policy[idx]]}"
            elif cell == "G":
                row_str += "  G "
            elif cell == "H":
                row_str += "  H "
            else:
                row_str += f"  {action_symbols[policy[idx]]} "
        print(row_str)
    print(f"{'='*40}\n")


def plot_policy_grid(policy, nrow, ncol, map_desc, title, filename):
    """绘制策略网格图"""
    arrow_dx = {0: -0.3, 1: 0, 2: 0.3, 3: 0}
    arrow_dy = {0: 0, 1: 0.3, 2: 0, 3: -0.3}

    fig, ax = plt.subplots(figsize=(max(ncol * 0.9, 6), max(nrow * 0.9, 6)))

    for r in range(nrow):
        for c in range(ncol):
            cell = map_desc[r][c]
            if cell == "H":
                color = "#2c3e50"
            elif cell == "G":
                color = "#27ae60"
            elif cell == "S":
                color = "#3498db"
            else:
                color = "#ecf0f1"
            rect = plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#7f8c8d", linewidth=1)
            ax.add_patch(rect)

            idx = r * ncol + c
            if cell in ("H", "G"):
                ax.text(c + 0.5, r + 0.5, cell, ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white")
            else:
                a = policy[idx]
                ax.annotate("", xy=(c + 0.5 + arrow_dx[a], r + 0.5 + arrow_dy[a]),
                            xytext=(c + 0.5, r + 0.5),
                            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))

    ax.set_xlim(0, ncol)
    ax.set_ylim(nrow, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(ncol) + 0.5)
    ax.set_xticklabels(range(ncol))
    ax.set_yticks(np.arange(nrow) + 0.5)
    ax.set_yticklabels(range(nrow))
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(length=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"[已保存] {OUTPUT_DIR}/{filename}")


def create_agent_animation(agent, map_desc, nrow, ncol, title, filename,
                           env_kwargs, max_attempts=50):
    """生成智能体按贪婪策略行走的 GIF 动画"""
    env = gym.make("FrozenLake-v1", **env_kwargs)

    trajectory = None
    for _ in range(max_attempts):
        states = []
        state, _ = env.reset()
        states.append(state)
        done = False
        while not done:
            action = int(np.argmax(agent.q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            states.append(state)
            done = terminated or truncated
        if reward > 0:
            trajectory = states
            break
    env.close()

    if trajectory is None:
        print(f"[跳过] {filename} — 未能在 {max_attempts} 次尝试中找到成功轨迹")
        return

    arrow_dx = {0: -0.3, 1: 0, 2: 0.3, 3: 0}
    arrow_dy = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    policy = agent.get_greedy_policy()
    cell_size = 0.85
    fig, ax = plt.subplots(figsize=(max(ncol * cell_size, 5), max(nrow * cell_size, 5)))

    def draw_base(ax):
        ax.clear()
        for r in range(nrow):
            for c in range(ncol):
                cell = map_desc[r][c]
                if cell == "H":
                    color = "#2c3e50"
                elif cell == "G":
                    color = "#27ae60"
                elif cell == "S":
                    color = "#3498db"
                else:
                    color = "#ecf0f1"
                rect = plt.Rectangle((c, r), 1, 1, facecolor=color,
                                     edgecolor="#7f8c8d", linewidth=1)
                ax.add_patch(rect)

                idx = r * ncol + c
                if cell in ("H", "G"):
                    ax.text(c + 0.5, r + 0.5, cell, ha="center", va="center",
                            fontsize=12, fontweight="bold", color="white")
                else:
                    a = policy[idx]
                    ax.annotate("", xy=(c + 0.5 + arrow_dx[a] * 0.6,
                                        r + 0.5 + arrow_dy[a] * 0.6),
                                xytext=(c + 0.5, r + 0.5),
                                arrowprops=dict(arrowstyle="->", color="#bdc3c7",
                                                lw=1.2))

        ax.set_xlim(0, ncol)
        ax.set_ylim(nrow, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(ncol) + 0.5)
        ax.set_xticklabels(range(ncol), fontsize=8)
        ax.set_yticks(np.arange(nrow) + 0.5)
        ax.set_yticklabels(range(nrow), fontsize=8)
        ax.tick_params(length=0)

    visited_patches = []
    agent_marker = [None]

    def update(frame):
        for p in visited_patches:
            p.remove()
        visited_patches.clear()
        if agent_marker[0] is not None:
            agent_marker[0].remove()
            agent_marker[0] = None

        draw_base(ax)

        for i in range(frame):
            s = trajectory[i]
            r, c = divmod(s, ncol)
            cell = map_desc[r][c]
            if cell not in ("H", "G"):
                p = plt.Rectangle((c + 0.1, r + 0.1), 0.8, 0.8,
                                  facecolor="#f39c12", alpha=0.25,
                                  edgecolor="none")
                ax.add_patch(p)
                visited_patches.append(p)

        s = trajectory[frame]
        r, c = divmod(s, ncol)
        marker = ax.plot(c + 0.5, r + 0.5, "o", markersize=max(18 - ncol, 10),
                         color="#e74c3c", markeredgecolor="white",
                         markeredgewidth=2, zorder=10)[0]
        agent_marker[0] = marker

        step_info = f"步骤 {frame}/{len(trajectory)-1}"
        if frame == len(trajectory) - 1:
            step_info += "  ★ 到达终点！"
        ax.set_title(f"{title}\n{step_info}", fontsize=12, pad=8)

    anim = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                   interval=400, repeat=True, repeat_delay=2000)
    anim.save(os.path.join(OUTPUT_DIR, filename), writer="pillow", dpi=100)
    plt.close(fig)
    print(f"[已保存] {OUTPUT_DIR}/{filename}")


# ==================== 主程序 ====================

def run_experiment(map_name, desc, n_episodes, alpha, gamma,
                   epsilon, epsilon_min, epsilon_decay,
                   q_init=0.0, max_episode_steps=None):
    """对单个地图执行完整的训练和测试流程"""
    kwargs = {"is_slippery": True}
    if desc is not None:
        kwargs["desc"] = desc
    else:
        kwargs["map_name"] = map_name
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make("FrozenLake-v1", **kwargs)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(
        n_states, n_actions,
        alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        q_init=q_init,
    )

    print(f"\n{'#'*50}")
    print(f"  开始训练: {map_name} | 回合数: {n_episodes}")
    print(f"  参数: α={alpha}, γ={gamma}, ε衰减={epsilon_decay}")
    print(f"{'#'*50}")

    rewards, steps, successes = train(env, agent, n_episodes)

    avg_r, sr, avg_s = test(env, agent, n_episodes=1000)
    print(f"\n  [测试结果] 平均奖励: {avg_r:.3f} | 成功率: {sr:.2%} | 平均步数: {avg_s:.1f}")

    unwrapped = env.unwrapped
    nrow, ncol = unwrapped.nrow, unwrapped.ncol
    map_desc = unwrapped.desc.astype(str).tolist()
    policy = agent.get_greedy_policy()

    print_policy(policy, nrow, ncol, map_desc)

    env.close()
    return agent, rewards, steps, successes, policy, nrow, ncol, map_desc


def main():
    # ---- 8x8 预设地图 ----
    agent_8, rewards_8, steps_8, success_8, policy_8, nr8, nc8, desc8 = run_experiment(
        map_name="8x8", desc=None,
        n_episodes=50000,
        alpha=0.15, gamma=0.99,
        epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.99995,
        q_init=1.0,
    )

    # ---- 12x12 自定义地图 ----
    agent_12, rewards_12, steps_12, success_12, policy_12, nr12, nc12, desc12 = run_experiment(
        map_name="12x12 (自定义)", desc=CUSTOM_12x12_MAP,
        n_episodes=100000,
        alpha=0.15, gamma=0.99,
        epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.99998,
        q_init=1.0,
        max_episode_steps=500,
    )

    # ---- 可视化 ----
    print("\n正在生成可视化图表...")
    plot_training_curves(rewards_8, rewards_12, success_8, success_12)
    plot_policy_grid(policy_8, nr8, nc8, desc8, "8×8 最终策略", "policy_8x8.png")
    plot_policy_grid(policy_12, nr12, nc12, desc12, "12×12 最终策略", "policy_12x12.png")

    # ---- 动态效果图 ----
    print("\n正在生成动态效果图...")
    create_agent_animation(
        agent_8, desc8, nr8, nc8,
        "8×8 智能体行走动画", "animation_8x8.gif",
        env_kwargs={"map_name": "8x8", "is_slippery": True},
    )
    create_agent_animation(
        agent_12, desc12, nr12, nc12,
        "12×12 智能体行走动画", "animation_12x12.gif",
        env_kwargs={"desc": CUSTOM_12x12_MAP, "is_slippery": True,
                    "max_episode_steps": 500},
    )

    # ---- 汇总比较 ----
    avg_r8, sr8, avg_s8 = test(
        gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True), agent_8
    )
    avg_r12, sr12, avg_s12 = test(
        gym.make("FrozenLake-v1", desc=CUSTOM_12x12_MAP, is_slippery=True,
                 max_episode_steps=500), agent_12
    )

    print(f"\n{'='*60}")
    print(f"  最终性能比较（贪婪策略，1000 回合测试）")
    print(f"{'='*60}")
    print(f"  {'指标':<12} {'8×8 地图':>12} {'12×12 地图':>12}")
    print(f"  {'-'*36}")
    print(f"  {'平均奖励':<12} {avg_r8:>12.3f} {avg_r12:>12.3f}")
    print(f"  {'成功率':<12} {sr8:>11.2%} {sr12:>11.2%}")
    print(f"  {'平均步数':<12} {avg_s8:>12.1f} {avg_s12:>12.1f}")
    print(f"{'='*60}")
    print(f"\n所有结果已保存至 '{OUTPUT_DIR}/' 目录。")


if __name__ == "__main__":
    main()
