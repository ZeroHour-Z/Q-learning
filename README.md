# 基于随机滑动 FrozenLake 的 Q-learning 求解实验

使用表格型 Q-learning 求解 Gymnasium FrozenLake-v1 环境，在 **8×8 预设地图**与**自定义 12×12 地图**（均启用 `is_slippery=True`）上进行训练，并对比两种地图的学习效果。

## 项目结构

```
rl/
├── q_learning_frozenlake.py   # 主脚本：训练、测试、可视化、动画
├── requirements.txt           # 依赖版本
├── report.md                  # 实验报告
├── todo.md                    # 实验要求
└── results/                   # 输出目录（运行后自动创建）
    ├── training_curves.png    # 训练奖励曲线与成功率曲线
    ├── policy_8x8.png         # 8×8 最终策略网格图
    ├── policy_12x12.png       # 12×12 最终策略网格图
    ├── animation_8x8.gif      # 8×8 智能体行走动画
    └── animation_12x12.gif    # 12×12 智能体行走动画
```

## 环境配置

```bash
conda create -n rl_course python=3.10
conda activate rl_course
pip install -r requirements.txt
```

## 运行

```bash
conda activate rl_course
python q_learning_frozenlake.py
```

运行完成后，所有图表和动画保存在 `results/` 目录。

> 完整训练（8×8 共 50,000 回合，12×12 共 100,000 回合）约需 3~5 分钟。

## 算法概述

**核心更新公式（Q-learning）：**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ R + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

**关键设计决策：**

| 设计 | 说明 |
|------|------|
| ε-greedy 探索 | 从 ε=1.0 指数衰减至 ε_min=0.02，兼顾探索与利用 |
| 乐观初始化 | Q 表初始值设为 1.0，激励在大地图上持续探索 |
| 慢速 ε 衰减 | 大地图随机探索首次到达终点需大量回合，衰减过快会导致训练失败 |

## 实验参数

| 参数 | 8×8 | 12×12 |
|------|-----|-------|
| 学习率 α | 0.15 | 0.15 |
| 折扣因子 γ | 0.99 | 0.99 |
| ε 衰减 | 0.99995 | 0.99998 |
| 训练回合数 | 50,000 | 100,000 |

## 测试结果

| 指标 | 8×8 | 12×12 |
|------|-----|-------|
| 成功率 | ~57% | ~100% |
| 平均步数 | ~76 | ~114 |

详细分析见 [report.md](report.md)。
