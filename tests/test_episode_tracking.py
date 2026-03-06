#!/usr/bin/env python3
"""
测试 Episode Reward 追踪功能

验证：
1. Episode 累积奖励正确计算
2. Episode 长度正确追踪
3. Episode 完成时正确记录
4. 多个环境并行追踪
5. 边界情况处理
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import Mock, MagicMock
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))


class TestEpisodeTracker:
    """测试 Episode 追踪器"""

    def test_single_episode_completion(self):
        """测试单个 episode 完成时的追踪"""
        num_envs = 4
        device = "cpu"

        # 初始化追踪状态
        episode_rewards = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)
        completed_returns = []
        completed_lengths = []

        # 模拟 5 步，第 3 步 env 0 完成
        rewards_sequence = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),  # step 0
            torch.tensor([1.0, 2.0, 3.0, 4.0]),  # step 1
            torch.tensor([1.0, 2.0, 3.0, 4.0]),  # step 2: env 0 完成
            torch.tensor([1.0, 2.0, 3.0, 4.0]),  # step 3
            torch.tensor([1.0, 2.0, 3.0, 4.0]),  # step 4
        ]
        dones_sequence = [
            torch.tensor([False, False, False, False]),
            torch.tensor([False, False, False, False]),
            torch.tensor([True, False, False, False]),  # env 0 完成
            torch.tensor([False, False, False, False]),
            torch.tensor([False, False, False, False]),
        ]

        for step, (rewards, dones) in enumerate(zip(rewards_sequence, dones_sequence)):
            # 累积奖励
            episode_rewards += rewards
            episode_lengths += 1

            # 检查完成的 episode
            done_indices = dones.nonzero(as_tuple=True)[0]
            if len(done_indices) > 0:
                completed_returns.extend(episode_rewards[done_indices].tolist())
                completed_lengths.extend(episode_lengths[done_indices].tolist())
                # 重置
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

        # 验证
        assert len(completed_returns) == 1, f"应该有 1 个完成的 episode，实际 {len(completed_returns)}"
        assert completed_returns[0] == 3.0, f"env 0 的 episode return 应该是 3.0，实际 {completed_returns[0]}"
        assert completed_lengths[0] == 3, f"env 0 的 episode length 应该是 3，实际 {completed_lengths[0]}"

        # env 0 重置后继续累积
        assert episode_rewards[0].item() == 2.0, f"env 0 重置后应该累积 2.0，实际 {episode_rewards[0].item()}"
        assert episode_lengths[0].item() == 2, f"env 0 重置后应该累积 2 步，实际 {episode_lengths[0].item()}"

        print("✓ test_single_episode_completion 通过")

    def test_multiple_episodes_completion(self):
        """测试多个 episode 同时完成"""
        num_envs = 4
        device = "cpu"

        episode_rewards = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)
        completed_returns = []
        completed_lengths = []

        # 模拟 3 步，第 2 步 env 0 和 env 2 同时完成
        rewards_sequence = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),  # env 0, 2 完成
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ]
        dones_sequence = [
            torch.tensor([False, False, False, False]),
            torch.tensor([True, False, True, False]),  # env 0, 2 完成
            torch.tensor([False, False, False, False]),
        ]

        for rewards, dones in zip(rewards_sequence, dones_sequence):
            episode_rewards += rewards
            episode_lengths += 1

            done_indices = dones.nonzero(as_tuple=True)[0]
            if len(done_indices) > 0:
                completed_returns.extend(episode_rewards[done_indices].tolist())
                completed_lengths.extend(episode_lengths[done_indices].tolist())
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

        # 验证
        assert len(completed_returns) == 2, f"应该有 2 个完成的 episode，实际 {len(completed_returns)}"
        assert 2.0 in completed_returns, "env 0 的 return 2.0 应该在列表中"
        assert 6.0 in completed_returns, "env 2 的 return 6.0 应该在列表中"

        print("✓ test_multiple_episodes_completion 通过")

    def test_no_episode_completion(self):
        """测试没有 episode 完成的情况"""
        num_envs = 4
        device = "cpu"

        episode_rewards = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)
        completed_returns = []
        completed_lengths = []

        # 模拟 5 步，没有 episode 完成
        for _ in range(5):
            rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
            dones = torch.tensor([False, False, False, False])

            episode_rewards += rewards
            episode_lengths += 1

            done_indices = dones.nonzero(as_tuple=True)[0]
            if len(done_indices) > 0:
                completed_returns.extend(episode_rewards[done_indices].tolist())
                completed_lengths.extend(episode_lengths[done_indices].tolist())
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

        # 验证
        assert len(completed_returns) == 0, "不应该有完成的 episode"
        assert episode_rewards.sum().item() == 50.0, "总累积奖励应该是 50.0"
        assert episode_lengths.sum().item() == 20, "总累积长度应该是 20"

        print("✓ test_no_episode_completion 通过")

    def test_episode_stats_calculation(self):
        """测试 episode 统计计算"""
        completed_returns = [10.0, 20.0, 30.0, 40.0]
        completed_lengths = [100, 200, 300, 400]
        rollout_mean_reward = 5.0

        if completed_returns:
            episode_stats = {
                "mean_return": sum(completed_returns) / len(completed_returns),
                "mean_length": sum(completed_lengths) / len(completed_lengths),
                "count": len(completed_returns),
                "rollout_mean_reward": rollout_mean_reward,
            }
        else:
            episode_stats = {
                "mean_return": 0.0,
                "mean_length": 0.0,
                "count": 0,
                "rollout_mean_reward": rollout_mean_reward,
            }

        # 验证
        assert episode_stats["mean_return"] == 25.0
        assert episode_stats["mean_length"] == 250.0
        assert episode_stats["count"] == 4
        assert episode_stats["rollout_mean_reward"] == 5.0

        print("✓ test_episode_stats_calculation 通过")

    def test_2d_tensor_handling(self):
        """测试 2D tensor 的处理（rewards 和 dones 可能是 [N, 1] 形状）"""
        num_envs = 4
        device = "cpu"

        episode_rewards = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)
        completed_returns = []

        # 模拟 2D tensor
        rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # [4, 1]
        dones = torch.tensor([[True], [False], [True], [False]])  # [4, 1]

        # 处理 2D tensor
        rewards_flat = rewards.squeeze(-1) if rewards.dim() == 2 else rewards
        dones_flat = dones.squeeze(-1) if dones.dim() == 2 else dones

        episode_rewards += rewards_flat
        episode_lengths += 1

        done_indices = dones_flat.nonzero(as_tuple=True)[0]
        if len(done_indices) > 0:
            completed_returns.extend(episode_rewards[done_indices].tolist())
            episode_rewards[done_indices] = 0
            episode_lengths[done_indices] = 0

        # 验证
        assert len(completed_returns) == 2
        assert 1.0 in completed_returns
        assert 3.0 in completed_returns

        print("✓ test_2d_tensor_handling 通过")


def run_all_tests():
    """运行所有测试"""
    print("开始测试 Episode Tracking 功能...\n")

    tracker = TestEpisodeTracker()
    tracker.test_single_episode_completion()
    tracker.test_multiple_episodes_completion()
    tracker.test_no_episode_completion()
    tracker.test_episode_stats_calculation()
    tracker.test_2d_tensor_handling()

    print("\n所有测试通过！✓")


if __name__ == "__main__":
    run_all_tests()
