"""
测试 episode_reward 日志功能

问题背景：
当前 train_student_rl_finetune.py 中缺少 episode_reward 日志记录，
无法判断策略是否真正在改进。

测试目标：
1. 验证 collect_rollouts 返回 episode 统计信息
2. 验证 run_training_iteration 返回 episode 统计信息
3. 验证 log_training_metrics 正确记录 episode 统计
"""

import sys
from pathlib import Path
import torch
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))


class TestEpisodeRewardLogging:
    """测试 episode_reward 日志功能"""

    def test_collect_rollouts_returns_episode_stats(self):
        """测试：collect_rollouts 应该返回 episode 统计信息"""
        from train_student_rl_finetune import collect_rollouts

        # 创建模拟环境
        mock_env = create_mock_env(num_envs=4, num_prop=53)

        # 创建模拟 PPO
        mock_ppo = create_mock_ppo()

        # 收集 rollouts
        result = collect_rollouts(
            env=mock_env,
            ppo=mock_ppo,
            domain_rand=None,
            num_steps=10,
            num_prop=53,
        )

        # 验证返回值包含 episode 统计信息
        assert "episode_stats" in result, (
            "collect_rollouts should return episode_stats"
        )
        # 新的 episode_stats 结构
        assert "mean_return" in result["episode_stats"], (
            "episode_stats should contain mean_return"
        )
        assert "mean_length" in result["episode_stats"], (
            "episode_stats should contain mean_length"
        )
        assert "count" in result["episode_stats"], (
            "episode_stats should contain count"
        )
        assert "rollout_mean_reward" in result["episode_stats"], (
            "episode_stats should contain rollout_mean_reward"
        )

    def test_run_training_iteration_returns_episode_stats(self):
        """测试：run_training_iteration 应该返回 episode 统计信息"""
        from train_student_rl_finetune import run_training_iteration

        # 创建模拟对象
        mock_env = create_mock_env(num_envs=4, num_prop=53)
        mock_ppo = create_mock_ppo()
        mock_actor_critic = create_mock_actor_critic()

        # 运行训练迭代
        train_info = run_training_iteration(
            env=mock_env,
            ppo=mock_ppo,
            actor_critic=mock_actor_critic,
            domain_rand=None,
            num_steps=10,
            iteration=0,
            num_prop=53,
        )

        # 验证返回值包含 episode 统计信息
        assert "episode_stats" in train_info, (
            "run_training_iteration should return episode_stats"
        )

    def test_log_training_metrics_logs_episode_reward(self):
        """测试：log_training_metrics 应该记录 episode_reward"""
        from train_student_rl_finetune import log_training_metrics

        # 创建模拟 logger
        mock_logger = Mock()
        mock_logger.enabled = True
        mock_logger.log = Mock()

        # 训练信息
        train_info = {
            "value_loss": 0.5,
            "surrogate_loss": 0.1,
            "entropy": 8.0,
            "kl": 0.01,
            "learning_rate": 1e-4,
        }

        # Episode 统计（新结构）
        episode_stats = {
            "mean_return": 150.5,
            "mean_length": 200.0,
            "count": 10,
            "rollout_mean_reward": 0.75,
        }

        # 记录日志
        log_training_metrics(
            logger=mock_logger,
            iteration=100,
            train_info=train_info,
            episode_stats=episode_stats,
        )

        # 验证 logger.log 被调用
        mock_logger.log.assert_called_once()

        # 获取调用参数
        call_args = mock_logger.log.call_args
        logged_metrics = call_args[0][0]  # 第一个位置参数

        # 验证 episode 统计被记录
        assert "episode/mean_return" in logged_metrics, (
            "episode/mean_return should be logged"
        )
        assert logged_metrics["episode/mean_return"] == 150.5
        assert "episode/mean_length" in logged_metrics
        assert "episode/count" in logged_metrics
        assert "episode/rollout_mean_reward" in logged_metrics


def create_mock_env(num_envs: int = 4, num_prop: int = 53) -> Mock:
    """创建模拟环境"""
    mock_env = Mock()
    mock_env.num_envs = num_envs

    # 模拟 get_observations 返回值
    obs_tensor = torch.randn(num_envs, num_prop + 10)
    extras = {
        "observations": {
            "depth_camera": torch.randn(num_envs, 58, 87),
        }
    }
    mock_env.get_observations.return_value = (obs_tensor, extras)

    # 模拟 step 返回值
    def mock_step(actions):
        rewards = torch.randn(num_envs, 1)
        dones = torch.zeros(num_envs, 1)
        infos = {
            "observations": {
                "depth_camera": torch.randn(num_envs, 58, 87),
            },
            "episode": {
                "reward": rewards.squeeze(-1),
            }
        }
        return obs_tensor, rewards, dones, infos

    mock_env.step.side_effect = mock_step

    return mock_env


def create_mock_ppo() -> Mock:
    """创建模拟 PPO"""
    mock_ppo = Mock()
    mock_ppo.act.return_value = torch.randn(4, 12)
    mock_ppo.process_env_step = Mock()
    mock_ppo.compute_returns = Mock()
    mock_ppo.update.return_value = {
        "value_loss": 0.5,
        "surrogate_loss": 0.1,
        "entropy": 8.0,
        "kl": 0.01,
        "learning_rate": 1e-4,
    }
    return mock_ppo


def create_mock_actor_critic() -> Mock:
    """创建模拟 ActorCritic"""
    mock_ac = Mock()
    mock_ac.evaluate.return_value = torch.randn(4, 1)
    return mock_ac


def run_tests():
    """手动运行测试"""
    test_class = TestEpisodeRewardLogging()
    tests = [
        ("test_collect_rollouts_returns_episode_stats",
         test_class.test_collect_rollouts_returns_episode_stats),
        ("test_run_training_iteration_returns_episode_stats",
         test_class.test_run_training_iteration_returns_episode_stats),
        ("test_log_training_metrics_logs_episode_reward",
         test_class.test_log_training_metrics_logs_episode_reward),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ PASSED: {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {name}")
            print(f"  Exception: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")
    return failed == 0


if __name__ == "__main__":
    run_tests()
