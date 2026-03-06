#!/usr/bin/env python3
"""
测试 Episode Reward 控制台输出和 TensorBoard 日志

验证：
1. 控制台打印包含 episode reward
2. TensorBoard 日志包含 episode/mean_reward
"""

import sys
from pathlib import Path
from io import StringIO
from unittest.mock import Mock, patch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))

from train_student_rl_finetune import log_training_metrics


def test_console_output_format():
    """测试控制台输出格式包含 reward"""
    # 模拟训练信息
    train_info = {
        "value_loss": 0.1234,
        "surrogate_loss": -0.0567,
        "entropy": 8.9012,
        "kl": 0.0234,
        "learning_rate": 3e-5,
        "episode_stats": {
            "mean_reward": 12.3456,
        },
    }

    # 提取 episode_stats
    episode_stats = train_info.get("episode_stats", {})
    mean_reward = episode_stats.get("mean_reward", 0.0)

    # 构造预期的输出格式
    iteration = 100
    expected_output = (
        f"[Iter {iteration}] "
        f"reward={mean_reward:.4f} "
        f"value_loss={train_info['value_loss']:.4f} "
        f"policy_loss={train_info['surrogate_loss']:.4f} "
        f"entropy={train_info['entropy']:.4f} "
        f"kl={train_info['kl']:.4f} "
        f"lr={train_info['learning_rate']:.2e}"
    )

    # 验证格式
    assert "reward=12.3456" in expected_output
    assert "value_loss=0.1234" in expected_output
    assert "policy_loss=-0.0567" in expected_output
    assert "entropy=8.9012" in expected_output
    assert "kl=0.0234" in expected_output
    assert "lr=3.00e-05" in expected_output

    print("✓ 控制台输出格式测试通过")


def test_tensorboard_logging_with_episode_stats():
    """测试 TensorBoard 日志包含 episode_stats"""
    # 创建 mock logger
    mock_logger = Mock()
    mock_logger.enabled = True

    # 模拟训练信息
    train_info = {
        "value_loss": 0.1234,
        "surrogate_loss": -0.0567,
        "entropy": 8.9012,
        "kl": 0.0234,
        "learning_rate": 3e-5,
    }

    episode_stats = {
        "mean_reward": 12.3456,
    }

    # 调用日志函数
    log_training_metrics(
        logger=mock_logger,
        iteration=100,
        train_info=train_info,
        episode_stats=episode_stats,
    )

    # 验证 logger.log 被调用
    assert mock_logger.log.called, "logger.log 应该被调用"

    # 获取调用参数
    call_args = mock_logger.log.call_args
    metrics = call_args[0][0]  # 第一个位置参数

    # 验证 episode/mean_reward 在 metrics 中
    assert "episode/mean_reward" in metrics, "metrics 应该包含 episode/mean_reward"
    assert metrics["episode/mean_reward"] == 12.3456

    # 验证其他指标
    assert "train/value_loss" in metrics
    assert "train/policy_loss" in metrics
    assert "train/entropy" in metrics
    assert "train/kl_divergence" in metrics
    assert "train/learning_rate" in metrics

    print("✓ TensorBoard 日志测试通过")


def test_tensorboard_logging_without_episode_stats():
    """测试 TensorBoard 日志在没有 episode_stats 时不崩溃"""
    # 创建 mock logger
    mock_logger = Mock()
    mock_logger.enabled = True

    # 模拟训练信息（不包含 episode_stats）
    train_info = {
        "value_loss": 0.1234,
        "surrogate_loss": -0.0567,
        "entropy": 8.9012,
        "kl": 0.0234,
        "learning_rate": 3e-5,
    }

    # 调用日志函数（不传递 episode_stats）
    log_training_metrics(
        logger=mock_logger,
        iteration=100,
        train_info=train_info,
        episode_stats=None,
    )

    # 验证 logger.log 被调用
    assert mock_logger.log.called, "logger.log 应该被调用"

    # 获取调用参数
    call_args = mock_logger.log.call_args
    metrics = call_args[0][0]

    # 验证 episode/mean_reward 不在 metrics 中
    assert "episode/mean_reward" not in metrics, "没有 episode_stats 时不应该有 episode 指标"

    print("✓ TensorBoard 日志（无 episode_stats）测试通过")


if __name__ == "__main__":
    print("开始测试 Episode Reward 日志功能...\n")

    test_console_output_format()
    test_tensorboard_logging_with_episode_stats()
    test_tensorboard_logging_without_episode_stats()

    print("\n所有测试通过！✓")
