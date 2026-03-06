"""
Phase 5: 训练脚本简化测试

测试 train_student_rl_finetune.py 的核心功能（不依赖 Isaac Lab）：
1. 配置解析和验证
2. Domain Randomization 集成
3. 检查点保存和加载
4. 训练循环逻辑（使用 Mock）

使用 TDD 方法：先编写测试，再实现功能。
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

import torch
import torch.nn as nn

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))


def test_parse_default_config():
    """测试默认配置解析"""
    from train_student_rl_finetune import parse_training_config

    config = parse_training_config()

    assert config.num_envs == 256
    assert config.num_steps_per_env == 64
    assert config.max_iterations == 10000
    assert config.learning_rate == 1e-4
    assert config.clip_param == 0.2
    assert config.gamma == 0.99
    assert config.lam == 0.95
    assert config.entropy_coef == 0.01
    assert config.value_loss_coef == 0.5
    assert config.max_grad_norm == 1.0
    print("✓ test_parse_default_config passed")


def test_parse_custom_config():
    """测试自定义配置解析"""
    from train_student_rl_finetune import parse_training_config

    config = parse_training_config(
        num_envs=128,
        learning_rate=5e-5,
        max_iterations=5000,
    )

    assert config.num_envs == 128
    assert config.learning_rate == 5e-5
    assert config.max_iterations == 5000
    print("✓ test_parse_custom_config passed")


def test_config_validation_invalid_learning_rate():
    """测试无效学习率验证"""
    from train_student_rl_finetune import parse_training_config

    try:
        parse_training_config(learning_rate=-1e-4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "learning_rate" in str(e)
    print("✓ test_config_validation_invalid_learning_rate passed")


def test_config_validation_invalid_num_envs():
    """测试无效环境数验证"""
    from train_student_rl_finetune import parse_training_config

    try:
        parse_training_config(num_envs=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_envs" in str(e)
    print("✓ test_config_validation_invalid_num_envs passed")


def test_create_domain_randomization():
    """测试创建 Domain Randomization 模块"""
    from train_student_rl_finetune import create_domain_randomization

    domain_rand = create_domain_randomization(
        enabled=True,
        use_curriculum=True,
    )

    assert domain_rand is not None
    assert hasattr(domain_rand, "apply_augmentation")
    assert hasattr(domain_rand, "update_curriculum")
    print("✓ test_create_domain_randomization passed")


def test_domain_rand_curriculum_update():
    """测试 Domain Randomization 课程学习更新"""
    from train_student_rl_finetune import create_domain_randomization

    domain_rand = create_domain_randomization(
        enabled=True,
        use_curriculum=True,
    )

    # 初始参数
    params_0 = domain_rand.get_current_params()

    # 更新到迭代 2000
    domain_rand.update_curriculum(iteration=2000)
    params_2000 = domain_rand.get_current_params()

    # 验证参数随迭代增加
    assert params_2000["noise_std"] >= params_0["noise_std"]
    print("✓ test_domain_rand_curriculum_update passed")


def test_apply_augmentation_to_depth():
    """测试对深度图像应用增强"""
    from train_student_rl_finetune import create_domain_randomization

    domain_rand = create_domain_randomization(
        enabled=True,
        use_curriculum=False,
    )

    # 创建测试深度图像
    depth = torch.randn(256, 4, 58, 87)

    # 应用增强
    augmented_depth = domain_rand.apply_augmentation(depth)

    # 验证形状不变
    assert augmented_depth.shape == depth.shape
    # 验证值有变化（增强生效）
    assert not torch.allclose(augmented_depth, depth)
    print("✓ test_apply_augmentation_to_depth passed")


def test_save_checkpoint():
    """测试保存检查点"""
    from train_student_rl_finetune import save_training_checkpoint

    mock_actor_critic = Mock()
    mock_actor_critic.state_dict.return_value = {"layer": torch.zeros(10)}

    mock_optimizer = Mock()
    mock_optimizer.state_dict.return_value = {"param_groups": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint_100.pt"

        save_training_checkpoint(
            path=checkpoint_path,
            actor_critic=mock_actor_critic,
            optimizer=mock_optimizer,
            iteration=100,
            config={"learning_rate": 1e-4},
        )

        assert checkpoint_path.exists()

        # 验证内容
        checkpoint = torch.load(checkpoint_path)
        assert "actor_critic_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["iteration"] == 100
    print("✓ test_save_checkpoint passed")


def test_load_checkpoint():
    """测试加载检查点"""
    from train_student_rl_finetune import load_training_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint_100.pt"

        # 创建测试检查点
        checkpoint = {
            "actor_critic_state_dict": {"layer": torch.zeros(10)},
            "optimizer_state_dict": {"param_groups": []},
            "iteration": 100,
            "config": {"learning_rate": 1e-4},
        }
        torch.save(checkpoint, checkpoint_path)

        mock_actor_critic = Mock()
        mock_optimizer = Mock()

        iteration = load_training_checkpoint(
            path=checkpoint_path,
            actor_critic=mock_actor_critic,
            optimizer=mock_optimizer,
        )

        assert iteration == 100
        mock_actor_critic.load_state_dict.assert_called_once()
        mock_optimizer.load_state_dict.assert_called_once()
    print("✓ test_load_checkpoint passed")


def test_training_config_dataclass():
    """测试训练配置数据类"""
    from train_student_rl_finetune import TrainingConfig

    config = TrainingConfig(
        num_envs=256,
        num_steps_per_env=64,
        max_iterations=10000,
        learning_rate=1e-4,
    )

    assert config.num_envs == 256
    assert config.num_steps_per_env == 64
    print("✓ test_training_config_dataclass passed")


def test_initialize_ppo_student():
    """测试 PPOStudent 初始化"""
    from train_student_rl_finetune import initialize_ppo_student

    # 创建模拟的 actor_critic
    mock_actor_critic = Mock()
    mock_actor_critic.parameters.return_value = [torch.zeros(10, requires_grad=True)]

    ppo = initialize_ppo_student(
        actor_critic=mock_actor_critic,
        learning_rate=1e-4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        device="cpu",
    )

    assert ppo is not None
    assert ppo.learning_rate == 1e-4
    assert ppo.clip_param == 0.2
    assert ppo.gamma == 0.99
    print("✓ test_initialize_ppo_student passed")


def test_collect_rollouts():
    """测试 rollout 收集"""
    from train_student_rl_finetune import collect_rollouts

    # 创建模拟对象
    mock_env = Mock()
    mock_env.num_envs = 256
    mock_env.get_observations.return_value = {
        "proprio": torch.randn(256, 53),
        "depth": torch.randn(256, 4, 58, 87),
    }
    mock_env.step.return_value = (
        {"proprio": torch.randn(256, 53), "depth": torch.randn(256, 4, 58, 87)},
        torch.randn(256, 1),  # rewards
        torch.zeros(256, 1),  # dones
        {},  # infos
    )

    mock_ppo = Mock()
    mock_ppo.act.return_value = torch.randn(256, 12)
    mock_ppo.storage = Mock()

    mock_domain_rand = Mock()
    mock_domain_rand.apply_augmentation.side_effect = lambda x: x

    # 收集 rollouts
    obs = collect_rollouts(
        env=mock_env,
        ppo=mock_ppo,
        domain_rand=mock_domain_rand,
        num_steps=64,
    )

    # 验证调用次数
    assert mock_ppo.act.call_count == 64
    assert mock_env.step.call_count == 64
    print("✓ test_collect_rollouts passed")


def test_compute_returns_and_update():
    """测试计算 returns 和 PPO 更新"""
    from train_student_rl_finetune import compute_returns_and_update

    mock_ppo = Mock()
    mock_ppo.compute_returns.return_value = None
    mock_ppo.update.return_value = {
        "value_loss": 0.5,
        "surrogate_loss": 0.1,
        "entropy": 0.05,
        "kl": 0.01,
        "learning_rate": 1e-4,
    }

    mock_actor_critic = Mock()
    mock_actor_critic.evaluate.return_value = torch.randn(256, 1)

    last_obs = {
        "proprio": torch.randn(256, 53),
        "depth": torch.randn(256, 4, 58, 87),
    }

    train_info = compute_returns_and_update(
        ppo=mock_ppo,
        actor_critic=mock_actor_critic,
        last_obs=last_obs,
    )

    assert "value_loss" in train_info
    assert "surrogate_loss" in train_info
    mock_ppo.compute_returns.assert_called_once()
    mock_ppo.update.assert_called_once()
    print("✓ test_compute_returns_and_update passed")


def test_run_training_iteration():
    """测试单次训练迭代"""
    from train_student_rl_finetune import run_training_iteration

    mock_env = Mock()
    mock_env.num_envs = 256
    mock_env.get_observations.return_value = {
        "proprio": torch.randn(256, 53),
        "depth": torch.randn(256, 4, 58, 87),
    }
    mock_env.step.return_value = (
        {"proprio": torch.randn(256, 53), "depth": torch.randn(256, 4, 58, 87)},
        torch.randn(256, 1),
        torch.zeros(256, 1),
        {"episode_return": torch.zeros(256)},
    )

    mock_ppo = Mock()
    mock_ppo.act.return_value = torch.randn(256, 12)
    mock_ppo.storage = Mock()
    mock_ppo.compute_returns.return_value = None
    mock_ppo.update.return_value = {
        "value_loss": 0.5,
        "surrogate_loss": 0.1,
        "entropy": 0.05,
        "kl": 0.01,
        "learning_rate": 1e-4,
    }

    mock_actor_critic = Mock()
    mock_actor_critic.evaluate.return_value = torch.randn(256, 1)

    mock_domain_rand = Mock()
    mock_domain_rand.apply_augmentation.side_effect = lambda x: x

    train_info = run_training_iteration(
        env=mock_env,
        ppo=mock_ppo,
        actor_critic=mock_actor_critic,
        domain_rand=mock_domain_rand,
        num_steps=64,
        iteration=0,
    )

    assert train_info is not None
    assert "value_loss" in train_info
    print("✓ test_run_training_iteration passed")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 Student RL Fine-tuning 训练脚本测试")
    print("=" * 60)

    tests = [
        test_parse_default_config,
        test_parse_custom_config,
        test_config_validation_invalid_learning_rate,
        test_config_validation_invalid_num_envs,
        test_create_domain_randomization,
        test_domain_rand_curriculum_update,
        test_apply_augmentation_to_depth,
        test_save_checkpoint,
        test_load_checkpoint,
        test_training_config_dataclass,
        test_initialize_ppo_student,
        test_collect_rollouts,
        test_compute_returns_and_update,
        test_run_training_iteration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"测试结果: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ 所有测试通过！")
        return True
    else:
        print(f"\n✗ {failed} 个测试失败")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
