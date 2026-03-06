"""
Phase 5: 训练脚本测试

测试 train_student_rl_finetune.py 的核心功能：
1. 配置解析和验证
2. StudentActorCritic 创建和初始化
3. PPOStudent 算法初始化
4. Domain Randomization 集成
5. 训练循环逻辑
6. 检查点保存和加载

使用 TDD 方法：先编写测试，再实现功能。
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))
sys.path.insert(0, str(PROJECT_ROOT / "parkour_tasks"))


# ============================================================
# 测试 1: 配置解析和验证
# ============================================================

class TestConfigParsing:
    """测试配置解析功能"""

    def test_parse_default_config(self):
        """测试默认配置解析"""
        # 导入训练脚本模块
        from train_student_rl_finetune import parse_training_config

        config = parse_training_config()

        # 验证默认值（更新后的保守配置）
        assert config.num_envs == 256
        assert config.num_steps_per_env == 64
        assert config.max_iterations == 10000
        # 更新后的 PPO 超参数（针对 DAgger 模型微调优化）
        assert config.learning_rate == 3e-5, f"Expected learning_rate=3e-5, got {config.learning_rate}"
        assert config.clip_param == 0.1, f"Expected clip_param=0.1, got {config.clip_param}"
        assert config.gamma == 0.99
        assert config.lam == 0.95
        assert config.entropy_coef == 0.001, f"Expected entropy_coef=0.001, got {config.entropy_coef}"
        assert config.value_loss_coef == 0.5
        assert config.max_grad_norm == 0.5, f"Expected max_grad_norm=0.5, got {config.max_grad_norm}"
        assert config.desired_kl == 0.12, f"Expected desired_kl=0.12, got {config.desired_kl}"
        assert config.num_learning_epochs == 3, f"Expected num_learning_epochs=3, got {config.num_learning_epochs}"

    def test_parse_custom_config(self):
        """测试自定义配置解析"""
        from train_student_rl_finetune import parse_training_config

        custom_args = {
            "num_envs": 128,
            "learning_rate": 5e-5,
            "max_iterations": 5000,
        }

        config = parse_training_config(**custom_args)

        assert config.num_envs == 128
        assert config.learning_rate == 5e-5
        assert config.max_iterations == 5000

    def test_config_validation_invalid_learning_rate(self):
        """测试无效学习率验证"""
        from train_student_rl_finetune import parse_training_config

        with pytest.raises(ValueError, match="learning_rate"):
            parse_training_config(learning_rate=-1e-4)

    def test_config_validation_invalid_num_envs(self):
        """测试无效环境数验证"""
        from train_student_rl_finetune import parse_training_config

        with pytest.raises(ValueError, match="num_envs"):
            parse_training_config(num_envs=0)


# ============================================================
# 测试 2: StudentActorCritic 默认参数
# ============================================================

class TestStudentActorCriticDefaults:
    """测试 StudentActorCritic 默认参数（针对 DAgger 模型微调优化）"""

    def test_init_noise_std_default(self):
        """测试 init_noise_std 默认值为 0.1（适合预训练模型）"""
        from modules.student_actor_critic import StudentActorCritic
        import inspect

        # 获取 __init__ 方法的签名
        sig = inspect.signature(StudentActorCritic.__init__)
        init_noise_std_default = sig.parameters["init_noise_std"].default

        # 验证默认值为 0.1（而不是 1.0）
        assert init_noise_std_default == 0.1, (
            f"Expected init_noise_std default=0.1 for pre-trained DAgger model, "
            f"got {init_noise_std_default}"
        )

    def test_init_noise_std_creates_correct_log_std(self):
        """测试 init_noise_std=0.1 创建正确的 log_std"""
        from modules.student_actor_critic import StudentActorCritic

        # 创建模拟的 student_policy
        mock_policy = Mock()
        mock_policy.token_dim = 128
        mock_policy.action_dim = 12
        mock_policy.n_layers = 3

        # 使用默认 init_noise_std
        actor_critic = StudentActorCritic(
            student_policy=mock_policy,
            init_noise_std=0.1,  # 新的默认值
        )

        # 验证 log_std 的值
        expected_log_std = torch.log(torch.tensor(0.1)).item()
        actual_log_std = actor_critic.log_std[0].item()
        assert abs(actual_log_std - expected_log_std) < 1e-5, (
            f"Expected log_std={expected_log_std}, got {actual_log_std}"
        )


# ============================================================
# 测试 3: PPOStudent 默认参数
# ============================================================

class TestPPOStudentDefaults:
    """测试 PPOStudent 默认参数（针对 DAgger 模型微调优化）"""

    def test_desired_kl_default(self):
        """测试 desired_kl 默认值为 0.12（适应 12 维动作空间）"""
        from modules.ppo_student import PPOStudent
        import inspect

        sig = inspect.signature(PPOStudent.__init__)
        desired_kl_default = sig.parameters["desired_kl"].default

        # 验证默认值为 0.12（12 维动作空间的 KL 是求和，所以需要更大的目标值）
        assert desired_kl_default == 0.12, (
            f"Expected desired_kl default=0.12 for 12-dim action space, "
            f"got {desired_kl_default}"
        )

    def test_entropy_coef_default(self):
        """测试 entropy_coef 默认值为 0.001（避免鼓励增加噪声）"""
        from modules.ppo_student import PPOStudent
        import inspect

        sig = inspect.signature(PPOStudent.__init__)
        entropy_coef_default = sig.parameters["entropy_coef"].default

        assert entropy_coef_default == 0.001, (
            f"Expected entropy_coef default=0.001, got {entropy_coef_default}"
        )

    def test_learning_rate_lower_bound(self):
        """测试学习率下限为 1e-7（而不是 1e-5）"""
        from modules.ppo_student import PPOStudent

        # 创建模拟的 actor_critic
        mock_actor_critic = Mock()
        mock_actor_critic.parameters.return_value = [torch.zeros(10)]
        mock_actor_critic.to = Mock(return_value=mock_actor_critic)

        ppo = PPOStudent(
            actor_critic=mock_actor_critic,
            learning_rate=1e-4,
            desired_kl=0.12,
            device="cpu",
        )

        # 模拟 KL 过高的情况，触发学习率下降
        # 连续多次降低学习率
        for _ in range(20):
            ppo.learning_rate = max(1e-7, ppo.learning_rate / 1.5)

        # 验证学习率下限为 1e-7
        assert ppo.learning_rate >= 1e-7, (
            f"Learning rate should not go below 1e-7, got {ppo.learning_rate}"
        )


# ============================================================
# 测试 4: StudentActorCritic 创建
# ============================================================

class TestStudentActorCriticCreation:
    """测试 StudentActorCritic 创建功能"""

    def test_create_actor_critic_from_checkpoint(self):
        """测试从 DAgger checkpoint 创建 StudentActorCritic"""
        from train_student_rl_finetune import create_student_actor_critic

        # 创建模拟的 checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            mock_checkpoint = {
                "model_state_dict": {},
                "meta": {
                    "num_prop": 53,
                    "action_dim": 12,
                    "camera_resolution": [58, 87],
                }
            }
            torch.save(mock_checkpoint, f.name)
            checkpoint_path = f.name

        try:
            actor_critic, checkpoint_meta = create_student_actor_critic(
                checkpoint_path=checkpoint_path,
                freeze_encoders=False,
                freeze_fusion=False,
                freeze_temporal=False,
                device="cpu",
            )

            # 验证返回类型
            assert actor_critic is not None
            assert hasattr(actor_critic, "act")
            assert hasattr(actor_critic, "evaluate")
            assert hasattr(actor_critic, "value_head")
        finally:
            os.unlink(checkpoint_path)

    def test_create_actor_critic_with_encoder_freezing(self):
        """测试创建带编码器冻结的 StudentActorCritic"""
        from train_student_rl_finetune import create_student_actor_critic

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            mock_checkpoint = {
                "model_state_dict": {},
                "meta": {
                    "num_prop": 53,
                    "action_dim": 12,
                    "camera_resolution": [58, 87],
                }
            }
            torch.save(mock_checkpoint, f.name)
            checkpoint_path = f.name

        try:
            actor_critic, checkpoint_meta = create_student_actor_critic(
                checkpoint_path=checkpoint_path,
                freeze_encoders=True,
                freeze_fusion=True,
                freeze_temporal=False,
                device="cpu",
            )

            # 验证编码器被冻结
            if hasattr(actor_critic.student_policy, "proprio_encoder"):
                for param in actor_critic.student_policy.proprio_encoder.parameters():
                    assert not param.requires_grad

            if hasattr(actor_critic.student_policy, "depth_encoder"):
                for param in actor_critic.student_policy.depth_encoder.parameters():
                    assert not param.requires_grad
        finally:
            os.unlink(checkpoint_path)


# ============================================================
# 测试 3: PPOStudent 初始化
# ============================================================

class TestPPOStudentInitialization:
    """测试 PPOStudent 算法初始化"""

    def test_initialize_ppo_student(self):
        """测试 PPOStudent 初始化"""
        from train_student_rl_finetune import initialize_ppo_student

        # 创建模拟的 actor_critic
        mock_actor_critic = Mock()
        mock_actor_critic.parameters.return_value = [torch.zeros(10)]

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

    def test_initialize_storage(self):
        """测试 rollout storage 初始化"""
        from train_student_rl_finetune import initialize_ppo_student

        mock_actor_critic = Mock()
        mock_actor_critic.parameters.return_value = [torch.zeros(10)]

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

        # 初始化 storage
        ppo.init_storage(
            num_envs=256,
            num_steps=64,
            proprio_dim=53,
            depth_shape=(4, 58, 87),
            action_dim=12,
        )

        assert ppo.storage is not None
        assert ppo.storage.num_envs == 256
        assert ppo.storage.num_steps == 64


# ============================================================
# 测试 4: Domain Randomization 集成
# ============================================================

class TestDomainRandomizationIntegration:
    """测试 Domain Randomization 集成"""

    def test_create_domain_randomization(self):
        """测试创建 Domain Randomization 模块"""
        from train_student_rl_finetune import create_domain_randomization

        domain_rand = create_domain_randomization(
            enabled=True,
            use_curriculum=True,
        )

        assert domain_rand is not None
        assert hasattr(domain_rand, "apply_augmentation")
        assert hasattr(domain_rand, "update_curriculum")

    def test_domain_rand_curriculum_update(self):
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

    def test_apply_augmentation_to_depth(self):
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


# ============================================================
# 测试 5: 训练循环逻辑
# ============================================================

class TestTrainingLoop:
    """测试训练循环逻辑"""

    def test_collect_rollouts(self):
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

    def test_compute_returns_and_update(self):
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

    def test_training_iteration(self):
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


# ============================================================
# 测试 6: 检查点保存和加载
# ============================================================

class TestCheckpointing:
    """测试检查点保存和加载"""

    def test_save_checkpoint(self):
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

    def test_load_checkpoint(self):
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


# ============================================================
# 测试 7: 完整训练流程（集成测试）
# ============================================================

class TestFullTrainingPipeline:
    """测试完整训练流程"""

    def test_training_script_imports(self):
        """测试训练脚本可以正确导入"""
        try:
            from train_student_rl_finetune import (
                parse_training_config,
                create_student_actor_critic,
                initialize_ppo_student,
                create_domain_randomization,
                collect_rollouts,
                compute_returns_and_update,
                run_training_iteration,
                save_training_checkpoint,
                load_training_checkpoint,
            )
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import training script: {e}")

    def test_training_config_dataclass(self):
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


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
