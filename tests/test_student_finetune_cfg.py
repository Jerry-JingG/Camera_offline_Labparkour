"""
测试 Student RL Fine-tuning 配置文件

测试配置加载、参数验证和默认值设置
"""

import pytest
import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)


class TestStudentFinetuneCfg:
    """测试 Student Fine-tuning 配置类"""

    def test_config_import(self):
        """测试配置文件可以正确导入"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        assert UnitreeGo2StudentFinetunePPORunnerCfg is not None

    def test_ppo_hyperparameters(self):
        """测试 PPO 超参数配置"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证 PPO 超参数
        assert cfg.algorithm.learning_rate == 1e-4, "学习率应为 1e-4（低于 teacher 的 2e-4）"
        assert cfg.algorithm.clip_param == 0.2, "Clip param 应为 0.2"
        assert cfg.algorithm.gamma == 0.99, "Gamma 应为 0.99"
        assert cfg.algorithm.lam == 0.95, "Lambda 应为 0.95"
        assert cfg.algorithm.entropy_coef == 0.01, "Entropy coef 应为 0.01"
        assert cfg.algorithm.value_loss_coef == 0.5, "Value loss coef 应为 0.5"
        assert cfg.algorithm.max_grad_norm == 1.0, "Max grad norm 应为 1.0"

    def test_environment_config(self):
        """测试环境配置"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证环境配置
        assert cfg.num_steps_per_env == 64, "每环境步数应为 64（匹配 TXL sequence length）"
        assert cfg.max_iterations == 10000, "最大迭代应为 10000"
        assert cfg.save_interval == 100, "保存间隔应为 100"
        assert cfg.log_interval == 10, "日志间隔应为 10"

    def test_training_config(self):
        """测试训练配置"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证训练配置
        assert cfg.algorithm.num_learning_epochs == 5, "学习 epoch 数应为 5"
        assert cfg.algorithm.num_mini_batches == 4, "Mini-batch 数应为 4"
        assert cfg.algorithm.schedule == "adaptive", "学习率调度应为 adaptive"
        assert cfg.algorithm.desired_kl == 0.01, "期望 KL 散度应为 0.01"

    def test_encoder_freezing_config(self):
        """测试编码器冻结策略配置"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证编码器冻结配置（默认不冻结）
        assert hasattr(cfg, "freeze_proprio_encoder"), "应有 freeze_proprio_encoder 配置"
        assert hasattr(cfg, "freeze_depth_encoder"), "应有 freeze_depth_encoder 配置"
        assert hasattr(cfg, "freeze_fusion_transformer"), "应有 freeze_fusion_transformer 配置"
        assert hasattr(cfg, "freeze_temporal_transformer"), "应有 freeze_temporal_transformer 配置"

        assert cfg.freeze_proprio_encoder == False, "默认不冻结 proprio encoder"
        assert cfg.freeze_depth_encoder == False, "默认不冻结 depth encoder"
        assert cfg.freeze_fusion_transformer == False, "默认不冻结 fusion transformer"
        assert cfg.freeze_temporal_transformer == False, "默认不冻结 temporal transformer"

    def test_domain_randomization_config(self):
        """测试 Domain Randomization 配置"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证 Domain Randomization 配置
        assert hasattr(cfg, "domain_rand_enabled"), "应有 domain_rand_enabled 配置"
        assert hasattr(cfg, "domain_rand_curriculum"), "应有 domain_rand_curriculum 配置"

        assert cfg.domain_rand_enabled == True, "默认启用 domain randomization"
        assert cfg.domain_rand_curriculum == True, "默认启用 curriculum"

    def test_experiment_name(self):
        """测试实验名称配置"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证实验名称
        assert hasattr(cfg, "experiment_name"), "应有 experiment_name 配置"
        assert "student" in cfg.experiment_name.lower(), "实验名称应包含 'student'"
        assert "finetune" in cfg.experiment_name.lower(), "实验名称应包含 'finetune'"

    def test_parameter_ranges(self):
        """测试参数范围验证"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证参数在合理范围内
        assert 0 < cfg.algorithm.learning_rate < 1e-3, "学习率应在合理范围内"
        assert 0 < cfg.algorithm.clip_param < 1.0, "Clip param 应在 (0, 1) 范围内"
        assert 0 < cfg.algorithm.gamma <= 1.0, "Gamma 应在 (0, 1] 范围内"
        assert 0 < cfg.algorithm.lam <= 1.0, "Lambda 应在 (0, 1] 范围内"
        assert cfg.algorithm.entropy_coef >= 0, "Entropy coef 应非负"
        assert cfg.algorithm.value_loss_coef > 0, "Value loss coef 应为正"
        assert cfg.algorithm.max_grad_norm > 0, "Max grad norm 应为正"

    def test_config_consistency(self):
        """测试配置一致性"""
        from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
            UnitreeGo2StudentFinetunePPORunnerCfg
        )

        cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

        # 验证配置一致性
        assert cfg.num_steps_per_env > 0, "步数应为正"
        assert cfg.max_iterations > 0, "最大迭代应为正"
        assert cfg.save_interval > 0, "保存间隔应为正"
        assert cfg.save_interval <= cfg.max_iterations, "保存间隔不应超过最大迭代"


if __name__ == "__main__":
    """直接运行测试"""
    test_suite = TestStudentFinetuneCfg()

    tests = [
        ("test_config_import", test_suite.test_config_import),
        ("test_ppo_hyperparameters", test_suite.test_ppo_hyperparameters),
        ("test_environment_config", test_suite.test_environment_config),
        ("test_training_config", test_suite.test_training_config),
        ("test_encoder_freezing_config", test_suite.test_encoder_freezing_config),
        ("test_domain_randomization_config", test_suite.test_domain_randomization_config),
        ("test_experiment_name", test_suite.test_experiment_name),
        ("test_parameter_ranges", test_suite.test_parameter_ranges),
        ("test_config_consistency", test_suite.test_config_consistency),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
