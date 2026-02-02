"""
简化的配置测试脚本

不依赖 Isaac Lab，直接验证配置文件的结构和值。
"""

import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_config_file_exists():
    """测试配置文件是否存在"""
    config_path = os.path.join(
        PROJECT_ROOT,
        "parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py"
    )
    assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
    print("✓ 配置文件存在")


def test_config_structure():
    """测试配置文件结构"""
    config_path = os.path.join(
        PROJECT_ROOT,
        "parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py"
    )

    with open(config_path, 'r') as f:
        content = f.read()

    # 检查必要的类定义
    assert "StudentFinetuneAlgorithmCfg" in content, "缺少 StudentFinetuneAlgorithmCfg 类"
    assert "UnitreeGo2StudentFinetunePPORunnerCfg" in content, "缺少 UnitreeGo2StudentFinetunePPORunnerCfg 类"

    # 检查关键配置参数
    assert "learning_rate" in content, "缺少 learning_rate 配置"
    assert "1e-4" in content, "learning_rate 应为 1e-4"
    assert "clip_param" in content, "缺少 clip_param 配置"
    assert "0.2" in content, "clip_param 应为 0.2"
    assert "gamma" in content, "缺少 gamma 配置"
    assert "0.99" in content, "gamma 应为 0.99"
    assert "lam" in content, "缺少 lam 配置"
    assert "0.95" in content, "lam 应为 0.95"
    assert "entropy_coef" in content, "缺少 entropy_coef 配置"
    assert "0.01" in content, "entropy_coef 应为 0.01"
    assert "value_loss_coef" in content, "缺少 value_loss_coef 配置"
    assert "0.5" in content, "value_loss_coef 应为 0.5"
    assert "max_grad_norm" in content, "缺少 max_grad_norm 配置"
    assert "1.0" in content, "max_grad_norm 应为 1.0"

    print("✓ 配置文件结构正确")


def test_environment_config():
    """测试环境配置"""
    config_path = os.path.join(
        PROJECT_ROOT,
        "parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py"
    )

    with open(config_path, 'r') as f:
        content = f.read()

    # 检查环境配置
    assert "num_steps_per_env" in content, "缺少 num_steps_per_env 配置"
    assert "64" in content, "num_steps_per_env 应为 64"
    assert "max_iterations" in content, "缺少 max_iterations 配置"
    assert "10000" in content, "max_iterations 应为 10000"
    assert "save_interval" in content, "缺少 save_interval 配置"
    assert "100" in content, "save_interval 应为 100"
    assert "log_interval" in content, "缺少 log_interval 配置"
    assert "10" in content, "log_interval 应为 10"

    print("✓ 环境配置正确")


def test_encoder_freezing_config():
    """测试编码器冻结配置"""
    config_path = os.path.join(
        PROJECT_ROOT,
        "parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py"
    )

    with open(config_path, 'r') as f:
        content = f.read()

    # 检查编码器冻结配置
    assert "freeze_proprio_encoder" in content, "缺少 freeze_proprio_encoder 配置"
    assert "freeze_depth_encoder" in content, "缺少 freeze_depth_encoder 配置"
    assert "freeze_fusion_transformer" in content, "缺少 freeze_fusion_transformer 配置"
    assert "freeze_temporal_transformer" in content, "缺少 freeze_temporal_transformer 配置"

    print("✓ 编码器冻结配置正确")


def test_domain_randomization_config():
    """测试 Domain Randomization 配置"""
    config_path = os.path.join(
        PROJECT_ROOT,
        "parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py"
    )

    with open(config_path, 'r') as f:
        content = f.read()

    # 检查 Domain Randomization 配置
    assert "domain_rand_enabled" in content, "缺少 domain_rand_enabled 配置"
    assert "domain_rand_curriculum" in content, "缺少 domain_rand_curriculum 配置"

    print("✓ Domain Randomization 配置正确")


def test_experiment_name():
    """测试实验名称"""
    config_path = os.path.join(
        PROJECT_ROOT,
        "parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py"
    )

    with open(config_path, 'r') as f:
        content = f.read()

    # 检查实验名称
    assert "experiment_name" in content, "缺少 experiment_name 配置"
    assert "student" in content.lower(), "实验名称应包含 'student'"
    assert "finetune" in content.lower(), "实验名称应包含 'finetune'"

    print("✓ 实验名称配置正确")


if __name__ == "__main__":
    """运行所有测试"""
    tests = [
        ("test_config_file_exists", test_config_file_exists),
        ("test_config_structure", test_config_structure),
        ("test_environment_config", test_environment_config),
        ("test_encoder_freezing_config", test_encoder_freezing_config),
        ("test_domain_randomization_config", test_domain_randomization_config),
        ("test_experiment_name", test_experiment_name),
    ]

    passed = 0
    failed = 0

    print("="*60)
    print("运行 Student Fine-tuning 配置测试")
    print("="*60)

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ 所有测试通过！")
