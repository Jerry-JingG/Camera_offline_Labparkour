"""
Phase 6: 鲁棒性评估脚本测试

测试 evaluate_student_robustness.py 的核心功能（不依赖 Isaac Lab）：
1. 评估配置数据类
2. 测试场景定义
3. 鲁棒性指标计算
4. 增强应用
5. 评估循环
6. 报告生成

使用 TDD 方法：先编写测试，再实现功能。
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))


# ============================================================
# 测试 1: 评估配置数据类
# ============================================================

def test_evaluation_config_default():
    """测试默认评估配置"""
    from evaluate_student_robustness import EvaluationConfig

    config = EvaluationConfig()

    # 验证默认值
    assert config.num_episodes == 100
    assert config.num_envs == 256
    assert config.device == "cuda"
    assert config.checkpoint_path == ""
    print("✓ test_evaluation_config_default passed")


def test_evaluation_config_custom():
    """测试自定义评估配置"""
    from evaluate_student_robustness import EvaluationConfig

    config = EvaluationConfig(
        num_episodes=50,
        num_envs=128,
        device="cpu",
        checkpoint_path="/path/to/checkpoint.pt",
    )

    assert config.num_episodes == 50
    assert config.num_envs == 128
    assert config.device == "cpu"
    assert config.checkpoint_path == "/path/to/checkpoint.pt"
    print("✓ test_evaluation_config_custom passed")


def test_evaluation_config_validation():
    """测试评估配置验证"""
    from evaluate_student_robustness import EvaluationConfig

    # 无效的 num_episodes
    try:
        config = EvaluationConfig(num_episodes=0)
        config.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_episodes" in str(e)

    # 无效的 num_envs
    try:
        config = EvaluationConfig(num_envs=-1)
        config.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_envs" in str(e)

    print("✓ test_evaluation_config_validation passed")


# ============================================================
# 测试 2: 测试场景枚举
# ============================================================

def test_test_scenario_enum():
    """测试场景枚举定义"""
    from evaluate_student_robustness import TestScenario

    # 验证所有场景存在
    assert hasattr(TestScenario, "CLEAN")
    assert hasattr(TestScenario, "HIGH_NOISE")
    assert hasattr(TestScenario, "HIGH_LATENCY")
    assert hasattr(TestScenario, "CAMERA_DROPOUT")
    assert hasattr(TestScenario, "COMBINED_STRESS")

    print("✓ test_test_scenario_enum passed")


def test_get_scenario_params():
    """测试获取场景参数"""
    from evaluate_student_robustness import TestScenario, get_scenario_params

    # Clean 场景 - 无增强
    clean_params = get_scenario_params(TestScenario.CLEAN)
    assert clean_params["gaussian_std"] == 0.0
    assert clean_params["salt_pepper_prob"] == 0.0
    assert clean_params["camera_dropout_prob"] == 0.0
    assert clean_params["depth_delay_frames"] == 0

    # High Noise 场景
    noise_params = get_scenario_params(TestScenario.HIGH_NOISE)
    assert noise_params["gaussian_std"] == 0.08
    assert noise_params["salt_pepper_prob"] == 0.04

    # High Latency 场景
    latency_params = get_scenario_params(TestScenario.HIGH_LATENCY)
    assert latency_params["depth_delay_frames"] == 5
    assert latency_params["action_delay_frames"] == 8

    # Camera Dropout 场景
    dropout_params = get_scenario_params(TestScenario.CAMERA_DROPOUT)
    assert dropout_params["camera_dropout_prob"] == 0.7

    # Combined Stress 场景 - 所有增强最大强度
    stress_params = get_scenario_params(TestScenario.COMBINED_STRESS)
    assert stress_params["gaussian_std"] >= 0.08
    assert stress_params["camera_dropout_prob"] >= 0.7
    assert stress_params["depth_delay_frames"] >= 5

    print("✓ test_get_scenario_params passed")


# ============================================================
# 测试 3: 鲁棒性指标数据类
# ============================================================

def test_robustness_metrics_dataclass():
    """测试鲁棒性指标数据类"""
    from evaluate_student_robustness import RobustnessMetrics

    metrics = RobustnessMetrics(
        scenario_name="HIGH_NOISE",
        episode_returns=[100.0, 95.0, 105.0, 90.0, 110.0],
        episode_lengths=[500, 480, 520, 460, 540],
        goal_progress=[0.8, 0.75, 0.85, 0.7, 0.9],
        timeout_rate=0.1,
    )

    assert metrics.scenario_name == "HIGH_NOISE"
    assert len(metrics.episode_returns) == 5
    assert len(metrics.episode_lengths) == 5
    assert len(metrics.goal_progress) == 5
    assert metrics.timeout_rate == 0.1

    print("✓ test_robustness_metrics_dataclass passed")


def test_robustness_metrics_statistics():
    """测试鲁棒性指标统计计算"""
    from evaluate_student_robustness import RobustnessMetrics

    metrics = RobustnessMetrics(
        scenario_name="TEST",
        episode_returns=[100.0, 100.0, 100.0, 100.0, 100.0],
        episode_lengths=[500, 500, 500, 500, 500],
        goal_progress=[0.8, 0.8, 0.8, 0.8, 0.8],
        timeout_rate=0.0,
    )

    stats = metrics.compute_statistics()

    assert stats["return_mean"] == 100.0
    assert stats["return_std"] == 0.0
    assert stats["length_mean"] == 500.0
    assert stats["goal_progress_mean"] == 0.8

    print("✓ test_robustness_metrics_statistics passed")


def test_robustness_metrics_to_dict():
    """测试鲁棒性指标转换为字典"""
    from evaluate_student_robustness import RobustnessMetrics

    metrics = RobustnessMetrics(
        scenario_name="CLEAN",
        episode_returns=[100.0],
        episode_lengths=[500],
        goal_progress=[0.8],
        timeout_rate=0.0,
    )

    result = metrics.to_dict()

    assert "scenario_name" in result
    assert "statistics" in result
    assert "raw_data" in result
    assert result["scenario_name"] == "CLEAN"

    print("✓ test_robustness_metrics_to_dict passed")


# ============================================================
# 测试 4: 增强应用函数
# ============================================================

def test_apply_gaussian_noise():
    """测试高斯噪声应用"""
    from evaluate_student_robustness import apply_gaussian_noise

    depth = torch.zeros(256, 4, 58, 87)
    noisy_depth = apply_gaussian_noise(depth, std=0.08)

    # 验证形状不变
    assert noisy_depth.shape == depth.shape
    # 验证有噪声添加
    assert not torch.allclose(noisy_depth, depth)
    # 验证噪声在合理范围内
    assert noisy_depth.std() < 0.2  # 噪声不应过大

    print("✓ test_apply_gaussian_noise passed")


def test_apply_salt_pepper_noise():
    """测试椒盐噪声应用"""
    from evaluate_student_robustness import apply_salt_pepper_noise

    depth = torch.ones(256, 4, 58, 87) * 0.5
    # 使用较高的概率确保有像素被修改
    noisy_depth = apply_salt_pepper_noise(depth, prob=0.1)

    # 验证形状不变
    assert noisy_depth.shape == depth.shape
    # 验证有像素被修改（使用更宽松的检查）
    num_modified = (noisy_depth != depth).sum().item()
    assert num_modified > 0, "Salt-pepper noise should modify some pixels"

    print("✓ test_apply_salt_pepper_noise passed")


def test_apply_camera_dropout():
    """测试相机丢失应用"""
    from evaluate_student_robustness import apply_camera_dropout

    depth = torch.randn(256, 4, 58, 87)
    dropout_depth = apply_camera_dropout(depth, prob=0.7)

    # 验证形状不变
    assert dropout_depth.shape == depth.shape

    print("✓ test_apply_camera_dropout passed")


def test_apply_depth_latency():
    """测试深度延迟应用"""
    from evaluate_student_robustness import DepthLatencyBuffer

    buffer = DepthLatencyBuffer(delay_frames=5)

    # 模拟多帧输入
    frames = [torch.randn(256, 4, 58, 87) * i for i in range(10)]

    outputs = []
    for frame in frames:
        output = buffer.apply(frame)
        outputs.append(output)

    # 前 5 帧应该返回第一帧（延迟效果）
    assert torch.allclose(outputs[0], outputs[1])
    assert torch.allclose(outputs[0], outputs[4])

    # 第 6 帧开始应该返回延迟的帧
    # outputs[5] 应该是 frames[0]
    assert torch.allclose(outputs[5], frames[0])

    print("✓ test_apply_depth_latency passed")


def test_apply_test_scenario():
    """测试应用完整测试场景"""
    from evaluate_student_robustness import TestScenario, apply_test_scenario

    depth = torch.randn(256, 4, 58, 87)

    # Clean 场景 - 不应修改
    clean_depth = apply_test_scenario(depth.clone(), TestScenario.CLEAN)
    assert torch.allclose(clean_depth, depth)

    # High Noise 场景 - 应该修改
    noisy_depth = apply_test_scenario(depth.clone(), TestScenario.HIGH_NOISE)
    assert not torch.allclose(noisy_depth, depth)

    print("✓ test_apply_test_scenario passed")


# ============================================================
# 测试 5: 评估函数
# ============================================================

def test_evaluate_single_episode():
    """测试单个 episode 评估"""
    from evaluate_student_robustness import evaluate_single_episode, TestScenario

    # 创建模拟对象
    mock_env = Mock()
    mock_env.reset.return_value = {
        "proprio": torch.randn(1, 53),
        "depth": torch.randn(1, 4, 58, 87),
    }
    mock_env.step.return_value = (
        {"proprio": torch.randn(1, 53), "depth": torch.randn(1, 4, 58, 87)},
        torch.tensor([1.0]),  # reward
        torch.tensor([True]),  # done
        {"goal_progress": torch.tensor([0.8]), "timeout": torch.tensor([False])},
    )

    mock_policy = Mock()
    mock_policy.act.return_value = torch.randn(1, 12)

    result = evaluate_single_episode(
        env=mock_env,
        policy=mock_policy,
        scenario=TestScenario.CLEAN,
        max_steps=1000,
    )

    assert "episode_return" in result
    assert "episode_length" in result
    assert "goal_progress" in result
    assert "timeout" in result

    print("✓ test_evaluate_single_episode passed")


def test_evaluate_robustness():
    """测试完整鲁棒性评估"""
    from evaluate_student_robustness import evaluate_robustness, TestScenario

    # 创建模拟对象
    mock_env = Mock()
    mock_env.reset.return_value = {
        "proprio": torch.randn(1, 53),
        "depth": torch.randn(1, 4, 58, 87),
    }
    mock_env.step.return_value = (
        {"proprio": torch.randn(1, 53), "depth": torch.randn(1, 4, 58, 87)},
        torch.tensor([1.0]),
        torch.tensor([True]),
        {"goal_progress": torch.tensor([0.8]), "timeout": torch.tensor([False])},
    )

    mock_policy = Mock()
    mock_policy.act.return_value = torch.randn(1, 12)

    metrics = evaluate_robustness(
        env=mock_env,
        policy=mock_policy,
        scenario=TestScenario.CLEAN,
        num_episodes=5,
    )

    assert metrics.scenario_name == "CLEAN"
    assert len(metrics.episode_returns) == 5
    assert len(metrics.episode_lengths) == 5

    print("✓ test_evaluate_robustness passed")


def test_evaluate_all_scenarios():
    """测试所有场景评估"""
    from evaluate_student_robustness import evaluate_all_scenarios, TestScenario

    # 创建模拟对象
    mock_env = Mock()
    mock_env.reset.return_value = {
        "proprio": torch.randn(1, 53),
        "depth": torch.randn(1, 4, 58, 87),
    }
    mock_env.step.return_value = (
        {"proprio": torch.randn(1, 53), "depth": torch.randn(1, 4, 58, 87)},
        torch.tensor([1.0]),
        torch.tensor([True]),
        {"goal_progress": torch.tensor([0.8]), "timeout": torch.tensor([False])},
    )

    mock_policy = Mock()
    mock_policy.act.return_value = torch.randn(1, 12)

    all_metrics = evaluate_all_scenarios(
        env=mock_env,
        policy=mock_policy,
        num_episodes=3,
    )

    # 验证所有场景都被评估
    assert len(all_metrics) == len(TestScenario)
    scenario_names = [m.scenario_name for m in all_metrics]
    assert "CLEAN" in scenario_names
    assert "HIGH_NOISE" in scenario_names
    assert "HIGH_LATENCY" in scenario_names
    assert "CAMERA_DROPOUT" in scenario_names
    assert "COMBINED_STRESS" in scenario_names

    print("✓ test_evaluate_all_scenarios passed")


# ============================================================
# 测试 6: 比较函数
# ============================================================

def test_compare_with_baseline():
    """测试与基线比较"""
    from evaluate_student_robustness import (
        compare_with_baseline,
        RobustnessMetrics,
    )

    baseline_metrics = RobustnessMetrics(
        scenario_name="CLEAN",
        episode_returns=[100.0] * 10,
        episode_lengths=[500] * 10,
        goal_progress=[0.8] * 10,
        timeout_rate=0.0,
    )

    test_metrics = RobustnessMetrics(
        scenario_name="HIGH_NOISE",
        episode_returns=[80.0] * 10,
        episode_lengths=[400] * 10,
        goal_progress=[0.6] * 10,
        timeout_rate=0.1,
    )

    comparison = compare_with_baseline(baseline_metrics, test_metrics)

    assert "return_ratio" in comparison
    assert "length_ratio" in comparison
    assert "goal_progress_diff" in comparison
    assert comparison["return_ratio"] == 0.8  # 80/100
    assert comparison["length_ratio"] == 0.8  # 400/500

    print("✓ test_compare_with_baseline passed")


def test_compare_with_dagger_baseline():
    """测试与 DAgger 基线比较"""
    from evaluate_student_robustness import compare_with_dagger_baseline

    finetuned_returns = [100.0, 105.0, 95.0, 110.0, 90.0]
    dagger_returns = [80.0, 85.0, 75.0, 90.0, 70.0]

    comparison = compare_with_dagger_baseline(finetuned_returns, dagger_returns)

    assert "improvement_ratio" in comparison
    assert "improvement_percentage" in comparison
    assert comparison["improvement_ratio"] > 1.0  # Fine-tuned 应该更好

    print("✓ test_compare_with_dagger_baseline passed")


# ============================================================
# 测试 7: 报告生成
# ============================================================

def test_generate_report():
    """测试生成评估报告"""
    from evaluate_student_robustness import generate_report, RobustnessMetrics

    metrics_list = [
        RobustnessMetrics(
            scenario_name="CLEAN",
            episode_returns=[100.0] * 5,
            episode_lengths=[500] * 5,
            goal_progress=[0.8] * 5,
            timeout_rate=0.0,
        ),
        RobustnessMetrics(
            scenario_name="HIGH_NOISE",
            episode_returns=[80.0] * 5,
            episode_lengths=[400] * 5,
            goal_progress=[0.6] * 5,
            timeout_rate=0.1,
        ),
    ]

    report = generate_report(metrics_list)

    assert "summary" in report
    assert "scenarios" in report
    assert "comparisons" in report
    assert len(report["scenarios"]) == 2

    print("✓ test_generate_report passed")


def test_save_report_json():
    """测试保存 JSON 报告"""
    from evaluate_student_robustness import save_report, RobustnessMetrics

    metrics_list = [
        RobustnessMetrics(
            scenario_name="CLEAN",
            episode_returns=[100.0] * 5,
            episode_lengths=[500] * 5,
            goal_progress=[0.8] * 5,
            timeout_rate=0.0,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.json"

        save_report(metrics_list, report_path)

        assert report_path.exists()

        # 验证 JSON 格式正确
        with open(report_path, "r") as f:
            loaded_report = json.load(f)

        assert "summary" in loaded_report
        assert "scenarios" in loaded_report

    print("✓ test_save_report_json passed")


def test_print_report_summary():
    """测试打印报告摘要"""
    from evaluate_student_robustness import print_report_summary, RobustnessMetrics

    metrics_list = [
        RobustnessMetrics(
            scenario_name="CLEAN",
            episode_returns=[100.0] * 5,
            episode_lengths=[500] * 5,
            goal_progress=[0.8] * 5,
            timeout_rate=0.0,
        ),
        RobustnessMetrics(
            scenario_name="HIGH_NOISE",
            episode_returns=[80.0] * 5,
            episode_lengths=[400] * 5,
            goal_progress=[0.6] * 5,
            timeout_rate=0.1,
        ),
    ]

    # 不应抛出异常
    print_report_summary(metrics_list)

    print("✓ test_print_report_summary passed")


# ============================================================
# 测试 8: 命令行参数解析
# ============================================================

def test_parse_args_default():
    """测试默认命令行参数"""
    from evaluate_student_robustness import parse_args

    # 模拟命令行参数
    with patch("sys.argv", ["evaluate_student_robustness.py", "--checkpoint", "/path/to/ckpt.pt"]):
        args = parse_args()

    assert args.checkpoint == "/path/to/ckpt.pt"
    assert args.num_episodes == 100
    assert args.device == "cuda"

    print("✓ test_parse_args_default passed")


def test_parse_args_custom():
    """测试自定义命令行参数"""
    from evaluate_student_robustness import parse_args

    with patch("sys.argv", [
        "evaluate_student_robustness.py",
        "--checkpoint", "/path/to/ckpt.pt",
        "--num_episodes", "50",
        "--device", "cpu",
        "--output", "/path/to/report.json",
        "--scenarios", "CLEAN", "HIGH_NOISE",
    ]):
        args = parse_args()

    assert args.checkpoint == "/path/to/ckpt.pt"
    assert args.num_episodes == 50
    assert args.device == "cpu"
    assert args.output == "/path/to/report.json"
    assert args.scenarios == ["CLEAN", "HIGH_NOISE"]

    print("✓ test_parse_args_custom passed")


# ============================================================
# 测试 9: 边界情况
# ============================================================

def test_empty_episode_returns():
    """测试空 episode returns 处理"""
    from evaluate_student_robustness import RobustnessMetrics

    metrics = RobustnessMetrics(
        scenario_name="TEST",
        episode_returns=[],
        episode_lengths=[],
        goal_progress=[],
        timeout_rate=0.0,
    )

    stats = metrics.compute_statistics()

    # 应该返回 NaN 或 0
    assert "return_mean" in stats

    print("✓ test_empty_episode_returns passed")


def test_single_episode():
    """测试单个 episode 评估"""
    from evaluate_student_robustness import RobustnessMetrics

    metrics = RobustnessMetrics(
        scenario_name="TEST",
        episode_returns=[100.0],
        episode_lengths=[500],
        goal_progress=[0.8],
        timeout_rate=0.0,
    )

    stats = metrics.compute_statistics()

    assert stats["return_mean"] == 100.0
    assert stats["return_std"] == 0.0

    print("✓ test_single_episode passed")


def test_negative_rewards():
    """测试负奖励处理"""
    from evaluate_student_robustness import RobustnessMetrics

    metrics = RobustnessMetrics(
        scenario_name="TEST",
        episode_returns=[-10.0, -20.0, -5.0],
        episode_lengths=[100, 50, 150],
        goal_progress=[0.1, 0.05, 0.15],
        timeout_rate=0.5,
    )

    stats = metrics.compute_statistics()

    assert stats["return_mean"] < 0
    assert stats["return_min"] == -20.0
    assert stats["return_max"] == -5.0

    print("✓ test_negative_rewards passed")


# ============================================================
# 测试 10: 集成测试
# ============================================================

def test_full_evaluation_pipeline():
    """测试完整评估流程"""
    from evaluate_student_robustness import (
        EvaluationConfig,
        TestScenario,
        evaluate_all_scenarios,
        generate_report,
        save_report,
    )

    # 创建模拟对象
    mock_env = Mock()
    mock_env.reset.return_value = {
        "proprio": torch.randn(1, 53),
        "depth": torch.randn(1, 4, 58, 87),
    }
    mock_env.step.return_value = (
        {"proprio": torch.randn(1, 53), "depth": torch.randn(1, 4, 58, 87)},
        torch.tensor([1.0]),
        torch.tensor([True]),
        {"goal_progress": torch.tensor([0.8]), "timeout": torch.tensor([False])},
    )

    mock_policy = Mock()
    mock_policy.act.return_value = torch.randn(1, 12)

    # 运行评估
    all_metrics = evaluate_all_scenarios(
        env=mock_env,
        policy=mock_policy,
        num_episodes=2,
    )

    # 生成报告
    report = generate_report(all_metrics)

    # 保存报告
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "evaluation_report.json"
        save_report(all_metrics, report_path)

        assert report_path.exists()

    print("✓ test_full_evaluation_pipeline passed")


# ============================================================
# 运行所有测试
# ============================================================

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 Phase 6 鲁棒性评估脚本测试")
    print("=" * 60)

    tests = [
        # 配置测试
        test_evaluation_config_default,
        test_evaluation_config_custom,
        test_evaluation_config_validation,
        # 场景测试
        test_test_scenario_enum,
        test_get_scenario_params,
        # 指标测试
        test_robustness_metrics_dataclass,
        test_robustness_metrics_statistics,
        test_robustness_metrics_to_dict,
        # 增强测试
        test_apply_gaussian_noise,
        test_apply_salt_pepper_noise,
        test_apply_camera_dropout,
        test_apply_depth_latency,
        test_apply_test_scenario,
        # 评估测试
        test_evaluate_single_episode,
        test_evaluate_robustness,
        test_evaluate_all_scenarios,
        # 比较测试
        test_compare_with_baseline,
        test_compare_with_dagger_baseline,
        # 报告测试
        test_generate_report,
        test_save_report_json,
        test_print_report_summary,
        # 命令行测试
        test_parse_args_default,
        test_parse_args_custom,
        # 边界情况测试
        test_empty_episode_returns,
        test_single_episode,
        test_negative_rewards,
        # 集成测试
        test_full_evaluation_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
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
