#!/usr/bin/env python3
"""
Phase 6: Student Policy 鲁棒性评估脚本

评估 RL Fine-tuning 后的 Student Policy 在各种测试场景下的鲁棒性。

主要功能：
1. 定义测试场景（Clean, High Noise, High Latency, Camera Dropout, Combined Stress）
2. 应用场景特定的增强
3. 评估 episode returns, lengths, goal progress
4. 与基线比较
5. 生成评估报告

使用方法：
    python evaluate_student_robustness.py --checkpoint <path> --num_episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# 确保项目路径可导入
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))


# ============================================================
# 评估配置数据类
# ============================================================

@dataclass
class EvaluationConfig:
    """评估配置数据类"""

    # 评估参数
    num_episodes: int = 100
    num_envs: int = 256
    max_steps_per_episode: int = 1000

    # 模型配置
    checkpoint_path: str = ""
    dagger_checkpoint_path: str = ""  # 用于比较

    # 输出配置
    output_path: str = ""
    verbose: bool = True

    # 设备
    device: str = "cuda"

    def validate(self) -> None:
        """验证配置参数"""
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode must be positive, got {self.max_steps_per_episode}"
            )


# ============================================================
# 测试场景枚举
# ============================================================

class TestScenario(Enum):
    """测试场景枚举

    定义不同的测试场景用于评估策略鲁棒性：
    - CLEAN: 无增强，测量最佳性能
    - HIGH_NOISE: 高噪声（Gaussian std=0.08, Salt-pepper prob=0.04）
    - HIGH_LATENCY: 高延迟（Depth delay=5 frames, Action delay=8 frames）
    - CAMERA_DROPOUT: 相机丢失（70% 离线概率）
    - COMBINED_STRESS: 组合压力测试（所有增强最大强度）
    """

    CLEAN = "clean"
    HIGH_NOISE = "high_noise"
    HIGH_LATENCY = "high_latency"
    CAMERA_DROPOUT = "camera_dropout"
    COMBINED_STRESS = "combined_stress"


def get_scenario_params(scenario: TestScenario) -> Dict[str, Any]:
    """获取场景参数

    Args:
        scenario: 测试场景

    Returns:
        场景参数字典
    """
    params = {
        TestScenario.CLEAN: {
            "gaussian_std": 0.0,
            "salt_pepper_prob": 0.0,
            "missing_pixel_prob": 0.0,
            "camera_dropout_prob": 0.0,
            "depth_delay_frames": 0,
            "action_delay_frames": 0,
            "brightness_range": (1.0, 1.0),
            "contrast_range": (1.0, 1.0),
        },
        TestScenario.HIGH_NOISE: {
            "gaussian_std": 0.08,  # 2x 训练最大值
            "salt_pepper_prob": 0.04,  # 2x 训练最大值
            "missing_pixel_prob": 0.02,
            "camera_dropout_prob": 0.0,
            "depth_delay_frames": 0,
            "action_delay_frames": 0,
            "brightness_range": (0.8, 1.2),
            "contrast_range": (0.8, 1.2),
        },
        TestScenario.HIGH_LATENCY: {
            "gaussian_std": 0.0,
            "salt_pepper_prob": 0.0,
            "missing_pixel_prob": 0.0,
            "camera_dropout_prob": 0.0,
            "depth_delay_frames": 5,
            "action_delay_frames": 8,
            "brightness_range": (1.0, 1.0),
            "contrast_range": (1.0, 1.0),
        },
        TestScenario.CAMERA_DROPOUT: {
            "gaussian_std": 0.0,
            "salt_pepper_prob": 0.0,
            "missing_pixel_prob": 0.0,
            "camera_dropout_prob": 0.7,  # 高于训练时的 50%
            "depth_delay_frames": 0,
            "action_delay_frames": 0,
            "brightness_range": (1.0, 1.0),
            "contrast_range": (1.0, 1.0),
        },
        TestScenario.COMBINED_STRESS: {
            "gaussian_std": 0.08,
            "salt_pepper_prob": 0.04,
            "missing_pixel_prob": 0.02,
            "camera_dropout_prob": 0.7,
            "depth_delay_frames": 5,
            "action_delay_frames": 8,
            "brightness_range": (0.7, 1.3),
            "contrast_range": (0.7, 1.3),
        },
    }

    return params.get(scenario, params[TestScenario.CLEAN])


# ============================================================
# 鲁棒性指标数据类
# ============================================================

@dataclass
class RobustnessMetrics:
    """鲁棒性指标数据类

    存储单个测试场景的评估结果。
    """

    scenario_name: str
    episode_returns: List[float]
    episode_lengths: List[int]
    goal_progress: List[float]
    timeout_rate: float

    def compute_statistics(self) -> Dict[str, float]:
        """计算统计指标

        Returns:
            统计指标字典
        """
        import numpy as np

        if len(self.episode_returns) == 0:
            return {
                "return_mean": float("nan"),
                "return_std": float("nan"),
                "return_min": float("nan"),
                "return_max": float("nan"),
                "length_mean": float("nan"),
                "length_std": float("nan"),
                "goal_progress_mean": float("nan"),
                "goal_progress_std": float("nan"),
            }

        returns = np.array(self.episode_returns)
        lengths = np.array(self.episode_lengths)
        progress = np.array(self.goal_progress)

        return {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "return_min": float(np.min(returns)),
            "return_max": float(np.max(returns)),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "goal_progress_mean": float(np.mean(progress)),
            "goal_progress_std": float(np.std(progress)),
            "timeout_rate": self.timeout_rate,
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            包含所有数据的字典
        """
        return {
            "scenario_name": self.scenario_name,
            "statistics": self.compute_statistics(),
            "raw_data": {
                "episode_returns": self.episode_returns,
                "episode_lengths": self.episode_lengths,
                "goal_progress": self.goal_progress,
                "timeout_rate": self.timeout_rate,
            },
        }


# ============================================================
# 增强函数
# ============================================================

def apply_gaussian_noise(depth: Tensor, std: float) -> Tensor:
    """应用高斯噪声

    Args:
        depth: 深度图像 [B, C, H, W]
        std: 噪声标准差

    Returns:
        添加噪声后的深度图像
    """
    if std <= 0:
        return depth

    noise = torch.randn_like(depth) * std
    return depth + noise


def apply_salt_pepper_noise(depth: Tensor, prob: float) -> Tensor:
    """应用椒盐噪声

    Args:
        depth: 深度图像 [B, C, H, W]
        prob: 噪声概率

    Returns:
        添加噪声后的深度图像
    """
    if prob <= 0:
        return depth

    noisy = depth.clone()

    # 生成随机掩码
    mask = torch.rand_like(depth)

    # Salt (设为 1.0，表示最大深度)
    salt_mask = mask < (prob / 2)
    noisy[salt_mask] = 1.0

    # Pepper (设为 0.0 或 -0.5，表示最小/无效深度)
    pepper_mask = (mask >= (prob / 2)) & (mask < prob)
    noisy[pepper_mask] = 0.0

    return noisy


def apply_camera_dropout(depth: Tensor, prob: float) -> Tensor:
    """应用相机丢失

    Args:
        depth: 深度图像 [B, C, H, W]
        prob: 丢失概率

    Returns:
        处理后的深度图像
    """
    if prob <= 0:
        return depth

    dropout = depth.clone()

    # 按 batch 维度随机丢失
    batch_size = depth.shape[0]
    dropout_mask = torch.rand(batch_size, device=depth.device) < prob

    # 将丢失的深度设为无效值（-0.5 在归一化空间）
    dropout[dropout_mask] = -0.5

    return dropout


class DepthLatencyBuffer:
    """深度延迟缓冲区

    模拟深度相机的处理延迟。
    """

    def __init__(self, delay_frames: int = 0):
        """初始化延迟缓冲区

        Args:
            delay_frames: 延迟帧数
        """
        self.delay_frames = delay_frames
        self.buffer: deque = deque(maxlen=delay_frames + 1)

    def reset(self) -> None:
        """重置缓冲区"""
        self.buffer.clear()

    def apply(self, depth: Tensor) -> Tensor:
        """应用延迟

        Args:
            depth: 当前深度图像

        Returns:
            延迟后的深度图像
        """
        if self.delay_frames <= 0:
            return depth

        self.buffer.append(depth.clone())

        # 如果缓冲区未满，返回第一帧
        if len(self.buffer) <= self.delay_frames:
            return self.buffer[0].clone()

        # 返回延迟的帧
        return self.buffer[0].clone()


def apply_test_scenario(
    depth: Tensor,
    scenario: TestScenario,
    latency_buffer: Optional[DepthLatencyBuffer] = None,
) -> Tensor:
    """应用测试场景增强

    Args:
        depth: 深度图像
        scenario: 测试场景
        latency_buffer: 延迟缓冲区（可选）

    Returns:
        增强后的深度图像
    """
    params = get_scenario_params(scenario)

    augmented = depth.clone()

    # 应用高斯噪声
    if params["gaussian_std"] > 0:
        augmented = apply_gaussian_noise(augmented, params["gaussian_std"])

    # 应用椒盐噪声
    if params["salt_pepper_prob"] > 0:
        augmented = apply_salt_pepper_noise(augmented, params["salt_pepper_prob"])

    # 应用相机丢失
    if params["camera_dropout_prob"] > 0:
        augmented = apply_camera_dropout(augmented, params["camera_dropout_prob"])

    # 应用延迟（如果提供了缓冲区）
    if latency_buffer is not None and params["depth_delay_frames"] > 0:
        augmented = latency_buffer.apply(augmented)

    return augmented


# ============================================================
# 评估函数
# ============================================================

def evaluate_single_episode(
    env: Any,
    policy: Any,
    scenario: TestScenario,
    max_steps: int = 1000,
    latency_buffer: Optional[DepthLatencyBuffer] = None,
) -> Dict[str, Any]:
    """评估单个 episode

    Args:
        env: 环境
        policy: 策略
        scenario: 测试场景
        max_steps: 最大步数
        latency_buffer: 延迟缓冲区

    Returns:
        episode 结果字典
    """
    obs = env.reset()
    done = False
    episode_return = 0.0
    episode_length = 0
    goal_progress = 0.0
    timeout = False

    # 重置延迟缓冲区
    if latency_buffer is not None:
        latency_buffer.reset()

    while not done and episode_length < max_steps:
        # 获取观测
        proprio = obs.get("proprio", obs.get("proprioception"))
        depth = obs.get("depth", obs.get("depth_image"))

        # 应用场景增强
        if depth is not None:
            depth = apply_test_scenario(depth, scenario, latency_buffer)

        # 获取动作
        with torch.no_grad():
            action = policy.act(proprio, depth) if depth is not None else policy.act(proprio)

        # 执行动作
        obs_next, reward, done_tensor, info = env.step(action)

        # 累积奖励
        if isinstance(reward, Tensor):
            episode_return += reward.item() if reward.numel() == 1 else reward.sum().item()
        else:
            episode_return += float(reward)

        episode_length += 1

        # 处理 done
        if isinstance(done_tensor, Tensor):
            done = done_tensor.item() if done_tensor.numel() == 1 else done_tensor.any().item()
        else:
            done = bool(done_tensor)

        # 获取 goal progress
        if "goal_progress" in info:
            gp = info["goal_progress"]
            goal_progress = gp.item() if isinstance(gp, Tensor) else float(gp)

        # 检查超时
        if "timeout" in info:
            to = info["timeout"]
            timeout = to.item() if isinstance(to, Tensor) else bool(to)

        obs = obs_next

    return {
        "episode_return": episode_return,
        "episode_length": episode_length,
        "goal_progress": goal_progress,
        "timeout": timeout,
    }


def evaluate_robustness(
    env: Any,
    policy: Any,
    scenario: TestScenario,
    num_episodes: int = 100,
    max_steps: int = 1000,
) -> RobustnessMetrics:
    """评估策略在特定场景下的鲁棒性

    Args:
        env: 环境
        policy: 策略
        scenario: 测试场景
        num_episodes: 评估 episode 数
        max_steps: 每个 episode 最大步数

    Returns:
        RobustnessMetrics 对象
    """
    episode_returns = []
    episode_lengths = []
    goal_progress_list = []
    timeout_count = 0

    # 创建延迟缓冲区
    params = get_scenario_params(scenario)
    latency_buffer = None
    if params["depth_delay_frames"] > 0:
        latency_buffer = DepthLatencyBuffer(params["depth_delay_frames"])

    for ep in range(num_episodes):
        result = evaluate_single_episode(
            env=env,
            policy=policy,
            scenario=scenario,
            max_steps=max_steps,
            latency_buffer=latency_buffer,
        )

        episode_returns.append(result["episode_return"])
        episode_lengths.append(result["episode_length"])
        goal_progress_list.append(result["goal_progress"])
        if result["timeout"]:
            timeout_count += 1

    timeout_rate = timeout_count / num_episodes if num_episodes > 0 else 0.0

    return RobustnessMetrics(
        scenario_name=scenario.name,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        goal_progress=goal_progress_list,
        timeout_rate=timeout_rate,
    )


def evaluate_all_scenarios(
    env: Any,
    policy: Any,
    num_episodes: int = 100,
    max_steps: int = 1000,
    scenarios: Optional[List[TestScenario]] = None,
) -> List[RobustnessMetrics]:
    """评估所有测试场景

    Args:
        env: 环境
        policy: 策略
        num_episodes: 每个场景的 episode 数
        max_steps: 每个 episode 最大步数
        scenarios: 要评估的场景列表（默认所有场景）

    Returns:
        所有场景的 RobustnessMetrics 列表
    """
    if scenarios is None:
        scenarios = list(TestScenario)

    all_metrics = []

    for scenario in scenarios:
        print(f"[INFO] Evaluating scenario: {scenario.name}")
        metrics = evaluate_robustness(
            env=env,
            policy=policy,
            scenario=scenario,
            num_episodes=num_episodes,
            max_steps=max_steps,
        )
        all_metrics.append(metrics)

        # 打印简要结果
        stats = metrics.compute_statistics()
        print(
            f"  Return: {stats['return_mean']:.2f} +/- {stats['return_std']:.2f}, "
            f"Goal Progress: {stats['goal_progress_mean']:.2%}"
        )

    return all_metrics


# ============================================================
# 比较函数
# ============================================================

def compare_with_baseline(
    baseline_metrics: RobustnessMetrics,
    test_metrics: RobustnessMetrics,
) -> Dict[str, float]:
    """与基线比较

    Args:
        baseline_metrics: 基线指标（通常是 CLEAN 场景）
        test_metrics: 测试指标

    Returns:
        比较结果字典
    """
    baseline_stats = baseline_metrics.compute_statistics()
    test_stats = test_metrics.compute_statistics()

    baseline_return = baseline_stats["return_mean"]
    test_return = test_stats["return_mean"]

    baseline_length = baseline_stats["length_mean"]
    test_length = test_stats["length_mean"]

    baseline_progress = baseline_stats["goal_progress_mean"]
    test_progress = test_stats["goal_progress_mean"]

    return {
        "return_ratio": test_return / baseline_return if baseline_return != 0 else 0.0,
        "length_ratio": test_length / baseline_length if baseline_length != 0 else 0.0,
        "goal_progress_diff": test_progress - baseline_progress,
        "return_diff": test_return - baseline_return,
        "length_diff": test_length - baseline_length,
        "baseline_scenario": baseline_metrics.scenario_name,
        "test_scenario": test_metrics.scenario_name,
    }


def compare_with_dagger_baseline(
    finetuned_returns: List[float],
    dagger_returns: List[float],
) -> Dict[str, float]:
    """与 DAgger 基线比较

    Args:
        finetuned_returns: Fine-tuned 策略的 returns
        dagger_returns: DAgger 策略的 returns

    Returns:
        比较结果字典
    """
    import numpy as np

    finetuned_mean = np.mean(finetuned_returns)
    dagger_mean = np.mean(dagger_returns)

    improvement = finetuned_mean - dagger_mean
    improvement_ratio = finetuned_mean / dagger_mean if dagger_mean != 0 else 0.0
    improvement_percentage = (improvement / abs(dagger_mean) * 100) if dagger_mean != 0 else 0.0

    return {
        "finetuned_mean": float(finetuned_mean),
        "dagger_mean": float(dagger_mean),
        "improvement": float(improvement),
        "improvement_ratio": float(improvement_ratio),
        "improvement_percentage": float(improvement_percentage),
    }


# ============================================================
# 报告生成函数
# ============================================================

def generate_report(metrics_list: List[RobustnessMetrics]) -> Dict[str, Any]:
    """生成评估报告

    Args:
        metrics_list: 所有场景的指标列表

    Returns:
        报告字典
    """
    import numpy as np

    # 找到 CLEAN 场景作为基线
    baseline_metrics = None
    for m in metrics_list:
        if m.scenario_name == "CLEAN":
            baseline_metrics = m
            break

    # 如果没有 CLEAN 场景，使用第一个作为基线
    if baseline_metrics is None and len(metrics_list) > 0:
        baseline_metrics = metrics_list[0]

    # 生成场景报告
    scenarios = []
    comparisons = []

    for metrics in metrics_list:
        scenario_data = metrics.to_dict()
        scenarios.append(scenario_data)

        # 与基线比较
        if baseline_metrics is not None and metrics.scenario_name != baseline_metrics.scenario_name:
            comparison = compare_with_baseline(baseline_metrics, metrics)
            comparisons.append(comparison)

    # 生成摘要
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_scenarios": len(metrics_list),
        "scenarios_evaluated": [m.scenario_name for m in metrics_list],
    }

    # 添加整体统计
    if len(metrics_list) > 0:
        all_returns = []
        for m in metrics_list:
            all_returns.extend(m.episode_returns)

        if len(all_returns) > 0:
            summary["overall_return_mean"] = float(np.mean(all_returns))
            summary["overall_return_std"] = float(np.std(all_returns))

    return {
        "summary": summary,
        "scenarios": scenarios,
        "comparisons": comparisons,
    }


def save_report(
    metrics_list: List[RobustnessMetrics],
    output_path: Union[str, Path],
) -> None:
    """保存评估报告到 JSON 文件

    Args:
        metrics_list: 所有场景的指标列表
        output_path: 输出文件路径
    """
    report = generate_report(metrics_list)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Report saved to: {output_path}")


def print_report_summary(metrics_list: List[RobustnessMetrics]) -> None:
    """打印报告摘要

    Args:
        metrics_list: 所有场景的指标列表
    """
    print("\n" + "=" * 60)
    print("鲁棒性评估报告摘要")
    print("=" * 60)

    # 找到基线
    baseline_metrics = None
    for m in metrics_list:
        if m.scenario_name == "CLEAN":
            baseline_metrics = m
            break

    for metrics in metrics_list:
        stats = metrics.compute_statistics()
        print(f"\n场景: {metrics.scenario_name}")
        print(f"  Episode Return: {stats['return_mean']:.2f} +/- {stats['return_std']:.2f}")
        print(f"  Episode Length: {stats['length_mean']:.1f} +/- {stats['length_std']:.1f}")
        print(f"  Goal Progress: {stats['goal_progress_mean']:.2%}")
        print(f"  Timeout Rate: {stats['timeout_rate']:.2%}")

        # 与基线比较
        if baseline_metrics is not None and metrics.scenario_name != "CLEAN":
            comparison = compare_with_baseline(baseline_metrics, metrics)
            print(f"  vs CLEAN: Return Ratio = {comparison['return_ratio']:.2%}")

    print("\n" + "=" * 60)


# ============================================================
# 命令行参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Student Policy 鲁棒性评估脚本"
    )

    # 必需参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Fine-tuned Student Policy checkpoint 路径",
    )

    # 评估参数
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="每个场景评估的 episode 数（默认: 100）",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="每个 episode 的最大步数（默认: 1000）",
    )

    # 场景选择
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="要评估的场景列表（默认: 所有场景）",
    )

    # 输出配置
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告路径（JSON 格式）",
    )

    # 比较配置
    parser.add_argument(
        "--dagger_checkpoint",
        type=str,
        default=None,
        help="DAgger baseline checkpoint 路径（用于比较）",
    )

    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="评估设备（默认: cuda）",
    )

    # 其他
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="详细输出",
    )

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> EvaluationConfig:
    """将命令行参数转换为配置对象"""
    return EvaluationConfig(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        checkpoint_path=args.checkpoint,
        dagger_checkpoint_path=args.dagger_checkpoint or "",
        output_path=args.output or "",
        verbose=args.verbose,
        device=args.device,
    )


# ============================================================
# 主函数入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Student Policy 鲁棒性评估脚本")
    print("=" * 60)
    print()
    print("此脚本需要 Isaac Lab 环境。")
    print("请在正确配置 Isaac Lab 后运行。")
    print()
    print("使用方法:")
    print("  python evaluate_student_robustness.py --checkpoint <path>")
    print()
    print("可用场景:")
    for scenario in TestScenario:
        print(f"  - {scenario.name}: {scenario.value}")
    print()
    print("测试时，可以直接导入模块中的函数:")
    print("  from evaluate_student_robustness import evaluate_robustness")
    print("=" * 60)
