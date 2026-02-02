"""
Student RL Fine-tuning 配置使用示例

演示如何使用和自定义 Student Fine-tuning 配置。
"""

import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)


def example_basic_usage():
    """示例 1: 基本使用"""
    print("\n" + "="*60)
    print("示例 1: 基本使用")
    print("="*60)

    from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
        UnitreeGo2StudentFinetunePPORunnerCfg
    )

    # 创建配置实例
    cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

    # 打印关键配置
    print(f"\n实验名称: {cfg.experiment_name}")
    print(f"学习率: {cfg.algorithm.learning_rate}")
    print(f"最大迭代: {cfg.max_iterations}")
    print(f"每环境步数: {cfg.num_steps_per_env}")
    print(f"Domain Randomization: {cfg.domain_rand_enabled}")
    print(f"Curriculum: {cfg.domain_rand_curriculum}")


def example_custom_hyperparameters():
    """示例 2: 自定义超参数"""
    print("\n" + "="*60)
    print("示例 2: 自定义超参数")
    print("="*60)

    from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
        UnitreeGo2StudentFinetunePPORunnerCfg
    )

    cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

    # 修改超参数
    cfg.algorithm.learning_rate = 5e-5  # 更保守的学习率
    cfg.algorithm.clip_param = 0.15  # 更小的 clip 范围
    cfg.max_iterations = 15000  # 更长的训练

    print(f"\n修改后的学习率: {cfg.algorithm.learning_rate}")
    print(f"修改后的 clip_param: {cfg.algorithm.clip_param}")
    print(f"修改后的最大迭代: {cfg.max_iterations}")


def example_freeze_encoders():
    """示例 3: 冻结编码器"""
    print("\n" + "="*60)
    print("示例 3: 冻结编码器（避免灾难性遗忘）")
    print("="*60)

    from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
        UnitreeGo2StudentFinetunePPORunnerCfg
    )

    cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

    # 冻结所有编码器
    cfg.freeze_proprio_encoder = True
    cfg.freeze_depth_encoder = True
    cfg.freeze_fusion_transformer = True
    cfg.freeze_temporal_transformer = False  # 只训练 temporal transformer

    print("\n编码器冻结状态:")
    print(f"  Proprio Encoder: {'冻结' if cfg.freeze_proprio_encoder else '训练'}")
    print(f"  Depth Encoder: {'冻结' if cfg.freeze_depth_encoder else '训练'}")
    print(f"  Fusion Transformer: {'冻结' if cfg.freeze_fusion_transformer else '训练'}")
    print(f"  Temporal Transformer: {'冻结' if cfg.freeze_temporal_transformer else '训练'}")


def example_disable_domain_randomization():
    """示例 4: 禁用 Domain Randomization"""
    print("\n" + "="*60)
    print("示例 4: 禁用 Domain Randomization")
    print("="*60)

    from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
        UnitreeGo2StudentFinetunePPORunnerCfg
    )

    cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

    # 禁用 domain randomization
    cfg.domain_rand_enabled = False
    cfg.domain_rand_curriculum = False

    print(f"\nDomain Randomization: {cfg.domain_rand_enabled}")
    print(f"Curriculum: {cfg.domain_rand_curriculum}")
    print("\n注意: 禁用 Domain Randomization 可能降低 sim-to-real 迁移性能")


def example_conservative_training():
    """示例 5: 保守训练配置（最小化遗忘风险）"""
    print("\n" + "="*60)
    print("示例 5: 保守训练配置")
    print("="*60)

    from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
        UnitreeGo2StudentFinetunePPORunnerCfg
    )

    cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

    # 保守配置
    cfg.algorithm.learning_rate = 5e-5  # 更低的学习率
    cfg.algorithm.clip_param = 0.15  # 更小的 clip
    cfg.algorithm.max_grad_norm = 0.5  # 更强的梯度裁剪
    cfg.freeze_depth_encoder = True  # 冻结 depth encoder
    cfg.freeze_fusion_transformer = True  # 冻结 fusion

    print("\n保守训练配置:")
    print(f"  学习率: {cfg.algorithm.learning_rate}")
    print(f"  Clip param: {cfg.algorithm.clip_param}")
    print(f"  Max grad norm: {cfg.algorithm.max_grad_norm}")
    print(f"  冻结 Depth Encoder: {cfg.freeze_depth_encoder}")
    print(f"  冻结 Fusion: {cfg.freeze_fusion_transformer}")


def example_aggressive_training():
    """示例 6: 激进训练配置（最大化性能提升）"""
    print("\n" + "="*60)
    print("示例 6: 激进训练配置")
    print("="*60)

    from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
        UnitreeGo2StudentFinetunePPORunnerCfg
    )

    cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

    # 激进配置
    cfg.algorithm.learning_rate = 2e-4  # 更高的学习率
    cfg.algorithm.entropy_coef = 0.02  # 更多探索
    cfg.max_iterations = 20000  # 更长训练
    cfg.domain_rand_enabled = True  # 启用 domain randomization
    cfg.domain_rand_curriculum = True  # 启用 curriculum

    # 所有编码器都训练
    cfg.freeze_proprio_encoder = False
    cfg.freeze_depth_encoder = False
    cfg.freeze_fusion_transformer = False
    cfg.freeze_temporal_transformer = False

    print("\n激进训练配置:")
    print(f"  学习率: {cfg.algorithm.learning_rate}")
    print(f"  Entropy coef: {cfg.algorithm.entropy_coef}")
    print(f"  最大迭代: {cfg.max_iterations}")
    print(f"  端到端训练: 是")
    print(f"  Domain Randomization: {cfg.domain_rand_enabled}")


def example_calculate_training_time():
    """示例 7: 计算训练时间"""
    print("\n" + "="*60)
    print("示例 7: 估算训练时间")
    print("="*60)

    from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
        UnitreeGo2StudentFinetunePPORunnerCfg
    )

    cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

    # 假设参数
    num_envs = 256
    steps_per_second = 1000  # 假设每秒 1000 步（取决于硬件）

    # 计算
    total_steps = num_envs * cfg.num_steps_per_env * cfg.max_iterations
    total_seconds = total_steps / steps_per_second
    total_hours = total_seconds / 3600

    print(f"\n训练参数:")
    print(f"  环境数: {num_envs}")
    print(f"  每环境步数: {cfg.num_steps_per_env}")
    print(f"  最大迭代: {cfg.max_iterations}")
    print(f"\n估算:")
    print(f"  总步数: {total_steps:,}")
    print(f"  总时间: {total_hours:.1f} 小时")
    print(f"  (假设 {steps_per_second} steps/sec)")


if __name__ == "__main__":
    """运行所有示例"""
    print("\n" + "="*60)
    print("Student RL Fine-tuning 配置使用示例")
    print("="*60)

    examples = [
        example_basic_usage,
        example_custom_hyperparameters,
        example_freeze_encoders,
        example_disable_domain_randomization,
        example_conservative_training,
        example_aggressive_training,
        example_calculate_training_time,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n错误: {e}")

    print("\n" + "="*60)
    print("所有示例运行完成")
    print("="*60)
