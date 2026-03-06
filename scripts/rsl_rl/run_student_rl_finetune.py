#!/usr/bin/env python3
"""
Student RL Fine-tuning 统一启动脚本

整合所有 RL Fine-tuning 组件，提供完整的训练和评估流程：
1. 加载 DAgger 预训练的 checkpoint
2. 创建 StudentActorCritic（添加 ValueHead）
3. 初始化 PPOStudent 算法
4. 集成 Domain Randomization
5. 运行训练循环
6. 可选：训练后自动评估

使用方法：
    # 训练模式（直接运行，无需 ./isaaclab.sh）
    python run_student_rl_finetune.py train --dagger_checkpoint <path>

    # 评估模式
    python run_student_rl_finetune.py evaluate --checkpoint <path>

    # 训练后自动评估
    python run_student_rl_finetune.py train --dagger_checkpoint <path> --auto_evaluate

注意：此脚本已集成 Isaac Lab 环境设置，可以直接通过 python 命令运行。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# Isaac Lab 环境设置（必须在其他导入之前）
# ============================================================

# 确保项目路径可导入
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))
sys.path.insert(0, str(PROJECT_ROOT / "parkour_tasks"))
sys.path.insert(0, str(PROJECT_ROOT / "parkour_isaaclab"))

# 设置 Isaac Lab 环境
from isaaclab_env_setup import setup_isaaclab_env, ensure_env_setup
ensure_env_setup()

# 导入 AppLauncher（在环境设置之后）
from isaaclab.app import AppLauncher


# ============================================================
# 命令行参数解析
# ============================================================

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Student RL Fine-tuning 统一启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练模式（直接运行，无需 ./isaaclab.sh）
  python run_student_rl_finetune.py train --dagger_checkpoint logs/dagger/best.pt

  # 评估模式
  python run_student_rl_finetune.py evaluate --checkpoint logs/finetune/final.pt

  # 训练后自动评估
  python run_student_rl_finetune.py train --dagger_checkpoint logs/dagger/best.pt --auto_evaluate
        """
    )

    # 添加 AppLauncher 参数
    AppLauncher.add_app_launcher_args(parser)

    # 子命令
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # 训练子命令
    train_parser = subparsers.add_parser("train", help="训练模式")
    _add_train_args(train_parser)

    # 评估子命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模式")
    _add_eval_args(eval_parser)

    return parser


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """添加训练参数"""
    # 必需参数
    parser.add_argument(
        "--dagger_checkpoint",
        type=str,
        required=True,
        help="DAgger 训练的 Student Policy checkpoint 路径",
    )

    # 环境配置
    env_group = parser.add_argument_group("环境配置")
    env_group.add_argument(
        "--task",
        type=str,
        default="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-v0",
        help="Isaac Lab 任务名称",
    )
    env_group.add_argument("--num_envs", type=int, default=256, help="环境数量")
    env_group.add_argument("--num_steps_per_env", type=int, default=64, help="每环境步数")
    env_group.add_argument("--max_iterations", type=int, default=10000, help="最大迭代次数")
    env_group.add_argument("--seed", type=int, default=42, help="随机种子")

    # PPO 超参数
    ppo_group = parser.add_argument_group("PPO 超参数")
    ppo_group.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    ppo_group.add_argument("--clip_param", type=float, default=0.2, help="PPO clip 参数")
    ppo_group.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    ppo_group.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    ppo_group.add_argument("--entropy_coef", type=float, default=0.01, help="熵系数")
    ppo_group.add_argument("--value_loss_coef", type=float, default=0.5, help="价值损失系数")
    ppo_group.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    ppo_group.add_argument("--num_learning_epochs", type=int, default=5, help="学习 epoch 数")
    ppo_group.add_argument("--num_mini_batches", type=int, default=4, help="Mini-batch 数")

    # 编码器冻结
    freeze_group = parser.add_argument_group("编码器冻结")
    freeze_group.add_argument("--freeze_encoders", action="store_true", help="冻结编码器")
    freeze_group.add_argument("--freeze_fusion", action="store_true", help="冻结融合 transformer")
    freeze_group.add_argument("--freeze_temporal", action="store_true", help="冻结时序模型")

    # Domain Randomization
    dr_group = parser.add_argument_group("Domain Randomization")
    dr_group.add_argument("--no_domain_rand", action="store_true", help="禁用 Domain Randomization")
    dr_group.add_argument("--no_curriculum", action="store_true", help="禁用课程学习")

    # 日志和检查点
    log_group = parser.add_argument_group("日志和检查点")
    log_group.add_argument(
        "--log_dir",
        type=str,
        default="logs/student_finetune",
        help="日志目录",
    )
    log_group.add_argument(
        "--experiment_name",
        type=str,
        default="student_finetune",
        help="实验名称",
    )
    log_group.add_argument("--save_interval", type=int, default=100, help="保存间隔")
    log_group.add_argument("--log_interval", type=int, default=10, help="日志间隔")

    # 恢复训练
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")

    # 自动评估
    parser.add_argument(
        "--auto_evaluate",
        action="store_true",
        help="训练完成后自动运行评估",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=50,
        help="自动评估时每个场景的 episode 数",
    )

    # 设备
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")

    # Isaac Lab 相关
    parser.add_argument("--headless", action="store_true", help="无头模式运行")
    parser.add_argument("--video", action="store_true", help="录制视频")

def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    """添加评估参数"""
    # 必需参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Fine-tuned Student Policy checkpoint 路径",
    )

    # 环境配置
    env_group = parser.add_argument_group("环境配置")
    env_group.add_argument(
        "--task",
        type=str,
        default="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-v0",
        help="Isaac Lab 任务名称",
    )
    env_group.add_argument("--num_envs", type=int, default=256, help="环境数量")

    # 评估参数
    eval_group = parser.add_argument_group("评估参数")
    eval_group.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="每个场景评估的 episode 数",
    )
    eval_group.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="每个 episode 的最大步数",
    )
    eval_group.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="要评估的场景列表（默认: 所有场景）",
    )

    # 比较配置
    parser.add_argument(
        "--dagger_checkpoint",
        type=str,
        default=None,
        help="DAgger baseline checkpoint 路径（用于比较）",
    )

    # 输出配置
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告路径（JSON 格式）",
    )

    # 设备
    parser.add_argument("--device", type=str, default="cuda", help="评估设备")

    # Isaac Lab 相关
    parser.add_argument("--headless", action="store_true", help="无头模式运行")


# ============================================================
# 训练模式
# ============================================================

def run_training(args: argparse.Namespace) -> Optional[Path]:
    """运行训练流程

    Args:
        args: 命令行参数

    Returns:
        最终检查点路径（如果训练成功）
    """
    import torch

    # 导入训练模块
    from train_student_rl_finetune import (
        TrainingConfig,
        create_student_actor_critic,
        initialize_ppo_student,
        create_domain_randomization,
        save_training_checkpoint,
        load_training_checkpoint,
    )

    print("=" * 60)
    print("Student RL Fine-tuning - 训练模式")
    print("=" * 60)

    # 验证 DAgger checkpoint 存在
    dagger_path = Path(args.dagger_checkpoint)
    if not dagger_path.exists():
        raise FileNotFoundError(f"DAgger checkpoint 不存在: {dagger_path}")

    print(f"[INFO] DAgger checkpoint: {dagger_path}")
    print(f"[INFO] 任务: {args.task}")
    print(f"[INFO] 环境数量: {args.num_envs}")
    print(f"[INFO] 最大迭代次数: {args.max_iterations}")
    print(f"[INFO] 设备: {args.device}")

    # 设置日志目录
    log_dir = Path(args.log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_dir / args.experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 日志目录: {run_dir}")

    # 创建训练配置
    config = TrainingConfig(
        num_envs=args.num_envs,
        num_steps_per_env=args.num_steps_per_env,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        clip_param=args.clip_param,
        gamma=args.gamma,
        lam=args.lam,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        num_learning_epochs=args.num_learning_epochs,
        num_mini_batches=args.num_mini_batches,
        freeze_proprio_encoder=args.freeze_encoders,
        freeze_depth_encoder=args.freeze_encoders,
        freeze_fusion_transformer=args.freeze_fusion,
        freeze_temporal_transformer=args.freeze_temporal,
        domain_rand_enabled=not args.no_domain_rand,
        domain_rand_curriculum=not args.no_curriculum,
        dagger_checkpoint=str(dagger_path),
        log_dir=str(run_dir),
        experiment_name=args.experiment_name,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
    )
    config.validate()

    # 打印配置摘要
    print("\n[配置摘要]")
    print(f"  PPO: lr={config.learning_rate}, clip={config.clip_param}")
    print(f"  GAE: gamma={config.gamma}, lam={config.lam}")
    print(f"  编码器冻结: encoders={args.freeze_encoders}, fusion={args.freeze_fusion}")
    print(f"  Domain Rand: enabled={config.domain_rand_enabled}, curriculum={config.domain_rand_curriculum}")

    # 创建 StudentActorCritic
    print("\n[INFO] 加载 DAgger checkpoint 并创建 StudentActorCritic...")
    actor_critic, checkpoint_meta = create_student_actor_critic(
        checkpoint_path=str(dagger_path),
        freeze_encoders=args.freeze_encoders,
        freeze_fusion=args.freeze_fusion,
        freeze_temporal=args.freeze_temporal,
        device=args.device,
    )
    print(f"[INFO] StudentActorCritic 创建成功")

    # 初始化 PPOStudent
    print("[INFO] 初始化 PPOStudent 算法...")
    ppo = initialize_ppo_student(
        actor_critic=actor_critic,
        learning_rate=config.learning_rate,
        clip_param=config.clip_param,
        gamma=config.gamma,
        lam=config.lam,
        entropy_coef=config.entropy_coef,
        value_loss_coef=config.value_loss_coef,
        max_grad_norm=config.max_grad_norm,
        num_learning_epochs=config.num_learning_epochs,
        num_mini_batches=config.num_mini_batches,
        device=args.device,
    )

    # 初始化 storage
    ppo.init_storage(
        num_envs=config.num_envs,
        num_steps=config.num_steps_per_env,
        proprio_dim=config.proprio_dim,
        depth_shape=config.depth_shape,
        action_dim=config.action_dim,
    )
    print("[INFO] PPOStudent 初始化成功")

    # 创建 Domain Randomization
    domain_rand = None
    if config.domain_rand_enabled:
        domain_rand = create_domain_randomization(
            enabled=True,
            use_curriculum=config.domain_rand_curriculum,
            num_envs=config.num_envs,
            device=args.device,
        )
        print("[INFO] Domain Randomization: 已启用")
        if config.domain_rand_curriculum:
            print("[INFO] Domain Randomization Curriculum: 已启用")

    # 恢复训练
    start_iteration = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_iteration = load_training_checkpoint(
                resume_path,
                actor_critic,
                ppo.optimizer,
            )
            print(f"[INFO] 从迭代 {start_iteration} 恢复训练")
        else:
            print(f"[WARNING] 恢复检查点不存在: {resume_path}")

    # 训练准备就绪
    print("\n" + "=" * 60)
    print("训练准备就绪")
    print("=" * 60)
    print()
    print("[INFO] Isaac Lab 环境已初始化，可以开始训练。")
    print(f"[INFO] 任务: {args.task}")
    print(f"[INFO] 环境数量: {args.num_envs}")
    print()
    print("=" * 60)

    # 保存初始检查点（用于验证）
    initial_checkpoint = run_dir / "checkpoint_initial.pt"
    save_training_checkpoint(
        path=initial_checkpoint,
        actor_critic=actor_critic,
        optimizer=ppo.optimizer,
        iteration=0,
        config=vars(config),
    )
    print(f"[INFO] 初始检查点已保存: {initial_checkpoint}")

    # 创建 Isaac Lab 环境
    print("\n[INFO] 创建 Isaac Lab 环境...")
    import gymnasium as gym
    # 触发 parkour_tasks 中 Gym 环境注册（包括 TeacherCam 任务）
    import parkour_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from vecenv_wrapper import ParkourRslRlVecEnvWrapper

    # 解析环境配置（与 train_student_dagger.py 保持一致）
    disable_fabric = getattr(args, "disable_fabric", False)
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not disable_fabric,
    )

    # 加载 agent 配置（仅用于获取 clip_actions）
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # 创建环境
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 使用 ParkourRslRlVecEnvWrapper 包装环境（提供 get_observations 等接口）
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] 环境创建成功: {args.task}")

    # 运行训练循环
    print("\n[INFO] 开始训练循环...")
    from train_student_rl_finetune import run_training_loop

    try:
        run_training_loop(
            env=env,
            config=config,
            resume_path=args.resume,
        )
    finally:
        # 确保环境被正确关闭
        env.close()
        print("[INFO] 环境已关闭")

    # 返回最终检查点路径
    final_checkpoint = run_dir / f"checkpoint_{config.max_iterations:06d}.pt"
    if final_checkpoint.exists():
        return final_checkpoint
    else:
        return initial_checkpoint


# ============================================================
# 评估模式
# ============================================================

def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """运行评估流程

    Args:
        args: 命令行参数

    Returns:
        评估结果字典
    """
    # 导入评估模块
    from evaluate_student_robustness import (
        EvaluationConfig,
        TestScenario,
        RobustnessMetrics,
        get_scenario_params,
        generate_report,
        save_report,
        print_report_summary,
    )

    print("=" * 60)
    print("Student RL Fine-tuning - 评估模式")
    print("=" * 60)

    # 验证 checkpoint 存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] 任务: {args.task}")
    print(f"[INFO] 环境数量: {args.num_envs}")
    print(f"[INFO] 每场景 episode 数: {args.num_episodes}")
    print(f"[INFO] 设备: {args.device}")

    # 解析场景
    scenarios = None
    if args.scenarios:
        scenarios = []
        for s in args.scenarios:
            try:
                scenario = TestScenario[s.upper()]
                scenarios.append(scenario)
            except KeyError:
                print(f"[WARNING] 未知场景: {s}，跳过")
        if not scenarios:
            scenarios = list(TestScenario)
    else:
        scenarios = list(TestScenario)

    print(f"[INFO] 评估场景: {[s.name for s in scenarios]}")

    # 创建评估配置
    config = EvaluationConfig(
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        max_steps_per_episode=args.max_steps,
        checkpoint_path=str(checkpoint_path),
        dagger_checkpoint_path=args.dagger_checkpoint or "",
        output_path=args.output or "",
        device=args.device,
    )
    config.validate()

    # 评估准备就绪
    print("\n" + "=" * 60)
    print("评估准备就绪")
    print("=" * 60)
    print()
    print("[INFO] Isaac Lab 环境已初始化，可以开始评估。")
    print(f"[INFO] 任务: {args.task}")
    print(f"[INFO] 环境数量: {args.num_envs}")

    # 打印场景参数
    print("测试场景参数:")
    for scenario in scenarios:
        params = get_scenario_params(scenario)
        print(f"\n  {scenario.name}:")
        print(f"    gaussian_std: {params['gaussian_std']}")
        print(f"    salt_pepper_prob: {params['salt_pepper_prob']}")
        print(f"    camera_dropout_prob: {params['camera_dropout_prob']}")
        print(f"    depth_delay_frames: {params['depth_delay_frames']}")

    print()
    print("=" * 60)

    return {"status": "ready", "scenarios": [s.name for s in scenarios]}


# ============================================================
# 主函数
# ============================================================

# 全局变量，用于存储 simulation_app
simulation_app = None


def main() -> None:
    """主函数入口"""
    global simulation_app

    parser = create_parser()
    args, hydra_args = parser.parse_known_args()

    # 如果启用视频录制，需要启用相机
    if hasattr(args, 'video') and args.video:
        args.enable_cameras = True

    # 清理 sys.argv 以便 Hydra 使用
    sys.argv = [sys.argv[0]] + hydra_args

    # 初始化 AppLauncher（启动 Isaac Sim）
    print("[INFO] 初始化 Isaac Lab 环境...")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[INFO] Isaac Lab 环境初始化完成")

    if args.mode is None:
        parser.print_help()
        print("\n请指定运行模式: train 或 evaluate")
        return

    try:
        if args.mode == "train":
            checkpoint_path = run_training(args)

            # 自动评估
            if args.auto_evaluate and checkpoint_path:
                print("\n" + "=" * 60)
                print("自动评估")
                print("=" * 60)

                # 创建评估参数
                eval_args = argparse.Namespace(
                    checkpoint=str(checkpoint_path),
                    task=args.task,
                    num_envs=args.num_envs,
                    num_episodes=args.eval_episodes,
                    max_steps=1000,
                    scenarios=None,
                    dagger_checkpoint=args.dagger_checkpoint,
                    output=str(checkpoint_path.parent / "evaluation_report.json"),
                    device=args.device,
                    headless=True,
                )
                run_evaluation(eval_args)

        elif args.mode == "evaluate":
            run_evaluation(args)

    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] 配置错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================
# 帮助信息
# ============================================================

def print_usage_info() -> None:
    """打印使用信息"""
    print("""
Student RL Fine-tuning 统一启动脚本
====================================

此脚本整合了 RL Fine-tuning 的所有组件，提供统一的训练和评估接口。
已集成 Isaac Lab 环境设置，可以直接通过 python 命令运行，无需 ./isaaclab.sh 包装。

组件概览:
---------
1. ValueHead (Phase 1)
   - 位置: parkour_tasks/.../modules/actionheads/value_head.py
   - 功能: 为 Student Policy 添加价值估计头

2. StudentActorCritic (Phase 2)
   - 位置: scripts/rsl_rl/modules/student_actor_critic.py
   - 功能: 包装 Student Policy，添加 PPO 所需接口

3. PPOStudent (Phase 2)
   - 位置: scripts/rsl_rl/modules/ppo_student.py
   - 功能: 适配 Student Policy 的 PPO 算法

4. Domain Randomization (Phase 3)
   - 位置: parkour_isaaclab/envs/mdp/domain_randomization.py
   - 功能: 深度图像增强和课程学习

5. 训练脚本 (Phase 5)
   - 位置: scripts/rsl_rl/train_student_rl_finetune.py
   - 功能: 完整训练循环

6. 评估脚本 (Phase 6)
   - 位置: scripts/rsl_rl/evaluate_student_robustness.py
   - 功能: 鲁棒性评估

使用示例:
---------
# 基本训练（直接运行，无需 ./isaaclab.sh）
python run_student_rl_finetune.py train \\
    --dagger_checkpoint logs/dagger/best.pt \\
    --headless

# 带 Domain Randomization 课程学习的训练
python run_student_rl_finetune.py train \\
    --dagger_checkpoint logs/dagger/best.pt \\
    --num_envs 512 \\
    --max_iterations 20000 \\
    --headless

# 冻结编码器的训练
python run_student_rl_finetune.py train \\
    --dagger_checkpoint logs/dagger/best.pt \\
    --freeze_encoders \\
    --freeze_fusion \\
    --headless

# 评估所有场景
python run_student_rl_finetune.py evaluate \\
    --checkpoint logs/finetune/final.pt \\
    --headless

# 评估特定场景
python run_student_rl_finetune.py evaluate \\
    --checkpoint logs/finetune/final.pt \\
    --scenarios CLEAN HIGH_NOISE CAMERA_DROPOUT \\
    --headless

# 训练后自动评估
python run_student_rl_finetune.py train \\
    --dagger_checkpoint logs/dagger/best.pt \\
    --auto_evaluate \\
    --eval_episodes 50 \\
    --headless

相关文档:
---------
- 设计文档: docs/plans/2026-02-02-rl-finetuning-design.md
- 架构文档: docs/architecture/2026-02-02-rl-finetuning-architecture.md
- Phase 5 总结: docs/phases/PHASE5_SUMMARY.md
- Phase 6 总结: docs/phases/PHASE6_SUMMARY.md
""")


if __name__ == "__main__":
    # 如果没有参数，打印使用信息
    if len(sys.argv) == 1:
        print_usage_info()
        print("\n使用 --help 查看完整参数列表")
    else:
        main()
        # 关闭 Isaac Sim
        if simulation_app is not None:
            simulation_app.close()
