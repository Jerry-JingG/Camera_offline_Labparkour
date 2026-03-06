# -*- coding: utf-8 -*-
"""
TensorBoard 集成测试

测试 TensorBoard 日志与训练脚本的集成。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# 确保项目路径可导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))


class TestTensorBoardIntegration(unittest.TestCase):
    """测试 TensorBoard 与训练脚本的集成"""

    def test_init_tensorboard_creates_logger(self):
        """测试 init_tensorboard 创建日志器"""
        from train_student_rl_finetune import (
            TrainingConfig,
            init_tensorboard,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(use_tensorboard=True)
            checkpoint_meta = {"num_prop": 53, "action_dim": 12}
            run_dir = Path(tmpdir)

            logger = init_tensorboard(
                config=config,
                checkpoint_meta=checkpoint_meta,
                run_dir=run_dir,
            )

            self.assertIsNotNone(logger)
            self.assertTrue(logger.enabled)

            # 检查 TensorBoard 目录是否创建
            tb_dir = run_dir / "tensorboard"
            self.assertTrue(tb_dir.exists())

            # 检查配置文件是否创建
            config_file = tb_dir / "config.json"
            self.assertTrue(config_file.exists())

            logger.finish()

    def test_init_tensorboard_disabled(self):
        """测试禁用 TensorBoard 时返回 None"""
        from train_student_rl_finetune import (
            TrainingConfig,
            init_tensorboard,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(use_tensorboard=False)
            checkpoint_meta = {"num_prop": 53, "action_dim": 12}
            run_dir = Path(tmpdir)

            logger = init_tensorboard(
                config=config,
                checkpoint_meta=checkpoint_meta,
                run_dir=run_dir,
            )

            self.assertIsNone(logger)

    def test_log_training_metrics(self):
        """测试记录训练指标"""
        from train_student_rl_finetune import (
            TrainingConfig,
            init_tensorboard,
            log_training_metrics,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(use_tensorboard=True)
            checkpoint_meta = {"num_prop": 53, "action_dim": 12}
            run_dir = Path(tmpdir)

            logger = init_tensorboard(
                config=config,
                checkpoint_meta=checkpoint_meta,
                run_dir=run_dir,
            )

            # 模拟训练信息
            train_info = {
                "value_loss": 0.5,
                "surrogate_loss": 0.1,
                "entropy": 0.3,
                "kl": 0.01,
                "learning_rate": 1e-4,
            }

            # 记录指标（不应抛出异常）
            log_training_metrics(
                logger=logger,
                iteration=100,
                train_info=train_info,
                domain_rand_params={"noise_std": 0.02},
                episode_stats={"reward": 10.0},
            )

            logger.finish()

    def test_log_training_metrics_with_none_logger(self):
        """测试 logger 为 None 时不抛出异常"""
        from train_student_rl_finetune import log_training_metrics

        train_info = {
            "value_loss": 0.5,
            "surrogate_loss": 0.1,
        }

        # 不应抛出异常
        log_training_metrics(
            logger=None,
            iteration=100,
            train_info=train_info,
        )

    def test_training_config_tensorboard_defaults(self):
        """测试 TrainingConfig 的 TensorBoard 默认值"""
        from train_student_rl_finetune import TrainingConfig

        config = TrainingConfig()

        # 验证默认值
        self.assertTrue(config.use_tensorboard)
        self.assertEqual(config.tensorboard_flush_secs, 10)

        # 验证没有 wandb 配置
        self.assertFalse(hasattr(config, 'use_wandb'))
        self.assertFalse(hasattr(config, 'wandb_project'))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTensorBoardIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
