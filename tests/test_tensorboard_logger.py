# -*- coding: utf-8 -*-
"""
TensorBoardLogger 单元测试

测试 TensorBoard 日志工具的核心功能。
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest import mock

# 确保项目路径可导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))


class TestTensorBoardLoggerInit(unittest.TestCase):
    """测试 TensorBoardLogger 初始化"""

    def test_init_creates_log_directory(self):
        """测试初始化时创建日志目录"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs", "test_run")
            logger = TensorBoardLogger(log_dir=log_dir)

            assert os.path.exists(log_dir)
            logger.finish()

    def test_init_with_existing_directory(self):
        """测试使用已存在的目录初始化"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            assert logger.enabled
            logger.finish()

    def test_init_sets_enabled_flag(self):
        """测试初始化后 enabled 标志为 True"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            assert logger.enabled is True
            logger.finish()


class TestTensorBoardLoggerLog(unittest.TestCase):
    """测试 TensorBoardLogger 日志记录功能"""

    def test_log_single_metric(self):
        """测试记录单个指标"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            result = logger.log({"train/loss": 0.5}, step=1)

            assert result is True
            logger.finish()

    def test_log_multiple_metrics(self):
        """测试记录多个指标"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            metrics = {
                "train/value_loss": 0.1,
                "train/policy_loss": 0.2,
                "train/entropy": 0.3,
            }
            result = logger.log(metrics, step=10)

            assert result is True
            logger.finish()

    def test_log_returns_false_when_disabled(self):
        """测试禁用时返回 False"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.enabled = False

            result = logger.log({"test": 1.0}, step=1)

            assert result is False
            logger.finish()

    def test_log_empty_metrics(self):
        """测试记录空指标字典"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            result = logger.log({}, step=1)

            assert result is True
            logger.finish()


class TestTensorBoardLoggerConfig(unittest.TestCase):
    """测试 TensorBoardLogger 配置记录功能"""

    def test_log_config_dict(self):
        """测试记录配置字典"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            config = {
                "learning_rate": 1e-4,
                "num_envs": 256,
                "max_iterations": 1000,
            }
            # 不应抛出异常
            logger.log_config(config)
            logger.finish()

    def test_log_config_creates_file(self):
        """测试配置记录创建文件"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            config = {"test_param": 123}
            logger.log_config(config)

            # 检查配置文件是否存在
            config_file = os.path.join(tmpdir, "config.json")
            assert os.path.exists(config_file)
            logger.finish()


class TestTensorBoardLoggerContextManager(unittest.TestCase):
    """测试 TensorBoardLogger 上下文管理器"""

    def test_context_manager_enter(self):
        """测试上下文管理器进入"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardLogger(log_dir=tmpdir) as logger:
                assert logger.enabled is True

    def test_context_manager_exit_calls_finish(self):
        """测试上下文管理器退出时调用 finish"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            with mock.patch.object(logger, 'finish') as mock_finish:
                with logger:
                    pass
                mock_finish.assert_called_once()


class TestTensorBoardLoggerFinish(unittest.TestCase):
    """测试 TensorBoardLogger 关闭功能"""

    def test_finish_sets_enabled_false(self):
        """测试 finish 后 enabled 为 False"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.finish()

            assert logger.enabled is False

    def test_finish_can_be_called_multiple_times(self):
        """测试 finish 可以多次调用"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            logger.finish()
            logger.finish()  # 不应抛出异常


class TestTensorBoardLoggerEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_log_with_none_value(self):
        """测试记录 None 值时的处理"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # None 值应该被跳过，不应抛出异常
            result = logger.log({"valid": 1.0, "invalid": None}, step=1)
            assert result is True
            logger.finish()

    def test_log_with_string_value(self):
        """测试记录字符串值时的处理"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # 字符串值应该被跳过，不应抛出异常
            result = logger.log({"valid": 1.0, "string": "test"}, step=1)
            assert result is True
            logger.finish()

    def test_log_with_negative_step(self):
        """测试负数 step"""
        from utils.tensorboard_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # 负数 step 应该正常工作
            result = logger.log({"test": 1.0}, step=-1)
            assert result is True
            logger.finish()


if __name__ == "__main__":
    import unittest
    # 使用 unittest 运行测试，避免 pytest 与 ROS 的冲突
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTensorBoardLoggerInit))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorBoardLoggerLog))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorBoardLoggerConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorBoardLoggerContextManager))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorBoardLoggerFinish))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorBoardLoggerEdgeCases))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
