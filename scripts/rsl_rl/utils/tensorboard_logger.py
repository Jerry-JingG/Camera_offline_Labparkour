# -*- coding: utf-8 -*-
"""
TensorBoard 日志工具

提供简洁的 TensorBoard 日志接口，用于 RL Fine-tuning 训练。
替代 wandb 日志，避免网络 I/O 阻塞训练。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# 检查 TensorBoard 是否可用
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[tensorboard] tensorboard 未安装，日志功能已禁用")
    print("[tensorboard] 请运行: pip install tensorboard")


class TensorBoardLogger:
    """TensorBoard 日志记录器

    提供与 WandbAsyncLogger 兼容的接口，用于无缝替换。

    特性：
    - 本地日志，无网络 I/O，不会阻塞训练
    - 简洁的 API，与 wandb 接口兼容
    - 支持上下文管理器
    """

    def __init__(
        self,
        log_dir: str,
        flush_secs: int = 10,
        comment: str = "",
    ):
        """初始化 TensorBoard 日志记录器

        Args:
            log_dir: 日志目录路径
            flush_secs: 刷新间隔（秒）
            comment: 运行注释
        """
        self.log_dir = log_dir
        self.enabled = False
        self._writer: Optional[SummaryWriter] = None

        if not TENSORBOARD_AVAILABLE:
            print("[tensorboard] tensorboard 未安装，日志功能已禁用")
            return

        try:
            # 创建日志目录
            os.makedirs(log_dir, exist_ok=True)

            # 初始化 SummaryWriter
            self._writer = SummaryWriter(
                log_dir=log_dir,
                flush_secs=flush_secs,
                comment=comment,
            )
            self.enabled = True
            print(f"[tensorboard] 初始化成功: {log_dir}")

        except Exception as e:
            print(f"[tensorboard] 初始化失败: {e}")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: int) -> bool:
        """记录指标

        Args:
            metrics: 指标字典，键为指标名称，值为数值
            step: 步数

        Returns:
            是否成功记录
        """
        if not self.enabled or self._writer is None:
            return False

        try:
            for key, value in metrics.items():
                # 跳过非数值类型
                if value is None:
                    continue
                if isinstance(value, str):
                    continue

                # 记录标量
                try:
                    self._writer.add_scalar(key, float(value), step)
                except (TypeError, ValueError):
                    # 无法转换为 float，跳过
                    continue

            return True

        except Exception as e:
            print(f"[tensorboard] 日志记录失败: {e}")
            return False

    def log_config(self, config: Dict[str, Any]) -> None:
        """记录配置

        将配置保存为 JSON 文件，并作为文本记录到 TensorBoard。

        Args:
            config: 配置字典
        """
        if not self.enabled:
            return

        try:
            # 保存配置到 JSON 文件
            config_path = os.path.join(self.log_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, default=str)
            print(f"[tensorboard] 配置已保存: {config_path}")

            # 记录配置到 TensorBoard 文本
            if self._writer is not None:
                config_text = json.dumps(config, indent=2, default=str)
                self._writer.add_text("config", f"```json\n{config_text}\n```", 0)

        except Exception as e:
            print(f"[tensorboard] 配置记录失败: {e}")

    def finish(self) -> None:
        """关闭日志记录器"""
        if self._writer is not None:
            try:
                self._writer.close()
                print("[tensorboard] 日志记录器已关闭")
            except Exception as e:
                print(f"[tensorboard] 关闭时出错: {e}")
            finally:
                self._writer = None

        self.enabled = False

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.finish()
