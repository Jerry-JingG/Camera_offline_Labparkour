"""
Wandb 异步日志工具

使用队列 + 后台线程实现非阻塞的 wandb 日志记录，
解决 Isaac Lab 多线程环境中 wandb.log() 阻塞训练的问题。

核心特性：
- 主训练线程将日志数据放入队列，立即返回（非阻塞）
- 后台线程从队列中取出数据，调用 wandb.log()
- 线程安全，使用 Python Queue
- 优雅退出，确保所有日志都已上传
"""

import queue
import threading
import time
from typing import Any, Dict, Optional

# 检查 wandb 是否可用
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[wandb] wandb 未安装，日志功能已禁用")


class WandbAsyncLogger:
    """Wandb 异步日志记录器

    使用后台线程异步上传日志，避免阻塞主训练循环。
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        resume: Optional[str] = None,
        id: Optional[str] = None,
        queue_maxsize: int = 1000,
    ):
        """初始化异步日志记录器

        Args:
            project: wandb 项目名称
            entity: wandb 实体名称（用户名或团队名）
            name: 运行名称
            config: 配置字典
            tags: 标签列表
            resume: 恢复模式 ("allow", "must", "never")
            id: 运行 ID（用于恢复）
            queue_maxsize: 日志队列最大长度
        """
        self.enabled = False
        self.run = None
        self._log_queue = queue.Queue(maxsize=queue_maxsize)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._error_count = 0
        self._max_errors = 10  # 最大连续错误次数

        if not WANDB_AVAILABLE:
            print("[wandb] wandb 未安装，日志功能已禁用")
            return

        try:
            # 初始化 wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                resume=resume,
                id=id,
            )
            self.enabled = True
            print(f"[wandb] 初始化成功: {self.run.url}")

            # 启动后台日志线程
            self._worker_thread = threading.Thread(
                target=self._log_worker,
                daemon=True,
                name="wandb-async-logger"
            )
            self._worker_thread.start()
            print("[wandb] 后台日志线程已启动")

        except Exception as e:
            print(f"[wandb] 初始化失败: {e}")
            self.enabled = False

    def _log_worker(self):
        """后台日志工作线程

        从队列中取出日志数据并上传到 wandb。
        """
        print("[wandb] 日志工作线程开始运行")

        while not self._stop_event.is_set():
            try:
                # 从队列中获取日志数据（超时 0.5 秒）
                try:
                    log_data = self._log_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # 上传日志到 wandb
                metrics = log_data["metrics"]
                step = log_data["step"]

                try:
                    wandb.log(metrics, step=step)
                    self._error_count = 0  # 重置错误计数
                except Exception as e:
                    self._error_count += 1
                    print(f"[wandb] 日志上传失败 (错误 {self._error_count}/{self._max_errors}): {e}")

                    # 如果连续错误过多，禁用日志
                    if self._error_count >= self._max_errors:
                        print(f"[wandb] 连续错误过多，禁用日志功能")
                        self.enabled = False
                        break

                finally:
                    self._log_queue.task_done()

            except Exception as e:
                print(f"[wandb] 日志工作线程异常: {e}")

        print("[wandb] 日志工作线程已停止")

    def log(self, metrics: Dict[str, Any], step: int) -> bool:
        """记录指标（非阻塞）

        将日志数据放入队列，立即返回。后台线程会异步上传。

        Args:
            metrics: 指标字典
            step: 步数

        Returns:
            是否成功放入队列
        """
        if not self.enabled or self.run is None:
            return False

        try:
            # 非阻塞放入队列
            log_data = {
                "metrics": metrics,
                "step": step,
            }
            self._log_queue.put_nowait(log_data)
            return True

        except queue.Full:
            print(f"[wandb] 警告: 日志队列已满，跳过 step {step}")
            return False

        except Exception as e:
            print(f"[wandb] 日志入队失败: {e}")
            return False

    def finish(self, timeout: float = 30.0):
        """结束日志记录，等待所有日志上传完成

        Args:
            timeout: 最大等待时间（秒）
        """
        if not self.enabled:
            return

        print(f"[wandb] 等待日志队列清空（剩余 {self._log_queue.qsize()} 条）...")

        # 等待队列清空
        start_time = time.time()
        while not self._log_queue.empty():
            if time.time() - start_time > timeout:
                print(f"[wandb] 警告: 等待超时，仍有 {self._log_queue.qsize()} 条日志未上传")
                break
            time.sleep(0.5)

        # 停止后台线程
        print("[wandb] 停止后台日志线程...")
        self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        # 关闭 wandb
        if self.run:
            print("[wandb] 关闭 wandb 运行...")
            wandb.finish()
            print(f"[wandb] 运行已结束: {self.run.url}")

    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return self._log_queue.qsize()

    def is_alive(self) -> bool:
        """检查后台线程是否存活"""
        return self._worker_thread and self._worker_thread.is_alive()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.finish()
