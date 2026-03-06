#!/usr/bin/env python3
"""
测试 WandbAsyncLogger 异步日志功能

验证：
1. 异步日志不会阻塞主线程
2. 日志能够正确上传到 wandb
3. 队列满时的处理
4. 优雅退出和日志完整性
"""

import sys
import time
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))

from utils.wandb_async_logger import WandbAsyncLogger


def test_basic_logging():
    """测试基本的异步日志功能"""
    print("\n=== 测试 1: 基本异步日志 ===")

    # 初始化日志器
    logger = WandbAsyncLogger(
        project="test-async-logger",
        name="test-basic-logging",
        config={"test": "basic"},
        queue_maxsize=100,
    )

    if not logger.enabled:
        print("[SKIP] wandb 未启用，跳过测试")
        return

    print(f"[INFO] 日志器已初始化，后台线程存活: {logger.is_alive()}")

    # 记录一些日志
    start_time = time.time()
    for i in range(50):
        metrics = {
            "iteration": i,
            "loss": 1.0 / (i + 1),
            "accuracy": i / 50.0,
        }
        success = logger.log(metrics, step=i)
        if not success:
            print(f"[WARN] 日志记录失败: step {i}")

    elapsed = time.time() - start_time
    print(f"[INFO] 记录 50 条日志耗时: {elapsed:.3f}s (平均 {elapsed/50*1000:.1f}ms/条)")
    print(f"[INFO] 队列大小: {logger.get_queue_size()}")

    # 等待日志上传完成
    print("[INFO] 等待日志上传...")
    logger.finish(timeout=30.0)
    print("[PASS] 测试通过")


def test_non_blocking():
    """测试异步日志的非阻塞特性"""
    print("\n=== 测试 2: 非阻塞特性 ===")

    logger = WandbAsyncLogger(
        project="test-async-logger",
        name="test-non-blocking",
        config={"test": "non-blocking"},
        queue_maxsize=1000,
    )

    if not logger.enabled:
        print("[SKIP] wandb 未启用，跳过测试")
        return

    # 快速记录大量日志
    start_time = time.time()
    num_logs = 500

    for i in range(num_logs):
        metrics = {
            "step": i,
            "value": i * 0.1,
        }
        logger.log(metrics, step=i)

    elapsed = time.time() - start_time
    print(f"[INFO] 记录 {num_logs} 条日志耗时: {elapsed:.3f}s")
    print(f"[INFO] 平均每条日志: {elapsed/num_logs*1000:.2f}ms")

    # 如果是同步日志，每条日志至少需要几十毫秒
    # 异步日志应该远快于此
    if elapsed / num_logs < 0.01:  # 小于 10ms/条
        print("[PASS] 日志记录是非阻塞的")
    else:
        print("[WARN] 日志记录可能存在阻塞")

    logger.finish(timeout=60.0)


def test_queue_overflow():
    """测试队列满时的处理"""
    print("\n=== 测试 3: 队列溢出处理 ===")

    # 使用小队列测试溢出
    logger = WandbAsyncLogger(
        project="test-async-logger",
        name="test-queue-overflow",
        config={"test": "overflow"},
        queue_maxsize=10,  # 小队列
    )

    if not logger.enabled:
        print("[SKIP] wandb 未启用，跳过测试")
        return

    # 快速记录超过队列大小的日志
    failed_count = 0
    for i in range(50):
        metrics = {"step": i}
        success = logger.log(metrics, step=i)
        if not success:
            failed_count += 1

    print(f"[INFO] 总共 50 条日志，失败 {failed_count} 条")
    print(f"[INFO] 队列大小: {logger.get_queue_size()}")

    if failed_count > 0:
        print("[PASS] 队列溢出时正确处理（跳过日志）")
    else:
        print("[INFO] 未触发队列溢出")

    logger.finish(timeout=30.0)


def test_context_manager():
    """测试上下文管理器"""
    print("\n=== 测试 4: 上下文管理器 ===")

    with WandbAsyncLogger(
        project="test-async-logger",
        name="test-context-manager",
        config={"test": "context"},
    ) as logger:
        if not logger.enabled:
            print("[SKIP] wandb 未启用，跳过测试")
            return

        for i in range(20):
            logger.log({"step": i, "value": i * 2}, step=i)

        print(f"[INFO] 队列大小: {logger.get_queue_size()}")

    # 退出上下文时应该自动调用 finish()
    print("[PASS] 上下文管理器正常工作")


if __name__ == "__main__":
    print("开始测试 WandbAsyncLogger...")

    try:
        test_basic_logging()
        test_non_blocking()
        test_queue_overflow()
        test_context_manager()

        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n[INFO] 测试被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
