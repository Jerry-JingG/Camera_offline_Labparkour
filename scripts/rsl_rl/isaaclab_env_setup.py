#!/usr/bin/env python3
"""
Isaac Lab 环境设置模块

提供在不使用 ./isaaclab.sh 包装的情况下运行 Isaac Lab 脚本所需的环境设置。

主要功能：
1. 自动检测并设置 ISAACLAB_PATH 环境变量
2. 配置 Python 路径以包含 Isaac Lab 源代码
3. 提供 AppLauncher 参数创建辅助函数

使用方法：
    # 在脚本开头导入并调用
    from isaaclab_env_setup import setup_isaaclab_env, create_app_launcher_args
    setup_isaaclab_env()

    # 然后可以正常导入 Isaac Lab 模块
    from isaaclab.app import AppLauncher
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def find_isaaclab_path() -> Path:
    """查找 Isaac Lab 安装路径

    按以下顺序查找：
    1. ISAACLAB_PATH 环境变量
    2. 当前项目的父目录（假设项目在 IsaacLab 目录下）
    3. 常见安装位置

    Returns:
        Isaac Lab 根目录路径

    Raises:
        RuntimeError: 如果无法找到 Isaac Lab 目录
    """
    # 1. 检查环境变量
    if "ISAACLAB_PATH" in os.environ:
        path = Path(os.environ["ISAACLAB_PATH"])
        if path.exists() and (path / "isaaclab.sh").exists():
            return path

    # 2. 从当前脚本位置推断
    # 假设脚本在 IsaacLab/Camera_offline_Labparkour/scripts/rsl_rl/ 下
    current_file = Path(__file__).resolve()

    # 向上查找包含 isaaclab.sh 的目录
    search_path = current_file.parent
    for _ in range(10):  # 最多向上查找 10 层
        if (search_path / "isaaclab.sh").exists():
            return search_path
        search_path = search_path.parent

    # 3. 检查常见安装位置
    common_paths = [
        Path.home() / "IsaacLab",
        Path("/opt/IsaacLab"),
        Path("/home/droplet/IsaacLab"),
    ]

    for path in common_paths:
        if path.exists() and (path / "isaaclab.sh").exists():
            return path

    raise RuntimeError(
        "无法找到 Isaac Lab 目录。请设置 ISAACLAB_PATH 环境变量，"
        "或确保脚本位于 Isaac Lab 项目目录下。"
    )


def setup_isaaclab_env() -> None:
    """设置 Isaac Lab 环境

    执行以下操作：
    1. 设置 ISAACLAB_PATH 环境变量
    2. 将 Isaac Lab source 目录添加到 Python 路径
    3. 设置其他必要的环境变量
    """
    # 如果设置了跳过标志，直接返回
    if os.environ.get("ISAACLAB_SKIP_INIT") == "1":
        return

    # 查找 Isaac Lab 路径
    isaaclab_path = find_isaaclab_path()

    # 设置环境变量
    os.environ["ISAACLAB_PATH"] = str(isaaclab_path)

    # 添加 source 目录到 Python 路径
    source_path = isaaclab_path / "source"
    if source_path.exists():
        # 添加 source 目录下的所有子目录
        for subdir in source_path.iterdir():
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                if str(subdir) not in sys.path:
                    sys.path.insert(0, str(subdir))

        # 添加 source 目录本身
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))

    # 设置 RESOURCE_NAME（用于 Isaac Sim 图标显示）
    os.environ.setdefault("RESOURCE_NAME", "IsaacSim")


def create_app_launcher_args(
    headless: bool = False,
    num_envs: Optional[int] = None,
    device: Optional[str] = None,
    enable_cameras: bool = False,
    experience: Optional[str] = None,
) -> argparse.Namespace:
    """创建 AppLauncher 参数

    Args:
        headless: 是否无头模式运行
        num_envs: 环境数量
        device: 设备（如 "cuda:0"）
        enable_cameras: 是否启用相机
        experience: Isaac Sim experience 文件路径

    Returns:
        argparse.Namespace: AppLauncher 参数对象
    """
    args = argparse.Namespace()
    args.headless = headless
    args.enable_cameras = enable_cameras
    args.experience = experience

    if device is not None:
        args.device = device

    if num_envs is not None:
        args.num_envs = num_envs

    return args


def get_isaaclab_python_path() -> str:
    """获取 Isaac Lab 推荐的 Python 解释器路径

    Returns:
        Python 解释器路径
    """
    # 如果在 conda 环境中，使用 conda 的 Python
    if "CONDA_PREFIX" in os.environ:
        conda_python = Path(os.environ["CONDA_PREFIX"]) / "bin" / "python"
        if conda_python.exists():
            return str(conda_python)

    # 否则使用当前 Python
    return sys.executable


# 模块级别的环境设置（导入时自动执行）
# 注意：这允许通过简单的 import 语句设置环境
_env_setup_done = False


def ensure_env_setup() -> None:
    """确保环境已设置（幂等操作）"""
    global _env_setup_done
    if not _env_setup_done:
        setup_isaaclab_env()
        _env_setup_done = True
