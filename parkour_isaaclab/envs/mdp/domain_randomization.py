"""
Domain Randomization 模块

实现深度图像的域随机化增强，用于提高策略的鲁棒性。

包含以下组件：
1. DepthNoiseAugmentation - 深度噪声增强
2. LatencySimulation - 延迟模拟
3. LightingAugmentation - 光照增强
4. DomainRandCurriculum - 课程学习调度
"""

from __future__ import annotations

import torch
from collections import deque
from typing import Tuple, Dict, Optional


class DepthNoiseAugmentation:
    """
    深度噪声增强类

    模拟真实深度相机的各种噪声：
    - 高斯噪声：传感器噪声
    - 椒盐噪声：死像素
    - 缺失像素：IR 反射失败
    - 深度量化：传感器位深限制
    - 尺度变化：校准误差
    """

    def __init__(self, cfg):
        """
        初始化深度噪声增强

        参数:
            cfg: 配置对象，包含以下属性：
                - gaussian_std: 高斯噪声标准差 (0.01-0.04)
                - salt_pepper_prob: 椒盐噪声概率 (0.005-0.02)
                - missing_pixel_prob: 缺失像素概率 (0.005-0.02)
                - quantization_levels: 量化级别 (0 表示不量化，256-1024)
                - scale_range: 尺度变化范围 (min, max)
        """
        self.gaussian_std = cfg.gaussian_std
        self.salt_pepper_prob = cfg.salt_pepper_prob
        self.missing_pixel_prob = cfg.missing_pixel_prob
        self.quantization_levels = cfg.quantization_levels
        self.scale_range = cfg.scale_range

    def apply(self, depth: torch.Tensor) -> torch.Tensor:
        """
        应用深度噪声增强

        参数:
            depth: 深度图像 [B, H, W] 或 [B, C, H, W]

        返回:
            augmented_depth: 增强后的深度图像，形状与输入相同
        """
        # 创建副本以保持不可变性
        augmented = depth.clone()

        # 1. 高斯噪声
        if self.gaussian_std > 0:
            noise = torch.randn_like(augmented) * self.gaussian_std
            augmented = augmented + noise

        # 2. 椒盐噪声
        if self.salt_pepper_prob > 0:
            mask = torch.rand_like(augmented) < self.salt_pepper_prob
            # 随机选择盐（最大值）或椒（最小值）
            salt_or_pepper = torch.rand_like(augmented) > 0.5
            augmented = torch.where(
                mask & salt_or_pepper,
                torch.ones_like(augmented) * augmented.max(),
                augmented
            )
            augmented = torch.where(
                mask & ~salt_or_pepper,
                torch.ones_like(augmented) * augmented.min(),
                augmented
            )

        # 3. 缺失像素（设置为无效深度 -0.5）
        if self.missing_pixel_prob > 0:
            missing_mask = torch.rand_like(augmented) < self.missing_pixel_prob
            augmented = torch.where(
                missing_mask,
                torch.ones_like(augmented) * (-0.5),
                augmented
            )

        # 4. 深度量化
        if self.quantization_levels > 0:
            # 量化到指定级别
            depth_min = augmented.min()
            depth_max = augmented.max()
            depth_range = depth_max - depth_min
            if depth_range > 0:
                # 归一化到 [0, 1]
                normalized = (augmented - depth_min) / depth_range
                # 量化
                quantized = torch.round(normalized * (self.quantization_levels - 1))
                # 反归一化
                augmented = (quantized / (self.quantization_levels - 1)) * depth_range + depth_min

        # 5. 尺度变化
        if self.scale_range != (1.0, 1.0):
            scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
            augmented = augmented * scale

        return augmented


class LatencySimulation:
    """
    延迟模拟类

    模拟真实硬件的处理延迟：
    - 帧延迟：使用 N 帧前的深度图像
    - 随机丢帧：模拟处理失败
    """

    def __init__(self, depth_delay_frames: int = 3, drop_prob: float = 0.0):
        """
        初始化延迟模拟

        参数:
            depth_delay_frames: 延迟帧数 (1-5)
            drop_prob: 丢帧概率 (0.0-0.2)
        """
        self.depth_delay = depth_delay_frames
        self.drop_prob = drop_prob
        self.depth_buffer = deque(maxlen=depth_delay_frames + 1)
        self.last_valid_depth = None

    def reset(self):
        """重置缓冲区"""
        self.depth_buffer.clear()
        self.last_valid_depth = None

    def apply(self, depth: torch.Tensor) -> torch.Tensor:
        """
        应用延迟模拟

        参数:
            depth: 当前帧深度图像

        返回:
            delayed_depth: 延迟后的深度图像
        """
        # 随机丢帧
        if self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob:
            # 返回上一帧有效深度
            if self.last_valid_depth is not None:
                return self.last_valid_depth.clone()
            else:
                return depth.clone()

        # 添加到缓冲区
        self.depth_buffer.append(depth.clone())

        # 如果缓冲区未满，返回最早的帧
        if len(self.depth_buffer) <= self.depth_delay:
            result = self.depth_buffer[0].clone()
        else:
            # 返回延迟帧
            result = self.depth_buffer[0].clone()

        # 更新最后有效深度
        self.last_valid_depth = result.clone()

        return result


class LightingAugmentation:
    """
    光照增强类

    模拟不同光照条件对深度感知的影响：
    - 亮度调整：环境光变化
    - 对比度调整：光照均匀性
    """

    def __init__(self, cfg):
        """
        初始化光照增强

        参数:
            cfg: 配置对象，包含以下属性：
                - brightness_range: 亮度范围 (min, max)
                - contrast_range: 对比度范围 (min, max)
        """
        self.brightness_range = cfg.brightness_range
        self.contrast_range = cfg.contrast_range

    def apply(self, depth: torch.Tensor) -> torch.Tensor:
        """
        应用光照增强

        参数:
            depth: 深度图像

        返回:
            augmented_depth: 增强后的深度图像
        """
        # 创建副本
        augmented = depth.clone()

        # 1. 亮度调整
        if self.brightness_range != (1.0, 1.0):
            brightness_factor = torch.empty(1).uniform_(
                self.brightness_range[0],
                self.brightness_range[1]
            ).item()
            augmented = augmented * brightness_factor

        # 2. 对比度调整
        if self.contrast_range != (1.0, 1.0):
            contrast_factor = torch.empty(1).uniform_(
                self.contrast_range[0],
                self.contrast_range[1]
            ).item()
            # 对比度调整：(x - mean) * factor + mean
            mean = augmented.mean()
            augmented = (augmented - mean) * contrast_factor + mean

        return augmented


class DomainRandCurriculum:
    """
    域随机化课程学习调度器

    根据训练迭代次数逐步增加增强强度，避免一开始就使用过强的增强导致训练不稳定。

    课程学习阶段：
    - 0-1000: 轻度增强
    - 1000-3000: 中度增强
    - 3000-5000: 重度增强
    - 5000+: 最大增强
    """

    # 课程学习时间表
    # 格式: (迭代次数, 噪声标准差, 椒盐概率, 相机丢失概率, 延迟帧数)
    SCHEDULE = [
        (0,    0.01, 0.005, 0.1, 1),
        (1000, 0.02, 0.01,  0.2, 2),
        (3000, 0.03, 0.015, 0.3, 3),
        (5000, 0.04, 0.02,  0.5, 3),
    ]

    def __init__(self):
        """初始化课程学习调度器"""
        pass

    def get_params(self, iteration: int) -> Dict[str, float]:
        """
        根据当前迭代次数获取增强参数

        参数:
            iteration: 当前训练迭代次数

        返回:
            params: 包含增强参数的字典
        """
        # 查找当前所在的阶段
        for i in range(len(self.SCHEDULE) - 1):
            iter_start, noise_start, salt_start, dropout_start, latency_start = self.SCHEDULE[i]
            iter_end, noise_end, salt_end, dropout_end, latency_end = self.SCHEDULE[i + 1]

            if iter_start <= iteration < iter_end:
                # 线性插值
                alpha = (iteration - iter_start) / (iter_end - iter_start)
                return {
                    'noise_std': self._lerp(noise_start, noise_end, alpha),
                    'salt_pepper': self._lerp(salt_start, salt_end, alpha),
                    'dropout': self._lerp(dropout_start, dropout_end, alpha),
                    'latency': int(self._lerp(latency_start, latency_end, alpha)),
                }

        # 超过最后阶段，返回最终参数
        _, noise, salt, dropout, latency = self.SCHEDULE[-1]
        return {
            'noise_std': noise,
            'salt_pepper': salt,
            'dropout': dropout,
            'latency': latency,
        }

    @staticmethod
    def _lerp(start: float, end: float, alpha: float) -> float:
        """线性插值"""
        return start + (end - start) * alpha


class CameraDropoutManagerWithCurriculum:
    """
    带课程学习的相机丢失管理器

    扩展原有的 CameraDropoutManager，添加课程学习功能，
    使相机丢失概率随训练迭代逐步增加。

    课程学习阶段：
    - 0-1000: 10% 丢失概率
    - 1000-3000: 20% 丢失概率
    - 3000-5000: 30% 丢失概率
    - 5000+: 50% 丢失概率
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dt: float = 0.02,
        initial_prob: float = 0.1,
        online_duration_range: Tuple[float, float] = (2.0, 10.0),
        offline_duration_range: Tuple[float, float] = (1.0, 7.0),
        enable_curriculum: bool = True,
    ):
        """
        初始化带课程学习的相机丢失管理器

        参数:
            num_envs: 环境数量
            device: PyTorch 设备
            dt: 时间步长
            initial_prob: 初始丢失概率
            online_duration_range: 在线持续时间范围（秒）
            offline_duration_range: 离线持续时间范围（秒）
            enable_curriculum: 是否启用课程学习
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.online_duration_range = online_duration_range
        self.offline_duration_range = offline_duration_range
        self.enable_curriculum = enable_curriculum

        # 当前丢失概率（会随课程学习更新）
        self.current_prob = initial_prob

        # 初始化状态
        self.offline_state = (torch.rand(num_envs, device=device) < self.current_prob)
        self.switching_countdown = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._sample_countdown(torch.ones(num_envs, device=device, dtype=torch.bool))

        # 离线类型（0: 全黑）
        self.offline_type = torch.zeros(num_envs, device=device, dtype=torch.long)

    def update_curriculum(self, iteration: int):
        """
        根据训练迭代次数更新丢失概率

        参数:
            iteration: 当前训练迭代次数
        """
        if not self.enable_curriculum:
            return

        # 课程学习时间表
        if iteration < 1000:
            self.current_prob = 0.1
        elif iteration < 3000:
            # 线性插值 1000-3000: 0.1 -> 0.2
            alpha = (iteration - 1000) / (3000 - 1000)
            self.current_prob = 0.1 + (0.2 - 0.1) * alpha
        elif iteration < 5000:
            # 线性插值 3000-5000: 0.2 -> 0.3
            alpha = (iteration - 3000) / (5000 - 3000)
            self.current_prob = 0.2 + (0.3 - 0.2) * alpha
        else:
            # 5000+: 0.5
            self.current_prob = 0.5

    def _sample_countdown(self, switching_mask: torch.Tensor):
        """采样状态切换倒计时"""
        if not switching_mask.any():
            return

        # 刚变成在线的环境
        newly_online = switching_mask & (~self.offline_state)
        if newly_online.any():
            low, high = self.online_duration_range
            dur_s = (torch.rand(newly_online.sum(), device=self.device) * (high - low) + low)
            self.switching_countdown[newly_online] = (dur_s / self.dt).long()

        # 刚变成离线的环境
        newly_offline = switching_mask & self.offline_state
        if newly_offline.any():
            low, high = self.offline_duration_range
            dur_s = (torch.rand(newly_offline.sum(), device=self.device) * (high - low) + low)
            self.switching_countdown[newly_offline] = (dur_s / self.dt).long()

    def update(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        更新状态并应用相机丢失遮掩

        参数:
            depth_image: 深度图像 [num_envs, H, W] 或 [num_envs, C, H, W]

        返回:
            modified_depth_image: 应用遮掩后的深度图像
        """
        # 1. 倒计时递减
        self.switching_countdown -= 1

        # 2. 状态切换
        switching_mask = self.switching_countdown <= 0
        if switching_mask.any():
            self.offline_state[switching_mask] = ~self.offline_state[switching_mask]
            self._sample_countdown(switching_mask)

        # 3. 应用遮掩（深度归一化范围是 (-0.5, 0.5)，-0.5 表示无效深度）
        if self.offline_state.any():
            depth_image[self.offline_state] = -0.5

        return depth_image

    def reset_env(self, dones_mask: torch.Tensor):
        """
        环境重置时的处理

        参数:
            dones_mask: 需要重置的环境掩码
        """
        if not dones_mask.any():
            return

        # 根据当前概率重新采样离线状态
        offline_state = (torch.rand(self.num_envs, device=self.device) < self.current_prob)
        self.offline_state[dones_mask] = offline_state[dones_mask]

        # 重新采样倒计时
        self._sample_countdown(dones_mask)

