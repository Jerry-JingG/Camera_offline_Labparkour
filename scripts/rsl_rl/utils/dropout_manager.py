"""image_features的返回值经过了归一化, 范围是(-0.5, 0.5), 所以-0.5才表示深度为0!!!"""

import numpy as np
import torch


class CameraDropoutManager:
    """
    管理相机掉线状态的管理器。
    维护每个环境的：
    1. offline_state: False (在线/正常), True (掉线/黑屏)
    2. switching_countdown: 距离下次状态切换的步数
    3. offline_type: 0 (全黑), 1 (噪声/雪花屏 - 预留), 2 (动态遮掩 - 预留)
    """
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dt: float = 0.02,
        prob_start_offline: float = 0.00,
        online_duration_range: tuple[float, float] = (2.0, 10.0),      # online_duration_range的最小值不应小于dt
        offline_duration_range: tuple[float, float] = (2.0, 10.0)    # offline_duration_range的最小值也不应小于dt
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.prob_start_offline = prob_start_offline

        # 1. 初始化状态 (False: Online, True: Offline)
        self.offline_state = (torch.rand(num_envs, device=device) < prob_start_offline)
        self.online_duration_range = online_duration_range
        self.offline_duration_range = offline_duration_range

        # 2. 初始化倒计时
        self.switching_countdown = torch.zeros(num_envs, device=device, dtype=torch.long)
        # 为初始状态生成第一次倒计时
        self._sample_countdown(torch.ones(num_envs, device=device, dtype=torch.bool))

        # 3. 掉线类型 (目前默认全为 0: Black Screen)
        self.offline_type = torch.zeros(num_envs, device=device, dtype=torch.long)

    def _sample_countdown(self, switching_mask: torch.Tensor):
        if not switching_mask.any():
            return

        # Group A: 刚刚变成 Online 的 (time_up 且 offline_state为False)
        newly_online = switching_mask & (~self.offline_state)
        if newly_online.any():
            low, high = self.online_duration_range
            dur_s = (torch.rand(newly_online.sum(), device=self.device) * (high - low) + low)
            self.switching_countdown[newly_online] = (dur_s / self.dt).long()

        # Group B: 刚刚变成 Offline 的 (time_up 且 offline_state为True)
        newly_offline = switching_mask & self.offline_state
        if newly_offline.any():
            low, high = self.offline_duration_range
            dur_s = (torch.rand(newly_offline.sum(), device=self.device) * (high - low) + low)
            self.switching_countdown[newly_offline] = (dur_s / self.dt).long()

    def update(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        更新状态并应用掉线遮掩。
        Args:
            depth_image: [num_envs, H, W] or [num_envs, 1, H, W]
        Returns:
            modified_depth_image
        """
        # 1. 倒计时递减
        self.switching_countdown -= 1

        # 2. 状态切换
        switching_mask = self.switching_countdown <= 0
        if switching_mask.any():
            self.offline_state[switching_mask] = ~self.offline_state[switching_mask]
            self._sample_countdown(switching_mask)

        # 3. 应用遮掩 (In-place)
        """image_features的返回值经过了归一化, 范围是(-0.5, 0.5), 所以-0.5才表示深度为0!!!"""
        if self.offline_state.any():
            depth_image[self.offline_state] = -0.5

        return depth_image

    def reset_env(self, dones_mask: torch.Tensor):
        """
        Step T: 采取动作，环境返回 obs_{T+1} 和 done_T。
        Start of Loop (T+1):
        检查 done_T。如果为 True, 说明 obs_{T+1} 是一个新 Episode 的初始帧。
        调用 reset_env(done_T)：强制相机恢复在线 (为了新 Episode)。
        调用 update(obs_{T+1})：对这帧新图像应用掉线逻辑。
        """
        if not dones_mask.any():
            return

        offline_state = (torch.rand(self.num_envs, device=self.device) < self.prob_start_offline)
        self.offline_state[dones_mask] = offline_state[dones_mask]

        self._sample_countdown(dones_mask)
