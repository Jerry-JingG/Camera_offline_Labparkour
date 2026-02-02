"""
相机全黑管理器 - 让相机一直保持全黑状态。
参考 CameraDropoutManager 中的黑屏逻辑实现。

注意: image_features的返回值经过了归一化, 范围是(-0.5, 0.5), 所以-0.5才表示深度为0!!!
"""

import torch


class CameraBlackoutManager:
    """
    相机全黑管理器。
    始终将深度图像设置为全黑状态 (-0.5)。
    """
    
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
    ):
        """
        初始化相机全黑管理器。
        
        Args:
            num_envs: 环境数量
            device: PyTorch 设备
        """
        self.num_envs = num_envs
        self.device = device
        # 所有环境始终处于全黑状态
        self.blackout_state = torch.ones(num_envs, device=device, dtype=torch.bool)
    
    def update(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        将所有深度图像设置为全黑。
        
        Args:
            depth_image: [num_envs, H, W] or [num_envs, 1, H, W]
        Returns:
            modified_depth_image: 全黑的深度图像
        """
        # 应用全黑遮掩 (In-place)
        # image_features的返回值经过了归一化, 范围是(-0.5, 0.5), 所以-0.5才表示深度为0!!!
        depth_image[self.blackout_state] = -0.5
        return depth_image
    
    def reset_env(self, dones_mask: torch.Tensor):
        """
        环境重置时的处理 (对于全黑管理器，无需额外操作)。
        
        Args:
            dones_mask: 需要重置的环境掩码
        """
        # 对于全黑管理器，始终保持全黑状态，无需重置
        pass
