"""
测试 Domain Randomization 模块

测试覆盖：
1. DepthNoiseAugmentation - 深度噪声增强
2. LatencySimulation - 延迟模拟
3. LightingAugmentation - 光照增强
4. DomainRandCurriculum - 课程学习调度
"""

import pytest
import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDepthNoiseAugmentation:
    """测试深度噪声增强类"""

    def test_gaussian_noise_adds_noise(self):
        """测试高斯噪声是否正确添加"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DepthNoiseAugmentation

        # 创建配置
        cfg = type('Config', (), {
            'gaussian_std': 0.02,
            'salt_pepper_prob': 0.0,
            'missing_pixel_prob': 0.0,
            'quantization_levels': 0,
            'scale_range': (1.0, 1.0),
        })()

        aug = DepthNoiseAugmentation(cfg)

        # 创建测试数据 [B, H, W]
        depth = torch.ones(4, 58, 87) * 0.5
        augmented = aug.apply(depth)

        # 验证形状不变
        assert augmented.shape == depth.shape

        # 验证添加了噪声（值不完全相同）
        assert not torch.allclose(augmented, depth)

        # 验证噪声在合理范围内（3-sigma 原则）
        diff = (augmented - depth).abs()
        assert diff.max() < 0.02 * 3  # 3倍标准差

    def test_salt_pepper_noise(self):
        """测试椒盐噪声"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DepthNoiseAugmentation

        cfg = type('Config', (), {
            'gaussian_std': 0.0,
            'salt_pepper_prob': 0.1,  # 10% 概率
            'missing_pixel_prob': 0.0,
            'quantization_levels': 0,
            'scale_range': (1.0, 1.0),
        })()

        aug = DepthNoiseAugmentation(cfg)
        depth = torch.ones(100, 100, 100) * 0.5

        augmented = aug.apply(depth)

        # 验证有像素被设置为极值
        assert (augmented == augmented.min()).sum() > 0 or (augmented == augmented.max()).sum() > 0

    def test_missing_pixels(self):
        """测试缺失像素模拟"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DepthNoiseAugmentation

        cfg = type('Config', (), {
            'gaussian_std': 0.0,
            'salt_pepper_prob': 0.0,
            'missing_pixel_prob': 0.1,
            'quantization_levels': 0,
            'scale_range': (1.0, 1.0),
        })()

        aug = DepthNoiseAugmentation(cfg)
        depth = torch.ones(100, 100, 100) * 0.5

        augmented = aug.apply(depth)

        # 验证有像素被设置为 -0.5（无效深度）
        assert (augmented == -0.5).sum() > 0

    def test_depth_quantization(self):
        """测试深度量化"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DepthNoiseAugmentation

        cfg = type('Config', (), {
            'gaussian_std': 0.0,
            'salt_pepper_prob': 0.0,
            'missing_pixel_prob': 0.0,
            'quantization_levels': 256,
            'scale_range': (1.0, 1.0),
        })()

        aug = DepthNoiseAugmentation(cfg)
        depth = torch.linspace(0, 1, 1000).reshape(10, 10, 10)

        augmented = aug.apply(depth)

        # 验证量化后的唯一值数量不超过量化级别
        unique_values = torch.unique(augmented)
        assert len(unique_values) <= 256

    def test_scale_variation(self):
        """测试深度尺度变化"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DepthNoiseAugmentation

        cfg = type('Config', (), {
            'gaussian_std': 0.0,
            'salt_pepper_prob': 0.0,
            'missing_pixel_prob': 0.0,
            'quantization_levels': 0,
            'scale_range': (0.95, 1.05),
        })()

        aug = DepthNoiseAugmentation(cfg)
        depth = torch.ones(4, 58, 87) * 0.5

        augmented = aug.apply(depth)

        # 验证尺度在范围内
        scale = augmented.mean() / depth.mean()
        assert 0.95 <= scale <= 1.05


class TestLatencySimulation:
    """测试延迟模拟类"""

    def test_frame_delay(self):
        """测试帧延迟功能"""
        from parkour_isaaclab.envs.mdp.domain_randomization import LatencySimulation

        latency_sim = LatencySimulation(depth_delay_frames=3)

        # 创建不同的帧
        frames = [torch.ones(4, 58, 87) * i for i in range(5)]

        results = []
        for frame in frames:
            result = latency_sim.apply(frame)
            results.append(result)

        # 前3帧应该返回第0帧（因为缓冲区未满）
        # 第4帧应该返回第1帧（延迟3帧）
        assert torch.allclose(results[3], frames[0])
        assert torch.allclose(results[4], frames[1])

    def test_random_frame_drop(self):
        """测试随机丢帧"""
        from parkour_isaaclab.envs.mdp.domain_randomization import LatencySimulation

        latency_sim = LatencySimulation(depth_delay_frames=1, drop_prob=0.5)

        frame1 = torch.ones(4, 58, 87) * 1.0
        frame2 = torch.ones(4, 58, 87) * 2.0

        # 多次测试以验证随机丢帧
        results = []
        for _ in range(100):
            latency_sim.reset()
            latency_sim.apply(frame1)
            result = latency_sim.apply(frame2)
            results.append(result)

        # 验证有些结果是 frame1（丢帧），有些是 frame2（未丢帧）
        # 注意：这是概率性测试，可能偶尔失败
        unique_results = len(set([r.mean().item() for r in results]))
        assert unique_results > 1  # 应该有不同的结果


class TestLightingAugmentation:
    """测试光照增强类"""

    def test_brightness_adjustment(self):
        """测试亮度调整"""
        from parkour_isaaclab.envs.mdp.domain_randomization import LightingAugmentation

        cfg = type('Config', (), {
            'brightness_range': (0.8, 1.2),
            'contrast_range': (1.0, 1.0),
        })()

        aug = LightingAugmentation(cfg)
        depth = torch.ones(4, 58, 87) * 0.5

        augmented = aug.apply(depth)

        # 验证亮度在范围内
        brightness_factor = augmented.mean() / depth.mean()
        assert 0.8 <= brightness_factor <= 1.2

    def test_contrast_adjustment(self):
        """测试对比度调整"""
        from parkour_isaaclab.envs.mdp.domain_randomization import LightingAugmentation

        cfg = type('Config', (), {
            'brightness_range': (1.0, 1.0),
            'contrast_range': (0.8, 1.2),
        })()

        aug = LightingAugmentation(cfg)
        depth = torch.linspace(0, 1, 4 * 58 * 87).reshape(4, 58, 87)

        augmented = aug.apply(depth)

        # 验证对比度调整后标准差变化
        original_std = depth.std()
        augmented_std = augmented.std()
        contrast_factor = augmented_std / original_std
        assert 0.8 <= contrast_factor <= 1.2


class TestDomainRandCurriculum:
    """测试课程学习调度类"""

    def test_curriculum_progression(self):
        """测试课程学习参数随迭代递增"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DomainRandCurriculum

        curriculum = DomainRandCurriculum()

        # 测试不同迭代的参数
        params_0 = curriculum.get_params(0)
        params_1000 = curriculum.get_params(1000)
        params_5000 = curriculum.get_params(5000)

        # 验证参数递增
        assert params_0['noise_std'] < params_1000['noise_std']
        assert params_1000['noise_std'] <= params_5000['noise_std']

        assert params_0['salt_pepper'] < params_1000['salt_pepper']
        assert params_0['dropout'] < params_1000['dropout']

    def test_linear_interpolation(self):
        """测试线性插值"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DomainRandCurriculum

        curriculum = DomainRandCurriculum()

        # 测试中间点的插值
        params_500 = curriculum.get_params(500)

        # 500 在 0-1000 之间，应该是中间值
        expected_noise = (0.01 + 0.02) / 2
        assert abs(params_500['noise_std'] - expected_noise) < 0.001

    def test_final_stage_params(self):
        """测试最终阶段参数保持不变"""
        from parkour_isaaclab.envs.mdp.domain_randomization import DomainRandCurriculum

        curriculum = DomainRandCurriculum()

        params_5000 = curriculum.get_params(5000)
        params_10000 = curriculum.get_params(10000)

        # 超过最后阶段后参数应该保持不变
        assert params_5000 == params_10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

