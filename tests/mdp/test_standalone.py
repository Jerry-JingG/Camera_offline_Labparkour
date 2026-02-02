"""
独立测试脚本 - 不依赖 Isaac Lab

直接测试 domain_randomization 模块的功能
"""

import torch
import sys
from pathlib import Path

# 直接导入模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 直接导入实现
exec(open('/home/droplet/IsaacLab/Camera_offline_Labparkour/parkour_isaaclab/envs/mdp/domain_randomization.py').read())


def test_gaussian_noise():
    """测试高斯噪声"""
    print("测试: 高斯噪声添加...")

    cfg = type('Config', (), {
        'gaussian_std': 0.02,
        'salt_pepper_prob': 0.0,
        'missing_pixel_prob': 0.0,
        'quantization_levels': 0,
        'scale_range': (1.0, 1.0),
    })()

    aug = DepthNoiseAugmentation(cfg)
    depth = torch.ones(4, 58, 87) * 0.5
    augmented = aug.apply(depth)

    # 验证形状
    assert augmented.shape == depth.shape, "形状不匹配"

    # 验证添加了噪声
    assert not torch.allclose(augmented, depth), "未添加噪声"

    # 验证噪声范围（使用更宽松的阈值，因为是随机的）
    diff = (augmented - depth).abs()
    assert diff.max() < 0.02 * 5, f"噪声超出范围: {diff.max()}"  # 使用 5-sigma 更安全

    print("✓ 高斯噪声测试通过")


def test_salt_pepper_noise():
    """测试椒盐噪声"""
    print("测试: 椒盐噪声...")

    cfg = type('Config', (), {
        'gaussian_std': 0.0,
        'salt_pepper_prob': 0.1,
        'missing_pixel_prob': 0.0,
        'quantization_levels': 0,
        'scale_range': (1.0, 1.0),
    })()

    aug = DepthNoiseAugmentation(cfg)
    depth = torch.ones(100, 100, 100) * 0.5
    augmented = aug.apply(depth)

    # 验证有极值像素
    has_extremes = (augmented == augmented.min()).sum() > 0 or (augmented == augmented.max()).sum() > 0
    assert has_extremes, "未添加椒盐噪声"

    print("✓ 椒盐噪声测试通过")


def test_missing_pixels():
    """测试缺失像素"""
    print("测试: 缺失像素...")

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

    # 验证有缺失像素
    assert (augmented == -0.5).sum() > 0, "未添加缺失像素"

    print("✓ 缺失像素测试通过")


def test_depth_quantization():
    """测试深度量化"""
    print("测试: 深度量化...")

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

    # 验证量化级别
    unique_values = torch.unique(augmented)
    assert len(unique_values) <= 256, f"量化级别过多: {len(unique_values)}"

    print("✓ 深度量化测试通过")


def test_scale_variation():
    """测试尺度变化"""
    print("测试: 尺度变化...")

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

    # 验证尺度范围
    scale = augmented.mean() / depth.mean()
    assert 0.95 <= scale <= 1.05, f"尺度超出范围: {scale}"

    print("✓ 尺度变化测试通过")


def test_frame_delay():
    """测试帧延迟"""
    print("测试: 帧延迟...")

    latency_sim = LatencySimulation(depth_delay_frames=3)

    frames = [torch.ones(4, 58, 87) * i for i in range(5)]
    results = []
    for frame in frames:
        result = latency_sim.apply(frame)
        results.append(result)

    # 验证延迟
    assert torch.allclose(results[3], frames[0]), "延迟不正确"
    assert torch.allclose(results[4], frames[1]), "延迟不正确"

    print("✓ 帧延迟测试通过")


def test_random_frame_drop():
    """测试随机丢帧"""
    print("测试: 随机丢帧...")

    latency_sim = LatencySimulation(depth_delay_frames=1, drop_prob=0.5)

    frame1 = torch.ones(4, 58, 87) * 1.0
    frame2 = torch.ones(4, 58, 87) * 2.0

    results = []
    for _ in range(100):
        latency_sim.reset()
        latency_sim.apply(frame1)
        result = latency_sim.apply(frame2)
        results.append(result.mean().item())

    # 验证有不同结果
    unique_results = len(set(results))
    assert unique_results > 1, "未实现随机丢帧"

    print("✓ 随机丢帧测试通过")


def test_brightness_adjustment():
    """测试亮度调整"""
    print("测试: 亮度调整...")

    cfg = type('Config', (), {
        'brightness_range': (0.8, 1.2),
        'contrast_range': (1.0, 1.0),
    })()

    aug = LightingAugmentation(cfg)
    depth = torch.ones(4, 58, 87) * 0.5
    augmented = aug.apply(depth)

    # 验证亮度范围
    brightness_factor = augmented.mean() / depth.mean()
    assert 0.8 <= brightness_factor <= 1.2, f"亮度超出范围: {brightness_factor}"

    print("✓ 亮度调整测试通过")


def test_contrast_adjustment():
    """测试对比度调整"""
    print("测试: 对比度调整...")

    cfg = type('Config', (), {
        'brightness_range': (1.0, 1.0),
        'contrast_range': (0.8, 1.2),
    })()

    aug = LightingAugmentation(cfg)
    depth = torch.linspace(0, 1, 4 * 58 * 87).reshape(4, 58, 87)
    augmented = aug.apply(depth)

    # 验证对比度调整
    original_std = depth.std()
    augmented_std = augmented.std()
    contrast_factor = augmented_std / original_std
    assert 0.8 <= contrast_factor <= 1.2, f"对比度超出范围: {contrast_factor}"

    print("✓ 对比度调整测试通过")


def test_curriculum_progression():
    """测试课程学习递增"""
    print("测试: 课程学习递增...")

    curriculum = DomainRandCurriculum()

    params_0 = curriculum.get_params(0)
    params_1000 = curriculum.get_params(1000)
    params_5000 = curriculum.get_params(5000)

    # 验证递增
    assert params_0['noise_std'] < params_1000['noise_std'], "噪声未递增"
    assert params_1000['noise_std'] <= params_5000['noise_std'], "噪声未递增"
    assert params_0['salt_pepper'] < params_1000['salt_pepper'], "椒盐未递增"
    assert params_0['dropout'] < params_1000['dropout'], "丢失未递增"

    print("✓ 课程学习递增测试通过")


def test_linear_interpolation():
    """测试线性插值"""
    print("测试: 线性插值...")

    curriculum = DomainRandCurriculum()

    params_500 = curriculum.get_params(500)

    # 验证中间值
    expected_noise = (0.01 + 0.02) / 2
    assert abs(params_500['noise_std'] - expected_noise) < 0.001, "插值不正确"

    print("✓ 线性插值测试通过")


def test_final_stage_params():
    """测试最终阶段参数"""
    print("测试: 最终阶段参数...")

    curriculum = DomainRandCurriculum()

    params_5000 = curriculum.get_params(5000)
    params_10000 = curriculum.get_params(10000)

    # 验证参数相同
    assert params_5000 == params_10000, "最终参数不一致"

    print("✓ 最终阶段参数测试通过")


def test_camera_dropout_curriculum():
    """测试相机丢失课程学习"""
    print("测试: 相机丢失课程学习...")

    manager = CameraDropoutManagerWithCurriculum(
        num_envs=10,
        device=torch.device('cpu'),
        enable_curriculum=True
    )

    # 测试不同迭代的概率
    manager.update_curriculum(0)
    prob_0 = manager.current_prob
    assert abs(prob_0 - 0.1) < 0.01, f"初始概率不正确: {prob_0}"

    manager.update_curriculum(2000)
    prob_2000 = manager.current_prob
    assert 0.1 < prob_2000 < 0.2, f"中间概率不正确: {prob_2000}"

    manager.update_curriculum(6000)
    prob_6000 = manager.current_prob
    assert abs(prob_6000 - 0.5) < 0.01, f"最终概率不正确: {prob_6000}"

    print("✓ 相机丢失课程学习测试通过")


def test_camera_dropout_apply():
    """测试相机丢失应用"""
    print("测试: 相机丢失应用...")

    manager = CameraDropoutManagerWithCurriculum(
        num_envs=100,
        device=torch.device('cpu'),
        enable_curriculum=False
    )

    # 强制一些环境离线，并设置很大的倒计时以防止状态切换
    manager.offline_state[:50] = True
    manager.offline_state[50:] = False
    manager.switching_countdown[:] = 10000  # 防止状态切换

    # 创建测试深度图像
    depth = torch.ones(100, 58, 87) * 0.5

    # 应用丢失
    result = manager.update(depth)

    # 验证离线环境被设置为 -0.5
    assert torch.allclose(result[:50], torch.ones(50, 58, 87) * (-0.5)), "离线环境未正确设置"
    # 注意：在线环境应该保持原值
    assert torch.allclose(result[50:], torch.ones(50, 58, 87) * 0.5), "在线环境被错误修改"

    print("✓ 相机丢失应用测试通过")



def main():
    """运行所有测试"""
    print("="*60)
    print("Domain Randomization 模块测试")
    print("="*60)

    tests = [
        test_gaussian_noise,
        test_salt_pepper_noise,
        test_missing_pixels,
        test_depth_quantization,
        test_scale_variation,
        test_frame_delay,
        test_random_frame_drop,
        test_brightness_adjustment,
        test_contrast_adjustment,
        test_curriculum_progression,
        test_linear_interpolation,
        test_final_stage_params,
        test_camera_dropout_curriculum,
        test_camera_dropout_apply,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} 失败: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"测试总结: {passed} 通过, {failed} 失败")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
