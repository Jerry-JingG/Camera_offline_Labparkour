"""
Domain Randomization 使用示例

演示如何使用域随机化模块增强深度图像
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from parkour_isaaclab.envs.mdp.domain_randomization import (
    DepthNoiseAugmentation,
    LatencySimulation,
    LightingAugmentation,
    DomainRandCurriculum,
    CameraDropoutManagerWithCurriculum,
)


def create_sample_depth_image():
    """创建示例深度图像"""
    # 创建一个简单的深度图像：中心近，边缘远
    h, w = 58, 87
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    depth = torch.sqrt(x**2 + y**2)
    # 归一化到 [-0.5, 0.5] 范围
    depth = (depth - depth.min()) / (depth.max() - depth.min()) - 0.5
    return depth.unsqueeze(0)  # [1, H, W]


def visualize_augmentations():
    """可视化各种增强效果"""
    # 创建原始深度图像
    original_depth = create_sample_depth_image()

    # 配置
    cfg = type('Config', (), {
        'gaussian_std': 0.03,
        'salt_pepper_prob': 0.02,
        'missing_pixel_prob': 0.01,
        'quantization_levels': 64,
        'scale_range': (0.95, 1.05),
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2),
    })()

    # 创建增强器
    noise_aug = DepthNoiseAugmentation(cfg)
    lighting_aug = LightingAugmentation(cfg)

    # 应用增强
    noisy_depth = noise_aug.apply(original_depth.clone())
    lit_depth = lighting_aug.apply(original_depth.clone())
    combined_depth = lighting_aug.apply(noise_aug.apply(original_depth.clone()))

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    images = [
        (original_depth[0].numpy(), "原始深度图像"),
        (noisy_depth[0].numpy(), "添加噪声"),
        (lit_depth[0].numpy(), "光照调整"),
        (combined_depth[0].numpy(), "组合增强"),
    ]

    for ax, (img, title) in zip(axes.flat, images):
        im = ax.imshow(img, cmap='viridis')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('domain_randomization_examples.png', dpi=150)
    print("✓ 可视化已保存到 domain_randomization_examples.png")


def demonstrate_curriculum():
    """演示课程学习"""
    curriculum = DomainRandCurriculum()

    iterations = [0, 500, 1000, 2000, 3000, 4000, 5000, 7000]
    params_list = [curriculum.get_params(it) for it in iterations]

    # 绘制课程学习曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics = ['noise_std', 'salt_pepper', 'dropout', 'latency']
    titles = ['高斯噪声标准差', '椒盐噪声概率', '相机丢失概率', '延迟帧数']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        values = [p[metric] for p in params_list]
        ax.plot(iterations, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('训练迭代次数', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title}的课程学习', fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('curriculum_learning.png', dpi=150)
    print("✓ 课程学习曲线已保存到 curriculum_learning.png")


def demonstrate_latency():
    """演示延迟模拟"""
    latency_sim = LatencySimulation(depth_delay_frames=3)

    # 创建一系列不同的帧
    frames = []
    for i in range(8):
        frame = torch.ones(1, 58, 87) * (i / 7.0) - 0.5
        frames.append(frame)

    # 应用延迟
    delayed_frames = []
    for frame in frames:
        delayed = latency_sim.apply(frame)
        delayed_frames.append(delayed)

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (ax_orig, ax_delayed) in enumerate(zip(axes[0], axes[1])):
        if i < len(frames):
            ax_orig.imshow(frames[i][0].numpy(), cmap='gray', vmin=-0.5, vmax=0.5)
            ax_orig.set_title(f'原始帧 {i}', fontsize=12)
            ax_orig.axis('off')

            ax_delayed.imshow(delayed_frames[i][0].numpy(), cmap='gray', vmin=-0.5, vmax=0.5)
            ax_delayed.set_title(f'延迟帧 {i}', fontsize=12)
            ax_delayed.axis('off')

    plt.suptitle('延迟模拟（延迟 3 帧）', fontsize=16)
    plt.tight_layout()
    plt.savefig('latency_simulation.png', dpi=150)
    print("✓ 延迟模拟已保存到 latency_simulation.png")


def demonstrate_camera_dropout():
    """演示相机丢失"""
    manager = CameraDropoutManagerWithCurriculum(
        num_envs=16,
        device=torch.device('cpu'),
        enable_curriculum=False
    )

    # 手动设置一些环境为离线状态
    manager.offline_state[::2] = True  # 偶数索引离线
    manager.switching_countdown[:] = 10000  # 防止状态切换

    # 创建深度图像
    depth = create_sample_depth_image().repeat(16, 1, 1)

    # 应用丢失
    result = manager.update(depth)

    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < 16:
            ax.imshow(result[i].numpy(), cmap='viridis')
            status = "离线" if manager.offline_state[i] else "在线"
            ax.set_title(f'环境 {i} ({status})', fontsize=10)
            ax.axis('off')

    plt.suptitle('相机丢失模拟（偶数环境离线）', fontsize=16)
    plt.tight_layout()
    plt.savefig('camera_dropout.png', dpi=150)
    print("✓ 相机丢失已保存到 camera_dropout.png")


def main():
    """运行所有示例"""
    print("="*60)
    print("Domain Randomization 使用示例")
    print("="*60)

    print("\n1. 生成增强可视化...")
    visualize_augmentations()

    print("\n2. 生成课程学习曲线...")
    demonstrate_curriculum()

    print("\n3. 生成延迟模拟示例...")
    demonstrate_latency()

    print("\n4. 生成相机丢失示例...")
    demonstrate_camera_dropout()

    print("\n" + "="*60)
    print("所有示例已生成完成！")
    print("="*60)


if __name__ == "__main__":
    main()
