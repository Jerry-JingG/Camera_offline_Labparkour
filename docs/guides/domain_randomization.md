# Domain Randomization 模块文档

## 概述

Domain Randomization 模块实现了深度图像的域随机化增强，用于提高强化学习策略对真实世界条件的鲁棒性。

## 模块位置

`/home/droplet/IsaacLab/Camera_offline_Labparkour/parkour_isaaclab/envs/mdp/domain_randomization.py`

## 组件

### 1. DepthNoiseAugmentation

模拟真实深度相机的各种噪声。

**功能：**
- 高斯噪声：模拟传感器噪声
- 椒盐噪声：模拟死像素
- 缺失像素：模拟 IR 反射失败
- 深度量化：模拟传感器位深限制
- 尺度变化：模拟校准误差

**使用示例：**
```python
from parkour_isaaclab.envs.mdp.domain_randomization import DepthNoiseAugmentation

# 创建配置
cfg = type('Config', (), {
    'gaussian_std': 0.02,
    'salt_pepper_prob': 0.01,
    'missing_pixel_prob': 0.01,
    'quantization_levels': 256,
    'scale_range': (0.95, 1.05),
})()

# 初始化增强器
aug = DepthNoiseAugmentation(cfg)

# 应用增强
depth_image = torch.randn(4, 58, 87)  # [B, H, W]
augmented = aug.apply(depth_image)
```

### 2. LatencySimulation

模拟真实硬件的处理延迟。

**功能：**
- 帧延迟：使用 N 帧前的深度图像
- 随机丢帧：模拟处理失败

**使用示例：**
```python
from parkour_isaaclab.envs.mdp.domain_randomization import LatencySimulation

# 初始化延迟模拟器
latency_sim = LatencySimulation(
    depth_delay_frames=3,  # 延迟 3 帧
    drop_prob=0.1          # 10% 丢帧概率
)

# 应用延迟
for frame in frames:
    delayed_frame = latency_sim.apply(frame)
```

### 3. LightingAugmentation

模拟不同光照条件对深度感知的影响。

**功能：**
- 亮度调整：模拟环境光变化
- 对比度调整：模拟光照均匀性

**使用示例：**
```python
from parkour_isaaclab.envs.mdp.domain_randomization import LightingAugmentation

# 创建配置
cfg = type('Config', (), {
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
})()

# 初始化增强器
aug = LightingAugmentation(cfg)

# 应用增强
augmented = aug.apply(depth_image)
```

### 4. DomainRandCurriculum

域随机化课程学习调度器，根据训练迭代次数逐步增加增强强度。

**课程学习阶段：**
| 迭代范围 | 噪声标准差 | 椒盐概率 | 相机丢失 | 延迟帧数 |
|---------|-----------|---------|---------|---------|
| 0-1000 | 0.01 | 0.005 | 0.1 | 1 |
| 1000-3000 | 0.02 | 0.01 | 0.2 | 2 |
| 3000-5000 | 0.03 | 0.015 | 0.3 | 3 |
| 5000+ | 0.04 | 0.02 | 0.5 | 3 |

**使用示例：**
```python
from parkour_isaaclab.envs.mdp.domain_randomization import DomainRandCurriculum

# 初始化课程学习调度器
curriculum = DomainRandCurriculum()

# 在训练循环中更新参数
for iteration in range(max_iterations):
    params = curriculum.get_params(iteration)

    # 使用当前参数配置增强器
    noise_aug.gaussian_std = params['noise_std']
    noise_aug.salt_pepper_prob = params['salt_pepper']
    # ...
```

### 5. CameraDropoutManagerWithCurriculum

带课程学习的相机丢失管理器，扩展原有的 CameraDropoutManager。

**功能：**
- 相机在线/离线状态管理
- 课程学习：丢失概率从 10% 逐步增加到 50%
- 自动状态切换

**使用示例：**
```python
from parkour_isaaclab.envs.mdp.domain_randomization import CameraDropoutManagerWithCurriculum

# 初始化管理器
manager = CameraDropoutManagerWithCurriculum(
    num_envs=256,
    device=torch.device('cuda'),
    enable_curriculum=True
)

# 在训练循环中
for iteration in range(max_iterations):
    # 更新课程学习
    manager.update_curriculum(iteration)

    # 在每个环境步骤中应用丢失
    depth_image = env.get_depth()
    depth_image = manager.update(depth_image)

    # 环境重置时
    if dones.any():
        manager.reset_env(dones)
```

## 测试

### 运行测试

```bash
cd /home/droplet/IsaacLab/Camera_offline_Labparkour
/home/droplet/anaconda3/envs/parkour/bin/python tests/mdp/test_standalone.py
```

### 测试覆盖

所有组件都有完整的单元测试：

- ✓ DepthNoiseAugmentation (5 个测试)
  - 高斯噪声
  - 椒盐噪声
  - 缺失像素
  - 深度量化
  - 尺度变化

- ✓ LatencySimulation (2 个测试)
  - 帧延迟
  - 随机丢帧

- ✓ LightingAugmentation (2 个测试)
  - 亮度调整
  - 对比度调整

- ✓ DomainRandCurriculum (3 个测试)
  - 课程学习递增
  - 线性插值
  - 最终阶段参数

- ✓ CameraDropoutManagerWithCurriculum (2 个测试)
  - 课程学习
  - 丢失应用

**总计：14 个测试，100% 通过率**

## 集成到训练流程

在 RL fine-tuning 训练脚本中集成：

```python
# 初始化域随机化组件
domain_rand_cfg = type('Config', (), {
    'gaussian_std': 0.01,
    'salt_pepper_prob': 0.005,
    'missing_pixel_prob': 0.005,
    'quantization_levels': 256,
    'scale_range': (0.95, 1.05),
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
})()

depth_noise = DepthNoiseAugmentation(domain_rand_cfg)
latency_sim = LatencySimulation(depth_delay_frames=1)
lighting_aug = LightingAugmentation(domain_rand_cfg)
curriculum = DomainRandCurriculum()
camera_dropout = CameraDropoutManagerWithCurriculum(
    num_envs=256,
    device=device,
    enable_curriculum=True
)

# 训练循环
for iteration in range(max_iterations):
    # 更新课程学习
    params = curriculum.get_params(iteration)
    depth_noise.gaussian_std = params['noise_std']
    depth_noise.salt_pepper_prob = params['salt_pepper']
    latency_sim.depth_delay = params['latency']
    camera_dropout.update_curriculum(iteration)

    # 收集 rollouts
    for step in range(num_steps):
        # 获取观测
        obs = env.get_observations()
        depth = obs['depth_camera']

        # 应用域随机化
        depth = depth_noise.apply(depth)
        depth = latency_sim.apply(depth)
        depth = lighting_aug.apply(depth)
        depth = camera_dropout.update(depth)

        # 使用增强后的深度图像
        actions = policy.act(obs['proprio'], depth)
        # ...
```

## 性能考虑

- 所有增强操作都是 GPU 兼容的（使用 PyTorch 张量）
- 遵循不可变性原则（使用 `.clone()` 创建副本）
- 支持批处理操作（向量化计算）
- 内存高效（原地操作在适当的地方）

## 未来扩展

可能的扩展方向：

1. **自适应课程学习**：根据策略性能动态调整增强强度
2. **更多噪声类型**：运动模糊、镜头畸变等
3. **空间相关噪声**：模拟真实相机的空间噪声模式
4. **时间相关噪声**：模拟连续帧之间的噪声相关性

## 参考

- 设计文档：`/home/droplet/IsaacLab/Camera_offline_Labparkour/docs/plans/2026-02-02-rl-finetuning-design.md`
- 架构文档：`/home/droplet/IsaacLab/Camera_offline_Labparkour/docs/architecture/2026-02-02-rl-finetuning-architecture.md`
