# Phase 3 快速参考：Domain Randomization

**版本：** 1.0
**日期：** 2026-02-03
**完整文档：** [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md)

---

## 快速开始

### 基本使用

```python
from parkour_isaaclab.envs.mdp.domain_randomization import (
    DepthNoiseAugmentation,
    LatencySimulation,
    LightingAugmentation,
    DomainRandCurriculum,
    CameraDropoutManagerWithCurriculum
)

# 1. 创建配置
cfg = type('Config', (), {
    'gaussian_std': 0.02,
    'salt_pepper_prob': 0.01,
    'missing_pixel_prob': 0.01,
    'quantization_levels': 256,
    'scale_range': (0.95, 1.05),
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
})()

# 2. 初始化组件
depth_noise = DepthNoiseAugmentation(cfg)
latency_sim = LatencySimulation(depth_delay_frames=3, drop_prob=0.1)
lighting_aug = LightingAugmentation(cfg)
curriculum = DomainRandCurriculum()
camera_dropout = CameraDropoutManagerWithCurriculum(
    num_envs=256,
    device=torch.device('cuda'),
    enable_curriculum=True
)

# 3. 在训练循环中使用
for iteration in range(max_iterations):
    # 更新课程学习
    params = curriculum.get_params(iteration)
    depth_noise.gaussian_std = params['noise_std']
    camera_dropout.update_curriculum(iteration)

    # 应用增强
    depth = env.get_depth()
    depth = depth_noise.apply(depth)
    depth = latency_sim.apply(depth)
    depth = lighting_aug.apply(depth)
    depth = camera_dropout.update(depth)
```

---

## 组件参考

### 1. DepthNoiseAugmentation

**用途：** 模拟深度相机噪声

**参数配置：**

| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `gaussian_std` | float | 0.01-0.04 | 0.02 | 高斯噪声标准差 |
| `salt_pepper_prob` | float | 0.005-0.02 | 0.01 | 椒盐噪声概率 |
| `missing_pixel_prob` | float | 0.005-0.02 | 0.01 | 缺失像素概率 |
| `quantization_levels` | int | 256-1024 | 256 | 量化级别（0=不量化）|
| `scale_range` | tuple | (0.95, 1.05) | (1.0, 1.0) | 尺度变化范围 |

**代码示例：**
```python
# 创建配置
cfg = type('Config', (), {
    'gaussian_std': 0.02,
    'salt_pepper_prob': 0.01,
    'missing_pixel_prob': 0.01,
    'quantization_levels': 256,
    'scale_range': (0.95, 1.05),
})()

# 初始化
aug = DepthNoiseAugmentation(cfg)

# 应用增强
depth = torch.randn(4, 58, 87)  # [B, H, W]
augmented = aug.apply(depth)
```

**噪声类型：**
1. **高斯噪声** - 传感器随机误差
2. **椒盐噪声** - 死像素
3. **缺失像素** - IR 反射失败（设置为 -0.5）
4. **深度量化** - 位深限制
5. **尺度变化** - 校准误差

---

### 2. LatencySimulation

**用途：** 模拟处理延迟和丢帧

**参数配置：**

| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `depth_delay_frames` | int | 1-5 | 3 | 延迟帧数 |
| `drop_prob` | float | 0.0-0.2 | 0.0 | 丢帧概率 |

**代码示例：**
```python
# 初始化
latency_sim = LatencySimulation(
    depth_delay_frames=3,  # 延迟 3 帧
    drop_prob=0.1          # 10% 丢帧概率
)

# 在循环中使用
for frame in frames:
    delayed_frame = latency_sim.apply(frame)

# 重置（episode 结束时）
latency_sim.reset()
```

**工作原理：**
- 维护一个大小为 `depth_delay_frames + 1` 的缓冲区
- 返回 N 帧前的深度图像
- 丢帧时返回上一帧有效深度

---

### 3. LightingAugmentation

**用途：** 模拟光照变化

**参数配置：**

| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `brightness_range` | tuple | (0.8, 1.2) | (1.0, 1.0) | 亮度范围 |
| `contrast_range` | tuple | (0.8, 1.2) | (1.0, 1.0) | 对比度范围 |

**代码示例：**
```python
# 创建配置
cfg = type('Config', (), {
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
})()

# 初始化
aug = LightingAugmentation(cfg)

# 应用增强
augmented = aug.apply(depth)
```

**增强类型：**
1. **亮度调整** - 乘以随机因子
2. **对比度调整** - `(x - mean) * factor + mean`

---

### 4. DomainRandCurriculum

**用途：** 课程学习调度器

**课程学习时间表：**

| 迭代范围 | 噪声标准差 | 椒盐概率 | 相机丢失 | 延迟帧数 |
|---------|-----------|---------|---------|---------|
| 0-1000 | 0.01 | 0.005 | 0.1 | 1 |
| 1000-3000 | 0.02 | 0.01 | 0.2 | 2 |
| 3000-5000 | 0.03 | 0.015 | 0.3 | 3 |
| 5000+ | 0.04 | 0.02 | 0.5 | 3 |

**代码示例：**
```python
# 初始化
curriculum = DomainRandCurriculum()

# 在训练循环中
for iteration in range(max_iterations):
    # 获取当前参数
    params = curriculum.get_params(iteration)

    # 更新增强器
    depth_noise.gaussian_std = params['noise_std']
    depth_noise.salt_pepper_prob = params['salt_pepper']
    latency_sim.depth_delay = params['latency']
    # camera_dropout 使用自己的 update_curriculum()
```

**返回参数：**
```python
{
    'noise_std': float,      # 噪声标准差
    'salt_pepper': float,    # 椒盐概率
    'dropout': float,        # 相机丢失概率
    'latency': int,          # 延迟帧数
}
```

---

### 5. CameraDropoutManagerWithCurriculum

**用途：** 带课程学习的相机丢失管理

**参数配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_envs` | int | - | 环境数量 |
| `device` | torch.device | - | PyTorch 设备 |
| `dt` | float | 0.02 | 时间步长（秒）|
| `initial_prob` | float | 0.1 | 初始丢失概率 |
| `online_duration_range` | tuple | (2.0, 10.0) | 在线持续时间（秒）|
| `offline_duration_range` | tuple | (1.0, 7.0) | 离线持续时间（秒）|
| `enable_curriculum` | bool | True | 是否启用课程学习 |

**代码示例：**
```python
# 初始化
manager = CameraDropoutManagerWithCurriculum(
    num_envs=256,
    device=torch.device('cuda'),
    dt=0.02,
    enable_curriculum=True
)

# 训练循环
for iteration in range(max_iterations):
    # 更新课程学习
    manager.update_curriculum(iteration)

    # 每个环境步骤
    for step in range(num_steps):
        depth = env.get_depth()
        depth = manager.update(depth)  # 应用丢失

    # 环境重置
    if dones.any():
        manager.reset_env(dones.nonzero().squeeze(-1))
```

**课程学习时间表：**
- 0-1000: 10% 丢失概率
- 1000-3000: 10% -> 20%（线性插值）
- 3000-5000: 20% -> 30%（线性插值）
- 5000+: 50% 丢失概率

---

## 完整集成示例

```python
import torch
from parkour_isaaclab.envs.mdp.domain_randomization import (
    DepthNoiseAugmentation,
    LatencySimulation,
    LightingAugmentation,
    DomainRandCurriculum,
    CameraDropoutManagerWithCurriculum
)

# 1. 初始化所有组件
device = torch.device('cuda')
num_envs = 256

# 配置
domain_rand_cfg = type('Config', (), {
    'gaussian_std': 0.01,
    'salt_pepper_prob': 0.005,
    'missing_pixel_prob': 0.005,
    'quantization_levels': 256,
    'scale_range': (0.95, 1.05),
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
})()

# 组件
depth_noise = DepthNoiseAugmentation(domain_rand_cfg)
latency_sim = LatencySimulation(depth_delay_frames=1, drop_prob=0.0)
lighting_aug = LightingAugmentation(domain_rand_cfg)
curriculum = DomainRandCurriculum()
camera_dropout = CameraDropoutManagerWithCurriculum(
    num_envs=num_envs,
    device=device,
    enable_curriculum=True
)

# 2. 训练循环
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

        # 应用域随机化（按顺序）
        depth = depth_noise.apply(depth)
        depth = latency_sim.apply(depth)
        depth = lighting_aug.apply(depth)
        depth = camera_dropout.update(depth)

        # 使用增强后的深度
        actions = policy.act(obs['proprio'], depth)
        obs, rewards, dones, infos = env.step(actions)

        # 处理重置
        if dones.any():
            done_ids = dones.nonzero().squeeze(-1)
            camera_dropout.reset_env(done_ids)

    # PPO 更新
    policy.update()
```

---

## 常见问题

### Q1: 应该使用什么顺序应用增强？

**A:** 推荐顺序：
1. DepthNoiseAugmentation（基础噪声）
2. LatencySimulation（时序延迟）
3. LightingAugmentation（光照变化）
4. CameraDropoutManager（相机丢失，最后）

### Q2: 如何选择合适的参数？

**A:**
- **初期训练**：使用保守参数（见下方配置）
- **后期训练**：使用激进参数
- **使用课程学习**：让系统自动调整

**保守配置：**
```python
cfg = {
    'gaussian_std': 0.01,
    'salt_pepper_prob': 0.005,
    'quantization_levels': 512,
    'scale_range': (0.98, 1.02),
    'depth_delay_frames': 1,
    'camera_dropout_prob': 0.1,
}
```

**激进配置：**
```python
cfg = {
    'gaussian_std': 0.04,
    'salt_pepper_prob': 0.02,
    'quantization_levels': 256,
    'scale_range': (0.95, 1.05),
    'depth_delay_frames': 3,
    'camera_dropout_prob': 0.5,
}
```

### Q3: 课程学习是必需的吗？

**A:** 强烈推荐。课程学习：
- ✅ 提高训练稳定性
- ✅ 避免早期训练崩溃
- ✅ 更快收敛
- ✅ 更好的最终性能

可以通过 `enable_curriculum=False` 禁用。

### Q4: 如何调试增强效果？

**A:**
```python
# 可视化增强前后的深度图像
import matplotlib.pyplot as plt

depth_original = env.get_depth()[0]  # 第一个环境
depth_augmented = depth_noise.apply(depth_original)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(depth_original.cpu())
axes[0].set_title('Original')
axes[1].imshow(depth_augmented.cpu())
axes[1].set_title('Augmented')
plt.show()
```

### Q5: 性能开销有多大？

**A:** 非常小：
- 总开销：~1.1 ms per step（256 环境）
- GPU 利用率：低
- 额外内存：~55 MB

不会成为训练瓶颈。

### Q6: 如何在评估时禁用增强？

**A:**
```python
# 方法 1：创建不增强的配置
eval_cfg = type('Config', (), {
    'gaussian_std': 0.0,
    'salt_pepper_prob': 0.0,
    'missing_pixel_prob': 0.0,
    'quantization_levels': 0,
    'scale_range': (1.0, 1.0),
    'brightness_range': (1.0, 1.0),
    'contrast_range': (1.0, 1.0),
})()

# 方法 2：直接跳过增强步骤
if not eval_mode:
    depth = depth_noise.apply(depth)
```

---

## 文件位置索引

| 文件 | 路径 |
|------|------|
| **实现** | `parkour_isaaclab/envs/mdp/domain_randomization.py` |
| **测试** | `tests/mdp/test_domain_randomization.py` |
| **文档** | `docs/domain_randomization.md` |
| **总结** | `docs/PHASE3_SUMMARY.md` |
| **快速参考** | `docs/PHASE3_QUICKREF.md` |

---

## 相关文档

- [Phase 3 完整总结](PHASE3_SUMMARY.md)
- [Phase 1 总结](PHASE1_SUMMARY.md)
- [Phase 2 总结](PHASE2_SUMMARY.md)
- [RL Fine-tuning 架构](architecture/2026-02-02-rl-finetuning-architecture.md)
- [Domain Randomization 详细文档](domain_randomization.md)

---

**版本：** 1.0
**最后更新：** 2026-02-03
