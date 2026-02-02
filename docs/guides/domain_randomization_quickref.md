# Domain Randomization 快速参考

## 快速开始

### 1. 导入模块

```python
from parkour_isaaclab.envs.mdp.domain_randomization import (
    DepthNoiseAugmentation,
    LatencySimulation,
    LightingAugmentation,
    DomainRandCurriculum,
    CameraDropoutManagerWithCurriculum,
)
```

### 2. 基本使用

```python
# 创建配置
cfg = type('Config', (), {
    'gaussian_std': 0.02,
    'salt_pepper_prob': 0.01,
    'missing_pixel_prob': 0.01,
    'quantization_levels': 256,
    'scale_range': (0.95, 1.05),
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
})()

# 初始化增强器
noise_aug = DepthNoiseAugmentation(cfg)
lighting_aug = LightingAugmentation(cfg)

# 应用增强
depth = torch.randn(4, 58, 87)
augmented = noise_aug.apply(depth)
augmented = lighting_aug.apply(augmented)
```

### 3. 课程学习

```python
# 初始化课程学习
curriculum = DomainRandCurriculum()

# 在训练循环中
for iteration in range(max_iterations):
    params = curriculum.get_params(iteration)
    noise_aug.gaussian_std = params['noise_std']
    # 使用更新后的参数
```

### 4. 相机丢失

```python
# 初始化管理器
camera_dropout = CameraDropoutManagerWithCurriculum(
    num_envs=256,
    device=torch.device('cuda'),
    enable_curriculum=True
)

# 更新和应用
camera_dropout.update_curriculum(iteration)
depth = camera_dropout.update(depth)

# 环境重置时
if dones.any():
    camera_dropout.reset_env(dones)
```

## 组件速查表

| 组件 | 功能 | 主要参数 |
|------|------|---------|
| DepthNoiseAugmentation | 深度噪声 | gaussian_std, salt_pepper_prob |
| LatencySimulation | 延迟模拟 | depth_delay_frames, drop_prob |
| LightingAugmentation | 光照调整 | brightness_range, contrast_range |
| DomainRandCurriculum | 课程学习 | iteration |
| CameraDropoutManagerWithCurriculum | 相机丢失 | num_envs, enable_curriculum |

## 测试命令

```bash
# 运行所有测试
/home/droplet/anaconda3/envs/parkour/bin/python tests/mdp/test_standalone.py

# 运行示例
/home/droplet/anaconda3/envs/parkour/bin/python examples/domain_randomization_demo.py
```

## 文件位置

- **实现：** `parkour_isaaclab/envs/mdp/domain_randomization.py`
- **测试：** `tests/mdp/test_standalone.py`
- **文档：** `docs/domain_randomization.md`
- **示例：** `examples/domain_randomization_demo.py`

## 常见问题

### Q: 如何调整增强强度？
A: 修改配置对象中的参数，或使用课程学习自动调整。

### Q: 如何禁用某个增强？
A: 将对应参数设置为 0 或 (1.0, 1.0)。

### Q: 如何自定义课程学习时间表？
A: 修改 `DomainRandCurriculum.SCHEDULE` 列表。

### Q: 增强是否支持 GPU？
A: 是的，所有操作都是 GPU 兼容的。

## 性能提示

1. 批量处理多个环境以提高效率
2. 在 GPU 上运行以加速计算
3. 使用课程学习避免过早的强增强
4. 监控训练性能以调整参数

## 下一步

- 集成到训练脚本
- 运行实验验证效果
- 根据结果调整参数
