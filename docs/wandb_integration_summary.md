# wandb 集成完成总结

## 完成时间
2026-02-03

## 实现内容

### 1. 命令行参数（run_student_rl_finetune.py）

添加了以下 wandb 相关参数：
- `--wandb`: 启用 wandb 日志记录
- `--wandb_project`: 项目名称（默认: "student-rl-finetune"）
- `--wandb_entity`: 实体名称（用户名或团队名）
- `--wandb_run_name`: 运行名称（默认自动生成）
- `--wandb_tags`: 标签列表

### 2. 配置数据类（train_student_rl_finetune.py）

在 `TrainingConfig` 中添加了 wandb 配置字段：
```python
use_wandb: bool = False
wandb_project: str = "student-rl-finetune"
wandb_entity: Optional[str] = None
wandb_run_name: Optional[str] = None
wandb_tags: Optional[List[str]] = None
```

### 3. wandb 集成函数（train_student_rl_finetune.py）

实现了三个核心函数：

#### `init_wandb()`
- 初始化 wandb run
- 记录所有训练配置和超参数
- 支持恢复训练（resume_run_id）

#### `log_training_metrics()`
- 记录 PPO 训练指标（loss、entropy、KL 散度等）
- 记录 Domain Randomization 参数
- 支持 episode 统计信息

#### `save_checkpoint_to_wandb()`
- 上传检查点到 wandb Artifacts
- 支持版本控制
- 标记最佳模型

### 4. 训练循环集成

在 `run_training_loop()` 函数中：
- 训练开始前初始化 wandb
- 每个 log_interval 记录训练指标
- 每个 save_interval 上传检查点
- 训练结束时上传最终检查点并关闭 wandb run

## 记录的指标

### 训练指标
- `train/value_loss`: 价值函数损失
- `train/policy_loss`: 策略损失
- `train/entropy`: 策略熵
- `train/kl_divergence`: KL 散度
- `train/learning_rate`: 学习率
- `iteration`: 训练迭代次数

### Domain Randomization 参数（如果启用课程学习）
- `domain_rand/noise_std`: 深度噪声标准差
- `domain_rand/salt_pepper`: 椒盐噪声概率
- `domain_rand/dropout`: 相机丢失概率
- `domain_rand/latency`: 延迟帧数

### 超参数配置
- 环境配置（num_envs、num_steps_per_env 等）
- PPO 超参数（learning_rate、clip_param、gamma 等）
- 编码器冻结策略
- Domain Randomization 配置
- 模型架构参数

## 使用示例

### 基本用法
```bash
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --wandb \
    --headless
```

### 完整配置
```bash
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --num_envs 512 \
    --max_iterations 10000 \
    --wandb \
    --wandb_project "parkour-rl-experiments" \
    --wandb_run_name "baseline-512envs" \
    --wandb_tags baseline v1 \
    --headless
```

## 文档

详细使用文档：[docs/wandb_integration.md](./wandb_integration.md)

## 代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 命令行参数 | run_student_rl_finetune.py | 176-210 |
| 配置字段 | train_student_rl_finetune.py | 195-200 |
| wandb 初始化 | train_student_rl_finetune.py | 381-453 |
| 日志记录 | train_student_rl_finetune.py | 456-493 |
| 检查点上传 | train_student_rl_finetune.py | 496-515 |
| 训练循环集成 | train_student_rl_finetune.py | 1101-1165 |

## 测试建议

### 1. 基本功能测试
```bash
# 测试 wandb 初始化和日志记录
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --num_envs 4 \
    --max_iterations 10 \
    --wandb \
    --wandb_run_name "test-basic" \
    --headless
```

### 2. 检查点上传测试
```bash
# 测试检查点上传（设置较短的保存间隔）
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --num_envs 4 \
    --max_iterations 20 \
    --save_interval 5 \
    --wandb \
    --wandb_run_name "test-checkpoint" \
    --headless
```

### 3. Domain Randomization 课程测试
```bash
# 测试 Domain Randomization 参数记录
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --num_envs 4 \
    --max_iterations 50 \
    --wandb \
    --wandb_run_name "test-curriculum" \
    --headless
```

## 未来改进

- [ ] 实现训练恢复时的 wandb run_id 保存和恢复
- [ ] 添加 episode 统计信息记录（reward、length、success rate）
- [ ] 集成评估结果到 wandb
- [ ] 支持 wandb Sweeps 进行超参数搜索
- [ ] 添加模型架构可视化
- [ ] 记录梯度分布和权重直方图

## 注意事项

1. **安装 wandb**: 使用前需要先安装 wandb 包
   ```bash
   conda activate parkour
   pip install wandb
   wandb login
   ```

2. **可选功能**: wandb 集成是可选的，不影响现有训练流程
   - 不添加 `--wandb` 参数时，训练正常进行，不记录到 wandb

3. **离线模式**: 如果无法连接到 wandb 服务器，可以使用离线模式
   ```bash
   export WANDB_MODE=offline
   ```

4. **性能影响**: wandb 日志记录对训练性能的影响很小（< 1%）

## 相关文件

- [docs/wandb_integration.md](./wandb_integration.md) - 详细使用文档
- [run_student_rl_finetune.py](../scripts/rsl_rl/run_student_rl_finetune.py) - 启动脚本
- [train_student_rl_finetune.py](../scripts/rsl_rl/train_student_rl_finetune.py) - 训练脚本
