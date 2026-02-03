# wandb 集成快速参考

## 快速开始

### 1. 安装和登录
```bash
pip install wandb
wandb login
```

### 2. 启用 wandb
```bash
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --wandb \
    --headless
```

## 常用命令

### 基线训练
```bash
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --num_envs 512 \
    --max_iterations 10000 \
    --wandb \
    --wandb_project "parkour-rl" \
    --wandb_run_name "baseline" \
    --wandb_tags baseline \
    --headless
```

### 冻结编码器
```bash
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --freeze_encoders \
    --freeze_fusion \
    --wandb \
    --wandb_run_name "freeze-encoders" \
    --wandb_tags ablation freeze \
    --headless
```

### 禁用 Domain Randomization
```bash
python run_student_rl_finetune.py train \
    --dagger_checkpoint logs/dagger/best.pt \
    --no_domain_rand \
    --wandb \
    --wandb_run_name "no-dr" \
    --wandb_tags ablation no-dr \
    --headless
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--wandb` | 启用 wandb | False |
| `--wandb_project` | 项目名称 | "student-rl-finetune" |
| `--wandb_entity` | 实体名称 | None |
| `--wandb_run_name` | 运行名称 | 自动生成 |
| `--wandb_tags` | 标签列表 | None |

## 记录的指标

### 训练指标
- `train/value_loss`
- `train/policy_loss`
- `train/entropy`
- `train/kl_divergence`
- `train/learning_rate`

### Domain Randomization（如果启用课程学习）
- `domain_rand/noise_std`
- `domain_rand/salt_pepper`
- `domain_rand/dropout`
- `domain_rand/latency`

## 查看结果

训练开始后，wandb 会输出 URL：
```
[wandb] View run at: https://wandb.ai/username/project/runs/xxxxx
```

## 离线模式

```bash
export WANDB_MODE=offline
python run_student_rl_finetune.py train --wandb ...
```

稍后同步：
```bash
wandb sync wandb/offline-run-xxxxx
```

## 故障排除

### wandb 未安装
```bash
pip install wandb
```

### 未登录
```bash
wandb login
```

### 查看帮助
```bash
python run_student_rl_finetune.py train --help
```

## 更多信息

详细文档：[docs/wandb_integration.md](./wandb_integration.md)
