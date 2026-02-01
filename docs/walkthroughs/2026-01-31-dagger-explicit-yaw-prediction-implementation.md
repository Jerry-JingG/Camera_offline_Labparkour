# Dagger 显式 Yaw 预测实现详解

**日期:** 2026-01-31

**目标:** 为 dagger 训练方法的深度编码器添加显式 yaw 预测功能，与原始 train.py 的实现方式保持一致。

## 概述

本次实现为学生模型在 DAGGER 训练过程中添加了显式 yaw 预测能力。核心思想是：学生模型应该从视觉特征（深度图像）中预测 yaw，而不是依赖特权信息（观测中的 delta_yaw）。

## 架构变更

### 1. DepthEncoder 修改

**文件:** `parkour_tasks/parkour_tasks/extreme_parkour_task/modules/tokenizers/depth_encoder.py`

**变更内容:**
- 在 `__init__` 中添加 `num_prop` 参数，用于可选的本体感知输入
- 添加 `yaw_head` - 一个3层MLP，从视觉特征预测2D yaw
- 修改 `forward()` 方法：
  - 接受可选的 `proprio` 张量（delta_yaw 已置零）
  - 输出拼接后的 `[visual_tokens_flat, yaw_pred]`，而不仅仅是 tokens
  - 输出形状从 `[B, num_tokens, token_dim]` 变为 `[B, num_tokens * token_dim + 2]`

**Yaw Head 架构:**
```
输入: visual_features (grid_size * grid_size * token_dim) + 可选的 proprio (num_prop)
  -> Linear(input_dim, 256) + GELU + Dropout
  -> Linear(256, 128) + GELU + Dropout
  -> Linear(128, 2)  # 2D yaw 输出
```

### 2. MultiModalStudentPolicy 修改

**文件:** `scripts/rsl_rl/train_student_from_dataset.py`

**变更内容:**

#### `__init__` 方法:
- 添加 `self.num_prop = proprio_dim` 存储本体感知维度
- 更新 `DepthEncoder` 初始化，传入 `num_prop` 参数

#### 新增辅助方法 `_get_last_frame_indices()`:
- 返回展平后本体感知历史中 delta_yaw 的起止索引
- 用于定位 delta_yaw 位置: `(prop_hist_len - 1) * num_prop + 6` 到 `+ 8`

#### `forward()` 方法:
- 在传入深度编码器前将 proprio 中的 delta_yaw（索引 6:8）置零
- 从深度编码器输出中提取 yaw 预测（最后2个维度）
- 将 yaw 预测缩放 1.5 倍（与原始实现一致）
- 用缩放后的 yaw 预测替换 proprio 中的 delta_yaw
- 同时返回 actions 和 yaw 预测

#### `forward_with_mems()` 方法:
- 添加 `delta_yaw_ok` 参数用于选择性 yaw 替换
- 实现完整的 yaw 预测流程：
  1. 将 proprio 中的 delta_yaw 置零后传给编码器
  2. 从深度编码器获取 depth tokens + yaw
  3. 将 yaw 缩放 1.5 倍
  4. 替换 proprio 中的 delta_yaw（遵循 delta_yaw_ok 掩码）
  5. 编码修改后的 proprio
  6. 融合并通过时序模型处理
- 返回 `(actions, yaw_pred_seq, new_mems)`

#### `forward_step()` 方法:
- 添加 `delta_yaw_ok` 参数
- 将 delta_yaw_ok 传递给 `forward_with_mems`
- 返回 `(actions_step, yaw_pred_step, new_mems)`

### 3. DAGGER 训练循环修改

**文件:** `scripts/rsl_rl/train_student_dagger.py`

**变更内容:**

#### Rollout 阶段:
- 添加 `yaws_buffer` 收集 yaw 预测误差
- 添加 `delta_yaw_ok` 掩码（当前全部为 True）
- 将 `delta_yaw_ok` 传递给 `student.forward_step()`
- 收集 yaw 误差: `true_yaw - yaw_pred_step * 1.5`

#### 训练阶段:
- 从 batch proprio 中提取真实 yaw（最后一帧，索引 6:8）
- 计算 yaw 损失: `MSE(yaw_pred * 1.5, true_yaw_train)`
- 总损失 = action_loss + yaw_loss

#### 日志记录:
- 控制台输出现在显示: `loss=X (action=Y, yaw=Z)`
- 在 wandb 指标中添加 `train/loss_yaw`

## 数据流

```
输入: proprio_seq [B, S, prop_hist_len * num_prop], depth_seq [B, S, depth_hist_len, H, W]
                                    |
                                    v
                    +-------------------------------+
                    |  将 delta_yaw (6:8) 置零     |
                    |  用于编码器输入               |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |      DepthEncoder            |
                    |  输入: depth + 置零的 prop   |
                    |  输出: [tokens_flat, yaw]    |
                    +-------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
            depth_tokens                      yaw_pred (2D)
                    |                               |
                    |                               v
                    |                    +-------------------+
                    |                    | 缩放 1.5 倍       |
                    |                    +-------------------+
                    |                               |
                    v                               v
                    +-------------------------------+
                    |  用缩放后的 yaw 替换         |
                    |  proprio 中的 delta_yaw      |
                    |  (遵循 delta_yaw_ok 掩码)    |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |     ProprioEncoder           |
                    |  (使用替换后的 yaw)          |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |   FusionTransformer          |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |   TemporalModel (TXL)        |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |      ActionHead              |
                    +-------------------------------+
                                    |
                                    v
                            actions [B, S, action_dim]
```

## 关键设计决策

1. **Yaw 缩放因子 (1.5):** 与原始 train.py 实现保持一致。这个缩放有助于模型学习更激进的 yaw 预测。

2. **Delta_yaw 置零:** 在将 proprio 传入深度编码器之前，delta_yaw 被置零。这迫使模型纯粹从视觉特征预测 yaw，而不是"作弊"使用真实值。

3. **内部 yaw 替换:** yaw 替换发生在模型的前向传播内部，创建一个闭环，使模型在推理时能看到自己的 yaw 预测。

4. **delta_yaw_ok 掩码:** 允许按环境选择性地进行 yaw 替换。当前设置为全部 True，但可用于课程学习。

5. **等权重损失:** Yaw 损失与 action 损失权重相等（都是 MSE）。

## 修改的文件

1. `parkour_tasks/parkour_tasks/extreme_parkour_task/modules/tokenizers/depth_encoder.py`
2. `scripts/rsl_rl/train_student_from_dataset.py`
3. `scripts/rsl_rl/train_student_dagger.py`

## 测试清单

- [ ] 训练运行无错误
- [ ] Yaw 损失随迭代下降
- [ ] Action 损失保持稳定
- [ ] Yaw 预测在合理范围内
- [ ] 模型检查点正确保存

## 未来改进

1. **delta_yaw_ok 课程学习:** 逐步增加使用预测 yaw 的环境比例
2. **Yaw 损失权重调优:** 可能需要调整 yaw 与 action 损失的相对权重
3. **Yaw 更新频率:** 原始实现每5步更新一次 yaw 以减少计算量
4. **更新 play_student.py:** 确保推理时正确使用 yaw 预测

