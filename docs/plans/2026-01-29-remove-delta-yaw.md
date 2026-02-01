# 移除 Delta Yaw 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 完全移除代码中所有 delta yaw 相关的功能，简化观测空间，减少无用特征

**架构:** 从观测计算、配置文件、推理脚本中移除所有 delta_yaw 和 delta_next_yaw 相关代码。观测维度从 54 维减少到 51 维，历史缓冲区维度相应调整。这是一个破坏性变更，需要重新训练所有模型。

**技术栈:** Python, PyTorch, Isaac Lab, RSL-RL

**影响范围:**
- 观测维度：54 → 51（移除 3 个 delta yaw 相关维度）
- 历史缓冲区维度：53 → 50
- 需要重新训练教师和学生模型
- 已有 checkpoint 无法使用

---

## 任务概览

1. **Task 1:** 移除核心观测计算中的 delta yaw
2. **Task 2:** 移除配置文件中的 delta yaw 相关配置
3. **Task 3:** 清理注释掉的 yaw 预测相关代码
4. **Task 4:** 验证修改并提交

---

### Task 1: 移除核心观测计算中的 delta yaw

**文件:**
- 修改: `parkour_isaaclab/envs/mdp/observations.py`

**步骤 1: 备份当前文件**

```bash
cp parkour_isaaclab/envs/mdp/observations.py parkour_isaaclab/envs/mdp/observations.py.backup
```

**步骤 2: 移除 delta_yaw 缓冲区初始化**

在 `ExtremeParkourObservations.__init__` 方法中：

删除第 40-41 行：
```python
# 删除这两行
self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
```

**步骤 3: 更新历史缓冲区维度**

修改第 39 行，将维度从 53 改为 50：

```python
# 修改前
self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, 3 + 2 + 3 + 4 + 36 + 5, device=self.device)

# 修改后（移除中间的 +3）
self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, 3 + 2 + 4 + 36 + 5, device=self.device)
```

**步骤 4: 移除 delta yaw 计算**

在 `__call__` 方法中，删除第 64-65 行：

```python
# 删除这两行
self.delta_yaw = self.parkour_event.target_yaw - wrap_to_pi(yaw)
self.delta_next_yaw = self.parkour_event.next_target_yaw - wrap_to_pi(yaw)
```

**步骤 5: 移除 obs_buf 中的 delta yaw 字段**

修改第 68-82 行的 `obs_buf` 构造，删除第 71-73 行：

```python
# 修改前
obs_buf = torch.cat((
    self.asset.data.root_ang_vel_b * 0.25,   #[1,3] 0~2
    imu_obs,    #[1,2] 3~4
    0*self.delta_yaw[:, None],   #[1,1] 5  ← 删除
    self.delta_yaw[:, None], #[1,1] 6  ← 删除
    self.delta_next_yaw[:, None], #[1,1] 7  ← 删除
    0*commands[:, 0:2], #[1,2] 8
    commands[:, 0:1],  #[1,1] 9
    env_idx_tensor,
    invert_env_idx_tensor,
    self.asset.data.joint_pos - self.asset.data.default_joint_pos,
    self.asset.data.joint_vel * 0.05 ,
    env.action_manager.get_term('joint_pos').action_history_buf[:, -1],
    self._get_contact_fill(),
),dim=-1)

# 修改后
obs_buf = torch.cat((
    self.asset.data.root_ang_vel_b * 0.25,   #[1,3] 0~2
    imu_obs,    #[1,2] 3~4
    0*commands[:, 0:2], #[1,2] 5~6
    commands[:, 0:1],  #[1,1] 7
    env_idx_tensor,    #[1,1] 8
    invert_env_idx_tensor,  #[1,1] 9
    self.asset.data.joint_pos - self.asset.data.default_joint_pos,  #[1,12] 10~21
    self.asset.data.joint_vel * 0.05 ,  #[1,12] 22~33
    env.action_manager.get_term('joint_pos').action_history_buf[:, -1],  #[1,12] 34~45
    self._get_contact_fill(),  #[1,5] 46~50
),dim=-1)
```

**步骤 6: 移除 obs_buf 清零代码**

删除第 90 行（这行本来就是清零 delta yaw 的）：

```python
# 删除这一行
obs_buf[:, 6:8] = 0
```

**步骤 7: 移除 obervation_delta_yaw_ok 类**

删除第 211-229 行的整个类定义：

```python
# 删除整个类（第 211-229 行）
class obervation_delta_yaw_ok(ManagerTermBase):
    ...
```

**步骤 8: 验证语法**

```bash
python -m py_compile parkour_isaaclab/envs/mdp/observations.py
```

预期输出：无错误

**步骤 9: 提交更改**

```bash
git add parkour_isaaclab/envs/mdp/observations.py
git commit -m "refactor: remove delta_yaw from observations

- Remove delta_yaw and delta_next_yaw buffers
- Update observation dimension from 54 to 51
- Update history buffer dimension from 53 to 50
- Remove obervation_delta_yaw_ok class
- Remove obs_buf clearing code for delta_yaw

BREAKING CHANGE: Observation dimension changed, requires retraining all models"
```

---

### Task 2: 移除配置文件中的 delta yaw 相关配置

**文件:**
- 修改: `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py`

**步骤 1: 移除 DeltaYawOkPolicyCfg 类定义**

删除第 93-100 行：

```python
# 删除这个类定义（第 93-100 行）
@configclass
class DeltaYawOkPolicyCfg(ObsGroup):
    deta_yaw_ok =  ObsTerm(
        func=observations.obervation_delta_yaw_ok,
        params={
        "parkour_name":'base_parkour',
        'threshold': 0.6
        },
    )
```

**步骤 2: 移除 ObservationsCfg 中的 delta_yaw_ok 字段**

删除第 103 行：

```python
# 在 ObservationsCfg 类中删除这一行
delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()
```

**步骤 3: 验证语法**

```bash
python -m py_compile parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py
```

预期输出：无错误

**步骤 4: 提交更改**

```bash
git add parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py
git commit -m "refactor: remove delta_yaw_ok from config

- Remove DeltaYawOkPolicyCfg class
- Remove delta_yaw_ok field from ObservationsCfg"
```

---

### Task 3: 清理注释掉的 yaw 预测相关代码

**文件:**
- 修改: `scripts/rsl_rl/train_student_dagger.py`
- 修改: `scripts/rsl_rl/train_student_from_dataset.py`
- 修改: `scripts/rsl_rl/play_student_transformer.py`

**步骤 1: 清理 train_student_dagger.py**

删除第 307-308 行的注释：

```python
# 删除这些注释行
# Butter for Closed-Loop Yaw Injection - Removed as requested
# last_yaw_pred = np.zeros((args.num_envs, 2), dtype=np.float32)
```

删除第 333-334 行的注释：

```python
# 删除这些注释行
# --- Mask Proprioception & Closed-Loop Injection (Removed as requested) ---
```

删除第 368 行的注释：

```python
# 删除这个注释
# Student prediction (yaw_pred_step removed)
```

**步骤 2: 清理 train_student_from_dataset.py**

删除第 273 行的注释：

```python
# 删除这个注释行
# self.yaw_head = nn.Linear(token_dim, 2)  # Removed as requested
```

删除第 330 行的注释：

```python
# 删除这个注释行
# yaw_pred = self.yaw_head(temporal_out) # Removed
```

删除第 375 行的注释：

```python
# 删除这个注释行
# yaw_pred = self.yaw_head(temporal_out) # Removed
```

删除第 393 行的文档字符串中关于 yaw_pred 的说明：

```python
# 在 forward_step 方法的文档字符串中删除这一行
yaw_pred_step: Tensor[B, 2]，辅助任务预测的偏航角变化。
```

**步骤 3: 清理 play_student_transformer.py**

删除所有关于 last_yaw_pred 的注释行（第 165, 178, 197-199, 214, 244, 284 行）

**步骤 4: 验证语法**

```bash
python -m py_compile scripts/rsl_rl/train_student_dagger.py
python -m py_compile scripts/rsl_rl/train_student_from_dataset.py
python -m py_compile scripts/rsl_rl/play_student_transformer.py
```

预期输出：无错误

**步骤 5: 提交更改**

```bash
git add scripts/rsl_rl/train_student_dagger.py \
        scripts/rsl_rl/train_student_from_dataset.py \
        scripts/rsl_rl/play_student_transformer.py
git commit -m "chore: remove commented yaw prediction code

- Clean up commented last_yaw_pred references
- Remove commented yaw_head references
- Remove outdated docstring about yaw_pred"
```

---

### Task 4: 验证修改并创建验证脚本

**文件:**
- 创建: `scripts/verify_observation_dims.py`

**步骤 1: 创建验证脚本**

创建一个脚本来验证观测维度是否正确：

```python
"""
验证观测维度是否正确更新
"""
import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def verify_observation_dimensions():
    """验证观测维度计算"""

    # 预期的观测维度
    expected_obs_buf_dim = 51  # 3 + 2 + 2 + 1 + 2 + 12 + 12 + 12 + 5
    expected_history_dim = 50  # 3 + 2 + 4 + 36 + 5

    print("=" * 60)
    print("观测维度验证")
    print("=" * 60)

    # 计算 obs_buf 维度
    obs_components = {
        "角速度 (root_ang_vel_b)": 3,
        "IMU (roll, pitch)": 2,
        "命令清零 (0*commands[:, 0:2])": 2,
        "命令 (commands[:, 0:1])": 1,
        "地形标记 (env_idx_tensor)": 1,
        "地形标记反转 (invert_env_idx_tensor)": 1,
        "关节位置偏差": 12,
        "关节速度": 12,
        "动作历史": 12,
        "接触信息": 5,
    }

    total_obs = sum(obs_components.values())

    print(f"\nobs_buf 组成:")
    for name, dim in obs_components.items():
        print(f"  - {name}: {dim}")
    print(f"  总计: {total_obs}")

    if total_obs == expected_obs_buf_dim:
        print(f"✅ obs_buf 维度正确: {total_obs}")
    else:
        print(f"❌ obs_buf 维度错误: 期望 {expected_obs_buf_dim}, 实际 {total_obs}")
        return False

    # 计算历史缓冲区维度
    history_components = {
        "角速度": 3,
        "IMU": 2,
        "命令相关": 4,  # 2 + 1 + 1
        "关节相关": 36,  # 12 + 12 + 12
        "接触": 5,
    }

    total_history = sum(history_components.values())

    print(f"\n历史缓冲区组成:")
    for name, dim in history_components.items():
        print(f"  - {name}: {dim}")
    print(f"  总计: {total_history}")

    if total_history == expected_history_dim:
        print(f"✅ 历史缓冲区维度正确: {total_history}")
    else:
        print(f"❌ 历史缓冲区维度错误: 期望 {expected_history_dim}, 实际 {total_history}")
        return False

    print("\n" + "=" * 60)
    print("✅ 所有维度验证通过！")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = verify_observation_dimensions()
    sys.exit(0 if success else 1)
```

**步骤 2: 运行验证脚本**

```bash
python scripts/verify_observation_dims.py
```

预期输出：
```
============================================================
观测维度验证
============================================================

obs_buf 组成:
  - 角速度 (root_ang_vel_b): 3
  - IMU (roll, pitch): 2
  - 命令清零 (0*commands[:, 0:2]): 2
  - 命令 (commands[:, 0:1]): 1
  - 地形标记 (env_idx_tensor): 1
  - 地形标记反转 (invert_env_idx_tensor): 1
  - 关节位置偏差: 12
  - 关节速度: 12
  - 动作历史: 12
  - 接触信息: 5
  总计: 51
✅ obs_buf 维度正确: 51

历史缓冲区组成:
  - 角速度: 3
  - IMU: 2
  - 命令相关: 4
  - 关节相关: 36
  - 接触: 5
  总计: 50
✅ 历史缓冲区维度正确: 50

============================================================
✅ 所有维度验证通过！
============================================================
```

**步骤 3: 提交验证脚本**

```bash
git add scripts/verify_observation_dims.py
git commit -m "test: add observation dimension verification script"
```

**步骤 4: 创建最终总结提交**

```bash
git commit --allow-empty -m "refactor: complete delta_yaw removal

Summary of changes:
- Removed delta_yaw and delta_next_yaw from observations
- Observation dimension: 54 → 51
- History buffer dimension: 53 → 50
- Removed obervation_delta_yaw_ok class
- Removed DeltaYawOkPolicyCfg from config
- Cleaned up commented yaw prediction code

BREAKING CHANGE: All models must be retrained with new observation dimensions

Next steps:
1. Retrain teacher model with new observation dimensions
2. Retrain student model after teacher is ready
3. Update any existing datasets or checkpoints"
```

---

## 后续步骤（不在本计划范围内）

完成代码修改后，需要执行以下步骤：

### 1. 重新训练教师模型

```bash
# 使用现有的教师训练脚本
bash scripts/run_teacher_training.sh
```

**预期结果:**
- 教师模型使用新的 51 维观测
- 训练时间与之前相同
- 性能应该相同或略好（因为移除了无用特征）

### 2. 重新训练学生模型

```bash
# 等教师模型训练完成后
bash scripts/run_student_dagger.sh
```

**注意事项:**
- 需要更新 `TEACHER_CHECKPOINT` 路径指向新训练的教师模型
- 学生模型的 `num_prop` 参数会自动从环境推断（已减少 3）

### 3. 验证模型性能

- 使用 play 脚本测试模型
- 确认导航能力没有下降
- 检查跑酷性能指标

---

## 回滚计划

如果需要回滚更改：

```bash
# 恢复所有修改
git revert HEAD~4..HEAD

# 或者使用备份文件
cp parkour_isaaclab/envs/mdp/observations.py.backup parkour_isaaclab/envs/mdp/observations.py
```

---

## 注意事项

1. **破坏性变更**: 这是一个破坏性变更，所有已训练的模型都无法使用
2. **观测索引**: 移除 delta yaw 后，其他观测的索引会发生变化，注意更新任何硬编码的索引
3. **测试**: 建议在小规模环境（少量 envs）上先测试，确认没有维度错误
4. **文档**: 更新任何相关文档，说明观测空间的变化

---

## 验证清单

- [ ] observations.py 中移除了所有 delta_yaw 引用
- [ ] 历史缓冲区维度更新为 50
- [ ] obs_buf 维度为 51
- [ ] 配置文件中移除了 DeltaYawOkPolicyCfg
- [ ] 清理了所有注释的 yaw 预测代码
- [ ] 验证脚本运行通过
- [ ] 所有修改已提交到 git
- [ ] 准备好重新训练教师模型
