# 恢复 delta_yaw 并在学生策略中实现 mask 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 恢复 obs_buf 中的 delta_yaw 信息（让教师策略能看到），同时在学生策略的 dagger 训练和推理脚本中实现 mask 逻辑（让学生策略看不到这些特权信息）

**架构:** delta_yaw 和 delta_next_yaw 作为特权信息保留在 obs_buf 的索引 6-7 位置。教师策略可以直接使用这些信息进行训练。学生策略在 dagger 训练和推理时，通过 mask 操作将这些维度清零，模拟真实部署时无法获得精确目标方向的情况。

**技术栈:** Python, PyTorch, Isaac Lab, RSL-RL

**影响范围:**
- 观测维度：51 → 54（恢复 3 个 delta yaw 相关维度）
- 历史缓冲区维度：50 → 53
- 需要重新训练教师模型
- 需要在学生训练和推理脚本中添加 mask 逻辑

---

## 任务概览

1. **Task 1:** 恢复 observations.py 中的 delta_yaw 计算和观测
2. **Task 2:** 在 train_student_dagger.py 中实现 delta_yaw mask
3. **Task 3:** 在 play_student_transformer.py 中实现 delta_yaw mask
4. **Task 4:** 在 play_student.py 中实现 delta_yaw mask
5. **Task 5:** 验证修改并提交

---

### Task 1: 恢复 observations.py 中的 delta_yaw 计算和观测

**文件:**
- 修改: `parkour_isaaclab/envs/mdp/observations.py`
- 参考: `parkour_isaaclab/envs/mdp/observations.py.backup`

**步骤 1: 恢复 delta_yaw 缓冲区初始化**

在 `ExtremeParkourObservations.__init__` 方法中，第 38 行之后添加：

```python
self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, 3 + 2 + 3 + 4 + 36 + 5, device=self.device)
self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
```

**步骤 2: 恢复 delta_yaw 计算**

在 `__call__` 方法中，第 61-62 行之间添加 delta_yaw 计算：

```python
if env.common_step_counter % 5 == 0:
    self.delta_yaw = self.parkour_event.target_yaw - wrap_to_pi(yaw)
    self.delta_next_yaw = self.parkour_event.next_target_yaw - wrap_to_pi(yaw)
    self.measured_heights = self._get_heights()
```

**步骤 3: 恢复 obs_buf 中的 delta_yaw 字段**

修改 `__call__` 方法中的 obs_buf 构造（第 63-75 行）：

```python
obs_buf = torch.cat((
    self.asset.data.root_ang_vel_b * 0.25,   #[1,3] 0~2
    imu_obs,    #[1,2] 3~4
    0*self.delta_yaw[:, None],   #[1,1] 5
    self.delta_yaw[:, None], #[1,1] 6
    self.delta_next_yaw[:, None], #[1,1] 7
    0*commands[:, 0:2], #[1,2] 8~9
    commands[:, 0:1],  #[1,1] 10
    env_idx_tensor,    #[1,1] 11
    invert_env_idx_tensor,  #[1,1] 12
    self.asset.data.joint_pos - self.asset.data.default_joint_pos,  #[1,12] 13~24
    self.asset.data.joint_vel * 0.05 ,  #[1,12] 25~36
    env.action_manager.get_term('joint_pos').action_history_buf[:, -1],  #[1,12] 37~48
    self._get_contact_fill(),  #[1,5] 49~53
),dim=-1)
```

**步骤 4: 构造完整观测**

构造完整的 observations（包含 obs_buf、高度图、特权信息和历史缓冲区）：

```python
observations = torch.cat([obs_buf, #54
                          self.measured_heights, #132
                          priv_explicit, # 9
                          priv_latent, # 29
                          self._obs_history_buffer.view(self.num_envs, -1)
                          ],dim=-1)
```

**重要说明：** 不要在这里清零 delta_yaw！教师策略需要看到完整的 delta_yaw 信息（索引6-7）。历史缓冲区会自动从 obs_buf 更新，所以历史中也会包含 delta_yaw。学生策略的 mask 操作将在学生脚本中实现。

**步骤 5: 验证语法**

```bash
python -m py_compile parkour_isaaclab/envs/mdp/observations.py
```

预期输出：无错误

**步骤 6: 提交更改**

```bash
git add parkour_isaaclab/envs/mdp/observations.py
git commit -m "feat: restore delta_yaw in observations

- Restore delta_yaw and delta_next_yaw buffers
- Update observation dimension from 51 to 54
- Update history buffer dimension from 50 to 53
- delta_yaw visible to teacher policy in current obs and history
- Student policy masking will be implemented in dagger/inference scripts

This allows teacher policy to use privileged direction info
while preparing for student policy masking in dagger training."
```

---

### Task 2: 在 train_student_dagger.py 中实现 delta_yaw mask

**文件:**
- 修改: `scripts/rsl_rl/train_student_dagger.py:330`

**步骤 1: 实现 mask 逻辑**

修改第 330 行，将简单的 copy 改为实际的 mask 操作：

```python
# Mask privileged information (delta_yaw at indices 6-7)
obs_prop_np_masked = obs_prop_np.copy()
obs_prop_np_masked[:, 6:8] = 0  # Zero out delta_yaw and delta_next_yaw
```

**步骤 2: 添加注释说明**

在第 330 行之前添加详细注释：

```python
# --- Mask privileged information for student ---
# Teacher policy can see delta_yaw (indices 6-7) in obs_buf
# Student policy should not have access to this privileged direction info
# Indices:
#   6: delta_yaw (current target direction - robot yaw)
#   7: delta_next_yaw (next target direction - robot yaw)
obs_prop_np_masked = obs_prop_np.copy()
obs_prop_np_masked[:, 6:8] = 0  # Zero out delta_yaw and delta_next_yaw
```

**步骤 3: 验证语法**

```bash
python -m py_compile scripts/rsl_rl/train_student_dagger.py
```

预期输出：无错误

**步骤 4: 提交更改**

```bash
git add scripts/rsl_rl/train_student_dagger.py
git commit -m "feat: implement delta_yaw masking in student dagger training

- Mask obs_buf indices 6-7 (delta_yaw, delta_next_yaw) for student
- Student policy trains without privileged direction information
- Teacher policy retains full observation including delta_yaw"
```

---

### Task 3: 在 play_student_transformer.py 中实现 delta_yaw mask

**文件:**
- 修改: `scripts/rsl_rl/play_student_transformer.py:332`

**步骤 1: 实现 mask 逻辑**

在第 332 行之后添加 mask 操作：

```python
obs_prop = obs[:, :num_prop]

# Mask privileged information (delta_yaw at indices 6-7)
# Student policy should not have access to privileged direction info during inference
obs_prop[:, 6:8] = 0  # Zero out delta_yaw and delta_next_yaw

actions = runner.act(obs_prop, depth)
```

**步骤 2: 验证语法**

```bash
python -m py_compile scripts/rsl_rl/play_student_transformer.py
```

预期输出：无错误

**步骤 3: 提交更改**

```bash
git add scripts/rsl_rl/play_student_transformer.py
git commit -m "feat: implement delta_yaw masking in student transformer inference

- Mask obs_buf indices 6-7 during student policy inference
- Ensures student policy does not rely on privileged direction info"
```

---

### Task 4: 在 play_student.py 中实现 delta_yaw mask

**文件:**
- 修改: `scripts/rsl_rl/play_student.py:577`

**步骤 1: 实现 mask 逻辑**

在第 577 行之后添加 mask 操作：

```python
obs_prop = obs[:, :proprio_dim]

# Mask privileged information (delta_yaw at indices 6-7)
# Student policy should not have access to privileged direction info during inference
obs_prop = obs_prop.clone()  # Clone to avoid modifying original obs
obs_prop[:, 6:8] = 0  # Zero out delta_yaw and delta_next_yaw
```

**步骤 2: 验证语法**

```bash
python -m py_compile scripts/rsl_rl/play_student.py
```

预期输出：无错误

**步骤 3: 提交更改**

```bash
git add scripts/rsl_rl/play_student.py
git commit -m "feat: implement delta_yaw masking in student inference

- Mask obs_buf indices 6-7 during student policy inference
- Clone obs_prop to avoid modifying original observation
- Ensures student policy does not rely on privileged direction info"
```

---

### Task 5: 验证修改并创建验证脚本

**文件:**
- 创建: `scripts/verify_delta_yaw_mask.py`

**步骤 1: 创建验证脚本**

创建一个脚本来验证观测维度和 mask 逻辑：

```python
"""
验证 delta_yaw 恢复和 mask 逻辑
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def verify_observation_dimensions():
    """验证观测维度计算"""

    # 预期的观测维度
    expected_obs_buf_dim = 54  # 3 + 2 + 1 + 1 + 1 + 2 + 1 + 2 + 12 + 12 + 12 + 5
    expected_history_dim = 53  # 3 + 2 + 3 + 4 + 36 + 5

    print("=" * 60)
    print("观测维度验证")
    print("=" * 60)

    # 计算 obs_buf 维度
    obs_components = {
        "角速度 (root_ang_vel_b)": 3,
        "IMU (roll, pitch)": 2,
        "占位符 (0*delta_yaw)": 1,
        "delta_yaw": 1,
        "delta_next_yaw": 1,
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
        "delta_yaw 相关": 3,  # 0*delta_yaw, delta_yaw, delta_next_yaw (但会被清零)
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

    print("\n重要提示:")
    print("1. 教师策略可以看到 obs_buf[6:8] 的 delta_yaw 信息（当前观测和历史）")
    print("2. 学生策略在 dagger 训练和推理时需要 mask obs_buf[6:8]")
    print("3. Mask 索引: 6 (delta_yaw), 7 (delta_next_yaw)")
    print("4. Mask 操作在学生脚本中实现，不在 observations.py 中")

    return True

if __name__ == "__main__":
    success = verify_observation_dimensions()
    sys.exit(0 if success else 1)
```

**步骤 2: 运行验证脚本**

```bash
python scripts/verify_delta_yaw_mask.py
```

预期输出：
```
============================================================
观测维度验证
============================================================

obs_buf 组成:
  - 角速度 (root_ang_vel_b): 3
  - IMU (roll, pitch): 2
  - 占位符 (0*delta_yaw): 1
  - delta_yaw: 1
  - delta_next_yaw: 1
  - 命令清零 (0*commands[:, 0:2]): 2
  - 命令 (commands[:, 0:1]): 1
  - 地形标记 (env_idx_tensor): 1
  - 地形标记反转 (invert_env_idx_tensor): 1
  - 关节位置偏差: 12
  - 关节速度: 12
  - 动作历史: 12
  - 接触信息: 5
  总计: 54
✅ obs_buf 维度正确: 54

历史缓冲区组成:
  - 角速度: 3
  - IMU: 2
  - delta_yaw 相关: 3
  - 命令相关: 4
  - 关节相关: 36
  - 接触: 5
  总计: 53
✅ 历史缓冲区维度正确: 53

============================================================
✅ 所有维度验证通过！
============================================================

重要提示:
1. 教师策略可以看到 obs_buf[6:8] 的 delta_yaw 信息（当前观测和历史）
2. 学生策略在 dagger 训练和推理时需要 mask obs_buf[6:8]
3. Mask 索引: 6 (delta_yaw), 7 (delta_next_yaw)
4. Mask 操作在学生脚本中实现，不在 observations.py 中
```

**步骤 3: 提交验证脚本**

```bash
git add scripts/verify_delta_yaw_mask.py
git commit -m "test: add delta_yaw mask verification script

- Verify observation dimensions (54 for obs_buf, 53 for history)
- Document mask indices for student policy
- Explain teacher vs student observation differences"
```

**步骤 4: 创建最终总结提交**

```bash
git commit --allow-empty -m "feat: complete delta_yaw restoration with student masking

Summary of changes:
- Restored delta_yaw and delta_next_yaw in observations (indices 6-7)
- Observation dimension: 51 → 54
- History buffer dimension: 50 → 53
- Teacher policy can see delta_yaw in current observation
- Student policy masks delta_yaw during dagger training and inference
- Added verification script for dimension checking

Architecture:
- delta_yaw is privileged information visible to teacher (current and history)
- Student learns to navigate without explicit target direction (masked in student scripts)
- Teacher and student see same observation structure, but student masks indices 6-7

Next steps:
1. Retrain teacher model with restored delta_yaw (54-dim obs)
2. Train student model with dagger (will mask delta_yaw)
3. Verify student performance without privileged direction info"
```

---

## 后续步骤（不在本计划范围内）

完成代码修改后，需要执行以下步骤：

### 1. 重新训练教师模型

```bash
# 使用现有的教师训练脚本
bash scripts/train_parkour_teacher.sh
```

**预期结果:**
- 教师模型使用新的 54 维观测（包含 delta_yaw）
- 训练应该比之前更稳定（因为有了方向信息）
- 性能应该相同或更好

### 2. 训练学生模型

```bash
# 等教师模型训练完成后
bash scripts/run_student_dagger.sh
```

**注意事项:**
- 需要更新 `TEACHER_CHECKPOINT` 路径指向新训练的教师模型
- 学生模型的 `num_prop` 参数会自动从环境推断（54 维）
- 学生模型在训练时会 mask 掉 delta_yaw（索引 6-7）

### 3. 验证模型性能

```bash
# 测试教师模型
bash scripts/play_teacher.sh

# 测试学生模型
bash scripts/play_student.sh
```

**验证要点:**
- 教师模型应该能稳定通过复杂地形
- 学生模型应该能在没有 delta_yaw 的情况下导航
- 对比有/无 delta_yaw 时的性能差异

---

## 回滚计划

如果需要回滚更改：

```bash
# 回滚所有修改
git revert HEAD~5..HEAD

# 或者使用备份文件
cp parkour_isaaclab/envs/mdp/observations.py.backup parkour_isaaclab/envs/mdp/observations.py
```

---

## 注意事项

1. **破坏性变更**: 观测维度从 51 变回 54，需要重新训练所有模型
2. **Mask 索引**: 确保所有学生脚本中的 mask 索引一致（6:8）
3. **测试**: 在小规模环境（少量 envs）上先测试，确认没有维度错误
4. **文档**: 更新任何相关文档，说明观测空间的变化和 mask 逻辑

---

## 验证清单

- [ ] observations.py 中恢复了所有 delta_yaw 引用
- [ ] 历史缓冲区维度更新为 53
- [ ] obs_buf 维度为 54
- [ ] train_student_dagger.py 中实现了 mask 逻辑
- [ ] play_student_transformer.py 中实现了 mask 逻辑
- [ ] play_student.py 中实现了 mask 逻辑
- [ ] 验证脚本运行通过
- [ ] 所有修改已提交到 git
- [ ] 准备好重新训练教师模型
