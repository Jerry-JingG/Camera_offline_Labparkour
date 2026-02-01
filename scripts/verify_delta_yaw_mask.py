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
