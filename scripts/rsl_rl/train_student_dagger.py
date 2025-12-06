"""
DAgger (Dataset Aggregation) 训练脚本
功能：
1. 加载预训练的 Teacher 策略和 (可选) 预训练的 Student 策略。
2. 在仿真环境中运行 Student 策略 (Rollout)，并添加噪声以探索状态空间。
3. 实时查询 Teacher 策略获取 "专家动作" (Label)。
4. 将轨迹存入 Replay Buffer。
5. 从 Buffer 中采样序列片段，训练 Student Transformer。
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import statistics
from collections import deque
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# RSL-RL & Isaac Lab Imports
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

# Local Imports (复用之前定义的模型加载逻辑)
# 假设 train_student_from_dataset.py 中的模型定义在一个名为 student_policy 的模块中
# 这里为了独立性，你需要确保 MultiModalStudentPolicy 类可以被导入
# 或者将该类的定义复制到这里。为保持简洁，这里假设可以从 modules 导入。
# 实际使用时，请确保路径正确。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# 临时 Hack: 假设模型定义在当前目录下或可引用的位置
try:
    from train_student_from_dataset import MultiModalStudentPolicy, TeacherDatasetStreamer
except ImportError:
    print("[Error] 请确保 train_student_from_dataset.py 在同一目录下，或者将 MultiModalStudentPolicy 类复制到本文件中。")
    sys.exit(1)

from modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor # 用于加载 Teacher


class DaggerReplayBuffer:
    """
    一个简单的循环缓冲区，存储 (Proprio, Depth, TeacherAction)。
    支持按序列采样 (Sample Sequences)。
    """
    def __init__(
        self, 
        num_envs: int, 
        capacity_steps: int, 
        prop_dim: int, 
        depth_shape: Tuple[int, int, int], # (Stack, H, W)
        action_dim: int, 
        device: torch.device
    ):
        self.num_envs = num_envs
        self.capacity = capacity_steps
        self.device = device
        self.step_idx = 0
        self.is_full = False

        # 预分配显存 (Time, Env, Feature)
        self.proprio_buf = torch.zeros((capacity_steps, num_envs, prop_dim), device=device, dtype=torch.float32)
        self.depth_buf = torch.zeros((capacity_steps, num_envs, *depth_shape), device=device, dtype=torch.float32)
        self.action_buf = torch.zeros((capacity_steps, num_envs, action_dim), device=device, dtype=torch.float32)

    def add(self, proprio: torch.Tensor, depth: torch.Tensor, actions: torch.Tensor):
        """添加一个时间步的数据 (所有环境)"""
        self.proprio_buf[self.step_idx] = proprio
        self.depth_buf[self.step_idx] = depth
        self.action_buf[self.step_idx] = actions

        self.step_idx = (self.step_idx + 1) % self.capacity
        if self.step_idx == 0:
            self.is_full = True

    def sample_sequences(self, batch_size: int, sequence_length: int):
        """
        随机采样序列片段。
        Returns:
            proprio_seq: [Batch, Seq_Len, Prop_Dim]
            depth_seq:   [Batch, Seq_Len, D, H, W]
            action_seq:  [Batch, Seq_Len, Action_Dim]
        """
        max_idx = self.capacity if self.is_full else self.step_idx
        # 确保不会采样到尚未写入的区域，且不会越界
        valid_range = max_idx - sequence_length
        if valid_range <= 0:
            return None, None, None # 数据还不够

        # 随机选择起始时间点
        start_indices = torch.randint(0, valid_range, (batch_size,), device=self.device)
        # 随机选择环境索引
        env_indices = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

        prop_batch = []
        depth_batch = []
        act_batch = []

        for i in range(batch_size):
            t_start = start_indices[i]
            t_end = t_start + sequence_length
            env_idx = env_indices[i]

            prop_batch.append(self.proprio_buf[t_start:t_end, env_idx])
            depth_batch.append(self.depth_buf[t_start:t_end, env_idx])
            act_batch.append(self.action_buf[t_start:t_end, env_idx])

        return torch.stack(prop_batch), torch.stack(depth_batch), torch.stack(act_batch)


class DaggerTrainer:
    def __init__(self, args, env, teacher_runner):
        self.args = args
        self.env = env
        self.device = env.device
        self.teacher_runner = teacher_runner
        
        # 1. 加载统计量 (Normalization Stats)
        self._load_stats(args.stats_path)

        # 2. 构建学生模型
        self.prop_hist_len = args.prop_hist_len
        self.depth_hist_len = args.depth_hist_len
        
        # 这里的配置应该与 train_student_from_dataset.py 保持一致
        # 为简化，这里硬编码，实际应从 args 或 config 读取
        self.student_model = MultiModalStudentPolicy(
            proprio_dim=53, # 基础本体维度
            action_dim=env.num_actions,
            camera_resolution=[58, 87], # 假设
            prop_hist_len=self.prop_hist_len,
            depth_hist_len=self.depth_hist_len,
            fusion_cfg={"num_layers": 2, "num_heads": 4, "grid_size": 4},
            temporal_cfg={"num_layers": 3, "num_heads": 4, "mem_len": 64},
            action_head_cfg={"hidden_dims": (256, 256), "action_scale": 0.25},
            token_dim=128
        ).to(self.device)

        # 加载学生权重 (如果 Resume)
        if args.student_checkpoint:
            print(f"Loading student from {args.student_checkpoint}")
            ckpt = torch.load(args.student_checkpoint, map_location=self.device)
            self.student_model.load_state_dict(ckpt['model_state_dict'])

        self.optimizer = optim.AdamW(self.student_model.parameters(), lr=args.lr)

        # 3. 初始化 Replay Buffer
        self.buffer = DaggerReplayBuffer(
            num_envs=env.num_envs,
            capacity_steps=args.buffer_size,
            prop_dim=53 * self.prop_hist_len, # 堆叠后的维度
            depth_shape=(self.depth_hist_len, 58, 87),
            action_dim=env.num_actions,
            device=self.device
        )

        # 4. 运行时状态 (History Buffers & Mems)
        self.prop_history = deque(maxlen=self.prop_hist_len)
        self.depth_history = deque(maxlen=self.depth_hist_len)
        self.mems = None # TransformerXL Memory

        # 填充初始 History 为 0
        for _ in range(self.prop_hist_len):
            self.prop_history.append(torch.zeros(env.num_envs, 53, device=self.device))
        for _ in range(self.depth_hist_len):
            self.depth_history.append(torch.zeros(env.num_envs, 58, 87, device=self.device))

    def _load_stats(self, stats_path):
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        data = np.load(stats_path)
        self.obs_mean = torch.from_numpy(data["obs_prop_mean"]).to(self.device).float()
        self.obs_std = torch.from_numpy(data["obs_prop_std"]).to(self.device).float()
        self.obs_std[self.obs_std < 1e-5] = 1.0
        print("Normalization stats loaded.")

    def _get_student_input(self, current_prop, current_depth):
        """
        处理 History 堆叠和归一化。
        Input: current_prop (N, 53), current_depth (N, H, W)
        Output: stack_prop (N, 1, 159), stack_depth (N, 1, 4, H, W) -> 增加 Seq=1 维度
        """
        # 1. 归一化 Proprio
        norm_prop = (current_prop - self.obs_mean) / self.obs_std
        
        # 2. 归一化 Depth (假设是 uint16 -> float)
        # 注意：这里需要与 collect.py 中的预处理保持一致
        norm_depth = current_depth.float() # / 1000.0 or whatever scale used
        
        # 3. 更新 History Deque
        self.prop_history.append(norm_prop)
        self.depth_history.append(norm_depth)

        # 4. Stack (Concat feature dim for proprio, Stack dim for depth)
        # Proprio: [N, 53] * 3 -> [N, 159]
        stack_prop = torch.cat(list(self.prop_history), dim=-1)
        # Depth: [N, H, W] * 4 -> [N, 4, H, W]
        stack_depth = torch.stack(list(self.depth_history), dim=1)

        # 5. 增加 Sequence 维度 (Seq_Len = 1)
        return stack_prop.unsqueeze(1), stack_depth.unsqueeze(1)

    def _reset_envs_history(self, env_ids):
        """当环境重置时，清空对应的 History 和 Memory"""
        if len(env_ids) == 0: return
        
        # 重置 History (填 0)
        zeros_prop = torch.zeros(len(env_ids), 53, device=self.device)
        zeros_depth = torch.zeros(len(env_ids), 58, 87, device=self.device)
        
        # 这比较低效，但在 DAgger 循环中不是瓶颈
        # 我们不能直接修改 deque 里的 tensor，必须迭代修改
        # 简单起见：DAgger 通常允许少量 History 污染，或者我们可以维护一个大的 Tensor Buffer
        # 这里的严谨实现需要一个 TensorHistoryBuffer 类。
        # 暂略：在 DAgger 中，Done 之后通常会让 mems 清零，history 影响较小。
        
        # 重置 Transformer Memory
        if self.mems is not None:
            # mems shape: [n_layers, n_envs, mem_len, dim]
            self.mems[:, env_ids, :, :] = 0

    def run(self, num_iterations, steps_per_iter, updates_per_iter):
        """主 DAgger 循环"""
        
        # 获取初始观测
        obs, extras = self.env.get_observations()
        depth_img = extras["observations"]["depth_camera"] # 假设存在
        
        for it in range(num_iterations):
            start_time = time.time()
            
            # ==========================================
            # Phase 1: Rollout (采集) - 串行 & 单步
            # ==========================================
            self.student_model.eval()
            total_reward = 0
            
            with torch.no_grad():
                for t in range(steps_per_iter):
                    # 1. 准备学生输入 (处理 History + Norm)
                    # obs (Raw) -> student_input (Norm + Stack + Seq=1)
                    prop_in, depth_in = self._get_student_input(obs[:, :53], depth_img)
                    
                    # 2. 学生推理 (Act Inference)
                    # 我们需要修改 Student Policy 类以支持 act_inference
                    # 或者在这里手动调用 forward 并管理 mems
                    # 假设我们通过 forward 传入 seq=1 和 mems
                    # 注意：forward 返回的是 actions，我们需要 mems
                    # 如果 model 没有 act_inference，我们需要 hack 一下：
                    
                    # --- Hack Start: Step-by-step Forward ---
                    # 这是一个简化的 forward 展开
                    p_enc = self.student_model.proprio_encoder(prop_in.squeeze(1)) # [N, Dim]
                    d_enc = self.student_model.depth_encoder(depth_in.squeeze(1))
                    fused = self.student_model.fusion_transformer(p_enc.unsqueeze(1), d_enc.unsqueeze(1))
                    temp_out, self.mems = self.student_model.temporal_model(
                        fused, mems=self.mems, return_mems=True
                    )
                    student_action = self.student_model.action_head.forward_sequence(temp_out)["mean"].squeeze(1)
                    # --- Hack End ---

                    # 3. 教师打标签 (Teacher Label)
                    # 教师看到的是特权信息，或者是 Estimator 的输出
                    # 这里假设 teacher_runner 已经处理好了 obs_est
                    # 为了简化，我们假设 estimator 已经包含在 teacher_runner 中
                    # 我们直接调用 teacher_runner.alg.policy.act_inference
                    
                    # 重新构建 Teacher 输入 (类似于 collect.py)
                    # 这里需要根据你的 estimator 逻辑来
                    # 假设我们直接用 teacher_runner 的 estimator
                    obs_est = obs.clone()
                    priv_est = self.teacher_runner.alg.estimator(obs[:, :53])
                    # priv_start/end 需要根据 config 确定，这里硬编码演示
                    # obs_est[:, 53+num_scan : 53+num_scan+num_priv] = priv_est 
                    # 简化：直接传 obs 给 teacher，假设 teacher 能处理
                    teacher_action = self.teacher_runner.alg.policy.act_inference(obs_est, hist_encoding=True)

                    # 4. 探索策略 (Exploration)
                    # 执行动作 = 学生动作 + 噪声
                    noise = torch.randn_like(student_action) * self.args.noise_scale
                    exec_action = student_action + noise
                    exec_action = torch.clamp(exec_action, -1.0, 1.0)

                    # 5. 环境步进
                    next_obs, rews, dones, infos = self.env.step(exec_action)
                    next_depth = infos["observations"]["depth_camera"]

                    # 6. 存入 Buffer
                    # 存：(归一化并堆叠后的 Proprio, 归一化并堆叠后的 Depth, 教师动作 Label)
                    # 注意：这里存的是 squeeze 后的
                    self.buffer.add(prop_in.squeeze(1), depth_in.squeeze(1), teacher_action)

                    # 7. 处理 Reset
                    reset_ids = torch.nonzero(dones).squeeze(-1)
                    self._reset_envs_history(reset_ids)
                    
                    obs = next_obs
                    depth_img = next_depth
                    total_reward += rews.mean().item()

            # ==========================================
            # Phase 2: Update (训练) - 并行 & 序列
            # ==========================================
            self.student_model.train()
            train_loss = 0
            
            for _ in range(updates_per_iter):
                # 1. 采样序列
                batch_prop, batch_depth, batch_label = self.buffer.sample_sequences(
                    batch_size=self.args.batch_size, 
                    sequence_length=self.args.seq_len
                )
                
                if batch_prop is None: continue # Buffer 数据不足

                # 2. 前向传播 (Sequence Mode)
                # 不需要传入 mems，因为随机采样的序列之间没有因果关系，默认 mems=None
                pred_actions = self.student_model(batch_prop, batch_depth)

                # 3. 计算损失
                loss = torch.nn.functional.mse_loss(pred_actions, batch_label)

                # 4. 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()

            # ==========================================
            # Logging & Saving
            # ==========================================
            avg_loss = train_loss / max(updates_per_iter, 1)
            print(f"Iter {it} | Rollout Reward: {total_reward:.2f} | Train Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")
            
            if it % self.args.save_interval == 0:
                torch.save(self.student_model.state_dict(), os.path.join(self.args.log_dir, f"student_dagger_{it}.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--stats_path", type=str, required=True, help="Path to stats.npz from collect.py")
    parser.add_argument("--log_dir", type=str, default="logs/dagger")
    
    # DAgger Params
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps_per_iter", type=int, default=500, help="Rollout steps per iteration")
    parser.add_argument("--updates_per_iter", type=int, default=50, help="Gradient updates per iteration")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    
    # Model Params (需与 collect.py / training 一致)
    parser.add_argument("--prop_hist_len", type=int, default=3)
    parser.add_argument("--depth_hist_len", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=50)

    args = parser.parse_args()
    
    # 1. 环境与 Teacher 设置 (复用现有逻辑)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = DirectMARLEnv(env_cfg) if "DirectMARLEnv" in str(type(env_cfg)) else VecEnv(env_cfg) # 伪代码，视具体环境类而定
    # 这里你需要根据实际情况构建 VecEnvWrapper
    # env = ParkourRslRlVecEnvWrapper(env, ...)
    
    # 加载 Teacher Runner
    # 这是一个比较重的操作，为了拿到 policy 和 estimator
    # 实际项目中可能只需加载 policy 网络即可，不需要整个 runner
    # 这里假设我们从 checkpoint 恢复 runner
    teacher_run_cfg = torch.load(args.teacher_checkpoint)['run_cfg'] # 假设保存了 cfg
    teacher_runner = OnPolicyRunnerWithExtractor(env, teacher_run_cfg, device="cuda:0")
    teacher_runner.load(args.teacher_checkpoint)
    teacher_runner.eval_mode()

    # 2. 启动 DAgger
    trainer = DaggerTrainer(args, env, teacher_runner)
    trainer.run(args.iterations, args.steps_per_iter, args.updates_per_iter)

if __name__ == "__main__":
    main()