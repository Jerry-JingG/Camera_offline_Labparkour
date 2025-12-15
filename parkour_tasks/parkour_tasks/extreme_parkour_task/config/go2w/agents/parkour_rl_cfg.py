# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class ParkourRslRlBaseCfg:
    """Base config for Go2W parkour (16 joints: 12 legs + 4 wheels)."""
    num_priv_explicit: int = 3 + 3 + 3  # 9
    # Go2W: 4 (mass) + 1 (friction) + 16 (stiffness) + 16 (damping) = 37
    num_priv_latent: int = 4 + 1 + 16 + 16  # 37
    # Go2W: 3 (ang_vel) + 2 (imu) + 3 (delta_yaw etc) + 4 (cmd) + 48 (joints*3) + 5 (contact) = 65
    # Actually from obs: ang_vel(3) + imu(2) + delta_yaw*3(3) + cmd(4) + env_idx(2) + joint_pos(16) + joint_vel(16) + action(12) + contact(4) = 62
    # But action only has 12 joints (legs), so: 3+2+3+4+2+16+16+12+4 = 62... 
    # Wait, let me check again: obs_buf includes action_history which is 12 joints only
    num_prop: int = 3 + 2 + 3 + 4 + 16 + 16 + 12 + 5  # 61  (Go2W: 16 joint_pos + 16 joint_vel + 12 actions + 5 contact/env)
    num_scan: int = 132
    num_hist: int = 10
    
@configclass
class ParkourRslRlStateHistEncoderCfg(ParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder" 
    channel_size: int = 10 
    
@configclass
class ParkourRslRlDepthEncoderCfg(ParkourRslRlBaseCfg):
    backbone_class_name: str = "DepthOnlyFCBackbone58x87" 
    encoder_class_name: str = "RecurrentDepthBackbone" 
    depth_shape: tuple[int] = (87, 58)
    hidden_dims: int = 512
    learning_rate: float = 1.e-3
    num_steps_per_env: int = 24 * 5

@configclass
class ParkourRslRlEstimatorCfg(ParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator" 
    train_with_estimated_states: bool = True 
    learning_rate: float = 1.e-4 
    hidden_dims: list[int] = MISSING 
    
@configclass
class ParkourRslRlActorCfg(ParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: ParkourRslRlStateHistEncoderCfg = MISSING

@configclass
class ParkourRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = 'ActorCriticRMA'
    tanh_encoder_output: bool = False 
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    actor: ParkourRslRlActorCfg = MISSING

@configclass
class ParkourRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'PPOWithExtractor'
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float] = [0, 0.1, 2000, 3000]

@configclass
class ParkourRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DistillationWithExtractor"

@configclass
class ParkourRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: ParkourRslRlPpoActorCriticCfg = MISSING
    estimator: ParkourRslRlEstimatorCfg = MISSING
    depth_encoder: ParkourRslRlDepthEncoderCfg | None = None
    algorithm: ParkourRslRlPpoAlgorithmCfg | ParkourRslRlDistillationAlgorithmCfg = MISSING
