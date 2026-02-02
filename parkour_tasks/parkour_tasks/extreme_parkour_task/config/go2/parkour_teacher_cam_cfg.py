# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""教师策略版本的跑酷环境（附加深度相机用于数据采集）。"""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import ( 
    randomize_rigid_body_mass,
    apply_external_force_torque,
    reset_joints_by_scale
)

from parkour_tasks.default_cfg import CAMERA_CFG, CAMERA_USD_CFG
from .parkour_teacher_cfg import (
    ParkourTeacherSceneCfg,
    UnitreeGo2TeacherParkourEnvCfg,
    UnitreeGo2TeacherParkourEnvCfg_EVAL,
    UnitreeGo2TeacherParkourEnvCfg_PLAY,
)
from .parkour_mdp_cfg import *
from parkour_isaaclab.envs.mdp import events


@configclass
class ParkourTeacherCamSceneCfg(ParkourTeacherSceneCfg):
    """在教师场景基础上添加深度相机资产，用于只读采集。"""

    depth_camera = CAMERA_CFG
    depth_camera_usd = CAMERA_USD_CFG


@configclass
class TeacherWithCameraObservationsCfg(TeacherObservationsCfg):
    """在教师观测上附加深度相机观测组，但不影响原有策略输入。"""

    depth_camera: StudentObservationsCfg.DepthCameraPolicyCfg = (
        StudentObservationsCfg.DepthCameraPolicyCfg()
    )


@configclass
class UnitreeGo2TeacherCamParkourEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    """训练版：保持教师配置，只是提供额外的深度相机观测。"""

    scene: ParkourTeacherCamSceneCfg = ParkourTeacherCamSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()


@configclass
class UnitreeGo2TeacherCamParkourEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    """评估版：支持 GUI/调试，可选 256 并行环境。"""

    scene: ParkourTeacherCamSceneCfg = ParkourTeacherCamSceneCfg(num_envs=256, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()


@configclass
class UnitreeGo2TeacherCamParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    """回放/数据采集版：默认 16 并行环境（可被 CLI 参数覆盖）。"""

    scene: ParkourTeacherCamSceneCfg = ParkourTeacherCamSceneCfg(num_envs=16, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()


@configclass
class UnitreeGo2TeacherCamParkourEnvCfg_COLLECT(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    """数据采集配置：保留了域随机化项以拓宽数据集coverage"""
    scene: ParkourTeacherCamSceneCfg = ParkourTeacherCamSceneCfg(num_envs=16, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 恢复相机位置随机化（教师策略中被禁用）
        self.events.random_camera_position = EventTerm(
            func=events.random_camera_position,
            mode="startup",
            params={
                'sensor_cfg': SceneEntityCfg("depth_camera"),
                'rot_noise_range': {'pitch': (-1, 1)},  
                'convention': 'ros',
            },
        )

        # 恢复并增强周期性推力扰动，这个应该是最有效的
        self.events.push_by_setting_velocity = EventTerm(
            func=events.push_by_setting_velocity,
            params={
                'velocity_range': {
                    "x": (-0.6, 0.6),  # 比训练时略大（训练为±0.5）
                    "y": (-0.6, 0.6),
                }
            },
            interval_range_s=(4., 8.),  # 随机间隔4-8秒
            is_global_time=True,
            mode="interval",
        )

        # 恢复质量随机化
        self.events.randomize_rigid_body_mass = EventTerm(
            func=randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-0.5, 2.0),  # 比训练时稍窄（训练为-1~3）
                "operation": "add",
            },
        )

        # 恢复质心随机化
        self.events.randomize_rigid_body_com = EventTerm(
            func=events.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    'x': (-0.015, 0.015),  # 比训练时稍小（训练为±0.02）
                    'y': (-0.015, 0.015),
                    'z': (-0.015, 0.015),
                }
            },
        )

        # 激活外部力矩扰动（在 PLAY 中被禁用）
        self.events.base_external_force_torque = EventTerm(
            func=apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (0.0, 0.0),      # 不添加持续力
                "torque_range": (-0.5, 0.5),    # 添加随机扭矩 ±0.5 Nm
            },
        )

        # 增大关节初始位置扰动（让教师从更多样的状态开始）
        self.events.reset_robot_joints = EventTerm(
            func=reset_joints_by_scale,
            params={
                "position_range": (0.90, 1.10),  # ±10%（训练为±5%）
                "velocity_range": (-0.1, 0.1),   # 添加初始速度（训练为0）
            },
            mode="reset",
        )
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)  # 更频繁地改变目标
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 1.0)  # 扩大速度范围
        self.commands.base_velocity.ranges.heading = (-1.8, 1.8)   # 扩大航向范围
