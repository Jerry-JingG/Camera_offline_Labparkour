# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher environment with camera for Go2W parkour tasks."""

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
    ParkourGo2WTeacherSceneCfg,
    UnitreeGo2WTeacherParkourEnvCfg,
    UnitreeGo2WTeacherParkourEnvCfg_EVAL,
    UnitreeGo2WTeacherParkourEnvCfg_PLAY,
)
from .parkour_mdp_cfg import *
from parkour_isaaclab.envs.mdp import events


@configclass
class ParkourGo2WTeacherCamSceneCfg(ParkourGo2WTeacherSceneCfg):
    depth_camera = CAMERA_CFG
    depth_camera_usd = CAMERA_USD_CFG


@configclass
class TeacherWithCameraObservationsCfg(TeacherObservationsCfg):
    depth_camera: StudentObservationsCfg.DepthCameraPolicyCfg = (
        StudentObservationsCfg.DepthCameraPolicyCfg()
    )


@configclass
class UnitreeGo2WTeacherCamParkourEnvCfg(UnitreeGo2WTeacherParkourEnvCfg):
    scene: ParkourGo2WTeacherCamSceneCfg = ParkourGo2WTeacherCamSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()


@configclass
class UnitreeGo2WTeacherCamParkourEnvCfg_EVAL(UnitreeGo2WTeacherParkourEnvCfg_EVAL):
    scene: ParkourGo2WTeacherCamSceneCfg = ParkourGo2WTeacherCamSceneCfg(num_envs=256, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()


@configclass
class UnitreeGo2WTeacherCamParkourEnvCfg_PLAY(UnitreeGo2WTeacherParkourEnvCfg_PLAY):
    scene: ParkourGo2WTeacherCamSceneCfg = ParkourGo2WTeacherCamSceneCfg(num_envs=16, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()


@configclass
class UnitreeGo2WTeacherCamParkourEnvCfg_COLLECT(UnitreeGo2WTeacherCamParkourEnvCfg_PLAY):
    scene: ParkourGo2WTeacherCamSceneCfg = ParkourGo2WTeacherCamSceneCfg(num_envs=16, env_spacing=1.0)
    observations: TeacherWithCameraObservationsCfg = TeacherWithCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.random_camera_position = EventTerm(
            func=events.random_camera_position,
            mode="startup",
            params={
                'sensor_cfg': SceneEntityCfg("depth_camera"),
                'rot_noise_range': {'pitch': (-1, 1)},  
                'convention': 'ros',
            },
        )
        self.events.push_by_setting_velocity = EventTerm(
            func=events.push_by_setting_velocity,
            params={'velocity_range': {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
            interval_range_s=(4., 8.),
            is_global_time=True,
            mode="interval",
        )
        self.events.randomize_rigid_body_mass = EventTerm(
            func=randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-0.5, 2.0),
                "operation": "add",
            },
        )
        self.events.randomize_rigid_body_com = EventTerm(
            func=events.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {'x': (-0.015, 0.015), 'y': (-0.015, 0.015), 'z': (-0.015, 0.015)}
            },
        )
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 1.0)
        self.commands.base_velocity.ranges.heading = (-1.8, 1.8)
