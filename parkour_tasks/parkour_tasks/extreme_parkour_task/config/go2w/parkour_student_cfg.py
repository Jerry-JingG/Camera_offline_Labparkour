# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student environment configurations for Go2W parkour tasks."""

from isaaclab.utils import configclass
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import ExtremeParkourRoughTerrainCfg
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import CAMERA_USD_CFG, CAMERA_CFG, VIEWER
from .parkour_teacher_cfg import ParkourGo2WTeacherSceneCfg


@configclass
class ParkourGo2WStudentSceneCfg(ParkourGo2WTeacherSceneCfg):
    depth_camera = CAMERA_CFG
    depth_camera_usd = None
    
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.1
        for key, sub_terrain in self.terrain.terrain_generator.sub_terrains.items():
            sub_terrain: ExtremeParkourRoughTerrainCfg
            sub_terrain.use_simplified = True 
            sub_terrain.horizontal_scale = 0.1
            if key == 'parkour_demo':
                sub_terrain.proportion = 0.15
            elif key == 'parkour_flat':
                sub_terrain.proportion = 0.05
            else:
                sub_terrain.proportion = 0.2
                if key != 'parkour':
                    sub_terrain.y_range = (-0.1, 0.1)


@configclass
class UnitreeGo2WStudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourGo2WStudentSceneCfg = ParkourGo2WStudentSceneCfg(num_envs=192, env_spacing=1.)
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: StudentRewardsCfg = StudentRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2WStudentParkourEnvCfg_EVAL(UnitreeGo2WStudentParkourEnvCfg):
    viewer = VIEWER 
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.commands.base_velocity.debug_vis = True
        self.scene.depth_camera_usd = CAMERA_USD_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.observations.depth_camera.depth_cam.params['debug_vis'] = True
        self.commands.base_velocity.resampling_time_range = (60., 60.)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6., 6.)
        self.events.random_camera_position.params['rot_noise_range'] = {'pitch': (0, 1)}


@configclass
class UnitreeGo2WStudentParkourEnvCfg_PLAY(UnitreeGo2WStudentParkourEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 60.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.events.push_by_setting_velocity = None
