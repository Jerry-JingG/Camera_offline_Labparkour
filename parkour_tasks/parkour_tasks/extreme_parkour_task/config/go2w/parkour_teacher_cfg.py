# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher environment configurations for Go2W parkour tasks."""

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import EXTREME_PARKOUR_TERRAINS_CFG
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import ParkourGo2WDefaultSceneCfg, VIEWER

@configclass
class ParkourGo2WTeacherSceneCfg(ParkourGo2WDefaultSceneCfg):
    """Teacher scene config for Go2W with height scanner and contact sensors."""
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=2, 
        track_air_time=True, 
        debug_vis=False,
        force_threshold=1.
    )
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_CFG
        
@configclass
class UnitreeGo2WTeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    """Training environment configuration for Go2W teacher policy."""
    scene: ParkourGo2WTeacherSceneCfg = ParkourGo2WTeacherSceneCfg(num_envs=6144, env_spacing=1.)
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
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
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1
        self.events.random_camera_position = None

@configclass
class UnitreeGo2WTeacherParkourEnvCfg_EVAL(UnitreeGo2WTeacherParkourEnvCfg):
    viewer = VIEWER 

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6., 6.)
        self.commands.base_velocity.resampling_time_range = (60., 60.)
                
@configclass
class UnitreeGo2WTeacherParkourEnvCfg_PLAY(UnitreeGo2WTeacherParkourEnvCfg_EVAL):
    viewer = VIEWER 

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.
        self.scene.num_envs = 16
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.events.push_by_setting_velocity = None
