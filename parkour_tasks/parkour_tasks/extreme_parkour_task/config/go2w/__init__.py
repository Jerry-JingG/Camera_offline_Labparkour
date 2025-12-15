# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for Go2W (wheeled quadruped) parkour environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Teacher environments
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2W-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2WTeacherParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2WParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2W-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2WTeacherParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2WParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2W-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2WTeacherParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2WParkourTeacherPPORunnerCfg",
    },
)

# Student environments
gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2W-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2WStudentParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2WParkourStudentPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2W-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2WStudentParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2WParkourStudentPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2W-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2WStudentParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2WParkourStudentPPORunnerCfg",
    },
)

# Teacher with camera environments
gym.register(
    id="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2W-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cam_cfg:UnitreeGo2WTeacherCamParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2WParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2W-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cam_cfg:UnitreeGo2WTeacherCamParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2WParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2W-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cam_cfg:UnitreeGo2WTeacherCamParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2WParkourTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2W-Collect-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cam_cfg:UnitreeGo2WTeacherCamParkourEnvCfg_COLLECT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2WParkourTeacherPPORunnerCfg",
    },
)
