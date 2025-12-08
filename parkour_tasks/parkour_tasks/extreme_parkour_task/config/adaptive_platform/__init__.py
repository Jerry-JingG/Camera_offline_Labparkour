# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the adaptive platform training pipeline."""

# We leave this file empty since we don't want to expose any configs in this package directly.
# We still need this file to import the "config" module in the parent package.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Adaptive-Platform-Teacher-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherAdaptivePlatformEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2AdaptivePlatformTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-Teacher-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherAdaptivePlatformEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2AdaptivePlatformTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-Teacher-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherAdaptivePlatformEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2AdaptivePlatformTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-Student-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentAdaptivePlatformEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2AdaptivePlatformStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-Student-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentAdaptivePlatformEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2AdaptivePlatformStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-Student-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentAdaptivePlatformEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2AdaptivePlatformStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-TeacherCam-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cam_cfg:UnitreeGo2TeacherCamAdaptivePlatformEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2AdaptivePlatformTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-TeacherCam-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cam_cfg:UnitreeGo2TeacherCamAdaptivePlatformEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2AdaptivePlatformTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Adaptive-Platform-TeacherCam-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cam_cfg:UnitreeGo2TeacherCamAdaptivePlatformEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2AdaptivePlatformTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_adaptive_platform_ppo_cfg.yaml",
    },
)
