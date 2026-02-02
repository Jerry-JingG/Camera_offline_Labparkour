# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations and Gym registrations for generic offroad locomotion tasks.

For now, the offroad task reuses the existing Go2 parkour teacher configuration
and RSL-RL hyperparameters. This keeps behavior identical to the parkour
teacher task while providing a separate task ID and config entry-points that
can be customized later without affecting the parkour setup.
"""

import gymnasium as gym

from ..go2 import agents as go2_agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Generic-Offroad-Teacher-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Offroad env & agent configs live in this submodule. They currently
        # mirror the parkour teacher configs but can diverge independently.
        "env_cfg_entry_point": f"{__name__}.offroad_go2_cfg:UnitreeGo2OffroadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.offroad_go2_cfg:UnitreeGo2OffroadTeacherPPORunnerCfg",
        # Reuse the same SKRL PPO config as parkour for now.
        "skrl_cfg_entry_point": f"{go2_agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

