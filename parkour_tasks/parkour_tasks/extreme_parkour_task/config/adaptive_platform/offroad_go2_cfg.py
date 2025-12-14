# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config entries for generic offroad locomotion with Unitree Go2.

These configs are intentionally very close to the existing parkour teacher
configs so that training behavior matches the parkour setup by default.
They provide a separate env and RSL-RL runner that can be tuned for
offroad / all-terrain training without impacting parkour.
"""

from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlActorCfg,
    ParkourRslRlEstimatorCfg,
    ParkourRslRlOnPolicyRunnerCfg,
    ParkourRslRlPpoActorCriticCfg,
    ParkourRslRlPpoAlgorithmCfg,
    ParkourRslRlStateHistEncoderCfg,
)
from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import (
    UnitreeGo2TeacherParkourEnvCfg,
)


@configclass
class UnitreeGo2OffroadEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    """Unitree Go2 generic offroad env config.

    Currently identical to the parkour teacher env. Terrain, events and
    curriculum can be customized here later for more diverse offroad setups
    without touching the original parkour config.
    """


@configclass
class UnitreeGo2OffroadTeacherPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner for generic offroad training.

    Hyperparameters are copied from the parkour teacher runner so that,
    at this stage, training behavior matches the existing parkour setup.
    The key difference is a separate experiment_name, which isolates logs.
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "unitree_go2_offroad"
    empirical_normalization = False
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=ParkourRslRlActorCfg(
            class_name="Actor",
            state_history_encoder=ParkourRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder"
            ),
        ),
    )
    estimator = ParkourRslRlEstimatorCfg(
        hidden_dims=[128, 64],
    )
    depth_encoder = None
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0.0, 0.1, 2000.0, 3000.0],
    )

