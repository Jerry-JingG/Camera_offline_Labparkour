# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""教师策略版本的跑酷环境（附加深度相机用于数据采集）。"""

from isaaclab.utils import configclass

from parkour_tasks.default_cfg import CAMERA_CFG, CAMERA_USD_CFG
from .parkour_teacher_cfg import (
    ParkourTeacherSceneCfg,
    UnitreeGo2TeacherParkourEnvCfg,
    UnitreeGo2TeacherParkourEnvCfg_EVAL,
    UnitreeGo2TeacherParkourEnvCfg_PLAY,
)
from .parkour_mdp_cfg import StudentObservationsCfg, TeacherObservationsCfg


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
