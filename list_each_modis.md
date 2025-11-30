# Xuanheng

commmit1&2: 添加parkour_tasks/pyproject.toml和parkour_isaaclab/__init__.py以正确安装环境

commit3:
修改了collect.py，扰动逻辑位于413行。如要启用，使用--noised_action argument
env.step()采用的是扰动后的动作，但采集的action是未加扰动的。采集的数据用于模仿学习，所以我们希望给学生模型用作label的action是没有扰动的
考虑了向量化环境，生成一个随机掩码，决定哪些环境在这个 step 使用噪声

commit4:
在collect.py中添加了观测扰动代码，位于439行。如要启用，使用--noised_observation argument
在parkour_teacher_cam_cfg.py中添加了UnitreeGo2TeacherCamParkourEnvCfg_COLLECT，通过定制self.events实现了域随机化
如果需要使用域随机化，修改run_collect.sh中的TASK_ID。已在__init__.py中注册COLLECT环境