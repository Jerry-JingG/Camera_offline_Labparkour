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

commit5&6:
大幅修改了train_student_from_dataset.py文件 
旧版代码batch_size!=num_envs并且切片得到的sequence_0, sequence_1, sequence_2是重叠的

TransformerXL网络在监督训练时，它的输入流是这样的：
输入batch0, batch1, batch2...  batch_i是不同环境同一段时间内教师模型与环境交互的切片
batch_i[j]与batch_i+1[j]必须是同一环境下连续的两片时间内教师模型与环境交互的切片
这样才可以训练transformerxl网络利用历史状态

因此：
batch_size必须等于num_envs，并且sequence_0, sequence_1, sequence_2应该相接！