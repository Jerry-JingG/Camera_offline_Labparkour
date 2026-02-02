# 语言设置 (Language Settings)

## 默认语言：中文

**关键要求**：在此项目中，所有与用户的交互、回答和生成的文档都必须使用**简体中文**。

### 适用范围

使用中文的场景：
- 所有对用户的回答和解释
- 生成的 Markdown 文档（.md 文件）
- 代码注释（除非是英文项目的标准注释）
- 提交信息（commit messages）可以使用中文或英文
- 错误信息的解释
- 技术方案的讨论

### 例外情况

以下内容可以保持英文：
- 代码本身（变量名、函数名、类名等）
- 配置文件中的键名
- 技术术语的原文（可在括号中标注，如：强化学习 (Reinforcement Learning)）
- 第三方库和框架的名称
- 命令行指令

### 示例

```markdown
# 正确示例

## 项目说明
这是一个基于 Isaac Lab 的强化学习项目，用于训练机器人进行跑酷任务。

## 安装步骤
1. 激活 conda 环境：`conda activate parkour`
2. 安装依赖包
3. 运行训练脚本

# 错误示例（不要这样做）

## Project Description
This is a reinforcement learning project based on Isaac Lab...
```

### 代码注释示例

```python
# 正确：使用中文注释
def train_policy(env, config):
    """
    训练策略网络

    参数:
        env: 训练环境
        config: 配置参数

    返回:
        训练好的策略模型
    """
    # 初始化训练器
    trainer = PPOTrainer(config)

    # 开始训练循环
    for epoch in range(config.num_epochs):
        # 收集经验数据
        data = collect_rollouts(env)

        # 更新策略
        trainer.update(data)

    return trainer.policy
```

## 重要提醒

- 即使用户使用英文提问，也应该用中文回答（除非用户明确要求使用英文）
- 技术术语可以中英文混用，但解释部分必须用中文
- 保持专业性和准确性，使用规范的技术中文表达
