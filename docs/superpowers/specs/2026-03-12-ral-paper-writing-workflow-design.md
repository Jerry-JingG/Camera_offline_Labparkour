# RAL 论文写作工作流设计文档

**日期**：2026-03-12
**项目**：Camera_offline_Labparkour（Go2 机器狗跑酷，相机故障鲁棒性）
**目标期刊**：IEEE Robotics and Automation Letters (RAL)

---

## 1. 概述

本文档描述一套在 VSCode 中运行的 Claude Code agent team 工作流，用于辅助撰写 RAL 学术论文。工作流采用**中央调度器模式**：用户只与主 agent（`paper-director`）对话，主 agent 在用户确认写作思路后并行调度专职子 agents 完成代码分析、风格学习、LaTeX 写作、参考文献查找和 AI 率审查。

---

## 2. 论文核心贡献

- 提出基于 Transformer Encoder + Transformer XL 的学生策略 DAgger 训练框架
- 处理视觉-本体多模态感知，赋予 Go2 机器狗长期记忆
- 解决相机掉线/黑屏故障下的鲁棒跑酷问题
- MuJoCo sim2sim 验证，准备 sim2real 部署

---

## 3. Agent 架构

### 3.1 中央调度器模式

```
用户
 ↕ 对话 / 审批
paper-director（主 Agent）
 ↓ 并行调度（达成共识后）
┌─────────────────────────────────────────┐
│ code-analyzer   style-analyzer          │
│ latex-writer    reference-finder        │
│ ai-reviewer                             │
└─────────────────────────────────────────┘
 ↓
VSCode LaTeX 工作区（IEEE RAL 模板）
```

### 3.2 Agent 列表

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| `paper-director` | 主 agent，与用户讨论写作思路，调度子 agents | 用户对话 | 写作决策、调度指令 |
| `code-analyzer` | 读取代码仓库，提取算法/网络结构/实验配置 | 代码仓库 | `context/code_summary.md` |
| `style-analyzer` | 分析示范 PDF 论文，提取写作风格 | `example_papers/*.pdf` | `context/style_guide.md` |
| `latex-writer` | 基于摘要和风格指南起草各章节 LaTeX | `code_summary.md` + `style_guide.md` | `sections/*.tex` |
| `reference-finder` | 联网搜索相关文献，生成 BibTeX 条目 | 章节内容 | `references.bib` + `\cite{}` |
| `ai-reviewer` | 检测 AI 写作特征，提出人性化修改建议 | `sections/*.tex` | 逐句反馈报告 |

---

## 4. 工作空间结构

采用**独立仓库 + VSCode Multi-root Workspace** 方案，彻底隔离代码和论文。

### 4.1 Agent 配置文件位置

所有 6 个 agent 配置文件放在**用户主目录**的全局 agents 目录，Claude Code 可识别：

```
~/.claude/agents/
├── paper-director.md
├── code-analyzer.md
├── style-analyzer.md
├── latex-writer.md
├── reference-finder.md
└── ai-reviewer.md
```

> 注：每个 agent 配置文件中必须用**绝对路径**引用两个仓库，不依赖相对路径或 VSCode workspace 上下文。例如：
> - 代码仓库：`/home/droplet/IsaacLab/Camera_offline_Labparkour/`
> - 论文仓库：`/home/droplet/ral-camera-fault-paper/`（Phase 0 初始化时确定）

### 4.2 代码仓库（现有，不变）

```
Camera_offline_Labparkour/
├── parkour_tasks/
├── scripts/
└── ...（现有代码不变）
```

### 4.3 论文仓库（新建）

```
ral-camera-fault-paper/
├── main.tex                  ← IEEE RAL 双栏模板入口
├── references.bib            ← reference-finder 自动维护
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experiments.tex
│   └── conclusion.tex
├── figures/                  ← 图表（手动放置）
├── example_papers/           ← 用户提供的示范 PDF（加入 .gitignore，勿提交）
├── context/
│   ├── code_summary.md       ← code-analyzer 输出
│   └── style_guide.md        ← style-analyzer 输出（格式见第 8 节）
└── .gitignore                ← 排除 *.aux *.log *.pdf *.synctex.gz example_papers/
```

> **版权提示**：`example_papers/` 中的 IEEE 论文 PDF 受版权保护，必须加入 `.gitignore`，仅本地保存，不提交到远程仓库。

### 4.3 VSCode Multi-root Workspace

在两个仓库的父目录创建 `parkour-paper.code-workspace`：

```json
{
  "folders": [
    { "name": "代码仓库", "path": "./Camera_offline_Labparkour" },
    { "name": "论文仓库", "path": "./ral-camera-fault-paper" }
  ],
  "settings": {
    "latex-workshop.latex.autoBuild.run": "onSave"
  }
}
```

---

## 5. 完整工作流程

### Phase 0：一次性初始化

1. 新建论文仓库，初始化 IEEE RAL LaTeX 模板，记录其绝对路径
2. 安装 VSCode 扩展：LaTeX Workshop
3. 创建 `.code-workspace` 文件
4. 将示范 PDF 放入 `example_papers/`，并将该目录加入 `.gitignore`
5. 在 `~/.claude/agents/` 下创建 6 个 agent 配置文件，**在每个文件中硬编码两个仓库的绝对路径**
6. 在论文仓库中安装 `pdftotext`（`poppler-utils` 包），供 `style-analyzer` 使用

### Phase 1：启动阶段（并行）

`paper-director` 同时调度：
- `code-analyzer`：扫描代码仓库的 `parkour_tasks/`、`scripts/rsl_rl/`、`*.py` 文件，重点提取：
  - 网络结构定义（Transformer Encoder + Transformer XL）
  - DAgger 训练逻辑
  - 观测空间设计（视觉 + 本体）
  - 实验配置和超参数
  - 输出 `context/code_summary.md`
- `style-analyzer`：
  1. 用 `pdftotext` 将 `example_papers/*.pdf` 转为 `.txt`
  2. 读取文本，提取写作风格特征
  3. 输出 `context/style_guide.md`（格式见第 8 节）

两者完成后，`paper-director` 向用户汇报分析结果。

### Phase 2：讨论阶段（强制门控）

`paper-director` 基于代码摘要与用户讨论：
- 论文整体结构和各节重点
- 核心 claim 和贡献表述
- 图表安排

**用户明确说"可以开始写了"之前，不产生任何 LaTeX 内容。**

### Phase 3：逐节写作循环

对每个章节重复以下循环（**单节最多 3 轮修订**）：

```
latex-writer 起草 .tex
        ↓
reference-finder ──┐  （并行）
ai-reviewer      ──┘
        ↓
latex-writer 按 ai-reviewer 反馈修订
        ↓
用户审阅 → 满意则进入下一节，否则继续循环
```

**终止条件**：
- 用户满意 → 进入下一节
- 已修订 3 轮仍未通过 → `paper-director` 向用户汇报，请求人工干预

### Phase 4：完稿阶段

1. 全文整合，检查章节衔接
2. **对完整 `main.tex` 运行 `ai-reviewer` 全文扫描**（检查章节衔接处的 AI 写作特征）
3. LaTeX Workshop 编译 PDF
4. 检查格式符合 IEEE RAL 要求（页数、图表格式、参考文献格式）
5. `references.bib` 去重和格式统一
6. 最终 PDF 输出

---

## 6. ai-reviewer 启发式规则

纯启发式实现，无需外部 API，检测以下 AI 写作特征：

| 特征 | 检测方式 |
|------|---------|
| 句子长度均匀 | 统计句子字数方差，方差过低则标记 |
| AI 高频词 | 检测 Furthermore / Moreover / It is worth noting / In this paper we propose 等 |
| 被动语态堆砌 | 统计被动句比例，超过 40% 则标记 |
| 表述笼统 | 检测缺乏具体数字的段落 |
| 段落三段式 | 检测段落结构是否过于规整 |
| 风格偏差 | 与 `style_guide.md` 中提取的句式模式对比 |

---

## 7. reference-finder 实现策略

- 优先搜索 **Semantic Scholar API**（免费，结构化数据）
- 补充搜索 **arXiv API**（机器人/强化学习领域）
- 自动生成标准 BibTeX 格式条目
- 在对应 `.tex` 文件中插入 `\cite{key}` 标记
- 避免重复引用（检查 `references.bib` 中已有条目）

---

## 8. style_guide.md 输出格式规范

`style-analyzer` 输出的 `context/style_guide.md` 必须包含以下结构化字段，供 `ai-reviewer` 和 `latex-writer` 使用：

```markdown
# 写作风格指南

## 1. 常用句式列表
- 示例句式 1
- 示例句式 2
...

## 2. 段落长度统计
- 平均段落长度：X 句
- 段落长度范围：Y-Z 句

## 3. 句子长度统计
- 平均句子长度：X 词
- 句子长度方差：Y

## 4. 引用密度
- 平均每段引用数：X 个

## 5. 术语使用习惯
- 高频术语：term1, term2, ...
- 避免使用的词：word1, word2, ...

## 6. 图表描述方式
- 示例描述 1
- 示例描述 2
```

此格式确保 `ai-reviewer` 可以解析并与待审查文本对比。
