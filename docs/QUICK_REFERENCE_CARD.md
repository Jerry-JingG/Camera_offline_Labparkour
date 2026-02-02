# 文档快速参考卡片

**项目：** RL Fine-tuning for Camera-based Parkour
**文档版本：** v1.1
**最后更新：** 2026-02-03

---

## 快速查找表

### 我想做什么？

| 需求 | 文档路径 | 预计阅读时间 |
|------|---------|-------------|
| 🚀 **快速开始训练** | [phases/PHASE4_QUICKREF.md](phases/PHASE4_QUICKREF.md) | 10 分钟 |
| 📚 **了解整体架构** | [architecture/2026-02-02-rl-finetuning-architecture.md](architecture/2026-02-02-rl-finetuning-architecture.md) | 15 分钟 |
| 🔧 **配置训练参数** | [config/student-finetune-config.md](config/student-finetune-config.md) | 15 分钟 |
| 🎯 **使用 ValueHead** | [phases/PHASE1_QUICKREF.md](phases/PHASE1_QUICKREF.md) | 10 分钟 |
| 🤖 **使用 PPOStudent** | [phases/PHASE2_QUICKREF.md](phases/PHASE2_QUICKREF.md) | 15 分钟 |
| 🌈 **配置领域随机化** | [guides/domain_randomization_quickref.md](guides/domain_randomization_quickref.md) | 10 分钟 |
| 🐛 **调试问题** | 各阶段 SUMMARY 的"经验教训"部分 | 按需 |
| 📖 **深入理解实现** | [phases/](phases/) 目录下的 SUMMARY 文档 | 2-3 小时 |

---

## 文档分类速查

### 📁 phases/ - 阶段实施文档

**Phase 1: 架构修改** ✅
- [PHASE1_SUMMARY.md](phases/PHASE1_SUMMARY.md) - 详细文档（19 KB）
- [PHASE1_QUICKREF.md](phases/PHASE1_QUICKREF.md) - 快速参考（5.1 KB）

**Phase 2: PPO 算法适配** ✅
- [PHASE2_SUMMARY.md](phases/PHASE2_SUMMARY.md) - 详细文档（25 KB）
- [PHASE2_QUICKREF.md](phases/PHASE2_QUICKREF.md) - 快速参考（11 KB）

**Phase 3: 领域随机化** ✅
- [PHASE3_SUMMARY.md](phases/PHASE3_SUMMARY.md) - 详细文档
- [PHASE3_QUICKREF.md](phases/PHASE3_QUICKREF.md) - 快速参考
- [phase3_implementation_summary.md](phases/phase3_implementation_summary.md) - 实施报告

**Phase 4: 训练配置** ✅
- [PHASE4_SUMMARY.md](phases/PHASE4_SUMMARY.md) - 详细文档（5.0 KB）
- [PHASE4_QUICKREF.md](phases/PHASE4_QUICKREF.md) - 快速参考（2.3 KB）
- [phase4-completion-report.md](phases/phase4-completion-report.md) - 完成报告

### 🏗️ architecture/ - 架构设计

- [2026-02-02-rl-finetuning-architecture.md](architecture/2026-02-02-rl-finetuning-architecture.md) - 完整架构设计

### 📋 plans/ - 设计计划

- [2026-02-02-rl-finetuning-design.md](plans/2026-02-02-rl-finetuning-design.md) - 详细实施方案

### ⚙️ config/ - 配置文档

- [student-finetune-config.md](config/student-finetune-config.md) - 训练配置说明

### 📖 guides/ - 使用指南

- [domain_randomization.md](guides/domain_randomization.md) - 领域随机化完整指南
- [domain_randomization_quickref.md](guides/domain_randomization_quickref.md) - 快速参考

---

## 学习路径

### 🌱 新手路径（1 小时）

```
1. 主 README (5 分钟)
   ↓
2. 架构设计 (15 分钟)
   ↓
3. Phase 1-4 快速参考 (40 分钟)
   ↓
4. 开始实践！
```

### 🌿 进阶路径（3 小时）

```
1. 完成新手路径
   ↓
2. Phase 1-4 详细文档 (2 小时)
   ↓
3. 领域随机化指南 (30 分钟)
   ↓
4. 配置文档 (15 分钟)
   ↓
5. 高级实践！
```

### 🌳 专家路径（全面掌握）

```
1. 完成进阶路径
   ↓
2. 设计计划文档
   ↓
3. 会话上下文
   ↓
4. 所有实施报告
   ↓
5. 贡献代码和文档！
```

---

## 常见问题快速定位

| 问题类型 | 查找位置 |
|---------|---------|
| 训练不收敛 | [phases/PHASE2_SUMMARY.md](phases/PHASE2_SUMMARY.md) - 经验教训 |
| 内存溢出 | [phases/PHASE2_SUMMARY.md](phases/PHASE2_SUMMARY.md) - 性能优化 |
| 配置参数不清楚 | [config/student-finetune-config.md](config/student-finetune-config.md) |
| 模块使用方法 | 对应阶段的 QUICKREF 文档 |
| 架构设计疑问 | [architecture/](architecture/) 目录 |
| 实现细节疑问 | 对应阶段的 SUMMARY 文档 |

---

## 文档更新记录

### v1.1 (2026-02-03)
- 重组文档目录结构
- 创建各子目录索引
- 改进导航系统

### v1.0 (2026-02-03)
- 完成所有阶段文档
- 创建主索引

---

## 获取帮助

1. **查看文档** - 从主 README 开始
2. **搜索关键词** - 使用 grep 搜索文档内容
3. **查看测试** - 测试文件包含使用示例
4. **查看代码** - 代码注释详细

---

**提示：** 将此文档加入书签，方便快速查找！

**文档根目录：** `/home/droplet/IsaacLab/Camera_offline_Labparkour/docs/`
