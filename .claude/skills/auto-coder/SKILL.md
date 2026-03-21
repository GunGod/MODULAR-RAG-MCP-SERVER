---
name: auto-coder
description: Autonomous spec-driven development agent. Syncs DEV_SPEC.md into chapter-based reference files, identifies the next pending task from the schedule, implements code following spec architecture and patterns, runs tests with up to 3 auto-fix rounds, and persists progress with atomic commits. Use when user says "auto code", "自动开发", "自动写代码", "auto dev", "一键开发", "autopilot", or wants fully automated spec-to-code workflow.
---

# Auto Coder

> **⚠️ 核心原则：宁可慢，不可乱**
>
> **严格按照以下步骤执行，不得跳过任何步骤，不得自作聪明改变流程。**
>
> 每个步骤都有其存在的意义，跳过步骤会导致：
> - 配置不同步
> - 进度跟踪混乱
> - 代码质量下降
> - 难以回溯和调试
>
> **记住：慢即是快。严格按照流程执行，避免返工才是最快的。**

One trigger completes **read spec → find task → code → test → persist progress**.

Optional modifiers: append a task ID (e.g. `auto code B2`) to target a specific task, or `--no-commit` to skip git commit.

---

## Pipeline (严格按顺序执行，不得跳过)

```
┌────────────────────────────────────────────────────────────────┐
│  1. Sync Spec → 2. Find Task → 3. Implement → 4. Test → 5. Persist │
│                                                                 │
│  ⚠️ 每一步都要显式完成，不要合并，不要跳过，不要自作主张         │
└────────────────────────────────────────────────────────────────┘
```

**关键原则：**
- ✅ 每个步骤开始前，先确认上一步已完成
- ✅ 每个步骤完成后，输出明确的完成标识
- ✅ 不要"顺便做"、"一起做"（这是跳过步骤的借口）
- ✅ 遇到疑问时，停下来确认，不要假设
- ✅ 宁可多花时间确认，也不要匆忙执行错误

**Pause only at the end for commit confirmation. Run everything else autonomously.**

> **⚠️ CRITICAL: Activate `.venv` before ANY `python`/`pytest` command (idempotent, re-run if unsure).**
> - **Windows**: `.\.venv\Scripts\Activate.ps1`
> - **macOS/Linux**: `source .venv/bin/activate`

## Reference Map

All files under `.claude/skills/auto-coder/references/`:

| File | Content | When to Read |
|------|---------|-------------|
| `01-overview.md` | Project overview & goals | First task or when needing project context |
| `02-features.md` | Feature specifications | When implementing feature-related tasks |
| `03-tech-stack.md` | Tech stack & dependencies | When choosing libraries or patterns |
| `04-testing.md` | Testing conventions | When writing tests |
| `05-architecture.md` | Architecture & module design | When creating/modifying modules |
| `06-schedule.md` | Task schedule & status | Every cycle (Sync Spec step) |
| `07-future.md` | Future roadmap | When planning or assessing scope |

---

### 1. Sync Spec（同步规范 - 必须执行！）

> **⚠️ 常见错误：跳过这一步，直接开始实现**
>
> **后果：** 使用过期的规范，导致实现不符合要求
>
> **正确做法：** 每个任务开始前，必须先运行同步脚本

```powershell
python .claude/skills/auto-coder/scripts/sync_spec.py
```

✅ **完成标识：** 看到 `synced N chapters` 输出

Then read the schedule file to get task statuses:
- Read `.claude/skills/auto-coder/references/06-schedule.md`

Task markers:

| Marker | Status |
|--------|--------|
| `[ ]` / `⬜` | Not started |
| `[~]` / `🔶` / `(进行中)` | In progress |
| `[x]` / `✅` / `(已完成)` | Completed |

---

### 2. Find Task

Pick the first `IN_PROGRESS` task, then the first `NOT_STARTED`. If user specified a task ID, use that directly.

Quick-check predecessor artifacts exist (file-level only). On mismatch, log a warning and continue — only stop if the target task itself is blocked.

---

### 3. Implement（实现 - 按子步骤顺序执行）

> **⚠️ 常见错误：跳过规划直接写代码，或者写完代码再补测试**
>
> **后果：** 代码结构混乱，测试覆盖率不足，返工率高
>
> **正确做法：** 严格按照3-1到3-6的顺序，每步完成后再进行下一步

3.1. **Read relevant spec** from `.claude/skills/auto-coder/references/`:
   - Architecture: `05-architecture.md`
   - Tech details: `03-tech-stack.md`
   - Testing conventions: `04-testing.md`

   ✅ **完成标识：** 明确列出任务的inputs/outputs/design principles

3.2. **Extract** from spec: inputs/outputs, design principles (Pluggable? Config-driven? Factory?), file list, acceptance criteria.

   ✅ **完成标识：** 列出要修改/创建的所有文件清单

3.3. **Plan** files to create/modify before writing any code.

   ✅ **完成标识：** 书面列出文件清单和每个文件的作用

3.4. **Code** — project-specific rules:
   - Treat spec as single source of truth
   - Use `config/settings.yaml` values, never hardcode
   - Match existing codebase patterns and style

   ✅ **完成标识：** 所有代码文件已创建

3.5. **Write tests** alongside code:
   - Place in `tests/unit/` or `tests/integration/` per spec
   - Mock external deps in unit tests

   ✅ **完成标识：** 测试文件已创建，覆盖主要场景

3.6. **Self-review** before running tests: verify all planned files exist and tests import correctly.

   ✅ **完成标识：** 手动import检查通过，文件清单核对无误

---

### 4. Test & Auto-Fix（测试与自动修复 - 必须运行pytest）

> **⚠️ 常见错误：写完代码不运行测试，或者运行失败不分析原因直接提交**
>
> **后果：** 提交有bug的代码，后续难以定位问题
>
> **正确做法：** 必须看到pytest全部通过的输出才能进入下一步

**前置要求：** 确保虚拟环境已激活
```bash
# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

```
Round 0（初次运行）:
  Run pytest on relevant test file
  If pass → ✅ go to step 5
  If fail → analyze error, apply fix, re-run

Round 1（第一次修复）:
  Run pytest again
  If pass → ✅ go to step 5
  If fail → analyze error, apply fix, re-run

Round 2（第二次修复）:
  Run pytest again
  If pass → ✅ go to step 5
  If fail → analyze error, apply fix, re-run

Round 3 still failing → ❌ STOP, show failure report to user
```

✅ **完成标识：** 看到 `X passed in Ys` 的pytest输出，且没有failed或error

---

### 5. Persist（持久化 - 更新进度，不要跳过！）

> **⚠️ 常见错误：直接git commit，忘记更新DEV_SPEC.md，或者不运行sync_spec.py**
>
> **后果：** 进度跟踪不准确，reference files不同步，下次任务可能基于过期的信息
>
> **正确做法：** 严格按照5.1 → 5.2 → 5.3的顺序执行

5.1. **Update `DEV_SPEC.md`** (global file): change task marker `[ ]` → `[x]`

    ✅ **完成标识：** DEV_SPEC.md中对应任务标记为 `[x]`，填写完成日期

5.2. **Re-sync**: `python .claude/skills/auto-coder/scripts/sync_spec.py --force`

    ✅ **完成标识：** 看到 `synced N chapters` 输出

5.3. **Show detailed summary & generate documentation** (必须执行！不要自己决定commit/skip/next)

> **⚠️ 必须提供详细的任务解读，帮助开发者理解完成的内容**
>
> **目的：** 让开发者能够检视和理解生成的代码，确保符合预期
>
> **必须包含：**
> 1. **文件清单**：创建了哪些文件，每个文件的作用
> 2. **代码解读**：用通俗易懂的语言解释关键代码做了什么
> 3. **设计决策**：为什么这样设计，有什么考虑
> 4. **测试验证**：测试覆盖了哪些场景
> 5. **与spec对应**：如何满足任务要求
>
> **针对Python不熟悉的开发者：**
> - 解释Python特有的概念（如@abstractmethod、dataclass、ABC）
> - 说明代码的作用，而不仅仅是语法
> - 用类比或实例帮助理解

5.3.1. **Generate task summary markdown file** (新增步骤)

> **⚠️ 每次任务完成后，必须生成任务总结文档**
>
> **目的：** 持久化任务完成记录，便于后续查阅和文档生成
>
> **要求：**
> - 文件名格式：`[任务编号]_[任务名称]_实现总结.md`（如：B7.2_Ollama_LLM_实现总结.md）
> - 存放位置：`docs/task-completion-summaries/`
> - 如果目录不存在，自动创建
>
> **文档内容应包含：**
> 1. **任务概述**：任务编号、名称、目标、验收标准
> 2. **创建的文件**：详细列出所有新增/修改的文件及其作用
> 3. **关键设计解读**：2-3个核心设计问题的解释（为什么这样做）
> 4. **测试验证**：测试结果和验收标准验证
> 5. **任务完成度**：满足任务要求和验收标准的清单
> 6. **项目进度**：更新后的进度统计
> 7. **使用示例**：可选，提供使用代码片段
> 8. **技术亮点**：总结本次实现的技术要点
>
> **✅ 完成标识：** md 文件已创建到 `docs/task-completion-summaries/` 目录

5.3.2. **Show summary and ask for user decision**

**Summary Template (for display):**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ✅ [任务编号] [任务名称]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📁 创建了哪些文件？

[列出所有新增/修改的文件]

### 文件1: `path/to/file.py`
**作用：** [一句话说明这个文件的作用]

**关键内容解读：**
- [类/函数1]：[做什么的，有什么参数，返回什么]
- [类/函数2]：[做什么的，有什么参数，返回什么]
- [Python概念解释]：[如果有特殊的Python概念，解释一下]

[对每个重要文件重复上述格式]

## 🔑 关键设计解读

### Q1: [设计问题1]
**解释：** [用通俗易懂的语言解释]

**类比：** [用生活中的例子类比]

**好处：** [这样设计的好处]

### Q2: [设计问题2]
[重复上述格式]

## 📊 测试验证

**[X]个测试全部通过，验证了：**
✅ [测试点1]
✅ [测试点2]
✅ [测试点3]

## 📐 满足任务要求

- ✅ [spec要求1]：[如何实现的]
- ✅ [spec要求2]：[如何实现的]

## 📈 项目进度

- [列出各阶段的进度]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**任务总结文档已生成：** docs/task-completion-summaries/[任务编号]_[任务名称]_实现总结.md

**Commit 信息格式（使用中文）：**

```
feat([模块名]): [任务编号] [中文简短描述]

使用 [提供商/技术] 实现 [功能特性]，支持 [关键能力]。

功能特性：
- [核心功能1]
- [核心功能2]
- [核心功能3]

技术亮点：
- [技术点1]
- [技术点2]

测试覆盖：
- [测试数量] 个测试用例全部通过
- 覆盖 [主要测试场景]

新增文件：
- [文件1]
- [文件2]

修改文件：
- [文件1]
- [文件2]

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

"commit" → git add + commit（使用上述中文提交信息格式）
"skip"   → end
"next"   → commit + start [下一个任务编号]
```

⚠️ **关键：必须等待用户选择，不要自作主张继续执行下一步**

On "next", loop back to step 1 and start the next task.
