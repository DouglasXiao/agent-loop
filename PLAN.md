# 改进路线图（基于 learn-claude-code/docs/zh）

> 来源参考：`shareAI-lab/learn-claude-code` 的 12 章 harness 课程（s01–s12）。
> 本仓库目标：把这些 harness 机制以**最小侵入**方式叠加到现有 OpenAI 兼容
> 流式 agent loop 之上，每个阶段独立 commit。

## 1. 现状快照（已具备的能力）

| Harness 层 | 现状 | 文件 |
| --- | --- | --- |
| s01 Agent loop | 已具备：`while True` + streaming 聚合 + `tool_calls` 解锁 | `agent_loop.py` |
| s02 Tool use + dispatch | 已具备：`execute_tool` 中央派发、ToolPolicy 沙箱 | `tools_execution.py`, `tools_registry.py` |
| s04 Subagent | 已具备：`run_sub_agent` / 并行版本，独立 OpenAI client | `sub_agent.py` |
| s06 Context compact | 部分：仅有"超过阈值整体摘要"，缺少 micro-compact 和 transcripts 落盘 | `context_memory.py` |
| s03 Todo / s05 Skill / s07 Task graph / s08 BG / s09–s12 Teams & Worktree | **未实现** | — |

主要短板（按价值排序）：

1. **没有规划工具**：模型在多步任务中容易跑偏；s03 的 `todo` + nag-reminder 是最便宜的修复。
2. **工具串行执行**：单轮多个 `tool_calls` 仍然顺序执行，浪费墙钟时间，并发化是低风险高回报。
3. **上下文压缩太粗**：只有"全文摘要"一种刀，缺少 s06 的 micro-compact（旧 tool_result → 占位符）以及可恢复的 transcripts 归档。

## 2. 本次落地范围（Phase 1–3）

### Phase 1：`todo_write` 工具 + 持久化任务列表（对齐 s03，吸收 s07 的"持久化"思想）

- 新增模块 `todo_manager.py`：状态机 `pending / in_progress / completed`，约束**同一时刻最多 1 个 in_progress**。
- 持久化到 `.claude/todos/current.json`，跨 context-compact / 跨 session 存活；同时维护内存视图。
- 工具 `todo_write`（actions：`set` / `add` / `update` / `complete` / `clear`），返回当前清单的 markdown 视图。
- 在 `system prompt` 中加入"什么时候用 todo"的 guideline；在 `core_agent_loop_streaming` 中实现 **nag reminder**：连续 N 轮（默认 3 轮）没有调用 todo 时，向最新 tool_result 注入 `<reminder>`。
- ToolPolicy：归类为 `mutate`（写本地状态）。

### Phase 2：tool 执行健壮性 + 同轮并行

- 在 `core_agent_loop_streaming` 中并行执行**只读**类（`read_file / glob_files / grep_files / web_fetch / get_weather`）的 `tool_calls`；写类工具仍串行执行，避免文件竞争。
- 抽出 `_run_one_tool_call` 函数：负责参数解析、SSE 事件、错误捕获，统一返回结构（含 `duration_ms` / `error`）。
- 每个工具调用增加耗时与错误事件 `tool_call_done`；保留 `tool_call_id ↔ tool` 配对的严格顺序。
- 用本地静态测试覆盖：参数缺失、JSON 解码失败、并行/串行混合 batch。

### Phase 3：context compact 升级（micro-compact + transcripts）

- `context_memory.py` 新增 `micro_compact_inplace(messages)`：在每轮模型调用前，对距离当前轮 > `KEEP_RECENT`（默认 3）的 `role=tool` 消息内容进行折叠：
  - `[micro-compacted: previously called {tool_name}; full output kept on disk at {spill_path or "—"}]`
  - 不删除消息（保留 tool_call_id 配对），只缩短 `content`。
- 触发"全量摘要"（既有 `maybe_compress_conversation`）前先 `micro-compact`；并把**整段被替换的早期对话**写入 `.claude/memory/transcripts/transcript_<ts>.jsonl`，可用 `read_file` 找回。
- 新增 SSE 事件 `micro_compact`，包含 `replaced_count` / `freed_chars`。
- 兼容现有 spill 机制；与 Phase 1 的 todo 持久化互补（todo 不进 micro-compact，因为它本身在工具结果里就只是几行）。

## 3. 后续阶段进度

- **Phase 4 ✅**：subagent 协议结构化（label、error_category、rounds_used、duration_ms、tools_used、tool_errors；config_error 提前快路径；wall-clock 超时）。
- **Phase 5 ✅**：`.claude/skills/<name>/SKILL.md` 按需加载（list_skills / load_skill；YAML frontmatter；system prompt 内联索引；附带 git-commit / code-review 两个 seed skill）。
- **Phase 6 ✅**：持久化 `task_graph`（s07，`.claude/tasks/`，blockedBy DAG）+ background tasks（s08，`bg_run` / `bg_check`，下一轮自动 drain 为 `<background-results>`）。
- **Phase 7 ✅**：worktree 隔离（s12，`.worktrees/index.json` + `events.jsonl`，与 `task` 绑定，`complete_task=true` 原子完成）+ team mailbox 雏形（s09 子集，`.team/inbox/<name>.jsonl` 文件型；不含 s10 协议握手 / s11 autonomous loop）。

## 4. 仍未覆盖（明确留白）

- **s10 协议握手**：shutdown / plan-approval 的 request_id ↔ response_id FSM；当前 mailbox 已经能承载消息载体，但没有强制状态机。
- **s11 autonomous claim loop**：teammate 自动扫看板认领任务；需要长期运行的子线程或独立进程，跨 session 设计点更多。
- **MCP 化**：把 sub_agent / task / worktree 抽成独立 MCP server 的可能性。

## 4. 提交策略

- 每个 phase 一个 commit，conventional 风格：`feat:` / `refactor:` / `chore:`。
- 完成即 `git push origin main`，方便在 GitHub 上回看。
- 任何破坏性改动（修改既有工具签名、改 `system prompt` 结构）都先在 `CLAUDE.md` 同步说明。
