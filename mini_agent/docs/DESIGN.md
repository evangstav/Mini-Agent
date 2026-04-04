# Mini-Agent: Architecture, Concepts & Design Decisions

## Overview

Mini-Agent is a production-quality AI coding agent in ~6,000 lines of Python. It implements a think-act-observe loop with 17 tools, permission sandboxing, context management, codebase awareness, and cross-session memory. It scores **44.4% on SWE-bench Verified** (astropy subset, MiniMax M2.7).

This document covers the architecture, the concept-driven design methodology, key decisions, and how Mini-Agent compares to Claude Code, Codex CLI, and OpenCode.

---

## Part 1: Concept-Driven Architecture

Mini-Agent is designed using Daniel Jackson's concept methodology (SPLASH 2025). Each behavioral concern is modeled as an independent concept with its own purpose, state, actions, and sync relationships. The 10 concepts compose into a validated app spec.

### The 10 Concepts

```
┌──────────────────────────────────────────────────────────────┐
│                        MiniAgent App                         │
│                                                              │
│  Configuration ────────────────────────────── (independent)  │
│                                                              │
│  RepoMap ──inject──▶ MessageLog ◀── snip ── ContextBudget   │
│                          ▲                       │           │
│                          │ append_tool_result     │ polls    │
│                          │                       ▼           │
│  Permission ──sync──▶ ToolExecution        AgentLoop         │
│       │                                     ▲    ▲           │
│       └── reads read_only flag              │    │           │
│                                             │    │           │
│  PhaseGating ──sync──▶ AgentLoop    Session─┘    │           │
│       │                                          │           │
│       └──sync──▶ ToolExecution          Memory───┘           │
│                   (tracks edits/tests)   (dreams on end)     │
└──────────────────────────────────────────────────────────────┘
```

| Concept | Module | Purpose |
|---------|--------|---------|
| **AgentLoop** | `agent.py` | Drive a bounded think-act-observe cycle to completion |
| **PhaseGating** | `phase_gating.py` | Prevent premature commitment, ensure verification |
| **ToolExecution** | `tool_execution.py` | Tool call lifecycle: request → permit → execute → result |
| **Permission** | `sandbox.py` + `permissions.py` | Layered allow/deny/ask decisions using tool properties |
| **MessageLog** | `message_log.py` | Ordered append-only history with snipping and compaction |
| **ContextBudget** | `context_budget.py` | Monitor token usage, trigger compaction when needed |
| **RepoMap** | `repo_map.py` | Codebase skeleton for structural awareness |
| **Session** | `agent.py` (lifecycle) | Bracket work with start/end hooks |
| **Memory** | `dream.py` | Cross-session knowledge consolidation |
| **Configuration** | `config.py` | Hierarchical settings with env auto-detection |

### Why Concepts?

Jackson's methodology solves a real problem: as agent harnesses grow, behaviors get tangled. Our code audit found phase nudges embedded in the agent loop, snipping logic in the context module, and permission decisions hardcoded in a frozenset. Extracting each behavior into a concept with a clear boundary made the code:

- **Testable** — each concept can be tested independently (PhaseGating has no dependency on the LLM)
- **Evolvable** — adding the verification gate was a 3-line change to PhaseGating, not a 30-line edit to agent.py
- **Auditable** — the app spec documents every sync relationship; you can verify the system's behavior by reading 60 lines of spec

### The App Composition

```
app MiniAgent
  concepts
    Configuration
    RepoMap [ProjectDir := workspace, CharBudget := 6000]
    MessageLog
    ContextBudget [Window := 200000, ThresholdPct := 0.85, Reserve := 20000]
    ToolExecution [ToolRegistry := RegisteredTools]
    Permission [Mode := auto]
    AgentLoop [MaxSteps := 50]
    PhaseGating [ExploreRatio := 0.3, LastResortRatio := 0.8]
    Session [SessionId := UUID]
    Memory [MemoryDir := ".claude/memory"]
```

Key sync chain:
1. User input → MessageLog.append_user → ContextBudget.check
2. AgentLoop.think → LLM call → MessageLog.append_assistant
3. AgentLoop.act → ToolExecution.request → Permission.evaluate (with read_only flag)
4. ToolExecution.succeed → MessageLog.append_tool_result
5. PhaseGating.tick → check_explore_budget / check_last_resort / check_verification
6. AgentLoop.complete → Session.end → Memory.dream

---

## Part 2: Key Design Decisions

### Decision 1: Phase gating over hard phase enforcement

**Context**: SWE-bench failure analysis showed 42% of failures were empty patches (agent explored forever) and 58% were wrong patches (agent edited too hastily).

**Choice**: Soft nudges at 30% and 80% of step budget, not hard phase transitions. The nudge asks the agent to *reason* ("state the file, explain the root cause, describe the fix") before editing, rather than forcing it to edit.

**Why**: Hard phase enforcement (you MUST edit now) caused panic-editing — the agent rushed into changes without understanding. Soft nudges with a reasoning checkpoint preserved the agent's autonomy while preventing the explore-forever failure mode.

**Result**: SWE-bench resolve rate went from 27% (astropy) to 44.4% after adding phase gating.

### Decision 2: Warn-and-write for linting, not block

**Context**: SWE-Agent's ACI shows that linting on edits prevents cascading failures. But blocking edits entirely prevents multi-step changes (add import in one edit, use it in the next).

**Choice**: EditTool validates syntax but only warns on NEW errors. It always writes the file. The warning appears in the tool result: `"WARNING: Python syntax error at line 12"`.

**Why**: Multi-step edits require intermediate invalid states. Blocking the first edit of a two-edit sequence breaks the workflow. Warning gives the agent the information to fix it in the next step.

### Decision 3: Observation snipping over full masking

**Context**: NeurIPS 2025 research showed observation masking halves cost while matching summarization quality. But full masking (replacing with a placeholder) loses signal.

**Choice**: Snip tool outputs — keep first half + last quarter, remove the verbose middle. Only for outputs >2000 chars older than the last 6 messages.

**Why**: Inspired by nano-claude-code. The beginning of a tool output is the command/context, the end is the conclusion/result. The middle is verbose detail. Keeping both endpoints preserves the key signal while cutting 25-50% of token usage.

### Decision 4: Polling ContextBudget, not reactive

**Context**: The concept originally specified syncing on every MessageLog.append (reactive model). The implementation polls once per step.

**Choice**: Match the polling code. ContextBudget.check() re-estimates tokens from scratch each step.

**Why**: Token estimation is approximate (chars/4). Reactive tracking would accumulate rounding errors. Polling from scratch is simpler, just as accurate, and avoids sync complexity.

### Decision 5: Tool-declared read_only over hardcoded safe sets

**Context**: Permission decisions were based on hardcoded frozensets (`_SAFE_TOOLS = {"read_file", "glob", ...}`). Adding a new read-only tool required editing sandbox.py.

**Choice**: Tools declare `read_only = True` on themselves. Permission.evaluate receives this flag from ToolExecution and uses it, falling back to the hardcoded set when not provided.

**Why**: Self-declaration follows the open-closed principle. MCP tools, custom tools, and new built-in tools automatically get correct permission classification without changing sandbox.py.

### Decision 6: Repo map in system prompt, not on-demand

**Context**: Aider injects a repo map (tree-sitter + PageRank) into every prompt. SWE-Agent relies on the agent searching interactively. Agentless uses hierarchical localization.

**Choice**: Inject an AST-parsed skeleton (file paths + class/function signatures) into the system prompt. 6,000 char budget (~1,500 tokens). Cached to `.runtime/repo_map.md`, invalidated on file changes.

**Why**: The research is clear — agents that start with structural awareness skip 5-8 exploration steps. System prompt injection is the simplest approach (no RAG, no embeddings, no tree-sitter dependency). The 6K budget is Aider's finding: enough for 100-150 signatures without bloating context.

### Decision 7: Unified diff feedback on edits

**Context**: nano-claude-code shows a unified diff after every edit. SWE-Agent shows the file state. Our original EditTool just said "Successfully edited file.py".

**Choice**: EditTool now returns a unified diff (capped at 60 lines) showing exactly what changed. The agent sees its own edit rendered back.

**Why**: Self-correction research shows external feedback (seeing the actual change) is far more effective than introspection (re-reading the file). The diff is free — no extra LLM call, just `difflib.unified_diff`.

---

## Part 3: Comparison with Claude Code, Codex CLI, and OpenCode

### Architecture Comparison

| Feature | Mini-Agent | Claude Code | Codex CLI | OpenCode |
|---------|-----------|-------------|-----------|----------|
| **Language** | Python (~6K lines) | TypeScript | Rust | Go |
| **Agent loop** | ReAct + PhaseGating | ReAct (blended) | ReAct (shell-first) | ReAct |
| **Tools** | 17 built-in + MCP | ~15 built-in + MCP | 1 (shell) + MCP | ~12 built-in |
| **Context window** | 200K (Anthropic) | 200K | 1M (GPT-5.4) | Model-dependent |
| **Compaction** | Snip + LLM summarize | Cache + clear + summarize | Opaque `encrypted_content` | Auto-summarize at 95% |
| **Permissions** | auto/readonly/full_access + TOML rules | 4 modes + LLM safety check | 3 modes + kernel sandbox | Dialog (allow once/session/deny) |
| **Sandbox** | Application-level | Application + LLM safety | **Kernel-level** (Seatbelt/Landlock) | Application-level |
| **Memory** | Dream consolidator + MEMORY.md | Auto-memory + CLAUDE.md | AGENTS.md (no auto-memory) | SQLite database |
| **Multi-agent** | Sub-agents via AgentTool | Agent Teams + worktrees | Sub-agents + worktrees | Basic sub-agents |
| **Repo awareness** | AST repo map in prompt | Prompt caching + deferred tools | Shell exploration | LSP integration |
| **Edit format** | Search/replace + diff view | Search/replace | `apply_patch` (diff) | Patch/Edit/Write |
| **SWE-bench** | 44.4% (MiniMax M2.7) | 76-82% (Claude 4.5/4.6) | 56-77% (GPT-5.x) | Not published |
| **Open source** | Yes (MIT) | No | Yes (Apache 2.0) | Yes (MIT, archived) |

### What Mini-Agent Does Differently

**1. Concept-driven design.** No other agent framework models its architecture as formal concept specs. This isn't just documentation — the concepts are validated, composed into an app spec, and the code is structurally aligned to concept boundaries. When we need to add a new behavior (e.g., concept-aware editing), we design the concept first, validate it, then implement.

**2. Phase gating.** Claude Code and Codex CLI rely entirely on the LLM to manage its own workflow. Mini-Agent's PhaseGating concept actively monitors progress and injects reasoning checkpoints. This is the feature that produced the biggest benchmark improvement (27% → 44%).

**3. Research-backed harness design.** Every major feature cites its research basis:
- Observation snipping (NeurIPS 2025)
- Windowed file output (SWE-Agent, NeurIPS 2024)
- Tool description optimization (Anthropic engineering)
- Best-of-N with majority voting (CodeMonkeys, Stanford 2025)
- Phased workflow prompts (every agent >50% on SWE-bench)

**4. Model-agnostic with provider auto-detection.** Works with Anthropic, OpenAI, and MiniMax APIs. Provider, model, and API base are auto-detected from environment variables.

**5. Benchmarking built in.** `mini-agent bench swebench --slice 0:50` runs SWE-bench directly. No external harness needed. Includes best-of-N mode and HumanEval+ support.

### What Mini-Agent Lacks (vs Competitors)

| Gap | Claude Code Has | Codex CLI Has | OpenCode Has |
|-----|----------------|---------------|--------------|
| **Kernel sandboxing** | No | **Yes** (Seatbelt/Landlock) | No |
| **LLM safety evaluator** | No | **Yes** (permission LLM call) | No |
| **IDE integration** | No | **Yes** (VS Code, JetBrains) | No |
| **Deferred tool loading** | No | **Yes** (ToolSearch) | No |
| **LSP integration** | No | Partial | **Yes** (full LSP client) |
| **Persistent server** | No | No | **Yes** (survives disconnects) |
| **Background agents** | No | **Yes** (cloud sandbox) | No |
| **Encrypted compaction** | No | **Yes** (opaque state) | No |

### Roadmap: Closing the Gaps

| Priority | Feature | Status |
|----------|---------|--------|
| P1 | Session continuity (persist codebase knowledge) | Open (`hq-joom`) |
| P1 | Non-interactive mode | **Shipped** |
| P2 | Semantic codebase search | Open (`hq-cr5h`) |
| P2 | Concept-aware editing | Open (`hq-2fg6`) |
| P2 | Pre/post tool-use hooks | Open (`hq-ao7l`) |
| P2 | Interactive refinement (learn from corrections) | Open (`hq-2jze`) |
| P3 | Tree-sitter + PageRank repo map | Open (`hq-lsnt`) |
| P3 | Agentless mode for CI | Open (`hq-psh3`) |

---

## Part 4: File Map

```
mini_agent/
  agent.py              349 lines   AgentLoop concept — core think-act-observe cycle
  phase_gating.py        97 lines   PhaseGating concept — nudges + verification gate
  tool_execution.py     159 lines   ToolExecution concept — permission + parallel execution
  message_log.py        140 lines   MessageLog concept — history with snipping
  context_budget.py      93 lines   ContextBudget concept — polling compaction monitor
  repo_map.py           195 lines   RepoMap concept — AST skeleton with caching
  sandbox.py            313 lines   Permission concept — modes + rules + tool flags
  permissions.py        163 lines   Permission rules — TOML-based rule engine
  dream.py              340 lines   Memory concept — cross-session consolidation
  config.py             117 lines   Configuration concept — hierarchical settings
  context.py            565 lines   Compaction, pruning, system prompt builder
  hooks.py              106 lines   Session lifecycle pub/sub
  events.py              96 lines   Typed event hierarchy
  retry.py              208 lines   Exponential backoff with jitter
  cost.py                53 lines   Per-model cost estimation
  session_memory.py     205 lines   Periodic session snapshots
  audit.py               78 lines   JSONL transcript logger
  cli/                             Rich CLI with subcommands
  llm/                             Anthropic + OpenAI provider clients
  tools/                           17 built-in tools
  benchmarks/                      SWE-bench + HumanEval+ adapters
  concepts/                        10 concept specs + app composition
```

---

## Part 5: SWE-bench Results

### Batch 1 (v1 harness, 50 instances)
- **19/50 resolved (38.0%)**
- 13 empty patches (explore-forever), 18 wrong patches
- Model: MiniMax M2.7, max_steps: 30

### Batch 2 (v2 harness, 18 astropy instances)
- **8/18 resolved (44.4%)**
- 0 empty patches (phase gating eliminated explore-forever)
- Model: MiniMax M2.7, max_steps: 50

### Key improvements between batches
- Phase gating (30%/80% nudges): eliminated empty patches entirely
- Repo map: faster localization
- Step counter: agent budgets its time
- Phased system prompt: explore → localize → reproduce → fix → verify
- `pip install -e .`: tests actually work now

---

*Mini-Agent v0.2.0 — MIT License — [github.com/evangstav/Mini-Agent](https://github.com/evangstav/Mini-Agent)*
