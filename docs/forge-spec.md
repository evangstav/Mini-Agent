# Forge: Multi-Agent Runtime for Mini-Agent

> Spec v0.1 — April 2026

## 1. Problem Statement

Building software with a single agent hits walls: long tasks overflow context,
complex tasks need different skills at different stages, and a single agent
can't parallelize. Gas Town solves this with polecats but requires heavy
infrastructure (Dolt, tmux, Go binary). We need multi-agent coordination
that runs in a single Python process.

## 2. Design Principles

1. **Orchestrator is just an Agent** — uses the same think→act→observe loop,
   same event model, same context management. Its tools happen to be
   multi-agent primitives.

2. **Agents are cheap** — spin up as async coroutines, not processes.
   Share the LLM client and prompt cache prefix.

3. **Communication through artifacts** — agents produce files, not messages.
   Any agent can read any artifact. No routing, no message bus.

4. **Dynamic specialization** — agents aren't predefined roles. The orchestrator
   creates blueprints on-the-fly based on the task.

5. **Re-planning** — the orchestrator revises the plan as agents complete work
   and new information emerges. Not a fixed pipeline.

6. **Minimal infrastructure** — in-memory work tracking, file-based artifacts,
   optional git worktrees. No databases, no servers.

## 3. Core Concepts

### 3.1 Blueprint

A blueprint defines what an agent can do. Created dynamically by the
orchestrator.

```python
@dataclass
class Blueprint:
    name: str                          # e.g. "api-designer"
    purpose: str                       # One sentence
    system_prompt: str                 # Role-specific instructions
    tools: list[Tool]                  # Subset of available tools
    workspace_strategy: WorkspaceStrategy
    constraints: Constraints
```

**WorkspaceStrategy:**
- `SHARED` — read/write main project directory (quick tasks)
- `ISOLATED` — temp directory, merge files back when done
- `WORKTREE` — git worktree, merge via git (best for code)
- `READONLY` — can only read files (analysis, review)

**Constraints:**
```python
@dataclass
class Constraints:
    max_steps: int = 30              # Agent step limit
    max_tokens: int = 100_000        # Context budget
    timeout_seconds: int = 300       # Wall-clock limit
```

### 3.2 WorkItem

Lightweight work tracking. No database — a list in memory.

```python
@dataclass
class WorkItem:
    id: str                           # Auto-generated short ID
    title: str                        # Human-readable summary
    description: str                  # What needs to be done and why
    status: Literal["pending", "running", "done", "failed", "cancelled"]
    assignee: str | None              # Blueprint name of assigned agent
    depends_on: list[str]             # WorkItem IDs that must complete first
    artifacts: list[str]              # Paths to produced artifacts
    result: str | None                # Structured summary when done
    error: str | None                 # Error message if failed
```

**Status transitions:**
```
pending → running → done
                  → failed
pending → cancelled (by re-planning)
```

**Dependency rules:**
- A WorkItem is "ready" when all items in `depends_on` have status `done`
- If a dependency fails, the dependent item is NOT automatically cancelled
  — the orchestrator decides what to do (re-plan, retry, skip)

### 3.3 Artifact

Agents communicate by producing artifacts — files on disk with metadata.

```python
@dataclass
class Artifact:
    path: Path                        # Relative to forge workspace
    producer: str                     # Agent/blueprint name
    work_item_id: str                 # Which work item produced this
    artifact_type: str                # "code", "plan", "review", "analysis", "test"
    summary: str                      # One-paragraph description
    created_at: datetime
```

**Why artifacts over messages:**
- Can be large (full source files, test suites, design docs)
- Any agent can read any artifact — no routing needed
- Naturally persistent on disk
- Orchestrator can inspect them to make decisions
- Version-controlled if using git worktrees

### 3.4 Workspace

Manages isolation between agents.

```python
class WorkspaceManager:
    forge_dir: Path                   # .forge/ in project root

    async def create(name: str, strategy: WorkspaceStrategy) -> Path
    async def merge(source: Path, target: Path) -> MergeResult
    async def cleanup(name: str)
```

**Directory layout:**
```
.forge/
├── work_items.json                  # Persisted work item state
├── artifacts.json                   # Artifact registry
├── workspaces/
│   ├── designer-a1b2/               # Isolated workspace
│   └── coder-c3d4/                  # Git worktree
└── artifacts/
    ├── design.md                    # Produced by designer
    └── review.md                    # Produced by reviewer
```

**MergeResult:**
```python
@dataclass
class MergeResult:
    success: bool
    merged_files: list[str]
    conflicts: list[str]             # Files with merge conflicts
    error: str | None
```

### 3.5 Forge

The runtime. Creates an orchestrator Agent whose tools are multi-agent
primitives.

```python
class Forge:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = DEFAULT_FORGE_PROMPT,
        available_tools: list[Tool] | None = None,
        project_dir: str = ".",
        max_concurrent: int = 3,
    ):
        ...

    async def run(self, goal: str) -> ForgeResult
    async def run_stream(self, goal: str) -> AsyncGenerator[ForgeEvent, None]
```

**ForgeResult:**
```python
@dataclass
class ForgeResult:
    success: bool
    summary: str                     # What was accomplished
    artifacts: list[Artifact]        # All produced artifacts
    work_items: list[WorkItem]       # Final state of all work
    total_steps: int                 # Across all agents
    wall_time: float                 # Total seconds
```

## 4. Orchestrator Tools

The orchestrator is an Agent with these tools. The LLM decides when and
how to use them.

### 4.1 plan_work

Decompose a goal into work items with dependencies.

```
plan_work(items: list[{title, description, depends_on}]) -> list[WorkItem]
```

Creates multiple WorkItems at once. The orchestrator calls this after
analyzing the goal. Can be called again to add more items (re-planning).

### 4.2 create_blueprint

Define a new agent type.

```
create_blueprint(
    name: str,
    purpose: str,
    system_prompt: str,
    tool_names: list[str],          # Subset of available_tools
    workspace_strategy: str,         # "shared" | "isolated" | "worktree" | "readonly"
    max_steps: int = 30,
) -> Blueprint
```

The orchestrator decides what tools each agent type needs. A reviewer
doesn't need write tools. A coder doesn't need web search.

### 4.3 dispatch

Assign a work item to a blueprint and run the agent.

```
dispatch(
    work_item_id: str,
    blueprint_name: str,
    context_artifacts: list[str],    # Artifact paths to inject into agent context
) -> str                             # Returns when agent completes
```

Creates an Agent from the blueprint, injects artifact content into its
context, runs it, and collects produced artifacts. Blocks until the agent
completes (or times out).

### 4.4 dispatch_parallel

Run multiple independent work items concurrently.

```
dispatch_parallel(
    assignments: list[{work_item_id, blueprint_name, context_artifacts}],
) -> list[str]                       # Results for each
```

Uses `asyncio.gather()`. Respects `max_concurrent` limit.

### 4.5 check_status

See current state of all work.

```
check_status() -> {
    pending: list[WorkItem],
    running: list[WorkItem],
    done: list[WorkItem],
    failed: list[WorkItem],
    ready: list[WorkItem],           # Pending with all deps satisfied
}
```

### 4.6 read_artifact

Read the content of an artifact produced by another agent.

```
read_artifact(path: str) -> str
```

### 4.7 merge_work

Integrate artifacts from completed agents into the main workspace.

```
merge_work(work_item_ids: list[str]) -> MergeResult
```

For worktree-based agents, performs git merge. For isolated agents, copies
files. Returns conflict information if any.

### 4.8 revise_plan

Modify the work plan based on new information.

```
revise_plan(
    add: list[{title, description, depends_on}] | None,
    cancel: list[str] | None,        # WorkItem IDs to cancel
    reason: str,
) -> list[WorkItem]                  # Updated full list
```

Called when: a review finds issues, an agent fails, or the orchestrator
realizes the approach needs to change.

## 5. Event Model

Extends the existing `AgentEvent` hierarchy for multi-agent visibility.

```python
class ForgeEvent(BaseModel):
    """Base class for forge-level events."""
    type: str
    timestamp: datetime

class WorkPlanned(ForgeEvent):
    type: Literal["work_planned"] = "work_planned"
    items: list[WorkItem]

class AgentSpawned(ForgeEvent):
    type: Literal["agent_spawned"] = "agent_spawned"
    agent_name: str
    blueprint: str
    work_item_id: str

class AgentProgress(ForgeEvent):
    type: Literal["agent_progress"] = "agent_progress"
    agent_name: str
    step: int
    max_steps: int
    last_action: str                 # e.g. "called bash", "wrote file"

class AgentCompleted(ForgeEvent):
    type: Literal["agent_completed"] = "agent_completed"
    agent_name: str
    work_item_id: str
    artifacts: list[str]
    steps: int
    duration: float

class AgentFailed(ForgeEvent):
    type: Literal["agent_failed"] = "agent_failed"
    agent_name: str
    work_item_id: str
    error: str

class WorkUnblocked(ForgeEvent):
    type: Literal["work_unblocked"] = "work_unblocked"
    work_item_id: str
    unblocked_by: str

class PlanRevised(ForgeEvent):
    type: Literal["plan_revised"] = "plan_revised"
    added: list[str]
    cancelled: list[str]
    reason: str

class MergeCompleted(ForgeEvent):
    type: Literal["merge_completed"] = "merge_completed"
    merged_items: list[str]
    conflicts: list[str]

class ForgeDone(ForgeEvent):
    type: Literal["forge_done"] = "forge_done"
    result: ForgeResult
```

## 6. Orchestrator System Prompt

The default orchestrator prompt:

```
You are a software engineering team lead. You decompose complex tasks
into focused work items, create specialized agent blueprints, and
coordinate their execution.

Workflow:
1. Analyze the goal and plan work items with dependencies
2. Create blueprints with the minimum tools each agent needs
3. Dispatch work, starting with items that have no dependencies
4. Use dispatch_parallel for independent items
5. Read artifacts from completed agents to inform next steps
6. Revise the plan if reviews find issues or agents fail
7. Merge all work when complete

Principles:
- Each work item should be completable in < 30 steps
- Give agents the MINIMUM tools they need (least privilege)
- Prefer readonly for analysis, worktree for code changes
- Always dispatch a reviewer after code is written
- Re-plan rather than letting a bad approach continue
- Merge incrementally, don't wait for everything to finish
```

## 7. Prompt Cache Sharing

All agents forked from the Forge share the same prompt cache prefix:
- Same `llm_client` instance
- System prompt of each agent starts with the same project context
  (CLAUDE.md, git info, memory files)
- Agent-specific instructions appended after the shared prefix

This means the first agent pays the full prompt cost, and subsequent
agents get cache hits on the shared prefix (~90% cost reduction).

## 8. TUI Integration

The TUI gets a `--forge` mode:

```bash
mini-agent --forge "Build a REST API with tests"
```

Display:
```
╭─ Forge ──────────────────────────────────────────────╮
│ Goal: Build a REST API with tests                     │
│                                                       │
│ designer   ████████████ done (2 steps, 3.1s)          │
│   └─ design.md                                        │
│ coder      ████████░░░░ step 8/30 (wrote routes.py)   │
│   └─ worktree: .forge/workspaces/coder-a1b2/          │
│ tester     ░░░░░░░░░░░░ pending (blocked by: coder)   │
│ reviewer   ░░░░░░░░░░░░ pending (blocked by: tester)  │
│                                                       │
│ 4 items │ 1 done │ 1 running │ 2 pending │ 0 failed   │
╰───────────────────────────────────────────────────────╯
```

Forge events stream into the display. User can Ctrl+C to cancel.

## 9. Error Handling

**Agent failure:**
- WorkItem status → "failed", error recorded
- Orchestrator receives the failure as a tool result
- Orchestrator decides: retry, revise plan, or abort

**Merge conflict:**
- MergeResult includes conflict file list
- Orchestrator can dispatch a "conflict resolver" agent
- Or merge manually and continue

**Timeout:**
- Agent cancelled after `constraints.timeout_seconds`
- WorkItem status → "failed" with timeout error
- Orchestrator decides next step

**Orchestrator failure:**
- If the orchestrator itself hits max_steps → ForgeResult.success=False
- All running agents cancelled
- Partial artifacts preserved in .forge/

## 10. Persistence

**During execution:**
- `work_items.json` updated after every status change
- `artifacts.json` updated when artifacts are registered
- Workspaces persist on disk

**After completion:**
- `.forge/` directory preserved with full state
- Can be inspected for debugging
- Future: resume interrupted forges from persisted state

## 11. Implementation Plan

| Phase | Component | LOC | Depends On |
|-------|-----------|-----|------------|
| 1 | `work_item.py` — WorkItem + status tracking | ~80 | — |
| 1 | `artifact.py` — Artifact registry | ~60 | — |
| 1 | `blueprint.py` — Blueprint + Constraints | ~50 | — |
| 2 | `workspace.py` — WorkspaceManager | ~100 | — |
| 2 | `forge_events.py` — Multi-agent events | ~40 | — |
| 3 | `forge_tools.py` — Orchestrator tool set | ~150 | Phases 1-2 |
| 3 | `forge.py` — Forge runtime | ~200 | All above |
| 4 | TUI `--forge` mode | ~100 | forge.py |
| **Total** | | **~680** | |

Phases 1-2 have no dependencies and can be built in parallel.
Phase 3 integrates everything. Phase 4 is TUI.

## 12. Open Questions

1. **Model selection per blueprint** — should the orchestrator be able to
   assign cheaper models (Haiku) for simple tasks and expensive models
   (Opus) for complex ones?

2. **Agent-to-agent communication** — artifacts cover most cases, but should
   agents be able to "ask" another running agent a question mid-execution?

3. **Nested forges** — should a work item be able to spawn its own sub-forge?
   This enables recursive decomposition but adds complexity.

4. **Persistence / resume** — how important is it to resume an interrupted
   forge? Requires checkpointing the orchestrator's conversation state.

5. **Cost tracking** — should the forge track per-agent and total API cost?
   Useful for budget-constrained deployments.

6. **Human-in-the-loop** — should certain work items require human approval
   before the agent starts? (e.g., "deploy to production")
