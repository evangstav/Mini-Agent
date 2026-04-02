# Mini-Agent Architecture

## Overview

Mini-Agent is a ~5,600-line Python implementation of an LLM-powered coding agent. It provides a Claude Code-like REPL with streaming output, tool execution, permission sandboxing, context management, and cross-session memory.

```
mini_agent/
  agent.py            349 lines   Core agent loop (think-act-observe)
  tool_execution.py   159 lines   Tool permission + execution lifecycle
  message_log.py      102 lines   Append-only message history
  context_budget.py    78 lines   Context window monitoring + compaction
  context.py          565 lines   Compaction, pruning, system prompts
  sandbox.py          313 lines   Permission modes + command classification
  permissions.py      163 lines   TOML-based rule engine
  dream.py            340 lines   Cross-session memory consolidation
  tui.py              760 lines   Interactive REPL
  llm/                           Anthropic + OpenAI provider clients
  tools/                         Bash, file, git, glob, grep, web, MCP, sub-agent
```

---

## Module Dependency Graph

```mermaid
flowchart TB
    subgraph TUI["TUI Layer"]
        tui["tui.py"]
    end

    subgraph Core["Agent Core"]
        agent["agent.py"]
        executor["tool_execution.py"]
        msglog["message_log.py"]
        budget["context_budget.py"]
    end

    subgraph LLM["LLM Providers"]
        wrapper["LLMClient"]
        anthropic["AnthropicClient"]
        openai["OpenAIClient"]
    end

    subgraph Security["Permission Layer"]
        sandbox["Sandbox"]
        perms["PermissionRuleset"]
    end

    subgraph Tools["Tool Registry"]
        bash["BashTool"]
        files["Read/Write/Edit"]
        git["Git tools"]
        search["Glob/Grep"]
        web["WebFetch/Search"]
        mcp["MCP tools"]
        subagent["AgentTool"]
    end

    subgraph Infra["Infrastructure"]
        hooks["HookRegistry"]
        dream["DreamConsolidator"]
        audit["AuditLogger"]
        smem["SessionMemory"]
        context["compact_messages"]
        config["load_config"]
        retry["RetryConfig"]
    end

    tui --> agent
    agent --> executor
    agent --> msglog
    agent --> budget
    agent --> wrapper
    agent --> hooks

    executor --> sandbox
    sandbox --> perms
    executor -.-> Tools

    budget --> context
    wrapper --> anthropic
    wrapper --> openai
    anthropic --> retry
    openai --> retry

    hooks --> dream
    hooks --> audit
    agent --> smem

    tui --> config

    style Core fill:#e1f5fe,stroke:#0288d1
    style Security fill:#fff3e0,stroke:#f57c00
    style LLM fill:#f3e5f5,stroke:#7b1fa2
    style Tools fill:#e8f5e9,stroke:#388e3c
    style Infra fill:#fce4ec,stroke:#c62828
    style TUI fill:#f5f5f5,stroke:#616161
```

---

## Agent Loop (Think-Act-Observe)

The core loop in `agent.py:run_stream()` follows this cycle:

```mermaid
sequenceDiagram
    actor User
    participant TUI
    participant Agent
    participant Budget as ContextBudget
    participant LLM
    participant Exec as ToolExecutor
    participant Sandbox
    participant Tool

    User->>TUI: input
    TUI->>Agent: add_user_message + run_stream

    loop step 0..max_steps
        Agent->>Budget: maybe_compact
        
        rect rgb(230, 245, 255)
        Note right of Agent: THINK
        Agent->>LLM: generate_stream
        LLM-->>TUI: TextChunk, ThinkingChunk
        LLM-->>Agent: message_complete
        end

        Agent->>Agent: log.append_assistant

        alt no tool_calls
            Agent-->>TUI: AgentDone
        else tool_calls present
            rect rgb(255, 243, 224)
            Note right of Agent: ACT
            Agent->>Exec: execute_batch
            Exec->>Sandbox: check each call
            Sandbox-->>Exec: ALLOW / ASK / DENY
            Exec->>Tool: execute (parallel)
            Tool-->>Exec: ToolResult
            Exec-->>TUI: ToolStart, ToolEnd
            end

            rect rgb(232, 245, 233)
            Note right of Agent: OBSERVE
            Agent->>Agent: log.extend results
            end
        end
    end
```

---

## Tool Execution Lifecycle

`ToolExecutor.execute_batch()` manages each tool call through a clean state progression:

```mermaid
stateDiagram-v2
    [*] --> requested: request(tool, args)
    
    requested --> permitted: Sandbox ALLOW
    requested --> gated: Sandbox ASK
    requested --> denied: Sandbox DENY
    
    gated --> permitted: User grants
    gated --> denied: User refuses
    gated --> denied: No callback
    
    permitted --> executing: begin execution
    
    executing --> succeeded: tool returns Ok
    executing --> failed: tool returns Error
    executing --> failed: exception thrown
    
    succeeded --> [*]
    failed --> [*]
    denied --> [*]
```

---

## Permission Decision Flow

Three layers combine to produce a decision:

```mermaid
flowchart TD
    call["Tool Call: name + args"] --> rules{"Permission Rules\n(TOML config)"}
    
    rules -->|"rule matches"| action["Rule Action:\nALLOW / DENY / ASK"]
    rules -->|"no match"| mode{"Permission Mode"}
    
    mode -->|full_access| allow["ALLOW"]
    mode -->|readonly| ro{"Read-only tool?"}
    mode -->|auto| auto{"Safe tool?"}
    
    ro -->|yes| allow
    ro -->|no| deny["DENY"]
    
    auto -->|"safe (read-only)"| allow
    auto -->|"write tool"| ask["ASK user"]
    auto -->|"bash: safe cmd"| allow
    auto -->|"bash: other"| ask
    auto -->|"web: allowed domain"| allow
    auto -->|"web: unknown domain"| ask
    auto -->|"unknown tool"| ask

    style allow fill:#c8e6c9,stroke:#2e7d32
    style deny fill:#ffcdd2,stroke:#c62828
    style ask fill:#fff9c4,stroke:#f9a825
```

**Rule evaluation order**: Rules are walked in *reverse* — last-added rule wins. Project rules (appended after global rules) naturally override global defaults.

---

## Context Management

```mermaid
flowchart LR
    subgraph MessageLog
        sys["System prompt"]
        msgs["User/Assistant/Tool messages"]
    end

    subgraph ContextBudget
        est["estimate_tokens()"]
        thresh["threshold = window * 0.85 - reserve"]
        check{"usage >= threshold?"}
    end

    subgraph Compaction
        summarize["LLM summarizes old turns"]
        replace["log.replace_prefix()"]
    end

    subgraph Pruning
        scan["Scan backward from recent"]
        prune["Replace large tool results\nwith placeholders"]
    end

    msgs --> est --> check
    check -->|yes| summarize --> replace
    check -->|no| continue["Continue loop"]
    msgs --> scan --> prune

    style Compaction fill:#e1f5fe,stroke:#0288d1
    style Pruning fill:#fff3e0,stroke:#f57c00
```

**Token estimation**: ~4 chars/token (conservative for code-heavy context). Compaction triggers at 85% of context window minus a 20K token reserve.

---

## Session Lifecycle & Memory

```mermaid
flowchart TB
    subgraph Session
        create["Session.create(id)"] --> start["Session.start()"]
        start --> active["active"]
        active -->|"AgentDone"| end_done["Session.end('done')"]
        active -->|"AgentCancelled"| end_cancel["Session.end('cancelled')"]
        active -->|"AgentError"| end_error["Session.end('errored')"]
    end

    subgraph Hooks["Lifecycle Hooks"]
        end_done --> hook_end["SESSION_END hook"]
        end_cancel --> hook_end
        end_error --> hook_end
    end

    subgraph Dream["Memory Consolidation"]
        hook_end --> orient["1. Orient: read existing memories"]
        orient --> gather["2. Gather: build transcript"]
        gather --> consolidate["3. Consolidate: LLM decides updates"]
        consolidate --> prune_mem["4. Apply ops + rebuild index"]
    end

    subgraph MemoryFiles["~/.claude/memory/"]
        user_mem["user_*.md"]
        feedback_mem["feedback_*.md"]
        project_mem["project_*.md"]
        ref_mem["reference_*.md"]
        index["MEMORY.md (index)"]
    end

    prune_mem --> MemoryFiles

    style Session fill:#e1bee7,stroke:#6a1b9a
    style Dream fill:#e1f5fe,stroke:#0288d1
    style MemoryFiles fill:#f5f5f5,stroke:#616161
```

---

## Concept Map

The codebase is modeled by 8 Jackson-style concepts (specs in `concepts/*.concept`):

```mermaid
flowchart LR
    AL["AgentLoop\nstate machine\nthink/act/observe"]
    TE["ToolExecution\nlifecycle\nrequest..succeed/fail"]
    PM["Permission\ndecision engine\nevaluate/grant/refuse"]
    ML["MessageLog\nappend-only\nfork/compact"]
    CB["ContextBudget\nmonitor\ntrack/compact"]
    SS["Session\nlifecycle\ncreate/start/end"]
    MM["Memory\npersistent store\ndream/create/update"]
    CF["Configuration\nlayer merge\nglobal/project/CLI"]

    SS -->|"sync: complete\ncancel, exhaust"| AL
    PM -->|"sync: request"| TE
    CB -->|"sync: append"| ML
    MM -->|"sync: end"| SS

    style AL fill:#bbdefb,stroke:#1565c0
    style TE fill:#bbdefb,stroke:#1565c0
    style PM fill:#ffe0b2,stroke:#e65100
    style ML fill:#c8e6c9,stroke:#2e7d32
    style CB fill:#c8e6c9,stroke:#2e7d32
    style SS fill:#e1bee7,stroke:#6a1b9a
    style MM fill:#e1bee7,stroke:#6a1b9a
    style CF fill:#f5f5f5,stroke:#616161
```

| Concept | File(s) | Purpose |
|---------|---------|---------|
| AgentLoop | `agent.py` | Bounded think-act-observe cycle |
| ToolExecution | `tool_execution.py` | Tool call permission + execution lifecycle |
| Permission | `sandbox.py` + `permissions.py` | Layered allow/deny/ask decisions |
| MessageLog | `message_log.py` | Ordered append-only message history |
| ContextBudget | `context_budget.py` | Monitor token usage, trigger compaction |
| Session | `agent.py` (lifecycle methods) | Bracket work with start/end hooks |
| Memory | `dream.py` | Cross-session knowledge consolidation |
| Configuration | `config.py` | Hierarchical config with layer overrides |

---

## LLM Provider Architecture

```mermaid
classDiagram
    class LLMClient {
        +generate(messages, tools) LLMResponse
        +generate_stream(messages, tools) StreamDelta
    }
    class LLMClientBase {
        <<abstract>>
        +generate()*
        +generate_stream()*
        #_prepare_request()*
        #_convert_messages()*
    }
    class AnthropicClient {
        +generate()
        +generate_stream()
        -_parse_response()
        -thinking_signature support
        -stream retry on 429/5xx
    }
    class OpenAIClient {
        +generate()
        +generate_stream()
        -_parse_response()
        -reasoning_split (MiniMax)
        -stream retry on 429/5xx
    }

    LLMClient --> LLMClientBase : delegates
    LLMClientBase <|-- AnthropicClient
    LLMClientBase <|-- OpenAIClient
```

Both clients support:
- **Streaming**: Real-time token delivery via async generators
- **Tool calling**: Automatic format conversion (Anthropic/OpenAI schemas)
- **Extended thinking**: Anthropic thinking blocks with signature replay
- **Retry**: Transient error retry (429, 5xx) on both `generate()` and `generate_stream()`
- **Token tracking**: Usage accumulation from API responses

---

## Configuration Hierarchy

```mermaid
flowchart TB
    global["~/.mini-agent/config.toml\n(user global)"] -->|"layer 1"| merge
    project[".mini-agent.toml\n(project local)"] -->|"layer 2"| merge
    cli["CLI args / env vars"] -->|"layer 3"| merge
    merge["MiniAgentConfig.merge()"] --> resolved["Resolved Config"]
    
    resolved --> model["model"]
    resolved --> provider["provider"]
    resolved --> api_base["api_base"]
    resolved --> max_steps["max_steps"]
    resolved --> permissions["permissions"]
    resolved --> tools_list["tools (allowlist)"]
```

Later layers override earlier ones. Non-None scalars replace; non-empty tool lists replace entirely.

---

## Key Design Decisions

### Why extract ToolExecutor?
The permission-check + parallel-execute + result-collect logic was duplicated between `run_stream()` (170 lines) and `_execute_tool_calls()` (73 lines). Extracting it eliminated the duplication and made tool execution independently testable.

### Why extract MessageLog?
Agent directly manipulated `self.messages` as a raw list from 9 different places. MessageLog provides a single mutation point, enabling future features like automatic budget tracking on append, and making `fork()` testable in isolation.

### Why extract ContextBudget?
Compaction logic (estimate tokens, check threshold, summarize, fire hook) was inlined in the main agent loop. Extracting it made `run_stream()` focus purely on think-act-observe.

### Why defer AgentLoop state machine?
The agent loop's "state" is implicit in the program counter of `run_stream()`. Adding explicit status tracking would add ceremony without preventing bugs — the `self._running` guard already prevents concurrent execution.

### Why defer Session extraction?
Session lifecycle is only ~40 lines. Extracting it into a separate class would add overhead disproportionate to its size.

---

## File Size Summary

| File | Lines | Role |
|------|------:|------|
| tui.py | 760 | Interactive REPL |
| context.py | 565 | Compaction, pruning, system prompts |
| agent.py | 349 | Core agent loop |
| dream.py | 340 | Memory consolidation |
| sandbox.py | 313 | Permission modes |
| tools/bash_tool.py | 310 | Shell execution |
| tools/mcp_loader.py | 234 | MCP server connections |
| retry.py | 208 | Exponential backoff |
| session_memory.py | 205 | Periodic session snapshots |
| tools/git_tool.py | 435 | Git operations |
| tools/file_tools.py | 170 | Read/Write/Edit |
| permissions.py | 163 | TOML rule engine |
| tool_execution.py | 159 | Tool lifecycle |
| tools/grep_tool.py | 139 | Content search |
| tools/agent_tool.py | 160 | Sub-agent spawning |
| config.py | 117 | Hierarchical config |
| hooks.py | 106 | Lifecycle pub/sub |
| message_log.py | 102 | Message history |
| events.py | 96 | Event types |
| tools/web_fetch.py | 88 | URL fetching |
| context_budget.py | 78 | Context monitoring |
| tools/web_search.py | 122 | DuckDuckGo search |
| tools/glob_tool.py | 72 | File pattern matching |
| audit.py | 78 | JSONL transcript |
| cost.py | 53 | Cost estimation |
| log.py | 50 | Logging setup |
| schema/schema.py | 75 | Core data models |
