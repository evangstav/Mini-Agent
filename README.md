# Mini-Agent

A production-quality AI coding agent in Python. Think-act-observe loop with tool use, permission sandboxing, context management, and SWE-bench benchmarking.

**38% resolve rate on SWE-bench Verified** (50 instances, MiniMax M2.7).

## Install

```bash
# From PyPI (when published)
uv tool install mini-agent

# From GitHub
uv tool install "git+https://github.com/evangstav/Mini-Agent.git"

# For development
git clone https://github.com/evangstav/Mini-Agent.git
cd Mini-Agent
uv sync
```

## Quick Start

```bash
# Set your API key (MiniMax or Anthropic)
export MINIMAX_API_KEY="your-key"
# or: export ANTHROPIC_API_KEY="your-key"

# Interactive REPL
mini-agent

# Non-interactive (for scripting/CI)
mini-agent run "fix the type error in auth.py"
mini-agent run "add input validation to the API" -p  # print-only mode
echo "explain this codebase" | mini-agent run

# Subcommands
mini-agent config show       # show resolved configuration
mini-agent config init       # create .mini-agent.toml template
mini-agent session list      # list saved sessions
mini-agent version           # show version
```

## Features

### Agent Loop
Think → Act → Observe cycle with streaming events, parallel tool execution, and automatic context compaction.

### Tools (17 built-in)
| Tool | Purpose |
|------|---------|
| `read_file` | Read files with 200-line windowed output |
| `write_file` | Write files with syntax warnings |
| `edit_file` | Search/replace with linter validation |
| `undo_edit` | Revert last edit to a file |
| `bash` | Shell execution with safety filters |
| `glob` | Fast file pattern matching |
| `grep` | Content search (ripgrep with fallback) |
| `find_definition` | AST-based Python class/function finder |
| `list_dir` | Directory tree view |
| `git_status/diff/commit/log/branch` | Git operations |
| `web_search` | DuckDuckGo search |
| `web_fetch` | URL fetching with HTML stripping |
| `agent` | Spawn sub-agents for parallel work |

### Context Management
- **Observation masking**: Old tool outputs replaced with placeholders (NeurIPS 2025 research)
- **LLM compaction**: Summarizes old conversation when context fills up
- **Repo map**: AST-based codebase skeleton injected into system prompt
- **Tool result storage**: Large outputs spilled to disk with previews

### Permission Sandboxing
Three modes: `auto` (safe reads allowed, writes gated), `readonly`, `full_access`. Layered with TOML-based rule engine for fine-grained control.

### CLI
Rich terminal output, tab completion for slash commands, status bar, session save/load, conversation forking.

## Configuration

Create `.mini-agent.toml` in your project:

```toml
[agent]
model = "claude-sonnet-4-20250514"
provider = "anthropic"
max_steps = 100

# Permission rules (evaluated in reverse — last match wins)
[[rules]]
tool = "bash"
action = "allow"
args = { command = "git *" }

[[rules]]
tool = "bash"
action = "allow"
args = { command = "pytest *" }
```

Or generate a template: `mini-agent config init`

Config hierarchy: `~/.mini-agent/config.toml` → `.mini-agent.toml` → CLI args (later wins).

## Benchmarking

```bash
# Install bench dependencies
uv sync --group bench

# HumanEval+ (quick sanity check, ~$2)
uv run --group bench mini-agent bench humaneval --slice 0:5

# SWE-bench Verified (real coding tasks)
uv run --group bench mini-agent bench swebench --slice 0:10
uv run --group bench mini-agent bench swebench --slice 0:50 --attempts 3  # best-of-3

# Evaluate results
uv pip install sb-cli
sb-cli gen-api-key your@email.com
sb-cli submit --predictions_path predictions.jsonl \
    --run_id my-run --dataset_name swe-bench_verified --split test
```

## As a Library

```python
import asyncio
from mini_agent import Agent, LLMClient
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.file_tools import ReadTool, EditTool

async def main():
    client = LLMClient(
        api_key="your-key",
        provider="anthropic",
        api_base="https://api.anthropic.com",
        model="claude-sonnet-4-20250514",
    )
    tools = [BashTool(), ReadTool(), EditTool()]
    agent = Agent(client, "You are a helpful coding assistant.", tools)
    agent.add_user_message("Fix the bug in auth.py")
    result = await agent.run()
    print(result)

asyncio.run(main())
```

### Streaming Events

```python
from mini_agent import TextChunk, ToolStart, ToolEnd, AgentDone

async for event in agent.run_stream():
    if isinstance(event, TextChunk):
        print(event.content, end="")
    elif isinstance(event, ToolStart):
        print(f"\n  ▶ {event.tool_name}")
    elif isinstance(event, ToolEnd):
        print(f"  {'✓' if event.success else '✗'}")
    elif isinstance(event, AgentDone):
        print(f"\nDone in {event.steps} steps.")
```

## Architecture

```
mini_agent/
  agent.py              Core agent loop (349 lines)
  tool_execution.py     Tool permission + execution lifecycle
  message_log.py        Append-only message history
  context_budget.py     Context window monitoring + compaction
  context.py            Compaction, observation masking, system prompts
  repo_map.py           AST-based codebase skeleton
  sandbox.py            Permission modes + command classification
  permissions.py        TOML-based rule engine
  dream.py              Cross-session memory consolidation
  cli/                  Rich CLI with subcommands
  llm/                  Anthropic + OpenAI provider clients
  tools/                17 built-in tools
  benchmarks/           SWE-bench + HumanEval+ adapters
  concepts/             8 Jackson-style concept specs
```

See [docs/ARCHITECTURE.md](mini_agent/docs/ARCHITECTURE.md) for detailed diagrams.

## Running Tests

```bash
uv sync --group dev
uv run pytest tests/           # 363+ unit tests
uv run pytest tests/ -m live   # live API tests (needs API key)
```

## License

MIT
