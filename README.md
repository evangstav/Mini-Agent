# Mini Agent

A minimal, readable agent harness in Python. ~2,000 lines of code you can read in an afternoon.

Built on the [MiniMax M2.5](https://www.minimax.io/) model with Anthropic-compatible API, but works with any Anthropic or OpenAI-compatible provider.

## What It Does

- **Agent loop**: Think → Act → Observe cycle with tool use
- **Streaming events**: `run_stream()` yields typed events (`TextChunk`, `ToolStart`, `ToolEnd`, `AgentCancelled`, etc.)
- **Tools**: Bash, Read/Write/Edit files, MCP integration
- **Context management**: Large tool results stored to disk, automatic compaction with COMPACT hook
- **System prompt injection**: Auto-injects CLAUDE.md, git info, and persistent memory
- **Two LLM providers**: Anthropic and OpenAI protocols
- **Permission gating**: Typed async callback to approve/deny tool calls
- **Lifecycle hooks**: SESSION_START, SESSION_END, COMPACT events with async callbacks
- **Retry with jitter**: Exponential backoff with full jitter to prevent thundering herd
- **Session persistence**: Save/load conversations with corrupt-file error handling
- **Timestamps**: All events and messages include UTC timestamps

## Quick Start

```bash
# Clone and install
git clone https://github.com/evangstav/Mini-Agent.git
cd Mini-Agent
uv sync  # or: pip install -e .

# Set your API key
export MINIMAX_API_KEY="your-key-here"
```

### Simple Usage

```python
import asyncio
from mini_agent import Agent, LLMClient
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.file_tools import ReadTool, WriteTool, EditTool

async def main():
    client = LLMClient(
        api_key="your-key",
        provider="anthropic",
        api_base="https://api.minimax.io",
        model="MiniMax-M2.5",
    )
    tools = [BashTool(), ReadTool(), WriteTool(), EditTool()]
    agent = Agent(client, "You are a helpful coding assistant.", tools)
    agent.add_user_message("List the files in this directory")
    result = await agent.run()
    print(result)

asyncio.run(main())
```

### Streaming Events

```python
from mini_agent import TextChunk, ToolStart, ToolEnd, AgentDone, AgentCancelled

async for event in agent.run_stream():
    if isinstance(event, TextChunk):
        print(event.content, end="")
    elif isinstance(event, ToolStart):
        print(f"\n  -> {event.tool_name}")
    elif isinstance(event, ToolEnd):
        print(f"  <- {'ok' if event.success else event.error}")
    elif isinstance(event, AgentCancelled):
        print(f"\nCancelled at step {event.steps}.")
    elif isinstance(event, AgentDone):
        print(f"\nDone in {event.steps} steps.")
```

### With Context Management

```python
from mini_agent.context import ToolResultStore, SystemPromptBuilder
from mini_agent.hooks import HookRegistry

agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful assistant.",
    tools=tools,
    tool_result_store=ToolResultStore(),     # Persist large results to disk
    context_window=200_000,                  # Total context window (tokens)
    compact_threshold_pct=0.85,              # Compact at 85% of context window
    compaction_reserve=20_000,               # Tokens reserved for summary output
    project_dir=".",                         # Inject CLAUDE.md + git info
    permission_callback=my_permission_fn,    # async (tool_name, args) -> bool
    hooks=HookRegistry(),                    # SESSION_START/END/COMPACT hooks
    session_id="my-session",                 # Optional session identifier
)
```

### Using Other Providers

```python
# Anthropic Claude
client = LLMClient(
    api_key="sk-ant-...",
    provider="anthropic",
    api_base="https://api.anthropic.com",
    model="claude-sonnet-4-6",
)

# OpenAI
client = LLMClient(
    api_key="sk-...",
    provider="openai",
    api_base="https://api.openai.com/v1",
    model="gpt-4o",
)
```

## Architecture

```
mini_agent/
├── agent.py          Core loop: run() + run_stream()
├── context.py        Tool result storage, compaction, system prompt
├── events.py         Typed event hierarchy with Literal discriminators
├── hooks.py          Lifecycle hooks (SESSION_START/END, COMPACT)
├── schema/           Message (with timestamps), ToolCall, LLMResponse
├── retry.py          Async retry with exponential backoff + jitter
├── dream.py          Dream consolidator for persistent memory
├── tui.py            Interactive REPL with slash commands
├── llm/
│   ├── base.py       Abstract LLM client
│   ├── llm_wrapper.py Provider router
│   ├── anthropic_client.py
│   └── openai_client.py
└── tools/
    ├── base.py        Tool + ToolResult
    ├── bash_tool.py   Shell execution
    ├── file_tools.py  Read/Write/Edit
    └── mcp_loader.py  MCP integration (stdio env merged with parent)
```

## MCP Integration

Load tools from any MCP server:

```python
from mini_agent.tools.mcp_loader import load_mcp_tools

tools = await load_mcp_tools("path/to/mcp-config.json")
agent = Agent(client, "You are helpful.", tools + [BashTool()])
```

## Running Tests

```bash
uv run pytest tests/
```

## License

MIT
