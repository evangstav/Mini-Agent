"""Agent/tools/config wiring — extracted from tui.py."""

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

from ..agent import Agent
from ..config import MiniAgentConfig, load_config
from ..context import ToolResultStore
from ..dream import DreamConsolidator
from ..hooks import HookRegistry
from ..llm import LLMClient
from ..log import setup_logging
from ..permissions import load_rules
from ..sandbox import PermissionMode, Sandbox
from ..schema import LLMProvider
from ..tools.agent_tool import AgentTool
from ..tools.bash_tool import BashTool, BashKillTool, BashOutputTool
from ..tools.file_tools import EditTool, ReadTool, WriteTool
from ..tools.git_tool import GitBranchTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from ..tools.glob_tool import GlobTool
from ..tools.grep_tool import GrepTool
from ..tools.mcp_loader import load_mcp_tools_async
from ..tools.web_fetch import WebFetchTool
from ..tools.web_search import WebSearchTool

from .commands import REPLContext, SlashCommandRegistry
from .commands.fork import ForkCommand, ForksCommand
from .commands.help import HelpCommand
from .commands.info import CostCommand, HistoryCommand
from .commands.model import ModelCommand
from .commands.plan import ExitCommand, PlanCommand
from .commands.session_cmds import ClearCommand, CompactCommand, LoadCommand, SaveCommand
from .permissions import PermissionManager
from .render import EventRenderer, console

logger = logging.getLogger(__name__)


def _detect_project_root(start: str) -> str:
    """Walk up to find project root (contains pyproject.toml or .git)."""
    current = Path(start).resolve()
    while True:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return str(current)
        parent = current.parent
        if parent == current:
            return start
        current = parent


def build_command_registry() -> SlashCommandRegistry:
    """Build and populate the slash command registry."""
    registry = SlashCommandRegistry()
    for cmd_cls in [
        HelpCommand, PlanCommand, ForkCommand, ForksCommand,
        ClearCommand, CompactCommand, ModelCommand, HistoryCommand,
        CostCommand, SaveCommand, LoadCommand, ExitCommand,
    ]:
        registry.register(cmd_cls())
    return registry


async def build_repl_context(
    api_key: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    api_base: str | None = None,
    workspace: str | None = None,
    system_prompt: str | None = None,
    max_steps: int = 50,
    session_file: str | None = None,
    enable_permissions: bool = True,
    verbose: bool = False,
) -> REPLContext:
    """Resolve config, create LLM client, tools, agent, and return a REPLContext."""
    setup_logging(verbose=verbose)
    workspace = workspace or _detect_project_root(os.getcwd())

    cli_overrides = MiniAgentConfig(
        model=model,
        provider=provider if provider != "anthropic" else None,
        api_base=api_base,
        max_steps=max_steps if max_steps != 50 else None,
        permissions=False if not enable_permissions else None,
    )
    cfg = load_config(project_dir=workspace, cli_overrides=cli_overrides)

    # Auto-detect API key and provider
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = api_key or minimax_key or anthropic_key or ""
    if not api_key:
        console.print("[error]Error:[/] Set MINIMAX_API_KEY or ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    is_minimax = bool(minimax_key) and api_key == minimax_key
    provider = cfg.provider or provider
    provider_enum = LLMProvider(provider)
    model = cfg.model or os.environ.get("MINI_AGENT_MODEL") or (
        "MiniMax-M2.7" if is_minimax else "claude-sonnet-4-20250514"
    )
    api_base = cfg.api_base or os.environ.get("MINI_AGENT_API_BASE") or (
        "https://api.minimax.io" if is_minimax else "https://api.anthropic.com"
    )
    max_steps = cfg.max_steps or max_steps
    enable_permissions = cfg.permissions if cfg.permissions is not None else enable_permissions

    llm_client = LLMClient(api_key=api_key, provider=provider_enum, api_base=api_base, model=model)

    tools = [
        ReadTool(workspace_dir=workspace),
        WriteTool(workspace_dir=workspace),
        EditTool(workspace_dir=workspace),
        BashTool(workspace_dir=workspace),
        BashOutputTool(),
        BashKillTool(),
        GitStatusTool(workspace_dir=workspace),
        GitDiffTool(workspace_dir=workspace),
        GitCommitTool(workspace_dir=workspace),
        GitLogTool(workspace_dir=workspace),
        GitBranchTool(workspace_dir=workspace),
        GlobTool(workspace_dir=workspace),
        GrepTool(workspace_dir=workspace),
        WebSearchTool(),
        WebFetchTool(),
    ]

    if cfg.tools:
        allowed = set(cfg.tools)
        tools = [t for t in tools if t.name in allowed]

    mcp_config = str(Path(workspace) / "mcp.json")
    mcp_tools = await load_mcp_tools_async(mcp_config)
    tools.extend(mcp_tools)

    perm_mgr = PermissionManager(console)
    perm_cb = perm_mgr.check if enable_permissions else None

    default_prompt = system_prompt or (
        "You are a helpful coding assistant. You have access to tools for "
        "reading/writing files, running shell commands, and more. "
        "Work step by step and verify your changes.\n\n"
        "## Grounding\n"
        "Only state facts you have verified by reading files or running commands. "
        "If you cannot find evidence for a claim, say so explicitly. "
        "Never fabricate code snippets, file contents, or test results. "
        "When search results return no matches, report that — do not infer "
        "what the missing content 'probably' contains.\n\n"
        "## Verification\n"
        "After making code changes, ALWAYS verify your work:\n"
        "1. Read the modified file to confirm the edit looks correct.\n"
        "2. Run relevant tests (look for pytest, npm test, go test, etc.).\n"
        "3. If tests fail, fix the issue before moving on.\n"
        "Do not claim changes work without running tests. External verification "
        "(test results, linter output) is far more reliable than re-reading your own edits.\n\n"
        "## Planning\n"
        "For complex multi-step tasks, outline your plan before executing. "
        "State what files you'll change and why. Revise the plan if you discover "
        "unexpected issues during execution."
    )

    tool_result_store = ToolResultStore(
        storage_dir=str(Path(workspace) / ".runtime" / "tool_results")
    )

    hooks = HookRegistry()
    dream = DreamConsolidator(
        memory_dir=str(Path(workspace) / ".claude" / "memory"),
        llm_client=llm_client,
    )
    dream.register(hooks)

    permission_rules = load_rules(project_dir=workspace)
    sandbox = Sandbox(
        mode=PermissionMode.AUTO if enable_permissions else PermissionMode.FULL_ACCESS,
        permission_rules=permission_rules if permission_rules.rules else None,
    )

    all_tools_by_name = {t.name: t for t in tools}
    agent_tool = AgentTool(
        llm_client=llm_client,
        available_tools=all_tools_by_name,
        sandbox=sandbox,
        permission_callback=perm_cb,
        max_steps=max_steps,
        workspace=workspace,
    )
    tools.append(agent_tool)

    session_id = str(uuid.uuid4())
    agent = Agent(
        llm_client=llm_client,
        system_prompt=default_prompt,
        tools=tools,
        max_steps=max_steps,
        tool_result_store=tool_result_store,
        project_dir=workspace,
        permission_callback=perm_cb,
        sandbox=sandbox,
        hooks=hooks,
        session_id=session_id,
        auto_end_session=False,
    )

    if session_file and Path(session_file).exists():
        from .session import load_session
        agent.messages = load_session(session_file)
        console.print(f"[info]Session loaded from[/] {session_file}")

    renderer = EventRenderer(console)

    ctx = REPLContext(
        agent=agent,
        console=console,
        renderer=renderer,
        model=model,
        api_key=api_key,
        provider_enum=provider_enum,
        api_base=api_base,
        forks={0: agent},
        current_fork_id=0,
        next_fork_id=1,
        cancel_event=asyncio.Event(),
        workspace=workspace,
        agent_tool=agent_tool,
        dream=dream,
    )

    return ctx
