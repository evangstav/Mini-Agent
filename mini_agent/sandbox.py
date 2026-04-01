"""Sandbox — permission modes and safe command classification.

Three permission modes matching Codex CLI:
- auto:        Safe reads execute freely; writes are gated for approval.
- readonly:    Browse only — no writes, no shell mutations.
- full_access: Unrestricted — all tools execute without prompting.

Replaces the old blocklist approach with a whitelist for auto mode.
Includes a network domain allowlist for web tools.
"""

from __future__ import annotations

import logging
import re
import shlex
from enum import Enum
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PermissionMode(str, Enum):
    """Agent permission mode."""

    AUTO = "auto"
    READONLY = "readonly"
    FULL_ACCESS = "full_access"


class Decision(str, Enum):
    """Sandbox decision for a tool call."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


# ── Safe command classification (whitelist for auto mode) ────────────────────

# Shell commands considered safe (read-only, no side effects)
_SAFE_COMMANDS: frozenset[str] = frozenset({
    # Filesystem reads
    "cat", "head", "tail", "less", "more", "wc", "file", "stat",
    "ls", "find", "tree", "du", "df",
    # Text processing (read-only)
    "grep", "rg", "ag", "awk", "sed",  # sed is read-only when used in pipeline
    "sort", "uniq", "cut", "tr", "diff", "comm", "paste",
    # Source/dev tools (read-only)
    "git log", "git status", "git diff", "git show", "git branch",
    "git remote", "git tag", "git blame", "git rev-parse",
    "python --version", "python3 --version", "node --version",
    "npm list", "pip list", "pip show", "uv pip list",
    "cargo --version", "go version", "rustc --version",
    # System info
    "echo", "printf", "date", "whoami", "hostname", "uname",
    "env", "printenv", "pwd", "which", "type", "command",
    "id", "groups",
    # Process info (read-only)
    "ps", "top", "htop",
})

# Single-word safe commands for fast lookup
_SAFE_WORDS: frozenset[str] = frozenset(
    cmd for cmd in _SAFE_COMMANDS if " " not in cmd
)

# Multi-word safe prefixes (e.g. "git log", "git status")
_SAFE_PREFIXES: tuple[str, ...] = tuple(
    cmd for cmd in _SAFE_COMMANDS if " " in cmd
)

# Shell operators that indicate writes or side effects
_WRITE_OPERATORS = re.compile(r"[>|]|>>|&&|\|\||;")

# Tools that are always safe (read-only)
_SAFE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "web_search",
    "web_fetch",
    "glob",
    "grep",
})

# Tools that always mutate state
_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
})

# Tools classified as read-only for readonly mode
_READONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "web_search",
    "web_fetch",
    "glob",
    "grep",
})


# ── Network domain allowlist ────────────────────────────────────────────────

# Default domains allowed for web tools in auto mode
DEFAULT_ALLOWED_DOMAINS: frozenset[str] = frozenset({
    # Search engines
    "html.duckduckgo.com",
    "duckduckgo.com",
    # Documentation
    "docs.python.org",
    "docs.rs",
    "doc.rust-lang.org",
    "developer.mozilla.org",
    "nodejs.org",
    "pypi.org",
    "pkg.go.dev",
    "crates.io",
    "npmjs.com",
    "www.npmjs.com",
    # Code hosting (read-only pages)
    "github.com",
    "raw.githubusercontent.com",
    "gitlab.com",
    "stackoverflow.com",
    "en.wikipedia.org",
})


def _extract_domain(url: str) -> str | None:
    """Extract domain from a URL, returning None for invalid URLs."""
    try:
        parsed = urlparse(url)
        return parsed.hostname
    except Exception:
        return None


def _is_domain_allowed(url: str, allowed_domains: frozenset[str]) -> bool:
    """Check if a URL's domain is in the allowlist."""
    domain = _extract_domain(url)
    if domain is None:
        return False
    # Check exact match and parent domain match (e.g. "api.github.com" matches "github.com")
    for allowed in allowed_domains:
        if domain == allowed or domain.endswith("." + allowed):
            return True
    return False


# ── Command classification ──────────────────────────────────────────────────

def is_command_safe(command: str) -> bool:
    """Classify a shell command as safe (read-only) or not.

    Uses a whitelist approach: only commands whose first token is in the
    safe set AND that don't contain write operators are considered safe.
    """
    stripped = command.strip()
    if not stripped:
        return False

    # Check for write operators (pipes to commands are OK, redirects are not)
    # Allow simple pipes (|) but block redirects (>, >>)
    if re.search(r">|>>", stripped):
        return False

    # For chained commands (&&, ||, ;), each sub-command must be safe
    if re.search(r"&&|\|\||;", stripped):
        parts = re.split(r"&&|\|\||;", stripped)
        return all(is_command_safe(part) for part in parts)

    # For piped commands, each segment must be safe
    if "|" in stripped:
        parts = stripped.split("|")
        return all(is_command_safe(part) for part in parts)

    # Extract the first token
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return False

    if not tokens:
        return False

    first = tokens[0]

    # Check multi-word prefixes first (e.g. "git log")
    for prefix in _SAFE_PREFIXES:
        if stripped.startswith(prefix):
            return True

    # Check single-word safe commands
    return first in _SAFE_WORDS


# ── Sandbox ─────────────────────────────────────────────────────────────────

class Sandbox:
    """Decides whether tool calls are allowed, denied, or need user approval.

    Args:
        mode: Permission mode (auto, readonly, full_access).
        allowed_domains: Network domains allowed for web tools.
            Defaults to DEFAULT_ALLOWED_DOMAINS.
    """

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.AUTO,
        allowed_domains: frozenset[str] | None = None,
    ):
        self.mode = mode
        self.allowed_domains = (
            allowed_domains if allowed_domains is not None else DEFAULT_ALLOWED_DOMAINS
        )

    def check(self, tool_name: str, arguments: dict[str, Any]) -> Decision:
        """Decide whether a tool call should be allowed, denied, or gated.

        Returns:
            Decision.ALLOW  — execute without prompting
            Decision.DENY   — block the call entirely
            Decision.ASK    — prompt the user for approval
        """
        if self.mode == PermissionMode.FULL_ACCESS:
            logger.debug("Full access: allowing %s", tool_name)
            return Decision.ALLOW

        if self.mode == PermissionMode.READONLY:
            return self._check_readonly(tool_name, arguments)

        # AUTO mode
        return self._check_auto(tool_name, arguments)

    def _check_readonly(self, tool_name: str, arguments: dict[str, Any]) -> Decision:
        """Readonly: only read tools allowed, everything else denied."""
        if tool_name in _READONLY_TOOLS:
            # Web tools still need domain check
            if tool_name in ("web_search", "web_fetch"):
                return self._check_web(tool_name, arguments)
            return Decision.ALLOW

        # Bash is allowed only for safe commands
        if tool_name == "bash":
            command = arguments.get("command", "")
            if is_command_safe(command):
                return Decision.ALLOW
            return Decision.DENY

        return Decision.DENY

    def _check_auto(self, tool_name: str, arguments: dict[str, Any]) -> Decision:
        """Auto: safe reads allowed, writes gated for approval."""
        # Safe tools always allowed
        if tool_name in _SAFE_TOOLS:
            # Web tools need domain check
            if tool_name in ("web_search", "web_fetch"):
                return self._check_web(tool_name, arguments)
            return Decision.ALLOW

        # Write tools need approval
        if tool_name in _WRITE_TOOLS:
            return Decision.ASK

        # Bash: safe commands auto-allowed, others gated
        if tool_name == "bash":
            command = arguments.get("command", "")
            if is_command_safe(command):
                return Decision.ALLOW
            return Decision.ASK

        # Unknown tools: gate them
        return Decision.ASK

    def _check_web(self, tool_name: str, arguments: dict[str, Any]) -> Decision:
        """Check web tool calls against domain allowlist."""
        # web_search doesn't hit arbitrary domains (goes through DDG)
        if tool_name == "web_search":
            return Decision.ALLOW

        # web_fetch: check the URL's domain
        url = arguments.get("url", "")
        if _is_domain_allowed(url, self.allowed_domains):
            return Decision.ALLOW

        # In auto mode, gate unknown domains; in readonly, deny them
        if self.mode == PermissionMode.READONLY:
            return Decision.DENY
        return Decision.ASK
