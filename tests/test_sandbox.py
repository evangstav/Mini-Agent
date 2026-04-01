"""Tests for sandbox permission modes and command classification."""

import pytest

from mini_agent.sandbox import (
    DEFAULT_ALLOWED_DOMAINS,
    Decision,
    PermissionMode,
    Sandbox,
    is_command_safe,
)


# ── is_command_safe tests ───────────────────────────────────────────────────


class TestCommandClassification:
    """Test the whitelist-based command classification."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls",
            "ls -la",
            "cat foo.py",
            "head -n 10 main.go",
            "tail -f log.txt",
            "grep -r 'pattern' .",
            "rg pattern",
            "find . -name '*.py'",
            "git status",
            "git log --oneline",
            "git diff HEAD~1",
            "git branch -a",
            "echo hello",
            "pwd",
            "whoami",
            "wc -l file.txt",
            "tree",
            "du -sh .",
            "df -h",
            "which python",
            "python --version",
            "node --version",
            "env",
            "date",
        ],
    )
    def test_safe_commands(self, command: str):
        assert is_command_safe(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "rm file.txt",
            "rm -rf /",
            "mkdir new_dir",
            "mv old new",
            "cp src dst",
            "pip install requests",
            "npm install",
            "curl https://example.com",
            "wget https://example.com",
            "python script.py",
            "node app.js",
            "make build",
            "docker run ubuntu",
            "sudo anything",
            "chmod 777 file",
            "kill -9 1234",
        ],
    )
    def test_unsafe_commands(self, command: str):
        assert is_command_safe(command) is False

    def test_redirect_makes_unsafe(self):
        assert is_command_safe("echo hello > file.txt") is False
        assert is_command_safe("cat foo >> bar") is False

    def test_pipe_safe_to_safe(self):
        assert is_command_safe("cat file.py | grep pattern") is True
        assert is_command_safe("ls -la | sort | head") is True

    def test_pipe_safe_to_unsafe(self):
        assert is_command_safe("curl url | python") is False

    def test_chained_safe_commands(self):
        assert is_command_safe("git status && git log --oneline") is True

    def test_chained_with_unsafe(self):
        assert is_command_safe("ls && rm file") is False

    def test_empty_command(self):
        assert is_command_safe("") is False
        assert is_command_safe("   ") is False


# ── Sandbox decision tests ──────────────────────────────────────────────────


class TestSandboxFullAccess:
    """Full access mode allows everything."""

    def setup_method(self):
        self.sandbox = Sandbox(mode=PermissionMode.FULL_ACCESS)

    def test_allows_read(self):
        assert self.sandbox.check("read_file", {"path": "foo.py"}) == Decision.ALLOW

    def test_allows_write(self):
        assert self.sandbox.check("write_file", {"path": "x", "content": "y"}) == Decision.ALLOW

    def test_allows_bash(self):
        assert self.sandbox.check("bash", {"command": "rm -rf /"}) == Decision.ALLOW

    def test_allows_unknown_tool(self):
        assert self.sandbox.check("unknown_tool", {}) == Decision.ALLOW


class TestSandboxReadonly:
    """Readonly mode: only reads allowed, writes denied."""

    def setup_method(self):
        self.sandbox = Sandbox(mode=PermissionMode.READONLY)

    def test_allows_read(self):
        assert self.sandbox.check("read_file", {"path": "foo.py"}) == Decision.ALLOW

    def test_denies_write(self):
        assert self.sandbox.check("write_file", {"path": "x", "content": "y"}) == Decision.DENY

    def test_denies_edit(self):
        assert self.sandbox.check("edit_file", {"path": "x", "old_str": "", "new_str": ""}) == Decision.DENY

    def test_allows_safe_bash(self):
        assert self.sandbox.check("bash", {"command": "ls -la"}) == Decision.ALLOW
        assert self.sandbox.check("bash", {"command": "git status"}) == Decision.ALLOW

    def test_denies_unsafe_bash(self):
        assert self.sandbox.check("bash", {"command": "rm file.txt"}) == Decision.DENY
        assert self.sandbox.check("bash", {"command": "pip install foo"}) == Decision.DENY

    def test_allows_web_search(self):
        assert self.sandbox.check("web_search", {"query": "python docs"}) == Decision.ALLOW

    def test_allows_web_fetch_allowed_domain(self):
        assert self.sandbox.check("web_fetch", {"url": "https://docs.python.org/3/"}) == Decision.ALLOW

    def test_denies_web_fetch_unknown_domain(self):
        assert self.sandbox.check("web_fetch", {"url": "https://evil.com/payload"}) == Decision.DENY

    def test_denies_unknown_tool(self):
        assert self.sandbox.check("some_new_tool", {}) == Decision.DENY


class TestSandboxAuto:
    """Auto mode: safe reads allowed, writes gated."""

    def setup_method(self):
        self.sandbox = Sandbox(mode=PermissionMode.AUTO)

    def test_allows_read(self):
        assert self.sandbox.check("read_file", {"path": "foo.py"}) == Decision.ALLOW

    def test_gates_write(self):
        assert self.sandbox.check("write_file", {"path": "x", "content": "y"}) == Decision.ASK

    def test_gates_edit(self):
        assert self.sandbox.check("edit_file", {"path": "x", "old_str": "", "new_str": ""}) == Decision.ASK

    def test_allows_safe_bash(self):
        assert self.sandbox.check("bash", {"command": "ls -la"}) == Decision.ALLOW
        assert self.sandbox.check("bash", {"command": "git log --oneline"}) == Decision.ALLOW

    def test_gates_unsafe_bash(self):
        assert self.sandbox.check("bash", {"command": "pip install requests"}) == Decision.ASK
        assert self.sandbox.check("bash", {"command": "python script.py"}) == Decision.ASK

    def test_allows_web_search(self):
        assert self.sandbox.check("web_search", {"query": "test"}) == Decision.ALLOW

    def test_allows_web_fetch_allowed_domain(self):
        assert self.sandbox.check("web_fetch", {"url": "https://github.com/foo/bar"}) == Decision.ALLOW

    def test_gates_web_fetch_unknown_domain(self):
        assert self.sandbox.check("web_fetch", {"url": "https://unknown-site.com/data"}) == Decision.ASK

    def test_gates_unknown_tool(self):
        assert self.sandbox.check("some_new_tool", {}) == Decision.ASK


class TestDomainAllowlist:
    """Test domain matching for web tools."""

    def test_exact_match(self):
        sandbox = Sandbox(mode=PermissionMode.AUTO)
        assert sandbox.check("web_fetch", {"url": "https://github.com/repo"}) == Decision.ALLOW

    def test_subdomain_match(self):
        sandbox = Sandbox(mode=PermissionMode.AUTO)
        assert sandbox.check("web_fetch", {"url": "https://api.github.com/repos"}) == Decision.ALLOW

    def test_custom_allowlist(self):
        sandbox = Sandbox(
            mode=PermissionMode.AUTO,
            allowed_domains=frozenset({"example.com"}),
        )
        assert sandbox.check("web_fetch", {"url": "https://example.com/page"}) == Decision.ALLOW
        assert sandbox.check("web_fetch", {"url": "https://github.com/foo"}) == Decision.ASK

    def test_empty_url(self):
        sandbox = Sandbox(mode=PermissionMode.AUTO)
        assert sandbox.check("web_fetch", {"url": ""}) == Decision.ASK

    def test_invalid_url(self):
        sandbox = Sandbox(mode=PermissionMode.AUTO)
        assert sandbox.check("web_fetch", {"url": "not-a-url"}) == Decision.ASK
