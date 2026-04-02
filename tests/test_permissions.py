"""Tests for the rule-based permission engine."""

import textwrap
from pathlib import Path

import pytest

from mini_agent.permissions import (
    PermissionRule,
    PermissionRuleset,
    RuleAction,
    load_rules_from_toml,
)
from mini_agent.sandbox import Decision, PermissionMode, Sandbox


# ── PermissionRule.matches ────────────────────────────────────────────────────


class TestPermissionRuleMatches:
    def test_exact_tool_match(self):
        rule = PermissionRule(tool="bash", action=RuleAction.ALLOW)
        assert rule.matches("bash", {}) is True
        assert rule.matches("read_file", {}) is False

    def test_glob_tool_match(self):
        rule = PermissionRule(tool="write_*", action=RuleAction.DENY)
        assert rule.matches("write_file", {}) is True
        assert rule.matches("write_config", {}) is True
        assert rule.matches("read_file", {}) is False

    def test_wildcard_matches_all(self):
        rule = PermissionRule(tool="*", action=RuleAction.ASK)
        assert rule.matches("anything", {}) is True

    def test_arg_constraint_match(self):
        rule = PermissionRule(
            tool="bash",
            action=RuleAction.ALLOW,
            args={"command": "git *"},
        )
        assert rule.matches("bash", {"command": "git status"}) is True
        assert rule.matches("bash", {"command": "git log --oneline"}) is True
        assert rule.matches("bash", {"command": "rm -rf /"}) is False

    def test_arg_constraint_missing_arg(self):
        rule = PermissionRule(
            tool="bash",
            action=RuleAction.ALLOW,
            args={"command": "git *"},
        )
        assert rule.matches("bash", {}) is False

    def test_multiple_arg_constraints(self):
        rule = PermissionRule(
            tool="write_file",
            action=RuleAction.DENY,
            args={"path": "*.env", "content": "*SECRET*"},
        )
        assert rule.matches("write_file", {"path": "prod.env", "content": "SECRET_KEY=x"}) is True
        assert rule.matches("write_file", {"path": "prod.env", "content": "debug=true"}) is False
        assert rule.matches("write_file", {"path": "main.py", "content": "SECRET_KEY=x"}) is False


# ── PermissionRuleset.evaluate ────────────────────────────────────────────────


class TestPermissionRuleset:
    def test_last_rule_wins(self):
        """Rules are evaluated in reverse order — last-added rule wins."""
        ruleset = PermissionRuleset(rules=[
            PermissionRule(tool="bash", action=RuleAction.DENY),
            PermissionRule(tool="bash", action=RuleAction.ALLOW, args={"command": "git *"}),
        ])
        # The ALLOW rule is last and more specific — it wins for git commands
        assert ruleset.evaluate("bash", {"command": "git status"}) == RuleAction.ALLOW
        # For non-git commands, the ALLOW rule doesn't match, so DENY (first from end) wins
        assert ruleset.evaluate("bash", {"command": "rm file"}) == RuleAction.DENY

    def test_no_match_returns_none(self):
        ruleset = PermissionRuleset(rules=[
            PermissionRule(tool="bash", action=RuleAction.ALLOW),
        ])
        assert ruleset.evaluate("read_file", {"path": "foo.py"}) is None

    def test_empty_ruleset(self):
        ruleset = PermissionRuleset()
        assert ruleset.evaluate("bash", {"command": "ls"}) is None


# ── TOML loading ──────────────────────────────────────────────────────────────


class TestLoadRulesFromToml:
    def test_load_valid_config(self, tmp_path: Path):
        config = tmp_path / "permissions.toml"
        config.write_text(textwrap.dedent("""\
            [[rules]]
            tool = "bash"
            action = "allow"
            args = { command = "git *" }

            [[rules]]
            tool = "write_file"
            action = "deny"
            args = { path = "*.env" }

            [[rules]]
            tool = "*"
            action = "ask"
        """))

        ruleset = load_rules_from_toml(config)
        assert len(ruleset.rules) == 3
        assert ruleset.rules[0].tool == "bash"
        assert ruleset.rules[0].action == RuleAction.ALLOW
        assert ruleset.rules[0].args == {"command": "git *"}
        assert ruleset.rules[1].tool == "write_file"
        assert ruleset.rules[1].action == RuleAction.DENY
        assert ruleset.rules[2].tool == "*"
        assert ruleset.rules[2].action == RuleAction.ASK

    def test_missing_file_returns_empty(self, tmp_path: Path):
        ruleset = load_rules_from_toml(tmp_path / "nonexistent.toml")
        assert len(ruleset.rules) == 0

    def test_invalid_toml_returns_empty(self, tmp_path: Path):
        config = tmp_path / "bad.toml"
        config.write_text("this is not valid [[[ toml")
        ruleset = load_rules_from_toml(config)
        assert len(ruleset.rules) == 0

    def test_unknown_action_defaults_to_ask(self, tmp_path: Path):
        config = tmp_path / "permissions.toml"
        config.write_text(textwrap.dedent("""\
            [[rules]]
            tool = "bash"
            action = "yolo"
        """))
        ruleset = load_rules_from_toml(config)
        assert ruleset.rules[0].action == RuleAction.ASK

    def test_empty_rules_section(self, tmp_path: Path):
        config = tmp_path / "permissions.toml"
        config.write_text("[metadata]\nversion = 1\n")
        ruleset = load_rules_from_toml(config)
        assert len(ruleset.rules) == 0


# ── Sandbox integration with rules ───────────────────────────────────────────


class TestSandboxWithRules:
    def test_rule_overrides_auto_mode(self):
        ruleset = PermissionRuleset(rules=[
            PermissionRule(tool="write_file", action=RuleAction.ALLOW),
        ])
        sandbox = Sandbox(mode=PermissionMode.AUTO, permission_rules=ruleset)
        # Normally write_file would be ASK in auto mode
        assert sandbox.check("write_file", {"path": "x", "content": "y"}) == Decision.ALLOW

    def test_rule_overrides_full_access_mode(self):
        ruleset = PermissionRuleset(rules=[
            PermissionRule(tool="bash", action=RuleAction.DENY, args={"command": "rm *"}),
        ])
        sandbox = Sandbox(mode=PermissionMode.FULL_ACCESS, permission_rules=ruleset)
        # Normally everything is ALLOW in full_access, but rules take priority
        assert sandbox.check("bash", {"command": "rm -rf /"}) == Decision.DENY
        # Non-matching falls through to full_access default
        assert sandbox.check("bash", {"command": "ls"}) == Decision.ALLOW

    def test_rule_overrides_readonly_mode(self):
        ruleset = PermissionRuleset(rules=[
            PermissionRule(tool="write_file", action=RuleAction.ALLOW, args={"path": "*.log"}),
        ])
        sandbox = Sandbox(mode=PermissionMode.READONLY, permission_rules=ruleset)
        # Normally write_file is DENY in readonly, but rule allows .log files
        assert sandbox.check("write_file", {"path": "debug.log", "content": ""}) == Decision.ALLOW
        # Non-matching falls through to readonly default
        assert sandbox.check("write_file", {"path": "main.py", "content": ""}) == Decision.DENY

    def test_no_rules_uses_default_behavior(self):
        sandbox = Sandbox(mode=PermissionMode.AUTO)
        assert sandbox.check("read_file", {"path": "foo.py"}) == Decision.ALLOW
        assert sandbox.check("write_file", {"path": "x", "content": "y"}) == Decision.ASK

    def test_empty_ruleset_uses_default(self):
        sandbox = Sandbox(mode=PermissionMode.AUTO, permission_rules=PermissionRuleset())
        assert sandbox.check("read_file", {"path": "foo.py"}) == Decision.ALLOW
        assert sandbox.check("write_file", {"path": "x", "content": "y"}) == Decision.ASK

    def test_bash_arg_pattern_rules(self):
        """With reversed evaluation, last-added rules win — put specific overrides last."""
        ruleset = PermissionRuleset(rules=[
            PermissionRule(tool="bash", action=RuleAction.DENY, args={"command": "npm *"}),
            PermissionRule(tool="bash", action=RuleAction.ALLOW, args={"command": "npm test*"}),
            PermissionRule(tool="bash", action=RuleAction.ALLOW, args={"command": "npm run lint*"}),
        ])
        sandbox = Sandbox(mode=PermissionMode.AUTO, permission_rules=ruleset)
        assert sandbox.check("bash", {"command": "npm test"}) == Decision.ALLOW
        assert sandbox.check("bash", {"command": "npm run lint"}) == Decision.ALLOW
        assert sandbox.check("bash", {"command": "npm install malware"}) == Decision.DENY
        # Non-npm commands fall through to auto mode default
        assert sandbox.check("bash", {"command": "ls"}) == Decision.ALLOW
