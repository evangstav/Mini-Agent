"""Rule-based permission engine for tool calls.

Permission rule syntax: Tool(pattern) → allow/deny/ask
Config in .mini-agent/permissions.toml or project .mini-agent.toml.

Rules are evaluated top-to-bottom; first match wins. If no rule matches,
the sandbox's default classification applies (fallthrough).
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class RuleAction(str, Enum):
    """Action to take when a permission rule matches."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class PermissionRule:
    """A single permission rule.

    Attributes:
        tool: Glob pattern matching tool names (e.g. "bash", "write_*", "*").
        action: What to do when the rule matches.
        args: Optional argument constraints. Keys are argument names,
              values are glob patterns that must match for the rule to apply.
    """

    tool: str
    action: RuleAction
    args: dict[str, str] = field(default_factory=dict)

    def matches(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Check if this rule matches a tool call."""
        if not fnmatch.fnmatch(tool_name, self.tool):
            return False

        for arg_key, arg_pattern in self.args.items():
            arg_value = str(arguments.get(arg_key, ""))
            if not fnmatch.fnmatch(arg_value, arg_pattern):
                return False

        return True


@dataclass
class PermissionRuleset:
    """Ordered set of permission rules.

    Rules are evaluated in reverse order (last-added rule wins).
    The first match encountered during this reverse walk determines the action.
    This means project-level rules (appended after global rules) take priority.
    """

    rules: list[PermissionRule] = field(default_factory=list)

    def evaluate(self, tool_name: str, arguments: dict[str, Any]) -> RuleAction | None:
        """Evaluate rules against a tool call.

        Rules are walked in reverse order so that later rules (e.g., project-local)
        override earlier ones (e.g., global defaults). Returns the action of the
        first matching rule in this reverse walk, or None if no rule matches
        (indicating the sandbox default should apply).
        """
        for rule in reversed(self.rules):
            if rule.matches(tool_name, arguments):
                logger.debug(
                    "Permission rule matched: %s(%s) → %s",
                    rule.tool,
                    rule.args or "*",
                    rule.action.value,
                )
                return rule.action
        return None


def _parse_rule(entry: dict[str, Any]) -> PermissionRule:
    """Parse a single rule entry from TOML config."""
    tool = entry.get("tool", "*")
    action_str = entry.get("action", "ask")
    args = entry.get("args", {})

    try:
        action = RuleAction(action_str.lower())
    except ValueError:
        logger.warning("Unknown rule action '%s', defaulting to 'ask'", action_str)
        action = RuleAction.ASK

    return PermissionRule(tool=tool, action=action, args=args)


def load_rules_from_toml(path: Path) -> PermissionRuleset:
    """Load permission rules from a TOML file.

    Expected format:
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
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        logger.debug("Permission config not found: %s", path)
        return PermissionRuleset()
    except Exception as e:
        logger.warning("Failed to parse permission config %s: %s", path, e)
        return PermissionRuleset()

    raw_rules = data.get("rules", [])
    rules = [_parse_rule(entry) for entry in raw_rules]
    logger.info("Loaded %d permission rules from %s", len(rules), path)
    return PermissionRuleset(rules=rules)


def load_rules(project_dir: str | None = None) -> PermissionRuleset:
    """Load permission rules from standard config locations.

    Search order (later files' rules are appended after earlier ones):
        1. ~/.mini-agent/permissions.toml  (user-global)
        2. <project>/.mini-agent.toml      (project-local)

    Rules from both files are concatenated into a single list.
    PermissionRuleset.evaluate() walks this list in reverse, so
    project rules (appended last) are checked first and naturally
    override global defaults.
    """
    combined_rules: list[PermissionRule] = []

    # User-global config
    global_path = Path.home() / ".mini-agent" / "permissions.toml"
    global_ruleset = load_rules_from_toml(global_path)
    combined_rules.extend(global_ruleset.rules)

    # Project-local config
    if project_dir:
        project_path = Path(project_dir) / ".mini-agent.toml"
        project_ruleset = load_rules_from_toml(project_path)
        combined_rules.extend(project_ruleset.rules)

    return PermissionRuleset(rules=combined_rules)
