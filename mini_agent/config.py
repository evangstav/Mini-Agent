"""Hierarchical configuration: global → project → CLI.

Reads settings from two TOML files and merges with CLI overrides:
    1. ~/.mini-agent/config.toml   (user-global defaults)
    2. <project>/.mini-agent.toml  (project-local overrides)
    3. CLI arguments / env vars    (highest priority)

Supported settings: model, provider, api_base, max_steps,
permissions (bool), tools (list of enabled tool names).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

GLOBAL_CONFIG_PATH = Path.home() / ".mini-agent" / "config.toml"
PROJECT_CONFIG_NAME = ".mini-agent.toml"


@dataclass
class MiniAgentConfig:
    """Resolved agent configuration."""

    model: str | None = None
    provider: str | None = None
    api_base: str | None = None
    max_steps: int | None = None
    permissions: bool | None = None
    tools: list[str] = field(default_factory=list)

    def merge(self, override: MiniAgentConfig) -> MiniAgentConfig:
        """Return a new config with *override* values taking priority.

        Non-None scalar fields in *override* replace this config's values.
        For ``tools``, a non-empty override list replaces entirely.
        """
        return MiniAgentConfig(
            model=override.model if override.model is not None else self.model,
            provider=override.provider if override.provider is not None else self.provider,
            api_base=override.api_base if override.api_base is not None else self.api_base,
            max_steps=override.max_steps if override.max_steps is not None else self.max_steps,
            permissions=override.permissions if override.permissions is not None else self.permissions,
            tools=override.tools if override.tools else self.tools,
        )


def _load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file, returning {} on missing/corrupt files."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        logger.debug("Config not found: %s", path)
        return {}
    except Exception as e:
        logger.warning("Failed to parse config %s: %s", path, e)
        return {}


def _parse_config(data: dict[str, Any]) -> MiniAgentConfig:
    """Extract known keys from a parsed TOML dict."""
    agent = data.get("agent", data)  # Support [agent] section or top-level
    return MiniAgentConfig(
        model=agent.get("model"),
        provider=agent.get("provider"),
        api_base=agent.get("api_base"),
        max_steps=agent.get("max_steps"),
        permissions=agent.get("permissions"),
        tools=agent.get("tools", []),
    )


def load_config(
    project_dir: str | None = None,
    *,
    cli_overrides: MiniAgentConfig | None = None,
) -> MiniAgentConfig:
    """Load and merge configuration from all layers.

    Resolution order (later wins):
        1. Global   ~/.mini-agent/config.toml
        2. Project  <project_dir>/.mini-agent.toml
        3. CLI      explicit arguments passed by the caller
    """
    # Layer 1: global
    global_data = _load_toml(GLOBAL_CONFIG_PATH)
    config = _parse_config(global_data)

    # Layer 2: project
    if project_dir:
        project_path = Path(project_dir) / PROJECT_CONFIG_NAME
        project_data = _load_toml(project_path)
        if project_data:
            config = config.merge(_parse_config(project_data))

    # Layer 3: CLI overrides
    if cli_overrides:
        config = config.merge(cli_overrides)

    logger.info(
        "Config resolved: model=%s provider=%s max_steps=%s",
        config.model,
        config.provider,
        config.max_steps,
    )
    return config
