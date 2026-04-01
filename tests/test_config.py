"""Tests for mini_agent.config — hierarchical config loading."""

import textwrap
from pathlib import Path

import pytest

from mini_agent.config import MiniAgentConfig, load_config, _load_toml, _parse_config


# ── MiniAgentConfig.merge ────────────────────────────────────────────────────


def test_merge_override_scalars():
    base = MiniAgentConfig(model="base-model", provider="anthropic", max_steps=50)
    override = MiniAgentConfig(model="override-model", max_steps=100)
    merged = base.merge(override)
    assert merged.model == "override-model"
    assert merged.provider == "anthropic"  # kept from base
    assert merged.max_steps == 100


def test_merge_none_does_not_overwrite():
    base = MiniAgentConfig(model="keep-me", api_base="https://example.com")
    override = MiniAgentConfig()  # all None
    merged = base.merge(override)
    assert merged.model == "keep-me"
    assert merged.api_base == "https://example.com"


def test_merge_tools_override():
    base = MiniAgentConfig(tools=["bash", "read"])
    override = MiniAgentConfig(tools=["bash"])
    merged = base.merge(override)
    assert merged.tools == ["bash"]


def test_merge_empty_tools_keeps_base():
    base = MiniAgentConfig(tools=["bash", "read"])
    override = MiniAgentConfig(tools=[])
    merged = base.merge(override)
    assert merged.tools == ["bash", "read"]


# ── _load_toml ───────────────────────────────────────────────────────────────


def test_load_toml_missing_file(tmp_path):
    result = _load_toml(tmp_path / "nope.toml")
    assert result == {}


def test_load_toml_valid(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text(textwrap.dedent("""\
        [agent]
        model = "gpt-4"
        max_steps = 30
    """))
    result = _load_toml(f)
    assert result["agent"]["model"] == "gpt-4"
    assert result["agent"]["max_steps"] == 30


def test_load_toml_corrupt(tmp_path):
    f = tmp_path / "bad.toml"
    f.write_text("not valid toml {{{{")
    result = _load_toml(f)
    assert result == {}


# ── _parse_config ────────────────────────────────────────────────────────────


def test_parse_config_agent_section():
    data = {"agent": {"model": "claude", "provider": "anthropic", "max_steps": 25}}
    cfg = _parse_config(data)
    assert cfg.model == "claude"
    assert cfg.provider == "anthropic"
    assert cfg.max_steps == 25


def test_parse_config_top_level():
    """Support flat config (no [agent] section)."""
    data = {"model": "gpt-4", "tools": ["bash"]}
    cfg = _parse_config(data)
    assert cfg.model == "gpt-4"
    assert cfg.tools == ["bash"]


# ── load_config integration ──────────────────────────────────────────────────


def test_load_config_global_only(tmp_path, monkeypatch):
    global_dir = tmp_path / ".mini-agent"
    global_dir.mkdir()
    (global_dir / "config.toml").write_text(textwrap.dedent("""\
        [agent]
        model = "global-model"
        max_steps = 40
    """))
    monkeypatch.setattr("mini_agent.config.GLOBAL_CONFIG_PATH", global_dir / "config.toml")

    cfg = load_config(project_dir=str(tmp_path))
    assert cfg.model == "global-model"
    assert cfg.max_steps == 40


def test_load_config_project_overrides_global(tmp_path, monkeypatch):
    # Global
    global_dir = tmp_path / ".mini-agent"
    global_dir.mkdir()
    (global_dir / "config.toml").write_text(textwrap.dedent("""\
        [agent]
        model = "global-model"
        max_steps = 40
        provider = "anthropic"
    """))
    monkeypatch.setattr("mini_agent.config.GLOBAL_CONFIG_PATH", global_dir / "config.toml")

    # Project
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".mini-agent.toml").write_text(textwrap.dedent("""\
        [agent]
        model = "project-model"
        max_steps = 20
    """))

    cfg = load_config(project_dir=str(project_dir))
    assert cfg.model == "project-model"
    assert cfg.max_steps == 20
    assert cfg.provider == "anthropic"  # inherited from global


def test_load_config_cli_overrides_all(tmp_path, monkeypatch):
    # Global
    global_dir = tmp_path / ".mini-agent"
    global_dir.mkdir()
    (global_dir / "config.toml").write_text('[agent]\nmodel = "global"\n')
    monkeypatch.setattr("mini_agent.config.GLOBAL_CONFIG_PATH", global_dir / "config.toml")

    # Project
    (tmp_path / ".mini-agent.toml").write_text('[agent]\nmodel = "project"\n')

    cli = MiniAgentConfig(model="cli-model")
    cfg = load_config(project_dir=str(tmp_path), cli_overrides=cli)
    assert cfg.model == "cli-model"


def test_load_config_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr("mini_agent.config.GLOBAL_CONFIG_PATH", tmp_path / "nope.toml")
    cfg = load_config(project_dir=str(tmp_path))
    assert cfg.model is None
    assert cfg.max_steps is None
    assert cfg.tools == []
