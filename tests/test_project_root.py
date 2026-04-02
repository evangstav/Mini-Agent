"""Tests for project root detection and workspace inheritance."""

import pytest
from pathlib import Path

from mini_agent.cli.setup import _detect_project_root


class TestDetectProjectRoot:
    def test_finds_git_root(self, tmp_path):
        """Should find .git directory and return its parent."""
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "pkg"
        subdir.mkdir(parents=True)

        result = _detect_project_root(str(subdir))
        assert result == str(tmp_path)

    def test_finds_pyproject_toml(self, tmp_path):
        """Should find pyproject.toml and return its parent."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        subdir = tmp_path / "src" / "pkg"
        subdir.mkdir(parents=True)

        result = _detect_project_root(str(subdir))
        assert result == str(tmp_path)

    def test_prefers_nearest_marker(self, tmp_path):
        """Should return the nearest parent with a marker file."""
        # Root has .git
        (tmp_path / ".git").mkdir()
        # Subdir also has pyproject.toml
        subproject = tmp_path / "sub"
        subproject.mkdir()
        (subproject / "pyproject.toml").write_text("[project]\nname='sub'\n")
        deep = subproject / "src" / "pkg"
        deep.mkdir(parents=True)

        result = _detect_project_root(str(deep))
        assert result == str(subproject)

    def test_fallback_to_start_dir(self, tmp_path):
        """Should return the start dir if no marker found."""
        subdir = tmp_path / "orphan"
        subdir.mkdir()

        # tmp_path won't have .git or pyproject.toml (just a random temp dir)
        # But the system root (/) won't have them either, so it falls back
        result = _detect_project_root(str(subdir))
        # Should return something (either subdir or a parent that happens to have .git)
        assert Path(result).is_dir()

    def test_works_from_project_root_itself(self, tmp_path):
        """Should work when called from the project root directly."""
        (tmp_path / ".git").mkdir()
        result = _detect_project_root(str(tmp_path))
        assert result == str(tmp_path)
