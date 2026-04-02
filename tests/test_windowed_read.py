"""Tests for windowed file reading (SWE-Agent research-backed feature)."""

import pytest

from mini_agent.tools.file_tools import ReadTool


class TestWindowedRead:
    @pytest.mark.asyncio
    async def test_small_file_returned_in_full(self, tmp_path):
        """Files under the line cap are returned completely."""
        f = tmp_path / "small.py"
        f.write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")

        tool = ReadTool(workspace_dir=str(tmp_path))
        result = await tool.execute(path=str(f))

        assert result.success
        assert "line 49" in result.content
        assert "Showing lines" not in result.content  # No truncation notice

    @pytest.mark.asyncio
    async def test_large_file_truncated_at_default(self, tmp_path):
        """Files over 200 lines get truncated with a notice."""
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")

        tool = ReadTool(workspace_dir=str(tmp_path))
        result = await tool.execute(path=str(f))

        assert result.success
        assert "line 199" in result.content  # Line 200 visible
        assert "line 300" not in result.content  # Line 300 not visible
        assert "Showing lines" in result.content
        assert "500" in result.content  # Total line count shown
        assert "offset=" in result.content  # Pagination hint

    @pytest.mark.asyncio
    async def test_explicit_limit_overrides_default(self, tmp_path):
        """Explicit limit= parameter overrides the 200-line default."""
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")

        tool = ReadTool(workspace_dir=str(tmp_path))
        result = await tool.execute(path=str(f), limit=10)

        assert result.success
        assert "line 9" in result.content
        assert "line 10" not in result.content
        # Explicit limit should NOT show "Showing lines" notice
        assert "Showing lines" not in result.content

    @pytest.mark.asyncio
    async def test_offset_works_with_default_cap(self, tmp_path):
        """Offset lets you paginate through a large file."""
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")

        tool = ReadTool(workspace_dir=str(tmp_path))
        result = await tool.execute(path=str(f), offset=201)

        assert result.success
        assert "line 200" in result.content
        assert "line 399" in result.content

    @pytest.mark.asyncio
    async def test_custom_max_lines(self, tmp_path):
        """Tool can be configured with a different line cap."""
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")

        tool = ReadTool(workspace_dir=str(tmp_path), max_lines=50)
        result = await tool.execute(path=str(f))

        assert result.success
        assert "Showing lines" in result.content
        assert "100" in result.content  # Total lines shown

    @pytest.mark.asyncio
    async def test_exactly_at_limit(self, tmp_path):
        """File with exactly max_lines lines should NOT show truncation notice."""
        f = tmp_path / "exact.py"
        f.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")

        tool = ReadTool(workspace_dir=str(tmp_path))
        result = await tool.execute(path=str(f))

        assert result.success
        assert "Showing lines" not in result.content
