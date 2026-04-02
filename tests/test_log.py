"""Tests for structured logging module."""

import logging
from pathlib import Path
from unittest.mock import patch

from mini_agent.log import LOG_DIR, setup_logging


class TestSetupLogging:
    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "logs"
        with patch("mini_agent.log.LOG_DIR", log_dir), \
             patch("mini_agent.log._configured", False):
            setup_logging(verbose=False)
            assert log_dir.exists()

    def test_configures_root_logger(self, tmp_path):
        log_dir = tmp_path / "logs"
        with patch("mini_agent.log.LOG_DIR", log_dir), \
             patch("mini_agent.log._configured", False):
            setup_logging(verbose=False)
            root = logging.getLogger("mini_agent")
            assert root.level == logging.DEBUG
            # Should have at least a file handler
            assert len(root.handlers) >= 1

    def test_verbose_adds_stderr_handler(self, tmp_path):
        log_dir = tmp_path / "logs"
        with patch("mini_agent.log.LOG_DIR", log_dir), \
             patch("mini_agent.log._configured", False):
            setup_logging(verbose=True)
            root = logging.getLogger("mini_agent")
            stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)
                               and not isinstance(h, logging.FileHandler)]
            assert len(stream_handlers) >= 1

    def test_idempotent(self, tmp_path):
        log_dir = tmp_path / "logs"
        with patch("mini_agent.log.LOG_DIR", log_dir), \
             patch("mini_agent.log._configured", False):
            setup_logging(verbose=False)
            handler_count = len(logging.getLogger("mini_agent").handlers)
            setup_logging(verbose=True)  # Second call should be no-op
            assert len(logging.getLogger("mini_agent").handlers) == handler_count
