"""Structured logging — file rotation to ~/.mini-agent/logs/, --verbose for debug."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path.home() / ".mini-agent" / "logs"
LOG_FILE = "mini-agent.log"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(*, verbose: bool = False) -> None:
    """Configure logging for the mini-agent package.

    Always writes INFO+ to ~/.mini-agent/logs/mini-agent.log with rotation.
    When verbose=True, also writes DEBUG to stderr.
    """
    global _configured
    if _configured:
        return
    _configured = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("mini_agent")
    root.setLevel(logging.DEBUG)

    # File handler — INFO+ with rotation
    fh = RotatingFileHandler(
        LOG_DIR / LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(fh)

    # Stderr handler — only when verbose
    if verbose:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(sh)
