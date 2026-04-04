"""PhaseGating — prevent premature commitment and ensure verification.

Matches the PhaseGating concept spec: monitors agent progress and injects
reasoning checkpoints when the agent explores too long without acting, or
tries to finish without verifying. This is the feature that moved SWE-bench
resolve rate from 27% to 44%.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PhaseGating:
    """Monitors agent progress and injects guidance at phase transitions."""

    def __init__(
        self,
        max_steps: int,
        explore_ratio: float = 0.3,
        last_resort_ratio: float = 0.8,
    ) -> None:
        self.max_steps = max_steps
        self.explore_ratio = explore_ratio
        self.last_resort_ratio = last_resort_ratio
        self.edited = False
        self.tested = False
        self.step_count = 0
        self._nudge_fired = False
        self._last_resort_fired = False
        self._verification_fired = False

    def tick(self) -> None:
        """Advance the step counter."""
        self.step_count += 1

    def record_edit(self) -> None:
        """Record that the agent has edited a file."""
        self.edited = True

    def record_test(self) -> None:
        """Record that the agent has run tests."""
        self.tested = True

    def track_tool_calls(self, tool_calls: list[Any]) -> None:
        """Scan tool calls to detect edits and test runs."""
        for tc in tool_calls:
            fn = tc.function.name
            if fn in ("edit_file", "write_file"):
                self.record_edit()
            if fn == "bash":
                cmd = tc.function.arguments.get("command", "")
                if any(kw in cmd for kw in (
                    "pytest", "python -m pytest", "test",
                    "npm test", "go test", "cargo test",
                )):
                    self.record_test()

    def check_explore_budget(self) -> str | None:
        """Returns nudge message if explore budget exceeded without edits."""
        explore_budget = max(10, int(self.max_steps * self.explore_ratio))
        if self.step_count == explore_budget and not self.edited and not self._nudge_fired:
            self._nudge_fired = True
            logger.info("Phase nudge at step %d: no edits yet", self.step_count)
            return (
                f"You've spent {self.step_count} steps exploring. Before editing, please:\n"
                f"1. State which file and function you believe contains the bug.\n"
                f"2. Explain the root cause in one sentence.\n"
                f"3. Describe the fix you plan to make.\n"
                f"Then implement the fix. Do NOT start editing until you've done steps 1-3."
            )
        return None

    def check_last_resort(self) -> str | None:
        """Returns nudge message if last-resort threshold hit without edits."""
        last_resort = int(self.max_steps * self.last_resort_ratio)
        if self.step_count == last_resort and not self.edited and not self._last_resort_fired:
            self._last_resort_fired = True
            remaining = self.max_steps - self.step_count
            logger.warning("Last-resort nudge at step %d: still no edits", self.step_count)
            return (
                f"You have only {remaining} steps left. "
                f"You must commit to a fix now. State the file and function, "
                f"then make the edit. An imperfect fix is better than no fix."
            )
        return None

    def check_verification(self) -> str | None:
        """Returns nudge message if trying to complete without testing after edits."""
        if self.edited and not self.tested and not self._verification_fired:
            self._verification_fired = True
            logger.info("Verification gate: agent edited files but didn't run tests")
            return (
                "You edited code but haven't run any tests to verify your changes. "
                "Please run the relevant tests before finishing."
            )
        return None
