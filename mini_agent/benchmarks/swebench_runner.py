"""SWE-bench adapter — run mini-agent against SWE-bench instances.

Usage:
    mini-agent bench swebench --slice 0:5 --model MiniMax-M2.7

The adapter:
1. Loads problem statements from the HuggingFace dataset
2. Clones each repo at the correct commit
3. Runs our Agent with the problem as the user message
4. Captures `git diff` as the patch
5. Writes predictions to JSONL for evaluation
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..agent import Agent
from ..llm import LLMClient
from ..sandbox import PermissionMode, Sandbox
from ..schema import LLMProvider
from ..tools.bash_tool import BashTool
from ..tools.file_tools import EditTool, ReadTool
from ..tools.glob_tool import GlobTool
from ..tools.grep_tool import GrepTool

logger = logging.getLogger(__name__)

SWEBENCH_SYSTEM_PROMPT = """\
You are an expert software engineer solving a GitHub issue.

Instructions:
1. Read the problem statement carefully.
2. Explore the repository to understand the codebase structure.
3. Locate the relevant source files.
4. Understand the root cause of the bug or the feature request.
5. Implement a fix using the available tools.
6. Verify your fix is correct by reading the changed files.

Rules:
- Only modify files that are necessary to fix the issue.
- Do not add tests unless the issue specifically requests them.
- Do not modify test files unless the fix requires it.
- Keep changes minimal and focused.
- Only state facts you have verified by reading files.
"""


class SWEBenchRunner:
    """Run mini-agent against SWE-bench instances."""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        provider: str = "anthropic",
        api_key: str | None = None,
        api_base: str | None = None,
        max_steps: int = 30,
    ) -> None:
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.api_base = api_base
        self.max_steps = max_steps

        if not self.api_key:
            raise ValueError("No API key found. Set MINIMAX_API_KEY or ANTHROPIC_API_KEY.")

    def _make_llm(self) -> LLMClient:
        is_minimax = "minimax" in self.api_key.lower() or "MINIMAX" in os.environ.get("MINIMAX_API_KEY", "")
        api_base = self.api_base or ("https://api.minimax.io" if is_minimax else "https://api.anthropic.com")
        return LLMClient(
            api_key=self.api_key,
            provider=LLMProvider(self.provider),
            api_base=api_base,
            model=self.model,
        )

    def _checkout_repo(self, repo: str, base_commit: str, workdir: Path) -> bool:
        """Clone a repo and checkout the specified commit."""
        try:
            # Clone with depth 1 for speed, then fetch the specific commit
            repo_url = f"https://github.com/{repo}.git"
            subprocess.run(
                ["git", "clone", "--depth", "50", repo_url, str(workdir)],
                capture_output=True, timeout=120, check=True,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=str(workdir), capture_output=True, timeout=30, check=True,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error("Failed to checkout %s@%s: %s", repo, base_commit, e)
            return False

    def _get_diff(self, workdir: Path) -> str:
        """Capture git diff from the working directory."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=str(workdir), capture_output=True, text=True, timeout=30,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return ""

    async def run_instance(self, instance: dict[str, Any], workdir: Path) -> dict[str, str]:
        """Run agent on a single SWE-bench instance.

        Returns: {instance_id, model_name_or_path, model_patch}
        """
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem = instance["problem_statement"]

        logger.info("Running instance %s (%s@%s)", instance_id, repo, base_commit[:8])

        # Checkout the repo
        if not self._checkout_repo(repo, base_commit, workdir):
            return {
                "instance_id": instance_id,
                "model_name_or_path": f"mini-agent-{self.model}",
                "model_patch": "",
            }

        # Create agent with tools scoped to the repo
        llm = self._make_llm()
        tools = [
            ReadTool(workspace_dir=str(workdir)),
            EditTool(workspace_dir=str(workdir)),
            GlobTool(workspace_dir=str(workdir)),
            GrepTool(workspace_dir=str(workdir)),
            BashTool(workspace_dir=str(workdir)),
        ]

        agent = Agent(
            llm_client=llm,
            system_prompt=SWEBENCH_SYSTEM_PROMPT,
            tools=tools,
            max_steps=self.max_steps,
            sandbox=Sandbox(PermissionMode.FULL_ACCESS),
        )

        agent.add_user_message(
            f"## GitHub Issue\n\n{problem}\n\n"
            f"The repository is checked out at `{workdir}`. "
            f"Explore the code, find the root cause, and fix the issue."
        )

        # Run the agent
        try:
            await agent.run()
        except Exception as e:
            logger.error("Agent failed on %s: %s", instance_id, e)

        # Capture the patch
        patch = self._get_diff(workdir)

        logger.info(
            "Instance %s: patch=%d chars, steps used, tokens=%d",
            instance_id, len(patch), agent.token_usage.total_tokens,
        )

        return {
            "instance_id": instance_id,
            "model_name_or_path": f"mini-agent-{self.model}",
            "model_patch": patch,
        }

    async def run_dataset(
        self,
        subset: str = "verified",
        split: str = "test",
        slice_range: str | None = None,
        output_path: str = "predictions.jsonl",
    ) -> Path:
        """Run the agent against multiple SWE-bench instances.

        Args:
            subset: "verified" (500 instances) or "lite" (300) or "full"
            split: dataset split (usually "test")
            slice_range: e.g. "0:5" for first 5 instances
            output_path: where to write predictions JSONL
        """
        from datasets import load_dataset

        dataset_name = {
            "verified": "princeton-nlp/SWE-bench_Verified",
            "lite": "princeton-nlp/SWE-bench_Lite",
            "full": "princeton-nlp/SWE-bench",
        }.get(subset, subset)

        logger.info("Loading dataset %s (split=%s)", dataset_name, split)
        ds = load_dataset(dataset_name, split=split)

        if slice_range:
            start, end = map(int, slice_range.split(":"))
            ds = ds.select(range(start, min(end, len(ds))))

        logger.info("Running %d instances with model=%s max_steps=%d", len(ds), self.model, self.max_steps)

        predictions = []
        out = Path(output_path)

        for i, instance in enumerate(ds):
            logger.info("[%d/%d] %s", i + 1, len(ds), instance["instance_id"])

            with tempfile.TemporaryDirectory(prefix=f"swebench_{i}_") as tmpdir:
                workdir = Path(tmpdir) / "repo"
                pred = await self.run_instance(instance, workdir)
                predictions.append(pred)

                # Write incrementally so we don't lose progress on crash
                with open(out, "w") as f:
                    for p in predictions:
                        f.write(json.dumps(p) + "\n")

        logger.info("Wrote %d predictions to %s", len(predictions), out)
        return out
