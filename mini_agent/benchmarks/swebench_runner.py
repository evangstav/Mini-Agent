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
from ..tools.find_definition import FindDefinitionTool
from ..tools.glob_tool import GlobTool
from ..tools.grep_tool import GrepTool
from ..tools.list_dir import ListDirTool

logger = logging.getLogger(__name__)

SWEBENCH_SYSTEM_PROMPT = """\
You are an expert software engineer solving a GitHub issue. Follow these phases strictly.

## Phase 1: EXPLORE (first 5-8 steps)
- Read the issue carefully. Identify what behavior is wrong and what is expected.
- Run `find . -type f -name "*.py" | head -20` or use glob to understand the repo structure.
- Read the README or setup.py/pyproject.toml to understand the project.
- DO NOT edit any files yet.

## Phase 2: LOCALIZE (next 3-5 steps)
- Search for relevant code using grep with specific terms from the issue.
- Read the files that are most likely to contain the bug.
- Identify the exact function/method and line range that needs to change.
- Think about the root cause before proposing a fix.

## Phase 3: REPRODUCE (1-2 steps, if possible)
- Write a small script that demonstrates the bug, then run it.
- If the issue mentions specific test commands, run them to see the failure.
- This confirms you understand the bug correctly.

## Phase 4: FIX (2-5 steps)
- Make the minimal change needed to fix the issue.
- If you need to add an import, do it in a separate edit.
- Keep changes focused — do not refactor unrelated code.

## Phase 5: VERIFY (2-3 steps)
- Re-read the changed file to confirm your edit looks correct.
- Run the reproduction script or tests to confirm the fix works.
- Run `python -m pytest <relevant_test_file> -x` if tests exist.

## Rules
- Only modify source files, not test files (unless the issue requires it).
- Only state facts you have verified by reading files.
- If you cannot find the relevant code, say so — do not guess.
- Keep changes minimal. One-line fixes are preferred when correct.
- If an edit introduces a syntax warning, fix it in the next step.
"""


class SWEBenchRunner:
    """Run mini-agent against SWE-bench instances."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_steps: int = 50,
    ) -> None:
        from ..llm import auto_detect_provider
        detected_key, detected_model, detected_base, detected_provider = auto_detect_provider()

        self.api_key = api_key or detected_key
        self.model = model or detected_model
        self.api_base = api_base or detected_base
        self.provider = LLMProvider(provider) if provider else detected_provider
        self.max_steps = max_steps

    def _make_llm(self) -> LLMClient:
        return LLMClient(
            api_key=self.api_key,
            provider=self.provider,
            api_base=self.api_base,
            model=self.model,
        )

    def _checkout_repo(self, repo: str, base_commit: str, workdir: Path) -> bool:
        """Clone a repo and checkout the specified commit."""
        repo_url = f"https://github.com/{repo}.git"
        try:
            # Try shallow clone first (fast), then deepen if commit not found
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(workdir)],
                capture_output=True, timeout=300, check=True,
            )
            # Fetch the specific commit
            subprocess.run(
                ["git", "fetch", "origin", base_commit],
                cwd=str(workdir), capture_output=True, timeout=300, check=True,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=str(workdir), capture_output=True, timeout=30, check=True,
            )
            return True
        except subprocess.CalledProcessError:
            # Shallow fetch failed — fall back to full clone
            logger.info("Shallow fetch failed for %s@%s, doing full clone...", repo, base_commit[:8])
            import shutil
            if workdir.exists():
                shutil.rmtree(workdir)
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(workdir)],
                    capture_output=True, timeout=600, check=True,
                )
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=str(workdir), capture_output=True, timeout=30, check=True,
                )
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error("Failed to checkout %s@%s: %s", repo, base_commit, e)
                return False
        except subprocess.TimeoutExpired as e:
            logger.error("Timeout cloning %s: %s", repo, e)
            return False

    def _setup_repo(self, workdir: Path) -> None:
        """Install repo in development mode so tests can run."""
        try:
            # Try pip install -e . (works for most Python repos)
            subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=str(workdir), capture_output=True, timeout=300,
            )
            logger.info("Installed repo via pip install -e .")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Could not install repo dependencies")

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

        # Install repo dependencies so tests can run
        self._setup_repo(workdir)

        # Create agent with tools scoped to the repo
        llm = self._make_llm()
        tools = [
            ReadTool(workspace_dir=str(workdir)),
            EditTool(workspace_dir=str(workdir)),
            FindDefinitionTool(workspace_dir=str(workdir)),
            ListDirTool(workspace_dir=str(workdir)),
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

    async def run_instance_best_of_n(
        self, instance: dict[str, Any], n: int = 3,
    ) -> dict[str, str]:
        """Run agent N times on the same instance, pick the most common non-empty patch.

        Based on CodeMonkeys (Stanford 2025): Best-of-N with majority voting
        gives +15-30% relative improvement for 3-5x cost.
        """
        import shutil
        from collections import Counter

        patches = []
        for attempt in range(n):
            with tempfile.TemporaryDirectory(prefix=f"swe_attempt_{attempt}_") as tmpdir:
                workdir = Path(tmpdir) / "repo"
                pred = await self.run_instance(instance, workdir)
                patch = pred["model_patch"]
                if patch.strip():
                    patches.append(patch)
                logger.info(
                    "Attempt %d/%d for %s: patch=%d chars",
                    attempt + 1, n, instance["instance_id"], len(patch),
                )

        if not patches:
            return {
                "instance_id": instance["instance_id"],
                "model_name_or_path": f"mini-agent-{self.model}",
                "model_patch": "",
            }

        # Majority voting: pick the most common patch
        counts = Counter(patches)
        best_patch, best_count = counts.most_common(1)[0]
        logger.info(
            "Best-of-%d for %s: %d/%d attempts produced patches, majority=%d",
            n, instance["instance_id"], len(patches), n, best_count,
        )

        return {
            "instance_id": instance["instance_id"],
            "model_name_or_path": f"mini-agent-{self.model}",
            "model_patch": best_patch,
        }

    async def run_dataset(
        self,
        subset: str = "verified",
        split: str = "test",
        slice_range: str | None = None,
        output_path: str = "predictions.jsonl",
        attempts: int = 1,
    ) -> Path:
        """Run the agent against multiple SWE-bench instances.

        Args:
            subset: "verified" (500 instances) or "lite" (300) or "full"
            split: dataset split (usually "test")
            slice_range: e.g. "0:5" for first 5 instances
            output_path: where to write predictions JSONL
            attempts: Best-of-N attempts per instance (1=single run, 3=majority voting)
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
            logger.info("[%d/%d] %s (attempts=%d)", i + 1, len(ds), instance["instance_id"], attempts)

            if attempts > 1:
                pred = await self.run_instance_best_of_n(instance, n=attempts)
            else:
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
