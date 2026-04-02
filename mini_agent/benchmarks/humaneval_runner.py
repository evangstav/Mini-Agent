"""HumanEval+ adapter — quick sanity check for code generation.

Usage:
    mini-agent bench humaneval --slice 0:5 --model MiniMax-M2.7

Much simpler than SWE-bench: no Docker, no repo cloning.
Sends function signature + docstring, gets completion, evaluates with evalplus.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from ..agent import Agent
from ..llm import LLMClient
from ..sandbox import PermissionMode, Sandbox
from ..schema import LLMProvider

logger = logging.getLogger(__name__)

HUMANEVAL_SYSTEM_PROMPT = """\
You are an expert Python programmer. You will be given a function signature \
with a docstring. Write the function body that correctly implements the \
specification. Return ONLY the function body code, no explanation, no \
markdown fences, no function signature — just the indented body lines.
"""


class HumanEvalRunner:
    """Run mini-agent against HumanEval+ problems."""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        provider: str = "anthropic",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.api_base = api_base

        if not self.api_key:
            raise ValueError("No API key found. Set MINIMAX_API_KEY or ANTHROPIC_API_KEY.")

    def _make_llm(self) -> LLMClient:
        is_minimax = bool(os.environ.get("MINIMAX_API_KEY"))
        api_base = self.api_base or ("https://api.minimax.io" if is_minimax else "https://api.anthropic.com")
        return LLMClient(
            api_key=self.api_key,
            provider=LLMProvider(self.provider),
            api_base=api_base,
            model=self.model,
        )

    async def run_problem(self, task_id: str, prompt: str) -> dict[str, str]:
        """Run agent on a single HumanEval problem.

        Returns: {task_id, completion}
        """
        llm = self._make_llm()

        # No tools needed — pure code generation
        agent = Agent(
            llm_client=llm,
            system_prompt=HUMANEVAL_SYSTEM_PROMPT,
            tools=[],
            max_steps=1,
            sandbox=Sandbox(PermissionMode.FULL_ACCESS),
        )

        agent.add_user_message(f"Complete this function:\n\n```python\n{prompt}\n```")

        try:
            result = await agent.run()
        except Exception as e:
            logger.error("Failed on %s: %s", task_id, e)
            result = "    pass"

        # Clean up: strip markdown fences if present
        completion = result.strip()
        if completion.startswith("```"):
            lines = completion.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            completion = "\n".join(lines)

        return {"task_id": task_id, "completion": completion}

    async def run_all(
        self,
        slice_range: str | None = None,
        output_path: str = "humaneval_results.jsonl",
    ) -> Path:
        """Run all HumanEval+ problems and write results.

        After running, evaluate with:
            evalplus.evaluate --dataset humaneval --samples humaneval_results.jsonl
        """
        from evalplus.data import get_human_eval_plus

        problems = get_human_eval_plus()
        task_ids = sorted(problems.keys())

        if slice_range:
            start, end = map(int, slice_range.split(":"))
            task_ids = task_ids[start:end]

        logger.info("Running %d HumanEval+ problems with model=%s", len(task_ids), self.model)

        results = []
        out = Path(output_path)

        for i, task_id in enumerate(task_ids):
            problem = problems[task_id]
            logger.info("[%d/%d] %s", i + 1, len(task_ids), task_id)

            pred = await self.run_problem(task_id, problem["prompt"])
            results.append(pred)

            # Write incrementally
            with open(out, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

        logger.info("Wrote %d results to %s", len(results), out)
        logger.info("Evaluate with: evalplus.evaluate --dataset humaneval --samples %s", out)
        return out
