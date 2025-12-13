"""DeepSWE environment adapter for AFFINETES framework

This adapter wraps the original DeepSWE SandboxEnv to work with AFFINETES.
It preserves all original logic including sandbox management, tool execution, and test evaluation.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import sys
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from datasets import load_dataset
from models import Challenge


class Actor:
    """AFFINETES Actor interface for DeepSWE environment
    
    Note: DeepSWE is a multi-turn sandbox environment. For AFFINETES integration,
    we provide a simplified interface that generates challenges. The full sandbox
    environment logic remains in the original deepswe.py for compatibility.
    """
    
    def __init__(
        self,
        dataset_name: str = "R2E-Gym/R2E-Gym-Subset",
        dataset_split: str = "train",
        dataset_test_size: float = 0.1,
        dataset_seed: int = 2025,
        max_turns: int = 50,
        **kwargs
    ):
        """Initialize DeepSWE Actor
        
        Args:
            dataset_name: Dataset to use (R2E-Gym/R2E-Gym-Subset, R2E-Gym/SWE-Bench-Lite, R2E-Gym/SWE-Bench-Verified)
            dataset_split: Dataset split
            dataset_test_size: Test split size
            dataset_seed: Random seed
            max_turns: Maximum conversation turns
            **kwargs: Additional arguments
        """
        # Load dataset
        split = "test" if dataset_name == "R2E-Gym/SWE-Bench-Verified" else dataset_split
        raw_dataset = load_dataset(dataset_name, split=split)
        
        # Process dataset
        def to_record(d):
            problem_statement = d.get("problem_statement", "")
            return {
                "prompt": self._format_prompt(problem_statement, max_turns),
                "info": {**d, "docker_image": d.get("docker_image", d.get("image_name"))},
                "answer": "",
            }
        
        raw_dataset = raw_dataset.map(to_record)
        split_data = raw_dataset.train_test_split(test_size=dataset_test_size, seed=dataset_seed)
        
        self.dataset = split_data["train"]
        self.eval_dataset = split_data["test"]
        self.max_turns = max_turns
        self.dataset_name = dataset_name
    
    def _format_prompt(self, problem_statement: str, max_turns: int) -> str:
        """Format the prompt for DeepSWE tasks"""
        return f"""Consider the following github issue:
<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

IMPORTANT TIP:
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
  2.1 reproduce_issue.py script finishes quickly after checking the error, fix etc. There should be no long running background servers for django for instance etc. It should be a quick script which checks the error and fix to provide a visible response.
  2.2 SUPER IMPORTANT: to ensure this reproduce_script.py must have a timeout logic of 20 seconds. If the script runs for more than 30 seconds, it should output a timeout message and you can interpret accordingly.
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well

VERY IMPORTANT: each response must include both reasoning and function call to solve the task.

You can take multiple turns to solve the task. So please only finish / submit when you are confident in your response. Don't rush. Be comprehensive.
You have to submit your final solution before reaching {max_turns} steps.
  
Your thinking should be thorough and so it's fine if it's very long.
VERY IMPORTANT: file_editor old_str and new_str must be w/o the line numbers. Line numbers are only shown in the view for clarity.

Also if a file_editor edit fails, it's a good idea to view the file near the edit location before trying to edit again. Don't keep trying the same edit over and over again. It will keep leading to the same failure.
Again, do not get stuck trying to do the same thing over and over again. Please be efficient.
"""
    
    async def generate(self, task_id: int = None) -> Challenge:
        """Generate a DeepSWE challenge
        
        Args:
            task_id: Optional task ID for deterministic selection
        
        Returns:
            Challenge object with prompt and metadata
        """
        if task_id is not None:
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        return Challenge(
            env="deepswe",
            prompt=sample["prompt"],
            extra={
                "task_id": task_id,
                "dataset_index": idx,
                "instance_id": sample["info"].get("instance_id", ""),
                "repo_name": sample["info"].get("repo_name", ""),
                "docker_image": sample["info"].get("docker_image", ""),
                "problem_statement": sample["info"].get("problem_statement", ""),
            }
        )
    
    async def evaluate(self, response: str, challenge: Challenge) -> tuple[float, dict]:
        """Evaluate response
        
        Note: Full evaluation requires sandbox execution which is handled by the
        original DeepSweSandboxEnv. This is a simplified interface for AFFINETES.
        For actual SWE-bench evaluation, use the original deepswe.py directly.
        
        Args:
            response: Model response (conversation or final patch)
            challenge: Original challenge
        
        Returns:
            Tuple of (score, extra_info)
        """
        # For AFFINETES integration, we return a placeholder score
        # The actual evaluation happens in the sandbox environment
        return 0.0, {
            "message": "DeepSWE evaluation requires full sandbox execution",
            "note": "Use the original DeepSweSandboxEnv for complete evaluation",
            "instance_id": challenge.extra.get("instance_id", ""),
        }