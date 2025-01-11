from typing import Dict, List, Optional
from pathlib import Path
import git
from git import Repo
from inspect_ai.solver import Solver, system_message, prompt_template, generate

class PromptLibrary:
    def __init__(self, prompts_dir="prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._init_repo()

    def _init_repo(self):
        if not self.prompts_dir.exists():
            self.prompts_dir.mkdir(parents=True)
            self.repo = Repo.init(self.prompts_dir)
        else:
            self.repo = Repo(self.prompts_dir)

    def get_prompt(self, name: str, ref: str = "main") -> str:
        """Get prompt content at specific git ref."""
        try:
            return self.repo.git.show(f'{ref}:{name}.txt')
        except git.exc.GitCommandError:
            raise ValueError(f"No prompt found for '{name}' at ref '{ref}'")