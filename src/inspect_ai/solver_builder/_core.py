from typing import Dict, List, Optional, Union
from pathlib import Path
import git
from git import Repo
from inspect_ai.solver import Solver, system_message, prompt_template, generate

class PromptLibrary:
    """Manages versioned prompts with file system and optional git backing."""
    
    def __init__(self, prompts_dir: Union[str, Path], use_git: bool = True):
        self.prompts_dir = Path(prompts_dir)
        self.use_git = use_git
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage backend (file system and optional git)."""
        if not self.prompts_dir.exists():
            self.prompts_dir.mkdir(parents=True)
        
        if self.use_git:
            if not (self.prompts_dir / '.git').exists():
                self.repo = Repo.init(self.prompts_dir)
                # Create initial commit so git commands work
                self.repo.index.commit("Initial commit")
            else:
                self.repo = Repo(self.prompts_dir)

    def get_prompt(self, name: str, version: str = "latest") -> str:
        """Get prompt content with optional version."""
        if version == "latest" or not self.use_git:
            # Use direct file access
            file_path = self.prompts_dir / f"{name}.txt"
            if not file_path.exists():
                raise ValueError(f"No prompt found for '{name}'")
            return file_path.read_text()
        
        # Use git for versioned access
        try:
            return self.repo.git.show(f'{version}:{name}.txt')
        except git.exc.GitCommandError:
            raise ValueError(f"No prompt found for '{name}' at version '{version}'")

class SolverBuilder:
    """Base class for all solver builders."""
    def build(self, prompt_lib: PromptLibrary, versions: Dict[str, str] = None) -> List[Solver]:
        """Build a solver using prompts from the library."""
        raise NotImplementedError

class BasicSolverBuilder(SolverBuilder):
    def build(self, prompt_lib: PromptLibrary, versions: Dict[str, str] = None) -> List[Solver]:
        v = versions or {}
        return [
            system_message(prompt_lib.get_prompt("system", v.get("system", "main"))),
            prompt_template(prompt_lib.get_prompt("forecast_method", v.get("forecast_method", "main"))),
            generate()
        ]

class CoTSolverBuilder(SolverBuilder):
    def build(self, prompt_lib: PromptLibrary, versions: Dict[str, str] = None) -> List[Solver]:
        v = versions or {}
        return [
            system_message(prompt_lib.get_prompt("system", v.get("system", "main"))),
            prompt_template(prompt_lib.get_prompt("cot_prefix", v.get("cot_prefix", "main"))),
            prompt_template(prompt_lib.get_prompt("forecast_method", v.get("forecast_method", "main"))),
            generate()
        ]