"""Tests for solver builder functionality."""
from pathlib import Path
import pytest
from git import Repo

from inspect_ai.solver import system_message, prompt_template, generate
from inspect_ai.solver_builder._core import (
    PromptLibrary,
    SolverBuilder,
    BasicSolverBuilder,
    CoTSolverBuilder,
)

@pytest.fixture
def temp_prompts(tmp_path):
    """Create temporary prompts directory with initial files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    
    # Create initial files
    (prompts_dir / "system.txt").write_text("You are a forecasting expert.")
    (prompts_dir / "forecast_method.txt").write_text("What is the probability: {prompt}")
    (prompts_dir / "cot_prefix.txt").write_text("Let's solve this step by step:\n")
    
    return prompts_dir

@pytest.fixture
def versioned_prompts(temp_prompts):
    """Create git repo with multiple versions of prompts."""
    repo = Repo.init(temp_prompts)
    repo.git.symbolic_ref('HEAD', 'refs/heads/main')
    
    # Version 1: Basic forecasting
    repo.index.add(["system.txt", "forecast_method.txt", "cot_prefix.txt"])
    repo.index.commit("Initial prompts")
    basic_version = repo.head.commit.hexsha
    
    # Version 2: Base rate forecasting
    (temp_prompts / "system.txt").write_text("You are a forecasting expert specializing in base rates.")
    (temp_prompts / "forecast_method.txt").write_text("Using historical base rates, what is the probability: {prompt}")
    (temp_prompts / "cot_prefix.txt").write_text("""
    Let's solve this step by step:
    1. Identify the reference class
    2. Find historical base rates
    3. Adjust for specifics
    """)
    repo.index.add(["system.txt", "forecast_method.txt", "cot_prefix.txt"])
    repo.index.commit("Add base rate prompts")
    base_rate_version = repo.head.commit.hexsha
    
    # Version 3: Expert opinion forecasting
    (temp_prompts / "system.txt").write_text("You are a domain expert in forecasting.")
    (temp_prompts / "forecast_method.txt").write_text("Based on expert knowledge, what is the probability: {prompt}")
    (temp_prompts / "cot_prefix.txt").write_text("""
    Let's solve this step by step:
    1. Apply domain expertise
    2. Consider similar cases
    3. Adjust for context
    """)
    repo.index.add(["system.txt", "forecast_method.txt", "cot_prefix.txt"])
    repo.index.commit("Add expert prompts")
    expert_version = repo.head.commit.hexsha
    
    return temp_prompts, {
        "basic": basic_version,
        "base_rate": base_rate_version,
        "expert": expert_version
    }

def test_prompt_library_init(temp_prompts):
    """Test basic PromptLibrary initialization."""
    lib = PromptLibrary(prompts_dir=temp_prompts, use_git=False)
    assert lib.prompts_dir == temp_prompts

def test_prompt_library_file_read(temp_prompts):
    """Test reading prompts from files (no versioning)."""
    lib = PromptLibrary(prompts_dir=temp_prompts, use_git=False)
    assert lib.get_prompt("system") == "You are a forecasting expert."

def test_versioned_prompt_access(versioned_prompts):
    """Test accessing different versions of prompts."""
    prompts_dir, versions = versioned_prompts
    lib = PromptLibrary(prompts_dir=prompts_dir, use_git=True)
    
    # Check each version
    assert "forecasting expert" in lib.get_prompt("system", version=versions["basic"])
    assert "base rates" in lib.get_prompt("forecast_method", version=versions["base_rate"])
    assert "expert knowledge" in lib.get_prompt("forecast_method", version=versions["expert"])

def test_basic_solver_with_versions(versioned_prompts):
    """Test BasicSolverBuilder with different prompt versions."""
    prompts_dir, versions = versioned_prompts
    lib = PromptLibrary(prompts_dir=prompts_dir, use_git=True)
    
    # Build solver with base rate version
    solver = BasicSolverBuilder().build(
        lib,
        versions={
            "system": versions["base_rate"],
            "forecast_method": versions["base_rate"]
        }
    )
    
    assert len(solver) == 3
    base_rate_text = lib.get_prompt("forecast_method", version=versions["base_rate"])
    assert "base rates" in base_rate_text

def test_cot_solver_with_versions(versioned_prompts):
    """Test CoTSolverBuilder with different prompt versions."""
    prompts_dir, versions = versioned_prompts
    lib = PromptLibrary(prompts_dir=prompts_dir, use_git=True)
    
    # Build solver with expert version
    solver = CoTSolverBuilder().build(
        lib,
        versions={
            "system": versions["expert"],
            "cot_prefix": versions["expert"],
            "forecast_method": versions["expert"]
        }
    )
    
    assert len(solver) == 4
    expert_text = lib.get_prompt("forecast_method", version=versions["expert"])
    assert "expert knowledge" in expert_text

def test_mixing_prompt_versions(versioned_prompts):
    """Test mixing different versions of prompts."""
    prompts_dir, versions = versioned_prompts
    lib = PromptLibrary(prompts_dir=prompts_dir, use_git=True)
    
    # Build solver with mixed versions
    solver = CoTSolverBuilder().build(
        lib,
        versions={
            "system": versions["expert"],          # Expert system prompt
            "cot_prefix": versions["base_rate"],   # Base rate reasoning steps
            "forecast_method": versions["basic"]    # Basic forecast method
        }
    )
    
    assert len(solver) == 4
    assert "expert" in lib.get_prompt("system", version=versions["expert"])
    assert "base rates" in lib.get_prompt("cot_prefix", version=versions["base_rate"])

def test_version_not_found(versioned_prompts):
    """Test behavior when version doesn't exist."""
    prompts_dir, versions = versioned_prompts
    lib = PromptLibrary(prompts_dir=prompts_dir, use_git=True)
    
    with pytest.raises(ValueError):
        lib.get_prompt("system", version="nonexistent")

def test_latest_version_fallback(versioned_prompts):
    """Test fallback to latest version."""
    prompts_dir, versions = versioned_prompts
    lib = PromptLibrary(prompts_dir=prompts_dir, use_git=True)
    
    # Latest version should be expert
    latest = lib.get_prompt("system")
    assert "domain expert" in latest