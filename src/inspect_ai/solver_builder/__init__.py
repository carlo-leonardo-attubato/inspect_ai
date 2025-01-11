"""Solver builder module for creating different types of forecasting solvers."""

from inspect_ai.solver import system_message, prompt_template, generate
from inspect_ai.solver_builder._core import (
    PromptLibrary,
    SolverBuilder,
    BasicSolverBuilder,
    CoTSolverBuilder,
)

# Direct solver definitions
SIMPLE_FORECAST = [
    system_message("You are a forecasting expert. Provide a probability between 0 and 1."),
    prompt_template("What is the probability that: {prompt}"),
    generate()
]

SIMPLE_COT = [
    system_message("You are a forecasting expert. Think step by step and provide a probability between 0 and 1."),
    prompt_template("""Let's approach this step by step:
1. First, let's understand what we're forecasting
2. Next, let's consider key factors
3. Finally, let's estimate a probability
Question: {prompt}"""),
    generate()
]

__all__ = [
    "PromptLibrary",
    "SolverBuilder",
    "BasicSolverBuilder",
    "CoTSolverBuilder",
    "SIMPLE_FORECAST",
    "SIMPLE_COT",
]