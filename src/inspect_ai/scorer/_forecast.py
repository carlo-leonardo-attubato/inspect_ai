import re
import math
import json
from typing import Optional
from inspect_ai._util.text import str_to_float
from inspect_ai.solver._task_state import TaskState
from ._metric import CORRECT, INCORRECT, Score
from ._scorer import Scorer, scorer
from ._metrics import mean, stderr, accuracy
from ._target import Target

def _extract_probability(text: str) -> tuple[str, float]:
    """Extract probability from text and normalize"""
    text = text.strip()
    try:
        prob = str_to_float(text)
        if 0 <= prob <= 1:
            return text, prob
        return "0.5", 0.5
    except ValueError:
        return "0.5", 0.5

@scorer(metrics=[mean(), stderr()])
def normalized_brier_score() -> Scorer:
    """Scorer that produces a normalized Brier score
    0 = random prediction
    100 = perfect prediction
    """
    async def score(state: TaskState, target: Target) -> Score:
        answer, pred_prob = _extract_probability(state.output.completion)
        true_prob = float(target.text)
        
        brier = (pred_prob - true_prob) ** 2
        # Scale to 0-100 where:
        # random (0.25) -> 0
        # perfect (0) -> 100
        normalized = (0.25 - brier) / 0.25 * 100
        
        return Score(
            value=float(normalized),
            answer=answer,
            explanation=state.output.completion if state.output.completion != answer else None
        )
    return score

@scorer(metrics=[mean(), stderr()])
def log_score() -> Scorer:
    """Scorer that produces a log score"""
    async def score(state: TaskState, target: Target) -> Score:
        answer, pred_prob = _extract_probability(state.output.completion)
        true_prob = float(target.text)
        
        log_score_val = math.log(pred_prob if true_prob == 1 else (1 - pred_prob))
        
        return Score(
            value=float(log_score_val),
            answer=answer,
            explanation=state.output.completion if state.output.completion != answer else None
        )
    return score

@scorer(metrics=[mean(), stderr()])
def peer_normalised_brier_score() -> Scorer:
    """Scorer that compares model Brier score to community predictions
    Returns the difference in normalized scores (0-100 scale)
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            target_data = json.loads(target.text)
            true_prob = float(target_data["outcome"])
            community_prob = float(target_data["community_prediction"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return Score(
                value=math.nan,
                answer="N/A",
                explanation="Invalid target data format"
            )

        answer, pred_prob = _extract_probability(state.output.completion)
        
        model_brier = (pred_prob - true_prob) ** 2
        community_brier = (community_prob - true_prob) ** 2
        
        # Scale both to 0-100
        model_normalized = (0.25 - model_brier) / 0.25 * 100
        community_normalized = (0.25 - community_brier) / 0.25 * 100
        
        return Score(
            value=float(model_normalized - community_normalized),
            answer=answer,
            explanation=state.output.completion if state.output.completion != answer else None
        )
    return score

@scorer(metrics=[mean(), stderr()])
def peer_log_score() -> Scorer:
    """Scorer that compares model log score to community predictions"""
    async def score(state: TaskState, target: Target) -> Score:
        try:
            target_data = json.loads(target.text)
            true_prob = float(target_data["outcome"])
            community_prob = float(target_data["community_prediction"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return Score(
                value=math.nan,
                answer="N/A",
                explanation="Invalid target data format"
            )

        answer, pred_prob = _extract_probability(state.output.completion)
        
        model_log = math.log(pred_prob if true_prob == 1 else (1 - pred_prob))
        community_log = math.log(community_prob if true_prob == 1 else (1 - community_prob))
        
        return Score(
            value=float(model_log - community_log),
            answer=answer,
            explanation=state.output.completion if state.output.completion != answer else None
        )
    return score

@scorer(metrics=[accuracy(), stderr()])
def probability_score() -> Scorer:
    """Scorer that checks if probability is within 0.1 of target"""
    async def score(state: TaskState, target: Target) -> Score:
        answer, pred_prob = _extract_probability(state.output.completion)
        target_prob = float(target.text)
        
        return Score(
            value=CORRECT if abs(pred_prob - target_prob) <= 0.1 else INCORRECT,
            answer=answer,
            explanation=state.output.completion if state.output.completion != answer else None
        )
    return score