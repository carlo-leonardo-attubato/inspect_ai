"""Scoring metrics for forecasting tasks."""

import math
import json
import re
from typing import Optional

from .._metric import Score, CORRECT, INCORRECT
from .._scorer import scorer
from .._metrics import mean, stderr, accuracy
from ..._util.pattern import AnswerPattern
from ...model import TaskState
from .._target import Target

@scorer(metrics=[mean(), stderr()])
def normalized_brier_score():
    async def score(state: TaskState, target: Target):
        # extract answer
        match = re.search(AnswerPattern.LINE, state.output.completion)
        if match:
            answer = match.group(1)
            try:
                pred_prob = float(answer)
                if 0 <= pred_prob <= 1:
                    true_prob = float(target.text)
                    # Calculate raw Brier score
                    brier = (pred_prob - true_prob) ** 2
                    # Normalize: random (0.25) -> 0, perfect (0) -> 1
                    normalized = (0.25 - brier) / 0.25
                    return Score(
                        value=normalized,  # Higher is better, range [-3,1]
                        answer=answer,
                        explanation=state.output.completion
                    )
                else:
                    # Invalid probability, treat as 0.5
                    true_prob = float(target.text)
                    pred_prob = 0.5
                    brier = (pred_prob - true_prob) ** 2
                    normalized = (0.25 - brier) / 0.25
                    return Score(
                        value=normalized,
                        answer="0.5",
                        explanation=f"Invalid probability {answer}, assuming 0.5"
                    )
            except ValueError:
                # Couldn't parse as float, treat as 0.5
                true_prob = float(target.text)
                pred_prob = 0.5
                brier = (pred_prob - true_prob) ** 2
                normalized = (0.25 - brier) / 0.25
                return Score(
                    value=normalized,
                    answer="0.5",
                    explanation="Could not parse probability, assuming 0.5"
                )
        else:
            # No answer found, assume 0.5
            true_prob = float(target.text)
            pred_prob = 0.5
            brier = (pred_prob - true_prob) ** 2
            normalized = (0.25 - brier) / 0.25
            return Score(
                value=normalized,
                answer="0.5",
                explanation="No probability found in output, assuming 0.5: " 
                + f"{state.output.completion}"
            )
    
    return score


@scorer(metrics=[mean(), stderr()])
def log_score():
    async def score(state: TaskState, target: Target):
        match = re.search(AnswerPattern.LINE, state.output.completion)
        if match:
            answer = match.group(1)
            try:
                pred_prob = float(answer)
                if 0 <= pred_prob <= 1:
                    true_prob = float(target.text)
                    # Calculate log score
                    log_score = math.log(pred_prob if true_prob == 1 else (1 - pred_prob))
                    return Score(
                        value=log_score,
                        answer=answer,
                        explanation=f"Log score: {log_score:.3f}"
                    )
            except ValueError:
                # Default to random prediction (0.5)
                true_prob = float(target.text)
                log_score = math.log(0.5)
                return Score(
                    value=log_score,
                    answer="0.5",
                    explanation=f"Invalid probability {answer}, using 0.5"
                )
        return Score(
            value=math.log(0.5),  # Random prediction performance
            answer="0.5",
            explanation="No valid probability found, using 0.5"
        )
    return score

@scorer(metrics=[mean(), stderr()])
def peer_normalised_brier_score():
    async def score(state: TaskState, target: Target):
        # Check if community predictions exist
        if not state.metadata.get('community_predictions'):
            return Score(
                value=None,
                answer="N/A",
                explanation="No community predictions available"
            )
        
        # Get community predictions and true outcome
        community_preds = json.loads(state.metadata['community_predictions'])
        community_prob = community_preds[-1][1]  # Last community prediction
        true_prob = float(target.text)
        
        # Get model prediction
        match = re.search(AnswerPattern.LINE, state.output.completion)
        pred_prob = 0.5  # Default prediction
        if match:
            try:
                answer_prob = float(match.group(1))
                if 0 <= answer_prob <= 1:
                    pred_prob = answer_prob
            except ValueError:
                pass
        
        # Calculate Brier scores
        model_brier = (pred_prob - true_prob) ** 2
        community_brier = (community_prob - true_prob) ** 2
        
        # Normalize both scores
        model_normalized = (0.25 - model_brier) / 0.25
        community_normalized = (0.25 - community_brier) / 0.25
        
        # Peer score is difference in normalized scores
        peer_score = model_normalized - community_normalized
        
        return Score(
            value=peer_score,
            answer=str(pred_prob),
            explanation=f"Model normalized Brier: {model_normalized:.3f}, Community normalized Brier: {community_normalized:.3f}"
        )
    return score

@scorer(metrics=[mean(), stderr()])
def peer_log_score():
    async def score(state: TaskState, target: Target):
        # Check if community predictions exist
        if not state.metadata.get('community_predictions'):
            return Score(
                value=None,
                answer="N/A",
                explanation="No community predictions available"
            )
        
        # Get community predictions and true outcome
        community_preds = json.loads(state.metadata['community_predictions'])
        community_prob = community_preds[-1][1]  # Last community prediction
        true_prob = float(target.text)
        
        # Get model prediction
        match = re.search(AnswerPattern.LINE, state.output.completion)
        pred_prob = 0.5  # Default prediction
        if match:
            try:
                answer_prob = float(match.group(1))
                if 0 <= answer_prob <= 1:
                    pred_prob = answer_prob
            except ValueError:
                pass
        
        # Calculate log scores
        model_log_score = math.log(pred_prob if true_prob == 1 else (1 - pred_prob))
        community_log_score = math.log(community_prob if true_prob == 1 else (1 - community_prob))
        
        # Peer score is difference in log scores
        peer_score = model_log_score - community_log_score
        
        return Score(
            value=peer_score,
            answer=str(pred_prob),
            explanation=f"Model log score: {model_log_score:.3f}, Community log score: {community_log_score:.3f}"
        )
    return score

@scorer(metrics=[accuracy(), stderr()])
def probability_score():
    async def score(state: TaskState, target: Target):
        # extract answer
        match = re.search(AnswerPattern.LINE, state.output.completion)
        if match:
            answer = match.group(1)
            try:
                prob = float(answer)
                target_prob = float(target.text)
                # Consider it correct if within 0.1 of target
                correct = abs(prob - target_prob) <= 0.1
                return Score(
                    value=CORRECT if correct else INCORRECT,
                    answer=answer,
                    explanation=state.output.completion
                )
            except ValueError:
                return Score(
                    value=INCORRECT,
                    explanation="Could not parse probability"
                )
        else:
            return Score(
                value=INCORRECT,
                explanation="Answer not found in model output: "
                + f"{state.output.completion}"
            )
    
    return score