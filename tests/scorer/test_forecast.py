import pytest
import json
import math
from test_helpers.utils import simple_task_state
from inspect_ai.scorer import Target
from inspect_ai.scorer._forecast import (
    normalized_brier_score,
    log_score, 
    peer_normalised_brier_score,
    peer_log_score,
    probability_score,
)
from inspect_ai.scorer._metric import CORRECT, INCORRECT

@pytest.mark.asyncio
async def test_normalized_brier_score_perfect():
    scorer = normalized_brier_score()
    state = simple_task_state(model_output="0.8")
    result = await scorer(state, Target(["0.8"]))
    assert result.value == 100  # Perfect prediction scores 100

@pytest.mark.asyncio
async def test_normalized_brier_score_invalid():
    scorer = normalized_brier_score()
    state = simple_task_state(model_output="invalid")
    result = await scorer(state, Target(["0.8"]))
    assert result.answer == "0.5"  # Default to 0.5 for invalid input

@pytest.mark.asyncio
async def test_log_score_perfect():
    scorer = log_score()
    state = simple_task_state(model_output="1.0")
    result = await scorer(state, Target(["1.0"]))
    assert result.value == 0.0  # Perfect prediction log score

@pytest.mark.asyncio
async def test_log_score_invalid():
    scorer = log_score()
    state = simple_task_state(model_output="invalid")
    result = await scorer(state, Target(["1.0"]))
    assert result.answer == "0.5"  # Default to 0.5 for invalid input

@pytest.mark.asyncio
async def test_peer_normalised_brier_with_community():
    scorer = peer_normalised_brier_score()
    state = simple_task_state(model_output="0.8")
    target_data = {
        "outcome": 1.0,
        "community_prediction": 0.7
    }
    result = await scorer(state, Target([json.dumps(target_data)]))
    assert isinstance(result.value, float)

@pytest.mark.asyncio
async def test_peer_normalised_brier_no_community():
    scorer = peer_normalised_brier_score()
    state = simple_task_state(model_output="0.8")
    result = await scorer(state, Target(["invalid json"]))
    assert result.value is math.nan
    assert result.answer == "N/A"

@pytest.mark.asyncio
async def test_peer_log_score_with_community():
    scorer = peer_log_score()
    state = simple_task_state(model_output="0.8")
    target_data = {
        "outcome": 1.0,
        "community_prediction": 0.7
    }
    result = await scorer(state, Target([json.dumps(target_data)]))
    assert isinstance(result.value, float)

@pytest.mark.asyncio
async def test_peer_log_score_no_community():
    scorer = peer_log_score()
    state = simple_task_state(model_output="0.8")
    result = await scorer(state, Target(["invalid json"]))
    assert result.value is math.nan
    assert result.answer == "N/A"

@pytest.mark.asyncio
async def test_probability_score_correct():
    scorer = probability_score()
    state = simple_task_state(model_output="0.8")
    result = await scorer(state, Target(["0.75"]))  # Within 0.1
    assert result.value == CORRECT

@pytest.mark.asyncio
async def test_probability_score_incorrect():
    scorer = probability_score()
    state = simple_task_state(model_output="0.8")
    result = await scorer(state, Target(["0.6"]))  # Outside 0.1
    assert result.value == INCORRECT

@pytest.mark.asyncio
async def test_probability_score_invalid():
    scorer = probability_score()
    state = simple_task_state(model_output="invalid")
    result = await scorer(state, Target(["0.8"]))
    assert result.value == INCORRECT