import pytest

from agentlab.llm.types import GenerationResponse, Usage
from agentlab.orchestration.state import (
    BudgetExceeded,
    BudgetTracker,
    ExecutionBudget,
)


def make_response(cost: float) -> GenerationResponse:
    return GenerationResponse(
        provider="scripted",
        requested_model="scripted/model",
        usage=Usage(input_tokens=10, output_tokens=10, estimated_cost=cost),
    )


def test_model_call_limit():
    tracker = BudgetTracker(budget=ExecutionBudget(maximum_model_calls=2))
    for _ in range(2):
        tracker.before_model_call()
        tracker.record_model_call(make_response(0.0))

    with pytest.raises(BudgetExceeded, match="model calls"):
        tracker.before_model_call()


def test_cost_limit():
    tracker = BudgetTracker(budget=ExecutionBudget(maximum_cost_usd=0.05))
    tracker.before_model_call()
    tracker.record_model_call(make_response(0.06))

    with pytest.raises(BudgetExceeded, match="cost"):
        tracker.before_model_call()


def test_tool_call_limit():
    tracker = BudgetTracker(budget=ExecutionBudget(maximum_tool_calls=1))
    tracker.before_tool_call()
    tracker.record_tool_call()

    with pytest.raises(BudgetExceeded, match="tool calls"):
        tracker.before_tool_call()
