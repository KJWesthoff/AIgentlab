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


def test_structured_retry_meters_both_round_trips():
    """A schema retry costs two calls; the budget must see two.

    Before usage travelled back from ``generate_structured``, a
    structured call was metered as one call with zero tokens — the
    tokens a retry actually spent were invisible to the cap and to the
    viewer's tally.
    """
    tracker = BudgetTracker(budget=ExecutionBudget(maximum_model_calls=3))
    tracker.record_model_usage(
        Usage(input_tokens=300, output_tokens=90, estimated_cost=0.004),
        calls=2,
    )

    assert tracker.model_calls == 2
    assert tracker.input_tokens == 300
    assert tracker.output_tokens == 90
    assert tracker.accumulated_cost_usd == pytest.approx(0.004)
