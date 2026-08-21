"""Execution budgets and the shared task state.

A single user task can fan out into many model calls, so budgets are
enforced at the runtime level before every model call and tool call.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..llm.types import GenerationResponse


class BudgetExceeded(RuntimeError):
    pass


class ExecutionBudget(BaseModel):
    maximum_model_calls: int = 12
    maximum_tool_calls: int = 6
    maximum_input_tokens: int = 200_000
    # Headroom for the researcher's 12k evidence cap plus the four later
    # stages and a revision; raising a per-agent cap without this only
    # moves the failure from truncation to BudgetExceeded. Cost stays the
    # real guard — output tokens are the cheap half of a run.
    maximum_output_tokens: int = 60_000
    maximum_cost_usd: float = 1.00


class BudgetTracker(BaseModel):
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)

    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    accumulated_cost_usd: float = 0.0

    def before_model_call(self) -> None:
        if self.model_calls >= self.budget.maximum_model_calls:
            raise BudgetExceeded("Maximum model calls reached.")
        if self.accumulated_cost_usd >= self.budget.maximum_cost_usd:
            raise BudgetExceeded("Maximum task cost reached.")
        if self.input_tokens >= self.budget.maximum_input_tokens:
            raise BudgetExceeded("Maximum input tokens reached.")
        if self.output_tokens >= self.budget.maximum_output_tokens:
            raise BudgetExceeded("Maximum output tokens reached.")

    def before_tool_call(self) -> None:
        if self.tool_calls >= self.budget.maximum_tool_calls:
            raise BudgetExceeded("Maximum tool calls reached.")

    def record_model_call(self, response: GenerationResponse) -> None:
        self.model_calls += 1
        usage = response.usage
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.accumulated_cost_usd += usage.estimated_cost or 0.0

    def record_tool_call(self) -> None:
        self.tool_calls += 1


class TaskState(BaseModel):
    task_id: str
    objective: str
    artifacts: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "running"
    final_answer: str | None = None

    def log(self, event: str, **details: Any) -> None:
        self.history.append({"event": event, **details})
