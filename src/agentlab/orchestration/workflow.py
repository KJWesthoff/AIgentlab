"""Deterministic workflow: research → analyze → write → review.

Routing is ordinary Python, not an LLM supervisor — every loop has a
deterministic limit and a measurable exit condition. This is still a
multi-agent system: each stage has its own role, instructions, model
profile, tools and budget.
"""

from __future__ import annotations

import json
from uuid import uuid4

from pydantic import BaseModel, Field

from ..agents.definitions import (
    AgentSpec,
    AnalysisResult,
    ResearchResult,
    ReviewResult,
)
from ..agents.runtime import AgentRuntime
from ..observability.trace import Tracer, budget_snapshot, truncate
from .state import BudgetTracker, TaskState

MAX_REVISIONS = 1


class WorkflowResult(BaseModel):
    task_id: str
    objective: str
    final_answer: str
    approved: bool
    revisions: int = 0

    research: ResearchResult
    analysis: AnalysisResult
    review: ReviewResult

    model_calls: int
    tool_calls: int
    accumulated_cost_usd: float
    history: list[dict] = Field(default_factory=list)


class Workflow:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        agents: dict[str, AgentSpec],
        tracker: BudgetTracker,
        tracer: Tracer | None = None,
    ) -> None:
        self._runtime = runtime
        self._agents = agents
        self._tracker = tracker
        self._tracer = tracer or Tracer()

    async def execute(self, objective: str) -> WorkflowResult:
        state = TaskState(task_id=str(uuid4()), objective=objective)
        self._tracer.emit(
            "run_started",
            task_id=state.task_id,
            objective=objective,
            agents=[
                {
                    "name": spec.name,
                    "model_profile": spec.model_profile,
                    "allowed_tools": sorted(spec.allowed_tools),
                    "max_calls": spec.max_calls,
                }
                for spec in self._agents.values()
            ],
            budget=budget_snapshot(self._tracker),
        )

        self._tracer.emit("stage_started", stage="research", agent="researcher")
        research = await self._runtime.run_structured(
            agent=self._agents["researcher"],
            state=state,
            task_input=f"Objective: {objective}\n\nCollect relevant evidence.",
            response_type=ResearchResult,
        )
        state.artifacts["research"] = research.model_dump()

        # Nothing downstream can succeed without evidence, and letting the
        # pipeline run anyway spends three more model calls to produce a
        # refusal that was knowable here — one that reads like the system
        # is broken rather than like the corpus simply lacks the material.
        if not research.evidence:
            return self._no_evidence(state, objective, research)

        self._tracer.emit("stage_started", stage="analyze", agent="analyst")
        analysis = await self._runtime.run_structured(
            agent=self._agents["analyst"],
            state=state,
            task_input=(
                f"Objective: {objective}\n\n"
                f"Evidence:\n{research.model_dump_json(indent=2)}\n\n"
                "Analyze only the available evidence."
            ),
            response_type=AnalysisResult,
        )
        state.artifacts["analysis"] = analysis.model_dump()

        draft = await self._write_draft(state, objective, research, analysis)

        review = await self._review(state, objective, research, draft)

        revisions = 0
        while not review.approved and revisions < MAX_REVISIONS:
            revisions += 1
            self._tracer.emit(
                "revision_started",
                revision=revisions,
                required_changes=review.required_changes,
            )
            draft = await self._write_draft(
                state,
                objective,
                research,
                analysis,
                revision_instructions=review.required_changes,
                previous_draft=draft,
            )
            review = await self._review(state, objective, research, draft)

        state.status = "completed"
        state.final_answer = draft
        self._tracer.emit(
            "run_finished",
            approved=review.approved,
            revisions=revisions,
            final_answer=truncate(draft),
            budget=budget_snapshot(self._tracker),
        )

        return WorkflowResult(
            task_id=state.task_id,
            objective=objective,
            final_answer=draft,
            approved=review.approved,
            revisions=revisions,
            research=research,
            analysis=analysis,
            review=review,
            model_calls=self._tracker.model_calls,
            tool_calls=self._tracker.tool_calls,
            accumulated_cost_usd=self._tracker.accumulated_cost_usd,
            history=state.history,
        )

    def _no_evidence(
        self, state: TaskState, objective: str, research: ResearchResult
    ) -> WorkflowResult:
        """Stop early and say why, rather than answer without grounding."""
        unanswered = research.unanswered_questions
        answer = (
            "No answer: the corpus contains no evidence relevant to this "
            "objective, and answering without evidence would mean inventing "
            "it.\n\nPoint --corpus-dir at a corpus that covers the topic, "
            "or add source documents to the current one."
        )
        if unanswered:
            answer += "\n\nThe researcher could not resolve:\n" + "\n".join(
                f"  - {question}" for question in unanswered
            )

        state.status = "completed"
        state.final_answer = answer
        state.log("no_evidence", agent="researcher")
        self._tracer.emit(
            "run_finished",
            approved=False,
            revisions=0,
            final_answer=answer,
            no_evidence=True,
            budget=budget_snapshot(self._tracker),
        )

        return WorkflowResult(
            task_id=state.task_id,
            objective=objective,
            final_answer=answer,
            approved=False,
            revisions=0,
            research=research,
            analysis=AnalysisResult(conclusions=[], confidence=0.0),
            review=ReviewResult(
                approved=False,
                required_changes=["Retrieval returned no evidence."],
            ),
            model_calls=self._tracker.model_calls,
            tool_calls=self._tracker.tool_calls,
            accumulated_cost_usd=self._tracker.accumulated_cost_usd,
            history=state.history,
        )

    async def _write_draft(
        self,
        state: TaskState,
        objective: str,
        research: ResearchResult,
        analysis: AnalysisResult,
        revision_instructions: list[str] | None = None,
        previous_draft: str | None = None,
    ) -> str:
        task_input = (
            f"Objective: {objective}\n\n"
            f"Evidence:\n{research.model_dump_json(indent=2)}\n\n"
            f"Analysis:\n{analysis.model_dump_json(indent=2)}\n\n"
            "Write a grounded answer using only the approved evidence and "
            "analysis above."
        )

        # The task input ends with the instruction the model weights most,
        # so a writer holding tools needs the reminder here rather than
        # only in its system prompt — otherwise it drafts prose and never
        # reaches for the tool it was granted.
        if self._agents["writer"].allowed_tools:
            task_input += (
                " If the objective asks for the answer to be saved, "
                "exported or written to a file, call save_report with the "
                "full answer before replying."
            )

        if revision_instructions:
            task_input += (
                f"\n\nPrevious draft:\n{previous_draft}\n\n"
                "The reviewer requires these changes:\n"
                f"{json.dumps(revision_instructions, indent=2)}"
            )

        self._tracer.emit("stage_started", stage="write", agent="writer")
        draft = await self._runtime.run_text(
            agent=self._agents["writer"],
            state=state,
            task_input=task_input,
        )
        self._tracer.emit(
            "draft_produced",
            agent="writer",
            chars=len(draft),
            draft=truncate(draft),
        )
        return draft

    async def _review(
        self,
        state: TaskState,
        objective: str,
        research: ResearchResult,
        draft: str,
    ) -> ReviewResult:
        self._tracer.emit("stage_started", stage="review", agent="reviewer")
        return await self._runtime.run_structured(
            agent=self._agents["reviewer"],
            state=state,
            task_input=(
                f"Objective: {objective}\n\n"
                f"Evidence:\n{research.model_dump_json(indent=2)}\n\n"
                f"Draft:\n{draft}\n\n"
                "Check the draft for claims unsupported by the evidence. "
                "Approve only if every substantive claim is supported."
            ),
            response_type=ReviewResult,
        )
