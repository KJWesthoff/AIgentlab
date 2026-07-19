"""Entry point.

Requires OPENROUTER_API_KEY in .env or the environment:

    python -m agentlab.main "Your question here"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from .agents.definitions import load_agents
from .agents.runtime import AgentRuntime
from .llm.openrouter import OpenRouterProvider
from .llm.registry import ModelRegistry
from .llm.service import LLMService
from .observability.trace import Tracer, TraceWriter
from .orchestration.state import BudgetTracker, ExecutionBudget
from .orchestration.workflow import Workflow
from .tools.registry import build_default_tools

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBJECTIVE = (
    "Explain the difference between retrieval-augmented generation and a "
    "plain database lookup."
)


async def run(
    objective: str,
    config_dir: Path,
    corpus_dir: Path,
    search_mode: str,
    tracer: Tracer | None = None,
) -> None:
    registry = ModelRegistry.from_yaml(config_dir / "models.yaml")
    agents = load_agents(config_dir / "agents.yaml")

    if search_mode == "vector":
        try:
            from .tools.vector_search import build_vector_tools
        except ImportError as exc:
            raise SystemExit(
                "Vector search needs the fastembed and numpy packages "
                f"({exc}). Reinstall with `pip install -e .` or run with "
                "--search-mode keyword."
            ) from exc
        tools = build_vector_tools(corpus_dir)
    else:
        tools = build_default_tools(corpus_dir)

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env."
        )
    provider = OpenRouterProvider(api_key=api_key)

    service = LLMService(
        model_registry=registry,
        providers={"openrouter": provider},
    )
    tracker = BudgetTracker(budget=ExecutionBudget())
    runtime = AgentRuntime(
        service=service, tools=tools, tracker=tracker, tracer=tracer
    )
    workflow = Workflow(
        runtime=runtime, agents=agents, tracker=tracker, tracer=tracer
    )

    try:
        result = await workflow.execute(objective)
    except BaseException as error:
        if tracer is not None:
            tracer.emit("run_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        await service.close()

    print(f"Task:      {result.objective}")
    print(f"Approved:  {result.approved} (revisions: {result.revisions})")
    print(
        f"Budget:    {result.model_calls} model calls, "
        f"{result.tool_calls} tool calls, "
        f"${result.accumulated_cost_usd:.4f}"
    )
    print()
    print(result.final_answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="agentlab multi-agent workflow")
    parser.add_argument(
        "objective",
        nargs="?",
        default=DEFAULT_OBJECTIVE,
        help="The task for the workflow.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "config",
        help="Directory containing models.yaml and agents.yaml.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus",
        help="Directory of .md files the researcher can search.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["vector", "keyword"],
        default="vector",
        help=(
            "How search_documents ranks the corpus: semantic embeddings "
            "(default; first run downloads a small local model) or plain "
            "keyword matching."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Serve a live trace viewer on localhost showing each agent's "
            "context window, tool/policy decisions and budget as the run "
            "progresses."
        ),
    )
    parser.add_argument(
        "--live-port",
        type=int,
        default=8642,
        help="Port for the live viewer (falls back to a free port if taken).",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help=(
            "Write the run trace (JSON lines) to this path, even without "
            "--live. With --live the default is data/last-run-trace.jsonl."
        ),
    )
    args = parser.parse_args()

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"Corpus directory not found: {args.corpus_dir}")

    tracer = None
    server = None
    if args.live or args.trace_file:
        trace_path = args.trace_file or (
            PROJECT_ROOT / "data" / "last-run-trace.jsonl"
        )
        tracer = TraceWriter(trace_path)
        if args.live:
            from .observability.server import TraceServer

            server = TraceServer(trace_path=trace_path, port=args.live_port)
            server.start()
            print(f"Live trace viewer: {server.url}")

    try:
        asyncio.run(
            run(
                args.objective,
                args.config_dir,
                args.corpus_dir,
                args.search_mode,
                tracer=tracer,
            )
        )
        if server is not None:
            print(f"\nRun finished — viewer still at {server.url} (Ctrl+C to exit).")
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()
        if tracer is not None:
            tracer.close()


if __name__ == "__main__":
    main()
