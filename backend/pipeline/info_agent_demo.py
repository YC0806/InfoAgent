from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running from repo root or backend/ without package installation.
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[2]
BACKEND_ROOT = CURRENT_DIR.parents[1]
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.orchestrator import InfoAgentOrchestrator
from utils.logger import get_logger, init_logging

init_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InfoAgent end-to-end demo")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Resume a previous run by its run_id",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="最近一周产业界和学术界的在AI领域的新发现、新进展、新趋势以及关键事件",
        help="Query text (ignored when --resume is used with existing brief)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logger.info("Starting info agent demo...")

    orchestrator = InfoAgentOrchestrator(timeout_seconds=300)
    output = await orchestrator.run(
        args.query,
        run_id=args.resume,
    )

    print(f"\n=== Run ID: {output.get('run_id', 'N/A')} ===")

    print("\n=== Status Summary ===")
    print(json.dumps(output.get("status_summary", {}), ensure_ascii=False, indent=2))

    print("\n=== Brief ===")
    print(json.dumps(output["brief"], ensure_ascii=False, indent=2))

    if "facet_results" in output:
        for i, facet in enumerate(output["facet_results"]):
            print(f"\n=== Facet {i+1}: {facet['objective']} ===")
            print(f"All gaps closed: {facet['all_gaps_closed']}")
            print(f"Summary: {facet['summary'][:500]}")
            events = facet["planner_dump"]["work_plan"]["events"]
            print(f"Events: {len(events)}")
            evidence = facet["planner_dump"]["work_plan"]["evidence"]
            print(f"Evidence items: {len(evidence)}")


if __name__ == "__main__":
    asyncio.run(main())
